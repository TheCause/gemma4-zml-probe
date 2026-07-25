// G12 — géométrie Gemma 4 12B Unified (jalon J2, plan docs/superpowers/plans/
// 2026-07-24-w4-j2-12b-unified.md, contrat docs/U_12B_CONTRACT.md).
//
// Valeurs = décision D1 (cartographie 24 juil, vérifiée sur pièce — config.json + modeling
// 5.9.0) : 48 couches, pas de YOCO (first_kv_shared = num_layers → aucun reader, chaque
// couche écrit son slot), d 3840, 16 Q ; sliding = GQA 8 KV × 256 (groupe 2), full = MQA
// 1 KV × 512 avec K=V (attention_k_eq_v : V = v_norm(k_proj brut), branche D4 du moteur) ;
// pas de PLE (ple_dim = 0 → bloc comptime-mort) ; couches full aux (i+1) % 6 == 0
// (5, 11, 17, 23, 29, 35, 41, 47).
//
// sliding_writer/full_writer : sans YOCO il n'y a AUCUN reader — ces champs ne sont jamais
// consommés (les branches reader de runLayerGen sont comptime-mortes) ; 0 = valeur inerte.
//
// Task 8 : structs G12LayerW/G12Model (poids packés du checkpoint, pattern DEUX SLICES —
// zml.io.load ne digère pas les optionnels, engine.zig:312-313) + toLayerW (assemblage par
// valeur d'un engine.LayerW, dequantW4 in-graph). Mécanique REPRISE du prototype éprouvé
// U7W/u7LayerW de gemma4_g12gate.zig (gate u7 PASS, pièges revue U6 corrigés/assertés) —
// les gates u6/u7 restent INTACTS (convention « gates immuables = oracles », duplication
// assumée, précédent bbs).

const std = @import("std");
const zml = @import("zml");
const engine = @import("engine.zig");
const w4 = @import("w4.zig");

pub const g12: engine.Geom = .{
    .num_layers = 48,
    .first_kv_shared = 48,
    .sliding_writer = 0,
    .full_writer = 0,
    .d = 3840,
    .nh = 16,
    .kvh_sliding = 8,
    .kvh_full = 1,
    .hd_sliding = 256,
    .hd_full = 512,
    .ple_dim = 0,
    .full_period = 6,
    .k_eq_v_full = true,
    .rope_theta_sliding = 1.0e4,
    .softcap = 30.0,
};

// ---------------------------------------------------------------------------- structs modèle (Task 8)

const N12: usize = g12.num_layers; // 48
const N12_FULL: usize = N12 / g12.full_period; // 8
const N12_SLIDING: usize = N12 - N12_FULL; // 40

/// Poids d'une couche 12B packée — champs COMMUNS aux 48 couches (SANS v_proj : les 8 couches
/// full n'en ont pas au checkpoint, attention_k_eq_v — pattern « deux structs distincts »,
/// prototype U6LayW du gate u6/u7).
pub const G12LayerW = struct {
    input_layernorm: zml.Tensor,
    q: w4.W4Lin,
    q_norm: zml.Tensor,
    k: w4.W4Lin,
    k_norm: zml.Tensor,
    o: w4.W4Lin,
    post_attention_layernorm: zml.Tensor,
    pre_feedforward_layernorm: zml.Tensor,
    gate: w4.W4Lin,
    up: w4.W4Lin,
    down: w4.W4Lin,
    post_feedforward_layernorm: zml.Tensor,
    layer_scalar: zml.Tensor, // [1] bf16 — consommé en fin de couche ET placeholder [1] (D4/PLE)

    fn init(v: zml.io.TensorStore.View) G12LayerW {
        const sa = v.withPrefix("self_attn");
        const mlp = v.withPrefix("mlp");
        return .{
            .input_layernorm = v.createTensor("input_layernorm.weight", .{.d}, null),
            .q = .init(sa, "q_proj"),
            .q_norm = sa.createTensor("q_norm.weight", .{.hd}, null),
            .k = .init(sa, "k_proj"),
            .k_norm = sa.createTensor("k_norm.weight", .{.hd}, null),
            .o = .init(sa, "o_proj"),
            .post_attention_layernorm = v.createTensor("post_attention_layernorm.weight", .{.d}, null),
            .pre_feedforward_layernorm = v.createTensor("pre_feedforward_layernorm.weight", .{.d}, null),
            .gate = .init(mlp, "gate_proj"),
            .up = .init(mlp, "up_proj"),
            .down = .init(mlp, "down_proj"),
            .post_feedforward_layernorm = v.createTensor("post_feedforward_layernorm.weight", .{.d}, null),
            .layer_scalar = v.createTensor("layer_scalar", .{.one}, null),
        };
    }
};

/// Modèle 12B complet au format packé : 48 couches (champs communs) + v_proj des 40 couches
/// SLIDING uniquement (les 8 full : K=V, pas de v_proj au checkpoint). Indexation de `vs` par
/// slidingSlot(i) = nombre de couches non-full < i — pattern « deux slices » J1 (W4Model.kv),
/// ici ENTRELACÉ (full tous les 6), pas préfixe. Pas de plmp/eptl : PLE mort (ple_dim = 0).
/// Prototype éprouvé : U7W (gemma4_g12gate.zig, gate u7 PASS).
pub const G12Model = struct {
    embed_tokens: zml.Tensor, // {voc,d} bf16 NON quantifié (D9) — gather embed ET head tied (D7)
    final_norm: zml.Tensor, // {d}
    layers: []G12LayerW, // 48
    vs: []w4.W4Lin, // 40 (sliding SEULEMENT, indexé slidingSlot — piège revue U6 (b))

    pub fn init(allocator: std.mem.Allocator, base: zml.io.TensorStore.View) !G12Model {
        const layers = try allocator.alloc(G12LayerW, N12);
        const vs = try allocator.alloc(w4.W4Lin, N12_SLIDING);
        const lb = base.withPrefix("layers");
        var vi: usize = 0;
        inline for (0..N12) |i| {
            layers[i] = G12LayerW.init(lb.withLayer(i));
            if (comptime !g12.isFull(i)) {
                // piège (b) : l'ordre d'ajout DOIT coïncider avec slidingSlot(i) — asserté.
                const slot_i = comptime blk: {
                    @setEvalBranchQuota(100_000); // slidingSlot = boucle comptime O(i) x 48 couches
                    break :blk @as(usize, @intCast(g12.slidingSlot(i)));
                };
                if (vi != slot_i) return error.SlidingIndexMismatch;
                vs[vi] = .init(lb.withLayer(i).withPrefix("self_attn"), "v_proj");
                vi += 1;
            }
        }
        if (vi != N12_SLIDING) return error.SlidingIndexMismatch; // piège (a) : 40 exactement
        return .{
            .embed_tokens = base.createTensor("embed_tokens.weight", .{ .voc, .d }, null),
            .final_norm = base.createTensor("norm.weight", .{.d}, null),
            .layers = layers,
            .vs = vs,
        };
    }

    pub fn load(self: *const G12Model, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, store: *const zml.io.TensorStore, shardings: []const zml.sharding.Sharding) !zml.Bufferized(G12Model) {
        return zml.io.load(G12Model, self, allocator, io, platform, store, .{ .shardings = shardings, .parallelism = 1, .dma_chunks = 1, .dma_chunk_size = 16 * 1024 * 1024 });
    }

    /// Assemblage par VALEUR d'un engine.LayerW (prototype u7LayerW, pièges revue U6 corrigés) :
    /// dequantW4 in-graph via toW(tags des sites moteur). Couche full (comptime) : v_proj =
    /// PLACEHOLDER de shape volontairement inconsommable [1] (layer_scalar) — JAMAIS k_proj
    /// recyclé (D4 : un placeholder shape-compatible consommé par erreur donnerait v_norm(kp)
    /// correct par accident) ; toute consommation accidentelle casse au compile. Couche
    /// sliding : vs[slidingSlot(j)] (JAMAIS vs[j] : entrelacement tous les 6, dès L6 vs[j]
    /// brut lirait la mauvaise arête EN SILENCE). Champs PLE = placeholders [1] (ple_dim=0,
    /// branche comptime-morte).
    pub fn toLayerW(self: G12Model, comptime j: usize) engine.LayerW {
        const lw = self.layers[j];
        return .{
            .input_layernorm = lw.input_layernorm,
            .q_proj = lw.q.toW(.{ .o, .d }),
            .q_norm = lw.q_norm,
            .k_proj = lw.k.toW(.{ .o, .d }),
            .k_norm = lw.k_norm,
            .v_proj = if (comptime g12.isFull(j))
                lw.layer_scalar // placeholder [1] (D4) — toute consommation casse au compile
            else
                self.vs[comptime blk: {
                    @setEvalBranchQuota(100_000); // slidingSlot = boucle comptime O(j) x 48 couches
                    break :blk @as(usize, @intCast(g12.slidingSlot(j)));
                }].toW(.{ .o, .d }),
            .o_proj = lw.o.toW(.{ .d, .m }),
            .post_attention_layernorm = lw.post_attention_layernorm,
            .pre_feedforward_layernorm = lw.pre_feedforward_layernorm,
            .gate_proj = lw.gate.toW(.{ .f, .d }),
            .up_proj = lw.up.toW(.{ .f, .d }),
            .down_proj = lw.down.toW(.{ .d, .f }),
            .post_feedforward_layernorm = lw.post_feedforward_layernorm,
            .per_layer_input_gate = lw.layer_scalar, // placeholders [1] — bloc PLE comptime-mort
            .per_layer_projection = lw.layer_scalar,
            .post_per_layer_input_norm = lw.layer_scalar,
            .layer_scalar = lw.layer_scalar,
        };
    }
};
