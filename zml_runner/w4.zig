// W4 — brique poids 4-bit w4a16 compressed-tensors (jalon J1, spec docs/superpowers/specs/
// 2026-07-18-w4-poids-4bit-12b-design.md, plan docs/superpowers/plans/2026-07-24-w4-j1-brique-e2b.md).
//
// Format (D1, vérifié sur le checkpoint Google 12B) : par Linear [out, in] :
//   weight_packed i32 [out, in/8]  — nibble j du mot w = colonne 8w+j, stocke q+8 (non signé)
//   weight_scale bf16 [out, in/32] — une scale par groupe de 32 le long de la dim d'ENTRÉE
// Dequant = (nibble - 8) * scale, en bf16 (== référence compressed-tensors _dequantize).
//
// Extraction LOGIQUE (shiftRightLogical + AND 0xF) puis -8 : le nibble est q+8 NON SIGNÉ,
// un shift arithmétique propagerait le bit 31 (nibble 7 >= 8) et fausserait les nibbles hauts.
const std = @import("std");
const zml = @import("zml");
const engine = @import("engine.zig");

/// Poids linéaire packé : les 2 tenseurs du store (+ tags canoniques .o/.gp et .o/.g).
pub const W4Lin = struct {
    pk: zml.Tensor, // {.o, .gp} i32, gp = in/8
    sc: zml.Tensor, // {.o, .g}  bf16, g  = in/32

    pub fn init(v: zml.io.TensorStore.View, comptime name: []const u8) W4Lin {
        return .{
            .pk = v.createTensor(name ++ ".weight_packed", .{ .o, .gp }, null),
            .sc = v.createTensor(name ++ ".weight_scale", .{ .o, .g }, null),
        };
    }

    /// dequant + retag vers les tags attendus par le site moteur (ex .{ .d, .m } pour o_proj).
    pub fn toW(self: W4Lin, tagz: anytype) zml.Tensor {
        return dequantW4(self.pk, self.sc).withTags(tagz);
    }
};

/// Unpack seul : {.o, .gp} i32 -> {.o, .d} i32, valeurs q dans [-8, 7]. Exposé pour le gate W1.
pub fn unpackW4(pk: zml.Tensor) zml.Tensor {
    var nibs: [8]zml.Tensor = undefined;
    inline for (0..8) |j| {
        nibs[j] = pk
            .shiftRightLogical(zml.Tensor.scalar(4 * j, .i32))
            .logical(.AND, zml.Tensor.scalar(0xF, .i32));
    }
    // stack en axe mineur .nib : ordre little-endian j=0..7 == colonnes 8w+0..8w+7
    const stacked = zml.Tensor.stack(&nibs, .last, .nib); // {.o, .gp, .nib=8}
    const q = stacked.sub(zml.Tensor.scalar(8, .i32)); // q = nibble - 8
    return q.merge(.{ .d = .{ .gp, .nib } }); // {.o, .d = in}
}

/// Dequant complet : unpack -> bf16 -> * scale par groupe de 32 -> {.o, .d} bf16.
pub fn dequantW4(pk: zml.Tensor, sc: zml.Tensor) zml.Tensor {
    const q = unpackW4(pk).convert(.bf16); // {.o, .d}
    const grouped = q.splitAxis(.d, .{ .g = .auto, .gi = 32 }); // {.o, .g, .gi=32}
    // broadcast auto binaryOp : tags identiques même ordre, sc {.o,.g,.gi=1} -> {.o,.g,.gi=32}
    const deq = grouped.mul(sc.appendAxes(.{.gi}));
    return deq.merge(.{ .d = .{ .g, .gi } }); // {.o, .d}
}

/// K/V/k_norm d'une couche PRODUCER (0-14). Les 20 readers YOCO (15-34) n'ont PAS ces modules
/// dans le checkpoint QAT (num_kv_shared_layers=20) — créer les clés planterait (crash .? sur
/// clé absente, zml/io.zig:168). Struct séparé -> slice de 15, chargée comme le reste.
pub const W4KV = struct {
    k: W4Lin,
    k_norm: zml.Tensor,
    v: W4Lin,

    pub fn init(sa: zml.io.TensorStore.View) W4KV {
        return .{
            .k = .init(sa, "k_proj"),
            .k_norm = sa.createTensor("k_norm.weight", .{.hd}, null),
            .v = .init(sa, "v_proj"),
        };
    }
};

/// Poids d'une couche E2B au format W4 : 7 linears packés + norms/scalar bf16 — SANS k/v
/// (portés par W4KV pour les seuls producers).
pub const W4LayerW = struct {
    input_layernorm: zml.Tensor,
    q: W4Lin,
    q_norm: zml.Tensor,
    o: W4Lin,
    post_attention_layernorm: zml.Tensor,
    pre_feedforward_layernorm: zml.Tensor,
    gate: W4Lin,
    up: W4Lin,
    down: W4Lin,
    post_feedforward_layernorm: zml.Tensor,
    plig: W4Lin,
    plp: W4Lin,
    post_per_layer_input_norm: zml.Tensor,
    layer_scalar: zml.Tensor,

    pub fn init(v: zml.io.TensorStore.View) W4LayerW {
        const sa = v.withPrefix("self_attn");
        const mlp = v.withPrefix("mlp");
        return .{
            .input_layernorm = v.createTensor("input_layernorm.weight", .{.d}, null),
            .q = .init(sa, "q_proj"),
            .q_norm = sa.createTensor("q_norm.weight", .{.hd}, null),
            .o = .init(sa, "o_proj"),
            .post_attention_layernorm = v.createTensor("post_attention_layernorm.weight", .{.d}, null),
            .pre_feedforward_layernorm = v.createTensor("pre_feedforward_layernorm.weight", .{.d}, null),
            .gate = .init(mlp, "gate_proj"),
            .up = .init(mlp, "up_proj"),
            .down = .init(mlp, "down_proj"),
            .post_feedforward_layernorm = v.createTensor("post_feedforward_layernorm.weight", .{.d}, null),
            .plig = .init(v, "per_layer_input_gate"),
            .plp = .init(v, "per_layer_projection"),
            .post_per_layer_input_norm = v.createTensor("post_per_layer_input_norm.weight", .{.d}, null),
            .layer_scalar = v.createTensor("layer_scalar", .{.one}, null),
        };
    }

    /// Dequant en graphe -> LayerW moteur, tags par site == LayerW.init (engine.zig:179-201).
    /// kv = null pour les readers : leurs champs k_proj/k_norm/v_proj reçoivent un PLACEHOLDER
    /// INERTE (tenseur existant, AUCUNE op émise) — jamais consommé au traçage, prouvé par
    /// engine.zig:415-423 (runLayerGen : la branche K/V est dans le `else` de `if (reader)`,
    /// les readers lisent les slots writers). Un placeholder consommé par erreur casserait
    /// au compile (shape/tags faux) — pas de faux silencieux possible.
    pub fn toLayerW(self: W4LayerW, kv: ?W4KV) engine.LayerW {
        return .{
            .input_layernorm = self.input_layernorm,
            .q_proj = self.q.toW(.{ .o, .d }),
            .q_norm = self.q_norm,
            .k_proj = if (kv) |x| x.k.toW(.{ .o, .d }) else self.input_layernorm,
            .k_norm = if (kv) |x| x.k_norm else self.input_layernorm,
            .v_proj = if (kv) |x| x.v.toW(.{ .o, .d }) else self.input_layernorm,
            .o_proj = self.o.toW(.{ .d, .m }),
            .post_attention_layernorm = self.post_attention_layernorm,
            .pre_feedforward_layernorm = self.pre_feedforward_layernorm,
            .gate_proj = self.gate.toW(.{ .f, .d }),
            .up_proj = self.up.toW(.{ .f, .d }),
            .down_proj = self.down.toW(.{ .d, .f }),
            .post_feedforward_layernorm = self.post_feedforward_layernorm,
            .per_layer_input_gate = self.plig.toW(.{ .p, .d }),
            .per_layer_projection = self.plp.toW(.{ .d, .p }),
            .post_per_layer_input_norm = self.post_per_layer_input_norm,
            .layer_scalar = self.layer_scalar,
        };
    }
};

/// Modèle E2B-W4 complet (miroir EngineModel.initWith, engine.zig:531-543).
pub const W4Model = struct {
    embed_tokens: zml.Tensor, // bf16 (ignore re:.*embed.*)
    plmp: W4Lin, // per_layer_model_projection : quantifié
    per_layer_projection_norm: zml.Tensor,
    final_norm: zml.Tensor,
    layers: []W4LayerW, // 35 (champs communs)
    kv: []W4KV, // 15 (producers 0-14 seulement)

    pub fn init(allocator: std.mem.Allocator, base: zml.io.TensorStore.View) !W4Model {
        const layers = try allocator.alloc(W4LayerW, engine.NUM_LAYERS);
        const kv = try allocator.alloc(W4KV, engine.FIRST_KV_SHARED);
        const lb = base.withPrefix("layers");
        for (layers, 0..) |*l, i| l.* = W4LayerW.init(lb.withLayer(i));
        for (kv, 0..) |*x, i| x.* = W4KV.init(lb.withLayer(i).withPrefix("self_attn"));
        return .{
            .embed_tokens = base.createTensor("embed_tokens.weight", .{ .voc, .d }, null),
            .plmp = .init(base, "per_layer_model_projection"),
            .per_layer_projection_norm = base.createTensor("per_layer_projection_norm.weight", .{.p}, null),
            .final_norm = base.createTensor("norm.weight", .{.d}, null),
            .layers = layers,
            .kv = kv,
        };
    }

    pub fn load(self: *const W4Model, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, store: *const zml.io.TensorStore, shardings: []const zml.sharding.Sharding) !zml.Bufferized(W4Model) {
        return zml.io.load(W4Model, self, allocator, io, platform, store, .{ .shardings = shardings, .parallelism = 1, .dma_chunks = 1, .dma_chunk_size = 16 * 1024 * 1024 });
    }
};
