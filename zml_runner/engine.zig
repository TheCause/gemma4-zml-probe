// Socle ZML modulaire — moteur decode invariant `EngineModel(comptime Brick)`.
//
// Extrait op-pour-op de gemma4_decode4.zig (P5.7.8) : même boucle de génération, même cache threadé,
// mêmes 35 couches, mêmes seuils numériques. La SEULE addition est un point d'extension comptime
// `post_v_norm` dans runLayerGen : si la brique l'implémente, V (post-v_norm, pré-cache) passe par elle ;
// sinon (struct{}) la branche est comptime-morte → graphe MLIR identique à decode4 → E1 bit-exact.
//
// decode4.zig reste intact (oracle de non-régression E1). Une brique = un type Zig avec ses constantes
// en champs Tensor + une méthode `post_v_norm(self, v, ctx) Tensor`. quantizeV (prouvée en Q3) est ici
// `pub` pour être partagée par les briques.

const std = @import("std");
const zml = @import("zml");
const log = std.log;

/// Géométrie du modèle — comptime. Défaut = E2B (neutralité prouvée par gate U1, HLO md5).
/// Portée par EngineCfg (`geom: Geom = e2b`) : les sites du moteur lisent cfg.geom, les alias
/// publics historiques (ci-dessous) restent exportés avec leurs TYPES d'origine (i64/f64).
pub const Geom = struct {
    num_layers: usize,
    first_kv_shared: usize, // = num_layers si pas de YOCO (aucun reader)
    sliding_writer: usize, // idx couche sliding qui écrit le K/V partagé YOCO ; ignoré si first_kv_shared == num_layers
    full_writer: usize, // idem pour la couche full (les readers lisent son slot)
    d: usize,
    nh: usize,
    kvh_sliding: usize, // E2B 1 (MQA), 12B 8 (GQA groupe 2)
    kvh_full: usize, // 1 partout (MQA)
    hd_sliding: usize,
    hd_full: usize,
    ple_dim: usize, // 0 => bloc PLE comptime-mort
    full_period: usize, // isFull(i) = (i+1) % full_period == 0  (E2B 5, 12B 6)
    k_eq_v_full: bool, // 12B : V = v_norm(k_proj brut) sur les couches full
    rope_theta_sliding: f32,
    softcap: f32,

    pub fn isFull(comptime g: Geom, comptime i: usize) bool {
        return (i + 1) % g.full_period == 0;
    }
    pub fn isReader(comptime g: Geom, comptime i: usize) bool {
        return i >= g.first_kv_shared;
    }
    /// Dérivations en f64 OBLIGATOIREMENT (les constantes historiques sont f64 ; une dérivation
    /// f32 changerait le scalaire HLO d'un ULP → md5 U1 FAIL par construction).
    pub fn embedScale(comptime g: Geom) f64 {
        return @sqrt(@as(f64, @floatFromInt(g.d)));
    }
    pub fn invSqrtHid(comptime g: Geom) f64 {
        return 1.0 / @sqrt(@as(f64, @floatFromInt(g.d)));
    }
    pub fn slidingSlot(comptime g: Geom, comptime i: usize) i64 {
        comptime var slot: i64 = 0;
        comptime var j: usize = 0;
        inline while (j < i) : (j += 1) {
            if (comptime !g.isFull(j)) slot += 1;
        }
        return slot;
    }
    pub fn fullSlot(comptime g: Geom, comptime i: usize) i64 {
        comptime var slot: i64 = 0;
        comptime var j: usize = 0;
        inline while (j < i) : (j += 1) {
            if (comptime g.isFull(j)) slot += 1;
        }
        return slot;
    }
};

/// Géométrie E2B — les valeurs historiques d'engine.zig, à l'identique (gate U1).
pub const e2b: Geom = .{
    .num_layers = 35,
    .first_kv_shared = 15,
    .sliding_writer = 13,
    .full_writer = 14,
    .d = 1536,
    .nh = 8,
    .kvh_sliding = 1,
    .kvh_full = 1,
    .hd_sliding = 256,
    .hd_full = 512,
    .ple_dim = 256,
    .full_period = 5,
    .k_eq_v_full = false,
    .rope_theta_sliding = 1.0e4,
    .softcap = 30.0,
};

// Alias publics conservés AVEC LEURS TYPES D'ORIGINE (i64/f64) — les runners existants
// (w4.zig, gemma4_w4auto.zig…) compilent sans modification. Valeurs = e2b.
pub const NUM_LAYERS: usize = e2b.num_layers;
pub const FIRST_KV_SHARED: usize = e2b.first_kv_shared;
pub const SLIDING_WRITER: usize = e2b.sliding_writer;
pub const FULL_WRITER: usize = e2b.full_writer;
pub const SLIDING_WRITER_SLOT: i64 = e2b.slidingSlot(e2b.sliding_writer); // 11
pub const FULL_WRITER_SLOT: i64 = e2b.fullSlot(e2b.full_writer); // 2

// B/S ne sont PLUS consommés par le moteur : depuis le gate batch T0, les 5 sites de reshape
// (q/k/v + les 2 du PLE) dérivent leurs dims batch/seq des SHAPES D'ENTRÉE (`h0.dim(.b)`,
// `embeds.dim(.s)`…) → le graphe est shape-polymorphe, un binaire unique sert tous les B.
// Déclarations gardées pour les runners existants (qui construisent encore des shapes à 1).
pub const B: i64 = 1;
pub const S: i64 = 1;
pub const D: i64 = @intCast(e2b.d);
pub const NH: i64 = @intCast(e2b.nh);
pub const KVH: i64 = @intCast(e2b.kvh_sliding);
pub const HD_SLIDING: i64 = @intCast(e2b.hd_sliding);
pub const HD_FULL: i64 = @intCast(e2b.hd_full);
pub const PLE_DIM: i64 = @intCast(e2b.ple_dim);

const RMS_EPS: f32 = 1.0e-6;
const ROPE_THETA_SLIDING: f32 = e2b.rope_theta_sliding;
const EMBED_SCALE: f64 = e2b.embedScale();
const INV_SQRT_HID: f64 = e2b.invSqrtHid();
const SQRT_PLE: f64 = 16.0;
const INV_SQRT_2: f64 = 0.7071067811865476;
const SOFTCAP: f64 = 30.0;
const INV_SOFTCAP: f64 = 1.0 / 30.0;

/// Wrappers historiques (géométrie E2B) — API préservée pour les runners existants ; le moteur,
/// lui, lit cfg.geom (méthodes Geom, comptime).
pub fn isFull(i: usize) bool {
    return (i + 1) % e2b.full_period == 0;
}
pub fn isReader(i: usize) bool {
    return i >= FIRST_KV_SHARED;
}

// NB : l'upcast prec-aware est `cvt(t, dt)` file-scope, dtype passé EXPLICITEMENT (G2.3 : prec est
// runtime, lu de self.prec). L'ancien pattern `const c = struct { fn call... }.call` ne marchait que
// parce que `prec` était comptime : un struct-fn Zig ne capture PAS les variables englobantes runtime.
// Neutralité : cvt(t, .f32) sur un tensor déjà f32 = `return self` (cf tensor.zig convert) = no-op HLO.
fn cvt(t: zml.Tensor, dt: zml.DataType) zml.Tensor {
    return t.convert(dt);
}
/// Encadre une valeur aux bornes d'une op non-GEMM (G2.3, spec §4) : arrondi d'entrée si la
/// famille est active. La sortie de l'op est re-upcastée par l'appelant (via .convert(compute)).
/// fam=null : `return x` — aucune op émise (neutralité).
fn inPrec(fam: ?zml.DataType, x: zml.Tensor) zml.Tensor {
    if (fam) |g| return x.convert(g);
    return x;
}
fn rmsScaleD(x: zml.Tensor, w: zml.Tensor) zml.Tensor {
    return zml.nn.rmsNorm(x, .d, RMS_EPS).mul(w.broad(x.shape()));
}
/// rmsScaleD prec-aware (famille norms, spec §4) : entrée ET poids arrondis, sortie re-upcastée.
/// `w` = poids BRUT du checkpoint (le cvt(w, compute) d'origine reste la base, ordre préservé).
/// fam=null : émission STRICTEMENT identique à `rmsScaleD(x, cvt(w, compute))` d'aujourd'hui
/// (convert same-dtype = `return self`). NB : zml.nn.rmsNorm upcaste en f32 en interne puis
/// re-converge au dtype d'entrée — bornes arrondies, calcul interne au gré d'XLA (le contrat).
/// ORDRE D'ÉMISSION : ici c'est l'INVERSE de rmsScaleHdPrec — `wi` (convert du poids) est émis
/// AVANT le rmsNorm (dans rmsScaleD), ne pas réordonner : une harmonisation naïve des deux
/// helpers casserait le byte-diff HLO (gate G2.3.0).
fn rmsScaleDPrec(fam: ?zml.DataType, compute: zml.DataType, x: zml.Tensor, w: zml.Tensor) zml.Tensor {
    const xi = inPrec(fam, x);
    const wi = inPrec(fam, w.convert(compute)); // le cvt() d'origine reste la base
    return rmsScaleD(xi, wi).convert(compute);
}
/// q_norm/k_norm (rmsNorm .hd + mul poids) prec-aware — même contrat que rmsScaleDPrec.
/// fam=null : identique à `zml.nn.rmsNorm(x, .hd, eps).mul(cvt(w, compute).broad(x.shape()))`.
/// ORDRE D'ÉMISSION (gate byte-diff HLO pré-opt) : l'inline d'origine évaluait le receiver rmsNorm
/// AVANT le convert du poids (argument du mul) — le poids checkpoint est bf16, son convert f32 est
/// une VRAIE op même en fam=null → `normalized` DOIT être construit avant `wi`, ne pas réordonner.
fn rmsScaleHdPrec(fam: ?zml.DataType, compute: zml.DataType, x: zml.Tensor, w: zml.Tensor) zml.Tensor {
    const xi = inPrec(fam, x);
    const normalized = zml.nn.rmsNorm(xi, .hd, RMS_EPS);
    const wi = inPrec(fam, w.convert(compute));
    return normalized.mul(wi.broad(xi.shape())).convert(compute);
}
fn rmsScaleP(x: zml.Tensor, w: zml.Tensor) zml.Tensor {
    const n = zml.nn.rmsNorm(x, .p, RMS_EPS);
    return n.mul(w.broad(n.shape()));
}
/// RoPE sliding prec-aware (famille rope). zml.nn.rope NATIF : ses cos/sin internes sont générés
/// en f32 puis convertis au dtype de x (cf zml nn.zig:rope) → arrondir x arrondit AUSSI cos/sin
/// (contrat : q/k ET cos/sin), sans patcher l'intérieur de zml. fam=null : identique + no-op convert.
fn slidingRope(fam: ?zml.DataType, compute: zml.DataType, x: zml.Tensor, pos: zml.Tensor, theta: f32) zml.Tensor {
    const xi = inPrec(fam, x);
    return zml.nn.rope(xi, pos, .{ .layout = .sequential, .scaling = .{ .default = .{ .rope_theta = theta } } }).convert(compute);
}
/// RoPE full prec-aware (famille rope) : q/k ET cos/sin arrondis en entrée, sortie re-upcastée.
/// fam=null : ops identiques à l'ancien manualRope + no-op convert.
fn manualRope(fam: ?zml.DataType, compute: zml.DataType, x: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor, half: i64) zml.Tensor {
    const xi = inPrec(fam, x);
    const cosi = inPrec(fam, cos);
    const sini = inPrec(fam, sin);
    const halves = xi.split(.hd, &.{ half, half });
    const rh = zml.Tensor.concatenate(&.{ halves[1].negate(), halves[0] }, .hd);
    return xi.mul(cosi.broad(xi.shape())).add(rh.mul(sini.broad(xi.shape()))).convert(compute);
}
/// Softcap cap·tanh(x/cap) prec-aware (famille softcap) : entrée arrondie, sortie re-upcastée.
/// `scale` émet sa constante au dtype du tensor (cf tensor.zig) → le chemin bf16 reste homogène.
/// `cap` vient de geom.softcap (f32 → f64 exact pour 30.0) : émission identique à l'historique
/// `raw.scale(INV_SOFTCAP).tanh().scale(SOFTCAP)` (gate U1).
fn softcapPrec(fam: ?zml.DataType, compute: zml.DataType, raw: zml.Tensor, cap: f64) zml.Tensor {
    return inPrec(fam, raw).scale(1.0 / cap).tanh().scale(cap).convert(compute);
}

/// Contexte (runtime) passé à un point d'extension, pour info/extensibilité. `is_full` n'y est PAS :
/// il est passé en paramètre **comptime** séparé (cf hook) car il sélectionne entre des constantes de
/// **shapes différentes** (codebook/Hadamard 256 vs 512) — un select runtime exigerait des shapes égales.
/// `pos` non inclus : YAGNI (aucune brique ne le consomme). S'ajoutent trivialement au besoin.
pub const LayerCtx = struct { layer_idx: usize };

/// Chaîne MSE V-only prouvée en Q3 (norm fp16 + Hadamard + nearest-centroid + inverse).
/// v:[.k,.hd], cb:[.c], Pi:[.e,.hd] -> v_hat:[.k,.hd]. `pub` pour partage par les briques.
pub fn quantizeV(v: zml.Tensor, cb: zml.Tensor, Pi: zml.Tensor) zml.Tensor {
    const norm = v.mul(v).sum(.hd).sqrt().convert(.f16).convert(.f32); // [.k,.hd=1]
    const u = v.div(norm); // broadcast (.hd=1)
    const y = u.dot(Pi, .hd); // [.k,.e]
    const target = zml.Shape.init(.{ y.dim(.k), y.dim(.e), cb.dim(.c) }, .f32)
        .withTags(.{ .k, .e, .c });
    const yr3 = y.appendAxes(.{.c}).broad(target);
    const cb3 = cb.insertAxes(0, .{ .k, .e }).broad(target);
    const diff = yr3.sub(cb3);
    const idx = diff.mul(diff).scale(-1.0).argMax(.c).indices.squeeze(.c); // [.k,.e]
    const y_hat = cb.gather(.{ .c = idx }, .{}); // [.k,.e]
    const u_hat = y_hat.dot(Pi, .e); // [.k,.hd]
    return u_hat.mul(norm);
}

pub const LayerW = struct {
    input_layernorm: zml.Tensor,
    q_proj: zml.Tensor,
    q_norm: zml.Tensor,
    k_proj: zml.Tensor,
    k_norm: zml.Tensor,
    v_proj: zml.Tensor,
    o_proj: zml.Tensor,
    post_attention_layernorm: zml.Tensor,
    pre_feedforward_layernorm: zml.Tensor,
    gate_proj: zml.Tensor,
    up_proj: zml.Tensor,
    down_proj: zml.Tensor,
    post_feedforward_layernorm: zml.Tensor,
    per_layer_input_gate: zml.Tensor,
    per_layer_projection: zml.Tensor,
    post_per_layer_input_norm: zml.Tensor,
    layer_scalar: zml.Tensor,

    pub fn init(v: zml.io.TensorStore.View) LayerW {
        const sa = v.withPrefix("self_attn");
        const mlp = v.withPrefix("mlp");
        return .{
            .input_layernorm = v.createTensor("input_layernorm.weight", .{.d}, null),
            .q_proj = sa.createTensor("q_proj.weight", .{ .o, .d }, null),
            .q_norm = sa.createTensor("q_norm.weight", .{.hd}, null),
            .k_proj = sa.createTensor("k_proj.weight", .{ .o, .d }, null),
            .k_norm = sa.createTensor("k_norm.weight", .{.hd}, null),
            .v_proj = sa.createTensor("v_proj.weight", .{ .o, .d }, null),
            .o_proj = sa.createTensor("o_proj.weight", .{ .d, .m }, null),
            .post_attention_layernorm = v.createTensor("post_attention_layernorm.weight", .{.d}, null),
            .pre_feedforward_layernorm = v.createTensor("pre_feedforward_layernorm.weight", .{.d}, null),
            .gate_proj = mlp.createTensor("gate_proj.weight", .{ .f, .d }, null),
            .up_proj = mlp.createTensor("up_proj.weight", .{ .f, .d }, null),
            .down_proj = mlp.createTensor("down_proj.weight", .{ .d, .f }, null),
            .post_feedforward_layernorm = v.createTensor("post_feedforward_layernorm.weight", .{.d}, null),
            .per_layer_input_gate = v.createTensor("per_layer_input_gate.weight", .{ .p, .d }, null),
            .per_layer_projection = v.createTensor("per_layer_projection.weight", .{ .d, .p }, null),
            .post_per_layer_input_norm = v.createTensor("post_per_layer_input_norm.weight", .{.d}, null),
            .layer_scalar = v.createTensor("layer_scalar", .{.one}, null),
        };
    }
};

// Cache empaqueté threadé (entrée + sortie du forward) : 2 types (sliding/full) × {slot,b,h,k,hd}.
pub const Cache = struct {
    sl_k: zml.Tensor,
    sl_v: zml.Tensor,
    fl_k: zml.Tensor,
    fl_v: zml.Tensor,

    pub fn init(v: zml.io.TensorStore.View) Cache {
        return .{
            .sl_k = v.createTensor("cache_sl_k", .{ .slot, .b, .h, .k, .hd }, null),
            .sl_v = v.createTensor("cache_sl_v", .{ .slot, .b, .h, .k, .hd }, null),
            .fl_k = v.createTensor("cache_fl_k", .{ .slot, .b, .h, .k, .hd }, null),
            .fl_v = v.createTensor("cache_fl_v", .{ .slot, .b, .h, .k, .hd }, null),
        };
    }

    pub fn load(self: *const Cache, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, store: *const zml.io.TensorStore, shardings: []const zml.sharding.Sharding) !zml.Bufferized(Cache) {
        return zml.io.load(Cache, self, allocator, io, platform, store, .{ .shardings = shardings, .parallelism = 1, .dma_chunks = 1, .dma_chunk_size = 16 * 1024 * 1024 });
    }

    /// Garde G2.3 (famille kv_store, mécanisme b) : le dtype de STOCKAGE du cache vient du header
    /// de la fixture (createTensor n'override pas le dtype) — vérifier AVANT compile que fixture et
    /// prec sont cohérents : kv_store actif ⇒ fixture variante à ce dtype (scripts/46 --kv-dtype
    /// bf16) ; kv_store null ⇒ fixture au dtype compute (f32). Un mismatch casserait sinon en aval
    /// avec une erreur obscure (scatter : update typé au dtype de l'opérande, cf zml ops.zig ;
    /// rethread step→step : sortie ≠ dtype d'entrée déclaré).
    pub fn checkDtype(self: *const Cache, prec: PrecRt) !void {
        const expected: zml.DataType = prec.kv_store orelse prec.compute;
        const entries = [_]struct { []const u8, zml.DataType }{
            .{ "cache_sl_k", self.sl_k.dtype() },
            .{ "cache_sl_v", self.sl_v.dtype() },
            .{ "cache_fl_k", self.fl_k.dtype() },
            .{ "cache_fl_v", self.fl_v.dtype() },
        };
        for (entries) |e| {
            if (e[1] != expected) {
                log.err("{s}: dtype fixture = {s} ≠ attendu = {s} (prec.kv_store {s}) — fixture standard f32 pour kv_store=null, variante (scripts/46 --kv-dtype bf16) pour kv_store=bf16", .{ e[0], @tagName(e[1]), @tagName(expected), if (prec.kv_store != null) "actif" else "null" });
                return error.KvCacheDtypeMismatch;
            }
        }
    }
};

// Entrées par-step empaquetées (constantes sur la boucle ; sélectionnées par dynamicSlice(.step)).
// Paramétré comptime par `MaskMode` (spec masques in-graph 2026-07-26 §4.2 ; ex-`two_masks: bool`) :
//   - .single (défaut) : un seul masque `masks` — strictement identique à l'ancien `Packed(false)`,
//     la fixture E1/E2 (KMAX=8) charge inchangée et le graphe HLO est préservé.
//   - .tables (génération longue) : deux masques `masks_sliding`/`masks_full` — ancien `Packed(true)`.
//   - .ingraph : PAS de tables masques (elles étaient {L_MAX,L_MAX} quadratiques) ; à la place un
//     scalaire runtime `window` ({} i32, la fenêtre glissante en DONNÉE — rebindable pour le
//     contre-test de vacuité). Les lignes de masque sont générées DANS le graphe (ingraphMaskLines).
// `zml.io.load` réfléchit RÉCURSIVEMENT sur les champs (chacun doit être un Tensor) → on retourne TROIS
// structs distincts (pas de champ `void` conditionnel, que load ne saurait pas traiter).
pub const MaskMode = enum { single, tables, ingraph };

// Valeur de masquage additive (== maskRows des runners : {0, MASK_MIN} STRICTEMENT — jamais
// d'addition de deux masques : -floatMax + -floatMax = -inf ≠ MASK_MIN, l'équivalence bit casserait).
pub const MASK_MIN: f32 = -std.math.floatMax(f32);

pub fn Packed(comptime mode: MaskMode) type {
    if (mode == .ingraph) return struct {
        embeds: zml.Tensor, // {step,b,s,d} bf16
        embptls: zml.Tensor, // {step,b,s,lf} bf16
        cos_full: zml.Tensor, // {step,b,s,hd=512}
        sin_full: zml.Tensor,
        positions: zml.Tensor, // {step} i32
        window: zml.Tensor, // {} i32 — fenêtre glissante runtime (12B : 1024) ; L_MAX = non mordante

        const Self = @This();
        pub fn init(v: zml.io.TensorStore.View) Self {
            return .{
                .embeds = v.createTensor("embeds", .{ .step, .b, .s, .d }, null),
                .embptls = v.createTensor("embptls", .{ .step, .b, .s, .lf }, null),
                .cos_full = v.createTensor("cos_full", .{ .step, .b, .s, .hd }, null),
                .sin_full = v.createTensor("sin_full", .{ .step, .b, .s, .hd }, null),
                .positions = v.createTensor("positions", .{.step}, null),
                .window = v.createTensor("window", .{}, null),
            };
        }
        pub fn load(self: *const Self, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, store: *const zml.io.TensorStore, shardings: []const zml.sharding.Sharding) !zml.Bufferized(Self) {
            return zml.io.load(Self, self, allocator, io, platform, store, .{ .shardings = shardings, .parallelism = 1, .dma_chunks = 1, .dma_chunk_size = 16 * 1024 * 1024 });
        }
    };
    if (mode == .tables) return struct {
        embeds: zml.Tensor, // {step,b,s,d} bf16
        embptls: zml.Tensor, // {step,b,s,lf} bf16
        cos_full: zml.Tensor, // {step,b,s,hd=512}
        sin_full: zml.Tensor,
        masks_sliding: zml.Tensor, // {step,b,h,q,k=KMAX_SLIDING} — fenêtre glissante
        masks_full: zml.Tensor, // {step,b,h,q,k=KMAX_FULL} — causal plein
        positions: zml.Tensor, // {step} i32

        const Self = @This();
        pub fn init(v: zml.io.TensorStore.View) Self {
            return .{
                .embeds = v.createTensor("embeds", .{ .step, .b, .s, .d }, null),
                .embptls = v.createTensor("embptls", .{ .step, .b, .s, .lf }, null),
                .cos_full = v.createTensor("cos_full", .{ .step, .b, .s, .hd }, null),
                .sin_full = v.createTensor("sin_full", .{ .step, .b, .s, .hd }, null),
                .masks_sliding = v.createTensor("masks_sliding", .{ .step, .b, .h, .q, .k }, null),
                .masks_full = v.createTensor("masks_full", .{ .step, .b, .h, .q, .k }, null),
                .positions = v.createTensor("positions", .{.step}, null),
            };
        }
        pub fn load(self: *const Self, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, store: *const zml.io.TensorStore, shardings: []const zml.sharding.Sharding) !zml.Bufferized(Self) {
            return zml.io.load(Self, self, allocator, io, platform, store, .{ .shardings = shardings, .parallelism = 1, .dma_chunks = 1, .dma_chunk_size = 16 * 1024 * 1024 });
        }
    };
    return struct {
        embeds: zml.Tensor, // {step,b,s,d} bf16
        embptls: zml.Tensor, // {step,b,s,lf} bf16
        cos_full: zml.Tensor, // {step,b,s,hd=512}
        sin_full: zml.Tensor,
        masks: zml.Tensor, // {step,b,h,q,k}
        positions: zml.Tensor, // {step} i32

        const Self = @This();
        pub fn init(v: zml.io.TensorStore.View) Self {
            return .{
                .embeds = v.createTensor("embeds", .{ .step, .b, .s, .d }, null),
                .embptls = v.createTensor("embptls", .{ .step, .b, .s, .lf }, null),
                .cos_full = v.createTensor("cos_full", .{ .step, .b, .s, .hd }, null),
                .sin_full = v.createTensor("sin_full", .{ .step, .b, .s, .hd }, null),
                .masks = v.createTensor("masks", .{ .step, .b, .h, .q, .k }, null),
                .positions = v.createTensor("positions", .{.step}, null),
            };
        }
        pub fn load(self: *const Self, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, store: *const zml.io.TensorStore, shardings: []const zml.sharding.Sharding) !zml.Bufferized(Self) {
            return zml.io.load(Self, self, allocator, io, platform, store, .{ .shardings = shardings, .parallelism = 1, .dma_chunks = 1, .dma_chunk_size = 16 * 1024 * 1024 });
        }
    };
}

// Compteur de step (scalaire u32, fourni/threadé par le main).
pub const Ctrl = struct {
    step: zml.Tensor,
    pub fn initSymbolic() Ctrl {
        return .{ .step = zml.Tensor.init(.{}, .u32) };
    }
};

fn pickStep(t: zml.Tensor, step: zml.Tensor) zml.Tensor {
    return t.dynamicSlice(.{ .step = zml.Tensor.DynSlice{ .start = step, .len = 1 } }).squeeze(.step);
}

// Masques in-graph (spec 2026-07-26 §4.3) : les deux lignes de masque additif du step courant,
// générées depuis positions[step] et le scalaire runtime `window` — remplace les tables host
// {L_MAX,L_MAX} du mode .tables (seul terme quadratique du design). Valeurs STRICTEMENT
// {0, MASK_MIN}, bornes identiques à maskRows (g12auto) : full = j <= p ; sliding = j <= p ET
// j >= p-(window-1). AND par select imbriqué (mêmes valeurs qu'un and booléen). Le pickStep
// positions est refait ici (le pos_i des forwards est extrait APRÈS les masques — on ne
// réordonne PAS les modes existants, byte-identité HLO oblige ; XLA dédupliquera).
// cmp broadcaste les scalaires rank-0 nativement (tensor.zig cmp) ; dtypes i32 alignés (iota
// {k <= maxInt(i32)} → i32, positions i32, window i32).
fn ingraphMaskLines(comptime kmax_sliding: i64, comptime kmax_full: i64, positions: zml.Tensor, window: zml.Tensor, step: zml.Tensor) struct { zml.Tensor, zml.Tensor } {
    const pos = pickStep(positions, step); // {} i32
    // full {k} : j <= p
    const iota_f = zml.Tensor.iota(zml.Shape.init(.{ .k = kmax_full }, .i32), .k);
    const le_f = iota_f.cmp(.LE, pos);
    const zero_f = zml.Tensor.scalar(0, .f32).broad(zml.Shape.init(.{ .k = kmax_full }, .f32).withTags(.{.k}));
    const min_f = zml.Tensor.scalar(MASK_MIN, .f32).broad(zml.Shape.init(.{ .k = kmax_full }, .f32).withTags(.{.k}));
    const full_line = le_f.select(zero_f, min_f);
    // sliding {k} : j <= p ET j >= p - (window-1)
    const iota_s = zml.Tensor.iota(zml.Shape.init(.{ .k = kmax_sliding }, .i32), .k);
    const le_s = iota_s.cmp(.LE, pos);
    const lo = pos.sub(window).addConstant(1); // {} i32 : p - window + 1
    const ge_s = iota_s.cmp(.GE, lo);
    const zero_s = zml.Tensor.scalar(0, .f32).broad(zml.Shape.init(.{ .k = kmax_sliding }, .f32).withTags(.{.k}));
    const min_s = zml.Tensor.scalar(MASK_MIN, .f32).broad(zml.Shape.init(.{ .k = kmax_sliding }, .f32).withTags(.{.k}));
    const sliding_line = le_s.select(ge_s.select(zero_s, min_s), min_s);
    // {k} → {b=1,h=1,q=1,k} : reshape layout-preserving + re-tag (pièges ZML #1/#2)
    return .{
        sliding_line.reshape(.{ 1, 1, 1, kmax_sliding }).withTags(.{ .b, .h, .q, .k }),
        full_line.reshape(.{ 1, 1, 1, kmax_full }).withTags(.{ .b, .h, .q, .k }),
    };
}

/// Config comptime du socle (cf engine DESIGN §3, §5). TOUS les champs ont une valeur par défaut qui
/// reproduit le comportement decode4/E1 : `EngineModel(Brick, .{})` est strictement neutre (aucune op
/// nouvelle émise) → graphe HLO byte-identique. La génération longue active `ring`/`two_masks` et fixe
/// les tailles de fenêtre. `kmax_sliding` n'est utilisé que comme scalaire du modulo ring (la dim `.k`
/// du cache, elle, est inférée de la fixture).
pub const EngineCfg = struct {
    // Géométrie du modèle (comptime). Défaut e2b = comportement historique à l'identique
    // (gate U1 : HLO byte-identique). ⚠ piège 4 : le struct dans la VALEUR cfg allonge le
    // @typeName des instanciations — parade pré-enregistrée si quota pjrt : GeomTag enum.
    geom: Geom = e2b,
    ring: bool = false, // scatter sliding circulaire pos % kmax_sliding
    two_masks: bool = false, // masque par type de couche (sliding/full) au lieu d'un masque unique
    kmax_sliding: i64 = 8, // modulo du ring-buffer sliding
    kmax_full: i64 = 8, // (info ; la dim full vient de la fixture)
    // Variante d'attention (chantier batching, Phase 2). `.manual` = chemin actuel roulé main
    // (scores bruts exposés → PrecRt s'y accroche) ; `.sdpa` = zml.nn.sdpa (chemin « produit »).
    // Le défaut `.manual` est strictement neutre : la branche `.sdpa` est comptime-morte → HLO
    // byte-identique (gate S1). NB : `.sdpa` NEUTRALISE les familles qk_scores/softmax/pv_ctx de
    // PrecRt (sdpa fait ses propres converts) → gates sdpa en fp32 pur, cf spec §3.4.
    attn: AttnKind = .manual,
    // Masques in-graph (spec 2026-07-26 §4.1) : les lignes de masque sont générées dans le graphe
    // (ingraphMaskLines) au lieu d'être lues des tables {L_MAX,L_MAX} de Packed(.tables). Exige
    // two_masks=true. Défaut false = strictement neutre (branches comptime-mortes, gate M0).
    ingraph_masks: bool = false,

    /// Mode Packed dérivé de la cfg (gardes comptime incluses).
    pub fn maskMode(comptime cfg: EngineCfg) MaskMode {
        if (cfg.ingraph_masks) {
            if (!cfg.two_masks) @compileError("ingraph_masks exige two_masks=true (chemin mono-masque E2B non couvert — spec §4.1)");
            return .ingraph;
        }
        return if (cfg.two_masks) .tables else .single;
    }
};

pub const AttnKind = enum { manual, sdpa };

/// Config de précision RUNTIME (G2.3, approche B de la spec). Un champ par FAMILLE d'ops ;
/// `null` = f32 (baseline). Portée par le modèle comme CHAMP RUNTIME (self.prec) — plus dans
/// EngineCfg comptime : le traçage émet les converts d'après la valeur au moment du compile,
/// le binaire est unique, le graphe diffère par run. Sémantique contractuelle (spec §4) :
/// « bf16 » = arrondi des opérandes aux bornes de l'op, calcul interne au gré d'XLA.
/// NEUTRALITÉ : tout-null doit émettre un graphe identique à la baseline (convert same-dtype
/// = `return self`, cf tensor.zig).
pub const PrecRt = struct {
    compute: zml.DataType = .f32,
    // familles GEMM (spec §4, 1-7)
    qkv_proj: ?zml.DataType = null,
    qk_scores: ?zml.DataType = null,
    pv_ctx: ?zml.DataType = null,
    o_proj: ?zml.DataType = null,
    mlp: ?zml.DataType = null,
    ple: ?zml.DataType = null,
    head: ?zml.DataType = null,
    // familles non-GEMM (spec §4, 8-12) — norms/softmax/rope/softcap câblées (Task 2) ;
    // kv_store câblée (Task 3, mécanisme b) : le DTYPE DE STOCKAGE vient du header de la fixture
    // (variante bf16 via scripts/46 --kv-dtype bf16) ; kv_store actif ⇒ writes arrondis avant
    // scatter + reads re-upcastés (runLayerGen). Cohérence fixture↔prec : Cache.checkDtype.
    norms: ?zml.DataType = null,
    softmax: ?zml.DataType = null,
    rope: ?zml.DataType = null,
    softcap: ?zml.DataType = null,
    kv_store: ?zml.DataType = null,

    /// Parse "fam1,fam2" (noms des champs) → PrecRt avec ces familles à .bf16. Erreur si nom inconnu.
    pub fn fromSpecList(list: []const u8) !PrecRt {
        var p: PrecRt = .{};
        if (list.len == 0) return p;
        var it = std.mem.splitScalar(u8, list, ',');
        while (it.next()) |name| {
            var matched = false;
            inline for (@typeInfo(PrecRt).@"struct".fields) |f| {
                if (comptime std.mem.eql(u8, f.name, "compute")) continue;
                if (std.mem.eql(u8, name, f.name)) {
                    @field(p, f.name) = .bf16;
                    matched = true;
                }
            }
            if (!matched) {
                log.err("famille inconnue: '{s}'", .{name});
                return error.UnknownFamily;
            }
        }
        return p;
    }
};

/// GEMM prec-aware par famille (G2.3). `fam` = le champ PrecRt de la famille du site d'appel.
/// fam=null : émission STRICTEMENT identique à `a.dot(convert(b))` d'aujourd'hui (neutralité).
fn dotPrec(fam: ?zml.DataType, compute: zml.DataType, a: zml.Tensor, b: zml.Tensor, comptime axis: @TypeOf(.enum_literal)) zml.Tensor {
    if (fam) |g| return a.convert(g).dot(b.convert(g), axis).convert(compute);
    return a.dot(b.convert(compute), axis);
}

/// Forward decode d'UNE couche i en mode génération (cache threadé). Producer scatter K/V du token à
/// (.slot, .k=pos) dans le cache empaqueté ; reader lit le slot du writer 13/14. `brick` est threadé
/// pour le point d'extension post_v_norm (comptime-mort si la brique ne l'implémente pas). `cfg` est
/// comptime : ses branches inactives ne sont pas émises (neutralité HLO en config défaut).
/// `prec` est RUNTIME (G2.3) : la valeur au moment du traçage décide des converts émis.
fn runLayerGen(layer: LayerW, comptime i: usize, comptime cfg: EngineCfg, prec: PrecRt, hidden: zml.Tensor, ple_i: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor, mask: zml.Tensor, pos_s: zml.Tensor, pos_u: zml.Tensor, cache: *Cache, brick: anytype) zml.Tensor {
    const geom = cfg.geom;
    const full = comptime geom.isFull(i);
    const reader = comptime geom.isReader(i);
    const hd: i64 = if (full) @intCast(geom.hd_full) else @intCast(geom.hd_sliding);
    const half: i64 = @divExact(hd, 2);
    const nh: i64 = @intCast(geom.nh);
    // D5 (GQA 12B) : kvh par TYPE de couche — E2B : 1 partout (MQA), même HLO (gate U1).
    const kvh: i64 = if (full) @intCast(geom.kvh_full) else @intCast(geom.kvh_sliding);

    const h0 = rmsScaleDPrec(prec.norms, prec.compute, hidden, layer.input_layernorm);

    var q = dotPrec(prec.qkv_proj, prec.compute, h0, layer.q_proj, .d).reshape(.{ h0.dim(.b), h0.dim(.s), nh, hd }).withTags(.{ .b, .s, .nh, .hd });
    q = rmsScaleHdPrec(prec.norms, prec.compute, q, layer.q_norm);
    q = if (full) manualRope(prec.rope, prec.compute, q, cos, sin, half) else slidingRope(prec.rope, prec.compute, q, pos_s, geom.rope_theta_sliding);
    const q_final = q.transpose(.{ .b, .nh, .s, .hd }).rename(.{ .nh = .h, .s = .q });

    const so = zml.Tensor.ScatterOpts{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override };
    var cache_k: zml.Tensor = undefined;
    var cache_v: zml.Tensor = undefined;
    if (reader) {
        if (full) {
            cache_k = cache.fl_k.choose1d(.slot, comptime geom.fullSlot(geom.full_writer));
            cache_v = cache.fl_v.choose1d(.slot, comptime geom.fullSlot(geom.full_writer));
        } else {
            cache_k = cache.sl_k.choose1d(.slot, comptime geom.slidingSlot(geom.sliding_writer));
            cache_v = cache.sl_v.choose1d(.slot, comptime geom.slidingSlot(geom.sliding_writer));
        }
    } else {
        // k_raw = k_proj brut post-reshape, capturé AVANT k_norm/rope — consommé par la branche
        // k_eq_v_full (D4) ci-dessous, comptime-morte pour E2B (même ordre d'émission qu'avant).
        const k_raw = dotPrec(prec.qkv_proj, prec.compute, h0, layer.k_proj, .d).reshape(.{ h0.dim(.b), h0.dim(.s), kvh, hd }).withTags(.{ .b, .s, .nh, .hd });
        var k = rmsScaleHdPrec(prec.norms, prec.compute, k_raw, layer.k_norm);
        k = if (full) manualRope(prec.rope, prec.compute, k, cos, sin, half) else slidingRope(prec.rope, prec.compute, k, pos_s, geom.rope_theta_sliding);
        const k_new = k.transpose(.{ .b, .nh, .s, .hd }).rename(.{ .nh = .h, .s = .k });

        // D4 (12B) : couches full sans v_proj — V part du k_proj BRUT (avant k_norm/rope), puis
        // v_norm + post_v_norm inchangés. `LayerW.v_proj` de ces couches = placeholder [1] jamais
        // consommé (toute consommation accidentelle casse au compile). Branche comptime : pour
        // E2B (k_eq_v_full=false) le chemin v_proj est émis à l'identique (gate U1).
        var v = if (comptime geom.k_eq_v_full and geom.isFull(i))
            k_raw
        else
            dotPrec(prec.qkv_proj, prec.compute, h0, layer.v_proj, .d).reshape(.{ h0.dim(.b), h0.dim(.s), kvh, hd }).withTags(.{ .b, .s, .nh, .hd });
        // v_norm n'a PAS de poids : entrée encadrée seulement, sortie re-upcastée (famille norms).
        v = zml.nn.rmsNorm(inPrec(prec.norms, v), .hd, RMS_EPS).convert(prec.compute);
        // === point d'extension post_v_norm (V post-v_norm, pré-cache) ===
        // comptime-mort pour une brique sans cette méthode (ex: struct{}) → V inchangé → bit-exact decode4.
        if (@hasDecl(@TypeOf(brick), "post_v_norm")) {
            // is_full passé en COMPTIME (la brique sélectionne une constante par shape). ctx = info runtime.
            v = brick.post_v_norm(v, comptime geom.isFull(i), LayerCtx{ .layer_idx = i });
        }
        const v_new = v.transpose(.{ .b, .nh, .s, .hd }).rename(.{ .nh = .h, .s = .k });

        // famille kv_store (G2.3, spec §4) — borne ÉCRITURE : k_new/v_new arrondis au dtype de
        // stockage AVANT scatterSlices (le cache est déclaré au dtype du header fixture, mécanisme b
        // — l'update d'un scatter doit matcher l'opérande). fam=null : inPrec = `return x`, zéro op.
        if (full) {
            const slot = zml.Tensor.scalar(@as(u32, @intCast(comptime geom.fullSlot(i))), .u32);
            cache.fl_k = cache.fl_k.scatterSlices(.{ .slot = slot, .k = pos_u }, inPrec(prec.kv_store, k_new), so);
            cache.fl_v = cache.fl_v.scatterSlices(.{ .slot = slot, .k = pos_u }, inPrec(prec.kv_store, v_new), so);
            cache_k = cache.fl_k.choose1d(.slot, comptime geom.fullSlot(i));
            cache_v = cache.fl_v.choose1d(.slot, comptime geom.fullSlot(i));
        } else {
            const slot = zml.Tensor.scalar(@as(u32, @intCast(comptime geom.slidingSlot(i))), .u32);
            // ring-buffer sliding : écriture circulaire à pos % kmax_sliding. `cfg.ring` est comptime →
            // en défaut (false) la branche `remainder` n'est PAS analysée ni émise → HLO == decode4.
            const write_k = if (cfg.ring) pos_u.remainder(zml.Tensor.scalar(@as(u32, @intCast(cfg.kmax_sliding)), .u32)) else pos_u;
            cache.sl_k = cache.sl_k.scatterSlices(.{ .slot = slot, .k = write_k }, inPrec(prec.kv_store, k_new), so);
            cache.sl_v = cache.sl_v.scatterSlices(.{ .slot = slot, .k = write_k }, inPrec(prec.kv_store, v_new), so);
            cache_k = cache.sl_k.choose1d(.slot, comptime geom.slidingSlot(i));
            cache_v = cache.sl_v.choose1d(.slot, comptime geom.slidingSlot(i));
        }
    }

    // famille kv_store — borne LECTURE : le cache lu (choose1d) est au dtype de stockage ; re-upcast
    // vers compute AVANT les dots QK/PV. Si qk_scores/pv_ctx sont AUSSI actifs, la chaîne émise reste
    // bf16→f32→bf16 telle quelle (contrat « aux bornes » : pas d'optimisation manuelle, XLA décide).
    // fam=null : branche non prise au traçage → aucune op émise, ordre d'émission INCHANGÉ.
    if (prec.kv_store != null) {
        cache_k = cache_k.convert(prec.compute);
        cache_v = cache_v.convert(prec.compute);
    }

    // Variante d'attention (cfg.attn, comptime → une seule branche émise).
    // `.manual` (défaut) : chemin historique, byte-identique (gate S1).
    // `.sdpa` : helper ZML. DEUX pièges neutralisés ici —
    //   (1) sdpa scale K par 1/√hd PAR DÉFAUT alors que Gemma 4 a scaling = 1.0 (la norme passe par
    //       q_norm) → `.scale = 1.0` OBLIGATOIRE, sinon les scores sont divisés par 16 ;
    //   (2) `q_final.dtype()` et NON `qs.dtype()` : `qs` est déclaré dans la branche `.manual`
    //       ci-dessous, donc hors scope ici (l'erreur ne sortirait qu'au premier build sdpa).
    // sdpa refait lui-même le splitAxis GQA (nn.zig) et sort `{b,q,h,hd}` après transpose+merge :
    // même forme que `ctx_attn.transpose(...)` du chemin manuel → le merge `.m` ci-dessous est commun.
    const ctx_bqhhd = if (cfg.attn == .sdpa) blk: {
        const one = zml.Tensor.scalar(1.0, q_final.dtype());
        const o = zml.nn.sdpa(q_final, cache_k, cache_v, .{ .attn_mask = mask, .scale = one });
        break :blk o.transpose(.{ .b, .q, .h, .hd });
    } else blk: {
        const qs = q_final.splitAxis(.h, .{ .h = cache_k.dim(.h), .hq = .auto });
        var scores = dotPrec(prec.qk_scores, prec.compute, qs, cache_k, .hd).merge(.{ .h = .{ .h, .hq } }).transpose(.{ .b, .h, .q, .k });
        scores = scores.add(mask.broad(scores.shape()));
        const probs = inPrec(prec.softmax, scores).softmax(.k).convert(prec.compute);

        const ps = probs.splitAxis(.h, .{ .h = cache_v.dim(.h), .hq = .auto });
        const ctx_attn = dotPrec(prec.pv_ctx, prec.compute, ps, cache_v, .k).merge(.{ .h = .{ .h, .hq } }).transpose(.{ .b, .h, .q, .hd });
        break :blk ctx_attn.transpose(.{ .b, .q, .h, .hd });
    };

    const attn_m = ctx_bqhhd.merge(.{ .m = .{ .h, .hd } });
    const attn_out = dotPrec(prec.o_proj, prec.compute, attn_m, layer.o_proj, .m).rename(.{ .q = .s });

    const h1 = hidden.add(rmsScaleDPrec(prec.norms, prec.compute, attn_out, layer.post_attention_layernorm));
    const xff = rmsScaleDPrec(prec.norms, prec.compute, h1, layer.pre_feedforward_layernorm);
    const mlp_out = dotPrec(prec.mlp, prec.compute, dotPrec(prec.mlp, prec.compute, xff, layer.gate_proj, .d).gelu().mul(dotPrec(prec.mlp, prec.compute, xff, layer.up_proj, .d)), layer.down_proj, .f);
    const h2 = h1.add(rmsScaleDPrec(prec.norms, prec.compute, mlp_out, layer.post_feedforward_layernorm));

    // Bloc PLE comptime-mort si geom.ple_dim == 0 (12B) : `ple_i` reste dans la signature,
    // inutilisé (licite en Zig). Pour E2B (ple_dim=256) : émission identique (gate U1).
    const h3 = if (comptime geom.ple_dim > 0) blk: {
        var g = dotPrec(prec.ple, prec.compute, h2, layer.per_layer_input_gate, .d).gelu();
        g = g.mul(ple_i);
        g = dotPrec(prec.ple, prec.compute, g, layer.per_layer_projection, .p);
        break :blk h2.add(rmsScaleDPrec(prec.norms, prec.compute, g, layer.post_per_layer_input_norm));
    } else h2;

    return h3.mul(cvt(layer.layer_scalar, prec.compute).asScalar());
}

/// Le socle : model decode générique paramétré comptime par une brique. `EngineModel(struct{})`
/// reproduit decode4 (gate E1) ; `EngineModel(MaBrique)` injecte une transformation au(x) point(s)
/// d'extension sans copier le moteur.
pub fn EngineModel(comptime Brick: type, comptime cfg: EngineCfg) type {
    return struct {
        embed_tokens: zml.Tensor, // {voc,d} lm_head tied
        per_layer_model_projection: zml.Tensor,
        per_layer_projection_norm: zml.Tensor,
        final_norm: zml.Tensor,
        layers: []LayerW,
        brick: Brick,
        // Précision RUNTIME (G2.3) : le runner l'écrit AVANT compile ; le traçage émet les converts
        // d'après cette valeur. Sans Tensor → strippé de Bufferized(Self) (cf zml meta.MapRestrict),
        // io.load ne le visite pas (précédent : brick struct{}).
        prec: PrecRt = .{},

        const Self = @This();

        /// Crée les poids (symboliques) depuis `base` (checkpoint) et assemble le model avec la brique
        /// fournie. Helper partagé par `init` (brique vide, E1) et `initBrick` (brique chargée, E2).
        fn initWith(allocator: std.mem.Allocator, base: zml.io.TensorStore.View, brick: Brick) !Self {
            const layers = try allocator.alloc(LayerW, cfg.geom.num_layers);
            const layers_base = base.withPrefix("layers");
            for (layers, 0..) |*layer, i| layer.* = LayerW.init(layers_base.withLayer(i));
            return .{
                .embed_tokens = base.createTensor("embed_tokens.weight", .{ .voc, .d }, null),
                .per_layer_model_projection = base.createTensor("per_layer_model_projection.weight", .{ .lf, .d }, null),
                .per_layer_projection_norm = base.createTensor("per_layer_projection_norm.weight", .{.p}, null),
                .final_norm = base.createTensor("norm.weight", .{.d}, null),
                .layers = layers,
                .brick = brick,
            };
        }

        /// E1 : poids depuis `base`, brique vide (`struct{}` → `.{}`).
        pub fn init(allocator: std.mem.Allocator, base: zml.io.TensorStore.View) !Self {
            return initWith(allocator, base, .{});
        }

        /// E2 : poids depuis `base` (store des poids), brique construite via `brick_view` (store des
        /// constantes brick, distinct). Les `Tensor` créés sont bindés à des stores différents : le LOAD
        /// se fait en deux passes côté main (poids vs brique) puis assemblage manuel du `Bufferized`.
        pub fn initBrick(allocator: std.mem.Allocator, base: zml.io.TensorStore.View, brick_view: zml.io.TensorStore.View) !Self {
            return initWith(allocator, base, Brick.init(brick_view));
        }

        pub fn load(self: *const Self, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, store: *const zml.io.TensorStore, shardings: []const zml.sharding.Sharding) !zml.Bufferized(Self) {
            return zml.io.load(Self, self, allocator, io, platform, store, .{ .shardings = shardings, .parallelism = 1, .dma_chunks = 1, .dma_chunk_size = 16 * 1024 * 1024 });
        }

        fn perLayerInputs(self: Self, embptl_slice: zml.Tensor, embeds: zml.Tensor) zml.Tensor {
            const prec = self.prec;
            const token_identity = embptl_slice
                .scale(SQRT_PLE).convert(.f32)
                .reshape(.{ embptl_slice.dim(.b), embptl_slice.dim(.s), cfg.geom.num_layers, cfg.geom.ple_dim }).withTags(.{ .b, .s, .layer, .p });
            const context = dotPrec(prec.ple, prec.compute, embeds, self.per_layer_model_projection, .d)
                .scale(comptime cfg.geom.invSqrtHid())
                .reshape(.{ embeds.dim(.b), embeds.dim(.s), cfg.geom.num_layers, cfg.geom.ple_dim }).withTags(.{ .b, .s, .layer, .p });
            const context_norm = rmsScaleP(context, cvt(self.per_layer_projection_norm, prec.compute));
            return context_norm.add(token_identity).scale(INV_SQRT_2);
        }

        /// Un pas de génération : sélectionne le step, embed+PLE, 35 couches (cache threadé) -> logits +
        /// cache grandi. Retour : {logits {b,s,voc}, sl_k, sl_v, fl_k, fl_v}.
        pub fn forward(self: Self, p: Packed(cfg.maskMode()), cache_in: Cache, ctrl: Ctrl) struct { zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor } {
            const prec = self.prec;
            const step = ctrl.step;
            const embed_slice = pickStep(p.embeds, step);
            const embptl_slice = pickStep(p.embptls, step);
            const cos = pickStep(p.cos_full, step);
            const sin = pickStep(p.sin_full, step);
            // Masque(s) extrait(s)/généré(s) UNE fois (hors boucle). Branches comptime par MaskMode :
            // la branche inactive n'est pas analysée (champs absents tolérés). En .ingraph, les lignes
            // sont générées depuis positions[step] + window (spec 2026-07-26 §4.3).
            const mask_single = if (cfg.two_masks) {} else pickStep(p.masks, step);
            const masks_gen = if (cfg.ingraph_masks) ingraphMaskLines(cfg.kmax_sliding, cfg.kmax_full, p.positions, p.window, step) else {};
            const mask_sliding = if (cfg.ingraph_masks) masks_gen[0] else if (cfg.two_masks) pickStep(p.masks_sliding, step) else {};
            const mask_full = if (cfg.ingraph_masks) masks_gen[1] else if (cfg.two_masks) pickStep(p.masks_full, step) else {};
            const pos_i = pickStep(p.positions, step); // {} i32
            const pos_s = pos_i.reshape(.{1}).withTags(.{.s});
            const pos_u = pos_i.convert(.u32);

            const embeds = embed_slice.convert(.f32).scale(comptime cfg.geom.embedScale());
            // PLE comptime-mort si geom.ple_dim == 0 (12B) : ple/ple_i valent embeds, jamais consommés.
            const ple = if (comptime cfg.geom.ple_dim > 0) self.perLayerInputs(embptl_slice, embeds) else embeds;
            var hidden = embeds;
            var cache = cache_in;
            inline for (0..cfg.geom.num_layers) |i| {
                const ple_i = if (comptime cfg.geom.ple_dim > 0) ple.choose1d(.layer, @as(i64, @intCast(i))) else ple;
                // sélection du masque par type de couche (comptime). En défaut : mask_single (== decode4).
                const mask = if (cfg.two_masks)
                    (if (comptime cfg.geom.isFull(i)) mask_full else mask_sliding)
                else
                    mask_single;
                hidden = runLayerGen(self.layers[i], i, cfg, prec, hidden, ple_i, cos, sin, mask, pos_s, pos_u, &cache, self.brick);
            }
            const last_hidden = rmsScaleDPrec(prec.norms, prec.compute, hidden, self.final_norm);
            const raw = dotPrec(prec.head, prec.compute, last_hidden, self.embed_tokens, .d);
            const logits = softcapPrec(prec.softcap, prec.compute, raw, cfg.geom.softcap);
            return .{ logits, cache.sl_k, cache.sl_v, cache.fl_k, cache.fl_v };
        }

        /// Variante CHUNKÉE du forward (perf) : exécute les couches [start,end) d'UN step, cache threadé.
        /// Découpe le graphe 35-couches en stages compilés séparément (borne le pic mémoire : moins de
        /// poids f32 coexistant). `first` → hidden = embeds (hidden_in ignoré) ; sinon hidden = hidden_in
        /// (sortie du stage précédent, threadée device→device). `last` → final norm + lm_head + softcap.
        /// Le PLE est recalculé ici (pur fonction de embeds, bit-exact). Type de retour UNIFORME (5 Tensors)
        /// : 1er = hidden_out (non-last) OU logits (last) ; + cache (sl_k,sl_v,fl_k,fl_v). Le calcul est
        /// identique à `forward` op-pour-op (runLayerGen partagé) → mêmes tokens, autre exécution.
        pub fn forwardStageGen(self: Self, comptime start: usize, comptime end: usize, comptime first: bool, comptime last: bool, p: Packed(cfg.maskMode()), cache_in: Cache, hidden_in: zml.Tensor, ctrl: Ctrl) struct { zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor } {
            const prec = self.prec;
            const step = ctrl.step;
            const embptl_slice = pickStep(p.embptls, step);
            const cos = pickStep(p.cos_full, step);
            const sin = pickStep(p.sin_full, step);
            const mask_single = if (cfg.two_masks) {} else pickStep(p.masks, step);
            const masks_gen = if (cfg.ingraph_masks) ingraphMaskLines(cfg.kmax_sliding, cfg.kmax_full, p.positions, p.window, step) else {};
            const mask_sliding = if (cfg.ingraph_masks) masks_gen[0] else if (cfg.two_masks) pickStep(p.masks_sliding, step) else {};
            const mask_full = if (cfg.ingraph_masks) masks_gen[1] else if (cfg.two_masks) pickStep(p.masks_full, step) else {};
            const pos_i = pickStep(p.positions, step);
            const pos_s = pos_i.reshape(.{1}).withTags(.{.s});
            const pos_u = pos_i.convert(.u32);

            const embeds = pickStep(p.embeds, step).convert(.f32).scale(comptime cfg.geom.embedScale());
            const ple = if (comptime cfg.geom.ple_dim > 0) self.perLayerInputs(embptl_slice, embeds) else embeds;
            var hidden = if (first) embeds else hidden_in;
            var cache = cache_in;
            inline for (start..end) |i| {
                const ple_i = if (comptime cfg.geom.ple_dim > 0) ple.choose1d(.layer, @as(i64, @intCast(i))) else ple;
                const mask = if (cfg.two_masks)
                    (if (comptime cfg.geom.isFull(i)) mask_full else mask_sliding)
                else
                    mask_single;
                hidden = runLayerGen(self.layers[i], i, cfg, prec, hidden, ple_i, cos, sin, mask, pos_s, pos_u, &cache, self.brick);
            }

            const out_first = if (last) blk: {
                const last_hidden = rmsScaleDPrec(prec.norms, prec.compute, hidden, self.final_norm);
                const raw = dotPrec(prec.head, prec.compute, last_hidden, self.embed_tokens, .d);
                break :blk softcapPrec(prec.softcap, prec.compute, raw, cfg.geom.softcap);
            } else hidden;
            return .{ out_first, cache.sl_k, cache.sl_v, cache.fl_k, cache.fl_v };
        }

        /// L2 — forward 1-step AUTONOME (host-orchestré) : les embeds/embptls viennent d'un gather HOST
        /// du token produit (token-dépendant), tandis que cos/sin/masques/positions viennent de `p` (la
        /// fixture L1a : position-only, INDÉPENDANTS du token → valides pour la génération autonome tant
        /// que les positions coïncident, i.e. même prompt+compte). `forward` mono (E1/E2) et `forwardStageGen`
        /// (chunké) sont INTACTS → preuve HLO et L1a inchangés ; cette méthode est une NOUVELLE entrée.
        ///
        /// `embeds_step` : {b,s,d} bf16 — embed_tokens[fed_tok] host-gathered (AVANT scale √1536, brut).
        /// `embptls_step` : {b,s,lf} bf16 — embed_tokens_per_layer[fed_tok] host-gathered.
        /// Retourne {logits, sl_k, sl_v, fl_k, fl_v} (== `forward` mono, op-pour-op identique hormis la
        /// source des embeds/embptls). Permet la boucle autonome : argmax → gather host → reinject.
        pub fn forwardStep(self: Self, embeds_step: zml.Tensor, embptls_step: zml.Tensor, p: Packed(cfg.maskMode()), cache_in: Cache, ctrl: Ctrl) struct { zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor } {
            if (cfg.ingraph_masks) @compileError("ingraph_masks non câblé sur forwardStep — runners 12B = forwardStageGen seul (spec 2026-07-26 §4.1)");
            const prec = self.prec;
            const step = ctrl.step;
            const cos = pickStep(p.cos_full, step);
            const sin = pickStep(p.sin_full, step);
            const mask_single = if (cfg.two_masks) {} else pickStep(p.masks, step);
            const mask_sliding = if (cfg.two_masks) pickStep(p.masks_sliding, step) else {};
            const mask_full = if (cfg.two_masks) pickStep(p.masks_full, step) else {};
            const pos_i = pickStep(p.positions, step);
            const pos_s = pos_i.reshape(.{1}).withTags(.{.s});
            const pos_u = pos_i.convert(.u32);

            const embeds = embeds_step.convert(.f32).scale(comptime cfg.geom.embedScale());
            const ple = if (comptime cfg.geom.ple_dim > 0) self.perLayerInputs(embptls_step, embeds) else embeds;
            var hidden = embeds;
            var cache = cache_in;
            inline for (0..cfg.geom.num_layers) |i| {
                const ple_i = if (comptime cfg.geom.ple_dim > 0) ple.choose1d(.layer, @as(i64, @intCast(i))) else ple;
                const mask = if (cfg.two_masks)
                    (if (comptime cfg.geom.isFull(i)) mask_full else mask_sliding)
                else
                    mask_single;
                hidden = runLayerGen(self.layers[i], i, cfg, prec, hidden, ple_i, cos, sin, mask, pos_s, pos_u, &cache, self.brick);
            }

            const last_hidden = rmsScaleDPrec(prec.norms, prec.compute, hidden, self.final_norm);
            const raw = dotPrec(prec.head, prec.compute, last_hidden, self.embed_tokens, .d);
            const logits = softcapPrec(prec.softcap, prec.compute, raw, cfg.geom.softcap);
            return .{ logits, cache.sl_k, cache.sl_v, cache.fl_k, cache.fl_v };
        }

        /// L2 CHUNKÉ — variante stage de `forwardStep` (autonome host-orchestré, chunké pour la mémoire).
        /// Comme `forwardStageGen` MAIS embeds/embptls viennent d'un gather HOST per-step (token-dépendant)
        /// au lieu de `pickStep(p.embeds/embptls)`. cos/sin/masques/positions restent de `p` (position-only).
        /// `first` → hidden = embeds_step ; `last` → final norm + lm_head + softcap. Même `runLayerGen`.
        /// Nécessaire car le mono `forwardStep` compile le graphe 35-couches (~33 Go, thrash) : le chunké
        /// borne le pic (cf GENERATION_LONGUE_CHUNKING_DESIGN). `forward`/`forwardStageGen`/`forwardStep`
        /// (E1/E2/L1a) sont INTACTS.
        pub fn forwardStageStep(self: Self, comptime start: usize, comptime end: usize, comptime first: bool, comptime last: bool, embeds_step: zml.Tensor, embptls_step: zml.Tensor, p: Packed(cfg.maskMode()), cache_in: Cache, hidden_in: zml.Tensor, ctrl: Ctrl) struct { zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor } {
            if (cfg.ingraph_masks) @compileError("ingraph_masks non câblé sur forwardStageStep — runners 12B = forwardStageGen seul (spec 2026-07-26 §4.1)");
            const prec = self.prec;
            const step = ctrl.step;
            const cos = pickStep(p.cos_full, step);
            const sin = pickStep(p.sin_full, step);
            const mask_single = if (cfg.two_masks) {} else pickStep(p.masks, step);
            const mask_sliding = if (cfg.two_masks) pickStep(p.masks_sliding, step) else {};
            const mask_full = if (cfg.two_masks) pickStep(p.masks_full, step) else {};
            const pos_i = pickStep(p.positions, step);
            const pos_s = pos_i.reshape(.{1}).withTags(.{.s});
            const pos_u = pos_i.convert(.u32);

            const embeds = embeds_step.convert(.f32).scale(comptime cfg.geom.embedScale());
            const ple = if (comptime cfg.geom.ple_dim > 0) self.perLayerInputs(embptls_step, embeds) else embeds;
            var hidden = if (first) embeds else hidden_in;
            var cache = cache_in;
            inline for (start..end) |i| {
                const ple_i = if (comptime cfg.geom.ple_dim > 0) ple.choose1d(.layer, @as(i64, @intCast(i))) else ple;
                const mask = if (cfg.two_masks)
                    (if (comptime cfg.geom.isFull(i)) mask_full else mask_sliding)
                else
                    mask_single;
                hidden = runLayerGen(self.layers[i], i, cfg, prec, hidden, ple_i, cos, sin, mask, pos_s, pos_u, &cache, self.brick);
            }
            const out_first = if (last) blk: {
                const last_hidden = rmsScaleDPrec(prec.norms, prec.compute, hidden, self.final_norm);
                const raw = dotPrec(prec.head, prec.compute, last_hidden, self.embed_tokens, .d);
                break :blk softcapPrec(prec.softcap, prec.compute, raw, cfg.geom.softcap);
            } else hidden;
            return .{ out_first, cache.sl_k, cache.sl_v, cache.fl_k, cache.fl_v };
        }
    };
}
