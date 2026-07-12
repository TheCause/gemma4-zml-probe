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

pub const NUM_LAYERS: usize = 35;
pub const FIRST_KV_SHARED: usize = 15;
pub const SLIDING_WRITER: usize = 13;
pub const FULL_WRITER: usize = 14;
pub const SLIDING_WRITER_SLOT: i64 = 11; // slidingSlot(13)
pub const FULL_WRITER_SLOT: i64 = 2; // fullSlot(14)

// B/S ne sont PLUS consommés par le moteur : depuis le gate batch T0, les 5 sites de reshape
// (q/k/v + les 2 du PLE) dérivent leurs dims batch/seq des SHAPES D'ENTRÉE (`h0.dim(.b)`,
// `embeds.dim(.s)`…) → le graphe est shape-polymorphe, un binaire unique sert tous les B.
// Déclarations gardées pour les runners existants (qui construisent encore des shapes à 1).
pub const B: i64 = 1;
pub const S: i64 = 1;
pub const D: i64 = 1536;
pub const NH: i64 = 8;
pub const KVH: i64 = 1;
pub const HD_SLIDING: i64 = 256;
pub const HD_FULL: i64 = 512;
pub const PLE_DIM: i64 = 256;

const RMS_EPS: f32 = 1.0e-6;
const ROPE_THETA_SLIDING: f32 = 1.0e4;
const EMBED_SCALE: f64 = @sqrt(1536.0);
const INV_SQRT_HID: f64 = 1.0 / @sqrt(1536.0);
const SQRT_PLE: f64 = 16.0;
const INV_SQRT_2: f64 = 0.7071067811865476;
const SOFTCAP: f64 = 30.0;
const INV_SOFTCAP: f64 = 1.0 / 30.0;

pub fn isFull(i: usize) bool {
    return (i + 1) % 5 == 0;
}
pub fn isReader(i: usize) bool {
    return i >= FIRST_KV_SHARED;
}
fn slidingSlot(i: usize) i64 {
    var slot: i64 = 0;
    var j: usize = 0;
    while (j < i) : (j += 1) {
        if (!isFull(j)) slot += 1;
    }
    return slot;
}
fn fullSlot(i: usize) i64 {
    var slot: i64 = 0;
    var j: usize = 0;
    while (j < i) : (j += 1) {
        if (isFull(j)) slot += 1;
    }
    return slot;
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
fn slidingRope(fam: ?zml.DataType, compute: zml.DataType, x: zml.Tensor, pos: zml.Tensor) zml.Tensor {
    const xi = inPrec(fam, x);
    return zml.nn.rope(xi, pos, .{ .layout = .sequential, .scaling = .{ .default = .{ .rope_theta = ROPE_THETA_SLIDING } } }).convert(compute);
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
/// Softcap 30·tanh(x/30) prec-aware (famille softcap) : entrée arrondie, sortie re-upcastée.
/// `scale` émet sa constante au dtype du tensor (cf tensor.zig) → le chemin bf16 reste homogène.
/// fam=null : émission identique à `raw.scale(INV_SOFTCAP).tanh().scale(SOFTCAP)`.
fn softcapPrec(fam: ?zml.DataType, compute: zml.DataType, raw: zml.Tensor) zml.Tensor {
    return inPrec(fam, raw).scale(INV_SOFTCAP).tanh().scale(SOFTCAP).convert(compute);
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
// Paramétré comptime par `two_masks` (cf engine DESIGN §5.3) :
//   - false (défaut) : un seul masque `masks` — strictement identique à l'ancien `Packed`, donc la
//     fixture E1/E2 (KMAX=8) charge inchangée et le graphe HLO est préservé.
//   - true (génération longue) : deux masques `masks_sliding`/`masks_full` de tailles `.k` distinctes.
// `zml.io.load` réfléchit RÉCURSIVEMENT sur les champs (chacun doit être un Tensor) → on retourne DEUX
// structs distincts (pas de champ `void` conditionnel, que load ne saurait pas traiter).
pub fn Packed(comptime two_masks: bool) type {
    if (two_masks) return struct {
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

/// Config comptime du socle (cf engine DESIGN §3, §5). TOUS les champs ont une valeur par défaut qui
/// reproduit le comportement decode4/E1 : `EngineModel(Brick, .{})` est strictement neutre (aucune op
/// nouvelle émise) → graphe HLO byte-identique. La génération longue active `ring`/`two_masks` et fixe
/// les tailles de fenêtre. `kmax_sliding` n'est utilisé que comme scalaire du modulo ring (la dim `.k`
/// du cache, elle, est inférée de la fixture).
pub const EngineCfg = struct {
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
    const full = isFull(i);
    const reader = isReader(i);
    const hd: i64 = if (full) HD_FULL else HD_SLIDING;
    const half: i64 = @divExact(hd, 2);

    const h0 = rmsScaleDPrec(prec.norms, prec.compute, hidden, layer.input_layernorm);

    var q = dotPrec(prec.qkv_proj, prec.compute, h0, layer.q_proj, .d).reshape(.{ h0.dim(.b), h0.dim(.s), NH, hd }).withTags(.{ .b, .s, .nh, .hd });
    q = rmsScaleHdPrec(prec.norms, prec.compute, q, layer.q_norm);
    q = if (full) manualRope(prec.rope, prec.compute, q, cos, sin, half) else slidingRope(prec.rope, prec.compute, q, pos_s);
    const q_final = q.transpose(.{ .b, .nh, .s, .hd }).rename(.{ .nh = .h, .s = .q });

    const so = zml.Tensor.ScatterOpts{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override };
    var cache_k: zml.Tensor = undefined;
    var cache_v: zml.Tensor = undefined;
    if (reader) {
        if (full) {
            cache_k = cache.fl_k.choose1d(.slot, FULL_WRITER_SLOT);
            cache_v = cache.fl_v.choose1d(.slot, FULL_WRITER_SLOT);
        } else {
            cache_k = cache.sl_k.choose1d(.slot, SLIDING_WRITER_SLOT);
            cache_v = cache.sl_v.choose1d(.slot, SLIDING_WRITER_SLOT);
        }
    } else {
        var k = dotPrec(prec.qkv_proj, prec.compute, h0, layer.k_proj, .d).reshape(.{ h0.dim(.b), h0.dim(.s), KVH, hd }).withTags(.{ .b, .s, .nh, .hd });
        k = rmsScaleHdPrec(prec.norms, prec.compute, k, layer.k_norm);
        k = if (full) manualRope(prec.rope, prec.compute, k, cos, sin, half) else slidingRope(prec.rope, prec.compute, k, pos_s);
        const k_new = k.transpose(.{ .b, .nh, .s, .hd }).rename(.{ .nh = .h, .s = .k });

        var v = dotPrec(prec.qkv_proj, prec.compute, h0, layer.v_proj, .d).reshape(.{ h0.dim(.b), h0.dim(.s), KVH, hd }).withTags(.{ .b, .s, .nh, .hd });
        // v_norm n'a PAS de poids : entrée encadrée seulement, sortie re-upcastée (famille norms).
        v = zml.nn.rmsNorm(inPrec(prec.norms, v), .hd, RMS_EPS).convert(prec.compute);
        // === point d'extension post_v_norm (V post-v_norm, pré-cache) ===
        // comptime-mort pour une brique sans cette méthode (ex: struct{}) → V inchangé → bit-exact decode4.
        if (@hasDecl(@TypeOf(brick), "post_v_norm")) {
            // is_full passé en COMPTIME (la brique sélectionne une constante par shape) ; `comptime isFull(i)`
            // car `i` est comptime mais `full` est un const runtime. ctx = info runtime.
            v = brick.post_v_norm(v, comptime isFull(i), LayerCtx{ .layer_idx = i });
        }
        const v_new = v.transpose(.{ .b, .nh, .s, .hd }).rename(.{ .nh = .h, .s = .k });

        // famille kv_store (G2.3, spec §4) — borne ÉCRITURE : k_new/v_new arrondis au dtype de
        // stockage AVANT scatterSlices (le cache est déclaré au dtype du header fixture, mécanisme b
        // — l'update d'un scatter doit matcher l'opérande). fam=null : inPrec = `return x`, zéro op.
        if (full) {
            const slot = zml.Tensor.scalar(@as(u32, @intCast(fullSlot(i))), .u32);
            cache.fl_k = cache.fl_k.scatterSlices(.{ .slot = slot, .k = pos_u }, inPrec(prec.kv_store, k_new), so);
            cache.fl_v = cache.fl_v.scatterSlices(.{ .slot = slot, .k = pos_u }, inPrec(prec.kv_store, v_new), so);
            cache_k = cache.fl_k.choose1d(.slot, fullSlot(i));
            cache_v = cache.fl_v.choose1d(.slot, fullSlot(i));
        } else {
            const slot = zml.Tensor.scalar(@as(u32, @intCast(slidingSlot(i))), .u32);
            // ring-buffer sliding : écriture circulaire à pos % kmax_sliding. `cfg.ring` est comptime →
            // en défaut (false) la branche `remainder` n'est PAS analysée ni émise → HLO == decode4.
            const write_k = if (cfg.ring) pos_u.remainder(zml.Tensor.scalar(@as(u32, @intCast(cfg.kmax_sliding)), .u32)) else pos_u;
            cache.sl_k = cache.sl_k.scatterSlices(.{ .slot = slot, .k = write_k }, inPrec(prec.kv_store, k_new), so);
            cache.sl_v = cache.sl_v.scatterSlices(.{ .slot = slot, .k = write_k }, inPrec(prec.kv_store, v_new), so);
            cache_k = cache.sl_k.choose1d(.slot, slidingSlot(i));
            cache_v = cache.sl_v.choose1d(.slot, slidingSlot(i));
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

    var g = dotPrec(prec.ple, prec.compute, h2, layer.per_layer_input_gate, .d).gelu();
    g = g.mul(ple_i);
    g = dotPrec(prec.ple, prec.compute, g, layer.per_layer_projection, .p);
    const h3 = h2.add(rmsScaleDPrec(prec.norms, prec.compute, g, layer.post_per_layer_input_norm));

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
            const layers = try allocator.alloc(LayerW, NUM_LAYERS);
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
                .reshape(.{ embptl_slice.dim(.b), embptl_slice.dim(.s), NUM_LAYERS, PLE_DIM }).withTags(.{ .b, .s, .layer, .p });
            const context = dotPrec(prec.ple, prec.compute, embeds, self.per_layer_model_projection, .d)
                .scale(INV_SQRT_HID)
                .reshape(.{ embeds.dim(.b), embeds.dim(.s), NUM_LAYERS, PLE_DIM }).withTags(.{ .b, .s, .layer, .p });
            const context_norm = rmsScaleP(context, cvt(self.per_layer_projection_norm, prec.compute));
            return context_norm.add(token_identity).scale(INV_SQRT_2);
        }

        /// Un pas de génération : sélectionne le step, embed+PLE, 35 couches (cache threadé) -> logits +
        /// cache grandi. Retour : {logits {b,s,voc}, sl_k, sl_v, fl_k, fl_v}.
        pub fn forward(self: Self, p: Packed(cfg.two_masks), cache_in: Cache, ctrl: Ctrl) struct { zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor } {
            const prec = self.prec;
            const step = ctrl.step;
            const embed_slice = pickStep(p.embeds, step);
            const embptl_slice = pickStep(p.embptls, step);
            const cos = pickStep(p.cos_full, step);
            const sin = pickStep(p.sin_full, step);
            // Masque(s) extrait(s) UNE fois (hors boucle) → 1 dynamicSlice en défaut, comme decode4.
            // `cfg.two_masks` comptime : la branche inactive n'est pas analysée (champs absents tolérés).
            const mask_single = if (cfg.two_masks) {} else pickStep(p.masks, step);
            const mask_sliding = if (cfg.two_masks) pickStep(p.masks_sliding, step) else {};
            const mask_full = if (cfg.two_masks) pickStep(p.masks_full, step) else {};
            const pos_i = pickStep(p.positions, step); // {} i32
            const pos_s = pos_i.reshape(.{1}).withTags(.{.s});
            const pos_u = pos_i.convert(.u32);

            const embeds = embed_slice.convert(.f32).scale(EMBED_SCALE);
            const ple = self.perLayerInputs(embptl_slice, embeds);
            var hidden = embeds;
            var cache = cache_in;
            inline for (0..NUM_LAYERS) |i| {
                const ple_i = ple.choose1d(.layer, @as(i64, @intCast(i)));
                // sélection du masque par type de couche (comptime). En défaut : mask_single (== decode4).
                const mask = if (cfg.two_masks)
                    (if (comptime isFull(i)) mask_full else mask_sliding)
                else
                    mask_single;
                hidden = runLayerGen(self.layers[i], i, cfg, prec, hidden, ple_i, cos, sin, mask, pos_s, pos_u, &cache, self.brick);
            }
            const last_hidden = rmsScaleDPrec(prec.norms, prec.compute, hidden, self.final_norm);
            const raw = dotPrec(prec.head, prec.compute, last_hidden, self.embed_tokens, .d);
            const logits = softcapPrec(prec.softcap, prec.compute, raw);
            return .{ logits, cache.sl_k, cache.sl_v, cache.fl_k, cache.fl_v };
        }

        /// Variante CHUNKÉE du forward (perf) : exécute les couches [start,end) d'UN step, cache threadé.
        /// Découpe le graphe 35-couches en stages compilés séparément (borne le pic mémoire : moins de
        /// poids f32 coexistant). `first` → hidden = embeds (hidden_in ignoré) ; sinon hidden = hidden_in
        /// (sortie du stage précédent, threadée device→device). `last` → final norm + lm_head + softcap.
        /// Le PLE est recalculé ici (pur fonction de embeds, bit-exact). Type de retour UNIFORME (5 Tensors)
        /// : 1er = hidden_out (non-last) OU logits (last) ; + cache (sl_k,sl_v,fl_k,fl_v). Le calcul est
        /// identique à `forward` op-pour-op (runLayerGen partagé) → mêmes tokens, autre exécution.
        pub fn forwardStageGen(self: Self, comptime start: usize, comptime end: usize, comptime first: bool, comptime last: bool, p: Packed(cfg.two_masks), cache_in: Cache, hidden_in: zml.Tensor, ctrl: Ctrl) struct { zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor } {
            const prec = self.prec;
            const step = ctrl.step;
            const embptl_slice = pickStep(p.embptls, step);
            const cos = pickStep(p.cos_full, step);
            const sin = pickStep(p.sin_full, step);
            const mask_single = if (cfg.two_masks) {} else pickStep(p.masks, step);
            const mask_sliding = if (cfg.two_masks) pickStep(p.masks_sliding, step) else {};
            const mask_full = if (cfg.two_masks) pickStep(p.masks_full, step) else {};
            const pos_i = pickStep(p.positions, step);
            const pos_s = pos_i.reshape(.{1}).withTags(.{.s});
            const pos_u = pos_i.convert(.u32);

            const embeds = pickStep(p.embeds, step).convert(.f32).scale(EMBED_SCALE);
            const ple = self.perLayerInputs(embptl_slice, embeds);
            var hidden = if (first) embeds else hidden_in;
            var cache = cache_in;
            inline for (start..end) |i| {
                const ple_i = ple.choose1d(.layer, @as(i64, @intCast(i)));
                const mask = if (cfg.two_masks)
                    (if (comptime isFull(i)) mask_full else mask_sliding)
                else
                    mask_single;
                hidden = runLayerGen(self.layers[i], i, cfg, prec, hidden, ple_i, cos, sin, mask, pos_s, pos_u, &cache, self.brick);
            }

            const out_first = if (last) blk: {
                const last_hidden = rmsScaleDPrec(prec.norms, prec.compute, hidden, self.final_norm);
                const raw = dotPrec(prec.head, prec.compute, last_hidden, self.embed_tokens, .d);
                break :blk softcapPrec(prec.softcap, prec.compute, raw);
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
        pub fn forwardStep(self: Self, embeds_step: zml.Tensor, embptls_step: zml.Tensor, p: Packed(cfg.two_masks), cache_in: Cache, ctrl: Ctrl) struct { zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor } {
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

            const embeds = embeds_step.convert(.f32).scale(EMBED_SCALE);
            const ple = self.perLayerInputs(embptls_step, embeds);
            var hidden = embeds;
            var cache = cache_in;
            inline for (0..NUM_LAYERS) |i| {
                const ple_i = ple.choose1d(.layer, @as(i64, @intCast(i)));
                const mask = if (cfg.two_masks)
                    (if (comptime isFull(i)) mask_full else mask_sliding)
                else
                    mask_single;
                hidden = runLayerGen(self.layers[i], i, cfg, prec, hidden, ple_i, cos, sin, mask, pos_s, pos_u, &cache, self.brick);
            }

            const last_hidden = rmsScaleDPrec(prec.norms, prec.compute, hidden, self.final_norm);
            const raw = dotPrec(prec.head, prec.compute, last_hidden, self.embed_tokens, .d);
            const logits = softcapPrec(prec.softcap, prec.compute, raw);
            return .{ logits, cache.sl_k, cache.sl_v, cache.fl_k, cache.fl_v };
        }

        /// L2 CHUNKÉ — variante stage de `forwardStep` (autonome host-orchestré, chunké pour la mémoire).
        /// Comme `forwardStageGen` MAIS embeds/embptls viennent d'un gather HOST per-step (token-dépendant)
        /// au lieu de `pickStep(p.embeds/embptls)`. cos/sin/masques/positions restent de `p` (position-only).
        /// `first` → hidden = embeds_step ; `last` → final norm + lm_head + softcap. Même `runLayerGen`.
        /// Nécessaire car le mono `forwardStep` compile le graphe 35-couches (~33 Go, thrash) : le chunké
        /// borne le pic (cf GENERATION_LONGUE_CHUNKING_DESIGN). `forward`/`forwardStageGen`/`forwardStep`
        /// (E1/E2/L1a) sont INTACTS.
        pub fn forwardStageStep(self: Self, comptime start: usize, comptime end: usize, comptime first: bool, comptime last: bool, embeds_step: zml.Tensor, embptls_step: zml.Tensor, p: Packed(cfg.two_masks), cache_in: Cache, hidden_in: zml.Tensor, ctrl: Ctrl) struct { zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor } {
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

            const embeds = embeds_step.convert(.f32).scale(EMBED_SCALE);
            const ple = self.perLayerInputs(embptls_step, embeds);
            var hidden = if (first) embeds else hidden_in;
            var cache = cache_in;
            inline for (start..end) |i| {
                const ple_i = ple.choose1d(.layer, @as(i64, @intCast(i)));
                const mask = if (cfg.two_masks)
                    (if (comptime isFull(i)) mask_full else mask_sliding)
                else
                    mask_single;
                hidden = runLayerGen(self.layers[i], i, cfg, prec, hidden, ple_i, cos, sin, mask, pos_s, pos_u, &cache, self.brick);
            }
            const out_first = if (last) blk: {
                const last_hidden = rmsScaleDPrec(prec.norms, prec.compute, hidden, self.final_norm);
                const raw = dotPrec(prec.head, prec.compute, last_hidden, self.embed_tokens, .d);
                break :blk softcapPrec(prec.softcap, prec.compute, raw);
            } else hidden;
            return .{ out_first, cache.sl_k, cache.sl_v, cache.fl_k, cache.fl_v };
        }
    };
}
