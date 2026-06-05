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

pub const NUM_LAYERS: usize = 35;
pub const FIRST_KV_SHARED: usize = 15;
pub const SLIDING_WRITER: usize = 13;
pub const FULL_WRITER: usize = 14;
pub const SLIDING_WRITER_SLOT: i64 = 11; // slidingSlot(13)
pub const FULL_WRITER_SLOT: i64 = 2; // fullSlot(14)

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

inline fn c(t: zml.Tensor) zml.Tensor {
    return t.convert(.f32);
}
fn rmsScaleD(x: zml.Tensor, w: zml.Tensor) zml.Tensor {
    return zml.nn.rmsNorm(x, .d, RMS_EPS).mul(w.broad(x.shape()));
}
fn rmsScaleP(x: zml.Tensor, w: zml.Tensor) zml.Tensor {
    const n = zml.nn.rmsNorm(x, .p, RMS_EPS);
    return n.mul(w.broad(n.shape()));
}
fn slidingRope(x: zml.Tensor, pos: zml.Tensor) zml.Tensor {
    return zml.nn.rope(x, pos, .{ .layout = .sequential, .scaling = .{ .default = .{ .rope_theta = ROPE_THETA_SLIDING } } });
}
fn manualRope(x: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor, half: i64) zml.Tensor {
    const halves = x.split(.hd, &.{ half, half });
    const rh = zml.Tensor.concatenate(&.{ halves[1].negate(), halves[0] }, .hd);
    return x.mul(cos.broad(x.shape())).add(rh.mul(sin.broad(x.shape())));
}

/// Contexte passé à un point d'extension. `layer_idx`/`is_full` sont comptime (issus de l'inline for).
/// `pos` (Tensor runtime) n'est PAS inclus : YAGNI — aucune brique actuelle ne le consomme (TurboQuant
/// route sur `is_full`). S'ajoute trivialement quand une brique le demande.
pub const LayerCtx = struct { layer_idx: usize, is_full: bool };

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
};

// Entrées par-step empaquetées (constantes sur la boucle ; sélectionnées par dynamicSlice(.step)).
pub const Packed = struct {
    embeds: zml.Tensor, // {step,b,s,d} bf16
    embptls: zml.Tensor, // {step,b,s,lf} bf16
    cos_full: zml.Tensor, // {step,b,s,hd=512}
    sin_full: zml.Tensor,
    masks: zml.Tensor, // {step,b,h,q,k}
    positions: zml.Tensor, // {step} i32

    pub fn init(v: zml.io.TensorStore.View) Packed {
        return .{
            .embeds = v.createTensor("embeds", .{ .step, .b, .s, .d }, null),
            .embptls = v.createTensor("embptls", .{ .step, .b, .s, .lf }, null),
            .cos_full = v.createTensor("cos_full", .{ .step, .b, .s, .hd }, null),
            .sin_full = v.createTensor("sin_full", .{ .step, .b, .s, .hd }, null),
            .masks = v.createTensor("masks", .{ .step, .b, .h, .q, .k }, null),
            .positions = v.createTensor("positions", .{.step}, null),
        };
    }

    pub fn load(self: *const Packed, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, store: *const zml.io.TensorStore, shardings: []const zml.sharding.Sharding) !zml.Bufferized(Packed) {
        return zml.io.load(Packed, self, allocator, io, platform, store, .{ .shardings = shardings, .parallelism = 1, .dma_chunks = 1, .dma_chunk_size = 16 * 1024 * 1024 });
    }
};

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

/// Forward decode d'UNE couche i en mode génération (cache threadé). Producer scatter K/V du token à
/// (.slot, .k=pos) dans le cache empaqueté ; reader lit le slot du writer 13/14. `brick` est threadé
/// pour le point d'extension post_v_norm (comptime-mort si la brique ne l'implémente pas).
fn runLayerGen(layer: LayerW, comptime i: usize, hidden: zml.Tensor, ple_i: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor, mask: zml.Tensor, pos_s: zml.Tensor, pos_u: zml.Tensor, cache: *Cache, brick: anytype) zml.Tensor {
    const full = isFull(i);
    const reader = isReader(i);
    const hd: i64 = if (full) HD_FULL else HD_SLIDING;
    const half: i64 = @divExact(hd, 2);

    const h0 = rmsScaleD(hidden, c(layer.input_layernorm));

    var q = h0.dot(c(layer.q_proj), .d).reshape(.{ B, S, NH, hd }).withTags(.{ .b, .s, .nh, .hd });
    q = zml.nn.rmsNorm(q, .hd, RMS_EPS).mul(c(layer.q_norm).broad(q.shape()));
    q = if (full) manualRope(q, cos, sin, half) else slidingRope(q, pos_s);
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
        var k = h0.dot(c(layer.k_proj), .d).reshape(.{ B, S, KVH, hd }).withTags(.{ .b, .s, .nh, .hd });
        k = zml.nn.rmsNorm(k, .hd, RMS_EPS).mul(c(layer.k_norm).broad(k.shape()));
        k = if (full) manualRope(k, cos, sin, half) else slidingRope(k, pos_s);
        const k_new = k.transpose(.{ .b, .nh, .s, .hd }).rename(.{ .nh = .h, .s = .k });

        var v = h0.dot(c(layer.v_proj), .d).reshape(.{ B, S, KVH, hd }).withTags(.{ .b, .s, .nh, .hd });
        v = zml.nn.rmsNorm(v, .hd, RMS_EPS);
        // === point d'extension post_v_norm (V post-v_norm, pré-cache) ===
        // comptime-mort pour une brique sans cette méthode (ex: struct{}) → V inchangé → bit-exact decode4.
        if (@hasDecl(@TypeOf(brick), "post_v_norm")) {
            const ctx = LayerCtx{ .layer_idx = i, .is_full = full };
            v = brick.post_v_norm(v, ctx);
        }
        const v_new = v.transpose(.{ .b, .nh, .s, .hd }).rename(.{ .nh = .h, .s = .k });

        if (full) {
            const slot = zml.Tensor.scalar(@as(u32, @intCast(fullSlot(i))), .u32);
            cache.fl_k = cache.fl_k.scatterSlices(.{ .slot = slot, .k = pos_u }, k_new, so);
            cache.fl_v = cache.fl_v.scatterSlices(.{ .slot = slot, .k = pos_u }, v_new, so);
            cache_k = cache.fl_k.choose1d(.slot, fullSlot(i));
            cache_v = cache.fl_v.choose1d(.slot, fullSlot(i));
        } else {
            const slot = zml.Tensor.scalar(@as(u32, @intCast(slidingSlot(i))), .u32);
            cache.sl_k = cache.sl_k.scatterSlices(.{ .slot = slot, .k = pos_u }, k_new, so);
            cache.sl_v = cache.sl_v.scatterSlices(.{ .slot = slot, .k = pos_u }, v_new, so);
            cache_k = cache.sl_k.choose1d(.slot, slidingSlot(i));
            cache_v = cache.sl_v.choose1d(.slot, slidingSlot(i));
        }
    }

    const qs = q_final.splitAxis(.h, .{ .h = cache_k.dim(.h), .hq = .auto });
    var scores = qs.dot(cache_k, .hd).merge(.{ .h = .{ .h, .hq } }).transpose(.{ .b, .h, .q, .k });
    scores = scores.add(mask.broad(scores.shape()));
    const probs = scores.softmax(.k);

    const ps = probs.splitAxis(.h, .{ .h = cache_v.dim(.h), .hq = .auto });
    const ctx_attn = ps.dot(cache_v, .k).merge(.{ .h = .{ .h, .hq } }).transpose(.{ .b, .h, .q, .hd });
    const attn_m = ctx_attn.transpose(.{ .b, .q, .h, .hd }).merge(.{ .m = .{ .h, .hd } });
    const attn_out = attn_m.dot(c(layer.o_proj), .m).rename(.{ .q = .s });

    const h1 = hidden.add(rmsScaleD(attn_out, c(layer.post_attention_layernorm)));
    const xff = rmsScaleD(h1, c(layer.pre_feedforward_layernorm));
    const mlp_out = xff.dot(c(layer.gate_proj), .d).gelu().mul(xff.dot(c(layer.up_proj), .d)).dot(c(layer.down_proj), .f);
    const h2 = h1.add(rmsScaleD(mlp_out, c(layer.post_feedforward_layernorm)));

    var g = h2.dot(c(layer.per_layer_input_gate), .d).gelu();
    g = g.mul(ple_i);
    g = g.dot(c(layer.per_layer_projection), .p);
    const h3 = h2.add(rmsScaleD(g, c(layer.post_per_layer_input_norm)));

    return h3.mul(c(layer.layer_scalar).asScalar());
}

/// Le socle : model decode générique paramétré comptime par une brique. `EngineModel(struct{})`
/// reproduit decode4 (gate E1) ; `EngineModel(MaBrique)` injecte une transformation au(x) point(s)
/// d'extension sans copier le moteur.
pub fn EngineModel(comptime Brick: type) type {
    return struct {
        embed_tokens: zml.Tensor, // {voc,d} lm_head tied
        per_layer_model_projection: zml.Tensor,
        per_layer_projection_norm: zml.Tensor,
        final_norm: zml.Tensor,
        layers: []LayerW,
        brick: Brick,

        const Self = @This();

        /// Initialise les poids depuis `base` (checkpoint). La brique est mise à `struct{}`-vide par
        /// défaut ; une brique avec constantes (E2) exposera `init(View)` et sera câblée sur la fixture
        /// (les constantes brick vivent dans un store distinct des poids — threading fait en Task 2).
        pub fn init(allocator: std.mem.Allocator, base: zml.io.TensorStore.View) !Self {
            const layers = try allocator.alloc(LayerW, NUM_LAYERS);
            const layers_base = base.withPrefix("layers");
            for (layers, 0..) |*layer, i| layer.* = LayerW.init(layers_base.withLayer(i));
            return .{
                .embed_tokens = base.createTensor("embed_tokens.weight", .{ .voc, .d }, null),
                .per_layer_model_projection = base.createTensor("per_layer_model_projection.weight", .{ .lf, .d }, null),
                .per_layer_projection_norm = base.createTensor("per_layer_projection_norm.weight", .{.p}, null),
                .final_norm = base.createTensor("norm.weight", .{.d}, null),
                .layers = layers,
                .brick = .{},
            };
        }

        pub fn load(self: *const Self, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, store: *const zml.io.TensorStore, shardings: []const zml.sharding.Sharding) !zml.Bufferized(Self) {
            return zml.io.load(Self, self, allocator, io, platform, store, .{ .shardings = shardings, .parallelism = 1, .dma_chunks = 1, .dma_chunk_size = 16 * 1024 * 1024 });
        }

        fn perLayerInputs(self: Self, embptl_slice: zml.Tensor, embeds: zml.Tensor) zml.Tensor {
            const token_identity = embptl_slice
                .scale(SQRT_PLE).convert(.f32)
                .reshape(.{ B, S, NUM_LAYERS, PLE_DIM }).withTags(.{ .b, .s, .layer, .p });
            const context = embeds.dot(c(self.per_layer_model_projection), .d)
                .scale(INV_SQRT_HID)
                .reshape(.{ B, S, NUM_LAYERS, PLE_DIM }).withTags(.{ .b, .s, .layer, .p });
            const context_norm = rmsScaleP(context, c(self.per_layer_projection_norm));
            return context_norm.add(token_identity).scale(INV_SQRT_2);
        }

        /// Un pas de génération : sélectionne le step, embed+PLE, 35 couches (cache threadé) -> logits +
        /// cache grandi. Retour : {logits {b,s,voc}, sl_k, sl_v, fl_k, fl_v}.
        pub fn forward(self: Self, p: Packed, cache_in: Cache, ctrl: Ctrl) struct { zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor } {
            const step = ctrl.step;
            const embed_slice = pickStep(p.embeds, step);
            const embptl_slice = pickStep(p.embptls, step);
            const cos = pickStep(p.cos_full, step);
            const sin = pickStep(p.sin_full, step);
            const mask = pickStep(p.masks, step);
            const pos_i = pickStep(p.positions, step); // {} i32
            const pos_s = pos_i.reshape(.{1}).withTags(.{.s});
            const pos_u = pos_i.convert(.u32);

            const embeds = embed_slice.convert(.f32).scale(EMBED_SCALE);
            const ple = self.perLayerInputs(embptl_slice, embeds);
            var hidden = embeds;
            var cache = cache_in;
            inline for (0..NUM_LAYERS) |i| {
                const ple_i = ple.choose1d(.layer, @as(i64, @intCast(i)));
                hidden = runLayerGen(self.layers[i], i, hidden, ple_i, cos, sin, mask, pos_s, pos_u, &cache, self.brick);
            }
            const last_hidden = rmsScaleD(hidden, c(self.final_norm));
            const raw = last_hidden.dot(c(self.embed_tokens), .d);
            const logits = raw.scale(INV_SOFTCAP).tanh().scale(SOFTCAP);
            return .{ logits, cache.sl_k, cache.sl_v, cache.fl_k, cache.fl_v };
        }
    };
}
