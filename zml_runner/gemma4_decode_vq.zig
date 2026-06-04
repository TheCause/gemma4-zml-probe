// Q4 — MOTEUR decode 1-step Gemma-4-E2B-it AVEC V-QUANT (MSE V-only) inséré au point cache.
//
// COPIE de gemma4_decode3.zig (gate decode immuable) + insertion du quantizer V dans la branche producer,
// entre v_norm (rmsNorm UNSCALED) et le transpose vers le cache. La chaine quantizeV (norm fp16 + Hadamard
// + nearest-centroid + inverse) est REPRISE telle quelle de gemma4_vquant.zig (prouvée bit-exact en Q3).
//
// V est 4D [.b,.s,.nh,.hd] mais en decode 1-step .b=.s=.nh=1 -> reshape à [.k=1,.hd] (axes attendus par
// quantizeV qui utilise y.dim(.k)), puis reshape back avant le transpose existant. K reste fp32 (V-only).
// Le couple {codebook, hadamard} est sélectionné par le flag full : cb_512/Pi_512 si full(i), sinon 256.
//
// Comparé à un oracle HF où V est quantifié de la MÊME façon (mêmes constantes Task 0) : portage seul ->
// bit-near attendu (last_hidden max_abs<=1e-2, mean_abs<=1e-4, argmax == HF-V-quant).
//
// CLI : gemma4_decode_vq <model.safetensors> <decode_vq.safetensors>

const std = @import("std");
const log = std.log;
const zml = @import("zml");

pub const std_options: std.Options = .{ .log_level = .info };

const NUM_LAYERS: usize = 35;
const FIRST_KV_SHARED: usize = 15;
const SLIDING_WRITER: usize = 13;
const FULL_WRITER: usize = 14;

const B: i64 = 1;
const S: i64 = 1;
const D: i64 = 1536;
const NH: i64 = 8;
const KVH: i64 = 1;
const HD_SLIDING: i64 = 256;
const HD_FULL: i64 = 512;
const PLE_DIM: i64 = 256;
const LF: i64 = 8960;
const VOC: i64 = 262144;

const RMS_EPS: f32 = 1.0e-6;
const ROPE_THETA_SLIDING: f32 = 1.0e4;
const EMBED_SCALE: f64 = @sqrt(1536.0);
const INV_SQRT_HID: f64 = 1.0 / @sqrt(1536.0);
const SQRT_PLE: f64 = 16.0;
const INV_SQRT_2: f64 = 0.7071067811865476;
const SOFTCAP: f64 = 30.0;
const INV_SOFTCAP: f64 = 1.0 / 30.0;

fn isFull(i: usize) bool {
    return (i + 1) % 5 == 0;
}
fn isReader(i: usize) bool {
    return i >= FIRST_KV_SHARED;
}
// Slot du cache d'un producer i dans le tenseur empaqueté correspondant (calculé au trace-time).
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

// MSE V-only quantizer chain — REPRISE bit-exacte de gemma4_vquant.zig (Q3).
// v:[.k,.hd], cb:[.c], Pi:[.e,.hd]  ->  v_hat:[.k,.hd]
fn quantizeV(v: zml.Tensor, cb: zml.Tensor, Pi: zml.Tensor) zml.Tensor {
    const norm = v.mul(v).sum(.hd).sqrt().convert(.f16).convert(.f32); // [.k,.hd=1]
    const u = v.div(norm); // broadcast (.hd=1)
    const y = u.dot(Pi, .hd); // [.k,.e]  (= u @ Pi.T)
    const target = zml.Shape.init(.{ y.dim(.k), y.dim(.e), cb.dim(.c) }, .f32)
        .withTags(.{ .k, .e, .c });
    const yr3 = y.appendAxes(.{.c}).broad(target);
    const cb3 = cb.insertAxes(0, .{ .k, .e }).broad(target);
    const diff = yr3.sub(cb3);
    const idx = diff.mul(diff).scale(-1.0).argMax(.c).indices.squeeze(.c); // [.k,.e]
    const y_hat = cb.gather(.{ .c = idx }, .{}); // [.k,.e]
    const u_hat = y_hat.dot(Pi, .e); // [.k,.hd]  (= y_hat @ Pi)
    return u_hat.mul(norm);
}

inline fn c(t: zml.Tensor) zml.Tensor {
    return t.convert(.f32);
}
fn rmsScaleD(x: zml.Tensor, w: zml.Tensor) zml.Tensor {
    const n = zml.nn.rmsNorm(x, .d, RMS_EPS);
    return n.mul(w.broad(n.shape()));
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

const LayerW = struct {
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

const SharedKV = struct {
    k_sliding: ?zml.Tensor = null,
    v_sliding: ?zml.Tensor = null,
    k_full: ?zml.Tensor = null,
    v_full: ?zml.Tensor = null,
};

const Runtime = struct {
    embed_slice: zml.Tensor, // {b,s,d} bf16
    embptl_slice: zml.Tensor, // {b,s,lf} bf16
    cos_full: zml.Tensor, // {b,s,hd=512} pos p
    sin_full: zml.Tensor,
    mask: zml.Tensor, // {b,h,q,k}
    cache_sl_k: zml.Tensor, // {slot=12,b,h,k=5,hd=256}
    cache_sl_v: zml.Tensor,
    cache_fl_k: zml.Tensor, // {slot=3,b,h,k=5,hd=512}
    cache_fl_v: zml.Tensor,
    pos: zml.Tensor, // {s=1} i32 = [p]
    // Constantes MSE V-quant (Task 0) — sélectionnées par head_dim (sliding 256 / full 512).
    codebook_256: zml.Tensor, // {c=16}
    hadamard_256: zml.Tensor, // {e=256,hd=256}
    codebook_512: zml.Tensor, // {c=16}
    hadamard_512: zml.Tensor, // {e=512,hd=512}

    pub fn init(v: zml.io.TensorStore.View) Runtime {
        return .{
            .embed_slice = v.createTensor("embed_slice", .{ .b, .s, .d }, null),
            .embptl_slice = v.createTensor("embptl_slice", .{ .b, .s, .lf }, null),
            .cos_full = v.createTensor("cos_full", .{ .b, .s, .hd }, null),
            .sin_full = v.createTensor("sin_full", .{ .b, .s, .hd }, null),
            .mask = v.createTensor("mask_decode", .{ .b, .h, .q, .k }, null),
            .cache_sl_k = v.createTensor("cache_sl_k", .{ .slot, .b, .h, .k, .hd }, null),
            .cache_sl_v = v.createTensor("cache_sl_v", .{ .slot, .b, .h, .k, .hd }, null),
            .cache_fl_k = v.createTensor("cache_fl_k", .{ .slot, .b, .h, .k, .hd }, null),
            .cache_fl_v = v.createTensor("cache_fl_v", .{ .slot, .b, .h, .k, .hd }, null),
            .pos = v.createTensor("pos_idx", .{.s}, null),
            .codebook_256 = v.createTensor("codebook_256", .{.c}, null),
            .hadamard_256 = v.createTensor("hadamard_256", .{ .e, .hd }, null),
            .codebook_512 = v.createTensor("codebook_512", .{.c}, null),
            .hadamard_512 = v.createTensor("hadamard_512", .{ .e, .hd }, null),
        };
    }

    pub fn load(self: *const Runtime, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, store: *const zml.io.TensorStore, shardings: []const zml.sharding.Sharding) !zml.Bufferized(Runtime) {
        return zml.io.load(Runtime, self, allocator, io, platform, store, .{ .shardings = shardings, .parallelism = 1, .dma_chunks = 1, .dma_chunk_size = 16 * 1024 * 1024 });
    }
};

/// Forward decode d'UNE couche i (S=1). Producer: compute K/V du token p, QUANTIFIE V (MSE V-only) au point
/// v_norm, scatter dans son cache prefill (empaqueté), attend sur le cache grandi ; publie aux writers 13/14.
/// Reader: réutilise shared (cache grandi du writer). Puis MLP + bloc PLE + layer_scalar (identiques prefill).
fn runLayerDecode(layer: LayerW, comptime i: usize, hidden: zml.Tensor, ple_i: zml.Tensor, rt: Runtime, shared: *SharedKV) zml.Tensor {
    const full = isFull(i);
    const reader = isReader(i);
    const hd: i64 = if (full) HD_FULL else HD_SLIDING;
    const half: i64 = @divExact(hd, 2);

    const input_ln = c(layer.input_layernorm);
    const q_proj = c(layer.q_proj);
    const q_norm = c(layer.q_norm);
    const o_proj = c(layer.o_proj);
    const post_attn_ln = c(layer.post_attention_layernorm);
    const pre_ff_ln = c(layer.pre_feedforward_layernorm);
    const gate_proj = c(layer.gate_proj);
    const up_proj = c(layer.up_proj);
    const down_proj = c(layer.down_proj);
    const post_ff_ln = c(layer.post_feedforward_layernorm);
    const ple_gate = c(layer.per_layer_input_gate);
    const ple_proj = c(layer.per_layer_projection);
    const ple_norm = c(layer.post_per_layer_input_norm);
    const layer_scalar = c(layer.layer_scalar);

    const h0 = rmsScaleD(hidden, input_ln);

    var q = h0.dot(q_proj, .d).reshape(.{ B, S, NH, hd }).withTags(.{ .b, .s, .nh, .hd });
    q = zml.nn.rmsNorm(q, .hd, RMS_EPS).mul(q_norm.broad(q.shape()));
    q = if (full) manualRope(q, rt.cos_full, rt.sin_full, half) else slidingRope(q, rt.pos);
    const q_final = q.transpose(.{ .b, .nh, .s, .hd }).rename(.{ .nh = .h, .s = .q });

    var cache_k: zml.Tensor = undefined;
    var cache_v: zml.Tensor = undefined;
    if (reader) {
        if (full) {
            cache_k = shared.k_full.?;
            cache_v = shared.v_full.?;
        } else {
            cache_k = shared.k_sliding.?;
            cache_v = shared.v_sliding.?;
        }
    } else {
        const k_proj = c(layer.k_proj);
        const k_norm = c(layer.k_norm);
        const v_proj = c(layer.v_proj);
        var k = h0.dot(k_proj, .d).reshape(.{ B, S, KVH, hd }).withTags(.{ .b, .s, .nh, .hd });
        k = zml.nn.rmsNorm(k, .hd, RMS_EPS).mul(k_norm.broad(k.shape()));
        k = if (full) manualRope(k, rt.cos_full, rt.sin_full, half) else slidingRope(k, rt.pos);
        const k_new = k.transpose(.{ .b, .nh, .s, .hd }).rename(.{ .nh = .h, .s = .k });

        var v = h0.dot(v_proj, .d).reshape(.{ B, S, KVH, hd }).withTags(.{ .b, .s, .nh, .hd });
        v = zml.nn.rmsNorm(v, .hd, RMS_EPS); // v_norm SANS scale

        // === Q4 : V-QUANT MSE V-only au point cache (entre v_norm et transpose) ===
        // decode 1-step : .b=.s=.nh=1 -> reshape V à [.k=1,.hd] (axes attendus par quantizeV),
        // sélectionner {codebook,hadamard} par head_dim, quantifier, puis reshape back à [.b,.s,.nh,.hd].
        const cb = if (full) rt.codebook_512 else rt.codebook_256;
        const Pi = if (full) rt.hadamard_512 else rt.hadamard_256;
        const v2d = v.reshape(.{ B, hd }).withTags(.{ .k, .hd });
        const vq2d = quantizeV(v2d, cb, Pi); // [.k=1,.hd]
        v = vq2d.reshape(.{ B, S, KVH, hd }).withTags(.{ .b, .s, .nh, .hd });
        // =========================================================================

        const v_new = v.transpose(.{ .b, .nh, .s, .hd }).rename(.{ .nh = .h, .s = .k });

        // cache prefill de ce producer (empaqueté multi-slots) + append à pos p
        const pref_k = if (full) rt.cache_fl_k.choose1d(.slot, fullSlot(i)) else rt.cache_sl_k.choose1d(.slot, slidingSlot(i));
        const pref_v = if (full) rt.cache_fl_v.choose1d(.slot, fullSlot(i)) else rt.cache_sl_v.choose1d(.slot, slidingSlot(i));
        const pos_u = rt.pos.squeeze(.s).convert(.u32);
        const so = zml.Tensor.ScatterOpts{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override };
        cache_k = pref_k.scatterSlices(.{ .k = pos_u }, k_new, so);
        cache_v = pref_v.scatterSlices(.{ .k = pos_u }, v_new, so);

        if (i == SLIDING_WRITER) {
            shared.k_sliding = cache_k;
            shared.v_sliding = cache_v;
        }
        if (i == FULL_WRITER) {
            shared.k_full = cache_k;
            shared.v_full = cache_v;
        }
    }

    const qs = q_final.splitAxis(.h, .{ .h = cache_k.dim(.h), .hq = .auto });
    var scores = qs.dot(cache_k, .hd).merge(.{ .h = .{ .h, .hq } }).transpose(.{ .b, .h, .q, .k });
    scores = scores.add(rt.mask.broad(scores.shape()));
    const probs = scores.softmax(.k);

    const ps = probs.splitAxis(.h, .{ .h = cache_v.dim(.h), .hq = .auto });
    const ctx = ps.dot(cache_v, .k).merge(.{ .h = .{ .h, .hq } }).transpose(.{ .b, .h, .q, .hd });
    const attn_m = ctx.transpose(.{ .b, .q, .h, .hd }).merge(.{ .m = .{ .h, .hd } });
    const attn_out = attn_m.dot(o_proj, .m).rename(.{ .q = .s });

    const h1 = hidden.add(rmsScaleD(attn_out, post_attn_ln));
    const xff = rmsScaleD(h1, pre_ff_ln);
    const mlp_out = xff.dot(gate_proj, .d).gelu().mul(xff.dot(up_proj, .d)).dot(down_proj, .f);
    const h2 = h1.add(rmsScaleD(mlp_out, post_ff_ln));

    var g = h2.dot(ple_gate, .d).gelu();
    g = g.mul(ple_i);
    g = g.dot(ple_proj, .p);
    const h3 = h2.add(rmsScaleD(g, ple_norm));

    return h3.mul(layer_scalar.asScalar());
}

const Engine = struct {
    embed_tokens: zml.Tensor, // {voc,d} = lm_head tied
    per_layer_model_projection: zml.Tensor, // {lf,d}
    per_layer_projection_norm: zml.Tensor, // {p}
    final_norm: zml.Tensor, // {d}
    layers: []LayerW,

    pub fn init(allocator: std.mem.Allocator, base: zml.io.TensorStore.View) !Engine {
        const layers = try allocator.alloc(LayerW, NUM_LAYERS);
        const layers_base = base.withPrefix("layers");
        for (layers, 0..) |*layer, i| layer.* = LayerW.init(layers_base.withLayer(i));
        return .{
            .embed_tokens = base.createTensor("embed_tokens.weight", .{ .voc, .d }, null),
            .per_layer_model_projection = base.createTensor("per_layer_model_projection.weight", .{ .lf, .d }, null),
            .per_layer_projection_norm = base.createTensor("per_layer_projection_norm.weight", .{.p}, null),
            .final_norm = base.createTensor("norm.weight", .{.d}, null),
            .layers = layers,
        };
    }

    pub fn load(self: *const Engine, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, store: *const zml.io.TensorStore, shardings: []const zml.sharding.Sharding) !zml.Bufferized(Engine) {
        return zml.io.load(Engine, self, allocator, io, platform, store, .{ .shardings = shardings, .parallelism = 1, .dma_chunks = 1, .dma_chunk_size = 16 * 1024 * 1024 });
    }

    fn perLayerInputs(self: Engine, rt: Runtime, embeds: zml.Tensor) zml.Tensor {
        const token_identity = rt.embptl_slice
            .scale(SQRT_PLE).convert(.f32)
            .reshape(.{ B, S, NUM_LAYERS, PLE_DIM }).withTags(.{ .b, .s, .layer, .p });
        const context = embeds.dot(c(self.per_layer_model_projection), .d)
            .scale(INV_SQRT_HID)
            .reshape(.{ B, S, NUM_LAYERS, PLE_DIM }).withTags(.{ .b, .s, .layer, .p });
        const context_norm = rmsScaleP(context, c(self.per_layer_projection_norm));
        return context_norm.add(token_identity).scale(INV_SQRT_2);
    }

    /// Forward decode e2e : embed -> PLE -> 35 couches (caches, V-quant) -> final norm -> lm_head + softcap.
    /// Retour : {last_hidden {b,s,d}, logits {b,s,voc}}.
    pub fn forward(self: Engine, rt: Runtime) struct { zml.Tensor, zml.Tensor } {
        const embeds = rt.embed_slice.convert(.f32).scale(EMBED_SCALE);
        const ple = self.perLayerInputs(rt, embeds);
        var hidden = embeds;
        var shared = SharedKV{};
        inline for (0..NUM_LAYERS) |i| {
            const ple_i = ple.choose1d(.layer, @as(i64, @intCast(i)));
            hidden = runLayerDecode(self.layers[i], i, hidden, ple_i, rt, &shared);
        }
        const last_hidden = rmsScaleD(hidden, c(self.final_norm));
        const raw = last_hidden.dot(c(self.embed_tokens), .d); // {b,s,voc} (lm_head tied)
        const logits = raw.scale(INV_SOFTCAP).tanh().scale(SOFTCAP); // softcap 30·tanh(x/30)
        return .{ last_hidden, logits };
    }
};

const Oracle = struct {
    last_hidden: zml.Tensor,
    logits: zml.Tensor,
    argmax: zml.Tensor,

    pub fn init(v: zml.io.TensorStore.View) Oracle {
        return .{
            .last_hidden = v.createTensor("last_hidden", .{ .b, .s, .d }, null),
            .logits = v.createTensor("logits", .{ .b, .voc }, null),
            .argmax = v.createTensor("argmax", .{.one}, null),
        };
    }

    pub fn load(self: *const Oracle, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, store: *const zml.io.TensorStore, shardings: []const zml.sharding.Sharding) !zml.Bufferized(Oracle) {
        return zml.io.load(Oracle, self, allocator, io, platform, store, .{ .shardings = shardings, .parallelism = 1, .dma_chunks = 1, .dma_chunk_size = 16 * 1024 * 1024 });
    }
};

fn scanHidden(allocator: std.mem.Allocator, io: std.Io, out_buf: *zml.Buffer, ref_buf: *zml.Buffer) !bool {
    var out_s = try out_buf.toSliceAlloc(allocator, io);
    defer out_s.free(allocator);
    var ref_s = try ref_buf.toSliceAlloc(allocator, io);
    defer ref_s.free(allocator);
    const out = out_s.items(f32);
    const ref = ref_s.items(f32);
    if (out.len != ref.len) return error.LengthMismatch;
    var max_abs: f32 = 0.0;
    var sum_abs: f64 = 0.0;
    var nan_inf = false;
    for (out, ref) |a, bb| {
        if (std.math.isNan(a) or std.math.isInf(a)) nan_inf = true;
        const diff = @abs(a - bb);
        if (diff > max_abs) max_abs = diff;
        sum_abs += @as(f64, diff);
    }
    const mean_abs = @as(f32, @floatCast(sum_abs / @as(f64, @floatFromInt(out.len))));
    const pass = !nan_inf and max_abs <= 1.0e-2 and mean_abs <= 1.0e-4;
    const verdict = if (nan_inf) "FAIL(NaN/Inf)" else if (pass) "PASS" else if (max_abs <= 1.0e-1) "WARN" else "FAIL";
    log.info("  last_hidden (n={d}) max_abs={e:.4} mean_abs={e:.4} -> {s}", .{ out.len, max_abs, mean_abs, verdict });
    return pass;
}

fn scanLogits(allocator: std.mem.Allocator, io: std.Io, out_buf: *zml.Buffer, ref_buf: *zml.Buffer) !bool {
    var out_s = try out_buf.toSliceAlloc(allocator, io);
    defer out_s.free(allocator);
    var ref_s = try ref_buf.toSliceAlloc(allocator, io);
    defer ref_s.free(allocator);
    const out = out_s.items(f32);
    const ref = ref_s.items(f32);
    if (out.len != ref.len) return error.LengthMismatch;
    var max_abs: f32 = 0.0;
    var sum_abs: f64 = 0.0;
    for (out, ref) |a, bb| {
        const diff = @abs(a - bb);
        if (diff > max_abs) max_abs = diff;
        sum_abs += @as(f64, diff);
    }
    const mean_abs = @as(f32, @floatCast(sum_abs / @as(f64, @floatFromInt(out.len))));
    const pass = max_abs <= 1.0e-1; // logits amplifiés (matmul lm_head) — critères liants = hidden+argmax
    log.info("  logits     (n={d}) max_abs={e:.4} mean_abs={e:.4} -> {s}", .{ out.len, max_abs, mean_abs, if (pass) "PASS" else "WARN" });
    return pass;
}

fn argmaxOf(allocator: std.mem.Allocator, io: std.Io, logits_buf: *zml.Buffer) !i64 {
    var s = try logits_buf.toSliceAlloc(allocator, io);
    defer s.free(allocator);
    const v = s.items(f32);
    var best: usize = 0;
    var best_val: f32 = v[0];
    for (v, 0..) |x, idx| {
        if (x > best_val) {
            best_val = x;
            best = idx;
        }
    }
    return @intCast(best);
}

pub fn main(init: std.process.Init) !void {
    const arena = init.arena;
    const allocator = init.gpa;
    const io = init.io;

    const process_args = try init.minimal.args.toSlice(arena.allocator());
    if (process_args.len < 3) {
        log.err("Usage: gemma4_decode_vq <model.safetensors> <decode_vq.safetensors>", .{});
        return error.MissingArgument;
    }
    const ckpt = process_args[1];
    const fixture = process_args[2];
    log.info("Q4 — moteur decode e2e V-QUANT (35 couches, 1 token, MSE V-only)", .{});

    var reg_ck: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, ckpt);
    var store_ck: zml.io.TensorStore = .fromRegistry(allocator, &reg_ck);
    var reg_fx: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, fixture);
    defer reg_fx.deinit();
    var store_fx: zml.io.TensorStore = .fromRegistry(allocator, &reg_fx);
    defer store_fx.deinit();

    const base = store_ck.view().withPrefix("model").withPrefix("language_model");
    const engine: Engine = try .init(arena.allocator(), base);
    const rt: Runtime = .init(store_fx.view());
    const oracle: Oracle = .init(store_fx.view());

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);
    const sharding = try zml.sharding.replicatedSharding(platform);

    log.info("Materializing weights (35 couches + frontend + lm_head + constantes V-quant)...", .{});
    const eng_buf = try engine.load(arena.allocator(), io, platform, &store_ck, &.{sharding});
    const rt_buf = try rt.load(arena.allocator(), io, platform, &store_fx, &.{sharding});
    var orc_buf = try oracle.load(arena.allocator(), io, platform, &store_fx, &.{sharding});
    store_ck.deinit();
    reg_ck.deinit();

    log.info("Compiling decode-vq forward...", .{});
    var exe = try platform.compile(allocator, io, engine, .forward, .{rt}, .{ .shardings = &.{sharding} });
    defer exe.deinit();
    var args = try exe.args(allocator);
    defer args.deinit(allocator);
    var results = try exe.results(allocator);
    defer results.deinit(allocator);
    args.set(.{ eng_buf, rt_buf });
    exe.call(args, &results);

    var r_hidden, var r_logits = results.get(struct { zml.Buffer, zml.Buffer });
    defer {
        r_hidden.deinit();
        r_logits.deinit();
    }

    log.info("Comparaison e2e vs oracle HF-V-quant :", .{});
    const hidden_ok = try scanHidden(allocator, io, &r_hidden, &orc_buf.last_hidden);
    const logits_ok = try scanLogits(allocator, io, &r_logits, &orc_buf.logits);

    const zml_argmax = try argmaxOf(allocator, io, &r_logits);
    var oa_s = try orc_buf.argmax.toSliceAlloc(allocator, io);
    defer oa_s.free(allocator);
    const hf_argmax: i64 = @intCast(oa_s.items(i32)[0]);
    const argmax_ok = zml_argmax == hf_argmax;
    log.info("  argmax (token suivant) : ZML={d} HF-V-quant={d} -> {s}", .{ zml_argmax, hf_argmax, if (argmax_ok) "PASS" else "FAIL" });

    if (hidden_ok and logits_ok and argmax_ok) {
        log.info("Q4 decode-vq PASS — V-quant inséré au point cache : decode 1-step ZML == HF-V-quant (last_hidden bit-near + argmax)", .{});
    } else {
        log.err("Q4 decode-vq : divergence (hidden_ok={} logits_ok={} argmax_ok={})", .{ hidden_ok, logits_ok, argmax_ok });
        return error.DecodeMismatch;
    }
}
