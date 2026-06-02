// P5.7.8 — BOUCLE DE GÉNÉRATION decode ZML (N tokens) -> séquence == HF greedy.
//
// Le tour d'honneur. Le moteur decode-3 répété en boucle, en threadant le cache grandi de step en step
// (idiome llama KvCache : scatter à (.slot, .k=pos) dans le cache empaqueté, retourné et réinjecté par
// le main). Validation teacher-forcing vérifié : on feed les tokens HF pré-gatherés (embeds dans la
// fixture) et on vérifie que l'argmax ZML de chaque step == le token suivant de HF.
//
// CLI : gemma4_decode4 <model.safetensors> <p5_7_8_gen.safetensors>

const std = @import("std");
const log = std.log;
const zml = @import("zml");

pub const std_options: std.Options = .{ .log_level = .info };

const NUM_LAYERS: usize = 35;
const FIRST_KV_SHARED: usize = 15;
const SLIDING_WRITER: usize = 13;
const FULL_WRITER: usize = 14;
const SLIDING_WRITER_SLOT: i64 = 11; // slidingSlot(13)
const FULL_WRITER_SLOT: i64 = 2; // fullSlot(14)

const B: i64 = 1;
const S: i64 = 1;
const D: i64 = 1536;
const NH: i64 = 8;
const KVH: i64 = 1;
const HD_SLIDING: i64 = 256;
const HD_FULL: i64 = 512;
const PLE_DIM: i64 = 256;
const NUM_STEPS: usize = 4;

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

// Cache empaqueté threadé (entrée + sortie du forward) : 2 types (sliding/full) × {slot,b,h,k,hd}.
const Cache = struct {
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
const Packed = struct {
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
const Ctrl = struct {
    step: zml.Tensor,
    pub fn initSymbolic() Ctrl {
        return .{ .step = zml.Tensor.init(.{}, .u32) };
    }
};

// Séquence attendue (HF), lue côté host.
const ExpW = struct {
    e: zml.Tensor,
    pub fn init(v: zml.io.TensorStore.View) ExpW {
        return .{ .e = v.createTensor("expected", .{.step}, null) };
    }
    pub fn load(self: *const ExpW, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, store: *const zml.io.TensorStore, shardings: []const zml.sharding.Sharding) !zml.Bufferized(ExpW) {
        return zml.io.load(ExpW, self, allocator, io, platform, store, .{ .shardings = shardings, .parallelism = 1, .dma_chunks = 1, .dma_chunk_size = 16 * 1024 * 1024 });
    }
};

fn pickStep(t: zml.Tensor, step: zml.Tensor) zml.Tensor {
    return t.dynamicSlice(.{ .step = zml.Tensor.DynSlice{ .start = step, .len = 1 } }).squeeze(.step);
}

/// Forward decode d'UNE couche i en mode génération (cache threadé). Producer scatter K/V du token à
/// (.slot, .k=pos) dans le cache empaqueté ; reader lit le slot du writer 13/14.
fn runLayerGen(layer: LayerW, comptime i: usize, hidden: zml.Tensor, ple_i: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor, mask: zml.Tensor, pos_s: zml.Tensor, pos_u: zml.Tensor, cache: *Cache) zml.Tensor {
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
    const ctx = ps.dot(cache_v, .k).merge(.{ .h = .{ .h, .hq } }).transpose(.{ .b, .h, .q, .hd });
    const attn_m = ctx.transpose(.{ .b, .q, .h, .hd }).merge(.{ .m = .{ .h, .hd } });
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

const Engine = struct {
    embed_tokens: zml.Tensor, // {voc,d} lm_head tied
    per_layer_model_projection: zml.Tensor,
    per_layer_projection_norm: zml.Tensor,
    final_norm: zml.Tensor,
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

    fn perLayerInputs(self: Engine, embptl_slice: zml.Tensor, embeds: zml.Tensor) zml.Tensor {
        const token_identity = embptl_slice
            .scale(SQRT_PLE).convert(.f32)
            .reshape(.{ B, S, NUM_LAYERS, PLE_DIM }).withTags(.{ .b, .s, .layer, .p });
        const context = embeds.dot(c(self.per_layer_model_projection), .d)
            .scale(INV_SQRT_HID)
            .reshape(.{ B, S, NUM_LAYERS, PLE_DIM }).withTags(.{ .b, .s, .layer, .p });
        const context_norm = rmsScaleP(context, c(self.per_layer_projection_norm));
        return context_norm.add(token_identity).scale(INV_SQRT_2);
    }

    /// Un pas de génération : sélectionne le step, embed+PLE, 35 couches (cache threadé), -> logits +
    /// cache grandi. Retour : {logits {b,s,voc}, sl_k, sl_v, fl_k, fl_v}.
    pub fn forward(self: Engine, p: Packed, cache_in: Cache, ctrl: Ctrl) struct { zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor } {
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
            hidden = runLayerGen(self.layers[i], i, hidden, ple_i, cos, sin, mask, pos_s, pos_u, &cache);
        }
        const last_hidden = rmsScaleD(hidden, c(self.final_norm));
        const raw = last_hidden.dot(c(self.embed_tokens), .d);
        const logits = raw.scale(INV_SOFTCAP).tanh().scale(SOFTCAP);
        return .{ logits, cache.sl_k, cache.sl_v, cache.fl_k, cache.fl_v };
    }
};

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
        log.err("Usage: gemma4_decode4 <model.safetensors> <p5_7_8_gen.safetensors>", .{});
        return error.MissingArgument;
    }
    const ckpt = process_args[1];
    const fixture = process_args[2];
    log.info("P5.7.8 — boucle génération decode ({d} tokens)", .{NUM_STEPS});

    var reg_ck: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, ckpt);
    var store_ck: zml.io.TensorStore = .fromRegistry(allocator, &reg_ck);
    var reg_fx: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, fixture);
    defer reg_fx.deinit();
    var store_fx: zml.io.TensorStore = .fromRegistry(allocator, &reg_fx);
    defer store_fx.deinit();

    const base = store_ck.view().withPrefix("model").withPrefix("language_model");
    const engine: Engine = try .init(arena.allocator(), base);
    const packed_in: Packed = .init(store_fx.view());
    const cache0: Cache = .init(store_fx.view());
    const ctrl_sym: Ctrl = .initSymbolic();

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);
    const sharding = try zml.sharding.replicatedSharding(platform);

    log.info("Materializing weights + packed inputs + caches...", .{});
    const eng_buf = try engine.load(arena.allocator(), io, platform, &store_ck, &.{sharding});
    const pk_buf = try packed_in.load(arena.allocator(), io, platform, &store_fx, &.{sharding});
    var cache_buf = try cache0.load(arena.allocator(), io, platform, &store_fx, &.{sharding});
    const exp_sym: ExpW = .init(store_fx.view());
    var exp_buf = try exp_sym.load(arena.allocator(), io, platform, &store_fx, &.{sharding});
    store_ck.deinit();
    reg_ck.deinit();

    var exp_slice = try exp_buf.e.toSliceAlloc(allocator, io);
    defer exp_slice.free(allocator);
    const expected_tokens = exp_slice.items(i32);

    log.info("Compiling gen step...", .{});
    var exe = try platform.compile(allocator, io, engine, .forward, .{ packed_in, cache0, ctrl_sym }, .{ .shardings = &.{sharding} });
    defer exe.deinit();

    var all_pass = true;
    var step_idx: usize = 0;
    while (step_idx < NUM_STEPS) : (step_idx += 1) {
        var step_buf = try zml.Buffer.scalar(io, platform, @as(u32, @intCast(step_idx)), .u32, sharding);
        const ctrl_buf = zml.Bufferized(Ctrl){ .step = step_buf };

        var args = try exe.args(allocator);
        var results = try exe.results(allocator);
        args.set(.{ eng_buf, pk_buf, cache_buf, ctrl_buf });
        exe.call(args, &results);
        var r_logits, const r_slk, const r_slv, const r_flk, const r_flv = results.get(struct {
            zml.Buffer, zml.Buffer, zml.Buffer, zml.Buffer, zml.Buffer,
        });

        const tok = try argmaxOf(allocator, io, &r_logits);
        const exp = @as(i64, @intCast(expected_tokens[step_idx]));
        const ok = tok == exp;
        if (!ok) all_pass = false;
        log.info("  step {d} (pos {d}) : argmax ZML={d} HF={d} -> {s}", .{ step_idx, 4 + step_idx, tok, exp, if (ok) "PASS" else "FAIL" });

        // thread le cache grandi vers le step suivant
        cache_buf.sl_k.deinit();
        cache_buf.sl_v.deinit();
        cache_buf.fl_k.deinit();
        cache_buf.fl_v.deinit();
        cache_buf = zml.Bufferized(Cache){ .sl_k = r_slk, .sl_v = r_slv, .fl_k = r_flk, .fl_v = r_flv };
        r_logits.deinit();
        step_buf.deinit();
        args.deinit(allocator);
        results.deinit(allocator);
    }
    cache_buf.sl_k.deinit();
    cache_buf.sl_v.deinit();
    cache_buf.fl_k.deinit();
    cache_buf.fl_v.deinit();

    if (all_pass) {
        log.info("P5.7.8 PASS — le moteur ZML génère {d} tokens, séquence == HF greedy", .{NUM_STEPS});
    } else {
        log.err("P5.7.8 : divergence de séquence vs HF", .{});
        return error.GenMismatch;
    }
}
