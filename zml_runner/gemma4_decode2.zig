// P5.7.7 decode-2 — PILOTE decode incrémental FULL (writer 14 full × reader 19 full).
//
// Symétrique à gemma4_decode1.zig mais sur les couches FULL : head_dim 512, RoPE manuelle partielle
// (partial_rotary 0.25, theta 1e6, proportional) appliquée via manualRope avec cos/sin oracle 512-wide
// à pos p (mécanisme prouvé prefill P5.7.4). attention_k_eq_v=False -> V séparé (comme decode-1).
// La SEULE nouveauté vs decode-1 : head_dim 512 + manualRope(cos_full/sin_full) au lieu de zml.nn.rope.
//
// CLI : gemma4_decode2 <model.safetensors> <p5_7_7_decode2.safetensors>

const std = @import("std");
const log = std.log;
const zml = @import("zml");

pub const std_options: std.Options = .{ .log_level = .info };

const B: i64 = 1;
const S: i64 = 1;
const NH: i64 = 8;
const KVH: i64 = 1;
const HD: i64 = 512; // full head_dim
const HALF: i64 = HD / 2; // 256 (rotate_half complet ; le partial est encodé dans cos/sin oracle)
const RMS_EPS: f32 = 1.0e-6;

inline fn c(t: zml.Tensor) zml.Tensor {
    return t.convert(.f32);
}
fn rmsScaleHd(x: zml.Tensor, w: zml.Tensor) zml.Tensor {
    const n = zml.nn.rmsNorm(x, .hd, RMS_EPS);
    return n.mul(w.broad(n.shape()));
}
// RoPE manuelle full (calque gemma4_prefill.zig:manualRope) : rotate_half complet + cos/sin oracle.
fn manualRope(x: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor) zml.Tensor {
    const halves = x.split(.hd, &.{ HALF, HALF });
    const rh = zml.Tensor.concatenate(&.{ halves[1].negate(), halves[0] }, .hd);
    return x.mul(cos.broad(x.shape())).add(rh.mul(sin.broad(x.shape())));
}

// Attention d'un nouveau token (q=1) sur un cache grandi {b,h=1,k,hd=512}. q calculé depuis q_src {b,s=1,d}.
fn attendCache(q_src: zml.Tensor, q_proj: zml.Tensor, q_norm: zml.Tensor, o_proj: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor, cache_k: zml.Tensor, cache_v: zml.Tensor, mask: zml.Tensor) zml.Tensor {
    var q = q_src.dot(q_proj, .d).reshape(.{ B, S, NH, HD }).withTags(.{ .b, .s, .nh, .hd });
    q = rmsScaleHd(q, q_norm);
    q = manualRope(q, cos, sin);
    const q_final = q.transpose(.{ .b, .nh, .s, .hd }).rename(.{ .nh = .h, .s = .q });

    const qs = q_final.splitAxis(.h, .{ .h = cache_k.dim(.h), .hq = .auto });
    var scores = qs.dot(cache_k, .hd).merge(.{ .h = .{ .h, .hq } }).transpose(.{ .b, .h, .q, .k });
    scores = scores.add(mask.broad(scores.shape()));
    const probs = scores.softmax(.k);

    const ps = probs.splitAxis(.h, .{ .h = cache_v.dim(.h), .hq = .auto });
    const ctx = ps.dot(cache_v, .k).merge(.{ .h = .{ .h, .hq } }).transpose(.{ .b, .h, .q, .hd });
    const attn_m = ctx.transpose(.{ .b, .q, .h, .hd }).merge(.{ .m = .{ .h, .hd } });
    return attn_m.dot(o_proj, .m).rename(.{ .q = .s });
}

const Out6 = struct { zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor };

const Engine = struct {
    // layer 14 : writer full (q/k/v proj, q/k norm, o_proj)
    q14: zml.Tensor,
    qn14: zml.Tensor,
    k14: zml.Tensor,
    kn14: zml.Tensor,
    v14: zml.Tensor,
    o14: zml.Tensor,
    // layer 19 : reader full (q proj, q norm, o_proj)
    q19: zml.Tensor,
    qn19: zml.Tensor,
    o19: zml.Tensor,

    pub fn init(base: zml.io.TensorStore.View) Engine {
        const layers = base.withPrefix("model").withPrefix("language_model").withPrefix("layers");
        const l14 = layers.withLayer(14).withPrefix("self_attn");
        const l19 = layers.withLayer(19).withPrefix("self_attn");
        return .{
            .q14 = l14.createTensor("q_proj.weight", .{ .o, .d }, null),
            .qn14 = l14.createTensor("q_norm.weight", .{.hd}, null),
            .k14 = l14.createTensor("k_proj.weight", .{ .o, .d }, null),
            .kn14 = l14.createTensor("k_norm.weight", .{.hd}, null),
            .v14 = l14.createTensor("v_proj.weight", .{ .o, .d }, null),
            .o14 = l14.createTensor("o_proj.weight", .{ .d, .m }, null),
            .q19 = l19.createTensor("q_proj.weight", .{ .o, .d }, null),
            .qn19 = l19.createTensor("q_norm.weight", .{.hd}, null),
            .o19 = l19.createTensor("o_proj.weight", .{ .d, .m }, null),
        };
    }

    pub fn load(self: *const Engine, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, store: *const zml.io.TensorStore, shardings: []const zml.sharding.Sharding) !zml.Bufferized(Engine) {
        return zml.io.load(Engine, self, allocator, io, platform, store, .{ .shardings = shardings, .parallelism = 1, .dma_chunks = 1, .dma_chunk_size = 16 * 1024 * 1024 });
    }

    pub fn forward(self: Engine, rt: Runtime) Out6 {
        const pos_u = rt.pos.squeeze(.s).convert(.u32);
        const cos = rt.cos_full;
        const sin = rt.sin_full;

        // --- Layer 14 (writer full) : k/v du token p ---
        var k = rt.attn_in_14.dot(c(self.k14), .d).reshape(.{ B, S, KVH, HD }).withTags(.{ .b, .s, .nh, .hd });
        k = rmsScaleHd(k, c(self.kn14));
        k = manualRope(k, cos, sin);
        const k_new = k.transpose(.{ .b, .nh, .s, .hd }).rename(.{ .nh = .h, .s = .k });

        var v = rt.attn_in_14.dot(c(self.v14), .d).reshape(.{ B, S, KVH, HD }).withTags(.{ .b, .s, .nh, .hd });
        v = zml.nn.rmsNorm(v, .hd, RMS_EPS); // v_norm SANS scale
        const v_new = v.transpose(.{ .b, .nh, .s, .hd }).rename(.{ .nh = .h, .s = .k });

        const so = zml.Tensor.ScatterOpts{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override };
        const cache_k = rt.cache_k_pref.scatterSlices(.{ .k = pos_u }, k_new, so);
        const cache_v = rt.cache_v_pref.scatterSlices(.{ .k = pos_u }, v_new, so);

        const attn14 = attendCache(rt.attn_in_14, c(self.q14), c(self.qn14), c(self.o14), cos, sin, cache_k, cache_v, rt.mask);
        const attn19 = attendCache(rt.attn_in_19, c(self.q19), c(self.qn19), c(self.o19), cos, sin, cache_k, cache_v, rt.mask);

        return .{ cache_k, cache_v, k_new, v_new, attn14, attn19 };
    }
};

const Runtime = struct {
    attn_in_14: zml.Tensor,
    attn_in_19: zml.Tensor,
    mask: zml.Tensor,
    cache_k_pref: zml.Tensor,
    cache_v_pref: zml.Tensor,
    pos: zml.Tensor,
    cos_full: zml.Tensor, // {b,s=1,hd=512} pos p
    sin_full: zml.Tensor,

    pub fn init(v: zml.io.TensorStore.View) Runtime {
        return .{
            .attn_in_14 = v.createTensor("attn_in_14", .{ .b, .s, .d }, null),
            .attn_in_19 = v.createTensor("attn_in_19", .{ .b, .s, .d }, null),
            .mask = v.createTensor("mask_decode", .{ .b, .h, .q, .k }, null),
            .cache_k_pref = v.createTensor("cache14_k_prefill", .{ .b, .h, .k, .hd }, null),
            .cache_v_pref = v.createTensor("cache14_v_prefill", .{ .b, .h, .k, .hd }, null),
            .pos = v.createTensor("pos_idx", .{.s}, null),
            .cos_full = v.createTensor("cos_full", .{ .b, .s, .hd }, null),
            .sin_full = v.createTensor("sin_full", .{ .b, .s, .hd }, null),
        };
    }

    pub fn load(self: *const Runtime, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, store: *const zml.io.TensorStore, shardings: []const zml.sharding.Sharding) !zml.Bufferized(Runtime) {
        return zml.io.load(Runtime, self, allocator, io, platform, store, .{ .shardings = shardings, .parallelism = 1, .dma_chunks = 1, .dma_chunk_size = 16 * 1024 * 1024 });
    }
};

const Oracle = struct {
    cache_k_after: zml.Tensor,
    cache_v_after: zml.Tensor,
    k_new: zml.Tensor,
    v_new: zml.Tensor,
    out14: zml.Tensor,
    out19: zml.Tensor,

    pub fn init(v: zml.io.TensorStore.View) Oracle {
        return .{
            .cache_k_after = v.createTensor("cache14_k_after", .{ .b, .h, .k, .hd }, null),
            .cache_v_after = v.createTensor("cache14_v_after", .{ .b, .h, .k, .hd }, null),
            .k_new = v.createTensor("k14_new", .{ .b, .h, .k, .hd }, null),
            .v_new = v.createTensor("v14_new", .{ .b, .h, .k, .hd }, null),
            .out14 = v.createTensor("attn_out_14", .{ .b, .s, .d }, null),
            .out19 = v.createTensor("attn_out_19", .{ .b, .s, .d }, null),
        };
    }

    pub fn load(self: *const Oracle, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, store: *const zml.io.TensorStore, shardings: []const zml.sharding.Sharding) !zml.Bufferized(Oracle) {
        return zml.io.load(Oracle, self, allocator, io, platform, store, .{ .shardings = shardings, .parallelism = 1, .dma_chunks = 1, .dma_chunk_size = 16 * 1024 * 1024 });
    }
};

fn cmp(allocator: std.mem.Allocator, io: std.Io, label: []const u8, out_buf: *zml.Buffer, ref_buf: *zml.Buffer) !bool {
    var out_s = try out_buf.toSliceAlloc(allocator, io);
    defer out_s.free(allocator);
    var ref_s = try ref_buf.toSliceAlloc(allocator, io);
    defer ref_s.free(allocator);
    const out = out_s.items(f32);
    const ref = ref_s.items(f32);
    if (out.len != ref.len) {
        log.err("  {s}: length mismatch out={d} ref={d}", .{ label, out.len, ref.len });
        return error.LengthMismatch;
    }
    var max_abs: f32 = 0.0;
    var sum_abs: f64 = 0.0;
    var out_max: f32 = 0.0;
    var ref_max: f32 = 0.0;
    var nan_inf = false;
    for (out, ref) |a, bb| {
        if (std.math.isNan(a) or std.math.isInf(a)) nan_inf = true;
        const diff = @abs(a - bb);
        if (diff > max_abs) max_abs = diff;
        sum_abs += @as(f64, diff);
        if (@abs(a) > out_max) out_max = @abs(a);
        if (@abs(bb) > ref_max) ref_max = @abs(bb);
    }
    const mean_abs = @as(f32, @floatCast(sum_abs / @as(f64, @floatFromInt(out.len))));
    const pass = !nan_inf and max_abs <= 1.0e-2 and mean_abs <= 1.0e-4;
    const verdict = if (nan_inf) "FAIL(NaN/Inf)" else if (pass) "PASS" else if (max_abs <= 1.0e-1) "WARN" else "FAIL";
    log.info("  {s:<22} n={d:>5} max_abs={e:.4} mean_abs={e:.4} | max|zml|={e:.3} max|ref|={e:.3} -> {s}", .{ label, out.len, max_abs, mean_abs, out_max, ref_max, verdict });
    return pass;
}

pub fn main(init: std.process.Init) !void {
    const arena = init.arena;
    const allocator = init.gpa;
    const io = init.io;

    const process_args = try init.minimal.args.toSlice(arena.allocator());
    if (process_args.len < 3) {
        log.err("Usage: gemma4_decode2 <model.safetensors> <p5_7_7_decode2.safetensors>", .{});
        return error.MissingArgument;
    }
    const ckpt = process_args[1];
    const fixture = process_args[2];
    log.info("P5.7.7 decode-2 — pilote writer 14 × reader 19 (full, head_dim 512)", .{});

    var reg_ck: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, ckpt);
    var store_ck: zml.io.TensorStore = .fromRegistry(allocator, &reg_ck);
    var reg_fx: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, fixture);
    defer reg_fx.deinit();
    var store_fx: zml.io.TensorStore = .fromRegistry(allocator, &reg_fx);
    defer store_fx.deinit();

    const engine: Engine = .init(store_ck.view());
    const rt: Runtime = .init(store_fx.view());
    const oracle: Oracle = .init(store_fx.view());

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);
    const sharding = try zml.sharding.replicatedSharding(platform);

    log.info("Materializing weights (layer 14 + 19 self_attn full)...", .{});
    const eng_buf = try engine.load(arena.allocator(), io, platform, &store_ck, &.{sharding});
    const rt_buf = try rt.load(arena.allocator(), io, platform, &store_fx, &.{sharding});
    var orc_buf = try oracle.load(arena.allocator(), io, platform, &store_fx, &.{sharding});
    store_ck.deinit();
    reg_ck.deinit();

    log.info("Compiling decode-2 forward...", .{});
    var exe = try platform.compile(allocator, io, engine, .forward, .{rt}, .{ .shardings = &.{sharding} });
    defer exe.deinit();
    var args = try exe.args(allocator);
    defer args.deinit(allocator);
    var results = try exe.results(allocator);
    defer results.deinit(allocator);
    args.set(.{ eng_buf, rt_buf });
    exe.call(args, &results);

    var r0, var r1, var r2, var r3, var r4, var r5 = results.get(struct {
        zml.Buffer, zml.Buffer, zml.Buffer, zml.Buffer, zml.Buffer, zml.Buffer,
    });
    defer {
        r0.deinit();
        r1.deinit();
        r2.deinit();
        r3.deinit();
        r4.deinit();
        r5.deinit();
    }

    log.info("Comparaison vs oracle (PASS: max_abs<=1e-2 ET mean_abs<=1e-4) :", .{});
    var all_pass = true;
    all_pass = (try cmp(allocator, io, "cache14_k_after", &r0, &orc_buf.cache_k_after)) and all_pass;
    all_pass = (try cmp(allocator, io, "cache14_v_after", &r1, &orc_buf.cache_v_after)) and all_pass;
    all_pass = (try cmp(allocator, io, "k14_new (token p)", &r2, &orc_buf.k_new)) and all_pass;
    all_pass = (try cmp(allocator, io, "v14_new (token p)", &r3, &orc_buf.v_new)) and all_pass;
    all_pass = (try cmp(allocator, io, "attn_out_14 (writer)", &r4, &orc_buf.out14)) and all_pass;
    all_pass = (try cmp(allocator, io, "attn_out_19 (reader)", &r5, &orc_buf.out19)) and all_pass;

    if (all_pass) {
        log.info("P5.7.7 decode-2 PASS — full head_dim 512 + manualRope(pos) + scatterSlices + reuse reader full", .{});
    } else {
        log.err("P5.7.7 decode-2 : divergence (cf 1er tap non-PASS)", .{});
        return error.DecodeMismatch;
    }
}
