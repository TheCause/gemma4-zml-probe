// P5.7.7 decode-1 — PILOTE decode incrémental ZML (writer 13 sliding × reader 15 sliding).
//
// Isole la mécanique decode neuve sur la plus petite unité : un cache KV qui grandit d'une colonne,
// l'append du token p via scatterSlices, le pos_idx absolu (RoPE à p=4), le mask S=1, et le reuse YOCO
// du cache grandi par un reader. On compare la sortie du module self_attn (post o_proj) vs l'oracle HF.
// Le bloc d'attention est calqué sur gemma4_prefill.zig:runLayer (GQA splitAxis, scores, softmax,
// context, o_proj) — la SEULE nouveauté est scatterSlices + pos_idx (idiome llama/qwen KvCache).
//
// CLI : gemma4_decode1 <model.safetensors> <p5_7_7_decode1.safetensors>

const std = @import("std");
const log = std.log;
const zml = @import("zml");

pub const std_options: std.Options = .{ .log_level = .info };

const B: i64 = 1;
const S: i64 = 1; // decode : un seul nouveau token
const D: i64 = 1536;
const NH: i64 = 8;
const KVH: i64 = 1;
const HD: i64 = 256; // sliding head_dim
const KMAX: i64 = 5; // S_prompt(4) + 1
const M: i64 = NH * HD; // 2048

const RMS_EPS: f32 = 1.0e-6;
const ROPE_THETA_SLIDING: f32 = 1.0e4;

inline fn c(t: zml.Tensor) zml.Tensor {
    return t.convert(.f32);
}
fn rmsScaleHd(x: zml.Tensor, w: zml.Tensor) zml.Tensor {
    const n = zml.nn.rmsNorm(x, .hd, RMS_EPS);
    return n.mul(w.broad(n.shape()));
}
fn slidingRope(x: zml.Tensor, pos: zml.Tensor) zml.Tensor {
    return zml.nn.rope(x, pos, .{ .layout = .sequential, .scaling = .{ .default = .{ .rope_theta = ROPE_THETA_SLIDING } } });
}

// Attention d'un nouveau token (q=1) sur un cache grandi {b,h=1,k,hd}. q calculé depuis q_src {b,s=1,d}.
// Identique à gemma4_prefill.zig:runLayer (lignes 185-237) restreint à S=1, cache fourni.
fn attendCache(q_src: zml.Tensor, q_proj: zml.Tensor, q_norm: zml.Tensor, o_proj: zml.Tensor, cache_k: zml.Tensor, cache_v: zml.Tensor, mask: zml.Tensor, pos: zml.Tensor) zml.Tensor {
    var q = q_src.dot(q_proj, .d).reshape(.{ B, S, NH, HD }).withTags(.{ .b, .s, .nh, .hd });
    q = rmsScaleHd(q, q_norm);
    q = slidingRope(q, pos);
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

// ---- Poids (lus du checkpoint complet, prefix model.language_model.layers.N.self_attn) + forward ----
const Engine = struct {
    // layer 13 : writer/producer sliding (q/k/v proj, q/k norm, o_proj)
    q13: zml.Tensor,
    qn13: zml.Tensor,
    k13: zml.Tensor,
    kn13: zml.Tensor,
    v13: zml.Tensor,
    o13: zml.Tensor,
    // layer 15 : reader sliding (q proj, q norm, o_proj — un reader n'a PAS de modules k/v, cf L1196)
    q15: zml.Tensor,
    qn15: zml.Tensor,
    o15: zml.Tensor,

    pub fn init(base: zml.io.TensorStore.View) Engine {
        const layers = base.withPrefix("model").withPrefix("language_model").withPrefix("layers");
        const l13 = layers.withLayer(13).withPrefix("self_attn");
        const l15 = layers.withLayer(15).withPrefix("self_attn");
        return .{
            .q13 = l13.createTensor("q_proj.weight", .{ .o, .d }, null),
            .qn13 = l13.createTensor("q_norm.weight", .{.hd}, null),
            .k13 = l13.createTensor("k_proj.weight", .{ .o, .d }, null),
            .kn13 = l13.createTensor("k_norm.weight", .{.hd}, null),
            .v13 = l13.createTensor("v_proj.weight", .{ .o, .d }, null),
            .o13 = l13.createTensor("o_proj.weight", .{ .d, .m }, null),
            .q15 = l15.createTensor("q_proj.weight", .{ .o, .d }, null),
            .qn15 = l15.createTensor("q_norm.weight", .{.hd}, null),
            .o15 = l15.createTensor("o_proj.weight", .{ .d, .m }, null),
        };
    }

    pub fn load(self: *const Engine, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, store: *const zml.io.TensorStore, shardings: []const zml.sharding.Sharding) !zml.Bufferized(Engine) {
        return zml.io.load(Engine, self, allocator, io, platform, store, .{ .shardings = shardings, .parallelism = 1, .dma_chunks = 1, .dma_chunk_size = 16 * 1024 * 1024 });
    }

    /// Forward decode-1 : layer 13 calcule k/v du token p, scatterSlices append à .k=p -> cache grandi ;
    /// attention writer (13) + attention reader (15) sur ce cache grandi. Retour :
    /// {cache_k_grown, cache_v_grown, k_new, v_new, attn13, attn15}.
    pub fn forward(self: Engine, rt: Runtime) Out6 {
        const pos = rt.pos;
        const pos_u = pos.squeeze(.s).convert(.u32); // index scalaire pour scatterSlices

        // --- Layer 13 (writer) : k/v du token p ---
        var k = rt.attn_in_13.dot(c(self.k13), .d).reshape(.{ B, S, KVH, HD }).withTags(.{ .b, .s, .nh, .hd });
        k = rmsScaleHd(k, c(self.kn13));
        k = slidingRope(k, pos);
        const k_new = k.transpose(.{ .b, .nh, .s, .hd }).rename(.{ .nh = .h, .s = .k }); // {b,h=1,k=1,hd}

        var v = rt.attn_in_13.dot(c(self.v13), .d).reshape(.{ B, S, KVH, HD }).withTags(.{ .b, .s, .nh, .hd });
        v = zml.nn.rmsNorm(v, .hd, RMS_EPS); // v_norm SANS scale
        const v_new = v.transpose(.{ .b, .nh, .s, .hd }).rename(.{ .nh = .h, .s = .k });

        // --- Append scatterSlices à la colonne p (la nouveauté du decode) ---
        const so = zml.Tensor.ScatterOpts{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override };
        const cache_k = rt.cache_k_pref.scatterSlices(.{ .k = pos_u }, k_new, so);
        const cache_v = rt.cache_v_pref.scatterSlices(.{ .k = pos_u }, v_new, so);

        // --- Attention writer (13) sur le cache grandi ---
        const attn13 = attendCache(rt.attn_in_13, c(self.q13), c(self.qn13), c(self.o13), cache_k, cache_v, rt.mask, pos);
        // --- Attention reader (15) : reuse YOCO du même cache grandi ---
        const attn15 = attendCache(rt.attn_in_15, c(self.q15), c(self.qn15), c(self.o15), cache_k, cache_v, rt.mask, pos);

        return .{ cache_k, cache_v, k_new, v_new, attn13, attn15 };
    }
};

// ---- Inputs runtime (fixture) ----
const Runtime = struct {
    attn_in_13: zml.Tensor, // {b,s=1,d}
    attn_in_15: zml.Tensor,
    mask: zml.Tensor, // {b,h,q=1,k=5}
    cache_k_pref: zml.Tensor, // {b,h=1,k=5,hd} (cols 0..3 réelles, col 4 = 0)
    cache_v_pref: zml.Tensor,
    pos: zml.Tensor, // {s=1} i32 = [4]

    pub fn init(v: zml.io.TensorStore.View) Runtime {
        return .{
            .attn_in_13 = v.createTensor("attn_in_13", .{ .b, .s, .d }, null),
            .attn_in_15 = v.createTensor("attn_in_15", .{ .b, .s, .d }, null),
            .mask = v.createTensor("mask_decode", .{ .b, .h, .q, .k }, null),
            .cache_k_pref = v.createTensor("cache13_k_prefill", .{ .b, .h, .k, .hd }, null),
            .cache_v_pref = v.createTensor("cache13_v_prefill", .{ .b, .h, .k, .hd }, null),
            .pos = v.createTensor("pos_idx", .{.s}, null),
        };
    }

    pub fn load(self: *const Runtime, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, store: *const zml.io.TensorStore, shardings: []const zml.sharding.Sharding) !zml.Bufferized(Runtime) {
        return zml.io.load(Runtime, self, allocator, io, platform, store, .{ .shardings = shardings, .parallelism = 1, .dma_chunks = 1, .dma_chunk_size = 16 * 1024 * 1024 });
    }
};

// ---- Références oracle (fixture) ----
const Oracle = struct {
    cache_k_after: zml.Tensor,
    cache_v_after: zml.Tensor,
    k_new: zml.Tensor,
    v_new: zml.Tensor,
    out13: zml.Tensor,
    out15: zml.Tensor,

    pub fn init(v: zml.io.TensorStore.View) Oracle {
        return .{
            .cache_k_after = v.createTensor("cache13_k_after", .{ .b, .h, .k, .hd }, null),
            .cache_v_after = v.createTensor("cache13_v_after", .{ .b, .h, .k, .hd }, null),
            .k_new = v.createTensor("k13_new", .{ .b, .h, .k, .hd }, null),
            .v_new = v.createTensor("v13_new", .{ .b, .h, .k, .hd }, null),
            .out13 = v.createTensor("attn_out_13", .{ .b, .s, .d }, null),
            .out15 = v.createTensor("attn_out_15", .{ .b, .s, .d }, null),
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
        log.err("Usage: gemma4_decode1 <model.safetensors> <p5_7_7_decode1.safetensors>", .{});
        return error.MissingArgument;
    }
    const ckpt = process_args[1];
    const fixture = process_args[2];
    log.info("P5.7.7 decode-1 — pilote writer 13 × reader 15 (sliding)", .{});

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

    log.info("Materializing weights (layer 13 + 15 self_attn)...", .{});
    const eng_buf = try engine.load(arena.allocator(), io, platform, &store_ck, &.{sharding});
    const rt_buf = try rt.load(arena.allocator(), io, platform, &store_fx, &.{sharding});
    var orc_buf = try oracle.load(arena.allocator(), io, platform, &store_fx, &.{sharding});
    store_ck.deinit();
    reg_ck.deinit();

    log.info("Compiling decode-1 forward...", .{});
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
    all_pass = (try cmp(allocator, io, "cache13_k_after", &r0, &orc_buf.cache_k_after)) and all_pass;
    all_pass = (try cmp(allocator, io, "cache13_v_after", &r1, &orc_buf.cache_v_after)) and all_pass;
    all_pass = (try cmp(allocator, io, "k13_new (token p)", &r2, &orc_buf.k_new)) and all_pass;
    all_pass = (try cmp(allocator, io, "v13_new (token p)", &r3, &orc_buf.v_new)) and all_pass;
    all_pass = (try cmp(allocator, io, "attn_out_13 (writer)", &r4, &orc_buf.out13)) and all_pass;
    all_pass = (try cmp(allocator, io, "attn_out_15 (reader)", &r5, &orc_buf.out15)) and all_pass;

    if (all_pass) {
        log.info("P5.7.7 decode-1 PASS — scatterSlices append + pos_idx + mask S=1 + reuse reader YOCO validés", .{});
    } else {
        log.err("P5.7.7 decode-1 : divergence (cf 1er tap non-PASS)", .{});
        return error.DecodeMismatch;
    }
}
