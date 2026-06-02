// P5.7.6 — ZML logits : lm_head (tied = embed_tokens, vocab complet 262144) + softcap 30, sur le
// last_hidden validé (P5.7.5). Test e2e décisif : top-1 (argmax = token prédit) + max_abs vs oracle.
//
// logits = softcap(last_hidden @ embed_tokens.weightᵀ), softcap(x)=30·tanh(x/30). lm_head = poids BRUT
// embed_tokens (sans embed_scale en sortie). embed_tokens depuis le checkpoint complet ; last_hidden +
// logits oracle depuis la fixture (arg de forward).
//
// CLI : gemma4_logits <model.safetensors> <p5_7_5_hybrid.safetensors>

const std = @import("std");
const log = std.log;
const zml = @import("zml");
const stdx = zml.stdx;

pub const std_options: std.Options = .{ .log_level = .info };

const B: i64 = 1;
const S: i64 = 4;
const D: i64 = 1536;
const VOCAB: usize = 262144;
const SOFTCAP: f64 = 30.0;

const ModelEmbed = struct {
    embed_tokens: zml.Tensor, // {.voc,.d} bf16 (lm_head tied)

    pub fn init(base: zml.io.TensorStore.View) ModelEmbed {
        return .{ .embed_tokens = base.createTensor("embed_tokens.weight", .{ .voc, .d }, null) };
    }
    pub fn load(self: *const ModelEmbed, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, store: *const zml.io.TensorStore, shardings: []const zml.sharding.Sharding) !zml.Bufferized(ModelEmbed) {
        return zml.io.load(ModelEmbed, self, allocator, io, platform, store, .{ .shardings = shardings, .parallelism = 1, .dma_chunks = 1, .dma_chunk_size = 16 * 1024 * 1024 });
    }

    /// logits = softcap(last_hidden @ embed_tokensᵀ). last_hidden fp32 (fixture).
    pub fn forward(self: ModelEmbed, last_hidden: zml.Tensor) zml.Tensor {
        const lm = self.embed_tokens.convert(.f32);
        const raw = last_hidden.dot(lm, .d); // {.b,.s,.voc}
        return raw.scale(1.0 / SOFTCAP).tanh().scale(SOFTCAP);
    }
};

const Fix = struct {
    last_hidden: zml.Tensor, // {.b,.s,.d}
    logits: zml.Tensor, // {.b,.s,.voc} oracle

    pub fn init(v: zml.io.TensorStore.View) Fix {
        return .{
            .last_hidden = v.createTensor("last_hidden", .{ .b, .s, .d }, null),
            .logits = v.createTensor("logits", .{ .b, .s, .voc }, null),
        };
    }
    pub fn load(self: *const Fix, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, store: *const zml.io.TensorStore, shardings: []const zml.sharding.Sharding) !zml.Bufferized(Fix) {
        return zml.io.load(Fix, self, allocator, io, platform, store, .{ .shardings = shardings, .parallelism = 1, .dma_chunks = 1, .dma_chunk_size = 16 * 1024 * 1024 });
    }
};

fn argmaxRow(row: []const f32) usize {
    var best: usize = 0;
    var best_v: f32 = row[0];
    for (row, 0..) |v, i| {
        if (v > best_v) {
            best_v = v;
            best = i;
        }
    }
    return best;
}

pub fn main(init: std.process.Init) !void {
    const arena = init.arena;
    const allocator = init.gpa;
    const io = init.io;

    const process_args = try init.minimal.args.toSlice(arena.allocator());
    if (process_args.len < 3) {
        log.err("Usage: gemma4_logits <model.safetensors> <p5_7_5_hybrid.safetensors>", .{});
        return error.MissingArgument;
    }
    const ckpt = process_args[1];
    const fixture = process_args[2];

    log.info("P5.7.6 — ZML logits (lm_head tied vocab {d} + softcap {d})", .{ VOCAB, @as(i64, @intFromFloat(SOFTCAP)) });

    var reg_ck: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, ckpt);
    var store_ck: zml.io.TensorStore = .fromRegistry(allocator, &reg_ck);
    var reg_fx: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, fixture);
    defer reg_fx.deinit();
    var store_fx: zml.io.TensorStore = .fromRegistry(allocator, &reg_fx);
    defer store_fx.deinit();

    const base = store_ck.view().withPrefix("model").withPrefix("language_model");
    const model: ModelEmbed = .init(base);
    const fix: Fix = .init(store_fx.view());

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);
    const sharding = try zml.sharding.replicatedSharding(platform);

    log.info("Loading embed_tokens (lm_head) + fixture...", .{});
    const m_buf = try model.load(arena.allocator(), io, platform, &store_ck, &.{sharding});
    var fx_buf = try fix.load(arena.allocator(), io, platform, &store_fx, &.{sharding});
    store_ck.deinit();
    reg_ck.deinit();

    log.info("Compiling lm_head + softcap...", .{});
    var exe = try platform.compile(allocator, io, model, .forward, .{fix.last_hidden}, .{ .shardings = &.{sharding} });
    defer exe.deinit();
    var args = try exe.args(allocator);
    defer args.deinit(allocator);
    var results = try exe.results(allocator);
    defer results.deinit(allocator);
    args.set(.{ m_buf, fx_buf.last_hidden });
    exe.call(args, &results);
    var logits_buf: zml.Buffer = results.get(zml.Buffer);
    defer logits_buf.deinit();

    var out_s = try logits_buf.toSliceAlloc(allocator, io);
    defer out_s.free(allocator);
    var ref_s = try fx_buf.logits.toSliceAlloc(allocator, io);
    defer ref_s.free(allocator);
    const out = out_s.items(f32);
    const ref = ref_s.items(f32);
    const expect: usize = @intCast(B * S * @as(i64, VOCAB));
    if (out.len != expect or ref.len != expect) {
        log.err("length mismatch out={d} ref={d} expected={d}", .{ out.len, ref.len, expect });
        return error.LengthMismatch;
    }

    var max_abs: f32 = 0.0;
    var nan_inf = false;
    for (out, ref) |a, b| {
        if (std.math.isNan(a) or std.math.isInf(a)) nan_inf = true;
        const diff = @abs(a - b);
        if (diff > max_abs) max_abs = diff;
    }

    // Top-1 (argmax) par position = token prédit. flip-rate temp=0 = désaccords d'argmax.
    var flips: usize = 0;
    log.info("Argmax (token prédit) par position s :", .{});
    for (0..@intCast(S)) |s| {
        const o_row = out[s * VOCAB .. (s + 1) * VOCAB];
        const r_row = ref[s * VOCAB .. (s + 1) * VOCAB];
        const o_tok = argmaxRow(o_row);
        const r_tok = argmaxRow(r_row);
        const ok = o_tok == r_tok;
        if (!ok) flips += 1;
        log.info("  s={d}: zml_argmax={d} (logit {d:.5}) | oracle_argmax={d} (logit {d:.5}) -> {s}", .{ s, o_tok, o_row[o_tok], r_tok, r_row[r_tok], if (ok) "MATCH" else "FLIP" });
    }

    log.info("logits max_abs vs oracle = {e:.4} | NaN/Inf={} | flips (temp=0) = {d}/{d}", .{ max_abs, nan_inf, flips, @as(usize, @intCast(S)) });
    if (nan_inf or flips != 0 or max_abs > 1.0e-1) {
        log.err("P5.7.6 FAIL (flips={d}, max_abs={e:.3})", .{ flips, max_abs });
        return error.LogitsMismatch;
    }
    log.info("P5.7.6 PASS : logits ZML == oracle (top-1 identique sur {d}/{d} positions, max_abs {e:.3})", .{ @as(usize, @intCast(S)), @as(usize, @intCast(S)), max_abs });
}
