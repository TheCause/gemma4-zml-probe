// G1 — Baseline fp32 sur GPU (P-GPU-1, cf docs/GPU_PORT_PLAN.md §10).
//
// Le moteur `engine.zig` est device-agnostic : le MÊME graphe XLA tourne sur CPU ou GPU. Ce runner force
// le backend CUDA (avec fallback auto) et AJOUTE un timer tok/s + logging platform/RSS. Le calcul est
// STRICTEMENT identique à `gemma4_gen_long.zig` (L1a mono) : `EngineModel(struct{}, .{...})` sans toucher
// à `self.prec` (PrecRt par défaut = tout-null = fp32). → G1 = "le moteur L1a, mais sur GPU", pour mesurer le gain brut du
// backend natif et valider que l'argmax == HF tient en fp32-CUDA (drift Eigen→CUDA caractérisé au G1).
//
// Critère G1 : argmax == HF sur les N tokens (séquence == L1a CPU == HF greedy) ; drift logits vs
// baseline CPU-L1a à reporter. Perf : tok/s (decode batch-1). Le chunking n'est PAS utilisé (le mur
// mémoire CPU ~33 Go disparaît sur GPU, cf GPU_PORT_PLAN §6) → mono-graphe direct.
//
// CLI : gemma4_gen_long_gpu <model.safetensors> <gen_long.safetensors> [max_steps] [--no-prealloc]
// Prérequis : libpjrt_cuda linké (cf GPU_PORT_PLAN §12) ; `nvidia-smi` pour la VRAM.
//
// --no-prealloc (G2.1) : coupe la préallocation BFC (preallocate=false) pour que nvidia-smi mesure la
// VRAM RÉELLEMENT utilisée. Avec preallocate=true (défaut, perf), BFC réserve memory_fraction×24 Go
// d'emblée → le ~22 Go observé au G1 était la RÉSERVE, pas l'usage (les poids sont chargés au dtype du
// checkpoint = bf16, cf createTensor/io.zig : dtype du header safetensors, jamais upcasté au load).

const std = @import("std");
const log = std.log;
const zml = @import("zml");
const engine = @import("engine.zig");
const mem_probe = @import("mem_probe.zig");

pub const std_options: std.Options = .{ .log_level = .info };

const L_MAX: i64 = 1024;
const B: i64 = 1;
const S: i64 = 1;
const D: i64 = 1536;
// G1 : fp32 (PrecRt défaut tout-null, `self.prec` non touché) — on n'active PAS le bf16 (c'est G2).
const Model = engine.EngineModel(struct {}, .{ .two_masks = true, .kmax_sliding = L_MAX, .kmax_full = L_MAX });
const PackedLong = engine.Packed(true);

const ExpW = struct {
    e: zml.Tensor,
    pub fn init(v: zml.io.TensorStore.View) ExpW {
        return .{ .e = v.createTensor("expected", .{.step}, null) };
    }
    pub fn load(self: *const ExpW, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, store: *const zml.io.TensorStore, shardings: []const zml.sharding.Sharding) !zml.Bufferized(ExpW) {
        return zml.io.load(ExpW, self, allocator, io, platform, store, .{ .shardings = shardings, .parallelism = 1, .dma_chunks = 1, .dma_chunk_size = 16 * 1024 * 1024 });
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
        log.err("Usage: gemma4_gen_long_gpu <model.safetensors> <gen_long.safetensors> [max_steps]", .{});
        return error.MissingArgument;
    }
    const ckpt = process_args[1];
    const fixture = process_args[2];
    const max_steps: ?usize = if (process_args.len >= 4) std.fmt.parseInt(usize, process_args[3], 10) catch null else null;
    var no_prealloc = false;
    for (process_args[3..]) |a| {
        if (std.mem.eql(u8, a, "--no-prealloc")) no_prealloc = true;
    }

    // === Backend : force CUDA (memory_fraction 0.90), fallback auto (CPU) si CUDA indisponible. ===
    const platform: *zml.Platform = blk: {
        const cuda_opts: zml.platform.CreateOptions = .{
            .cuda = .{ .allocator = .{ .bfc = .{ .preallocate = !no_prealloc, .memory_fraction = 0.90 } } },
        };
        if (zml.Platform.init(allocator, io, .cuda, cuda_opts)) |p| break :blk p else |_| {}
        log.warn("CUDA indisponible (libpjrt_cuda absent ?) — repli sur Platform.auto (probablement CPU).", .{});
        break :blk try zml.Platform.auto(allocator, io, .{});
    };
    defer platform.deinit(allocator);
    log.info("G1 — backend = {s} (cible : cuda). Prérequis : libpjrt_cuda linké ; VRAM via nvidia-smi.", .{@tagName(platform.target)});
    if (no_prealloc) log.info("G2.1 — preallocate=false : nvidia-smi mesure l'usage VRAM réel (pas la réserve BFC).", .{});
    const sharding = try zml.sharding.replicatedSharding(platform);

    var reg_ck: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, ckpt);
    var store_ck: zml.io.TensorStore = .fromRegistry(allocator, &reg_ck);
    var reg_fx: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, fixture);
    defer reg_fx.deinit();
    var store_fx: zml.io.TensorStore = .fromRegistry(allocator, &reg_fx);
    defer store_fx.deinit();

    const base = store_ck.view().withPrefix("model").withPrefix("language_model");
    const model: Model = try .init(arena.allocator(), base);
    const packed_in: PackedLong = .init(store_fx.view());
    const cache0: engine.Cache = .init(store_fx.view());
    const ctrl_sym: engine.Ctrl = .initSymbolic();

    log.info("Materializing weights + packed inputs + caches (L_MAX={d}) ...", .{L_MAX});
    const eng_buf = try model.load(arena.allocator(), io, platform, &store_ck, &.{sharding});
    const pk_buf = try packed_in.load(arena.allocator(), io, platform, &store_fx, &.{sharding});
    var cache_buf = try cache0.load(arena.allocator(), io, platform, &store_fx, &.{sharding});
    const exp_sym: ExpW = .init(store_fx.view());
    var exp_buf = try exp_sym.load(arena.allocator(), io, platform, &store_fx, &.{sharding});
    store_ck.deinit();
    reg_ck.deinit();
    mem_probe.logMem(io, "post-load (host RSS ; VRAM via nvidia-smi)");

    var exp_slice = try exp_buf.e.toSliceAlloc(allocator, io);
    defer exp_slice.free(allocator);
    const expected_tokens = exp_slice.items(i32);
    const total = expected_tokens.len;
    const num_steps = if (max_steps) |m| @min(m, total) else total;
    log.info("NUM_STEPS = {d} (max_steps={?d})", .{ num_steps, max_steps });

    log.info("Compiling gen step (mono-graphe 35 couches, fp32) ...", .{});
    const t_compile: std.Io.Timestamp = .now(io, .awake);
    var exe = try platform.compile(allocator, io, model, .forward, .{ packed_in, cache0, ctrl_sym }, .{ .shardings = &.{sharding} });
    defer exe.deinit();
    log.info("  compile: {f}", .{t_compile.untilNow(io, .awake)});
    mem_probe.logMem(io, "post-compile (host RSS ; go/no-go GPU = nvidia-smi)");

    var all_pass = true;
    var n_match: usize = 0;
    var first_fail: i64 = -1;
    var step_idx: usize = 0;
    const t0: std.Io.Timestamp = .now(io, .awake);
    while (step_idx < num_steps) : (step_idx += 1) {
        var step_buf = try zml.Buffer.scalar(io, platform, @as(u32, @intCast(step_idx)), .u32, sharding);
        const ctrl_buf = zml.Bufferized(engine.Ctrl){ .step = step_buf };

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
        if (ok) n_match += 1 else {
            all_pass = false;
            if (first_fail < 0) first_fail = @intCast(step_idx);
            if (step_idx - @as(usize, @intCast(@max(first_fail, 0))) < 8) {
                log.err("  FAIL step {d} (pos {d}) : argmax ZML={d} HF={d}", .{ step_idx, 4 + step_idx, tok, exp });
            }
        }
        if ((step_idx + 1) % 256 == 0) log.info("  ... {d}/{d} steps, {d} match", .{ step_idx + 1, num_steps, n_match });

        cache_buf.sl_k.deinit();
        cache_buf.sl_v.deinit();
        cache_buf.fl_k.deinit();
        cache_buf.fl_v.deinit();
        cache_buf = zml.Bufferized(engine.Cache){ .sl_k = r_slk, .sl_v = r_slv, .fl_k = r_flk, .fl_v = r_flv };
        r_logits.deinit();
        step_buf.deinit();
        args.deinit(allocator);
        results.deinit(allocator);
    }
    const elapsed = t0.untilNow(io, .awake);
    cache_buf.sl_k.deinit();
    cache_buf.sl_v.deinit();
    cache_buf.fl_k.deinit();
    cache_buf.fl_v.deinit();

    const elapsed_ns = elapsed.toNanoseconds();
    const elapsed_s = @as(f64, @floatFromInt(elapsed_ns)) / std.time.ns_per_s;
    const tok_per_s = if (elapsed_s > 0) @as(f64, @floatFromInt(num_steps)) / elapsed_s else 0;
    const ms_per_tok = if (num_steps > 0) @as(f64, @floatFromInt(elapsed_ns)) / 1e6 / @as(f64, @floatFromInt(num_steps)) else 0;
    log.info("G1 PERF : {d} tokens en {d:.2}s → {d:.1} tok/s ({d:.1} ms/tok) [backend={s}, fp32, batch-1, mono-graphe]", .{ num_steps, elapsed_s, tok_per_s, ms_per_tok, @tagName(platform.target) });
    mem_probe.logMem(io, "post-run (host RSS ; VRAM GPU via nvidia-smi)");

    log.info("G1 : {d}/{d} tokens argmax-match (vs HF)", .{ n_match, num_steps });
    if (all_pass) {
        log.info("G1 PASS — fp32-{s} reproduit HF greedy ({d} tokens) ; baseline GPU établi.", .{ @tagName(platform.target), num_steps });
    } else {
        log.err("G1 : divergence (1er fail step {d}, {d} match) — drift fp32-CUDA > tol ? (cf GPU_PORT_PLAN §10.1)", .{ first_fail, n_match });
        return error.GenMismatch;
    }
}
