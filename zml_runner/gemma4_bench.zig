// gemma4_bench — Benchmark decode GPU (P-GPU-1/G8, cf docs/GPU_PORT_PLAN.md §9.7).
//
// Mesure le débit decode batch-1 (tok/s) du moteur ZML sur le backend sélectionné (CUDA si dispo).
// Réutilise la fixture L1a (gen_long.safetensors) : les cos/sin/masques/positions/expected sont
// position-only (indépendants du backend) → le bench est reproductible et compare CPU vs GPU à
// calcul identique. Warmup (1 step, cold-start CUDA/cuBLAS écarté) puis timing sur N steps.
//
// Métriques : compile (ms), warmup, tok/s, ms/tok, pic RSS host, platform. VRAM GPU via `nvidia-smi`
// (le moteur n'expose pas un compteur VRAM portable ; TODO G8 : lire via PJRT device memory).
// Sanity : argmax == HF (si divergence, le bench mesure du bruit → alerte).
//
// CLI : gemma4_bench <model.safetensors> <gen_long.safetensors> [max_steps]
// Ex : bazel run //examples/rqz:gemma4_bench -- <ckpt> gen_long.safetensors 256

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
        log.err("Usage: gemma4_bench <model.safetensors> <gen_long.safetensors> [max_steps]", .{});
        return error.MissingArgument;
    }
    const ckpt = process_args[1];
    const fixture = process_args[2];
    const max_steps: ?usize = if (process_args.len >= 4) std.fmt.parseInt(usize, process_args[3], 10) catch null else null;

    const platform: *zml.Platform = blk: {
        const cuda_opts: zml.platform.CreateOptions = .{
            .cuda = .{ .allocator = .{ .bfc = .{ .preallocate = true, .memory_fraction = 0.90 } } },
        };
        if (zml.Platform.init(allocator, io, .cuda, cuda_opts)) |p| break :blk p else |_| {}
        log.warn("CUDA indisponible — repli sur Platform.auto.", .{});
        break :blk try zml.Platform.auto(allocator, io, .{});
    };
    defer platform.deinit(allocator);
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

    const eng_buf = try model.load(arena.allocator(), io, platform, &store_ck, &.{sharding});
    const pk_buf = try packed_in.load(arena.allocator(), io, platform, &store_fx, &.{sharding});
    var cache_buf = try cache0.load(arena.allocator(), io, platform, &store_fx, &.{sharding});
    const exp_sym: ExpW = .init(store_fx.view());
    var exp_buf = try exp_sym.load(arena.allocator(), io, platform, &store_fx, &.{sharding});
    store_ck.deinit();
    reg_ck.deinit();

    var exp_slice = try exp_buf.e.toSliceAlloc(allocator, io);
    defer exp_slice.free(allocator);
    const expected_tokens = exp_slice.items(i32);
    const total = expected_tokens.len;
    const num_steps = if (max_steps) |m| @min(m, total) else total;
    if (num_steps < 2) { log.err("num_steps < 2 : pas assez pour warmup+mesure", .{}); return error.MissingArgument; }

    log.info("BENCH — backend={s}, fp32, batch-1, mono-graphe ; warmup=1 + measure={d} steps", .{ @tagName(platform.target), num_steps - 1 });
    const t_compile: std.Io.Timestamp = .now(io, .awake);
    var exe = try platform.compile(allocator, io, model, .forward, .{ packed_in, cache0, ctrl_sym }, .{ .shardings = &.{sharding} });
    defer exe.deinit();
    log.info("  compile: {f}", .{t_compile.untilNow(io, .awake)});
    mem_probe.logMem(io, "post-compile");

    // run 1 step (idx 0) en warmup : cold-start cuBLAS/autotune écarté ; son cache est jeté (on recommence).
    var all_pass = true;
    var n_match: usize = 0;
    var step_idx: usize = 0;

    // helper inline pour 1 step (retourne le token, met à jour cache_buf).
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
        if (tok == exp) n_match += 1 else all_pass = false;
        cache_buf.sl_k.deinit();
        cache_buf.sl_v.deinit();
        cache_buf.fl_k.deinit();
        cache_buf.fl_v.deinit();
        cache_buf = zml.Bufferized(engine.Cache){ .sl_k = r_slk, .sl_v = r_slv, .fl_k = r_flk, .fl_v = r_flv };
        r_logits.deinit();
        step_buf.deinit();
        args.deinit(allocator);
        results.deinit(allocator);
        if (step_idx == 0) {
            // warmup done ; reset du timer pour la mesure (le cache accumule normalement).
            // IMPORTANT : avancer step_idx AVANT le break (le `: (step_idx += 1)` de la while n'est pas
            // exécuté sur break). Sinon la boucle de mesure ré-exécute l'étape 0 sur un cache déjà avancé
            // → étape 0 double-comptée (tok/s sous-estimé, n_match/num_steps > 100%).
            step_idx += 1;
            break;
        }
    }
    const measure_steps = num_steps - 1;
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
        if (tok == exp) n_match += 1 else all_pass = false;
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
    const tok_per_s = if (elapsed_s > 0) @as(f64, @floatFromInt(measure_steps)) / elapsed_s else 0;
    const ms_per_tok = if (measure_steps > 0) @as(f64, @floatFromInt(elapsed_ns)) / 1e6 / @as(f64, @floatFromInt(measure_steps)) else 0;
    mem_probe.logMem(io, "post-run (host RSS ; VRAM GPU via nvidia-smi)");

    log.info("========== gemma4_bench ==========", .{});
    log.info("  backend      : {s}", .{@tagName(platform.target)});
    log.info("  precision    : fp32 (G1 baseline ; bf16 = G2/G3)", .{});
    log.info("  config       : batch-1, mono-graphe 35 couches, L_MAX={d}", .{L_MAX});
    log.info("  warmup       : 1 step (écarté)", .{});
    log.info("  measured     : {d} steps", .{measure_steps});
    log.info("  throughput   : {d:.1} tok/s  ({d:.2} ms/tok)", .{ tok_per_s, ms_per_tok });
    log.info("  total decode : {d:.2}s", .{elapsed_s});
    log.info("  sanity       : argmax==HF {d}/{d} ({s})", .{ n_match, num_steps, if (all_pass) "OK" else "DIVERGENCE — bench mesure du bruit, investiguer drift" });
    log.info("  VRAM (GPU)   : lancer `nvidia-smi` (pic) ; TODO G8 : read PJRT device memory.", .{});
    log.info("===================================", .{});
    if (!all_pass) log.warn("sanity KO : le bench mesure des tokens divergents — valider le drift fp32-{s} avant de trust les tok/s.", .{@tagName(platform.target)});
}
