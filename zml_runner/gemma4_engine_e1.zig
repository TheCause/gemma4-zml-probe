// Gate E1 — non-régression du socle modulaire.
//
// EngineModel(struct{}) (brique vide → point post_v_norm comptime-mort) doit reproduire decode4 (P5.7.8) :
// même boucle de génération, même cache threadé, MÊME fixture (p5_7_8_gen.safetensors), MÊME oracle
// (les 4 tokens `expected` de HF greedy). La branche brick étant comptime-morte, le graphe MLIR est
// identique à decode4 → l'égalité des tokens doit tenir. Une divergence = erreur d'extraction.
//
// CLI : gemma4_engine_e1 <model.safetensors> <p5_7_8_gen.safetensors>

const std = @import("std");
const log = std.log;
const zml = @import("zml");
const engine = @import("engine.zig");

pub const std_options: std.Options = .{ .log_level = .info };

const NUM_STEPS: usize = 4;

const Model = engine.EngineModel(struct {});

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
        log.err("Usage: gemma4_engine_e1 <model.safetensors> <p5_7_8_gen.safetensors>", .{});
        return error.MissingArgument;
    }
    const ckpt = process_args[1];
    const fixture = process_args[2];
    log.info("E1 — non-régression EngineModel(struct{{}}) == decode4 ({d} tokens)", .{NUM_STEPS});

    var reg_ck: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, ckpt);
    var store_ck: zml.io.TensorStore = .fromRegistry(allocator, &reg_ck);
    var reg_fx: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, fixture);
    defer reg_fx.deinit();
    var store_fx: zml.io.TensorStore = .fromRegistry(allocator, &reg_fx);
    defer store_fx.deinit();

    const base = store_ck.view().withPrefix("model").withPrefix("language_model");
    const model: Model = try .init(arena.allocator(), base);
    const packed_in: engine.Packed = .init(store_fx.view());
    const cache0: engine.Cache = .init(store_fx.view());
    const ctrl_sym: engine.Ctrl = .initSymbolic();

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);
    const sharding = try zml.sharding.replicatedSharding(platform);

    log.info("Materializing weights + packed inputs + caches...", .{});
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

    log.info("Compiling gen step (EngineModel(struct{{}}))...", .{});
    var exe = try platform.compile(allocator, io, model, .forward, .{ packed_in, cache0, ctrl_sym }, .{ .shardings = &.{sharding} });
    defer exe.deinit();

    var all_pass = true;
    var step_idx: usize = 0;
    while (step_idx < NUM_STEPS) : (step_idx += 1) {
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
        if (!ok) all_pass = false;
        log.info("  step {d} (pos {d}) : argmax ZML={d} HF={d} -> {s}", .{ step_idx, 4 + step_idx, tok, exp, if (ok) "PASS" else "FAIL" });

        // thread le cache grandi vers le step suivant
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
    cache_buf.sl_k.deinit();
    cache_buf.sl_v.deinit();
    cache_buf.fl_k.deinit();
    cache_buf.fl_v.deinit();

    if (all_pass) {
        log.info("E1 PASS — EngineModel(struct{{}}) génère {d} tokens, séquence == decode4 (== HF greedy)", .{NUM_STEPS});
    } else {
        log.err("E1 : divergence de séquence vs decode4 — erreur d'extraction engine.zig", .{});
        return error.GenMismatch;
    }
}
