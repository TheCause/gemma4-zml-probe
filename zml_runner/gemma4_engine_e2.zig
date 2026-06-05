// Gate E2 — la brique branchée == le POC, sans copie.
//
// EngineModel(TurboQuantVBrick) doit reproduire gemma4_gen_vq.zig (Q5, le POC V-quant copié-collé) :
// même fixture (decode_vq_gen.safetensors), même oracle (tokens `expected` HF-V-quant), même séquence.
// Le socle remplace la copie : le moteur n'est PAS réécrit, la transformation V-quant est INJECTÉE par
// la brique au point post_v_norm.
//
// Plomberie multi-store (cf engine DESIGN §3.4) : les poids vivent dans le checkpoint (store_ck) et les
// constantes de la brique (codebook_*/hadamard_*) dans la fixture (store_fx). `zml.io.load` résout une
// struct contre UN seul store → on charge poids et brique SÉPARÉMENT, puis on assemble le
// `Bufferized(EngineModel(TurboQuantVBrick))` à la main (même structure de champs → mapping positionnel).
//
// CLI : gemma4_engine_e2 <model.safetensors> <decode_vq_gen.safetensors>

const std = @import("std");
const log = std.log;
const zml = @import("zml");
const engine = @import("engine.zig");
const TurboQuantVBrick = @import("brick_turboquant.zig").TurboQuantVBrick;

pub const std_options: std.Options = .{ .log_level = .info };

const NUM_STEPS: usize = 4;

const Model = engine.EngineModel(TurboQuantVBrick); // model branché (compile)
const Wmodel = engine.EngineModel(struct {}); // poids seuls (load depuis store_ck)

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
        log.err("Usage: gemma4_engine_e2 <model.safetensors> <decode_vq_gen.safetensors>", .{});
        return error.MissingArgument;
    }
    const ckpt = process_args[1];
    const fixture = process_args[2];
    log.info("E2 — EngineModel(TurboQuantVBrick) == gen_vq ({d} tokens, brique sans copie)", .{NUM_STEPS});

    var reg_ck: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, ckpt);
    var store_ck: zml.io.TensorStore = .fromRegistry(allocator, &reg_ck);
    var reg_fx: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, fixture);
    defer reg_fx.deinit();
    var store_fx: zml.io.TensorStore = .fromRegistry(allocator, &reg_fx);
    defer store_fx.deinit();

    const base = store_ck.view().withPrefix("model").withPrefix("language_model");
    const fixture_view = store_fx.view();

    // Model symbolique branché (poids via base_ck + brique via fixture_view) — sert au compile/trace.
    const model: Model = try .initBrick(arena.allocator(), base, fixture_view);
    // Model poids-seuls (même structure de poids) — sert à charger les poids depuis store_ck.
    const wmodel: Wmodel = try .init(arena.allocator(), base);
    // Brique symbolique — sert à charger les constantes depuis store_fx.
    const brick_sym: TurboQuantVBrick = .init(fixture_view);

    const packed_in: engine.Packed = .init(fixture_view);
    const cache0: engine.Cache = .init(fixture_view);
    const ctrl_sym: engine.Ctrl = .initSymbolic();

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);
    const sharding = try zml.sharding.replicatedSharding(platform);

    log.info("Materializing weights (store_ck) + brick constants (store_fx) + packed + caches...", .{});
    const w_buf = try wmodel.load(arena.allocator(), io, platform, &store_ck, &.{sharding});
    const brick_buf = try zml.io.load(TurboQuantVBrick, &brick_sym, arena.allocator(), io, platform, &store_fx, .{ .shardings = &.{sharding}, .parallelism = 1, .dma_chunks = 1, .dma_chunk_size = 16 * 1024 * 1024 });
    const pk_buf = try packed_in.load(arena.allocator(), io, platform, &store_fx, &.{sharding});
    var cache_buf = try cache0.load(arena.allocator(), io, platform, &store_fx, &.{sharding});
    const exp_sym: ExpW = .init(fixture_view);
    var exp_buf = try exp_sym.load(arena.allocator(), io, platform, &store_fx, &.{sharding});
    store_ck.deinit();
    reg_ck.deinit();

    // Assemblage manuel du Bufferized(Model) = poids (w_buf) + brique (brick_buf). Mapping positionnel :
    // Bufferized(Model) a les MÊMES champs/ordre que Bufferized(Wmodel) pour les poids, + .brick.
    const eng_buf = zml.Bufferized(Model){
        .embed_tokens = w_buf.embed_tokens,
        .per_layer_model_projection = w_buf.per_layer_model_projection,
        .per_layer_projection_norm = w_buf.per_layer_projection_norm,
        .final_norm = w_buf.final_norm,
        .layers = w_buf.layers,
        .brick = brick_buf,
    };

    var exp_slice = try exp_buf.e.toSliceAlloc(allocator, io);
    defer exp_slice.free(allocator);
    const expected_tokens = exp_slice.items(i32);

    log.info("Compiling gen step (EngineModel(TurboQuantVBrick))...", .{});
    var exe = try platform.compile(allocator, io, model, .forward, .{ packed_in, cache0, ctrl_sym }, .{ .shardings = &.{sharding} });
    defer exe.deinit();

    var all_pass = true;
    var n_match: usize = 0;
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
        if (ok) n_match += 1 else all_pass = false;
        log.info("  step {d} (pos {d}) : argmax ZML-brick={d} HF-V-quant={d} -> {s}", .{ step_idx, 4 + step_idx, tok, exp, if (ok) "PASS" else "FAIL" });

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

    log.info("E2 brique : {d}/{d} tokens argmax-match (EngineModel(TurboQuantVBrick) == gen_vq)", .{ n_match, NUM_STEPS });
    if (all_pass) {
        log.info("E2 PASS — la brique branchée reproduit gen_vq SANS copier le moteur ({d} tokens)", .{NUM_STEPS});
    } else {
        log.err("E2 : divergence vs gen_vq ({d}/{d} match) — brique/assemblage à corriger", .{ n_match, NUM_STEPS });
        return error.GenMismatch;
    }
}
