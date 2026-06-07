// L1a CHUNKÉ (perf) — decode découpé en stages compilés séparément pour borner le pic mémoire.
//
// Adapte le mode `chain` du prefill au decode (cf docs/GENERATION_LONGUE_CHUNKING_DESIGN.md) :
// les 35 couches sont découpées en stages de CHUNK couches, chacun compilé via `compileFn` (fn-factory
// comptime). À chaque step on exécute les stages en séquence, threadant hidden + cache device→device,
// avec sync (toSliceAlloc) après chaque pour libérer le working set. Calcul == forward mono (runLayerGen
// partagé) → mêmes tokens, mais pic mémoire borné (moins de poids f32 coexistant).
//
// GATE 0 (cette version) : compile N stages, mesure le pic POST-COMPILE (les N exe résidents = go/no-go),
// puis exécute NUM_STEPS_GATE0 steps pour vérifier l'équivalence (tokens == expected). Gestion mémoire
// best-effort (petite fuite tolérée sur peu de steps ; le pic post-compile est mesuré AVANT les steps).
//
// CLI : gemma4_gen_long_chunked <model.safetensors> <gen_long.safetensors>

const std = @import("std");
const log = std.log;
const zml = @import("zml");
const engine = @import("engine.zig");

pub const std_options: std.Options = .{ .log_level = .info };

const L_MAX: i64 = 1024;
const NUM_LAYERS: usize = 35;
const CHUNK: usize = 5; // couches/stage (divise 15 → pas de stage mixte producer/reader)
const NUM_STEPS_GATE0: usize = 4; // gate 0 : équivalence sur quelques steps (fuite cache tolérée)
const B: i64 = 1;
const S: i64 = 1;
const D: i64 = 1536;

const Model = engine.EngineModel(struct {}, .{ .two_masks = true, .kmax_sliding = L_MAX, .kmax_full = L_MAX });
const PackedLong = engine.Packed(true);
const StageOut = struct { zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor };

const Stage = struct { start: usize, end: usize, first: bool, last: bool };
const N_STAGES: usize = (NUM_LAYERS + CHUNK - 1) / CHUNK;
const STAGES: [N_STAGES]Stage = blk: {
    var s: [N_STAGES]Stage = undefined;
    var i: usize = 0;
    var start: usize = 0;
    while (start < NUM_LAYERS) : (start += CHUNK) {
        const end = @min(start + CHUNK, NUM_LAYERS);
        s[i] = .{ .start = start, .end = end, .first = (start == 0), .last = (end == NUM_LAYERS) };
        i += 1;
    }
    break :blk s;
};

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
    @setEvalBranchQuota(200000); // inline for sur N_STAGES × compileFn générique déborde le quota par défaut
    const arena = init.arena;
    const allocator = init.gpa;
    const io = init.io;

    const process_args = try init.minimal.args.toSlice(arena.allocator());
    if (process_args.len < 3) {
        log.err("Usage: gemma4_gen_long_chunked <model.safetensors> <gen_long.safetensors>", .{});
        return error.MissingArgument;
    }
    const ckpt = process_args[1];
    const fixture = process_args[2];
    log.info("L1a CHUNKÉ — {d} couches en {d} stages de {d} (L_MAX={d})", .{ NUM_LAYERS, N_STAGES, CHUNK, L_MAX });

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
    const hidden_sym = zml.Tensor.init(.{ B, S, D }, .f32).withTags(.{ .b, .s, .d });

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);
    const sharding = try zml.sharding.replicatedSharding(platform);

    log.info("Materializing weights + packed + cache0...", .{});
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
    const num_steps = expected_tokens.len; // run complet (équivalence sur toute la séquence)

    // dummy hidden {b,s,d} (entrée ignorée du first stage).
    const hidden_shape = zml.Shape.init(.{ B, S, D }, .f32).withTags(.{ .b, .s, .d });
    const zeros = try arena.allocator().alloc(u8, @intCast(B * S * D * 4));
    @memset(zeros, 0);
    var dummy_hidden = try zml.Buffer.fromBytes(io, platform, hidden_shape, sharding, zeros);
    defer dummy_hidden.deinit();

    // ===== Compile les N stages (fn-factory comptime) =====
    log.info("Compiling {d} stages...", .{N_STAGES});
    var exes: [N_STAGES]zml.Exe = undefined;
    inline for (STAGES, 0..) |stage, si| {
        const F = struct {
            fn f(m: Model, p: PackedLong, ca: engine.Cache, h: zml.Tensor, ct: engine.Ctrl) StageOut {
                return m.forwardStageGen(stage.start, stage.end, stage.first, stage.last, p, ca, h, ct);
            }
        }.f;
        exes[si] = try platform.compileFn(allocator, io, F, .{ model, packed_in, cache0, hidden_sym, ctrl_sym }, .{ .shardings = &.{sharding} });
        log.info("  stage {d} [{d},{d}) first={} last={} compilé", .{ si, stage.start, stage.end, stage.first, stage.last });
    }
    defer for (&exes) |*e| e.deinit();
    log.info("=== POST-COMPILE : {d} stages résidents (mesurer le pic mémoire ICI) ===", .{N_STAGES});

    // ===== Boucle steps (gate 0 : équivalence sur {d} steps) =====
    var all_pass = true;
    var n_match: usize = 0;
    var step_idx: usize = 0;
    while (step_idx < num_steps) : (step_idx += 1) {
        var step_buf = try zml.Buffer.scalar(io, platform, @as(u32, @intCast(step_idx)), .u32, sharding);
        const ctrl_buf = zml.Bufferized(engine.Ctrl){ .step = step_buf };

        var hidden_buf = dummy_hidden; // first stage : ignoré
        var tok: i64 = -1;
        inline for (STAGES, 0..) |stage, si| {
            var args = try exes[si].args(allocator);
            var results = try exes[si].results(allocator);
            args.set(.{ eng_buf, pk_buf, cache_buf, hidden_buf, ctrl_buf });
            exes[si].call(args, &results);
            var out0, const nsl, const nsv, const nfl, const nfv = results.get(struct {
                zml.Buffer, zml.Buffer, zml.Buffer, zml.Buffer, zml.Buffer,
            });

            // sync out0 (matérialise → libère le working set du stage)
            {
                var s = try out0.toSliceAlloc(allocator, io);
                s.free(allocator);
            }

            // thread cache : deinit l'ancien (pattern e1 — les buffers d'entrée ne sont pas « donnés » par
            // call, donc deinitables après ; les reader-stages retournent une copie du cache, pas un alias).
            var old_cache = cache_buf;
            cache_buf = zml.Bufferized(engine.Cache){ .sl_k = nsl, .sl_v = nsv, .fl_k = nfl, .fl_v = nfv };
            old_cache.sl_k.deinit();
            old_cache.sl_v.deinit();
            old_cache.fl_k.deinit();
            old_cache.fl_v.deinit();

            // thread hidden
            if (si != 0) hidden_buf.deinit();
            if (stage.last) {
                tok = try argmaxOf(allocator, io, &out0);
                out0.deinit();
            } else {
                hidden_buf = out0;
            }
            args.deinit(allocator);
            results.deinit(allocator);
        }

        const exp = @as(i64, @intCast(expected_tokens[step_idx]));
        const ok = tok == exp;
        if (ok) n_match += 1 else all_pass = false;
        if (!ok) log.err("  FAIL step {d} (pos {d}) : argmax ZML={d} HF={d}", .{ step_idx, 4 + step_idx, tok, exp });
        if ((step_idx + 1) % 256 == 0) log.info("  ... {d}/{d} steps, {d} match", .{ step_idx + 1, num_steps, n_match });
        step_buf.deinit();
    }

    log.info("L1a CHUNKÉ : {d}/{d} tokens match", .{ n_match, num_steps });
    if (all_pass) {
        log.info("L1a CHUNKÉ PASS — moteur chunké == HF greedy sur {d} tokens (== mono, exécution borné mémoire)", .{num_steps});
    } else {
        log.err("L1a CHUNKÉ : divergence vs expected", .{});
        return error.GenMismatch;
    }
}
