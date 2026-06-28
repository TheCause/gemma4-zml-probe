// L1b — Replay génération longue : VRAI ring-buffer 512 + masque CIRCULAIRE (replay, chunké).
//
// Configuration : EngineModel(struct{}, .{ .ring=true, .two_masks=true, .kmax_sliding=512,
//                                        .kmax_full=L_MAX }). Différences vs L1a (gemma4_gchunk) :
//   - ring=true  → scatter sliding CIRCULAIRE à `pos % 512` (au lieu de `pos` linéaire).
//   - kmax_sliding=512 → cache sliding `.k=512` (anneau), masque `masks_sliding` CIRCULAIRE (.k=512).
//   - kmax_full=L_MAX   → cache full reste LINÉAIRE `.k=L_MAX` (les couches full ne sont JAMAIS fenêtrées).
//
// La séquence greedy HF est IDENTIQUE à L1a (le ring 512 encode le même attention des 512 dernières
// positions) → PASS = argmax == expected sur les N tokens, y compris APRÈS le wrap (pos ≥ 512).
//
// Contre-test de non-vacuité (PLAN L1b step 277) : lancer ce runner sur `gen_long_ring_naive.safetensors`
// (masque non-remappé) → doit DIVERGER à partir de p≈512 (la bande déborde le ring, slots masqués à tort).
//
// Chemin d'exécution chunké (== gchunk) pour éviter le thrash mémoire du mono-graphe à `.k≥512`.
// Inclut l'instrumentation RSS (R1, mem_probe.zig).
//
// CLI : gemma4_gchunk_ring <model.safetensors> <gen_long_ring.safetensors|gen_long_ring_naive.safetensors>

const std = @import("std");
const log = std.log;
const zml = @import("zml");
const engine = @import("engine.zig");
const mem_probe = @import("mem_probe.zig");

pub const std_options: std.Options = .{ .log_level = .info };

const L_MAX: i64 = 1024;
const NUM_LAYERS: usize = 35;
const CHUNK: usize = 5;
const SYNC_EVERY: usize = 1;
const RSS_EVERY: usize = 64;
const B: i64 = 1;
const S: i64 = 1;
const D: i64 = 1536;

// L1b : ring 512 + masque circulaire. kmax_full reste L_MAX (full jamais fenêtré).
const Model = engine.EngineModel(struct {}, .{ .ring = true, .two_masks = true, .kmax_sliding = 512, .kmax_full = L_MAX });
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
    @setEvalBranchQuota(200000);
    const arena = init.arena;
    const allocator = init.gpa;
    const io = init.io;

    const process_args = try init.minimal.args.toSlice(arena.allocator());
    if (process_args.len < 3) {
        log.err("Usage: gemma4_gchunk_ring <model.safetensors> <gen_long_ring.safetensors|..._naive.safetensors>", .{});
        return error.MissingArgument;
    }
    const ckpt = process_args[1];
    const fixture = process_args[2];
    const is_naive = std.mem.indexOf(u8, fixture, "naive") != null;
    log.info("L1b — ring 512 + masque circulaire (chunké) ; fixture={s}{s}", .{ fixture, if (is_naive) " [NAIVE → attend divergence ~p=512]" else "" });

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

    const eng_buf = try model.load(arena.allocator(), io, platform, &store_ck, &.{sharding});
    const pk_buf = try packed_in.load(arena.allocator(), io, platform, &store_fx, &.{sharding});
    var cache_buf = try cache0.load(arena.allocator(), io, platform, &store_fx, &.{sharding});
    const exp_sym: ExpW = .init(store_fx.view());
    var exp_buf = try exp_sym.load(arena.allocator(), io, platform, &store_fx, &.{sharding});
    store_ck.deinit();
    reg_ck.deinit();
    mem_probe.logMem(io, "post-load");

    var exp_slice = try exp_buf.e.toSliceAlloc(allocator, io);
    defer exp_slice.free(allocator);
    const expected_tokens = exp_slice.items(i32);
    const num_steps = expected_tokens.len;

    const hidden_shape = zml.Shape.init(.{ B, S, D }, .f32).withTags(.{ .b, .s, .d });
    const zeros = try arena.allocator().alloc(u8, @intCast(B * S * D * 4));
    @memset(zeros, 0);
    var dummy_hidden = try zml.Buffer.fromBytes(io, platform, hidden_shape, sharding, zeros);
    defer dummy_hidden.deinit();

    log.info("Compiling {d} stages (ring=true, kmax_sliding=512)...", .{N_STAGES});
    var exes: [N_STAGES]zml.Exe = undefined;
    inline for (STAGES, 0..) |stage, si| {
        const F = struct {
            fn f(m: Model, p: PackedLong, ca: engine.Cache, h: zml.Tensor, ct: engine.Ctrl) StageOut {
                return m.forwardStageGen(stage.start, stage.end, stage.first, stage.last, p, ca, h, ct);
            }
        }.f;
        exes[si] = try platform.compileFn(allocator, io, F, .{ model, packed_in, cache0, hidden_sym, ctrl_sym }, .{ .shardings = &.{sharding} });
    }
    defer for (&exes) |*e| e.deinit();
    mem_probe.logMem(io, "post-compile (go/no-go)");

    var all_pass = true;
    var n_match: usize = 0;
    var first_fail: i64 = -1;
    var step_idx: usize = 0;
    const rss0 = mem_probe.rssKb(io);
    while (step_idx < num_steps) : (step_idx += 1) {
        var step_buf = try zml.Buffer.scalar(io, platform, @as(u32, @intCast(step_idx)), .u32, sharding);
        const ctrl_buf = zml.Bufferized(engine.Ctrl){ .step = step_buf };

        var hidden_buf = dummy_hidden;
        var tok: i64 = -1;
        inline for (STAGES, 0..) |stage, si| {
            var args = try exes[si].args(allocator);
            var results = try exes[si].results(allocator);
            args.set(.{ eng_buf, pk_buf, cache_buf, hidden_buf, ctrl_buf });
            exes[si].call(args, &results);
            var out0, const nsl, const nsv, const nfl, const nfv = results.get(struct {
                zml.Buffer, zml.Buffer, zml.Buffer, zml.Buffer, zml.Buffer,
            });

            const do_sync = (si % SYNC_EVERY == SYNC_EVERY - 1);
            if (do_sync) {
                var s = try out0.toSliceAlloc(allocator, io);
                s.free(allocator);
            }

            var old_cache = cache_buf;
            cache_buf = zml.Bufferized(engine.Cache){ .sl_k = nsl, .sl_v = nsv, .fl_k = nfl, .fl_v = nfv };
            old_cache.sl_k.deinit();
            old_cache.sl_v.deinit();
            old_cache.fl_k.deinit();
            old_cache.fl_v.deinit();

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
        if (ok) n_match += 1 else {
            all_pass = false;
            if (first_fail < 0) first_fail = @intCast(step_idx);
            if (first_fail >= 0 and (step_idx - @as(usize, @intCast(first_fail)) < 8)) {
                log.info("  DIVERGENCE step {d} (pos {d}) : ZML={d} HF={d}", .{ step_idx, 4 + step_idx, tok, exp });
            }
        }
        if ((step_idx + 1) % 256 == 0) log.info("  ... {d}/{d} steps, {d} match", .{ step_idx + 1, num_steps, n_match });
        if ((step_idx % RSS_EVERY == RSS_EVERY - 1) and (rss0 != null)) {
            var tag_buf: [32]u8 = undefined;
            const tag = std.fmt.bufPrint(&tag_buf, "step {d}", .{step_idx}) catch "step";
            mem_probe.logMem(io, tag);
        }
        step_buf.deinit();
    }
    cache_buf.sl_k.deinit();
    cache_buf.sl_v.deinit();
    cache_buf.fl_k.deinit();
    cache_buf.fl_v.deinit();
    mem_probe.logMem(io, "post-run");

    log.info("L1b RING : {d}/{d} tokens match (first_fail step {d})", .{ n_match, num_steps, first_fail });
    if (is_naive) {
        // Contre-test : le masque non-remappé doit DIVERGER à partir de p≈512 (pos 512 = step ~508).
        const ff_pos: i64 = if (first_fail >= 0) 4 + first_fail else -1;
        if (n_match < num_steps) {
            log.info("L1b NON-VACUITÉ RING OK — divergence (1re au pos {d}) prouve le wrap circulaire consommé", .{ff_pos});
            if (ff_pos >= 0 and ff_pos < 508) log.warn("  1re divergence précoce (pos {d}<508) : investiguer", .{ff_pos});
        } else {
            log.err("L1b NON-VACUITÉ RING FAIL : aucune divergence malgré masque non-remappé (wrap non consommé !)", .{});
            return error.Vacuity;
        }
    } else {
        if (all_pass) {
            log.info("L1b RING PASS — {d} tokens == HF greedy (ring 512 + masque circulaire, wrap franchi)", .{num_steps});
        } else {
            log.err("L1b RING : divergence vs expected (1re au step {d}) — ring/masque circulaire à investiguer", .{first_fail});
            return error.GenMismatch;
        }
    }
}
