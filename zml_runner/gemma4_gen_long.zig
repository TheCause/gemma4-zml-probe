// L1a — replay de GÉNÉRATION LONGUE : cache sliding LINÉAIRE borné (.k = L_MAX) + masque BANDE.
//
// Rejoue la fixture L0 (gen_long.safetensors) : à chaque step le moteur ZML embed le token `fed[step]`
// (déjà empaqueté), SCATTER son KV à .k=position, et lit le cache via le masque par type de couche
// (masks_sliding = bande [p-511,p] / masks_full = causal). On compare l'argmax à `expected[step]`.
//
// Config moteur : EngineModel(struct{}, .{ .two_masks=true, .kmax_sliding=L_MAX, .kmax_full=L_MAX }).
//   - ring=false → cache sliding LINÉAIRE (.k = L_MAX), scatter à `pos` (pas de modulo). C'est L1a ;
//     L1b passera en ring=true / kmax_sliding=512.
//   - two_masks=true → Packed(true) (masks_sliding + masks_full), sélection par comptime isFull(i).
//
// PASS = argmax ZML[k] == expected[k] pour tout k (séquence == HF greedy sliding window 512).
// CLI : gemma4_gen_long <model.safetensors> <gen_long.safetensors>

const std = @import("std");
const log = std.log;
const zml = @import("zml");
const engine = @import("engine.zig");

pub const std_options: std.Options = .{ .log_level = .info };

const L_MAX: i64 = 1024; // réduit de 2048 : pic compile .k=2048 (~34Go) > hôte 32Go → swap thrash. 1024 franchit 512.
const Model = engine.EngineModel(struct {}, .{ .two_masks = true, .kmax_sliding = L_MAX, .kmax_full = L_MAX });
const PackedLong = engine.Packed(true);

// Séquence attendue (HF), lue côté host ; `len` = NUM_STEPS (dynamique).
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
        log.err("Usage: gemma4_gen_long <model.safetensors> <gen_long.safetensors>", .{});
        return error.MissingArgument;
    }
    const ckpt = process_args[1];
    const fixture = process_args[2];
    log.info("L1a — replay génération longue (cache linéaire .k={d} + masque bande)", .{L_MAX});

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

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);
    const sharding = try zml.sharding.replicatedSharding(platform);

    log.info("Materializing weights + packed inputs + caches (L_MAX={d}) ...", .{L_MAX});
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
    const num_steps = expected_tokens.len;
    log.info("NUM_STEPS (= len expected) = {d}", .{num_steps});

    log.info("Compiling gen step (EngineModel two_masks, ring=false) ...", .{});
    var exe = try platform.compile(allocator, io, model, .forward, .{ packed_in, cache0, ctrl_sym }, .{ .shardings = &.{sharding} });
    defer exe.deinit();

    var all_pass = true;
    var n_match: usize = 0;
    var first_fail: i64 = -1;
    var step_idx: usize = 0;
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
        if (ok) {
            n_match += 1;
        } else {
            all_pass = false;
            if (first_fail < 0) first_fail = @intCast(step_idx);
            if (step_idx - @as(usize, @intCast(@max(first_fail, 0))) < 8) {
                log.err("  FAIL step {d} (pos {d}) : argmax ZML={d} HF={d}", .{ step_idx, 4 + step_idx, tok, exp });
            }
        }
        if ((step_idx + 1) % 256 == 0) {
            log.info("  ... {d}/{d} steps, {d} match", .{ step_idx + 1, num_steps, n_match });
        }

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

    log.info("L1a : {d}/{d} tokens argmax-match", .{ n_match, num_steps });
    if (all_pass) {
        log.info("L1a PASS — replay {d} tokens, séquence == HF greedy (sliding window 512, cache linéaire)", .{num_steps});
    } else {
        log.err("L1a : divergence (1er fail au step {d} / {d} match) — masque bande ou scatter à revoir", .{ first_fail, n_match });
        return error.GenMismatch;
    }
}
