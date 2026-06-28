// L1a — CONTRE-TEST DE NON-VACUITÉ (R2, cf analyse point 10).
//
// But : prouver que le masque bande `masks_sliding` est RÉELLEMENT consommé par l'attention sliding
// (réfuter l'aliasing / un PASS trompeur où le masque serait ignoré). C'est la contrepartie obligatoire
// du gate L1a (méthode du projet : non-vacuité, cf docs/GENERATION_LONGUE_PLAN.md Step 4 + DESIGN §7).
//
// Corruption : on rebind le buffer `masks_sliding` sur le buffer `masks_full` (causal plein [0,p]) au
// lieu de la bande [max(0,p-511), p]. Effet : les couches sliding voient TOUT le passé (fenêtre 512
// DÉSACTIVÉE) au lieu des 512 dernières positions.
//   - p < 511 : bande ≡ causal (lo=0) → tokens IDENTIQUES à HF (pas de divergence avant ~p=511).
//   - p > 511 (p>=512) : la bande HF tronque (lo=p-511>0) à [p-511,p], notre version voit [0,p] → attention différente →
//     logits différents → argmax différent → DIVERGENCE.
//
// Critère INVERSÉ (PASS = divergence observée) :
//   - Si argmax diverge d'`expected` sur >= 1 position (typiquement à partir de p~512, step ~508) →
//     le masque est bien consommé → NON-VACUITÉ PROUVÉE → on log "VACUITY-OK" (le test « réussit » en
//     échouant à reproduire HF).
//   - Si argmax == expected sur TOUTES les positions → le masque est ignoré (aliasing/vacuité) →
//     BUG → on return error.Vacuity (le test « échoue » = alerte rouge).
//
// Diagnostic : on rapporte `first_fail` (position de 1re divergence, attendue ~512/step ~508) et le compte de
// divergences. Une divergence bien AVANT 511 est suspecte (les masques bande/causal coïncident pour
// p<511 → une divergence précoce indiquerait une autre cause, à investiguer).
//
// On réutilise l'exécution chunkée (== gchunk, chemin L1a prouvé) pour rester fidèle au gate qu'on
// contre-teste. max_steps (3e arg optionnel) permet un run court (ex: 600) capturant la divergence
// au p~512 sans attendre les 1020 steps complets.
//
// CLI : gemma4_gchunk_vacuity <model.safetensors> <gen_long.safetensors> [max_steps]

const std = @import("std");
const log = std.log;
const zml = @import("zml");
const engine = @import("engine.zig");
const mem_probe = @import("mem_probe.zig");

pub const std_options: std.Options = .{ .log_level = .info };

const L_MAX: i64 = 1024;
const NUM_LAYERS: usize = 35;
const CHUNK: usize = 5;
const SYNC_EVERY: usize = 1; // == gchunk L1a (chemin contre-testé)
const B: i64 = 1;
const S: i64 = 1;
const D: i64 = 1536;
const SLIDING_WINDOW: i64 = 512;
const SEQ_LEN: i64 = 4; // positions décodées = SEQ_LEN + step_idx → p=511 ≈ step 507

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
    @setEvalBranchQuota(200000);
    const arena = init.arena;
    const allocator = init.gpa;
    const io = init.io;

    const process_args = try init.minimal.args.toSlice(arena.allocator());
    if (process_args.len < 3) {
        log.err("Usage: gemma4_gchunk_vacuity <model.safetensors> <gen_long.safetensors> [max_steps]", .{});
        return error.MissingArgument;
    }
    const ckpt = process_args[1];
    const fixture = process_args[2];
    const max_steps: ?usize = if (process_args.len >= 4)
        std.fmt.parseInt(usize, process_args[3], 10) catch null
    else
        null;
    log.info("L1a NON-VACUITÉ — masks_sliding corrompu en causal (fenêtre 512 OFF), attend 1re divergence ~p={d} (step ~508)", .{SLIDING_WINDOW});

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

    var exp_slice = try exp_buf.e.toSliceAlloc(allocator, io);
    defer exp_slice.free(allocator);
    const expected_tokens = exp_slice.items(i32);
    const total = expected_tokens.len;
    const num_steps = if (max_steps) |m| @min(m, total) else total;

    // ===== CORRUPTION : rebind masks_sliding <- masks_full (fenêtre OFF, causal [0,p]) =====
    // Bufferized(PackedLong) = struct de Buffers (handles device). On construit une copie où le champ
    // masks_sliding pointe sur le buffer masks_full. Les shapes .k=L_MAX coïncident (fixture L0). Les
    // buffers sont en lecture seule (inputs) → l'aliasing d'un même buffer pour 2 champs est sûr.
    const pk_corrupt = zml.Bufferized(PackedLong){
        .embeds = pk_buf.embeds,
        .embptls = pk_buf.embptls,
        .cos_full = pk_buf.cos_full,
        .sin_full = pk_buf.sin_full,
        .masks_sliding = pk_buf.masks_full, // ← CORRUPTION : bande -> causal (fenêtre OFF)
        .masks_full = pk_buf.masks_full,
        .positions = pk_buf.positions,
    };

    const hidden_shape = zml.Shape.init(.{ B, S, D }, .f32).withTags(.{ .b, .s, .d });
    const zeros = try arena.allocator().alloc(u8, @intCast(B * S * D * 4));
    @memset(zeros, 0);
    var dummy_hidden = try zml.Buffer.fromBytes(io, platform, hidden_shape, sharding, zeros);
    defer dummy_hidden.deinit();

    log.info("Compiling {d} stages (chemin == gchunk L1a)...", .{N_STAGES});
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
    mem_probe.logMem("post-compile");

    // ===== Boucle : on attend DIVERGENCE (critère inversé) =====
    var n_match: usize = 0;
    var n_diverge: usize = 0;
    var first_fail: i64 = -1;
    var step_idx: usize = 0;
    while (step_idx < num_steps) : (step_idx += 1) {
        var step_buf = try zml.Buffer.scalar(io, platform, @as(u32, @intCast(step_idx)), .u32, sharding);
        const ctrl_buf = zml.Bufferized(engine.Ctrl){ .step = step_buf };

        var hidden_buf = dummy_hidden;
        var tok: i64 = -1;
        inline for (STAGES, 0..) |stage, si| {
            var args = try exes[si].args(allocator);
            var results = try exes[si].results(allocator);
            args.set(.{ eng_buf, pk_corrupt, cache_buf, hidden_buf, ctrl_buf }); // ← pk_corrupt
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
        const pos = SEQ_LEN + @as(i64, @intCast(step_idx));
        if (tok == exp) {
            n_match += 1;
        } else {
            n_diverge += 1;
            if (first_fail < 0) first_fail = @intCast(step_idx);
            if (n_diverge <= 8 or pos >= SLIDING_WINDOW - 8) {
                log.info("  DIVERGENCE step {d} (pos {d}) : ZML(corrompu)={d} HF={d}", .{ step_idx, pos, tok, exp });
            }
        }
        step_buf.deinit();
    }
    cache_buf.sl_k.deinit();
    cache_buf.sl_v.deinit();
    cache_buf.fl_k.deinit();
    cache_buf.fl_v.deinit();

    log.info("NON-VACUITÉ : {d}/{d} divergent, {d} match (first_fail step {d})", .{ n_diverge, num_steps, n_match, first_fail });

    // Critère inversé : divergence observée => masque consommé => non-vacuité PROUVÉE.
    if (n_diverge > 0) {
        const ff_pos = if (first_fail >= 0) SEQ_LEN + first_fail else -1;
        log.info("VACUITY-OK (non-vacuité prouvée) — le masque bande est bien consommé ; 1re divergence au pos {d}", .{ff_pos});
        if (ff_pos >= 0 and ff_pos < SLIDING_WINDOW - 4) {
            log.warn("  1re divergence précoce (pos {d} < {d}-4) : suspect (bande≡causal pour p<511) → investiguer", .{ ff_pos, SLIDING_WINDOW });
        }
    } else {
        // Aucune divergence malgré la fenêtre désactivée → le masque est ignoré → VACUITÉ (bug).
        log.err("VACUITY-FAIL : {d}/{d} match malgré masks_sliding corrompu — le masque bande N'EST PAS consommé (aliasing/vacuité) !", .{ n_match, num_steps });
        return error.Vacuity;
    }
}
