// Contre-test de NON-VACUITÉ par LOGITS (rigoureux) — tranche définitivement si le masque sliding est
// réellement consommé par le moteur, là où le contre-test argmax échouait (greedy trop robuste : ajouter
// des tokens anciens de faible poids ne fait pas basculer le top-1).
//
// Principe : cache LINÉAIRE (.k=1024). À chaque step >= THRESHOLD on exécute DEUX forwards chunkés depuis
// le MÊME cache d'entrée :
//   (A) masque correct   : masks_sliding = bande [max(0,p-511), p]
//   (B) masque corrompu  : masks_sliding <- masks_full = causal plein [0, p]  (fenêtre 512 OFF)
// puis on mesure max_abs(logits_A - logits_B). Les K/V écrits au cache sont INDÉPENDANTS du masque (le
// scatter écrit le token courant quel que soit le masque) → on thread le cache du run A, le run B est jeté.
//
// Lecture (AUTO-VALIDANTE) :
//   - p < 512  : bande == causal → masques identiques → max_abs == 0 EXACT (valide la machinerie de
//     comparaison : même entrée → même sortie).
//   - p >= 512 : causal ajoute les positions anciennes [0, p-512]. Si le masque est CONSOMMÉ, les logits
//     DIFFÈRENT (max_abs > 0, croissant avec p). Si max_abs reste 0 malgré p>=512 → le masque n'est PAS
//     consommé (vacuité réelle) OU l'aliasing du buffer corrompu ne prend pas → on le signale.
//
// La métrique logits est SENSIBLE : une seule position ancienne incluse modifie le softmax → les logits
// changent (même de ~1e-6), même quand l'argmax greedy ne bascule pas.
//
// CLI : gemma4_vacuity_logits <model.safetensors> <gen_long.safetensors> [max_steps]   (défaut 540)

const std = @import("std");
const log = std.log;
const zml = @import("zml");
const engine = @import("engine.zig");
const mem_probe = @import("mem_probe.zig");

pub const std_options: std.Options = .{ .log_level = .info };

const L_MAX: i64 = 1024;
const NUM_LAYERS: usize = 35;
const CHUNK: usize = 5;
const B: i64 = 1;
const S: i64 = 1;
const D: i64 = 1536;
const SLIDING_WINDOW: i64 = 512;
const SEQ_LEN: i64 = 4; // p = SEQ_LEN + step
const THRESHOLD: usize = 505; // double-forward à partir d'ici (couvre p<512 sanity ET p>=512 test)
const DEFAULT_MAX_STEPS: usize = 540;

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

const FwdOut = struct { logits: zml.Buffer, cache: zml.Bufferized(engine.Cache) };

fn deinitCache(c: *zml.Bufferized(engine.Cache)) void {
    c.sl_k.deinit();
    c.sl_v.deinit();
    c.fl_k.deinit();
    c.fl_v.deinit();
}

/// Un forward chunké complet (7 stages) depuis `cache_in`, threadant hidden+cache stage-à-stage.
/// `preserve_input` : si true, NE deinit PAS `cache_in` (réutilisé pour un 2e forward) ; les caches
/// intermédiaires sont toujours libérés. Retourne (logits du dernier stage, cache de sortie).
fn runFwd(
    allocator: std.mem.Allocator,
    io: std.Io,
    exes: []zml.Exe,
    eng_buf: zml.Bufferized(Model),
    pk: zml.Bufferized(PackedLong),
    cache_in: zml.Bufferized(engine.Cache),
    dummy_hidden: zml.Buffer,
    ctrl_buf: zml.Bufferized(engine.Ctrl),
    comptime preserve_input: bool,
) !FwdOut {
    var hidden_buf = dummy_hidden; // first stage : ignoré (jamais deinit ici — dummy partagé)
    var cache_cur = cache_in;
    var logits: zml.Buffer = undefined;
    inline for (STAGES, 0..) |stage, si| {
        var args = try exes[si].args(allocator);
        var results = try exes[si].results(allocator);
        args.set(.{ eng_buf, pk, cache_cur, hidden_buf, ctrl_buf });
        exes[si].call(args, &results);
        var out0, const nsl, const nsv, const nfl, const nfv = results.get(struct {
            zml.Buffer, zml.Buffer, zml.Buffer, zml.Buffer, zml.Buffer,
        });

        // libère le cache courant SAUF l'entrée préservée (si==0 && preserve_input).
        if (si != 0 or !preserve_input) deinitCache(&cache_cur);
        cache_cur = zml.Bufferized(engine.Cache){ .sl_k = nsl, .sl_v = nsv, .fl_k = nfl, .fl_v = nfv };

        if (si != 0) hidden_buf.deinit();
        if (stage.last) {
            logits = out0; // matérialisé/comparé par l'appelant
        } else {
            var s = try out0.toSliceAlloc(allocator, io); // sync (borne le working set, == gchunk)
            s.free(allocator);
            hidden_buf = out0;
        }
        args.deinit(allocator);
        results.deinit(allocator);
    }
    return .{ .logits = logits, .cache = cache_cur };
}

fn maxAbsDiff(allocator: std.mem.Allocator, io: std.Io, a: *zml.Buffer, b: *zml.Buffer) !f32 {
    var sa = try a.toSliceAlloc(allocator, io);
    defer sa.free(allocator);
    var sb = try b.toSliceAlloc(allocator, io);
    defer sb.free(allocator);
    const va = sa.items(f32);
    const vb = sb.items(f32);
    var m: f32 = 0;
    for (va, vb) |x, y| {
        const d = @abs(x - y);
        if (d > m) m = d;
    }
    return m;
}

pub fn main(init: std.process.Init) !void {
    @setEvalBranchQuota(200000);
    const arena = init.arena;
    const allocator = init.gpa;
    const io = init.io;

    const process_args = try init.minimal.args.toSlice(arena.allocator());
    if (process_args.len < 3) {
        log.err("Usage: gemma4_vacuity_logits <model.safetensors> <gen_long.safetensors> [max_steps]", .{});
        return error.MissingArgument;
    }
    const ckpt = process_args[1];
    const fixture = process_args[2];
    const max_steps_arg: ?usize = if (process_args.len >= 4) std.fmt.parseInt(usize, process_args[3], 10) catch null else null;
    log.info("NON-VACUITÉ par LOGITS — double forward (masque correct vs causal) ; THRESHOLD={d}, transition attendue p={d}", .{ THRESHOLD, SLIDING_WINDOW });

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
    const total = exp_slice.items(i32).len;
    const num_steps = if (max_steps_arg) |m| @min(m, total) else @min(DEFAULT_MAX_STEPS, total);

    const hidden_shape = zml.Shape.init(.{ B, S, D }, .f32).withTags(.{ .b, .s, .d });
    const zeros = try arena.allocator().alloc(u8, @intCast(B * S * D * 4));
    @memset(zeros, 0);
    var dummy_hidden = try zml.Buffer.fromBytes(io, platform, hidden_shape, sharding, zeros);
    defer dummy_hidden.deinit();

    log.info("Compiling {d} stages...", .{N_STAGES});
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
    mem_probe.logMem(io, "post-compile");

    // Corruption : masks_sliding <- masks_full (causal plein, fenêtre 512 OFF). Mêmes shapes .k=L_MAX.
    const pk_corrupt = zml.Bufferized(PackedLong){
        .embeds = pk_buf.embeds,
        .embptls = pk_buf.embptls,
        .cos_full = pk_buf.cos_full,
        .sin_full = pk_buf.sin_full,
        .masks_sliding = pk_buf.masks_full, // ← CORRUPTION
        .masks_full = pk_buf.masks_full,
        .positions = pk_buf.positions,
    };

    var max_global: f32 = 0;
    var max_below: f32 = 0; // p<512 : doit rester 0 (sanity machinerie)
    var first_nonzero_p: i64 = -1;
    var n_tested: usize = 0;
    var step_idx: usize = 0;
    while (step_idx < num_steps) : (step_idx += 1) {
        var step_buf = try zml.Buffer.scalar(io, platform, @as(u32, @intCast(step_idx)), .u32, sharding);
        const ctrl_buf = zml.Bufferized(engine.Ctrl){ .step = step_buf };
        const p = SEQ_LEN + @as(i64, @intCast(step_idx));

        if (step_idx < THRESHOLD) {
            // accumulation : single forward (masque correct), thread le cache.
            var o = try runFwd(allocator, io, exes[0..], eng_buf, pk_buf, cache_buf, dummy_hidden, ctrl_buf, false);
            cache_buf = o.cache;
            o.logits.deinit();
        } else {
            // double forward depuis le MÊME cache_buf.
            var oA = try runFwd(allocator, io, exes[0..], eng_buf, pk_buf, cache_buf, dummy_hidden, ctrl_buf, true);
            var oB = try runFwd(allocator, io, exes[0..], eng_buf, pk_corrupt, cache_buf, dummy_hidden, ctrl_buf, true);
            const md = try maxAbsDiff(allocator, io, &oA.logits, &oB.logits);

            deinitCache(&cache_buf); // l'entrée préservée, on peut la libérer
            deinitCache(&oB.cache); // run B jeté (K/V == A)
            oA.logits.deinit();
            oB.logits.deinit();
            cache_buf = oA.cache;

            n_tested += 1;
            if (md > max_global) max_global = md;
            if (p < SLIDING_WINDOW) {
                if (md > max_below) max_below = md;
            } else if (md > 0 and first_nonzero_p < 0) {
                first_nonzero_p = p;
            }
            log.info("  step {d} (p={d}) max_abs(logits correct vs causal) = {e}", .{ step_idx, p, md });
        }
        step_buf.deinit();
    }
    deinitCache(&cache_buf);
    mem_probe.logMem(io, "post-run");

    log.info("=== VERDICT NON-VACUITÉ (LOGITS) ===", .{});
    log.info("steps testés (double fwd) = {d} ; max_abs global = {e}", .{ n_tested, max_global });
    log.info("max_abs pour p<{d} (sanity, doit == 0) = {e} ; 1re diff>0 à p = {d}", .{ SLIDING_WINDOW, max_below, first_nonzero_p });

    if (max_below != 0) {
        log.warn("ANOMALIE : max_abs != 0 pour p<{d} alors que bande==causal → machinerie suspecte (investiguer).", .{SLIDING_WINDOW});
    }
    if (max_global > 1.0e-6) {
        log.info("NON-VACUITÉ PROUVÉE — corrompre masks_sliding (bande→causal) CHANGE les logits (max_abs={e} à partir de p={d}) → le masque sliding EST réellement consommé par le moteur.", .{ max_global, first_nonzero_p });
    } else {
        log.err("VACUITÉ DÉTECTÉE — logits IDENTIQUES malgré masque corrompu (max_abs={e}) → le masque sliding N'EST PAS consommé (ou aliasing inopérant). À corriger.", .{max_global});
        return error.Vacuity;
    }
}
