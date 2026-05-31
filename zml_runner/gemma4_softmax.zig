// P5.2.E.softmax — ZML softmax only : reader layer 15 (sliding) x KV layer 13 (sliding).
//
// Objectif : valider la transformation `scores_masked -> probs` côté ZML, limitée au
// softmax sur l'axe .k (APRÈS QK scores de E.1 et masque de E.mask, AVANT context).
// Comparer byte-équivalent (~1e-5 attendu) contre l'oracle PyTorch fp32 `probs`
// (= torch.softmax(scores_masked, dim=-1, fp32)) produit en P5.2.E.0.
//
// Pipeline ZML strict (cf zml/nn.zig sdpa L1112 : attn_weights.convert(.f32).softmax(.k)) :
//   scores_masked [.b=1, .h=8, .q=4, .k=4]   (input, scores + masque causal additif finfo.min)
//   probs = scores_masked.softmax(.k)         [.b=1, .h=8, .q=4, .k=4]
//
// `Tensor.softmax` (tensor.zig L1369) soustrait le max par ligne (stable), convertit en
// f32, exp, normalise, et renvoie 0 pour une ligne entièrement masquée (-inf). Ici aucune
// ligne n'est full-masquée (q0 voit toujours k0) -> comportement identique à torch.softmax.
//
// Indépendance de l'oracle : `probs` de référence vient de torch.softmax (E.0) ; ce runner
// utilise l'implémentation ZML native. Aucun code partagé — seul le contrat numérique l'est.
//
// Interdits stricts P5.2.E.softmax :
//   - context (dot avec V)
//   - toute opération sur V
//   - masque réel S=8/window=3 (testé en E.mask, fermé)
//   - layer 14 (full attention, p-RoPE proportional)
//   - softcap d'attention
//   - scaling 1/sqrt(head_dim)
//
// CLI : gemma4_softmax <path-to-p5_2_esoftmax_layer15_kv13.safetensors>

const std = @import("std");
const log = std.log;

const zml = @import("zml");
const stdx = zml.stdx;

pub const std_options: std.Options = .{
    .log_level = .info,
};

// Invariants P5.2.E.0/E.softmax (cf manifest p5_2_esoftmax_layer15_kv13_manifest.json).
const B: i64 = 1;
const NH: i64 = 8; // n query heads (reader layer 15, post repeat_kv)
const SQ: i64 = 4; // query positions (layer 15)
const SK: i64 = 4; // key positions (layer 13)

// Tolerances.
const SOFTMAX_TOLERANCE: f32 = 1.0e-4; // vs oracle probs
const SUM_TOLERANCE: f32 = 1.0e-5; // |sum(probs, .k) - 1|
const FUTURE_TOLERANCE: f32 = 1.0e-9; // proba sur position future masquée
const FLAT_LEN: usize = @intCast(B * NH * SQ * SK); // 128

// Strides du tenseur probs [b, h, q, k] (row-major).
const STRIDE_B: usize = @intCast(NH * SQ * SK); // 128
const STRIDE_H: usize = @intCast(SQ * SK); // 16
const STRIDE_Q: usize = @intCast(SK); // 4

// Fixed-point oracle (probs fp32 extrait du fixture E.0, cf logs/21_*.log).
// flat_offset = h * STRIDE_H + q * STRIDE_Q  (b=0).
const ProbBlock = struct {
    label: []const u8,
    flat_offset: usize,
    expected: []const f32,
};

const PROB_BLOCKS = [_]ProbBlock{
    .{
        .label = "probs[0,0,0,:4]",
        .flat_offset = 0, // h=0, q=0 -> q0 ne voit que k0 (causal) -> [1,0,0,0]
        .expected = &.{ 1.0000000000, 0.0000000000, 0.0000000000, 0.0000000000 },
    },
    .{
        .label = "probs[0,0,3,:4]",
        .flat_offset = 12, // h=0, q=3 -> 3*4
        .expected = &.{ 0.0098362984, 0.0352034718, 0.7669014931, 0.1880586743 },
    },
    .{
        .label = "probs[0,7,3,:4]",
        .flat_offset = 124, // h=7, q=3 -> 7*16 + 3*4
        .expected = &.{ 0.0436253324, 0.5618983507, 0.2381305397, 0.1563457400 },
    },
};

/// Fixture E.softmax chargée depuis p5_2_esoftmax_layer15_kv13.safetensors.
/// 2 tenseurs : scores_masked (input), probs (oracle de comparaison).
const SoftmaxFixture = struct {
    scores_masked: zml.Tensor,
    probs_oracle: zml.Tensor,

    pub fn init(store: zml.io.TensorStore.View) SoftmaxFixture {
        return .{
            .scores_masked = store.createTensor(
                "scores_masked",
                .{ .b, .h, .q, .k },
                null,
            ),
            .probs_oracle = store.createTensor(
                "probs",
                .{ .b, .h, .q, .k },
                null,
            ),
        };
    }

    pub fn load(
        self: *const SoftmaxFixture,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *const zml.io.TensorStore,
        shardings: []const zml.sharding.Sharding,
    ) !zml.Bufferized(SoftmaxFixture) {
        return zml.io.load(SoftmaxFixture, self, allocator, io, platform, store, .{
            .shardings = shardings,
            .parallelism = 1,
            .dma_chunks = 1,
            .dma_chunk_size = 16 * 1024 * 1024,
        });
    }

    pub fn unloadBuffers(self: *zml.Bufferized(SoftmaxFixture)) void {
        self.scores_masked.deinit();
        self.probs_oracle.deinit();
    }

    /// Forward E.softmax : softmax sur l'axe .k uniquement. Pas de mask (déjà dans
    /// scores_masked), pas de context, pas de V. Convention sdpa ZML : softmax(.k) en f32.
    pub fn forward(self: SoftmaxFixture) zml.Tensor {
        return self.scores_masked.softmax(.k);
    }
};

pub fn main(init: std.process.Init) !void {
    const arena = init.arena;
    const allocator = init.gpa;
    const io = init.io;

    const process_args = try init.minimal.args.toSlice(arena.allocator());
    if (process_args.len < 2) {
        log.err("Usage: gemma4_softmax <path-to-p5_2_esoftmax_layer15_kv13.safetensors>", .{});
        return error.MissingArgument;
    }
    const fixture_path = process_args[1];

    log.info("P5.2.E.softmax — ZML softmax only (reader layer 15 x KV layer 13, axe .k, fp32, no context/V)", .{});
    log.info("Loading fixture from {s}", .{fixture_path});

    var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, fixture_path);
    defer registry.deinit();

    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();

    const model: SoftmaxFixture = .init(store.view());

    log.info("Symbolic shapes:", .{});
    log.info("  scores_masked : {f}", .{model.scores_masked});
    log.info("  probs_oracle  : {f}", .{model.probs_oracle});

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);

    const replicated_sharding = try zml.sharding.replicatedSharding(platform);

    log.info("Materializing buffers...", .{});
    var buffers = try model.load(arena.allocator(), io, platform, &store, &.{replicated_sharding});
    defer SoftmaxFixture.unloadBuffers(&buffers);
    log.info("Buffers loaded (2 tensors).", .{});

    log.info("Compiling forward (softmax over .k, fp32)...", .{});
    var exe = try platform.compile(
        allocator,
        io,
        model,
        .forward,
        .{},
        .{ .shardings = &.{replicated_sharding} },
    );
    defer exe.deinit();

    var args = try exe.args(allocator);
    defer args.deinit(allocator);

    var results = try exe.results(allocator);
    defer results.deinit(allocator);

    args.set(.{buffers});
    exe.call(args, &results);

    var result: zml.Buffer = results.get(zml.Buffer);
    defer result.deinit();

    log.info("Forward result shape: {f} (expected [b=1, h=8, q=4, k=4])", .{result.shape()});

    var slice = try result.toSliceAlloc(allocator, io);
    defer slice.free(allocator);
    const data = slice.items(f32);

    if (data.len != FLAT_LEN) {
        log.err("length mismatch: data={d} expected={d}", .{ data.len, FLAT_LEN });
        return error.SoftmaxLengthMismatch;
    }

    // === Checks de distribution sur la sortie ZML (probs valide ?) ===
    log.info("Distribution checks (ZML probs):", .{});
    var max_sum_err: f32 = 0.0;
    var future_prob_max: f32 = 0.0;
    var has_nan_inf = false;
    {
        var h: usize = 0;
        while (h < @as(usize, @intCast(NH))) : (h += 1) {
            var q: usize = 0;
            while (q < @as(usize, @intCast(SQ))) : (q += 1) {
                const row_off = h * STRIDE_H + q * STRIDE_Q;
                var row_sum: f32 = 0.0;
                var k: usize = 0;
                while (k < @as(usize, @intCast(SK))) : (k += 1) {
                    const v = data[row_off + k];
                    if (std.math.isNan(v) or std.math.isInf(v)) has_nan_inf = true;
                    row_sum += v;
                    // Position future (k > q) masquée -> proba ~ 0.
                    if (k > q) {
                        const av = @abs(v);
                        if (av > future_prob_max) future_prob_max = av;
                    }
                }
                const sum_err = @abs(row_sum - 1.0);
                if (sum_err > max_sum_err) max_sum_err = sum_err;
            }
        }
    }
    log.info("  max|sum(probs, .k) - 1| = {e:.3} (tol {e:.1})", .{ max_sum_err, SUM_TOLERANCE });
    log.info("  max proba sur futur masqué = {e:.3} (tol {e:.1})", .{ future_prob_max, FUTURE_TOLERANCE });
    log.info("  NaN/Inf present = {}", .{has_nan_inf});
    if (has_nan_inf) {
        log.err("BLOCK: ZML probs contains NaN/Inf", .{});
        return error.SoftmaxNanInf;
    }
    if (max_sum_err > SUM_TOLERANCE) {
        log.err("BLOCK: rows do not sum to 1 (err {e:.3} > tol {e:.1})", .{ max_sum_err, SUM_TOLERANCE });
        return error.SoftmaxSumFailed;
    }
    if (future_prob_max > FUTURE_TOLERANCE) {
        log.err("BLOCK: attention leak on future (max {e:.3} > tol {e:.1})", .{ future_prob_max, FUTURE_TOLERANCE });
        return error.SoftmaxFutureLeak;
    }

    // === Fixed-point blocks (3 x 4 valeurs) vs oracle probs ===
    log.info("Fixed-point blocks vs oracle probs (fp32):", .{});
    var max_block: f32 = 0.0;
    for (PROB_BLOCKS) |block| {
        var block_max: f32 = 0.0;
        log.info("  Block {s} (flat_offset={d}):", .{ block.label, block.flat_offset });
        for (block.expected, 0..) |expected, i| {
            const actual = data[block.flat_offset + i];
            const diff = @abs(actual - expected);
            if (diff > block_max) block_max = diff;
            log.info("    +{d:>3}: actual={d:.8} expected={d:.8} diff={e:.3}", .{
                i, actual, expected, diff,
            });
        }
        log.info("    block max_diff: {e:.6}", .{block_max});
        if (block_max > max_block) max_block = block_max;
    }
    log.info("  -> 3 blocks max_diff: {e:.6}", .{max_block});

    // === Scan global 128 valeurs vs oracle ===
    log.info("Scanning full tensor ({d} fp32) vs oracle probs:", .{FLAT_LEN});
    var ref_slice = try buffers.probs_oracle.toSliceAlloc(allocator, io);
    defer ref_slice.free(allocator);
    const ref_data = ref_slice.items(f32);

    if (ref_data.len != FLAT_LEN) {
        log.err("length mismatch: ref={d} expected={d}", .{ ref_data.len, FLAT_LEN });
        return error.SoftmaxLengthMismatch;
    }

    var max_global: f32 = 0.0;
    var sum_abs: f64 = 0.0;
    var max_idx: usize = 0;
    for (data, ref_data, 0..) |actual, expected, i| {
        const diff = @abs(actual - expected);
        if (diff > max_global) {
            max_global = diff;
            max_idx = i;
        }
        sum_abs += @as(f64, diff);
    }
    const mean_abs = @as(f32, @floatCast(sum_abs / @as(f64, @floatFromInt(FLAT_LEN))));

    const b_idx = max_idx / STRIDE_B;
    const h_idx = (max_idx % STRIDE_B) / STRIDE_H;
    const q_idx = (max_idx % STRIDE_H) / STRIDE_Q;
    const k_idx = max_idx % STRIDE_Q;
    log.info("  -> full tensor max_abs : {e:.6} at flat_index {d} (b={d}, h={d}, q={d}, k={d})", .{
        max_global, max_idx, b_idx, h_idx, q_idx, k_idx,
    });
    log.info("  -> full tensor mean_abs: {e:.6}", .{mean_abs});

    const max_diff = if (max_global > max_block) max_global else max_block;
    log.info("softmax global max_diff: {e:.6} (tolerance {e:.1})", .{ max_diff, SOFTMAX_TOLERANCE });
    log.info("  Expected << 1e-4 (softmax stable ; jitter QK ~2.4e-6 de E.1 propagé, peu amplifié)", .{});

    if (max_diff > SOFTMAX_TOLERANCE) {
        log.err("BLOCK: softmax max_diff exceeds tolerance", .{});
        return error.SoftmaxFailed;
    }
    log.info("P5.2.E.softmax PASS: ZML softmax reader layer 15 x KV layer 13 validated vs PyTorch oracle", .{});
    log.info("  (softmax over .k, fp32, sum=1, no future leak, no context, no V, no layer 14)", .{});
}
