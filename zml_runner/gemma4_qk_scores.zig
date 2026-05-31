// P5.2.E.1 — ZML QK scores only : reader layer 15 (sliding) x KV layer 13 (sliding).
//
// Objectif : valider le PREMIER calcul d'attention en ZML, limité aux scores
// bruts Q·Kᵀ (AVANT masque / softmax / context). Comparer byte-equivalent
// (~1e-5 attendu) contre l'oracle PyTorch fp32 `scores_raw` produit en P5.2.E.0.
//
// Pipeline ZML strict (cf zml/nn.zig sdpa, mais SANS scaling 1/sqrt(hd)) :
//   q_final [.b=1, .h=8, .q=4, .hd=256]   (reader layer 15, après q_norm + RoPE)
//   k_final [.b=1, .h=1, .k=4, .hd=256]   (writer layer 13, après k_norm + RoPE, cache layout)
//   GQA  : q_split = q.splitAxis(.h, .{ .h = k.dim(.h)=1, .hq = .auto=8 })  (convention Llama/sdpa)
//   dot  : scores  = q_split.dot(k, .hd)          -> [.b, .h=1, .hq=8, .q=4, .k=4]
//   merge: scores  = scores.merge(.{ .h = .{ .h, .hq } })  -> [.b, .h=8, .q=4, .k=4]
//   order: scores  = scores.transpose(.{ .b, .h, .q, .k })  pour matcher l'oracle [b,h,sq,sk]
//
// Gemma 4 : scaling = 1.0 (PAS 1/sqrt(head_dim) — la norm passe par q_norm/k_norm),
// PAS de softcap d'attention. On ne multiplie donc PAS par 1/sqrt(hd).
//
// Fixture consommée : `fixtures/p5_2_e1_qk_scores_layer15_kv13.safetensors`
// (3 tenseurs : q_final, k_final, scores_raw oracle), slim-export depuis le .pt E.0.
//
// Interdits stricts P5.2.E.1 :
//   - masque (causal / sliding)
//   - softmax
//   - context (dot avec V)
//   - layer 14 (full attention, p-RoPE proportional)
//   - softcap d'attention
//   - scaling 1/sqrt(head_dim)
//
// CLI : gemma4_qk_scores <path-to-p5_2_e1_qk_scores_layer15_kv13.safetensors>

const std = @import("std");
const log = std.log;

const zml = @import("zml");
const stdx = zml.stdx;

pub const std_options: std.Options = .{
    .log_level = .info,
};

// Invariants P5.2.E.0/E.1 (cf manifest p5_2_e1_qk_scores_layer15_kv13_manifest.json).
const B: i64 = 1;
const NH: i64 = 8; // n query heads (reader layer 15)
const NKV: i64 = 1; // n kv heads (writer layer 13)
const SQ: i64 = 4; // query positions (layer 15)
const SK: i64 = 4; // key positions (layer 13)
const HD: i64 = 256; // head_dim

// Tolerance / sanity expectations.
const QK_TOLERANCE: f32 = 1.0e-4;
const QK_FLAT_LEN: usize = @intCast(B * NH * SQ * SK); // 128

// Strides du tenseur scores [b, h, q, k] (row-major).
const STRIDE_B: usize = @intCast(NH * SQ * SK); // 128
const STRIDE_H: usize = @intCast(SQ * SK); // 16
const STRIDE_Q: usize = @intCast(SK); // 4

// Fixed-point oracle (scores_raw fp32 extrait du fixture E.0, cf logs/21_*.log).
// flat_offset = h * STRIDE_H + q * STRIDE_Q  (b=0).
const QkBlock = struct {
    label: []const u8,
    flat_offset: usize,
    expected: []const f32,
};

const QK_BLOCKS = [_]QkBlock{
    .{
        .label = "scores_raw[0,0,0,:4]",
        .flat_offset = 0, // h=0, q=0
        .expected = &.{ 0.6519736052, -0.4114219546, -1.1377084255, -1.0763149261 },
    },
    .{
        .label = "scores_raw[0,0,3,:4]",
        .flat_offset = 12, // h=0, q=3 -> 3*4
        .expected = &.{ -2.5638093948, -1.2887440920, 1.7924696207, 0.3868653178 },
    },
    .{
        .label = "scores_raw[0,7,3,:4]",
        .flat_offset = 124, // h=7, q=3 -> 7*16 + 3*4
        .expected = &.{ -2.0650737286, 0.4906092286, -0.3678926528, -0.7886418700 },
    },
};

/// Fixture E.1 chargée depuis p5_2_e1_qk_scores_layer15_kv13.safetensors.
/// 3 tenseurs : q_final (input), k_final (input), scores_raw (oracle de comparaison).
const QkScoresFixture = struct {
    q_final: zml.Tensor,
    k_final: zml.Tensor,
    scores_raw_oracle: zml.Tensor,

    pub fn init(store: zml.io.TensorStore.View) QkScoresFixture {
        return .{
            // q_final [1,8,4,256] : dim positions taggée .q directement (reader layer 15).
            .q_final = store.createTensor(
                "q_final",
                .{ .b, .h, .q, .hd },
                null,
            ),
            // k_final [1,1,4,256] : kv head taggé .h (size 1), positions .k (writer layer 13).
            .k_final = store.createTensor(
                "k_final",
                .{ .b, .h, .k, .hd },
                null,
            ),
            // scores_raw oracle [1,8,4,4] : [b, h, q(=sq), k(=sk)].
            .scores_raw_oracle = store.createTensor(
                "scores_raw",
                .{ .b, .h, .q, .k },
                null,
            ),
        };
    }

    pub fn load(
        self: *const QkScoresFixture,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *const zml.io.TensorStore,
        shardings: []const zml.sharding.Sharding,
    ) !zml.Bufferized(QkScoresFixture) {
        return zml.io.load(QkScoresFixture, self, allocator, io, platform, store, .{
            .shardings = shardings,
            .parallelism = 1,
            .dma_chunks = 1,
            .dma_chunk_size = 16 * 1024 * 1024,
        });
    }

    pub fn unloadBuffers(self: *zml.Bufferized(QkScoresFixture)) void {
        self.q_final.deinit();
        self.k_final.deinit();
        self.scores_raw_oracle.deinit();
    }

    /// Forward E.1 : scores bruts Q·Kᵀ uniquement (scaling 1.0, GQA via split des têtes Q).
    ///   q_split = q.splitAxis(.h, .{ .h = k.dim(.h), .hq = .auto })  [.b,.h=1,.hq=8,.q,.hd]
    ///   scores  = q_split.dot(k, .hd)                                [.b,.h=1,.hq=8,.q,.k]
    ///   scores  = scores.merge(.{ .h = .{ .h, .hq } })               [.b,.h=8,.q,.k]
    ///   scores  = scores.transpose(.{ .b, .h, .q, .k })              ordre physique [b,h,q,k]
    /// Pas de mul par 1/sqrt(hd) : Gemma4 scaling = 1.0. Pas de mask/softmax/context.
    pub fn forward(self: QkScoresFixture) zml.Tensor {
        const q_split = self.q_final.splitAxis(.h, .{ .h = self.k_final.dim(.h), .hq = .auto });
        const scores = q_split.dot(self.k_final, .hd);
        const merged = scores.merge(.{ .h = .{ .h, .hq } });
        return merged.transpose(.{ .b, .h, .q, .k });
    }
};

pub fn main(init: std.process.Init) !void {
    const arena = init.arena;
    const allocator = init.gpa;
    const io = init.io;

    const process_args = try init.minimal.args.toSlice(arena.allocator());
    if (process_args.len < 2) {
        log.err("Usage: gemma4_qk_scores <path-to-p5_2_e1_qk_scores_layer15_kv13.safetensors>", .{});
        return error.MissingArgument;
    }
    const fixture_path = process_args[1];

    log.info("P5.2.E.1 — ZML QK scores only (reader layer 15 x KV layer 13, scaling 1.0, no mask/softmax)", .{});
    log.info("Loading fixture from {s}", .{fixture_path});

    var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, fixture_path);
    defer registry.deinit();

    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();

    const model: QkScoresFixture = .init(store.view());

    log.info("Symbolic shapes:", .{});
    log.info("  q_final           : {f}", .{model.q_final});
    log.info("  k_final           : {f}", .{model.k_final});
    log.info("  scores_raw_oracle : {f}", .{model.scores_raw_oracle});

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);

    const replicated_sharding = try zml.sharding.replicatedSharding(platform);

    log.info("Materializing buffers...", .{});
    var buffers = try model.load(arena.allocator(), io, platform, &store, &.{replicated_sharding});
    defer QkScoresFixture.unloadBuffers(&buffers);
    log.info("Buffers loaded (3 tensors).", .{});

    log.info("Compiling forward (splitAxis GQA -> dot .hd -> merge -> transpose, scaling 1.0)...", .{});
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

    // === Fixed-point blocks (3 x 4 valeurs) vs oracle scores_raw ===
    log.info("Fixed-point blocks vs oracle scores_raw (fp32):", .{});
    var max_block: f32 = 0.0;
    for (QK_BLOCKS) |block| {
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
    log.info("Scanning full tensor ({d} fp32) vs oracle scores_raw:", .{QK_FLAT_LEN});
    var ref_slice = try buffers.scores_raw_oracle.toSliceAlloc(allocator, io);
    defer ref_slice.free(allocator);
    const ref_data = ref_slice.items(f32);

    if (ref_data.len != QK_FLAT_LEN or data.len != QK_FLAT_LEN) {
        log.err("length mismatch: ref={d} data={d} expected={d}", .{ ref_data.len, data.len, QK_FLAT_LEN });
        return error.QkLengthMismatch;
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
    const mean_abs = @as(f32, @floatCast(sum_abs / @as(f64, @floatFromInt(QK_FLAT_LEN))));

    const b_idx = max_idx / STRIDE_B;
    const h_idx = (max_idx % STRIDE_B) / STRIDE_H;
    const q_idx = (max_idx % STRIDE_H) / STRIDE_Q;
    const k_idx = max_idx % STRIDE_Q;
    log.info("  -> full tensor max_abs : {e:.6} at flat_index {d} (b={d}, h={d}, q={d}, k={d})", .{
        max_global, max_idx, b_idx, h_idx, q_idx, k_idx,
    });
    log.info("  -> full tensor mean_abs: {e:.6}", .{mean_abs});

    const max_diff = if (max_global > max_block) max_global else max_block;
    log.info("qk_scores global max_diff: {e:.6} (tolerance {e:.1})", .{ max_diff, QK_TOLERANCE });
    log.info("  Expected ~1e-5 (matmul QK PJRT-CPU Eigen-like vs PyTorch BLAS ; plancher jitter fp32 ~5e-7)", .{});

    if (max_diff > QK_TOLERANCE) {
        log.err("BLOCK: qk_scores max_diff exceeds tolerance", .{});
        return error.QkScoresFailed;
    }
    log.info("P5.2.E.1 PASS: ZML QK scores reader layer 15 x KV layer 13 validated vs PyTorch oracle", .{});
    log.info("  (scaling 1.0, GQA split, no mask, no softmax, no context, no layer 14)", .{});
}
