// P5.2.C.2 — ZML q_norm (reshape + RMSNorm + mul), reader layer 15 (sliding).
//
// Objectif : etendre P5.2.C.1 (q_proj seul, PASS) avec le pipeline q_norm
// pattern Llama (`normalized.mul(weight)`, PAS Qwen `(1+weight)`).
// Comparer contre l'oracle PyTorch fp32 `q_after_norm` du fixture C.0.
//
// Pipeline ZML strict :
//   q_after_proj = hidden_input.dot(q_proj_weight, .h)            // [.b, .s, .o]   reuse C.1
//   q_4d         = q_after_proj.reshape({1,4,8,256})              // perd les tags
//                    .withTags(.{.b, .s, .n, .d})                  // re-tag (piège #1)
//   q_normalized = zml.nn.rmsNorm(q_4d, .d, RMS_EPS)
//   q_after_norm = q_normalized.mul(q_norm_weight.broad(q_normalized.shape()))
//
// Interdits stricts P5.2.C.2 :
//   - RoPE (apply_rotary_pos_emb)
//   - transpose [B, n_heads, S, head_dim]
//   - K/V projection
//   - attention scores / matmul QK / softmax
//   - cache / sliding mask
//
// CLI : gemma4_q_norm <path-to-q_only_reader_layer15.safetensors>

const std = @import("std");
const log = std.log;

const zml = @import("zml");
const stdx = zml.stdx;

pub const std_options: std.Options = .{
    .log_level = .info,
};

// Invariants P5.2.C.0 (cf manifest q_only_reader_layer15_manifest.json).
const B: i64 = 1;
const S: i64 = 4;
const H: i64 = 1536;
const O: i64 = 2048; // n_heads (8) * head_dim (256)
const N: i64 = 8; // num_attention_heads
const D: i64 = 256; // head_dim
const RMS_EPS: f32 = 1.0e-6;

// Tolerance / sanity.
const Q_NORM_TOLERANCE: f32 = 1.0e-4;
const Q_NORM_FLAT_LEN: usize = @intCast(B * S * N * D); // 8192

// Fixed-point oracle extraits depuis fixture C.0 :
// q_after_norm[0, {0,3}, {0,7}, :8] en fp32 (PyTorch RMSNorm Gemma4 = pattern Llama).
// flat_offset (row-major, shape [1,4,8,256], strides (8192, 2048, 256, 1)) :
//   [0,s,n,:8] -> s*2048 + n*256.
const QNormBlock = struct {
    label: []const u8,
    flat_offset: usize,
    expected: []const f32,
};

const Q_NORM_BLOCKS = [_]QNormBlock{
    .{
        .label = "A [0,0,0,:8]",
        .flat_offset = 0,
        .expected = &.{
            0.2076200247, 0.2199150771, 2.0028545856, 1.1477159262,
            -0.2349513322, 1.7933709621, 4.2625265121, 0.3524217606,
        },
    },
    .{
        .label = "B [0,0,7,:8]",
        .flat_offset = 1792,
        .expected = &.{
            -0.4908642471, 0.1689578891, 2.2943143845, -2.0207083225,
            -0.0024840503, 2.9049944878, 1.7889719009, -0.3205637634,
        },
    },
    .{
        .label = "C [0,3,0,:8]",
        .flat_offset = 6144,
        .expected = &.{
            -0.0925704613, -0.5016254187, 3.3350615501, 4.4669680595,
            0.1783346832, 1.9709883928, 4.1253089905, 0.0569067970,
        },
    },
    .{
        .label = "D [0,3,7,:8]",
        .flat_offset = 7936,
        .expected = &.{
            0.2608506382, 0.4000130594, -1.2275550365, 4.0930047035,
            0.0877366289, -5.2869048119, 0.2385167331, -0.1336933672,
        },
    },
};

/// Fixture C.0 — 4 tenseurs consommes en C.2 (sur 9 du fixture P5.2.C.0).
/// Les 5 autres (rotary_cos/sin, q_after_proj, q_after_rope, q_final) sont
/// ignores en C.2 (mais q_after_proj a servi d'oracle en C.1).
const QNormFixture = struct {
    hidden_input: zml.Tensor,
    q_proj_weight: zml.Tensor,
    q_norm_weight: zml.Tensor,
    q_after_norm_oracle: zml.Tensor,

    pub fn init(store: zml.io.TensorStore.View) QNormFixture {
        return .{
            .hidden_input = store.createTensor(
                "hidden_input",
                .{ .b, .s, .h },
                null,
            ),
            .q_proj_weight = store.createTensor(
                "q_proj_weight",
                .{ .o, .h },
                null,
            ),
            .q_norm_weight = store.createTensor(
                "q_norm_weight",
                .{.d},
                null,
            ),
            .q_after_norm_oracle = store.createTensor(
                "q_after_norm",
                .{ .b, .s, .n, .d },
                null,
            ),
        };
    }

    pub fn load(
        self: *const QNormFixture,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *const zml.io.TensorStore,
        shardings: []const zml.sharding.Sharding,
    ) !zml.Bufferized(QNormFixture) {
        return zml.io.load(QNormFixture, self, allocator, io, platform, store, .{
            .shardings = shardings,
            .parallelism = 1,
            .dma_chunks = 1,
            .dma_chunk_size = 16 * 1024 * 1024,
        });
    }

    pub fn unloadBuffers(self: *zml.Bufferized(QNormFixture)) void {
        self.hidden_input.deinit();
        self.q_proj_weight.deinit();
        self.q_norm_weight.deinit();
        self.q_after_norm_oracle.deinit();
    }

    /// Forward C.2 : q_proj (C.1 PASS) + reshape + withTags + rmsNorm(.d) + mul(weight.broad).
    /// Resultat : Tensor({b=1, s=4, n=8, d=256, f32}).
    pub fn forward(self: QNormFixture) zml.Tensor {
        // Reuse C.1 path.
        const q_after_proj = self.hidden_input.dot(self.q_proj_weight, .h);

        // Reshape perd les tags -> re-tag (piège ZML #1 capitalise P4.4.2 Gate H).
        const q_4d = q_after_proj
            .reshape(.{ B, S, N, D })
            .withTags(.{ .b, .s, .n, .d });

        // RMSNorm sur axe .d (head_dim) — Gemma4RMSNorm = pattern Llama
        // (normalized * weight, PAS Qwen (1+weight)).
        const q_normalized = zml.nn.rmsNorm(q_4d, .d, RMS_EPS);
        return q_normalized.mul(
            self.q_norm_weight.broad(q_normalized.shape()),
        );
    }
};

pub fn main(init: std.process.Init) !void {
    const arena = init.arena;
    const allocator = init.gpa;
    const io = init.io;

    const process_args = try init.minimal.args.toSlice(arena.allocator());
    if (process_args.len < 2) {
        log.err("Usage: gemma4_q_norm <path-to-q_only_reader_layer15.safetensors>", .{});
        return error.MissingArgument;
    }
    const fixture_path = process_args[1];

    log.info("P5.2.C.2 — ZML q_norm (q_proj + reshape + rmsNorm + mul), reader layer 15 sliding", .{});
    log.info("Loading fixture from {s}", .{fixture_path});

    var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, fixture_path);
    defer registry.deinit();

    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();

    const model: QNormFixture = .init(store.view());

    log.info("Symbolic shapes:", .{});
    log.info("  hidden_input        : {f}", .{model.hidden_input});
    log.info("  q_proj_weight       : {f}", .{model.q_proj_weight});
    log.info("  q_norm_weight       : {f}", .{model.q_norm_weight});
    log.info("  q_after_norm_oracle : {f}", .{model.q_after_norm_oracle});

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);

    const replicated_sharding = try zml.sharding.replicatedSharding(platform);

    log.info("Materializing buffers...", .{});
    var buffers = try model.load(arena.allocator(), io, platform, &store, &.{replicated_sharding});
    defer QNormFixture.unloadBuffers(&buffers);
    log.info("Buffers loaded (4 tensors).", .{});

    log.info("Compiling forward (q_proj + reshape + rmsNorm + mul, no RoPE/transpose/K/V)...", .{});
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

    log.info("Forward result shape: {f}", .{result.shape()});

    var slice = try result.toSliceAlloc(allocator, io);
    defer slice.free(allocator);
    const data = slice.items(f32);

    // === 4 fixed-point blocks (extraits oracle) ===
    log.info("Fixed-point blocks vs oracle q_after_norm (fp32):", .{});
    var max_block: f32 = 0.0;
    for (Q_NORM_BLOCKS) |block| {
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
    log.info("  -> 4 blocks max_diff: {e:.6}", .{max_block});

    // === Scan global 8192 valeurs vs oracle ===
    log.info("Scanning full tensor ({d} fp32) vs oracle q_after_norm:", .{Q_NORM_FLAT_LEN});
    var ref_slice = try buffers.q_after_norm_oracle.toSliceAlloc(allocator, io);
    defer ref_slice.free(allocator);
    const ref_data = ref_slice.items(f32);

    if (ref_data.len != Q_NORM_FLAT_LEN or data.len != Q_NORM_FLAT_LEN) {
        log.err("length mismatch: ref={d} data={d} expected={d}", .{ ref_data.len, data.len, Q_NORM_FLAT_LEN });
        return error.QNormLengthMismatch;
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
    const mean_abs = @as(f32, @floatCast(sum_abs / @as(f64, @floatFromInt(Q_NORM_FLAT_LEN))));

    // Decompose max_idx into (s, n, d).
    const stride_s: usize = 2048;
    const stride_n: usize = 256;
    const s_idx = max_idx / stride_s;
    const n_idx = (max_idx % stride_s) / stride_n;
    const d_idx = max_idx % stride_n;

    log.info("  -> full tensor max_abs : {e:.6} at flat_index {d} (s={d}, n={d}, d={d})", .{
        max_global, max_idx, s_idx, n_idx, d_idx,
    });
    log.info("  -> full tensor mean_abs: {e:.6}", .{mean_abs});

    const max_diff = if (max_global > max_block) max_global else max_block;
    log.info("q_norm global max_diff: {e:.6} (tolerance {e:.1})", .{ max_diff, Q_NORM_TOLERANCE });
    log.info("  Expected ~ C.1 residual + RMSNorm normalization (probably <= 1e-4)", .{});

    if (max_diff > Q_NORM_TOLERANCE) {
        log.err("BLOCK: q_norm max_diff exceeds tolerance", .{});
        log.err("  suspects: (1) wrong RMSNorm axis, (2) (1+weight) instead of pure weight,", .{});
        log.err("            (3) reshape without withTags, (4) mul without explicit broad", .{});
        return error.QNormFailed;
    }
    log.info("P5.2.C.2 PASS: ZML q_norm reader layer 15 validated vs PyTorch oracle", .{});
    log.info("  (no RoPE, no transpose, no K/V, no attention)", .{});
}
