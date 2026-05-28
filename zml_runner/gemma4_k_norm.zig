// P5.2.D.3 — ZML k_norm (k_proj + reshape + RMSNorm + mul), producer/writer
// layer 13 (sliding).
//
// Objectif : etendre P5.2.D.1 (k_proj seul, PASS) avec le pipeline k_norm
// pattern Llama (`normalized.mul(weight)`, PAS Qwen `(1+weight)`).
// Comparer contre l'oracle PyTorch fp32 `k_after_norm` du fixture D.0.
//
// Pipeline ZML strict :
//   k_after_proj = hidden_input.dot(k_proj_weight, .h)           // [.b, .s, .kv]  reuse D.1
//   k_4d         = k_after_proj.reshape({1,4,1,256})              // perd les tags
//                    .withTags(.{.b, .s, .kvh, .d})               // re-tag (piège #1)
//   k_normalized = zml.nn.rmsNorm(k_4d, .d, RMS_EPS)
//   k_after_norm = k_normalized.mul(k_norm_weight.broad(k_normalized.shape()))
//
// Diff vs C.2 q_norm :
//   - n_kv = 1 (vs n_heads = 8) -> axe head_count taggue .kvh (vs .n)
//   - shape sortie [1,4,1,256] (vs [1,4,8,256])
//   - V non normé en Gemma 4 (v_norm absent du checkpoint, hors scope D.3)
//
// Interdits stricts P5.2.D.3 :
//   - v_proj
//   - v_norm
//   - RoPE (apply_rotary_pos_emb)
//   - transpose [B, n_kv, S, head_dim]
//   - cache / sliding mask
//   - attention scores / matmul QK / softmax
//
// CLI : gemma4_k_norm <path-to-p5_2_d3_k_norm_layer13.safetensors>

const std = @import("std");
const log = std.log;

const zml = @import("zml");
const stdx = zml.stdx;

pub const std_options: std.Options = .{
    .log_level = .info,
};

// Invariants P5.2.D.0 (cf manifest p5_2_d0_kv_oracle_layer13_manifest.json).
const B: i64 = 1;
const S: i64 = 4;
const H: i64 = 1536;
const KV: i64 = 256; // n_kv (1) * head_dim (256)
const NKV: i64 = 1; // num_key_value_heads
const D: i64 = 256; // head_dim
const RMS_EPS: f32 = 1.0e-6;

// Tolerance / sanity.
const K_NORM_TOLERANCE: f32 = 1.0e-4;
const K_NORM_FLAT_LEN: usize = @intCast(B * S * NKV * D); // 1024

// Fixed-point blocks (reported per-position vs oracle, no hardcoded expected) :
// k_after_norm[0, {0,3}, 0, :8] en fp32 (PyTorch Gemma4RMSNorm = pattern Llama).
// shape [1,4,1,256], strides (1024, 256, 256, 1).
// [0,s,0,:8] -> flat = s * 256.
const KNormBlock = struct {
    label: []const u8,
    flat_offset: usize,
    width: usize,
};

const K_NORM_BLOCKS = [_]KNormBlock{
    .{ .label = "A [0,0,0,:8]", .flat_offset = 0, .width = 8 },
    .{ .label = "B [0,3,0,:8]", .flat_offset = 768, .width = 8 },
};

/// Fixture D.3 slim (4 tenseurs) chargee depuis safetensors.
const KNormFixture = struct {
    hidden_input: zml.Tensor,
    k_proj_weight: zml.Tensor,
    k_norm_weight: zml.Tensor,
    k_after_norm_oracle: zml.Tensor,

    pub fn init(store: zml.io.TensorStore.View) KNormFixture {
        return .{
            .hidden_input = store.createTensor(
                "hidden_input",
                .{ .b, .s, .h },
                null,
            ),
            .k_proj_weight = store.createTensor(
                "k_proj_weight",
                .{ .kv, .h },
                null,
            ),
            .k_norm_weight = store.createTensor(
                "k_norm_weight",
                .{.d},
                null,
            ),
            .k_after_norm_oracle = store.createTensor(
                "k_after_norm",
                .{ .b, .s, .kvh, .d },
                null,
            ),
        };
    }

    pub fn load(
        self: *const KNormFixture,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *const zml.io.TensorStore,
        shardings: []const zml.sharding.Sharding,
    ) !zml.Bufferized(KNormFixture) {
        return zml.io.load(KNormFixture, self, allocator, io, platform, store, .{
            .shardings = shardings,
            .parallelism = 1,
            .dma_chunks = 1,
            .dma_chunk_size = 16 * 1024 * 1024,
        });
    }

    pub fn unloadBuffers(self: *zml.Bufferized(KNormFixture)) void {
        self.hidden_input.deinit();
        self.k_proj_weight.deinit();
        self.k_norm_weight.deinit();
        self.k_after_norm_oracle.deinit();
    }

    /// Forward D.3 : k_proj (D.1 PASS) + reshape + withTags + rmsNorm(.d) + mul(weight.broad).
    /// Resultat : Tensor({b=1, s=4, kvh=1, d=256, f32}).
    pub fn forward(self: KNormFixture) zml.Tensor {
        // Reuse D.1 path.
        const k_after_proj = self.hidden_input.dot(self.k_proj_weight, .h);

        // Reshape perd les tags -> re-tag (piège ZML #1 capitalise P4.4.2 Gate H).
        const k_4d = k_after_proj
            .reshape(.{ B, S, NKV, D })
            .withTags(.{ .b, .s, .kvh, .d });

        // RMSNorm sur axe .d (head_dim) — Gemma4RMSNorm = pattern Llama
        // (normalized * weight, PAS Qwen (1+weight)).
        const k_normalized = zml.nn.rmsNorm(k_4d, .d, RMS_EPS);
        return k_normalized.mul(
            self.k_norm_weight.broad(k_normalized.shape()),
        );
    }
};

pub fn main(init: std.process.Init) !void {
    const arena = init.arena;
    const allocator = init.gpa;
    const io = init.io;

    const process_args = try init.minimal.args.toSlice(arena.allocator());
    if (process_args.len < 2) {
        log.err("Usage: gemma4_k_norm <path-to-p5_2_d3_k_norm_layer13.safetensors>", .{});
        return error.MissingArgument;
    }
    const fixture_path = process_args[1];

    log.info("P5.2.D.3 — ZML k_norm (k_proj + reshape + rmsNorm + mul), producer layer 13 sliding", .{});
    log.info("Loading fixture from {s}", .{fixture_path});

    var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, fixture_path);
    defer registry.deinit();

    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();

    const model: KNormFixture = .init(store.view());

    log.info("Symbolic shapes:", .{});
    log.info("  hidden_input        : {f}", .{model.hidden_input});
    log.info("  k_proj_weight       : {f}", .{model.k_proj_weight});
    log.info("  k_norm_weight       : {f}", .{model.k_norm_weight});
    log.info("  k_after_norm_oracle : {f}", .{model.k_after_norm_oracle});

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);

    const replicated_sharding = try zml.sharding.replicatedSharding(platform);

    log.info("Materializing buffers...", .{});
    var buffers = try model.load(arena.allocator(), io, platform, &store, &.{replicated_sharding});
    defer KNormFixture.unloadBuffers(&buffers);
    log.info("Buffers loaded (4 tensors).", .{});

    log.info("Compiling forward (k_proj + reshape + rmsNorm + mul, no RoPE/transpose/V/cache)...", .{});
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

    // === Oracle slice (used for both fixed-point blocks AND global scan). ===
    var ref_slice = try buffers.k_after_norm_oracle.toSliceAlloc(allocator, io);
    defer ref_slice.free(allocator);
    const ref_data = ref_slice.items(f32);

    if (ref_data.len != K_NORM_FLAT_LEN or data.len != K_NORM_FLAT_LEN) {
        log.err("length mismatch: ref={d} data={d} expected={d}", .{ ref_data.len, data.len, K_NORM_FLAT_LEN });
        return error.KNormLengthMismatch;
    }

    // === Fixed-point blocks (2 x 8 valeurs) vs oracle ===
    log.info("Fixed-point blocks vs oracle k_after_norm (fp32):", .{});
    var max_block: f32 = 0.0;
    for (K_NORM_BLOCKS) |block| {
        var block_max: f32 = 0.0;
        log.info("  Block {s} (flat_offset={d}):", .{ block.label, block.flat_offset });
        var i: usize = 0;
        while (i < block.width) : (i += 1) {
            const actual = data[block.flat_offset + i];
            const expected = ref_data[block.flat_offset + i];
            const diff = @abs(actual - expected);
            if (diff > block_max) block_max = diff;
            log.info("    +{d:>3}: actual={d:.8} expected={d:.8} diff={e:.3}", .{
                i, actual, expected, diff,
            });
        }
        log.info("    block max_diff: {e:.6}", .{block_max});
        if (block_max > max_block) max_block = block_max;
    }
    log.info("  -> 2 blocks max_diff: {e:.6}", .{max_block});

    // === Scan global 1024 valeurs vs oracle ===
    log.info("Scanning full tensor ({d} fp32) vs oracle k_after_norm:", .{K_NORM_FLAT_LEN});

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
    const mean_abs = @as(f32, @floatCast(sum_abs / @as(f64, @floatFromInt(K_NORM_FLAT_LEN))));

    // Decompose max_idx into (s, kvh, d).
    // strides : s -> NKV*D = 256, kvh -> D = 256, d -> 1.
    const stride_s: usize = @intCast(NKV * D);
    const stride_kvh: usize = @intCast(D);
    const s_idx = max_idx / stride_s;
    const kvh_idx = (max_idx % stride_s) / stride_kvh;
    const d_idx = max_idx % stride_kvh;

    log.info("  -> full tensor max_abs : {e:.6} at flat_index {d} (s={d}, kvh={d}, d={d})", .{
        max_global, max_idx, s_idx, kvh_idx, d_idx,
    });
    log.info("  -> full tensor mean_abs: {e:.6}", .{mean_abs});

    const max_diff = if (max_global > max_block) max_global else max_block;
    log.info("k_norm global max_diff: {e:.6} (tolerance {e:.1})", .{ max_diff, K_NORM_TOLERANCE });
    log.info("  Expected ~ D.1 k_proj residual ~5e-6 amorti par RMSNorm normalization", .{});

    if (max_diff > K_NORM_TOLERANCE) {
        log.err("BLOCK: k_norm max_diff exceeds tolerance", .{});
        log.err("  suspects: (1) wrong RMSNorm axis, (2) (1+weight) au lieu de pure weight,", .{});
        log.err("            (3) reshape sans withTags, (4) mul sans broad explicite,", .{});
        log.err("            (5) v_norm utilise par erreur (n'existe pas dans le checkpoint)", .{});
        return error.KNormFailed;
    }
    log.info("P5.2.D.3 PASS: ZML k_norm producer layer 13 validated vs PyTorch oracle", .{});
    log.info("  (no v_proj, no v_norm, no RoPE, no transpose, no cache, no attention)", .{});
}
