// P5.6.K — ZML full_attention K-rope manuelle partielle (layer 14). Ferme le gap K-full-rope.
// Technique IDENTIQUE à P5.6 (Q) appliquée à K (1 tête kv, head_dim 512). cos/sin 512-wide oracle,
// rotate_half via split/neg/concat, k*cos + rotate_half(k)*sin.
// CLI : gemma4_full_krope <path-to-p5_6k_full_krope_layer14.safetensors>

const std = @import("std");
const log = std.log;
const zml = @import("zml");
const stdx = zml.stdx;

pub const std_options: std.Options = .{ .log_level = .info };

const B: i64 = 1;
const S: i64 = 4;
const NKV: i64 = 1;
const HD: i64 = 512;
const HALF: i64 = 256;
const HIN: i64 = 1536;
const O: i64 = NKV * HD; // 512
const RMS_EPS: f32 = 1.0e-6;

const ROPE_TOLERANCE: f32 = 1.0e-4;
const MAGNITUDE_CEIL: f32 = 5.0;
const FLAT_LEN: usize = @intCast(B * S * NKV * HD); // 2048
const STRIDE_S: usize = @intCast(NKV * HD); // 512

const RBlock = struct { label: []const u8, flat_offset: usize, expected: []const f32 };

const R_BLOCKS = [_]RBlock{
    .{
        .label = "k_after_rope[0,0,0,:8]",
        .flat_offset = 0,
        .expected = &.{ -0.0111265974, 0.0167657249, -0.0003671990, 0.0230035521, -0.0012761118, 0.0098401103, 0.0013963099, -0.0246327315 },
    },
    .{
        .label = "k_after_rope[0,3,0,:8]",
        .flat_offset = 1536, // s=3 -> 3*512
        .expected = &.{ 0.0132894563, 0.0422661938, 0.0008076691, -0.0169820655, -0.0124669513, 0.0142035764, 0.0000887966, -0.0050656814 },
    },
};

const FullKRopeFixture = struct {
    hidden_input: zml.Tensor,
    k_proj_weight: zml.Tensor,
    k_norm_weight: zml.Tensor,
    cos_full: zml.Tensor,
    sin_full: zml.Tensor,
    k_after_rope_oracle: zml.Tensor,

    pub fn init(store: zml.io.TensorStore.View) FullKRopeFixture {
        return .{
            .hidden_input = store.createTensor("hidden_input", .{ .b, .s, .h }, null),
            .k_proj_weight = store.createTensor("k_proj_weight", .{ .o, .h }, null),
            .k_norm_weight = store.createTensor("k_norm_weight", .{.hd}, null),
            .cos_full = store.createTensor("cos_full", .{ .b, .s, .hd }, null),
            .sin_full = store.createTensor("sin_full", .{ .b, .s, .hd }, null),
            .k_after_rope_oracle = store.createTensor("k_after_rope", .{ .b, .s, .nh, .hd }, null),
        };
    }

    pub fn load(
        self: *const FullKRopeFixture,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *const zml.io.TensorStore,
        shardings: []const zml.sharding.Sharding,
    ) !zml.Bufferized(FullKRopeFixture) {
        return zml.io.load(FullKRopeFixture, self, allocator, io, platform, store, .{
            .shardings = shardings,
            .parallelism = 1,
            .dma_chunks = 1,
            .dma_chunk_size = 16 * 1024 * 1024,
        });
    }

    pub fn unloadBuffers(self: *zml.Bufferized(FullKRopeFixture)) void {
        self.hidden_input.deinit();
        self.k_proj_weight.deinit();
        self.k_norm_weight.deinit();
        self.cos_full.deinit();
        self.sin_full.deinit();
        self.k_after_rope_oracle.deinit();
    }

    pub fn forward(self: FullKRopeFixture) zml.Tensor {
        const k_proj = self.hidden_input.dot(self.k_proj_weight, .h);
        const k_4d = k_proj.reshape(.{ B, S, NKV, HD }).withTags(.{ .b, .s, .nh, .hd });
        const k_normed = zml.nn.rmsNorm(k_4d, .hd, RMS_EPS);
        const k = k_normed.mul(self.k_norm_weight.broad(k_normed.shape()));
        const halves = k.split(.hd, &.{ HALF, HALF });
        const rh = zml.Tensor.concatenate(&.{ halves[1].negate(), halves[0] }, .hd);
        const term_cos = k.mul(self.cos_full.broad(k.shape()));
        const term_sin = rh.mul(self.sin_full.broad(k.shape()));
        return term_cos.add(term_sin);
    }
};

pub fn main(init: std.process.Init) !void {
    const arena = init.arena;
    const allocator = init.gpa;
    const io = init.io;

    const process_args = try init.minimal.args.toSlice(arena.allocator());
    if (process_args.len < 2) {
        log.err("Usage: gemma4_full_krope <path-to-p5_6k_full_krope_layer14.safetensors>", .{});
        return error.MissingArgument;
    }
    const fixture_path = process_args[1];

    log.info("P5.6.K — ZML full_attention K-rope manuelle (layer 14, head_dim 512, partial, 1 kv head)", .{});
    log.info("Loading fixture from {s}", .{fixture_path});

    var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, fixture_path);
    defer registry.deinit();

    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();

    const model: FullKRopeFixture = .init(store.view());

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);

    const replicated_sharding = try zml.sharding.replicatedSharding(platform);

    var buffers = try model.load(arena.allocator(), io, platform, &store, &.{replicated_sharding});
    defer FullKRopeFixture.unloadBuffers(&buffers);
    log.info("Buffers loaded (6 tensors).", .{});

    var exe = try platform.compile(allocator, io, model, .forward, .{}, .{ .shardings = &.{replicated_sharding} });
    defer exe.deinit();

    var args = try exe.args(allocator);
    defer args.deinit(allocator);

    var results = try exe.results(allocator);
    defer results.deinit(allocator);

    args.set(.{buffers});
    exe.call(args, &results);

    var result: zml.Buffer = results.get(zml.Buffer);
    defer result.deinit();

    log.info("Forward result shape: {f} (expected [b=1, s=4, nh=1, hd=512])", .{result.shape()});

    var slice = try result.toSliceAlloc(allocator, io);
    defer slice.free(allocator);
    const data = slice.items(f32);
    if (data.len != FLAT_LEN) return error.RopeLengthMismatch;

    var has_nan_inf = false;
    var max_mag: f32 = 0.0;
    for (data) |v| {
        if (std.math.isNan(v) or std.math.isInf(v)) has_nan_inf = true;
        const av = @abs(v);
        if (av > max_mag) max_mag = av;
    }
    log.info("Sanity: NaN/Inf={}, max|k_rope|={d:.4} (ceil {d:.1})", .{ has_nan_inf, max_mag, MAGNITUDE_CEIL });
    if (has_nan_inf or max_mag > MAGNITUDE_CEIL) return error.RopeSanity;

    var max_block: f32 = 0.0;
    for (R_BLOCKS) |block| {
        var block_max: f32 = 0.0;
        log.info("  Block {s} (flat_offset={d}):", .{ block.label, block.flat_offset });
        for (block.expected, 0..) |expected, i| {
            const actual = data[block.flat_offset + i];
            const diff = @abs(actual - expected);
            if (diff > block_max) block_max = diff;
            log.info("    +{d:>3}: actual={d:.8} expected={d:.8} diff={e:.3}", .{ i, actual, expected, diff });
        }
        log.info("    block max_diff: {e:.6}", .{block_max});
        if (block_max > max_block) max_block = block_max;
    }

    var ref_slice = try buffers.k_after_rope_oracle.toSliceAlloc(allocator, io);
    defer ref_slice.free(allocator);
    const ref_data = ref_slice.items(f32);
    if (ref_data.len != FLAT_LEN) return error.RopeLengthMismatch;

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
    log.info("  -> full tensor max_abs : {e:.6} at flat_index {d}", .{ max_global, max_idx });
    log.info("  -> full tensor mean_abs: {e:.6}", .{mean_abs});

    const max_diff = if (max_global > max_block) max_global else max_block;
    log.info("full_krope global max_diff: {e:.6} (tolerance {e:.1})", .{ max_diff, ROPE_TOLERANCE });
    if (max_diff > ROPE_TOLERANCE) return error.RopeFailed;
    log.info("P5.6.K PASS: ZML full_attention K-rope manuelle (layer 14) validated -> gap K-full-rope fermé", .{});
}
