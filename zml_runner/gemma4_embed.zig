// P5.4 — ZML embedding lookup (gather) + scale. Entrée du modèle.
//
// Gemma4TextScaledWordEmbedding : embed_tokens(input_ids) * sqrt(hidden_size=1536).
// P4.4 Gate B avait validé le SCALE bit-exact (slice pré-gathered) ; ici on valide le GATHER ZML
// (`weight.gather(.{.voc = tokens})`, cf qwen3_5/lfm2/zml.nn.embed) + scale, end-to-end.
//
// Table complète [262144,1536] impraticable -> gather sur slice vocab 4096 (gather pleine table
// = mécaniquement identique). Pipeline :
//   gathered  = embed_slice.gather(.{ .voc = input_ids }, .{})   {.b,.s,.d}
//   embed_out = gathered.scale(sqrt(1536))
//
// CLI : gemma4_embed <path-to-p5_4_embed.safetensors>

const std = @import("std");
const log = std.log;

const zml = @import("zml");
const stdx = zml.stdx;

pub const std_options: std.Options = .{ .log_level = .info };

const B: i64 = 1;
const S: i64 = 4;
const D: i64 = 1536;
const EMBED_SCALE: f64 = @sqrt(1536.0); // sqrt(hidden_size)

const EMBED_TOLERANCE: f32 = 1.0e-4;
const MAGNITUDE_CEIL: f32 = 50.0;
const FLAT_LEN: usize = @intCast(B * S * D); // 6144
const STRIDE_S: usize = @intCast(D); // 1536

const EBlock = struct { label: []const u8, flat_offset: usize, expected: []const f32 };

const E_BLOCKS = [_]EBlock{
    .{
        .label = "embed_out[0,0,:8]",
        .flat_offset = 0,
        .expected = &.{ -1.6361826658, -1.5309311152, 0.1877782792, -1.4830895662, -0.9951052666, -0.0272099096, -0.4449268579, 0.2738931477 },
    },
    .{
        .label = "embed_out[0,3,:8]",
        .flat_offset = 4608, // q=3 -> 3*1536
        .expected = &.{ 1.4735212326, 0.4951605499, -0.7271922827, -1.6744558811, -0.3635961413, -0.0294524841, 0.5812754035, 1.0094577074 },
    },
};

const EmbedFixture = struct {
    embed_slice: zml.Tensor,
    input_ids: zml.Tensor,
    embed_out_oracle: zml.Tensor,

    pub fn init(store: zml.io.TensorStore.View) EmbedFixture {
        return .{
            .embed_slice = store.createTensor("embed_slice", .{ .voc, .d }, null),
            .input_ids = store.createTensor("input_ids", .{ .b, .s }, null),
            .embed_out_oracle = store.createTensor("embed_out", .{ .b, .s, .d }, null),
        };
    }

    pub fn load(
        self: *const EmbedFixture,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *const zml.io.TensorStore,
        shardings: []const zml.sharding.Sharding,
    ) !zml.Bufferized(EmbedFixture) {
        return zml.io.load(EmbedFixture, self, allocator, io, platform, store, .{
            .shardings = shardings,
            .parallelism = 1,
            .dma_chunks = 1,
            .dma_chunk_size = 16 * 1024 * 1024,
        });
    }

    pub fn unloadBuffers(self: *zml.Bufferized(EmbedFixture)) void {
        self.embed_slice.deinit();
        self.input_ids.deinit();
        self.embed_out_oracle.deinit();
    }

    /// Forward P5.4 : gather (embedding lookup) + scale sqrt(1536).
    pub fn forward(self: EmbedFixture) zml.Tensor {
        const gathered = self.embed_slice.gather(.{ .voc = self.input_ids }, .{}); // {.b,.s,.d}
        return gathered.scale(EMBED_SCALE);
    }
};

pub fn main(init: std.process.Init) !void {
    const arena = init.arena;
    const allocator = init.gpa;
    const io = init.io;

    const process_args = try init.minimal.args.toSlice(arena.allocator());
    if (process_args.len < 2) {
        log.err("Usage: gemma4_embed <path-to-p5_4_embed.safetensors>", .{});
        return error.MissingArgument;
    }
    const fixture_path = process_args[1];

    log.info("P5.4 — ZML embedding gather + scale sqrt(1536) (vocab slice 4096)", .{});
    log.info("Loading fixture from {s}", .{fixture_path});

    var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, fixture_path);
    defer registry.deinit();

    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();

    const model: EmbedFixture = .init(store.view());

    log.info("Symbolic shapes:", .{});
    log.info("  embed_slice : {f}", .{model.embed_slice});
    log.info("  input_ids   : {f}", .{model.input_ids});
    log.info("  embed_out   : {f}", .{model.embed_out_oracle});

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);

    const replicated_sharding = try zml.sharding.replicatedSharding(platform);

    var buffers = try model.load(arena.allocator(), io, platform, &store, &.{replicated_sharding});
    defer EmbedFixture.unloadBuffers(&buffers);
    log.info("Buffers loaded (3 tensors).", .{});

    log.info("Compiling forward (gather .voc -> scale)...", .{});
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

    log.info("Forward result shape: {f} (expected [b=1, s=4, d=1536])", .{result.shape()});

    var slice = try result.toSliceAlloc(allocator, io);
    defer slice.free(allocator);
    const data = slice.items(f32);

    if (data.len != FLAT_LEN) {
        log.err("length mismatch: data={d} expected={d}", .{ data.len, FLAT_LEN });
        return error.EmbedLengthMismatch;
    }

    var has_nan_inf = false;
    var max_mag: f32 = 0.0;
    for (data) |v| {
        if (std.math.isNan(v) or std.math.isInf(v)) has_nan_inf = true;
        const av = @abs(v);
        if (av > max_mag) max_mag = av;
    }
    log.info("Sanity: NaN/Inf={}, max|embed_out|={d:.4} (ceil {d:.1})", .{ has_nan_inf, max_mag, MAGNITUDE_CEIL });
    if (has_nan_inf) return error.EmbedNanInf;
    if (max_mag > MAGNITUDE_CEIL) return error.EmbedMagnitude;

    log.info("Fixed-point blocks vs oracle embed_out (fp32):", .{});
    var max_block: f32 = 0.0;
    for (E_BLOCKS) |block| {
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

    log.info("Scanning full tensor ({d} fp32) vs oracle embed_out:", .{FLAT_LEN});
    var ref_slice = try buffers.embed_out_oracle.toSliceAlloc(allocator, io);
    defer ref_slice.free(allocator);
    const ref_data = ref_slice.items(f32);
    if (ref_data.len != FLAT_LEN) return error.EmbedLengthMismatch;

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
    const s_idx = (max_idx % @as(usize, @intCast(S * D))) / STRIDE_S;
    const d_idx = max_idx % STRIDE_S;
    log.info("  -> full tensor max_abs : {e:.6} at flat_index {d} (s={d}, d={d})", .{ max_global, max_idx, s_idx, d_idx });
    log.info("  -> full tensor mean_abs: {e:.6}", .{mean_abs});

    const max_diff = if (max_global > max_block) max_global else max_block;
    log.info("embed global max_diff: {e:.6} (tolerance {e:.1})", .{ max_diff, EMBED_TOLERANCE });
    log.info("  Expected ~0 (gather = sélection exacte de lignes, scale = mul scalaire)", .{});

    if (max_diff > EMBED_TOLERANCE) {
        log.err("BLOCK: embed max_diff exceeds tolerance", .{});
        return error.EmbedFailed;
    }
    log.info("P5.4 PASS: ZML embedding gather + scale validated vs PyTorch oracle", .{});
}
