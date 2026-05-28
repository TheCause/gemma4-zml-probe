// P4.4.2 — Mini-runner ZML PLE-only pour gemma-4-E2B-it.
//
// Gate A : charger ple_fixture.safetensors, declarer les 5 tenseurs symboliques,
//          materialiser via zml.io.load. PASS.
// Gate B : forward = embed_tokens_slice.scale(sqrt(1536)). PASS bit-exact.
// Gate C : forward = embed_tokens_per_layer_slice.scale(16.0). PASS bit-exact.
// Gate D : forward = embed_tokens_per_layer_slice.scale(16.0).reshape({1,4,35,256}).
//          Validation structurelle du layout via 5 blocs d'indices couvrant le 4D.
//
// CLI:
//   gemma4_ple_fixture <path-to-ple_fixture.safetensors>

const std = @import("std");
const log = std.log;

const zml = @import("zml");
const stdx = zml.stdx;

pub const std_options: std.Options = .{
    .log_level = .info,
};

// Invariants Gemma 4 E2B (figes dans fixtures/fixture_manifest.json)
const B: i64 = 1;
const S: i64 = 4;
const L: i64 = 35;
const H: i64 = 1536;
const D: i64 = 256;

// Constantes de scaling P4.4.2
const SQRT_H: f32 = 39.191835884530846;
const SQRT_D: f32 = 16.0;
const INV_SQRT_H: f32 = 0.02551551815399144;
const INV_SQRT_2: f32 = 0.7071067811865475;
const RMS_EPS: f32 = 1.0e-6;

// Reference Gate D : (embed_tokens_per_layer_slice * 16.0).reshape(1,4,35,256)
// 5 blocs d'indices couvrant le 4D ; flat_offset = b*(S*L*D) + s*(L*D) + l*D + d
// Calculee cote M1 via numpy fp32, layout consistency check FLAT == ref[idx] valide.
const GateDBlock = struct {
    label: []const u8,
    flat_offset: usize,
    expected: []const f32,
};

const GATE_D_BLOCKS = [_]GateDBlock{
    .{
        .label = "A [0,0,0,0:8]",
        .flat_offset = 0,
        .expected = &.{ 0.16796875, 0.04345703125, -2.1875, -0.0213623046875, 0.1494140625, -0.02490234375, 1.953125, -0.65625 },
    },
    .{
        .label = "B [0,0,0,252:256]",
        .flat_offset = 252,
        .expected = &.{ 0.1318359375, -0.486328125, -0.78125, 0.173828125 },
    },
    .{
        .label = "C [0,0,1,0:8]",
        .flat_offset = 256,
        .expected = &.{ 0.4296875, 0.2294921875, -0.1416015625, -2.375, -1.0390625, 1.1328125, -0.2294921875, -2.984375 },
    },
    .{
        .label = "D [0,0,34,0:8]",
        .flat_offset = 8704,
        .expected = &.{ 0.33203125, -0.64453125, 0.2890625, 0.2421875, 2.0625, -0.625, -0.1767578125, 1.171875 },
    },
    .{
        .label = "E [0,1,0,0:8]",
        .flat_offset = 8960,
        .expected = &.{ -0.71875, 1.125, 2.109375, 0.2451171875, 2.703125, -0.5390625, -1.2890625, -1.0078125 },
    },
};
const GATE_D_TOLERANCE: f32 = 1.0e-4;

/// Fixture PLE Gemma 4 charge depuis ple_fixture.safetensors.
const PleFixture = struct {
    embed_tokens_slice: zml.Tensor,
    embed_tokens_per_layer_slice: zml.Tensor,
    per_layer_model_projection: zml.Tensor,
    per_layer_projection_norm: zml.Tensor,
    ple_reference_final: zml.Tensor,

    pub fn init(store: zml.io.TensorStore.View) PleFixture {
        return .{
            .embed_tokens_slice = store.createTensor(
                "embed_tokens_slice",
                .{ .s, .h },
                null,
            ),
            .embed_tokens_per_layer_slice = store.createTensor(
                "embed_tokens_per_layer_slice",
                .{ .s, .lp },
                null,
            ),
            .per_layer_model_projection = store.createTensor(
                "per_layer_model_projection",
                .{ .lp, .h },
                null,
            ),
            .per_layer_projection_norm = store.createTensor(
                "per_layer_projection_norm",
                .{.d},
                null,
            ),
            .ple_reference_final = store.createTensor(
                "ple_reference_final",
                .{ .b, .s, .l, .d },
                null,
            ),
        };
    }

    pub fn load(
        self: *const PleFixture,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *const zml.io.TensorStore,
        shardings: []const zml.sharding.Sharding,
    ) !zml.Bufferized(PleFixture) {
        return zml.io.load(PleFixture, self, allocator, io, platform, store, .{
            .shardings = shardings,
            .parallelism = 1,
            .dma_chunks = 1,
            .dma_chunk_size = 16 * 1024 * 1024,
        });
    }

    pub fn unloadBuffers(self: *zml.Bufferized(PleFixture)) void {
        self.embed_tokens_slice.deinit();
        self.embed_tokens_per_layer_slice.deinit();
        self.per_layer_model_projection.deinit();
        self.per_layer_projection_norm.deinit();
        self.ple_reference_final.deinit();
    }

    /// Gate D forward : scale(16) puis reshape vers [1, 4, 35, 256].
    /// Toujours pas de dot, pas de rmsnorm, pas d'addition.
    pub fn forward(self: PleFixture) zml.Tensor {
        return self.embed_tokens_per_layer_slice.scale(SQRT_D).reshape(.{ 1, 4, 35, 256 });
    }
};

pub fn main(init: std.process.Init) !void {
    const arena = init.arena;
    const allocator = init.gpa;
    const io = init.io;

    const process_args = try init.minimal.args.toSlice(arena.allocator());
    if (process_args.len < 2) {
        log.err("Usage: gemma4_ple_fixture <path-to-ple_fixture.safetensors>", .{});
        return error.MissingArgument;
    }
    const fixture_path = process_args[1];

    log.info("Loading PLE fixture from {s}", .{fixture_path});

    var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, fixture_path);
    defer registry.deinit();

    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();

    const model: PleFixture = .init(store.view());

    log.info("Symbolic shapes:", .{});
    log.info("  embed_tokens_slice           : {f}", .{model.embed_tokens_slice});
    log.info("  embed_tokens_per_layer_slice : {f}", .{model.embed_tokens_per_layer_slice});
    log.info("  per_layer_model_projection   : {f}", .{model.per_layer_model_projection});
    log.info("  per_layer_projection_norm    : {f}", .{model.per_layer_projection_norm});
    log.info("  ple_reference_final          : {f}", .{model.ple_reference_final});

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);

    const replicated_sharding = try zml.sharding.replicatedSharding(platform);

    // === Gate A: materialize buffers ===
    log.info("Materializing buffers...", .{});
    var buffers = try model.load(arena.allocator(), io, platform, &store, &.{replicated_sharding});
    defer PleFixture.unloadBuffers(&buffers);
    log.info("Gate A PASS: 5 tensors loaded into device buffers", .{});

    // === Gate D: compile + run forward (scale by 16 + reshape [1,4,35,256]) ===
    log.info("Gate D: compiling forward (scale(16) + reshape [1,4,35,256])...", .{});
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
    log.info("Validating 5 layout blocks vs numpy fp32 ref (reshape [1,4,35,256] row-major):", .{});

    var max_diff_global: f32 = 0.0;
    for (GATE_D_BLOCKS) |block| {
        var block_max: f32 = 0.0;
        log.info("  Block {s} (flat_offset={}):", .{ block.label, block.flat_offset });
        for (block.expected, 0..) |expected, i| {
            const actual = data[block.flat_offset + i];
            const diff = @abs(actual - expected);
            if (diff > block_max) block_max = diff;
            log.info("    +{:>3}: actual={d:.10} expected={d:.10} diff={e:.3}", .{ i, actual, expected, diff });
        }
        log.info("    block max_diff: {e:.6}", .{block_max});
        if (block_max > max_diff_global) max_diff_global = block_max;
    }

    log.info("Gate D global max_diff: {e:.6} (tolerance {e:.1})", .{ max_diff_global, GATE_D_TOLERANCE });

    if (max_diff_global > GATE_D_TOLERANCE) {
        log.err("BLOCK: Gate D max_diff exceeds tolerance", .{});
        return error.GateDFailed;
    }
    log.info("Gate D PASS: scale(16) + reshape [1,4,35,256] matches numpy reference (row-major layout confirmed)", .{});
}
