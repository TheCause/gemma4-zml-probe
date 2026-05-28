// P4.4.2 — Mini-runner ZML PLE-only pour gemma-4-E2B-it.
//
// Gate A : charger ple_fixture.safetensors, declarer les 5 tenseurs symboliques,
//          materialiser via zml.io.load. PASS.
// Gate B : forward = embed_tokens_slice.scale(sqrt(1536)). PASS bit-exact.
// Gate C : forward = embed_tokens_per_layer_slice.scale(16.0). PASS bit-exact.
// Gate D : forward = embed_tokens_per_layer_slice.scale(16.0).reshape({1,4,35,256}). PASS.
// Gate E : forward = (embed_tokens_slice.scale(sqrt(1536)))
//                    .dot(per_layer_model_projection, .h). PASS bit-exact.
// Gate F : ajoute .scale(1/sqrt(1536)) apres le dot.
//          context_scaled = context_proj * INV_SQRT_H.
//          Toujours pas de reshape, pas de rmsnorm.
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

// Reference Gate F : ((embed * sqrt(1536)) @ proj.T) * (1/sqrt(1536))
// Result shape [s=4, lp=8960] row-major, flat 35840.
// Memes 5 blocs strategiques que Gate E, valeurs / sqrt(1536) ~= / 39.19.
const GateBlock = struct {
    label: []const u8,
    flat_offset: usize,
    expected: []const f32,
};

const GATE_F_BLOCKS = [_]GateBlock{
    .{
        .label = "A [0, 0:8]",
        .flat_offset = 0,
        .expected = &.{
            -0.3701806664466858, 0.018873091787099838, 0.013589124195277691, 0.15025092661380768,
            0.07879049330949783, -0.2266412079334259, -0.049442194402217865, 0.040296513587236404,
        },
    },
    .{
        .label = "B [0, 255:263]",
        .flat_offset = 255,
        .expected = &.{
            -0.5515657067298889, -0.18700061738491058, -0.08512131869792938, 0.09595407545566559,
            -0.2180565446615219, 0.24011623859405518, -0.22540313005447388, -0.023091372102499008,
        },
    },
    .{
        .label = "C [0, 8952:8960]",
        .flat_offset = 8952,
        .expected = &.{
            0.038720518350601196, -0.3203299045562744, -0.032808203250169754, -0.023142557591199875,
            -0.027749860659241676, -0.01926911063492298, -0.004224673379212618, 0.14686381816864014,
        },
    },
    .{
        .label = "D [1, 0:8]",
        .flat_offset = 8960,
        .expected = &.{
            -0.4754802882671356, -0.007106372155249119, 0.06471488624811172, 0.24352103471755981,
            0.09363850206136703, -0.2992176115512848, -0.0505719929933548, -0.009212813340127468,
        },
    },
    .{
        .label = "E [3, 8952:8960]",
        .flat_offset = 35832,
        .expected = &.{
            0.01903591677546501, -0.23062092065811157, -0.06775687634944916, -0.09522943198680878,
            0.03517763689160347, 0.03412840887904167, 0.03436551243066788, 0.0333789698779583,
        },
    },
};
const GATE_F_TOLERANCE: f32 = 1.0e-4;

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

    /// Gate F forward : inputs_embeds = embed_tokens_slice * sqrt(H)
    ///                  context_proj = inputs_embeds @ per_layer_model_projection.T
    ///                  context_scaled = context_proj * (1/sqrt(H))
    /// Resultat : Tensor({s=4, lp=8960, f32}).
    pub fn forward(self: PleFixture) zml.Tensor {
        const inputs_embeds = self.embed_tokens_slice.scale(SQRT_H);
        const context_proj = inputs_embeds.dot(self.per_layer_model_projection, .h);
        return context_proj.scale(INV_SQRT_H);
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

    // === Gate F: scale(sqrt(H)) + dot(proj, .h) + scale(1/sqrt(H)) ===
    log.info("Gate F: compiling forward (scale(sqrt(H)) + dot(proj, .h) + scale(1/sqrt(H)))...", .{});
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
    log.info("Validating 5 context_scaled blocks vs numpy fp32 ref:", .{});

    var max_diff_global: f32 = 0.0;
    for (GATE_F_BLOCKS) |block| {
        var block_max: f32 = 0.0;
        log.info("  Block {s} (flat_offset={}):", .{ block.label, block.flat_offset });
        for (block.expected, 0..) |expected, i| {
            const actual = data[block.flat_offset + i];
            const diff = @abs(actual - expected);
            if (diff > block_max) block_max = diff;
            log.info("    +{:>3}: actual={d:.8} expected={d:.8} diff={e:.3}", .{ i, actual, expected, diff });
        }
        log.info("    block max_diff: {e:.6}", .{block_max});
        if (block_max > max_diff_global) max_diff_global = block_max;
    }

    log.info("Gate F global max_diff: {e:.6} (tolerance {e:.1})", .{ max_diff_global, GATE_F_TOLERANCE });

    if (max_diff_global > GATE_F_TOLERANCE) {
        log.err("BLOCK: Gate F max_diff exceeds tolerance", .{});
        return error.GateFFailed;
    }
    log.info("Gate F PASS: context_scaled matches numpy reference", .{});
}
