// P4.4.2 — Mini-runner ZML PLE-only pour gemma-4-E2B-it.
//
// Gate A : charger ple_fixture.safetensors, declarer les 5 tenseurs symboliques,
//          materialiser via zml.io.load. PASS.
// Gate B : forward = embed_tokens_slice.scale(sqrt(1536)). PASS bit-exact.
// Gate C : forward = embed_tokens_per_layer_slice.scale(16.0). PASS bit-exact.
// Gate D : forward = embed_tokens_per_layer_slice.scale(16.0).reshape({1,4,35,256}). PASS.
// Gate E : forward = (embed_tokens_slice.scale(sqrt(1536)))
//                    .dot(per_layer_model_projection, .h)
//          Premier matmul. Resultat attendu shape {s=4, lp=8960}.
//          Pas de scale 1/sqrt(H), pas de reshape, pas de rmsnorm.
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

// Reference Gate E : (embed_tokens_slice * sqrt(1536)) @ per_layer_model_projection.T
// Result shape [s=4, lp=8960] row-major, flat 35840.
// 5 blocs strategiques (token 0 deb/mid/fin, token 1 deb, token 3 fin).
// Pre-calcule cote M1 via numpy fp32.
const GateBlock = struct {
    label: []const u8,
    flat_offset: usize,
    expected: []const f32,
};

const GATE_E_BLOCKS = [_]GateBlock{
    .{
        .label = "A [0, 0:8]",
        .flat_offset = 0,
        .expected = &.{
            -14.50805950164795, 0.7396711111068726, 0.532582700252533, 5.888609409332275,
            3.0879440307617188, -8.882484436035156, -1.937730312347412, 1.5792943239212036,
        },
    },
    .{
        .label = "B [0, 255:263]",
        .flat_offset = 255,
        .expected = &.{
            -21.616872787475586, -7.328896999359131, -3.3360605239868164, 3.7606163024902344,
            -8.546035766601562, 9.410595893859863, -8.833962440490723, -0.9049932360649109,
        },
    },
    .{
        .label = "C [0, 8952:8960]",
        .flat_offset = 8952,
        .expected = &.{
            1.5175281763076782, -12.554316520690918, -1.2858136892318726, -0.9069992899894714,
            -1.087567925453186, -0.7551918029785156, -0.16557270288467407, 5.755862712860107,
        },
    },
    .{
        .label = "D [1, 0:8]",
        .flat_offset = 8960,
        .expected = &.{
            -18.634944915771484, -0.27851176261901855, 2.536295175552368, 9.544035911560059,
            3.6698646545410156, -11.726886749267578, -1.9820091724395752, -0.3610670566558838,
        },
    },
    .{
        .label = "E [3, 8952:8960]",
        .flat_offset = 35832,
        .expected = &.{
            0.7460525035858154, -9.038456916809082, -2.6555161476135254, -3.7322161197662354,
            1.3786760568618774, 1.337554931640625, 1.346847414970398, 1.3081830739974976,
        },
    },
};
const GATE_E_TOLERANCE: f32 = 1.0e-4;

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

    /// Gate E forward : inputs_embeds = embed_tokens_slice * sqrt(H)
    ///                  context_proj = inputs_embeds @ per_layer_model_projection.T
    /// Le tag .h dans .dot(W, .h) contracte sur la dim hidden=1536.
    /// Resultat : Tensor({s=4, lp=8960, f32}).
    pub fn forward(self: PleFixture) zml.Tensor {
        const inputs_embeds = self.embed_tokens_slice.scale(SQRT_H);
        return inputs_embeds.dot(self.per_layer_model_projection, .h);
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

    // === Gate E: compile + run forward (inputs_embeds @ proj over .h) ===
    log.info("Gate E: compiling forward (scale(sqrt(H)) + dot(proj, .h))...", .{});
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
    log.info("Validating 5 dot-product blocks vs numpy fp32 ref:", .{});

    var max_diff_global: f32 = 0.0;
    for (GATE_E_BLOCKS) |block| {
        var block_max: f32 = 0.0;
        log.info("  Block {s} (flat_offset={}):", .{ block.label, block.flat_offset });
        for (block.expected, 0..) |expected, i| {
            const actual = data[block.flat_offset + i];
            const diff = @abs(actual - expected);
            if (diff > block_max) block_max = diff;
            log.info("    +{:>3}: actual={d:.6} expected={d:.6} diff={e:.3}", .{ i, actual, expected, diff });
        }
        log.info("    block max_diff: {e:.6}", .{block_max});
        if (block_max > max_diff_global) max_diff_global = block_max;
    }

    log.info("Gate E global max_diff: {e:.6} (tolerance {e:.1})", .{ max_diff_global, GATE_E_TOLERANCE });

    if (max_diff_global > GATE_E_TOLERANCE) {
        log.err("BLOCK: Gate E max_diff exceeds tolerance", .{});
        return error.GateEFailed;
    }
    log.info("Gate E PASS: scale(sqrt(H)) + dot(proj, .h) matches numpy reference", .{});
}
