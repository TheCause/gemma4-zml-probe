// P4.4.2 — Mini-runner ZML PLE-only pour gemma-4-E2B-it.
//
// Gate A : charger ple_fixture.safetensors, declarer les 5 tenseurs symboliques,
//          materialiser via zml.io.load. PASS.
// Gate B : forward = embed_tokens_slice.scale(sqrt(1536)). PASS bit-exact.
// Gate C : forward = embed_tokens_per_layer_slice.scale(16.0). PASS bit-exact.
// Gate D : forward = embed_tokens_per_layer_slice.scale(16.0).reshape({1,4,35,256}). PASS.
// Gate E : forward = (embed_tokens_slice.scale(sqrt(1536)))
//                    .dot(per_layer_model_projection, .h). PASS bit-exact.
// Gate F : ajoute .scale(1/sqrt(1536)) apres le dot. PASS bit-exact.
// Gate G : ajoute .reshape(.{1, 4, 35, 256}) en fin de chaine.
//          Reshape structurel pur, attendu bit-exact (8960 = 35*256).
// Gate H : RMSNorm(context_4d, axe .d) puis mul(per_layer_projection_norm.weight).
//          Pattern Llama : normalized.mul(weight). PAS Qwen (1+weight).
//          Reference numpy fp32 (Gates E/F/G upstream tous bit-exact).
//          max_diff = 1.49e-8 (tolerance 1e-4, ~7000x sous le seuil). PASS.
// Gate I : token_identity.add(context_normalized). Fusion pure des 2 branches PLE.
//          Branche A = sortie Gate D (re-taggee), branche B = sortie Gate H.
//          PAS de /√2 ici (reserve Gate J avec comparaison ple_reference_final).
//          max_diff = 1.49e-8 (heritage Gate H, add n'ajoute pas de drift). PASS.
// Gate J : ple_pre_final.scale(INV_SQRT_2) = ple_final. Ferme P4.4.2 PLE-only.
//          Premier oracle = fixture ple_reference_final (produit P3 PyTorch fp32),
//          PAS reference numpy calculee a la volee. Donc matmul Eigen-like (PJRT)
//          vs matmul PyTorch BLAS -> ~1.5e-5 residu attendu (cf. P4.3).
//          4 points fixes + scan global max_abs/mean_abs sur les 35840 valeurs.
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

// Reference Gate J : ple_final = (token_identity + context_normalized) / sqrt(2)
// Oracle = fixture fp32 `ple_reference_final.npy` (produit en P3 via PyTorch fp32).
// Shape [1, 4, 35, 256] row-major, flat 35840.
const GATE_J_BLOCKS = [_]GateBlock{
    .{
        .label = "A [0,0,0,:4]",
        .flat_offset = 0,
        .expected = &.{
            -0.0127844661, 0.0874446332, -1.5129654408, 0.0800682306,
        },
    },
    .{
        .label = "B [0,0,34,:4]",
        .flat_offset = 8704,
        .expected = &.{
            0.3450092971, -0.3198175728, 0.2380623370, 0.0752871484,
        },
    },
    .{
        .label = "C [0,3,0,:4]",
        .flat_offset = 26880,
        .expected = &.{
            0.4198024273, 0.0997980982, 1.6338746548, 0.9890668392,
        },
    },
    .{
        .label = "D [0,3,34,:4]",
        .flat_offset = 35584,
        .expected = &.{
            0.4773316383, 0.4231876135, 1.4620192051, -0.3281090856,
        },
    },
};
// Cible : 1e-4 (cohérent avec gates précédents). Attendu ~1.5e-5 (matmul PJRT
// Eigen-like vs PyTorch BLAS, residu observe en P4.3 numpy vs fixture).
const GATE_J_TOLERANCE: f32 = 1.0e-4;
// Sanity expectations sur tenseur entier (35840 valeurs fp32).
const PLE_FINAL_FLAT_LEN: usize = 35840;

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

    /// Gate J forward : chaîne PLE complete A→J pour gemma-4-E2B-it.
    /// Gate I (token_identity + context_normalized) puis Gate J (.scale(1/√2)).
    /// Resultat : Tensor({b=1, s=4, l=35, d=256, f32}) = ple_final.
    pub fn forward(self: PleFixture) zml.Tensor {
        // Branche A — token_identity (Gate D), reshape perd les tags -> re-tag.
        const token_identity = self.embed_tokens_per_layer_slice
            .scale(SQRT_D)
            .reshape(.{ 1, 4, 35, 256 })
            .withTags(.{ .b, .s, .l, .d });

        // Branche B — context_normalized (Gates E/F/G/H), idem re-tag pour rmsNorm.
        const inputs_embeds = self.embed_tokens_slice.scale(SQRT_H);
        const context_proj = inputs_embeds.dot(self.per_layer_model_projection, .h);
        const context_scaled = context_proj.scale(INV_SQRT_H);
        const context_4d = context_scaled
            .reshape(.{ 1, 4, 35, 256 })
            .withTags(.{ .b, .s, .l, .d });
        const normalized = zml.nn.rmsNorm(context_4d, .d, RMS_EPS);
        const context_normalized = normalized.mul(
            self.per_layer_projection_norm.broad(normalized.shape()),
        );

        // Gate I — fusion pure des 2 branches (add element-wise).
        const ple_pre_final = token_identity.add(context_normalized);

        // Gate J — scaling final 1/√2, ferme P4.4.2 PLE-only.
        return ple_pre_final.scale(INV_SQRT_2);
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

    // === Gate J: ple_pre_final.scale(INV_SQRT_2), ferme P4.4.2 PLE-only ===
    log.info("Gate J: compiling forward (chaîne A→J complete, scale 1/√2 final)...", .{});
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
    log.info("Validating 4 fixed-point 4D blocks vs fixture fp32 oracle (ple_reference_final):", .{});

    var max_diff_blocks: f32 = 0.0;
    for (GATE_J_BLOCKS) |block| {
        var block_max: f32 = 0.0;
        log.info("  Block {s} (flat_offset={}):", .{ block.label, block.flat_offset });
        for (block.expected, 0..) |expected, i| {
            const actual = data[block.flat_offset + i];
            const diff = @abs(actual - expected);
            if (diff > block_max) block_max = diff;
            log.info("    +{:>3}: actual={d:.8} expected={d:.8} diff={e:.3}", .{ i, actual, expected, diff });
        }
        log.info("    block max_diff: {e:.6}", .{block_max});
        if (block_max > max_diff_blocks) max_diff_blocks = block_max;
    }
    log.info("  → 4 blocks max_diff: {e:.6}", .{max_diff_blocks});

    // Scan global sur les 35840 valeurs : compare result device vs fixture
    // ple_reference_final (deja charge en buffer device, jamais materialise host).
    log.info("Scanning full tensor (35840 fp32 values) vs fixture ple_reference_final:", .{});
    var ref_slice = try buffers.ple_reference_final.toSliceAlloc(allocator, io);
    defer ref_slice.free(allocator);
    const ref_data = ref_slice.items(f32);

    if (ref_data.len != PLE_FINAL_FLAT_LEN or data.len != PLE_FINAL_FLAT_LEN) {
        log.err("Length mismatch: ref={} data={} expected={}", .{ ref_data.len, data.len, PLE_FINAL_FLAT_LEN });
        return error.GateJLengthMismatch;
    }

    var max_abs_global: f32 = 0.0;
    var sum_abs: f64 = 0.0;
    var max_idx: usize = 0;
    for (data, ref_data, 0..) |actual, expected, i| {
        const diff = @abs(actual - expected);
        if (diff > max_abs_global) {
            max_abs_global = diff;
            max_idx = i;
        }
        sum_abs += @as(f64, diff);
    }
    const mean_abs = @as(f32, @floatCast(sum_abs / @as(f64, @floatFromInt(PLE_FINAL_FLAT_LEN))));
    log.info("  → full tensor max_abs : {e:.6} at flat_index {}", .{ max_abs_global, max_idx });
    log.info("  → full tensor mean_abs: {e:.6}", .{mean_abs});

    const max_diff_global = if (max_abs_global > max_diff_blocks) max_abs_global else max_diff_blocks;
    log.info("Gate J global max_diff: {e:.6} (tolerance {e:.1})", .{ max_diff_global, GATE_J_TOLERANCE });
    log.info("  Expected ~1.5e-5 (matmul PJRT-CPU Eigen-like vs PyTorch BLAS, cf. P4.3)", .{});

    if (max_diff_global > GATE_J_TOLERANCE) {
        log.err("BLOCK: Gate J max_diff exceeds tolerance", .{});
        return error.GateJFailed;
    }
    log.info("Gate J PASS: PLE-only ZML minimal validated end-to-end (closes P4.4.2)", .{});
}
