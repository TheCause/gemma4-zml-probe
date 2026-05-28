// P5.2.D.1 — ZML k_proj uniquement, producer/writer layer 13 (sliding).
//
// Objectif : valider une seule projection lineaire K en ZML, comparer le
// resultat byte-equivalent (~1e-5 attendu, cf C.1) contre l'oracle PyTorch
// fp32 `k_after_proj` produit en P5.2.D.0 (fixture slim re-exportee par
// `scripts/15_p5_2_d1_export_fixture.py`).
//
// Pipeline ZML strict :
//   k_after_proj = hidden_input.dot(k_proj_weight, .h)
//   shape : [.b=1, .s=4, .h=1536] dot [.kv=256, .h=1536] -> [.b=1, .s=4, .kv=256]
//
// Note tags : pour D.1 on prefere .kv (lisible, distingue de .o du Q-path).
//
// Fixture consommee : `fixtures/p5_2_d1_k_proj_layer13.safetensors`
// 3 tenseurs : hidden_input, k_proj_weight, k_after_proj.
//
// Interdits stricts P5.2.D.1 :
//   - v_proj
//   - k_norm
//   - RoPE
//   - reshape [B,S,n_kv,head_dim]
//   - transpose [B,n_kv,S,head_dim]
//   - cache slot
//   - attention scores / matmul QK / softmax
//   - sliding mask
//
// CLI : gemma4_k_proj <path-to-p5_2_d1_k_proj_layer13.safetensors>

const std = @import("std");
const log = std.log;

const zml = @import("zml");
const stdx = zml.stdx;

pub const std_options: std.Options = .{
    .log_level = .info,
};

// Invariants P5.2.D.0 (cf manifest p5_2_d0_kv_oracle_layer13_manifest.json).
// num_key_value_heads=1, head_dim=256 -> k_proj output features = 256.
const B: i64 = 1;
const S: i64 = 4;
const H: i64 = 1536;
const KV: i64 = 256; // n_kv (1) * head_dim (256)

// Tolerance / sanity expectations.
const K_PROJ_TOLERANCE: f32 = 1.0e-4;
const K_PROJ_FLAT_LEN: usize = @intCast(B * S * KV); // 1024

// Fixed-point blocks (reported per-position vs oracle, no hardcoded expected) :
// k_after_proj[0, {0,1,3}, :8]. flat_offset = s * KV + 0 (avec KV=256).
const KProjBlock = struct {
    label: []const u8,
    flat_offset: usize,
    width: usize,
};

const K_PROJ_BLOCKS = [_]KProjBlock{
    .{ .label = "A [0,0,:8]", .flat_offset = 0, .width = 8 },
    .{ .label = "B [0,1,:8]", .flat_offset = 256, .width = 8 },
    .{ .label = "C [0,3,:8]", .flat_offset = 768, .width = 8 },
};

/// Fixture D.1 slim (3 tenseurs) chargee depuis safetensors.
const KProjFixture = struct {
    hidden_input: zml.Tensor,
    k_proj_weight: zml.Tensor,
    k_after_proj_oracle: zml.Tensor,

    pub fn init(store: zml.io.TensorStore.View) KProjFixture {
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
            .k_after_proj_oracle = store.createTensor(
                "k_after_proj",
                .{ .b, .s, .kv },
                null,
            ),
        };
    }

    pub fn load(
        self: *const KProjFixture,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *const zml.io.TensorStore,
        shardings: []const zml.sharding.Sharding,
    ) !zml.Bufferized(KProjFixture) {
        return zml.io.load(KProjFixture, self, allocator, io, platform, store, .{
            .shardings = shardings,
            .parallelism = 1,
            .dma_chunks = 1,
            .dma_chunk_size = 16 * 1024 * 1024,
        });
    }

    pub fn unloadBuffers(self: *zml.Bufferized(KProjFixture)) void {
        self.hidden_input.deinit();
        self.k_proj_weight.deinit();
        self.k_after_proj_oracle.deinit();
    }

    /// Forward D.1 : un seul matmul K.
    ///   hidden_input [.b, .s, .h]  dot  k_proj_weight [.kv, .h]  -> [.b, .s, .kv]
    /// Reduction sur .h, output garde .b, .s, .kv.
    pub fn forward(self: KProjFixture) zml.Tensor {
        return self.hidden_input.dot(self.k_proj_weight, .h);
    }
};

pub fn main(init: std.process.Init) !void {
    const arena = init.arena;
    const allocator = init.gpa;
    const io = init.io;

    const process_args = try init.minimal.args.toSlice(arena.allocator());
    if (process_args.len < 2) {
        log.err("Usage: gemma4_k_proj <path-to-p5_2_d1_k_proj_layer13.safetensors>", .{});
        return error.MissingArgument;
    }
    const fixture_path = process_args[1];

    log.info("P5.2.D.1 — ZML k_proj only (producer layer 13 sliding, no k_norm/RoPE/V)", .{});
    log.info("Loading fixture from {s}", .{fixture_path});

    var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, fixture_path);
    defer registry.deinit();

    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();

    const model: KProjFixture = .init(store.view());

    log.info("Symbolic shapes:", .{});
    log.info("  hidden_input       : {f}", .{model.hidden_input});
    log.info("  k_proj_weight      : {f}", .{model.k_proj_weight});
    log.info("  k_after_proj_oracle: {f}", .{model.k_after_proj_oracle});

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);

    const replicated_sharding = try zml.sharding.replicatedSharding(platform);

    log.info("Materializing buffers...", .{});
    var buffers = try model.load(arena.allocator(), io, platform, &store, &.{replicated_sharding});
    defer KProjFixture.unloadBuffers(&buffers);
    log.info("Buffers loaded (3 tensors).", .{});

    log.info("Compiling forward (single dot, reduce .h, no k_norm/RoPE)...", .{});
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
    var ref_slice = try buffers.k_after_proj_oracle.toSliceAlloc(allocator, io);
    defer ref_slice.free(allocator);
    const ref_data = ref_slice.items(f32);

    if (ref_data.len != K_PROJ_FLAT_LEN or data.len != K_PROJ_FLAT_LEN) {
        log.err("length mismatch: ref={d} data={d} expected={d}", .{ ref_data.len, data.len, K_PROJ_FLAT_LEN });
        return error.KProjLengthMismatch;
    }

    // === Fixed-point blocks (3 x 8 valeurs) vs oracle ===
    log.info("Fixed-point blocks vs oracle k_after_proj (fp32):", .{});
    var max_block: f32 = 0.0;
    for (K_PROJ_BLOCKS) |block| {
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
    log.info("  -> 3 blocks max_diff: {e:.6}", .{max_block});

    // === Scan global 1024 valeurs vs oracle ===
    log.info("Scanning full tensor ({d} fp32) vs oracle k_after_proj:", .{K_PROJ_FLAT_LEN});

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
    const mean_abs = @as(f32, @floatCast(sum_abs / @as(f64, @floatFromInt(K_PROJ_FLAT_LEN))));
    log.info("  -> full tensor max_abs : {e:.6} at flat_index {d} (s={d}, d={d})", .{
        max_global, max_idx, max_idx / @as(usize, @intCast(KV)), max_idx % @as(usize, @intCast(KV)),
    });
    log.info("  -> full tensor mean_abs: {e:.6}", .{mean_abs});

    const max_diff = if (max_global > max_block) max_global else max_block;
    log.info("k_proj global max_diff: {e:.6} (tolerance {e:.1})", .{ max_diff, K_PROJ_TOLERANCE });
    log.info("  Expected ~1e-5 (matmul PJRT-CPU Eigen-like vs PyTorch BLAS, cf C.1 q_proj)", .{});

    if (max_diff > K_PROJ_TOLERANCE) {
        log.err("BLOCK: k_proj max_diff exceeds tolerance", .{});
        return error.KProjFailed;
    }
    log.info("P5.2.D.1 PASS: ZML k_proj producer layer 13 validated vs PyTorch oracle", .{});
    log.info("  (no v_proj, no k_norm, no RoPE, no reshape, no transpose, no cache, no attention)", .{});
}
