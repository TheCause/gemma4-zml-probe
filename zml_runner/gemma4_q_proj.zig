// P5.2.C.1 — ZML q_proj uniquement, reader layer 15 (sliding).
//
// Objectif : valider une seule projection lineaire en ZML, comparer le
// resultat byte-equivalent (~1e-5 attendu) contre l'oracle PyTorch fp32
// `q_after_proj` produit en P5.2.C.0.
//
// Pipeline ZML strict :
//   q_after_proj = hidden_input.dot(q_proj_weight, .h)
//   shape : [.b=1, .s=4, .h=1536] dot [.o=2048, .h=1536] -> [.b=1, .s=4, .o=2048]
//
// Fixture consommee : `fixtures/q_only_reader_layer15.safetensors` (3 tenseurs
// utilises sur 9 du fixture P5.2.C.0 : hidden_input, q_proj_weight, q_after_proj).
// Les 6 autres tenseurs sont declares mais ignores (q_norm/RoPE non touches en C.1).
//
// Interdits stricts P5.2.C.1 :
//   - q_norm
//   - reshape [B,S,n_heads,head_dim]
//   - RoPE
//   - transpose [B,n_heads,S,head_dim]
//   - K/V projection
//   - attention scores / matmul QK / softmax
//   - cache / sliding mask
//
// CLI : gemma4_q_proj <path-to-q_only_reader_layer15.safetensors>

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

// Tolerance / sanity expectations.
const Q_PROJ_TOLERANCE: f32 = 1.0e-4;
const Q_PROJ_FLAT_LEN: usize = @intCast(B * S * O); // 8192

// Fixed-point oracle (extracted from fixture sur 3090) :
// q_after_proj[0, {0,1,3}, :8] en fp32. Resultats PyTorch BLAS.
// flat_offset = s * O + i  (avec O=2048).
const QProjBlock = struct {
    label: []const u8,
    flat_offset: usize,
    expected: []const f32,
};

const Q_PROJ_BLOCKS = [_]QProjBlock{
    .{
        .label = "A [0,0,:8]",
        .flat_offset = 0,
        .expected = &.{
            0.2643347681, 0.2799884081, 2.5499663353, 1.4612329006,
            -0.2991320491, 2.2832589149, 5.4269037247, 0.4486914277,
        },
    },
    .{
        .label = "B [0,1,:8]",
        .flat_offset = 2048,
        .expected = &.{
            0.1150695682, 0.0802796185, 2.6355469227, 1.5939798355,
            -0.3821163774, 0.1613903046, -1.8868535757, -0.0016172305,
        },
    },
    .{
        .label = "C [0,3,:8]",
        .flat_offset = 6144,
        .expected = &.{
            -0.1293292940, -0.7008160353, 4.6593823433, 6.2407579422,
            0.2491496652, 2.7536489964, 5.7634291649, 0.0795039386,
        },
    },
};

/// Fixture C.0 charge depuis q_only_reader_layer15.safetensors.
/// 3 tenseurs declares (les 6 autres du fixture P5.2.C.0 sont ignores ici).
const QProjFixture = struct {
    hidden_input: zml.Tensor,
    q_proj_weight: zml.Tensor,
    q_after_proj_oracle: zml.Tensor,

    pub fn init(store: zml.io.TensorStore.View) QProjFixture {
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
            .q_after_proj_oracle = store.createTensor(
                "q_after_proj",
                .{ .b, .s, .o },
                null,
            ),
        };
    }

    pub fn load(
        self: *const QProjFixture,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *const zml.io.TensorStore,
        shardings: []const zml.sharding.Sharding,
    ) !zml.Bufferized(QProjFixture) {
        return zml.io.load(QProjFixture, self, allocator, io, platform, store, .{
            .shardings = shardings,
            .parallelism = 1,
            .dma_chunks = 1,
            .dma_chunk_size = 16 * 1024 * 1024,
        });
    }

    pub fn unloadBuffers(self: *zml.Bufferized(QProjFixture)) void {
        self.hidden_input.deinit();
        self.q_proj_weight.deinit();
        self.q_after_proj_oracle.deinit();
    }

    /// Forward C.1 : un seul matmul Q.
    ///   hidden_input [.b, .s, .h]  dot  q_proj_weight [.o, .h]  -> [.b, .s, .o]
    /// Reduction sur .h, output garde .b, .s, .o.
    pub fn forward(self: QProjFixture) zml.Tensor {
        return self.hidden_input.dot(self.q_proj_weight, .h);
    }
};

pub fn main(init: std.process.Init) !void {
    const arena = init.arena;
    const allocator = init.gpa;
    const io = init.io;

    const process_args = try init.minimal.args.toSlice(arena.allocator());
    if (process_args.len < 2) {
        log.err("Usage: gemma4_q_proj <path-to-q_only_reader_layer15.safetensors>", .{});
        return error.MissingArgument;
    }
    const fixture_path = process_args[1];

    log.info("P5.2.C.1 — ZML q_proj only (reader layer 15 sliding, no q_norm/RoPE)", .{});
    log.info("Loading fixture from {s}", .{fixture_path});

    var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, fixture_path);
    defer registry.deinit();

    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();

    const model: QProjFixture = .init(store.view());

    log.info("Symbolic shapes:", .{});
    log.info("  hidden_input       : {f}", .{model.hidden_input});
    log.info("  q_proj_weight      : {f}", .{model.q_proj_weight});
    log.info("  q_after_proj_oracle: {f}", .{model.q_after_proj_oracle});

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);

    const replicated_sharding = try zml.sharding.replicatedSharding(platform);

    log.info("Materializing buffers...", .{});
    var buffers = try model.load(arena.allocator(), io, platform, &store, &.{replicated_sharding});
    defer QProjFixture.unloadBuffers(&buffers);
    log.info("Buffers loaded (3 tensors).", .{});

    log.info("Compiling forward (single dot, reduce .h, no q_norm/RoPE)...", .{});
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

    // === Fixed-point blocks (3 x 8 valeurs) ===
    log.info("Fixed-point blocks vs oracle q_after_proj (fp32):", .{});
    var max_block: f32 = 0.0;
    for (Q_PROJ_BLOCKS) |block| {
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

    // === Scan global 8192 valeurs vs oracle ===
    log.info("Scanning full tensor ({d} fp32) vs oracle q_after_proj:", .{Q_PROJ_FLAT_LEN});
    var ref_slice = try buffers.q_after_proj_oracle.toSliceAlloc(allocator, io);
    defer ref_slice.free(allocator);
    const ref_data = ref_slice.items(f32);

    if (ref_data.len != Q_PROJ_FLAT_LEN or data.len != Q_PROJ_FLAT_LEN) {
        log.err("length mismatch: ref={d} data={d} expected={d}", .{ ref_data.len, data.len, Q_PROJ_FLAT_LEN });
        return error.QProjLengthMismatch;
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
    const mean_abs = @as(f32, @floatCast(sum_abs / @as(f64, @floatFromInt(Q_PROJ_FLAT_LEN))));
    log.info("  -> full tensor max_abs : {e:.6} at flat_index {d} (s={d}, o={d})", .{
        max_global, max_idx, max_idx / @as(usize, @intCast(O)), max_idx % @as(usize, @intCast(O)),
    });
    log.info("  -> full tensor mean_abs: {e:.6}", .{mean_abs});

    const max_diff = if (max_global > max_block) max_global else max_block;
    log.info("q_proj global max_diff: {e:.6} (tolerance {e:.1})", .{ max_diff, Q_PROJ_TOLERANCE });
    log.info("  Expected ~1.5e-5 (matmul PJRT-CPU Eigen-like vs PyTorch BLAS, cf P4.4.2 Gate E/J)", .{});

    if (max_diff > Q_PROJ_TOLERANCE) {
        log.err("BLOCK: q_proj max_diff exceeds tolerance", .{});
        return error.QProjFailed;
    }
    log.info("P5.2.C.1 PASS: ZML q_proj reader layer 15 validated vs PyTorch oracle", .{});
    log.info("  (no q_norm, no RoPE, no transpose, no K/V, no attention)", .{});
}
