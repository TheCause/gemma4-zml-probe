// P5.2.C.3 — ZML RoPE Q-only, reader layer 15 (sliding).
//
// Objectif : etendre P5.2.C.2 (q_proj + q_norm, PASS) avec la rotation
// positionnelle RoPE sur Q. Comparer contre l'oracle PyTorch fp32
// `q_after_rope` du fixture C.0.
//
// Pipeline ZML strict :
//   q_after_proj = hidden_input.dot(q_proj_weight, .h)                  (reuse C.1)
//   q_4d         = q_after_proj.reshape({1,4,8,256})
//                    .withTags(.{.b, .s, .nh, .hd})                     (piège #1)
//   q_normalized = zml.nn.rmsNorm(q_4d, .hd, RMS_EPS)
//   q_after_norm = q_normalized.mul(q_norm_weight.broad(...))           (pattern Llama)
//   q_after_rope = zml.nn.rope(q_after_norm, null, opts)                (default pos_idx = arange(0,S))
//
// RoPE opts pour Gemma 4 sliding (cf P5.0 config + Transformers source) :
//   layout  = .sequential   (HF style, split-half)
//   scaling = .default      (attention_scaling = 1.0)
//   theta   = 10_000        (sliding_attention rope_theta)
//
// Math equivalence verifiee :
//   ZML        : y_real = x_real*cos - x_imag*sin ; y_imag = x_real*sin + x_imag*cos
//   HF Gemma 4 : q_embed = q*cos + rotate_half(q)*sin (cos/sin dupliques en 2 halves)
//   -> equivalent strict pour layout .sequential (cf preuve mathematique
//      dans la cartographie P5.0 + zml/nn.zig L260-L295)
//
// Convention tags ZML rope (cf llama/model.zig L508) :
//   - x doit avoir .s (sequence) et .hd (head dim, even)
//   - autres axes (ici .b, .nh) preserves
//
// Interdits stricts P5.2.C.3 :
//   - transpose final [.b, .nh, .s, .hd]
//   - K/V projection
//   - attention scores / matmul QK / softmax
//   - cache / sliding mask
//
// CLI : gemma4_q_rope <path-to-q_only_reader_layer15.safetensors>

const std = @import("std");
const log = std.log;

const zml = @import("zml");
const stdx = zml.stdx;

pub const std_options: std.Options = .{
    .log_level = .info,
};

// Invariants P5.2.C.0.
const B: i64 = 1;
const S: i64 = 4;
const H: i64 = 1536;
const O: i64 = 2048;
const N: i64 = 8;
const D: i64 = 256;
const RMS_EPS: f32 = 1.0e-6;
const ROPE_THETA: f32 = 10_000;

const Q_ROPE_TOLERANCE: f32 = 1.0e-4;
const Q_ROPE_FLAT_LEN: usize = @intCast(B * S * N * D);

// Fixed-point oracle extraits depuis fixture C.0.
// q_after_rope[0, {0,3}, {0,7}, :8].
// flat_offset = s*2048 + n*256 (shape [1,4,8,256], strides (8192,2048,256,1)).
const QRopeBlock = struct {
    label: []const u8,
    flat_offset: usize,
    expected: []const f32,
};

const Q_ROPE_BLOCKS = [_]QRopeBlock{
    // Position 0 : RoPE est l'identite (cos=1, sin=0).
    // q_after_rope[0,0,*,*] == q_after_norm[0,0,*,*] exactement.
    .{
        .label = "A [0,0,0,:8]  pos=0 RoPE identity",
        .flat_offset = 0,
        .expected = &.{
            0.2076200247, 0.2199150771, 2.0028545856, 1.1477159262,
            -0.2349513322, 1.7933709621, 4.2625265121, 0.3524217606,
        },
    },
    .{
        .label = "B [0,0,7,:8]  pos=0 RoPE identity",
        .flat_offset = 1792,
        .expected = &.{
            -0.4908642471, 0.1689578891, 2.2943143845, -2.0207083225,
            -0.0024840503, 2.9049944878, 1.7889719009, -0.3205637634,
        },
    },
    // Position 3 : RoPE active (rotation non triviale).
    .{
        .label = "C [0,3,0,:8]  pos=3 RoPE active",
        .flat_offset = 6144,
        .expected = &.{
            0.0838326439, 0.4273985922, -1.4685909748, -1.8279951811,
            0.1036225408, 0.2575258613, -2.7275817394, -0.0115460772,
        },
    },
    .{
        .label = "D [0,3,7,:8]  pos=3 RoPE active",
        .flat_offset = 7936,
        .expected = &.{
            -0.2664306164, -0.3861649334, -0.4999011755, -3.2045185566,
            0.0073138960, 0.5368316174, -1.5780971050, 0.3352223933,
        },
    },
};

/// Fixture C.0 — 4 tenseurs consommes en C.3 (sur 9). Les 5 autres ignores.
/// On ne consomme PAS rotary_cos / rotary_sin du fixture car ZML recalcule
/// la RoPE depuis (theta, head_dim, pos_idx) en interne. Si les deux paths
/// (PyTorch precomputed vs ZML recompute) donnent des resultats au-dessus
/// de tolerance, on saura qu'il y a un mismatch d'inv_freq/scaling.
const QRopeFixture = struct {
    hidden_input: zml.Tensor,
    q_proj_weight: zml.Tensor,
    q_norm_weight: zml.Tensor,
    q_after_rope_oracle: zml.Tensor,

    pub fn init(store: zml.io.TensorStore.View) QRopeFixture {
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
                .{.hd},
                null,
            ),
            .q_after_rope_oracle = store.createTensor(
                "q_after_rope",
                .{ .b, .s, .nh, .hd },
                null,
            ),
        };
    }

    pub fn load(
        self: *const QRopeFixture,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *const zml.io.TensorStore,
        shardings: []const zml.sharding.Sharding,
    ) !zml.Bufferized(QRopeFixture) {
        return zml.io.load(QRopeFixture, self, allocator, io, platform, store, .{
            .shardings = shardings,
            .parallelism = 1,
            .dma_chunks = 1,
            .dma_chunk_size = 16 * 1024 * 1024,
        });
    }

    pub fn unloadBuffers(self: *zml.Bufferized(QRopeFixture)) void {
        self.hidden_input.deinit();
        self.q_proj_weight.deinit();
        self.q_norm_weight.deinit();
        self.q_after_rope_oracle.deinit();
    }

    /// Forward C.3 : q_proj + q_norm (C.2 PASS) + RoPE Q-only.
    /// Resultat : Tensor({b=1, s=4, nh=8, hd=256, f32}).
    pub fn forward(self: QRopeFixture) zml.Tensor {
        // C.1 : projection lineaire Q, reduit .h.
        const q_after_proj = self.hidden_input.dot(self.q_proj_weight, .h);

        // C.2 : reshape + re-tag (piege #1), rmsNorm, mul broad (pattern Llama).
        const q_4d = q_after_proj
            .reshape(.{ B, S, N, D })
            .withTags(.{ .b, .s, .nh, .hd });
        const q_normalized = zml.nn.rmsNorm(q_4d, .hd, RMS_EPS);
        const q_after_norm = q_normalized.mul(
            self.q_norm_weight.broad(q_normalized.shape()),
        );

        // C.3 : RoPE Q-only via helper natif ZML.
        // pos_idx = null -> default arange(0, x.dim(.s)) = [0,1,2,3] tag .s
        // layout = .sequential (HF style, split-half) match Gemma 4 apply_rotary_pos_emb.
        // scaling = .default (attention_scaling = 1.0) match Gemma 4 sliding rope_type=default.
        const rope_opts: zml.nn.RopeOpts = .{
            .layout = .sequential,
            .scaling = .{ .default = .{ .rope_theta = ROPE_THETA } },
        };
        return zml.nn.rope(q_after_norm, null, rope_opts);
    }
};

pub fn main(init: std.process.Init) !void {
    const arena = init.arena;
    const allocator = init.gpa;
    const io = init.io;

    const process_args = try init.minimal.args.toSlice(arena.allocator());
    if (process_args.len < 2) {
        log.err("Usage: gemma4_q_rope <path-to-q_only_reader_layer15.safetensors>", .{});
        return error.MissingArgument;
    }
    const fixture_path = process_args[1];

    log.info("P5.2.C.3 — ZML RoPE Q-only (q_proj + q_norm + rope), reader layer 15 sliding", .{});
    log.info("  layout=.sequential  scaling=.default  rope_theta={d}", .{ROPE_THETA});
    log.info("Loading fixture from {s}", .{fixture_path});

    var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, fixture_path);
    defer registry.deinit();

    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();

    const model: QRopeFixture = .init(store.view());

    log.info("Symbolic shapes:", .{});
    log.info("  hidden_input        : {f}", .{model.hidden_input});
    log.info("  q_proj_weight       : {f}", .{model.q_proj_weight});
    log.info("  q_norm_weight       : {f}", .{model.q_norm_weight});
    log.info("  q_after_rope_oracle : {f}", .{model.q_after_rope_oracle});

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);

    const replicated_sharding = try zml.sharding.replicatedSharding(platform);

    log.info("Materializing buffers...", .{});
    var buffers = try model.load(arena.allocator(), io, platform, &store, &.{replicated_sharding});
    defer QRopeFixture.unloadBuffers(&buffers);
    log.info("Buffers loaded (4 tensors).", .{});

    log.info("Compiling forward (q_proj + reshape + rmsNorm + mul + rope, no transpose/K/V)...", .{});
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

    // === 4 fixed-point blocks ===
    log.info("Fixed-point blocks vs oracle q_after_rope (fp32):", .{});
    var max_block: f32 = 0.0;
    for (Q_ROPE_BLOCKS) |block| {
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

    // === Scan global ===
    log.info("Scanning full tensor ({d} fp32) vs oracle q_after_rope:", .{Q_ROPE_FLAT_LEN});
    var ref_slice = try buffers.q_after_rope_oracle.toSliceAlloc(allocator, io);
    defer ref_slice.free(allocator);
    const ref_data = ref_slice.items(f32);

    if (ref_data.len != Q_ROPE_FLAT_LEN or data.len != Q_ROPE_FLAT_LEN) {
        log.err("length mismatch: ref={d} data={d} expected={d}", .{ ref_data.len, data.len, Q_ROPE_FLAT_LEN });
        return error.QRopeLengthMismatch;
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
    const mean_abs = @as(f32, @floatCast(sum_abs / @as(f64, @floatFromInt(Q_ROPE_FLAT_LEN))));

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
    log.info("q_rope global max_diff: {e:.6} (tolerance {e:.1})", .{ max_diff, Q_ROPE_TOLERANCE });

    if (max_diff > Q_ROPE_TOLERANCE) {
        log.err("BLOCK: q_rope max_diff exceeds tolerance", .{});
        log.err("  suspects: (1) wrong layout (interleaved vs sequential),", .{});
        log.err("            (2) wrong rope_theta, (3) wrong inv_freq formula,", .{});
        log.err("            (4) wrong rotation formula sign, (5) wrong pos_idx", .{});
        return error.QRopeFailed;
    }
    log.info("P5.2.C.3 PASS: ZML RoPE Q-only reader layer 15 validated vs PyTorch oracle", .{});
    log.info("  (no transpose, no K/V, no attention)", .{});
}
