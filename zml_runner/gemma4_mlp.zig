// P5.2.H — ZML MLP feed-forward : sous-couche FFN (layer 15). Avec P5.2.G, ferme une COUCHE
// DÉCODEUR sliding complète.
//
// Reproduit la 2e moitié de Gemma4TextDecoderLayer.forward (L1408-1427) + Gemma4TextMLP :
//   residual = hidden_states                          (= attn_sublayer_out de P5.2.G)
//   x   = pre_feedforward_layernorm(residual)
//   mlp_out = down_proj( gelu_pytorch_tanh(gate_proj(x)) * up_proj(x) )
//   y   = post_feedforward_layernorm(mlp_out)
//   out = residual + y
//
// IMPORTANT : layer 15 est KV-shared (reader) + config.use_double_wide_mlp=True
// -> intermediate = 2 * 6144 = 12288 (modeling_gemma4 L1057-1060). Layers 0-14 = 6144.
//
// gelu_pytorch_tanh = zml Tensor.gelu (0.5x(1+tanh(sqrt(2/pi)(x+0.044715x^3)))), confirmé identique.
// Gating : gelu(gate) * up (activation sur gate SEUL), PAS gelu(gate*up). Linears bias=False.
// pre/post_feedforward_layernorm = Gemma4RMSNorm with_scale (pattern Llama * weight).
//
// Pipeline ZML (dot mirror C.1 q_proj, rmsNorm+mul mirror C.2/P5.2.G) :
//   residual [.b=1,.q=4,.d=1536]
//   gate/up_proj_weight [.f=12288,.d=1536] ; down_proj_weight [.d=1536,.f=12288]
//   x      = rmsNorm(residual,.d,1e-6).mul(pre_ff_w.broad)
//   gate   = x.dot(gate_proj_weight,.d)   {.b,.q,.f}
//   up     = x.dot(up_proj_weight,.d)     {.b,.q,.f}
//   gated  = gate.gelu().mul(up)          {.b,.q,.f}
//   mlp_out= gated.dot(down_proj_weight,.f)   {.b,.q,.d}
//   y      = rmsNorm(mlp_out,.d,1e-6).mul(post_ff_w.broad)
//   out    = residual.add(y)
//
// Comparer vs oracle PyTorch `mlp_sublayer_out` (modules réels Gemma4RMSNorm + ACT2FN), tol 1e-4.
// Interdits : attention, input_layernorm, per_layer_input (PLE), layer 14.
//
// CLI : gemma4_mlp <path-to-p5_2_h_mlp_layer15.safetensors>

const std = @import("std");
const log = std.log;

const zml = @import("zml");
const stdx = zml.stdx;

pub const std_options: std.Options = .{ .log_level = .info };

const B: i64 = 1;
const SQ: i64 = 4;
const D: i64 = 1536; // hidden
const Fdim: i64 = 12288; // intermediate (double-wide)
const RMS_EPS: f32 = 1.0e-6;

const MLP_TOLERANCE: f32 = 1.0e-4;
const MAGNITUDE_CEIL: f32 = 200.0; // out observé max|.| ~ 104 (résidual stream Gemma)
const FLAT_LEN: usize = @intCast(B * SQ * D); // 6144

const STRIDE_B: usize = @intCast(SQ * D); // 6144
const STRIDE_Q: usize = @intCast(D); // 1536

const HBlock = struct { label: []const u8, flat_offset: usize, expected: []const f32 };

const H_BLOCKS = [_]HBlock{
    .{
        .label = "mlp_sublayer_out[0,0,:8]",
        .flat_offset = 0,
        .expected = &.{ 6.1159014702, 7.8243889809, -5.3175573349, -8.8693828583, 2.8058314323, -3.2395372391, -8.6709432602, 4.3606405258 },
    },
    .{
        .label = "mlp_sublayer_out[0,3,:8]",
        .flat_offset = 4608, // q=3 -> 3*1536
        .expected = &.{ 4.0757942200, -9.4939498901, -8.2037515640, -3.4798879623, 2.0659213066, 7.8533105850, 3.9700179100, -0.6982295513 },
    },
};

const MlpFixture = struct {
    residual: zml.Tensor,
    gate_proj_weight: zml.Tensor,
    up_proj_weight: zml.Tensor,
    down_proj_weight: zml.Tensor,
    pre_ff_norm_weight: zml.Tensor,
    post_ff_norm_weight: zml.Tensor,
    mlp_sublayer_out_oracle: zml.Tensor,

    pub fn init(store: zml.io.TensorStore.View) MlpFixture {
        return .{
            .residual = store.createTensor("residual", .{ .b, .q, .d }, null),
            .gate_proj_weight = store.createTensor("gate_proj_weight", .{ .f, .d }, null),
            .up_proj_weight = store.createTensor("up_proj_weight", .{ .f, .d }, null),
            .down_proj_weight = store.createTensor("down_proj_weight", .{ .d, .f }, null),
            .pre_ff_norm_weight = store.createTensor("pre_ff_norm_weight", .{.d}, null),
            .post_ff_norm_weight = store.createTensor("post_ff_norm_weight", .{.d}, null),
            .mlp_sublayer_out_oracle = store.createTensor("mlp_sublayer_out", .{ .b, .q, .d }, null),
        };
    }

    pub fn load(
        self: *const MlpFixture,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *const zml.io.TensorStore,
        shardings: []const zml.sharding.Sharding,
    ) !zml.Bufferized(MlpFixture) {
        return zml.io.load(MlpFixture, self, allocator, io, platform, store, .{
            .shardings = shardings,
            .parallelism = 1,
            .dma_chunks = 1,
            .dma_chunk_size = 16 * 1024 * 1024,
        });
    }

    pub fn unloadBuffers(self: *zml.Bufferized(MlpFixture)) void {
        self.residual.deinit();
        self.gate_proj_weight.deinit();
        self.up_proj_weight.deinit();
        self.down_proj_weight.deinit();
        self.pre_ff_norm_weight.deinit();
        self.post_ff_norm_weight.deinit();
        self.mlp_sublayer_out_oracle.deinit();
    }

    /// Forward P5.2.H : sous-couche MLP complète.
    pub fn forward(self: MlpFixture) zml.Tensor {
        const x_norm = zml.nn.rmsNorm(self.residual, .d, RMS_EPS);
        const x = x_norm.mul(self.pre_ff_norm_weight.broad(x_norm.shape()));
        const gate = x.dot(self.gate_proj_weight, .d); // {.b,.q,.f}
        const up = x.dot(self.up_proj_weight, .d); // {.b,.q,.f}
        const gated = gate.gelu().mul(up); // {.b,.q,.f}
        const mlp_out = gated.dot(self.down_proj_weight, .f); // {.b,.q,.d}
        const y_norm = zml.nn.rmsNorm(mlp_out, .d, RMS_EPS);
        const y = y_norm.mul(self.post_ff_norm_weight.broad(y_norm.shape()));
        return self.residual.add(y);
    }
};

pub fn main(init: std.process.Init) !void {
    const arena = init.arena;
    const allocator = init.gpa;
    const io = init.io;

    const process_args = try init.minimal.args.toSlice(arena.allocator());
    if (process_args.len < 2) {
        log.err("Usage: gemma4_mlp <path-to-p5_2_h_mlp_layer15.safetensors>", .{});
        return error.MissingArgument;
    }
    const fixture_path = process_args[1];

    log.info("P5.2.H — ZML MLP feed-forward (layer 15, intermediate=12288 double-wide, gelu_pytorch_tanh)", .{});
    log.info("Loading fixture from {s}", .{fixture_path});

    var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, fixture_path);
    defer registry.deinit();

    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();

    const model: MlpFixture = .init(store.view());

    log.info("Symbolic shapes:", .{});
    log.info("  residual         : {f}", .{model.residual});
    log.info("  gate_proj_weight : {f}", .{model.gate_proj_weight});
    log.info("  down_proj_weight : {f}", .{model.down_proj_weight});
    log.info("  mlp_sublayer_out : {f}", .{model.mlp_sublayer_out_oracle});

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);

    const replicated_sharding = try zml.sharding.replicatedSharding(platform);

    log.info("Materializing buffers (7 tensors, ~226 MB)...", .{});
    var buffers = try model.load(arena.allocator(), io, platform, &store, &.{replicated_sharding});
    defer MlpFixture.unloadBuffers(&buffers);
    log.info("Buffers loaded.", .{});

    log.info("Compiling forward (pre_norm -> gate/up -> gelu*up -> down -> post_norm -> +residual)...", .{});
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

    log.info("Forward result shape: {f} (expected [b=1, q=4, d=1536])", .{result.shape()});

    var slice = try result.toSliceAlloc(allocator, io);
    defer slice.free(allocator);
    const data = slice.items(f32);

    if (data.len != FLAT_LEN) {
        log.err("length mismatch: data={d} expected={d}", .{ data.len, FLAT_LEN });
        return error.MlpLengthMismatch;
    }

    var has_nan_inf = false;
    var max_mag: f32 = 0.0;
    for (data) |v| {
        if (std.math.isNan(v) or std.math.isInf(v)) has_nan_inf = true;
        const av = @abs(v);
        if (av > max_mag) max_mag = av;
    }
    log.info("Sanity: NaN/Inf={}, max|out|={d:.4} (ceil {d:.1})", .{ has_nan_inf, max_mag, MAGNITUDE_CEIL });
    if (has_nan_inf) {
        log.err("BLOCK: ZML out contains NaN/Inf", .{});
        return error.MlpNanInf;
    }
    if (max_mag > MAGNITUDE_CEIL) {
        log.err("BLOCK: |out| {d:.4} > ceil {d:.1}", .{ max_mag, MAGNITUDE_CEIL });
        return error.MlpMagnitude;
    }

    log.info("Fixed-point blocks vs oracle mlp_sublayer_out (fp32):", .{});
    var max_block: f32 = 0.0;
    for (H_BLOCKS) |block| {
        var block_max: f32 = 0.0;
        log.info("  Block {s} (flat_offset={d}):", .{ block.label, block.flat_offset });
        for (block.expected, 0..) |expected, i| {
            const actual = data[block.flat_offset + i];
            const diff = @abs(actual - expected);
            if (diff > block_max) block_max = diff;
            log.info("    +{d:>3}: actual={d:.7} expected={d:.7} diff={e:.3}", .{ i, actual, expected, diff });
        }
        log.info("    block max_diff: {e:.6}", .{block_max});
        if (block_max > max_block) max_block = block_max;
    }
    log.info("  -> 2 blocks max_diff: {e:.6}", .{max_block});

    log.info("Scanning full tensor ({d} fp32) vs oracle mlp_sublayer_out:", .{FLAT_LEN});
    var ref_slice = try buffers.mlp_sublayer_out_oracle.toSliceAlloc(allocator, io);
    defer ref_slice.free(allocator);
    const ref_data = ref_slice.items(f32);

    if (ref_data.len != FLAT_LEN) {
        log.err("length mismatch: ref={d} expected={d}", .{ ref_data.len, FLAT_LEN });
        return error.MlpLengthMismatch;
    }

    var max_global: f32 = 0.0;
    var sum_abs: f64 = 0.0;
    var max_idx: usize = 0;
    var max_rel: f32 = 0.0;
    for (data, ref_data, 0..) |actual, expected, i| {
        const diff = @abs(actual - expected);
        if (diff > max_global) {
            max_global = diff;
            max_idx = i;
        }
        const rel = diff / (@abs(expected) + 1.0e-6);
        if (rel > max_rel) max_rel = rel;
        sum_abs += @as(f64, diff);
    }
    const mean_abs = @as(f32, @floatCast(sum_abs / @as(f64, @floatFromInt(FLAT_LEN))));

    const b_idx = max_idx / STRIDE_B;
    const q_idx = (max_idx % STRIDE_B) / STRIDE_Q;
    const d_idx = max_idx % STRIDE_Q;
    log.info("  -> full tensor max_abs : {e:.6} at flat_index {d} (b={d}, q={d}, d={d})", .{ max_global, max_idx, b_idx, q_idx, d_idx });
    log.info("  -> full tensor mean_abs: {e:.6} | max_rel: {e:.6}", .{ mean_abs, max_rel });

    const max_diff = if (max_global > max_block) max_global else max_block;
    log.info("mlp global max_diff: {e:.6} (tolerance {e:.1})", .{ max_diff, MLP_TOLERANCE });
    log.info("  Note: réduction down_proj .f=12288 + magnitudes ~100 -> résidu absolu plus élevé (relatif ~1e-6)", .{});

    if (max_diff > MLP_TOLERANCE) {
        log.err("BLOCK: mlp max_diff exceeds absolute tolerance (vérifier max_rel ci-dessus)", .{});
        return error.MlpFailed;
    }
    log.info("P5.2.H PASS: ZML MLP feed-forward layer 15 validated vs PyTorch oracle", .{});
    log.info("  -> avec P5.2.G : COUCHE DÉCODEUR SLIDING COMPLÈTE (attention + MLP)", .{});
}
