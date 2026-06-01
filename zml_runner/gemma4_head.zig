// P5.5 — ZML tête de sortie : final norm + lm_head (tied) + softcap. Fin du forward.
//
// hidden -> norm (Gemma4RMSNorm) -> lm_head (= embed_tokens.weight tied) -> 30*tanh(logits/30).
// Op neuve = softcap (Tensor.tanh + scale). lm_head pleine table 262144 = 1.6GB impraticable
// -> slice vocab 4096 (lm_head pleine table = mécaniquement identique).
//
// Pipeline ZML :
//   normed = rmsNorm(hidden_final,.d,1e-6).mul(norm_weight.broad)   {.b,.s,.d}
//   logits = normed.dot(lm_head_slice,.d)                           {.b,.s,.voc}
//   logits_out = logits.scale(1/30).tanh().scale(30)                softcap
//
// Comparer vs oracle PyTorch `logits_out` (module réel Gemma4RMSNorm + F.linear + softcap), tol 1e-4.
// CLI : gemma4_head <path-to-p5_5_head.safetensors>

const std = @import("std");
const log = std.log;

const zml = @import("zml");
const stdx = zml.stdx;

pub const std_options: std.Options = .{ .log_level = .info };

const B: i64 = 1;
const S: i64 = 4;
const D: i64 = 1536;
const VOC: i64 = 4096;
const RMS_EPS: f32 = 1.0e-6;
const SOFTCAP: f64 = 30.0;

const HEAD_TOLERANCE: f32 = 1.0e-4;
const MAGNITUDE_CEIL: f32 = 31.0; // softcap borne ±30
const FLAT_LEN: usize = @intCast(B * S * VOC); // 16384
const STRIDE_S: usize = @intCast(VOC); // 4096

const HBlock = struct { label: []const u8, flat_offset: usize, expected: []const f32 };

const H_BLOCKS = [_]HBlock{
    .{
        .label = "logits_out[0,0,:8]",
        .flat_offset = 0,
        .expected = &.{ 24.6611194611, -7.7168369293, -15.6979589462, 24.8031272888, 18.6575393677, 21.0394477844, 20.4767494202, 23.1501712799 },
    },
    .{
        .label = "logits_out[0,3,:8]",
        .flat_offset = 12288, // q=3 -> 3*4096
        .expected = &.{ -12.2130241394, -20.3034820557, 1.6355872154, -12.5123329163, -23.6553554535, 8.8580255508, -9.0673875809, -4.1106548309 },
    },
};

const HeadFixture = struct {
    hidden_final: zml.Tensor,
    norm_weight: zml.Tensor,
    lm_head_slice: zml.Tensor,
    logits_out_oracle: zml.Tensor,

    pub fn init(store: zml.io.TensorStore.View) HeadFixture {
        return .{
            .hidden_final = store.createTensor("hidden_final", .{ .b, .s, .d }, null),
            .norm_weight = store.createTensor("norm_weight", .{.d}, null),
            .lm_head_slice = store.createTensor("lm_head_slice", .{ .voc, .d }, null),
            .logits_out_oracle = store.createTensor("logits_out", .{ .b, .s, .voc }, null),
        };
    }

    pub fn load(
        self: *const HeadFixture,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *const zml.io.TensorStore,
        shardings: []const zml.sharding.Sharding,
    ) !zml.Bufferized(HeadFixture) {
        return zml.io.load(HeadFixture, self, allocator, io, platform, store, .{
            .shardings = shardings,
            .parallelism = 1,
            .dma_chunks = 1,
            .dma_chunk_size = 16 * 1024 * 1024,
        });
    }

    pub fn unloadBuffers(self: *zml.Bufferized(HeadFixture)) void {
        self.hidden_final.deinit();
        self.norm_weight.deinit();
        self.lm_head_slice.deinit();
        self.logits_out_oracle.deinit();
    }

    /// Forward P5.5 : final norm -> lm_head -> softcap.
    pub fn forward(self: HeadFixture) zml.Tensor {
        const n = zml.nn.rmsNorm(self.hidden_final, .d, RMS_EPS);
        const normed = n.mul(self.norm_weight.broad(n.shape()));
        const logits = normed.dot(self.lm_head_slice, .d); // {.b,.s,.voc}
        // softcap = 30 * tanh(logits / 30)
        return logits.scale(1.0 / SOFTCAP).tanh().scale(SOFTCAP);
    }
};

pub fn main(init: std.process.Init) !void {
    const arena = init.arena;
    const allocator = init.gpa;
    const io = init.io;

    const process_args = try init.minimal.args.toSlice(arena.allocator());
    if (process_args.len < 2) {
        log.err("Usage: gemma4_head <path-to-p5_5_head.safetensors>", .{});
        return error.MissingArgument;
    }
    const fixture_path = process_args[1];

    log.info("P5.5 — ZML head : final norm + lm_head(tied) + softcap 30*tanh(x/30) (vocab slice 4096)", .{});
    log.info("Loading fixture from {s}", .{fixture_path});

    var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, fixture_path);
    defer registry.deinit();

    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();

    const model: HeadFixture = .init(store.view());

    log.info("Symbolic shapes:", .{});
    log.info("  hidden_final  : {f}", .{model.hidden_final});
    log.info("  lm_head_slice : {f}", .{model.lm_head_slice});
    log.info("  logits_out    : {f}", .{model.logits_out_oracle});

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);

    const replicated_sharding = try zml.sharding.replicatedSharding(platform);

    var buffers = try model.load(arena.allocator(), io, platform, &store, &.{replicated_sharding});
    defer HeadFixture.unloadBuffers(&buffers);
    log.info("Buffers loaded (4 tensors).", .{});

    log.info("Compiling forward (rmsNorm -> dot lm_head -> softcap tanh)...", .{});
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

    log.info("Forward result shape: {f} (expected [b=1, s=4, voc=4096])", .{result.shape()});

    var slice = try result.toSliceAlloc(allocator, io);
    defer slice.free(allocator);
    const data = slice.items(f32);

    if (data.len != FLAT_LEN) {
        log.err("length mismatch: data={d} expected={d}", .{ data.len, FLAT_LEN });
        return error.HeadLengthMismatch;
    }

    var has_nan_inf = false;
    var max_mag: f32 = 0.0;
    for (data) |v| {
        if (std.math.isNan(v) or std.math.isInf(v)) has_nan_inf = true;
        const av = @abs(v);
        if (av > max_mag) max_mag = av;
    }
    log.info("Sanity: NaN/Inf={}, max|logits|={d:.4} (ceil {d:.1}, softcap borne ±30)", .{ has_nan_inf, max_mag, MAGNITUDE_CEIL });
    if (has_nan_inf) return error.HeadNanInf;
    if (max_mag > MAGNITUDE_CEIL) {
        log.err("BLOCK: |logits| {d:.4} > ceil {d:.1} (softcap cassé ?)", .{ max_mag, MAGNITUDE_CEIL });
        return error.HeadMagnitude;
    }

    log.info("Fixed-point blocks vs oracle logits_out (fp32):", .{});
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

    log.info("Scanning full tensor ({d} fp32) vs oracle logits_out:", .{FLAT_LEN});
    var ref_slice = try buffers.logits_out_oracle.toSliceAlloc(allocator, io);
    defer ref_slice.free(allocator);
    const ref_data = ref_slice.items(f32);
    if (ref_data.len != FLAT_LEN) return error.HeadLengthMismatch;

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
    const s_idx = (max_idx % @as(usize, @intCast(S * VOC))) / STRIDE_S;
    const voc_idx = max_idx % STRIDE_S;
    log.info("  -> full tensor max_abs : {e:.6} at flat_index {d} (s={d}, voc={d})", .{ max_global, max_idx, s_idx, voc_idx });
    log.info("  -> full tensor mean_abs: {e:.6}", .{mean_abs});

    const max_diff = if (max_global > max_block) max_global else max_block;
    log.info("head global max_diff: {e:.6} (tolerance {e:.1})", .{ max_diff, HEAD_TOLERANCE });
    log.info("  Expected ~1e-5 (lm_head matmul .d=1536 ; softcap tanh smooth)", .{});

    if (max_diff > HEAD_TOLERANCE) {
        log.err("BLOCK: head max_diff exceeds tolerance", .{});
        return error.HeadFailed;
    }
    log.info("P5.5 PASS: ZML head (final norm + lm_head tied + softcap) validated vs PyTorch oracle", .{});
}
