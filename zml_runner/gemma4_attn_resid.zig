// P5.2.G — ZML post_attention_layernorm + résiduel : ferme la SOUS-COUCHE ATTENTION (layer 15).
//
// Reproduit la fin du bloc attention de Gemma4TextDecoderLayer.forward (modeling_gemma4.py
// 5.9.0 L1405-1406) :
//   hidden_states = post_attention_layernorm(attn_output)   // Gemma4RMSNorm with_scale
//   hidden_states = residual + hidden_states                 // résiduel
//
// Pipeline ZML strict (mirror C.2 q_norm pour le rmsNorm + mul, plus un add) :
//   attn_output [.b=1, .q=4, .d=1536]   (input, o_proj_out de P5.2.F)
//   residual    [.b=1, .q=4, .d=1536]   (input, stand-in = hidden_input C.0)
//   pa_norm_weight [.d=1536]            (input, post_attention_layernorm.weight)
//   normed = zml.nn.rmsNorm(attn_output, .d, 1e-6)
//   scaled = normed.mul(pa_norm_weight.broad(normed.shape()))   // pattern Llama (* weight)
//   out    = residual.add(scaled)
//
// Gemma4RMSNorm = pattern Llama (normalized * weight, init weight=1), PAS Qwen (1+weight).
// Comparer out vs l'oracle PyTorch `attn_sublayer_out` (= module réel Gemma4RMSNorm + add),
// figé en P5.2.G. Tolérance 1e-4.
//
// Note résiduel : `residual` est un stand-in (hidden_input C.0). Le vrai résiduel est le hidden
// state pré-input_layernorm ; le pilote synthétique ne modélise pas input_layernorm. La gate
// valide l'OP (post_attn_norm + add) ; oracle et ZML consomment le MÊME residual.
//
// Indépendance oracle : `attn_sublayer_out` vient du module Gemma4RMSNorm + add (torch) ; ce
// runner utilise zml.nn.rmsNorm + mul + add ZML natif. Aucun code partagé.
//
// Interdits stricts P5.2.G : MLP, pre/post_feedforward_layernorm, input_layernorm, layer 14.
//
// CLI : gemma4_attn_resid <path-to-p5_2_g_attn_residual_layer15.safetensors>

const std = @import("std");
const log = std.log;

const zml = @import("zml");
const stdx = zml.stdx;

pub const std_options: std.Options = .{
    .log_level = .info,
};

const B: i64 = 1;
const SQ: i64 = 4;
const D: i64 = 1536; // hidden_size
const RMS_EPS: f32 = 1.0e-6;

const G_TOLERANCE: f32 = 1.0e-4;
const MAGNITUDE_CEIL: f32 = 20.0; // out observé max|.| ~ 9.0
const FLAT_LEN: usize = @intCast(B * SQ * D); // 6144

// Strides du tenseur out [b, q, d] (row-major).
const STRIDE_B: usize = @intCast(SQ * D); // 6144
const STRIDE_Q: usize = @intCast(D); // 1536

// Fixed-point oracle (attn_sublayer_out fp32 extrait du log de scripts/27_*.py).
const GBlock = struct {
    label: []const u8,
    flat_offset: usize,
    expected: []const f32,
};

const G_BLOCKS = [_]GBlock{
    .{
        .label = "attn_sublayer_out[0,0,:8]",
        .flat_offset = 0, // q=0
        .expected = &.{ -0.3466566801, -0.1356179118, 1.2983992100, -2.7226529121, 0.5595096350, -1.4223707914, 0.3725267649, -0.1077312753 },
    },
    .{
        .label = "attn_sublayer_out[0,3,:8]",
        .flat_offset = 4608, // q=3 -> 3*1536
        .expected = &.{ 2.5543491840, -1.2166686058, 0.8269227743, -0.7040416002, 0.8201341629, 2.7698321342, -0.4033173323, 0.2113059163 },
    },
};

/// Fixture P5.2.G : attn_output, residual, pa_norm_weight (inputs), attn_sublayer_out (oracle).
const AttnResidFixture = struct {
    attn_output: zml.Tensor,
    residual: zml.Tensor,
    pa_norm_weight: zml.Tensor,
    attn_sublayer_out_oracle: zml.Tensor,

    pub fn init(store: zml.io.TensorStore.View) AttnResidFixture {
        return .{
            .attn_output = store.createTensor("attn_output", .{ .b, .q, .d }, null),
            .residual = store.createTensor("residual", .{ .b, .q, .d }, null),
            .pa_norm_weight = store.createTensor("pa_norm_weight", .{.d}, null),
            .attn_sublayer_out_oracle = store.createTensor("attn_sublayer_out", .{ .b, .q, .d }, null),
        };
    }

    pub fn load(
        self: *const AttnResidFixture,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *const zml.io.TensorStore,
        shardings: []const zml.sharding.Sharding,
    ) !zml.Bufferized(AttnResidFixture) {
        return zml.io.load(AttnResidFixture, self, allocator, io, platform, store, .{
            .shardings = shardings,
            .parallelism = 1,
            .dma_chunks = 1,
            .dma_chunk_size = 16 * 1024 * 1024,
        });
    }

    pub fn unloadBuffers(self: *zml.Bufferized(AttnResidFixture)) void {
        self.attn_output.deinit();
        self.residual.deinit();
        self.pa_norm_weight.deinit();
        self.attn_sublayer_out_oracle.deinit();
    }

    /// Forward P5.2.G : out = residual + post_attention_layernorm(attn_output).
    ///   normed = zml.nn.rmsNorm(attn_output, .d, 1e-6)
    ///   scaled = normed.mul(pa_norm_weight.broad(...))    (pattern Llama * weight)
    ///   out    = residual.add(scaled)
    /// Pas de MLP/feedforward norms/input_layernorm.
    pub fn forward(self: AttnResidFixture) zml.Tensor {
        const normed = zml.nn.rmsNorm(self.attn_output, .d, RMS_EPS);
        const scaled = normed.mul(self.pa_norm_weight.broad(normed.shape()));
        return self.residual.add(scaled);
    }
};

pub fn main(init: std.process.Init) !void {
    const arena = init.arena;
    const allocator = init.gpa;
    const io = init.io;

    const process_args = try init.minimal.args.toSlice(arena.allocator());
    if (process_args.len < 2) {
        log.err("Usage: gemma4_attn_resid <path-to-p5_2_g_attn_residual_layer15.safetensors>", .{});
        return error.MissingArgument;
    }
    const fixture_path = process_args[1];

    log.info("P5.2.G — ZML post_attention_layernorm + résiduel (layer 15, ferme la sous-couche attention)", .{});
    log.info("Loading fixture from {s}", .{fixture_path});

    var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, fixture_path);
    defer registry.deinit();

    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();

    const model: AttnResidFixture = .init(store.view());

    log.info("Symbolic shapes:", .{});
    log.info("  attn_output              : {f}", .{model.attn_output});
    log.info("  residual                 : {f}", .{model.residual});
    log.info("  pa_norm_weight           : {f}", .{model.pa_norm_weight});
    log.info("  attn_sublayer_out_oracle : {f}", .{model.attn_sublayer_out_oracle});

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);

    const replicated_sharding = try zml.sharding.replicatedSharding(platform);

    log.info("Materializing buffers...", .{});
    var buffers = try model.load(arena.allocator(), io, platform, &store, &.{replicated_sharding});
    defer AttnResidFixture.unloadBuffers(&buffers);
    log.info("Buffers loaded (4 tensors).", .{});

    log.info("Compiling forward (rmsNorm .d -> mul weight -> add residual)...", .{});
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

    log.info("Forward result shape: {f} (expected [b=1, q=4, d=1536])", .{result.shape()});

    var slice = try result.toSliceAlloc(allocator, io);
    defer slice.free(allocator);
    const data = slice.items(f32);

    if (data.len != FLAT_LEN) {
        log.err("length mismatch: data={d} expected={d}", .{ data.len, FLAT_LEN });
        return error.GLengthMismatch;
    }

    // === Sanity sortie ZML ===
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
        return error.GNanInf;
    }
    if (max_mag > MAGNITUDE_CEIL) {
        log.err("BLOCK: |out| {d:.4} > ceil {d:.1}", .{ max_mag, MAGNITUDE_CEIL });
        return error.GMagnitude;
    }

    // === Fixed-point blocks (2 x 8 valeurs) vs oracle ===
    log.info("Fixed-point blocks vs oracle attn_sublayer_out (fp32):", .{});
    var max_block: f32 = 0.0;
    for (G_BLOCKS) |block| {
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
    log.info("  -> 2 blocks max_diff: {e:.6}", .{max_block});

    // === Scan global 6144 valeurs vs oracle ===
    log.info("Scanning full tensor ({d} fp32) vs oracle attn_sublayer_out:", .{FLAT_LEN});
    var ref_slice = try buffers.attn_sublayer_out_oracle.toSliceAlloc(allocator, io);
    defer ref_slice.free(allocator);
    const ref_data = ref_slice.items(f32);

    if (ref_data.len != FLAT_LEN) {
        log.err("length mismatch: ref={d} expected={d}", .{ ref_data.len, FLAT_LEN });
        return error.GLengthMismatch;
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
    const mean_abs = @as(f32, @floatCast(sum_abs / @as(f64, @floatFromInt(FLAT_LEN))));

    const b_idx = max_idx / STRIDE_B;
    const q_idx = (max_idx % STRIDE_B) / STRIDE_Q;
    const d_idx = max_idx % STRIDE_Q;
    log.info("  -> full tensor max_abs : {e:.6} at flat_index {d} (b={d}, q={d}, d={d})", .{
        max_global, max_idx, b_idx, q_idx, d_idx,
    });
    log.info("  -> full tensor mean_abs: {e:.6}", .{mean_abs});

    const max_diff = if (max_global > max_block) max_global else max_block;
    log.info("attn_resid global max_diff: {e:.6} (tolerance {e:.1})", .{ max_diff, G_TOLERANCE });
    log.info("  Expected ~1e-6 (rmsNorm + mul + add fp32, cf q_norm C.2 6.7e-6)", .{});

    if (max_diff > G_TOLERANCE) {
        log.err("BLOCK: attn_resid max_diff exceeds tolerance", .{});
        return error.GFailed;
    }
    log.info("P5.2.G PASS: ZML post_attention_layernorm + résiduel layer 15 validated vs PyTorch oracle", .{});
    log.info("  -> SOUS-COUCHE ATTENTION COMPLÈTE : qkv proj/norm/rope -> QK -> mask -> softmax -> context -> o_proj -> post_attn_norm -> +residual", .{});
}
