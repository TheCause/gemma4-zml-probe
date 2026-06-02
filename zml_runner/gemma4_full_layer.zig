// P5.7.4 — ZML couche décodeur FULL attention COMPLÈTE (layer 14 producer). Dispatcher full path.
//
// Généralise P5.3 (couche sliding) à full_attention : head_dim 512, RoPE MANUELLE partielle
// (split/neg/concat + cos/sin oracle 512-wide, cf P5.6/P5.6.K) au lieu de zml.nn.rope. Le reste
// de la couche (QKV/norms/QK/mask/softmax/context/o_proj/MLP/bloc PLE/layer_scalar) = identique
// au sliding avec dims 512. Oracle = module RÉEL Gemma4TextDecoderLayer(14).
//
// Tags : .h=hidden(1536), .nh=heads, .hd=head_dim(512), .s/.q/.k=seq(4), .f=mlp(6144), .p=ple(256),
// .m=concat têtes(4096). CLI : gemma4_full_layer <path-to-p5_7_4_full_layer14.safetensors>

const std = @import("std");
const log = std.log;
const zml = @import("zml");
const stdx = zml.stdx;

pub const std_options: std.Options = .{ .log_level = .info };

const B: i64 = 1;
const S: i64 = 4;
const H: i64 = 1536;
const NH: i64 = 8;
const HD: i64 = 512; // full head_dim
const HALF: i64 = 256;
const RMS_EPS: f32 = 1.0e-6;
const LAYER_SCALAR: f64 = 0.028564453125;

const LAYER_TOLERANCE: f32 = 5.0e-4;
const MAGNITUDE_CEIL: f32 = 50.0;
const FLAT_LEN: usize = @intCast(B * S * H); // 6144
const STRIDE_S: usize = @intCast(H);

const LBlock = struct { label: []const u8, flat_offset: usize, expected: []const f32 };
const L_BLOCKS = [_]LBlock{
    .{ .label = "layer_out[0,0,:8]", .flat_offset = 0, .expected = &.{ -0.3454087675, 0.5323283076, 1.5569403172, 1.4758952856, 0.0665154308, -0.4156963527, 1.2878988981, -1.2639403343 } },
    .{ .label = "layer_out[0,3,:8]", .flat_offset = 4608, .expected = &.{ 0.4377623498, -1.9567040205, -1.4764313698, 0.0504851043, 0.2480003685, -0.4823885560, 0.5605818629, 0.6772518754 } },
};

const FullLayerFixture = struct {
    layer_input: zml.Tensor,
    per_layer_input: zml.Tensor,
    cos_full: zml.Tensor,
    sin_full: zml.Tensor,
    attn_mask: zml.Tensor,
    layer_out_oracle: zml.Tensor,
    input_ln: zml.Tensor,
    q_proj: zml.Tensor,
    q_norm: zml.Tensor,
    k_proj: zml.Tensor,
    k_norm: zml.Tensor,
    v_proj: zml.Tensor,
    o_proj: zml.Tensor,
    post_attn_ln: zml.Tensor,
    pre_ff_ln: zml.Tensor,
    gate_proj: zml.Tensor,
    up_proj: zml.Tensor,
    down_proj: zml.Tensor,
    post_ff_ln: zml.Tensor,
    ple_gate: zml.Tensor,
    ple_proj: zml.Tensor,
    ple_norm: zml.Tensor,

    pub fn init(store: zml.io.TensorStore.View) FullLayerFixture {
        return .{
            .layer_input = store.createTensor("layer_input", .{ .b, .s, .h }, null),
            .per_layer_input = store.createTensor("per_layer_input", .{ .b, .s, .p }, null),
            .cos_full = store.createTensor("cos_full", .{ .b, .s, .hd }, null),
            .sin_full = store.createTensor("sin_full", .{ .b, .s, .hd }, null),
            .attn_mask = store.createTensor("attn_mask", .{ .b, .h, .q, .k }, null),
            .layer_out_oracle = store.createTensor("layer_out", .{ .b, .s, .h }, null),
            .input_ln = store.createTensor("w__input_layernorm__weight", .{.h}, null),
            .q_proj = store.createTensor("w__self_attn__q_proj__weight", .{ .o, .h }, null),
            .q_norm = store.createTensor("w__self_attn__q_norm__weight", .{.hd}, null),
            .k_proj = store.createTensor("w__self_attn__k_proj__weight", .{ .o, .h }, null),
            .k_norm = store.createTensor("w__self_attn__k_norm__weight", .{.hd}, null),
            .v_proj = store.createTensor("w__self_attn__v_proj__weight", .{ .o, .h }, null),
            .o_proj = store.createTensor("w__self_attn__o_proj__weight", .{ .out, .m }, null),
            .post_attn_ln = store.createTensor("w__post_attention_layernorm__weight", .{.h}, null),
            .pre_ff_ln = store.createTensor("w__pre_feedforward_layernorm__weight", .{.h}, null),
            .gate_proj = store.createTensor("w__mlp__gate_proj__weight", .{ .f, .h }, null),
            .up_proj = store.createTensor("w__mlp__up_proj__weight", .{ .f, .h }, null),
            .down_proj = store.createTensor("w__mlp__down_proj__weight", .{ .h, .f }, null),
            .post_ff_ln = store.createTensor("w__post_feedforward_layernorm__weight", .{.h}, null),
            .ple_gate = store.createTensor("w__per_layer_input_gate__weight", .{ .p, .h }, null),
            .ple_proj = store.createTensor("w__per_layer_projection__weight", .{ .h, .p }, null),
            .ple_norm = store.createTensor("w__post_per_layer_input_norm__weight", .{.h}, null),
        };
    }

    pub fn load(
        self: *const FullLayerFixture,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *const zml.io.TensorStore,
        shardings: []const zml.sharding.Sharding,
    ) !zml.Bufferized(FullLayerFixture) {
        return zml.io.load(FullLayerFixture, self, allocator, io, platform, store, .{
            .shardings = shardings,
            .parallelism = 1,
            .dma_chunks = 1,
            .dma_chunk_size = 16 * 1024 * 1024,
        });
    }

    fn rmsScale(x: zml.Tensor, w: zml.Tensor) zml.Tensor {
        const n = zml.nn.rmsNorm(x, .h, RMS_EPS);
        return n.mul(w.broad(n.shape()));
    }

    /// RoPE manuelle partielle : x {.b,.s,.nh,.hd=512} ; cos/sin {.b,.s,.hd}.
    /// rotate_half = cat(-x[256:],x[:256]) ; x*cos + rh*sin.
    fn manualRope(x: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor) zml.Tensor {
        const halves = x.split(.hd, &.{ HALF, HALF });
        const rh = zml.Tensor.concatenate(&.{ halves[1].negate(), halves[0] }, .hd);
        return x.mul(cos.broad(x.shape())).add(rh.mul(sin.broad(x.shape())));
    }

    pub fn forward(self: FullLayerFixture) zml.Tensor {
        const li = self.layer_input;
        const h0 = rmsScale(li, self.input_ln);

        // Q (8 têtes, head_dim 512, rope MANUELLE)
        var q = h0.dot(self.q_proj, .h).reshape(.{ B, S, NH, HD }).withTags(.{ .b, .s, .nh, .hd });
        q = zml.nn.rmsNorm(q, .hd, RMS_EPS).mul(self.q_norm.broad(q.shape()));
        q = manualRope(q, self.cos_full, self.sin_full);
        const q_final = q.transpose(.{ .b, .nh, .s, .hd }).rename(.{ .nh = .h, .s = .q });

        // K (1 tête kv)
        var k = h0.dot(self.k_proj, .h).reshape(.{ B, S, 1, HD }).withTags(.{ .b, .s, .nh, .hd });
        k = zml.nn.rmsNorm(k, .hd, RMS_EPS).mul(self.k_norm.broad(k.shape()));
        k = manualRope(k, self.cos_full, self.sin_full);
        const k_final = k.transpose(.{ .b, .nh, .s, .hd }).rename(.{ .nh = .h, .s = .k });

        // V (1 tête kv, v_norm SANS scale)
        var v = h0.dot(self.v_proj, .h).reshape(.{ B, S, 1, HD }).withTags(.{ .b, .s, .nh, .hd });
        v = zml.nn.rmsNorm(v, .hd, RMS_EPS);
        const v_final = v.transpose(.{ .b, .nh, .s, .hd }).rename(.{ .nh = .h, .s = .k });

        // QK scores (GQA, scaling 1.0) + masque causal + softmax
        const qs = q_final.splitAxis(.h, .{ .h = k_final.dim(.h), .hq = .auto });
        var scores = qs.dot(k_final, .hd).merge(.{ .h = .{ .h, .hq } }).transpose(.{ .b, .h, .q, .k });
        scores = scores.add(self.attn_mask.broad(scores.shape()));
        const probs = scores.softmax(.k);

        // context (GQA) + o_proj (concat têtes -> 4096)
        const ps = probs.splitAxis(.h, .{ .h = v_final.dim(.h), .hq = .auto });
        const ctx = ps.dot(v_final, .k).merge(.{ .h = .{ .h, .hq } }).transpose(.{ .b, .h, .q, .hd });
        const attn_m = ctx.transpose(.{ .b, .q, .h, .hd }).merge(.{ .m = .{ .h, .hd } }); // {.b,.q,.m=4096}
        const attn_out = attn_m.dot(self.o_proj, .m).rename(.{ .q = .s, .out = .h });

        const h1 = li.add(rmsScale(attn_out, self.post_attn_ln));

        // MLP (6144, layer 14 producer)
        const x = rmsScale(h1, self.pre_ff_ln);
        const mlp_out = x.dot(self.gate_proj, .h).gelu().mul(x.dot(self.up_proj, .h)).dot(self.down_proj, .f);
        const h2 = h1.add(rmsScale(mlp_out, self.post_ff_ln));

        // bloc PLE per-layer
        var g = h2.dot(self.ple_gate, .h).gelu();
        g = g.mul(self.per_layer_input);
        g = g.dot(self.ple_proj, .p);
        const h3 = h2.add(rmsScale(g, self.ple_norm));

        return h3.scale(LAYER_SCALAR);
    }
};

pub fn main(init: std.process.Init) !void {
    const arena = init.arena;
    const allocator = init.gpa;
    const io = init.io;

    const process_args = try init.minimal.args.toSlice(arena.allocator());
    if (process_args.len < 2) {
        log.err("Usage: gemma4_full_layer <path-to-p5_7_4_full_layer14.safetensors>", .{});
        return error.MissingArgument;
    }
    const fixture_path = process_args[1];

    log.info("P5.7.4 — ZML couche FULL attention COMPLÈTE (layer 14 producer, head_dim 512, RoPE manuelle)", .{});
    var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, fixture_path);
    defer registry.deinit();
    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();
    const model: FullLayerFixture = .init(store.view());

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);
    const replicated_sharding = try zml.sharding.replicatedSharding(platform);

    log.info("Materializing buffers (23 tensors)...", .{});
    var buffers = try model.load(arena.allocator(), io, platform, &store, &.{replicated_sharding});
    _ = &buffers;

    log.info("Compiling forward (full attention layer : rope manuelle + composition complète)...", .{});
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

    log.info("Forward result shape: {f} (expected [b=1, s=4, h=1536])", .{result.shape()});
    var slice = try result.toSliceAlloc(allocator, io);
    defer slice.free(allocator);
    const data = slice.items(f32);
    if (data.len != FLAT_LEN) return error.LayerLengthMismatch;

    var has_nan_inf = false;
    var max_mag: f32 = 0.0;
    for (data) |val| {
        if (std.math.isNan(val) or std.math.isInf(val)) has_nan_inf = true;
        const av = @abs(val);
        if (av > max_mag) max_mag = av;
    }
    log.info("Sanity: NaN/Inf={}, max|out|={d:.4} (ceil {d:.1})", .{ has_nan_inf, max_mag, MAGNITUDE_CEIL });
    if (has_nan_inf or max_mag > MAGNITUDE_CEIL) return error.LayerSanity;

    var max_block: f32 = 0.0;
    for (L_BLOCKS) |block| {
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

    var ref_slice = try buffers.layer_out_oracle.toSliceAlloc(allocator, io);
    defer ref_slice.free(allocator);
    const ref_data = ref_slice.items(f32);
    if (ref_data.len != FLAT_LEN) return error.LayerLengthMismatch;
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
    const s_idx = (max_idx % @as(usize, @intCast(S * H))) / STRIDE_S;
    const h_idx = max_idx % STRIDE_S;
    log.info("  -> full tensor max_abs : {e:.6} at (s={d}, h={d}), mean_abs {e:.6}", .{ max_global, s_idx, h_idx, mean_abs });

    const max_diff = if (max_global > max_block) max_global else max_block;
    log.info("full_layer global max_diff: {e:.6} (tolerance {e:.1})", .{ max_diff, LAYER_TOLERANCE });
    if (max_diff > LAYER_TOLERANCE) return error.LayerFailed;
    log.info("P5.7.4 PASS: ZML couche FULL attention complète (layer 14) validated vs PyTorch oracle", .{});
    log.info("  -> dispatcher full path validé : head_dim 512 + RoPE manuelle dans la composition complète", .{});
}
