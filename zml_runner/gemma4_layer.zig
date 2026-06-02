// P5.3 — ZML COUCHE DÉCODEUR sliding COMPLÈTE (layer 13 producer). Capstone de composition.
//
// Compose en UN forward tous les maillons validés (E/F/G/H) + input_layernorm + bloc PLE per-layer
// + layer_scalar, vs l'oracle module RÉEL Gemma4TextDecoderLayer (L1395-1438).
//
// Producer sliding : calcule sa propre K/V (head_dim 256), intermediate MLP 6144. RoPE sliding =
// zml.nn.rope (default theta 10000, validé C.3/D.4). Attention scaling 1.0 (q_norm porte la norm).
//
// Tags : .h=hidden(1536) en proj ; .nh=heads, .hd=head_dim(256), .s/.q/.k=seq(4) ; .f=mlp(6144) ;
// .p=ple(256) ; .m=concat têtes(2048). rename pour distinguer query-seq (.q) et key-seq (.k) au QK.
//
// layer_scalar = 0.08837890625 (buffer checkpoint ; réduit la sortie ~11x).
//
// CLI : gemma4_layer <path-to-p5_3_layer13.safetensors>

const std = @import("std");
const log = std.log;

const zml = @import("zml");
const stdx = zml.stdx;

pub const std_options: std.Options = .{ .log_level = .info };

const B: i64 = 1;
const S: i64 = 4;
const H: i64 = 1536;
const NH: i64 = 8;
const HD: i64 = 256;
const RMS_EPS: f32 = 1.0e-6;
const ROPE_THETA: f32 = 10000.0;
const LAYER_SCALAR: f64 = 0.08837890625;

const LAYER_TOLERANCE: f32 = 5.0e-4;
const MAGNITUDE_CEIL: f32 = 30.0;
const FLAT_LEN: usize = @intCast(B * S * H); // 6144
const STRIDE_S: usize = @intCast(H); // 1536

const LBlock = struct { label: []const u8, flat_offset: usize, expected: []const f32 };

const L_BLOCKS = [_]LBlock{
    .{
        .label = "layer_out[0,0,:8]",
        .flat_offset = 0,
        .expected = &.{ 1.2880382538, -1.0072543621, -0.7650865316, -0.4851128161, 1.2144860029, -0.4182901084, 0.7489302754, 0.8393705487 },
    },
    .{
        .label = "layer_out[0,3,:8]",
        .flat_offset = 4608,
        .expected = &.{ 1.1569122076, -1.6173239946, -0.6220484972, -0.1988275349, 3.4741263390, 5.2958526611, 0.7212421298, -0.7273856401 },
    },
};

const LayerFixture = struct {
    layer_input: zml.Tensor,
    per_layer_input: zml.Tensor,
    attn_mask: zml.Tensor,
    layer_out_oracle: zml.Tensor,
    // poids (noms fixture w__<sub avec __>)
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

    pub fn init(store: zml.io.TensorStore.View) LayerFixture {
        return .{
            .layer_input = store.createTensor("layer_input", .{ .b, .s, .h }, null),
            .per_layer_input = store.createTensor("per_layer_input", .{ .b, .s, .p }, null),
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
        self: *const LayerFixture,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *const zml.io.TensorStore,
        shardings: []const zml.sharding.Sharding,
    ) !zml.Bufferized(LayerFixture) {
        return zml.io.load(LayerFixture, self, allocator, io, platform, store, .{
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

    /// Forward complet de la couche décodeur sliding (producer).
    pub fn forward(self: LayerFixture) zml.Tensor {
        const li = self.layer_input; // {.b,.s,.h} residual1

        // --- input_layernorm ---
        const h0 = rmsScale(li, self.input_ln); // {.b,.s,.h}

        // --- Q path (8 têtes, head_dim 256, rope sliding) ---
        var q = h0.dot(self.q_proj, .h).reshape(.{ B, S, NH, HD }).withTags(.{ .b, .s, .nh, .hd });
        q = zml.nn.rmsNorm(q, .hd, RMS_EPS).mul(self.q_norm.broad(q.shape()));
        q = zml.nn.rope(q, null, .{ .layout = .sequential, .scaling = .{ .default = .{ .rope_theta = ROPE_THETA } } });
        const q_final = q.transpose(.{ .b, .nh, .s, .hd }).rename(.{ .nh = .h, .s = .q }); // {.b,.h=8,.q,.hd}

        // --- K path (1 tête kv) ---
        var k = h0.dot(self.k_proj, .h).reshape(.{ B, S, 1, HD }).withTags(.{ .b, .s, .nh, .hd });
        k = zml.nn.rmsNorm(k, .hd, RMS_EPS).mul(self.k_norm.broad(k.shape()));
        k = zml.nn.rope(k, null, .{ .layout = .sequential, .scaling = .{ .default = .{ .rope_theta = ROPE_THETA } } });
        const k_final = k.transpose(.{ .b, .nh, .s, .hd }).rename(.{ .nh = .h, .s = .k }); // {.b,.h=1,.k,.hd}

        // --- V path (1 tête kv, v_norm SANS scale) ---
        var v = h0.dot(self.v_proj, .h).reshape(.{ B, S, 1, HD }).withTags(.{ .b, .s, .nh, .hd });
        v = zml.nn.rmsNorm(v, .hd, RMS_EPS); // with_scale=False -> pas de mul
        const v_final = v.transpose(.{ .b, .nh, .s, .hd }).rename(.{ .nh = .h, .s = .k }); // {.b,.h=1,.k,.hd}

        // --- QK scores (GQA split, scaling 1.0) ---
        const qs = q_final.splitAxis(.h, .{ .h = k_final.dim(.h), .hq = .auto });
        var scores = qs.dot(k_final, .hd).merge(.{ .h = .{ .h, .hq } }).transpose(.{ .b, .h, .q, .k });
        scores = scores.add(self.attn_mask.broad(scores.shape()));
        const probs = scores.softmax(.k);

        // --- context (probs @ V, GQA) ---
        const ps = probs.splitAxis(.h, .{ .h = v_final.dim(.h), .hq = .auto });
        const ctx = ps.dot(v_final, .k).merge(.{ .h = .{ .h, .hq } }).transpose(.{ .b, .h, .q, .hd });

        // --- o_proj (concat têtes + dot) ---
        const attn_m = ctx.transpose(.{ .b, .q, .h, .hd }).merge(.{ .m = .{ .h, .hd } }); // {.b,.q,.m=2048}
        const attn_out = attn_m.dot(self.o_proj, .m).rename(.{ .q = .s, .out = .h }); // {.b,.s,.h}

        // --- post_attn_norm + residual1 ---
        const h1 = li.add(rmsScale(attn_out, self.post_attn_ln)); // {.b,.s,.h}

        // --- MLP (intermediate 6144) ---
        const x = rmsScale(h1, self.pre_ff_ln);
        const gate = x.dot(self.gate_proj, .h);
        const up = x.dot(self.up_proj, .h);
        const mlp_out = gate.gelu().mul(up).dot(self.down_proj, .f); // {.b,.s,.h}
        const h2 = h1.add(rmsScale(mlp_out, self.post_ff_ln));

        // --- bloc PLE per-layer ---
        var g = h2.dot(self.ple_gate, .h).gelu(); // {.b,.s,.p}
        g = g.mul(self.per_layer_input);
        g = g.dot(self.ple_proj, .p); // {.b,.s,.h}
        const h3 = h2.add(rmsScale(g, self.ple_norm));

        // --- layer_scalar ---
        return h3.scale(LAYER_SCALAR);
    }
};

pub fn main(init: std.process.Init) !void {
    const arena = init.arena;
    const allocator = init.gpa;
    const io = init.io;

    const process_args = try init.minimal.args.toSlice(arena.allocator());
    if (process_args.len < 2) {
        log.err("Usage: gemma4_layer <path-to-p5_3_layer13.safetensors>", .{});
        return error.MissingArgument;
    }
    const fixture_path = process_args[1];

    log.info("P5.3 — ZML couche décodeur sliding COMPLÈTE (layer 13 producer : attn + MLP + PLE block)", .{});
    log.info("Loading fixture from {s}", .{fixture_path});

    var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, fixture_path);
    defer registry.deinit();

    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();

    const model: LayerFixture = .init(store.view());

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);

    const replicated_sharding = try zml.sharding.replicatedSharding(platform);

    log.info("Materializing buffers (23 tensors, ~145 MB)...", .{});
    var buffers = try model.load(arena.allocator(), io, platform, &store, &.{replicated_sharding});
    log.info("Buffers loaded.", .{});

    log.info("Compiling forward (input_ln -> attn(QKV/rope/QK/mask/softmax/ctx/o_proj) -> +res -> MLP -> +res -> PLE -> scalar)...", .{});
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

    if (data.len != FLAT_LEN) {
        log.err("length mismatch: data={d} expected={d}", .{ data.len, FLAT_LEN });
        return error.LayerLengthMismatch;
    }

    var has_nan_inf = false;
    var max_mag: f32 = 0.0;
    for (data) |v| {
        if (std.math.isNan(v) or std.math.isInf(v)) has_nan_inf = true;
        const av = @abs(v);
        if (av > max_mag) max_mag = av;
    }
    log.info("Sanity: NaN/Inf={}, max|out|={d:.4} (ceil {d:.1})", .{ has_nan_inf, max_mag, MAGNITUDE_CEIL });
    if (has_nan_inf) return error.LayerNanInf;
    if (max_mag > MAGNITUDE_CEIL) return error.LayerMagnitude;

    log.info("Fixed-point blocks vs oracle layer_out (fp32):", .{});
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

    log.info("Scanning full tensor ({d} fp32) vs oracle layer_out:", .{FLAT_LEN});
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
    log.info("  -> full tensor max_abs : {e:.6} at flat_index {d} (s={d}, h={d})", .{ max_global, max_idx, s_idx, h_idx });
    log.info("  -> full tensor mean_abs: {e:.6}", .{mean_abs});

    const max_diff = if (max_global > max_block) max_global else max_block;
    log.info("layer global max_diff: {e:.6} (tolerance {e:.1})", .{ max_diff, LAYER_TOLERANCE });
    log.info("  Note: composition ~7 matmuls + rope + softmax + PLE ; layer_scalar 0.088 réduit le résidu ~11x", .{});

    if (max_diff > LAYER_TOLERANCE) {
        log.err("BLOCK: layer max_diff exceeds tolerance", .{});
        return error.LayerFailed;
    }
    log.info("P5.3 PASS: ZML couche décodeur sliding complète (layer 13) validated vs PyTorch oracle", .{});
    log.info("  -> COMPOSITION END-TO-END VALIDÉE : input_ln + attention + MLP + bloc PLE + layer_scalar", .{});
}
