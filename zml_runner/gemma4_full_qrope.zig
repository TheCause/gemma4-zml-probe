// P5.6 — ZML full_attention Q-rope MANUELLE partielle (layer 14). Dé-risque les couches full.
//
// full_attention = head_dim 512 (global_head_dim, pas 256), partial rotary 0.25 (128/512 dims
// tournent), rope_type=proportional theta=1e6, scaling=1.0. `zml.nn.rope` ne couvre PAS
// proportional -> RoPE MANUELLE : cos/sin (512-wide, du module réel, oracle) en fixture +
// application `q*cos + rotate_half(q)*sin` à la main. La structure partielle est portée par les
// valeurs de cos/sin (384/512 = identité) -> aucune logique "partial" côté ZML.
//
// Pipeline ZML (Q-path layer 14) :
//   q = hidden_input.dot(q_proj_weight,.h)                       {.b,.s,.o=4096}
//   q = q.reshape([1,4,8,512]).withTags(.{.b,.s,.nh,.hd})        (piège #1 reshape perd tags)
//   q = rmsNorm(q,.hd,1e-6).mul(q_norm_weight.broad)             {.b,.s,.nh,.hd=512}
//   halves = q.split(.hd, {256,256}) -> [first, second]
//   rh = concatenate(-second, first, .hd)                        rotate_half(q)
//   q_rope = q.mul(cos.broad).add(rh.mul(sin.broad))             cos/sin {.s,.hd=512}
//
// rotate_half(x) = cat(-x[256:512], x[0:256]) (split-half, = modeling_gemma4 rotate_half).
// Comparer vs oracle PyTorch `q_after_rope` [1,4,8,512] (= apply_rotary_pos_emb), tol 1e-4.
// Sanity inline confirmé côté oracle : RoPE manuelle == apply_rotary_pos_emb à 0.0.
//
// Le reste du chemin full (K-rope idem, QK/softmax/context/o_proj) est identique au sliding (E/F)
// avec head_dim 512 -> mécaniquement couvert. Ici on valide la TECHNIQUE rope manuelle partielle.
//
// Interdits : K/V, QK/softmax/context, o_proj, MLP, sliding.
// CLI : gemma4_full_qrope <path-to-p5_6_full_qrope_layer14.safetensors>

const std = @import("std");
const log = std.log;

const zml = @import("zml");
const stdx = zml.stdx;

pub const std_options: std.Options = .{ .log_level = .info };

const B: i64 = 1;
const S: i64 = 4;
const NH: i64 = 8;
const HD: i64 = 512; // full head_dim
const HALF: i64 = 256; // HD/2
const HIN: i64 = 1536;
const O: i64 = NH * HD; // 4096
const RMS_EPS: f32 = 1.0e-6;

const ROPE_TOLERANCE: f32 = 1.0e-4;
const MAGNITUDE_CEIL: f32 = 30.0;
const FLAT_LEN: usize = @intCast(B * S * NH * HD); // 16384

const STRIDE_S: usize = @intCast(NH * HD); // 4096
const STRIDE_NH: usize = @intCast(HD); // 512

const RBlock = struct { label: []const u8, flat_offset: usize, expected: []const f32 };

const R_BLOCKS = [_]RBlock{
    .{
        .label = "q_after_rope[0,0,0,:8]",
        .flat_offset = 0, // s=0, nh=0
        .expected = &.{ -0.2494254410, -0.7813053727, -2.7487080097, 1.0633299351, 1.4764531851, -0.7891241312, 0.5839917660, -0.1105189696 },
    },
    .{
        .label = "q_after_rope[0,3,0,:8]",
        .flat_offset = 12288, // s=3, nh=0 -> 3*4096
        .expected = &.{ -0.4871959388, -1.3078932762, -2.4838907719, -0.2659058869, -0.6251965761, -0.5595185161, -2.1747450829, 0.7821418047 },
    },
};

const FullQRopeFixture = struct {
    hidden_input: zml.Tensor,
    q_proj_weight: zml.Tensor,
    q_norm_weight: zml.Tensor,
    cos_full: zml.Tensor,
    sin_full: zml.Tensor,
    q_after_rope_oracle: zml.Tensor,

    pub fn init(store: zml.io.TensorStore.View) FullQRopeFixture {
        return .{
            .hidden_input = store.createTensor("hidden_input", .{ .b, .s, .h }, null),
            .q_proj_weight = store.createTensor("q_proj_weight", .{ .o, .h }, null),
            .q_norm_weight = store.createTensor("q_norm_weight", .{.hd}, null),
            .cos_full = store.createTensor("cos_full", .{ .b, .s, .hd }, null),
            .sin_full = store.createTensor("sin_full", .{ .b, .s, .hd }, null),
            .q_after_rope_oracle = store.createTensor("q_after_rope", .{ .b, .s, .nh, .hd }, null),
        };
    }

    pub fn load(
        self: *const FullQRopeFixture,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *const zml.io.TensorStore,
        shardings: []const zml.sharding.Sharding,
    ) !zml.Bufferized(FullQRopeFixture) {
        return zml.io.load(FullQRopeFixture, self, allocator, io, platform, store, .{
            .shardings = shardings,
            .parallelism = 1,
            .dma_chunks = 1,
            .dma_chunk_size = 16 * 1024 * 1024,
        });
    }

    pub fn unloadBuffers(self: *zml.Bufferized(FullQRopeFixture)) void {
        self.hidden_input.deinit();
        self.q_proj_weight.deinit();
        self.q_norm_weight.deinit();
        self.cos_full.deinit();
        self.sin_full.deinit();
        self.q_after_rope_oracle.deinit();
    }

    /// Forward P5.6 : Q-path full attention avec RoPE manuelle partielle.
    pub fn forward(self: FullQRopeFixture) zml.Tensor {
        const q_proj = self.hidden_input.dot(self.q_proj_weight, .h); // {.b,.s,.o=4096}
        const q_4d = q_proj.reshape(.{ B, S, NH, HD }).withTags(.{ .b, .s, .nh, .hd });
        const q_normed = zml.nn.rmsNorm(q_4d, .hd, RMS_EPS);
        const q = q_normed.mul(self.q_norm_weight.broad(q_normed.shape()));

        // rotate_half(q) = cat(-q[256:512], q[0:256]) sur .hd
        const halves = q.split(.hd, &.{ HALF, HALF });
        const rh = zml.Tensor.concatenate(&.{ halves[1].negate(), halves[0] }, .hd);

        // q_rope = q*cos + rotate_half(q)*sin (cos/sin {.s,.hd} broadcast sur .b,.nh)
        const term_cos = q.mul(self.cos_full.broad(q.shape()));
        const term_sin = rh.mul(self.sin_full.broad(q.shape()));
        return term_cos.add(term_sin);
    }
};

pub fn main(init: std.process.Init) !void {
    const arena = init.arena;
    const allocator = init.gpa;
    const io = init.io;

    const process_args = try init.minimal.args.toSlice(arena.allocator());
    if (process_args.len < 2) {
        log.err("Usage: gemma4_full_qrope <path-to-p5_6_full_qrope_layer14.safetensors>", .{});
        return error.MissingArgument;
    }
    const fixture_path = process_args[1];

    log.info("P5.6 — ZML full_attention Q-rope manuelle (layer 14, head_dim=512, partial 0.25, theta1e6)", .{});
    log.info("Loading fixture from {s}", .{fixture_path});

    var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, fixture_path);
    defer registry.deinit();

    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();

    const model: FullQRopeFixture = .init(store.view());

    log.info("Symbolic shapes:", .{});
    log.info("  hidden_input  : {f}", .{model.hidden_input});
    log.info("  q_proj_weight : {f}", .{model.q_proj_weight});
    log.info("  cos_full      : {f}", .{model.cos_full});
    log.info("  q_after_rope  : {f}", .{model.q_after_rope_oracle});

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);

    const replicated_sharding = try zml.sharding.replicatedSharding(platform);

    log.info("Materializing buffers (6 tensors)...", .{});
    var buffers = try model.load(arena.allocator(), io, platform, &store, &.{replicated_sharding});
    defer FullQRopeFixture.unloadBuffers(&buffers);
    log.info("Buffers loaded.", .{});

    log.info("Compiling forward (q_proj -> q_norm -> rotate_half(split/neg/concat) -> q*cos+rh*sin)...", .{});
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

    log.info("Forward result shape: {f} (expected [b=1, s=4, nh=8, hd=512])", .{result.shape()});

    var slice = try result.toSliceAlloc(allocator, io);
    defer slice.free(allocator);
    const data = slice.items(f32);

    if (data.len != FLAT_LEN) {
        log.err("length mismatch: data={d} expected={d}", .{ data.len, FLAT_LEN });
        return error.RopeLengthMismatch;
    }

    var has_nan_inf = false;
    var max_mag: f32 = 0.0;
    for (data) |v| {
        if (std.math.isNan(v) or std.math.isInf(v)) has_nan_inf = true;
        const av = @abs(v);
        if (av > max_mag) max_mag = av;
    }
    log.info("Sanity: NaN/Inf={}, max|q_rope|={d:.4} (ceil {d:.1})", .{ has_nan_inf, max_mag, MAGNITUDE_CEIL });
    if (has_nan_inf) return error.RopeNanInf;
    if (max_mag > MAGNITUDE_CEIL) {
        log.err("BLOCK: |q_rope| {d:.4} > ceil {d:.1}", .{ max_mag, MAGNITUDE_CEIL });
        return error.RopeMagnitude;
    }

    log.info("Fixed-point blocks vs oracle q_after_rope (fp32):", .{});
    var max_block: f32 = 0.0;
    for (R_BLOCKS) |block| {
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

    log.info("Scanning full tensor ({d} fp32) vs oracle q_after_rope:", .{FLAT_LEN});
    var ref_slice = try buffers.q_after_rope_oracle.toSliceAlloc(allocator, io);
    defer ref_slice.free(allocator);
    const ref_data = ref_slice.items(f32);

    if (ref_data.len != FLAT_LEN) return error.RopeLengthMismatch;

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

    const s_idx = (max_idx % @as(usize, @intCast(S * NH * HD))) / STRIDE_S;
    const nh_idx = (max_idx % STRIDE_S) / STRIDE_NH;
    const hd_idx = max_idx % STRIDE_NH;
    log.info("  -> full tensor max_abs : {e:.6} at flat_index {d} (s={d}, nh={d}, hd={d})", .{ max_global, max_idx, s_idx, nh_idx, hd_idx });
    log.info("  -> full tensor mean_abs: {e:.6}", .{mean_abs});

    const max_diff = if (max_global > max_block) max_global else max_block;
    log.info("full_qrope global max_diff: {e:.6} (tolerance {e:.1})", .{ max_diff, ROPE_TOLERANCE });
    log.info("  Expected ~1e-5 (q_proj matmul .h=1536 ; rope = mul/add, RoPE orthogonale préserve)", .{});

    if (max_diff > ROPE_TOLERANCE) {
        log.err("BLOCK: full_qrope max_diff exceeds tolerance", .{});
        return error.RopeFailed;
    }
    log.info("P5.6 PASS: ZML full_attention Q-rope manuelle partielle (layer 14) validated vs PyTorch oracle", .{});
    log.info("  -> technique RoPE manuelle (cos/sin oracle + split/neg/concat) validée pour head_dim 512 partial", .{});
}
