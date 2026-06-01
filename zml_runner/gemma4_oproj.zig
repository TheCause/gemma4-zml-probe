// P5.2.F — ZML o_proj : projection de sortie de l'attention (layer 15 reader).
//
// Dernier maillon du BLOC attention (après context E.context, AVANT résiduel/MLP). Reproduit
// `Gemma4TextAttention.forward` fin (modeling_gemma4.py 5.9.0 L1272-1273) + le transpose(1,2)
// de eager_attention_forward (L834) que le `context` E.0 ne contient pas :
//   attn_output = context.transpose(1,2).reshape(b, q, n_heads*head_dim)   [1,4,2048]
//   o_out       = o_proj(attn_output)   (nn.Linear 2048->1536, bias=False, PAS de clipping)
//
// o_proj TEXTE = nn.Linear simple (Gemma4ClippableLinear = attention VISION, hors sujet).
//
// Pipeline ZML strict (concat têtes via transpose+merge, dot mirror C.1 q_proj) :
//   context [.b=1, .h=8, .q=4, .hd=256]   (input, sortie E.context, layout [b,h,q,hd])
//   o_proj_weight [.o=1536, .m=2048]      (input, o=hidden out, m=n_heads*head_dim in)
//   attn  = context.transpose(.{ .b, .q, .h, .hd })           // [.b,.q,.h,.hd]  = transpose(1,2)
//   attn  = attn.merge(.{ .m = .{ .h, .hd } })                // [.b,.q,.m=2048] concat h-major
//   o_out = attn.dot(o_proj_weight, .m)                       // [.b,.q,.o=1536] contract .m
//
// merge(.{ .m = .{ .h, .hd } }) sur [.b,.q,.h,.hd] -> m = h*head_dim + hd (h-major, hd-mineur),
// exactement le `reshape(b,q,h*hd)` PyTorch après transpose(1,2). Comparer o_out vs l'oracle
// PyTorch `o_proj_out` (= F.linear(context.transpose(1,2).reshape, W)) figé en P5.2.F. Tol 1e-4.
//
// Indépendance oracle : `o_proj_out` vient de torch (F.linear) ; ce runner fait transpose/merge/dot
// ZML natif. Aucun code partagé.
//
// Interdits stricts P5.2.F : résiduel, post_attention_layernorm, MLP, layer 14, re-attention,
// clipping (vision only).
//
// CLI : gemma4_oproj <path-to-p5_2_f_oproj_layer15.safetensors>

const std = @import("std");
const log = std.log;

const zml = @import("zml");
const stdx = zml.stdx;

pub const std_options: std.Options = .{
    .log_level = .info,
};

const B: i64 = 1;
const NH: i64 = 8;
const SQ: i64 = 4;
const HD: i64 = 256;
const HIDDEN: i64 = 1536; // o_proj out
const M: i64 = NH * HD; // 2048 = concat têtes

const OPROJ_TOLERANCE: f32 = 1.0e-4;
const MAGNITUDE_CEIL: f32 = 20.0; // o_out observé max|.| ~ 13.6
const FLAT_LEN: usize = @intCast(B * SQ * HIDDEN); // 6144

// Strides du tenseur o_out [b, q, o] (row-major).
const STRIDE_B: usize = @intCast(SQ * HIDDEN); // 6144
const STRIDE_Q: usize = @intCast(HIDDEN); // 1536

// Fixed-point oracle (o_proj_out fp32 extrait du log de scripts/26_*.py).
const OBlock = struct {
    label: []const u8,
    flat_offset: usize,
    expected: []const f32,
};

const O_BLOCKS = [_]OBlock{
    .{
        .label = "o_proj_out[0,0,:8]",
        .flat_offset = 0, // q=0
        .expected = &.{ -0.4991632700, -0.0621126890, 1.5691689253, -1.7105869055, -0.0627050102, -1.3702884912, -0.5507981181, -0.1628580093 },
    },
    .{
        .label = "o_proj_out[0,3,:8]",
        .flat_offset = 4608, // q=3 -> 3*1536
        .expected = &.{ 0.6877590418, -0.1717909276, 0.7460835576, 0.7599042058, 0.4761676788, 1.8151590824, -0.5304908752, -0.3873683214 },
    },
};

/// Fixture P5.2.F : context (input), o_proj_weight (input), o_proj_out (oracle).
const OProjFixture = struct {
    context: zml.Tensor,
    o_proj_weight: zml.Tensor,
    o_proj_out_oracle: zml.Tensor,

    pub fn init(store: zml.io.TensorStore.View) OProjFixture {
        return .{
            // context [1,8,4,256] : layout [b,h,q,hd] (sortie E.context, avant transpose(1,2)).
            .context = store.createTensor("context", .{ .b, .h, .q, .hd }, null),
            // o_proj_weight [1536,2048] : .o = hidden out, .m = n_heads*head_dim in (réduit).
            .o_proj_weight = store.createTensor("o_proj_weight", .{ .o, .m }, null),
            // o_proj_out oracle [1,4,1536] : [b, q, o].
            .o_proj_out_oracle = store.createTensor("o_proj_out", .{ .b, .q, .o }, null),
        };
    }

    pub fn load(
        self: *const OProjFixture,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *const zml.io.TensorStore,
        shardings: []const zml.sharding.Sharding,
    ) !zml.Bufferized(OProjFixture) {
        return zml.io.load(OProjFixture, self, allocator, io, platform, store, .{
            .shardings = shardings,
            .parallelism = 1,
            .dma_chunks = 1,
            .dma_chunk_size = 16 * 1024 * 1024,
        });
    }

    pub fn unloadBuffers(self: *zml.Bufferized(OProjFixture)) void {
        self.context.deinit();
        self.o_proj_weight.deinit();
        self.o_proj_out_oracle.deinit();
    }

    /// Forward P5.2.F : concat des têtes (transpose+merge) puis projection o_proj.
    ///   attn  = context.transpose(.{ .b, .q, .h, .hd })     [.b,.q,.h,.hd]  (= transpose(1,2))
    ///   attn  = attn.merge(.{ .m = .{ .h, .hd } })          [.b,.q,.m=2048] (h-major, hd-mineur)
    ///   o_out = attn.dot(o_proj_weight, .m)                 [.b,.q,.o=1536] (mirror C.1 q_proj)
    /// Pas de résiduel/norm/MLP/clipping : QUE la projection de sortie.
    pub fn forward(self: OProjFixture) zml.Tensor {
        const attn_t = self.context.transpose(.{ .b, .q, .h, .hd });
        const attn = attn_t.merge(.{ .m = .{ .h, .hd } });
        return attn.dot(self.o_proj_weight, .m);
    }
};

pub fn main(init: std.process.Init) !void {
    const arena = init.arena;
    const allocator = init.gpa;
    const io = init.io;

    const process_args = try init.minimal.args.toSlice(arena.allocator());
    if (process_args.len < 2) {
        log.err("Usage: gemma4_oproj <path-to-p5_2_f_oproj_layer15.safetensors>", .{});
        return error.MissingArgument;
    }
    const fixture_path = process_args[1];

    log.info("P5.2.F — ZML o_proj (layer 15 reader, concat têtes + nn.Linear 2048->1536, no bias/clip/résiduel)", .{});
    log.info("Loading fixture from {s}", .{fixture_path});

    var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, fixture_path);
    defer registry.deinit();

    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();

    const model: OProjFixture = .init(store.view());

    log.info("Symbolic shapes:", .{});
    log.info("  context           : {f}", .{model.context});
    log.info("  o_proj_weight     : {f}", .{model.o_proj_weight});
    log.info("  o_proj_out_oracle : {f}", .{model.o_proj_out_oracle});

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);

    const replicated_sharding = try zml.sharding.replicatedSharding(platform);

    log.info("Materializing buffers...", .{});
    var buffers = try model.load(arena.allocator(), io, platform, &store, &.{replicated_sharding});
    defer OProjFixture.unloadBuffers(&buffers);
    log.info("Buffers loaded (3 tensors).", .{});

    log.info("Compiling forward (transpose -> merge têtes -> dot .m)...", .{});
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

    log.info("Forward result shape: {f} (expected [b=1, q=4, o=1536])", .{result.shape()});

    var slice = try result.toSliceAlloc(allocator, io);
    defer slice.free(allocator);
    const data = slice.items(f32);

    if (data.len != FLAT_LEN) {
        log.err("length mismatch: data={d} expected={d} (.m mal contracté ou merge faux ?)", .{ data.len, FLAT_LEN });
        return error.OProjLengthMismatch;
    }

    // === Sanity sortie ZML (NaN/Inf + magnitude — détecte concat têtes scramblé) ===
    var has_nan_inf = false;
    var max_mag: f32 = 0.0;
    for (data) |v| {
        if (std.math.isNan(v) or std.math.isInf(v)) has_nan_inf = true;
        const av = @abs(v);
        if (av > max_mag) max_mag = av;
    }
    log.info("Sanity: NaN/Inf={}, max|o_out|={d:.4} (ceil {d:.1})", .{ has_nan_inf, max_mag, MAGNITUDE_CEIL });
    if (has_nan_inf) {
        log.err("BLOCK: ZML o_out contains NaN/Inf", .{});
        return error.OProjNanInf;
    }
    if (max_mag > MAGNITUDE_CEIL) {
        log.err("BLOCK: |o_out| {d:.4} > ceil {d:.1} -> concat des têtes probablement scramblé", .{ max_mag, MAGNITUDE_CEIL });
        return error.OProjMagnitude;
    }

    // === Fixed-point blocks (2 x 8 valeurs) vs oracle o_proj_out ===
    log.info("Fixed-point blocks vs oracle o_proj_out (fp32):", .{});
    var max_block: f32 = 0.0;
    for (O_BLOCKS) |block| {
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
    log.info("Scanning full tensor ({d} fp32) vs oracle o_proj_out:", .{FLAT_LEN});
    var ref_slice = try buffers.o_proj_out_oracle.toSliceAlloc(allocator, io);
    defer ref_slice.free(allocator);
    const ref_data = ref_slice.items(f32);

    if (ref_data.len != FLAT_LEN) {
        log.err("length mismatch: ref={d} expected={d}", .{ ref_data.len, FLAT_LEN });
        return error.OProjLengthMismatch;
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
    const o_idx = max_idx % STRIDE_Q;
    log.info("  -> full tensor max_abs : {e:.6} at flat_index {d} (b={d}, q={d}, o={d})", .{
        max_global, max_idx, b_idx, q_idx, o_idx,
    });
    log.info("  -> full tensor mean_abs: {e:.6}", .{mean_abs});

    const max_diff = if (max_global > max_block) max_global else max_block;
    log.info("o_proj global max_diff: {e:.6} (tolerance {e:.1})", .{ max_diff, OPROJ_TOLERANCE });
    log.info("  Expected ~1e-5 (matmul réduction .m=2048 PJRT-CPU vs PyTorch BLAS, cf q_proj C.1 1.14e-5)", .{});

    if (max_diff > OPROJ_TOLERANCE) {
        log.err("BLOCK: o_proj max_diff exceeds tolerance", .{});
        return error.OProjFailed;
    }
    log.info("P5.2.F PASS: ZML o_proj reader layer 15 validated vs PyTorch oracle", .{});
    log.info("  (concat têtes transpose+merge + nn.Linear 2048->1536 ; no bias, no clip, no résiduel/norm/MLP)", .{});
}
