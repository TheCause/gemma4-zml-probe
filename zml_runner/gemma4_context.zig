// P5.2.E.context — ZML context dot : reader layer 15 (sliding) x KV layer 13 (sliding).
//
// DERNIER maillon de l'attention forward : context = probs @ V (après QK/mask/softmax,
// AVANT o_proj). Ferme P5.2.E -> première chaîne d'attention ZML complète.
//
// Pipeline ZML strict (GQA par split des têtes Q, miroir exact de E.1 QK scores) :
//   probs   [.b=1, .h=8, .q=4, .k=4]    (input, softmax déjà fait en E.softmax)
//   v_final [.b=1, .h=1, .k=4, .hd=256] (input, writer layer 13, V RMSNorm no-scale D.0b ;
//                                        tête KV taggée .h size 1, comme k_final en E.1)
//   probs_split = probs.splitAxis(.h, .{ .h = v_final.dim(.h)=1, .hq = .auto=8 })  [.b,.h=1,.hq=8,.q,.k]
//   context     = probs_split.dot(v_final, .k)                                     [.b,.h=1,.hq=8,.q,.hd]
//   context     = context.merge(.{ .h = .{ .h, .hq } })                            [.b,.h=8,.q,.hd]
//   context     = context.transpose(.{ .b, .h, .q, .hd })                          ordre physique [b,h,q,hd]
//
// GQA : l'unique tête KV (.h=1) est partagée par les 8 têtes Q (.hq=8) via le batch .h=1 ;
// merge(.h,.hq) -> head_merged = h*8 + hq = hq (h∈{0}), exactement comme repeat_kv(v,8) côté
// PyTorch (toutes les têtes Q attendent la même V). Comparer vs l'oracle PyTorch `context`
// (= matmul(probs, repeat_kv(v_final,8))) produit en P5.2.E.0. Tolérance 1e-4.
//
// Gemma 4 : scaling=1.0 (déjà appliqué), softmax fp32 (déjà faite), PAS de softcap. Ici on ne
// fait QUE la contraction probs@V — pas de re-scaling, pas de re-softmax, pas de o_proj.
//
// Indépendance oracle : `context` de référence vient de torch.matmul (E.0) ; ce runner utilise
// la chaîne ZML native splitAxis/dot/merge. Aucun code partagé.
//
// Interdits stricts P5.2.E.context :
//   - o_proj (projection de sortie, vient APRÈS context)
//   - re-softmax / masque (faits en E.softmax / E.mask)
//   - scaling 1/sqrt(head_dim) / softcap d'attention
//   - layer 14 (full attention, p-RoPE proportional)
//
// CLI : gemma4_context <path-to-p5_2_econtext_layer15_kv13.safetensors>

const std = @import("std");
const log = std.log;

const zml = @import("zml");
const stdx = zml.stdx;

pub const std_options: std.Options = .{
    .log_level = .info,
};

// Invariants P5.2.E.0/E.context.
const B: i64 = 1;
const NH: i64 = 8; // n query heads (reader layer 15)
const NKV: i64 = 1; // n kv heads (writer layer 13)
const SQ: i64 = 4; // query positions
const SK: i64 = 4; // key positions (contracted)
const HD: i64 = 256; // head_dim

// Tolerances / sanity.
const CTX_TOLERANCE: f32 = 1.0e-4; // vs oracle context
const MAGNITUDE_CEIL: f32 = 10.0; // |context| borné (probs convexe, V ~ RMSNorm std~1)
const FLAT_LEN: usize = @intCast(B * NH * SQ * HD); // 8192

// Strides du tenseur context [b, h, q, hd] (row-major).
const STRIDE_B: usize = @intCast(NH * SQ * HD); // 8192
const STRIDE_H: usize = @intCast(SQ * HD); // 1024
const STRIDE_Q: usize = @intCast(HD); // 256

// Fixed-point oracle (context fp32 extrait du fixture E.0, cf logs/21_*.log).
// flat_offset = h * STRIDE_H + q * STRIDE_Q  (b=0).
const CtxBlock = struct {
    label: []const u8,
    flat_offset: usize,
    expected: []const f32,
};

const CTX_BLOCKS = [_]CtxBlock{
    .{
        .label = "context[0,0,0,:8]",
        .flat_offset = 0, // h=0, q=0
        .expected = &.{ -1.4737527370, 1.8738104105, 0.1378140152, -0.4008703232, 0.9987069368, -0.7353876233, -0.9977726936, -0.6727036238 },
    },
    .{
        .label = "context[0,0,3,:8]",
        .flat_offset = 768, // h=0, q=3 -> 3*256
        .expected = &.{ 0.6404302120, -0.9312928319, 0.8357610106, 1.4575493336, -0.7226662636, 0.3802416325, -0.9020456672, -1.6392571926 },
    },
    .{
        .label = "context[0,7,3,:8]",
        .flat_offset = 7936, // h=7, q=3 -> 7*1024 + 3*256
        .expected = &.{ 0.5298202634, -0.5505211353, 0.8041542768, 0.1258839369, -0.7421149611, -0.0225164890, -0.6404543519, -1.0496249199 },
    },
};

/// Fixture E.context : probs (input), v_final (input), context (oracle).
const ContextFixture = struct {
    probs: zml.Tensor,
    v_final: zml.Tensor,
    context_oracle: zml.Tensor,

    pub fn init(store: zml.io.TensorStore.View) ContextFixture {
        return .{
            // probs [1,8,4,4] : 8 têtes Q déjà mergées.
            .probs = store.createTensor("probs", .{ .b, .h, .q, .k }, null),
            // v_final [1,1,4,256] : tête KV taggée .h (size 1), positions .k, head_dim .hd.
            .v_final = store.createTensor("v_final", .{ .b, .h, .k, .hd }, null),
            // context oracle [1,8,4,256] : [b, h, q, hd].
            .context_oracle = store.createTensor("context", .{ .b, .h, .q, .hd }, null),
        };
    }

    pub fn load(
        self: *const ContextFixture,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *const zml.io.TensorStore,
        shardings: []const zml.sharding.Sharding,
    ) !zml.Bufferized(ContextFixture) {
        return zml.io.load(ContextFixture, self, allocator, io, platform, store, .{
            .shardings = shardings,
            .parallelism = 1,
            .dma_chunks = 1,
            .dma_chunk_size = 16 * 1024 * 1024,
        });
    }

    pub fn unloadBuffers(self: *zml.Bufferized(ContextFixture)) void {
        self.probs.deinit();
        self.v_final.deinit();
        self.context_oracle.deinit();
    }

    /// Forward E.context : context = probs @ V via GQA (split têtes Q, miroir E.1).
    ///   probs_split = probs.splitAxis(.h, .{ .h = v.dim(.h)=1, .hq = .auto=8 })  [.b,.h=1,.hq=8,.q,.k]
    ///   context     = probs_split.dot(v_final, .k)                               [.b,.h=1,.hq=8,.q,.hd]
    ///   context     = context.merge(.{ .h = .{ .h, .hq } })                      [.b,.h=8,.q,.hd]
    ///   context     = context.transpose(.{ .b, .h, .q, .hd })                    ordre physique
    /// Pas de scaling/softcap/o_proj : QUE la contraction probs@V sur .k.
    pub fn forward(self: ContextFixture) zml.Tensor {
        const probs_split = self.probs.splitAxis(.h, .{ .h = self.v_final.dim(.h), .hq = .auto });
        const context = probs_split.dot(self.v_final, .k);
        const merged = context.merge(.{ .h = .{ .h, .hq } });
        return merged.transpose(.{ .b, .h, .q, .hd });
    }
};

pub fn main(init: std.process.Init) !void {
    const arena = init.arena;
    const allocator = init.gpa;
    const io = init.io;

    const process_args = try init.minimal.args.toSlice(arena.allocator());
    if (process_args.len < 2) {
        log.err("Usage: gemma4_context <path-to-p5_2_econtext_layer15_kv13.safetensors>", .{});
        return error.MissingArgument;
    }
    const fixture_path = process_args[1];

    log.info("P5.2.E.context — ZML context dot (reader layer 15 x KV layer 13, probs@V .k, GQA, no o_proj)", .{});
    log.info("Loading fixture from {s}", .{fixture_path});

    var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, fixture_path);
    defer registry.deinit();

    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();

    const model: ContextFixture = .init(store.view());

    log.info("Symbolic shapes:", .{});
    log.info("  probs          : {f}", .{model.probs});
    log.info("  v_final        : {f}", .{model.v_final});
    log.info("  context_oracle : {f}", .{model.context_oracle});

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);

    const replicated_sharding = try zml.sharding.replicatedSharding(platform);

    log.info("Materializing buffers...", .{});
    var buffers = try model.load(arena.allocator(), io, platform, &store, &.{replicated_sharding});
    defer ContextFixture.unloadBuffers(&buffers);
    log.info("Buffers loaded (3 tensors).", .{});

    log.info("Compiling forward (splitAxis GQA -> dot .k -> merge -> transpose)...", .{});
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

    log.info("Forward result shape: {f} (expected [b=1, h=8, q=4, hd=256])", .{result.shape()});

    var slice = try result.toSliceAlloc(allocator, io);
    defer slice.free(allocator);
    const data = slice.items(f32);

    if (data.len != FLAT_LEN) {
        log.err("length mismatch: data={d} expected={d} (rank/shape wrong -> .k mal contracté ?)", .{ data.len, FLAT_LEN });
        return error.ContextLengthMismatch;
    }

    // === Sanity sortie ZML (NaN/Inf + magnitude — détecte un layout merge scrambled) ===
    var has_nan_inf = false;
    var max_mag: f32 = 0.0;
    for (data) |v| {
        if (std.math.isNan(v) or std.math.isInf(v)) has_nan_inf = true;
        const av = @abs(v);
        if (av > max_mag) max_mag = av;
    }
    log.info("Sanity: NaN/Inf={}, max|context|={d:.4} (ceil {d:.1})", .{ has_nan_inf, max_mag, MAGNITUDE_CEIL });
    if (has_nan_inf) {
        log.err("BLOCK: ZML context contains NaN/Inf", .{});
        return error.ContextNanInf;
    }
    if (max_mag > MAGNITUDE_CEIL) {
        log.err("BLOCK: |context| {d:.4} > ceil {d:.1} -> layout merge probablement scrambled", .{ max_mag, MAGNITUDE_CEIL });
        return error.ContextMagnitude;
    }

    // === Fixed-point blocks (3 x 8 valeurs) vs oracle context ===
    log.info("Fixed-point blocks vs oracle context (fp32):", .{});
    var max_block: f32 = 0.0;
    for (CTX_BLOCKS) |block| {
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
    log.info("  -> 3 blocks max_diff: {e:.6}", .{max_block});

    // === Scan global 8192 valeurs vs oracle ===
    log.info("Scanning full tensor ({d} fp32) vs oracle context:", .{FLAT_LEN});
    var ref_slice = try buffers.context_oracle.toSliceAlloc(allocator, io);
    defer ref_slice.free(allocator);
    const ref_data = ref_slice.items(f32);

    if (ref_data.len != FLAT_LEN) {
        log.err("length mismatch: ref={d} expected={d}", .{ ref_data.len, FLAT_LEN });
        return error.ContextLengthMismatch;
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
    const h_idx = (max_idx % STRIDE_B) / STRIDE_H;
    const q_idx = (max_idx % STRIDE_H) / STRIDE_Q;
    const hd_idx = max_idx % STRIDE_Q;
    log.info("  -> full tensor max_abs : {e:.6} at flat_index {d} (b={d}, h={d}, q={d}, hd={d})", .{
        max_global, max_idx, b_idx, h_idx, q_idx, hd_idx,
    });
    log.info("  -> full tensor mean_abs: {e:.6}", .{mean_abs});

    const max_diff = if (max_global > max_block) max_global else max_block;
    log.info("context global max_diff: {e:.6} (tolerance {e:.1})", .{ max_diff, CTX_TOLERANCE });
    log.info("  Expected << 1e-4 (matmul probs@V PJRT-CPU vs PyTorch BLAS, réduction .k=4 ; jitter QK propagé)", .{});

    if (max_diff > CTX_TOLERANCE) {
        log.err("BLOCK: context max_diff exceeds tolerance", .{});
        return error.ContextFailed;
    }
    log.info("P5.2.E.context PASS: ZML context dot reader layer 15 x KV layer 13 validated vs PyTorch oracle", .{});
    log.info("  (probs@V .k via GQA split, merge, transpose ; no o_proj, no re-softmax, no scaling, no layer 14)", .{});
    log.info("  -> P5.2.E COMPLET : chaîne d'attention ZML QK -> mask -> softmax -> context validée bout-en-bout", .{});
}
