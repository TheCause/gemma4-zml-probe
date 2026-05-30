// P5.2.D.5 — ZML KV slot mock, producer/writer layer 13 (sliding).
//
// Objectif : packager le writer sliding dans un slot KV factice. Calcule
// le couple (k_after_rope, v_after_proj_reshaped) en compute layout, puis
// applique le transpose vers cache layout (mirror PyTorch k_final/v_final).
// Aucune attention, aucun cache dynamique réel, aucun Q path.
//
// Decision layout (cf manifest D.5) :
//   compute layout : [1, 4, 1, 256] = {.b, .s, .kvh, .hd}
//   cache layout   : [1, 1, 4, 256] = {.b, .kvh, .s, .hd}
//   transition     : tensor.transpose(.{.b, .kvh, .s, .hd})
//
// Avec n_kv = 1, le transpose est un NO-OP en mémoire (dim singleton ne
// reordonne pas en row-major). Verifié côté Python dans la sanity D.5.
// Reste explicite ZML-side pour préparer les futures couches non-singleton.
//
// Pipeline ZML strict (return tuple K_slot, V_slot, V_raw) :
//   K : k_proj + reshape + withTags + rmsNorm + mul + rope + transpose
//   V : v_proj + reshape + withTags + rmsNorm (UNSCALED, no mul) + transpose  (pas de RoPE)
//   V_raw : v_4d avant norm, expose pour la sanity anti-regression D.5 corrige.
//
// RoPE opts identique a D.4 :
//   layout=.sequential, scaling=.default, theta=10000
//
// Interdits stricts P5.2.D.5 :
//   - attention scores / matmul QK / softmax
//   - Q path (q_proj, q_norm, etc.)
//   - reader (layers 15-34)
//   - layer 14 full attention (proportional RoPE)
//   - sliding mask
//   - cache dynamique réel (scatter / dynamicSlice)
//
// CLI : gemma4_kv_slot <path-to-p5_2_d5_kv_slot_layer13.safetensors>

const std = @import("std");
const log = std.log;

const zml = @import("zml");
const stdx = zml.stdx;

pub const std_options: std.Options = .{
    .log_level = .info,
};

// Invariants P5.2.D.0.
const B: i64 = 1;
const S: i64 = 4;
const H: i64 = 1536;
const KV: i64 = 256;
const NKV: i64 = 1;
const D: i64 = 256;
const RMS_EPS: f32 = 1.0e-6;
const ROPE_THETA: f32 = 10_000;

const KV_SLOT_TOLERANCE: f32 = 1.0e-4;
const KV_SLOT_FLAT_LEN: usize = @intCast(B * NKV * S * D); // 1024

// Fixed-point blocks (per-position vs oracle k_final / v_final).
// Cache layout [1,1,4,256], strides (1024, 1024, 256, 1) :
// [0, 0, s, :8] -> flat = s * 256.
const SlotBlock = struct {
    label: []const u8,
    flat_offset: usize,
    width: usize,
};

const SLOT_BLOCKS = [_]SlotBlock{
    .{ .label = "[0,0,0,:8]  pos=0", .flat_offset = 0, .width = 8 },
    .{ .label = "[0,0,3,:8]  pos=3", .flat_offset = 768, .width = 8 },
};

/// Fixture D.5 — 6 tenseurs : 4 entrées (inputs+poids) + 2 oracles (k_final, v_final).
const KvSlotFixture = struct {
    hidden_input: zml.Tensor,
    k_proj_weight: zml.Tensor,
    k_norm_weight: zml.Tensor,
    v_proj_weight: zml.Tensor,
    k_final_oracle: zml.Tensor,
    v_final_oracle: zml.Tensor,

    pub fn init(store: zml.io.TensorStore.View) KvSlotFixture {
        return .{
            .hidden_input = store.createTensor(
                "hidden_input",
                .{ .b, .s, .h },
                null,
            ),
            .k_proj_weight = store.createTensor(
                "k_proj_weight",
                .{ .kv, .h },
                null,
            ),
            .k_norm_weight = store.createTensor(
                "k_norm_weight",
                .{.hd},
                null,
            ),
            .v_proj_weight = store.createTensor(
                "v_proj_weight",
                .{ .kv, .h },
                null,
            ),
            .k_final_oracle = store.createTensor(
                "k_final",
                .{ .b, .kvh, .s, .hd },
                null,
            ),
            .v_final_oracle = store.createTensor(
                "v_final",
                .{ .b, .kvh, .s, .hd },
                null,
            ),
        };
    }

    pub fn load(
        self: *const KvSlotFixture,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *const zml.io.TensorStore,
        shardings: []const zml.sharding.Sharding,
    ) !zml.Bufferized(KvSlotFixture) {
        return zml.io.load(KvSlotFixture, self, allocator, io, platform, store, .{
            .shardings = shardings,
            .parallelism = 1,
            .dma_chunks = 1,
            .dma_chunk_size = 16 * 1024 * 1024,
        });
    }

    pub fn unloadBuffers(self: *zml.Bufferized(KvSlotFixture)) void {
        self.hidden_input.deinit();
        self.k_proj_weight.deinit();
        self.k_norm_weight.deinit();
        self.v_proj_weight.deinit();
        self.k_final_oracle.deinit();
        self.v_final_oracle.deinit();
    }

    /// Forward D.5 : (k_slot, v_slot) en cache layout {.b, .kvh, .s, .hd}.
    ///
    ///   K : k_proj (D.1) + reshape + withTags + rmsNorm (D.3) + mul + rope (D.4)
    ///       -> compute layout {.b, .s, .kvh, .hd} = [1,4,1,256]
    ///       -> transpose -> cache layout {.b, .kvh, .s, .hd} = [1,1,4,256]
    ///
    ///   V : v_proj (D.2) + reshape + withTags + rmsNorm UNSCALED (D.2b, no mul)
    ///       -> compute layout {.b, .s, .kvh, .hd} = [1,4,1,256]
    ///       -> transpose -> cache layout {.b, .kvh, .s, .hd} = [1,1,4,256]
    ///       (V RMSNormé sans scale, non roté en Gemma 4)
    pub fn forward(self: KvSlotFixture) struct { zml.Tensor, zml.Tensor, zml.Tensor } {
        // --- K branche (inchangée) ---
        const k_after_proj = self.hidden_input.dot(self.k_proj_weight, .h);
        const k_4d = k_after_proj
            .reshape(.{ B, S, NKV, D })
            .withTags(.{ .b, .s, .kvh, .hd });
        const k_normalized = zml.nn.rmsNorm(k_4d, .hd, RMS_EPS);
        const k_after_norm = k_normalized.mul(
            self.k_norm_weight.broad(k_normalized.shape()),
        );
        const rope_opts: zml.nn.RopeOpts = .{
            .layout = .sequential,
            .scaling = .{ .default = .{ .rope_theta = ROPE_THETA } },
        };
        const k_after_rope = zml.nn.rope(k_after_norm, null, rope_opts);
        // Transpose compute -> cache. Avec n_kv=1 c'est un no-op mémoire,
        // mais reste explicite pour préparer les couches futures.
        const k_slot = k_after_rope.transpose(.{ .b, .kvh, .s, .hd });

        // --- V branche (corrigée D.5 : RMSNorm UNSCALED, cf D.2b) ---
        const v_after_proj = self.hidden_input.dot(self.v_proj_weight, .h);
        const v_4d = v_after_proj
            .reshape(.{ B, S, NKV, D })
            .withTags(.{ .b, .s, .kvh, .hd });
        // V : RMSNorm UNSCALED (Gemma4 v_norm = with_scale=False) — normalise
        // SANS poids (pas de mul), pas de RoPE. value = v_after_norm.
        const v_after_norm = zml.nn.rmsNorm(v_4d, .hd, RMS_EPS);
        const v_slot = v_after_norm.transpose(.{ .b, .kvh, .s, .hd });

        // v_4d (V brut, avant norm) expose pour la sanity anti-regression host :
        // max|v_slot - v_raw| doit valoir ~0.777, sinon le v_norm a saute.
        return .{ k_slot, v_slot, v_4d };
    }
};

/// Compare slice `data` vs `oracle` au seuil `tolerance` sur `len` valeurs.
/// Renvoie (max_abs, mean_abs, max_idx). Loggue les fixed-point blocks.
fn compareSlot(
    name: []const u8,
    data: []const f32,
    oracle: []const f32,
    tolerance: f32,
) !struct { max_abs: f32, mean_abs: f32, max_idx: usize, max_block: f32 } {
    if (data.len != oracle.len or data.len != KV_SLOT_FLAT_LEN) {
        log.err("{s}: length mismatch data={d} oracle={d} expected={d}", .{
            name, data.len, oracle.len, KV_SLOT_FLAT_LEN,
        });
        return error.SlotLengthMismatch;
    }

    log.info("{s}: fixed-point blocks vs oracle (fp32):", .{name});
    var max_block: f32 = 0.0;
    for (SLOT_BLOCKS) |block| {
        var block_max: f32 = 0.0;
        log.info("  Block {s} (flat_offset={d}):", .{ block.label, block.flat_offset });
        var i: usize = 0;
        while (i < block.width) : (i += 1) {
            const actual = data[block.flat_offset + i];
            const expected = oracle[block.flat_offset + i];
            const diff = @abs(actual - expected);
            if (diff > block_max) block_max = diff;
            log.info("    +{d:>3}: actual={d:.8} expected={d:.8} diff={e:.3}", .{
                i, actual, expected, diff,
            });
        }
        log.info("    block max_diff: {e:.6}", .{block_max});
        if (block_max > max_block) max_block = block_max;
    }
    log.info("  -> {s} 2 blocks max_diff: {e:.6}", .{ name, max_block });

    var max_abs: f32 = 0.0;
    var sum_abs: f64 = 0.0;
    var max_idx: usize = 0;
    for (data, oracle, 0..) |actual, expected, i| {
        const diff = @abs(actual - expected);
        if (diff > max_abs) {
            max_abs = diff;
            max_idx = i;
        }
        sum_abs += @as(f64, diff);
    }
    const mean_abs = @as(f32, @floatCast(sum_abs / @as(f64, @floatFromInt(KV_SLOT_FLAT_LEN))));

    // Decompose max_idx into (b, kvh, s, d). Cache layout strides:
    //   b   -> NKV*S*D = 1024
    //   kvh -> S*D     = 1024
    //   s   -> D       = 256
    //   d   -> 1
    const stride_b: usize = @intCast(NKV * S * D);
    const stride_kvh: usize = @intCast(S * D);
    const stride_s: usize = @intCast(D);
    const b_idx = max_idx / stride_b;
    const kvh_idx = (max_idx % stride_b) / stride_kvh;
    const s_idx = (max_idx % stride_kvh) / stride_s;
    const d_idx = max_idx % stride_s;

    log.info("{s}: full tensor max_abs : {e:.6} at flat_index {d} (b={d}, kvh={d}, s={d}, d={d})", .{
        name, max_abs, max_idx, b_idx, kvh_idx, s_idx, d_idx,
    });
    log.info("{s}: full tensor mean_abs: {e:.6}", .{ name, mean_abs });

    const max_diff = if (max_abs > max_block) max_abs else max_block;
    if (max_diff > tolerance) {
        log.err("BLOCK: {s} max_diff {e:.6} exceeds tolerance {e:.1}", .{ name, max_diff, tolerance });
        return error.SlotFailed;
    }

    return .{ .max_abs = max_abs, .mean_abs = mean_abs, .max_idx = max_idx, .max_block = max_block };
}

pub fn main(init: std.process.Init) !void {
    const arena = init.arena;
    const allocator = init.gpa;
    const io = init.io;

    const process_args = try init.minimal.args.toSlice(arena.allocator());
    if (process_args.len < 2) {
        log.err("Usage: gemma4_kv_slot <path-to-p5_2_d5_kv_slot_layer13.safetensors>", .{});
        return error.MissingArgument;
    }
    const fixture_path = process_args[1];

    log.info("P5.2.D.5 — ZML KV slot mock (no attention, no cache real), producer layer 13 sliding", .{});
    log.info("  Layout decision : compute [1,4,1,256] {{b,s,kvh,hd}} -> cache [1,1,4,256] {{b,kvh,s,hd}}", .{});
    log.info("  RoPE opts : layout=.sequential  scaling=.default  rope_theta={d}", .{ROPE_THETA});
    log.info("Loading fixture from {s}", .{fixture_path});

    var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, fixture_path);
    defer registry.deinit();

    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();

    const model: KvSlotFixture = .init(store.view());

    log.info("Symbolic shapes:", .{});
    log.info("  hidden_input    : {f}", .{model.hidden_input});
    log.info("  k_proj_weight   : {f}", .{model.k_proj_weight});
    log.info("  k_norm_weight   : {f}", .{model.k_norm_weight});
    log.info("  v_proj_weight   : {f}", .{model.v_proj_weight});
    log.info("  k_final_oracle  : {f}", .{model.k_final_oracle});
    log.info("  v_final_oracle  : {f}", .{model.v_final_oracle});

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);

    const replicated_sharding = try zml.sharding.replicatedSharding(platform);

    log.info("Materializing buffers...", .{});
    var buffers = try model.load(arena.allocator(), io, platform, &store, &.{replicated_sharding});
    defer KvSlotFixture.unloadBuffers(&buffers);
    log.info("Buffers loaded (6 tensors).", .{});

    log.info("Compiling forward (K full pipeline + V proj + transposes to cache layout)...", .{});
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

    var k_slot_buf, var v_slot_buf, var v_raw_buf = results.get(struct { zml.Buffer, zml.Buffer, zml.Buffer });
    defer k_slot_buf.deinit();
    defer v_slot_buf.deinit();
    defer v_raw_buf.deinit();

    log.info("Forward K_slot shape: {f}", .{k_slot_buf.shape()});
    log.info("Forward V_slot shape: {f}", .{v_slot_buf.shape()});

    var k_slice = try k_slot_buf.toSliceAlloc(allocator, io);
    defer k_slice.free(allocator);
    var v_slice = try v_slot_buf.toSliceAlloc(allocator, io);
    defer v_slice.free(allocator);
    var v_raw_slice = try v_raw_buf.toSliceAlloc(allocator, io);
    defer v_raw_slice.free(allocator);
    const k_data = k_slice.items(f32);
    const v_data = v_slice.items(f32);
    const v_raw_data = v_raw_slice.items(f32);

    // === Sanity obligatoire (D.5 corrigé) : v_slot (normé) DOIT différer du V
    // brut (v_4d). Avec n_kv=1 les deux partagent le flat layout (1024). Si la
    // RMSNorm V a sauté, max|v_slot - v_raw| ~ 0 -> le bug 'V non normé' revient. ===
    if (v_raw_data.len == v_data.len) {
        var v_norm_change: f32 = 0.0;
        for (v_data, v_raw_data) |normed, raw| {
            const c = @abs(normed - raw);
            if (c > v_norm_change) v_norm_change = c;
        }
        log.info("sanity max|v_slot - v_raw(brut)| = {e:.6} (expected ~0.777, RMSNorm V active)", .{v_norm_change});
        if (v_norm_change < 1.0e-2) {
            log.err("BLOCK: V slot ~ raw V — la RMSNorm V a saute, le bug 'V non norme' est revenu", .{});
            return error.VNormRegressed;
        }
    } else {
        log.err("v_raw length {d} != v_data length {d}", .{ v_raw_data.len, v_data.len });
        return error.SlotLengthMismatch;
    }

    var k_ref_slice = try buffers.k_final_oracle.toSliceAlloc(allocator, io);
    defer k_ref_slice.free(allocator);
    var v_ref_slice = try buffers.v_final_oracle.toSliceAlloc(allocator, io);
    defer v_ref_slice.free(allocator);
    const k_ref = k_ref_slice.items(f32);
    const v_ref = v_ref_slice.items(f32);

    // === Compare K slot (expected ~ D.4 résidu ~5e-7) ===
    const k_stats = try compareSlot("K_slot", k_data, k_ref, KV_SLOT_TOLERANCE);

    // === Compare V slot (v_proj matmul + v_norm UNSCALED ; résidu ~ D.2 ~5e-6) ===
    const v_stats = try compareSlot("V_slot", v_data, v_ref, KV_SLOT_TOLERANCE);

    log.info("---", .{});
    log.info("Summary :", .{});
    log.info("  K_slot max_abs={e:.6} mean_abs={e:.6} (expected ~ D.4 = 5.36e-7)", .{ k_stats.max_abs, k_stats.mean_abs });
    log.info("  V_slot max_abs={e:.6} mean_abs={e:.6} (v_proj matmul through v_norm, ~ D.2 5.25e-6)", .{ v_stats.max_abs, v_stats.mean_abs });
    log.info("  Tolerance {e:.1} on each slot", .{KV_SLOT_TOLERANCE});

    log.info("P5.2.D.5 PASS: ZML KV slot mock producer layer 13 validated vs PyTorch oracle", .{});
    log.info("  K_slot ≡ k_after_rope.transpose(cache) ≡ k_final  (cache layout {{b,kvh,s,hd}}=[1,1,4,256])", .{});
    log.info("  V_slot ≡ v_after_norm.transpose(cache) ≡ v_final  (V RMSNormé sans scale, non roté)", .{});
    log.info("  (no attention, no Q path, no reader, no layer 14, no sliding mask, no dynamic cache)", .{});
}
