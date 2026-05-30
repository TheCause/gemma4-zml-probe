// P5.2.D.2b — ZML v_norm (reshape + RMSNorm UNSCALED), producer/writer
// layer 13 (sliding).
//
// Objectif : valider la RMSNorm de V. Gemma4 v_norm = Gemma4RMSNorm(head_dim,
// eps, with_scale=False) -> normalisation RMS appliquee SANS poids appris.
// C'est le miroir exact de D.3 k_norm, mais SANS le `.mul(weight)`.
// Comparer contre l'oracle PyTorch fp32 `v_after_norm` du fixture D.0b.
//
// Le bug D.0 -> D.0b : "pas de v_norm.weight au checkpoint" avait ete lu a tort
// comme "V non norme". En realite V est RMSNorme sans scale. D.2b verifie le
// portage ZML de cette normalisation.
//
// Pipeline ZML strict :
//   v_4d         = v_after_proj.reshape({1,4,1,256})              // perd les tags
//                    .withTags(.{.b, .s, .kvh, .hd})              // re-tag (piège #1)
//   v_after_norm = zml.nn.rmsNorm(v_4d, .hd, RMS_EPS)             // PAS de .mul(weight)
//
// Entree : v_after_proj [1,4,256] (V deja projete, fourni par l'oracle D.0b).
// Oracle : v_after_norm [1,4,1,256] (PyTorch Gemma4RMSNorm with_scale=False).
//
// Interdits stricts P5.2.D.2b :
//   - .mul(weight) / v_norm.weight  (with_scale=False : pas de poids)
//   - v_proj  (on part de v_after_proj)
//   - RoPE (apply_rotary_pos_emb)
//   - transpose [B, n_kv, S, head_dim]
//   - cache / sliding mask
//   - attention scores / matmul QK / softmax
//
// CLI : gemma4_v_norm <path-to-p5_2_d2b_v_norm_layer13.safetensors>

const std = @import("std");
const log = std.log;

const zml = @import("zml");
const stdx = zml.stdx;

pub const std_options: std.Options = .{
    .log_level = .info,
};

// Invariants P5.2.D.0b (cf manifest p5_2_d0_kv_oracle_layer13_manifest.json).
const B: i64 = 1;
const S: i64 = 4;
const KV: i64 = 256; // n_kv (1) * head_dim (256) — tag d'entree v_after_proj
const NKV: i64 = 1; // num_key_value_heads
const D: i64 = 256; // head_dim
const RMS_EPS: f32 = 1.0e-6;

// Tolerance / sanity.
const V_NORM_TOLERANCE: f32 = 1.0e-4;
const V_NORM_FLAT_LEN: usize = @intCast(B * S * NKV * D); // 1024
// La RMSNorm doit modifier V de maniere mesurable (sinon regression "V non norme").
const V_NORM_CHANGE_MIN: f32 = 1.0e-3;

// Fixed-point blocks (reported per-position vs oracle, no hardcoded expected) :
// v_after_norm[0, {0,3}, 0, :8] en fp32.
// shape [1,4,1,256], strides (1024, 256, 256, 1). [0,s,0,:8] -> flat = s * 256.
const VNormBlock = struct {
    label: []const u8,
    flat_offset: usize,
    width: usize,
};

const V_NORM_BLOCKS = [_]VNormBlock{
    .{ .label = "A [0,0,0,:8]", .flat_offset = 0, .width = 8 },
    .{ .label = "B [0,3,0,:8]", .flat_offset = 768, .width = 8 },
};

/// Fixture D.2b slim (2 tenseurs) chargee depuis safetensors.
const VNormFixture = struct {
    v_after_proj: zml.Tensor,
    v_after_norm_oracle: zml.Tensor,

    pub fn init(store: zml.io.TensorStore.View) VNormFixture {
        return .{
            .v_after_proj = store.createTensor(
                "v_after_proj",
                .{ .b, .s, .kv },
                null,
            ),
            .v_after_norm_oracle = store.createTensor(
                "v_after_norm",
                .{ .b, .s, .kvh, .hd },
                null,
            ),
        };
    }

    pub fn load(
        self: *const VNormFixture,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *const zml.io.TensorStore,
        shardings: []const zml.sharding.Sharding,
    ) !zml.Bufferized(VNormFixture) {
        return zml.io.load(VNormFixture, self, allocator, io, platform, store, .{
            .shardings = shardings,
            .parallelism = 1,
            .dma_chunks = 1,
            .dma_chunk_size = 16 * 1024 * 1024,
        });
    }

    pub fn unloadBuffers(self: *zml.Bufferized(VNormFixture)) void {
        self.v_after_proj.deinit();
        self.v_after_norm_oracle.deinit();
    }

    /// Forward D.2b : reshape + withTags + rmsNorm(.hd) SANS mul (with_scale=False).
    /// Resultat : Tensor({b=1, s=4, kvh=1, hd=256, f32}).
    pub fn forward(self: VNormFixture) zml.Tensor {
        // Reshape perd les tags -> re-tag (piège ZML #1 capitalise P4.4.2 Gate H).
        const v_4d = self.v_after_proj
            .reshape(.{ B, S, NKV, D })
            .withTags(.{ .b, .s, .kvh, .hd });

        // RMSNorm sur axe .hd (head_dim) — Gemma4 v_norm = with_scale=False :
        // normalisation RMS PURE, AUCUNE multiplication par un poids.
        return zml.nn.rmsNorm(v_4d, .hd, RMS_EPS);
    }
};

pub fn main(init: std.process.Init) !void {
    const arena = init.arena;
    const allocator = init.gpa;
    const io = init.io;

    const process_args = try init.minimal.args.toSlice(arena.allocator());
    if (process_args.len < 2) {
        log.err("Usage: gemma4_v_norm <path-to-p5_2_d2b_v_norm_layer13.safetensors>", .{});
        return error.MissingArgument;
    }
    const fixture_path = process_args[1];

    log.info("P5.2.D.2b — ZML v_norm (reshape + rmsNorm UNSCALED, no mul), producer layer 13 sliding", .{});
    log.info("Loading fixture from {s}", .{fixture_path});

    var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, fixture_path);
    defer registry.deinit();

    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();

    const model: VNormFixture = .init(store.view());

    log.info("Symbolic shapes:", .{});
    log.info("  v_after_proj        : {f}", .{model.v_after_proj});
    log.info("  v_after_norm_oracle : {f}", .{model.v_after_norm_oracle});

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);

    const replicated_sharding = try zml.sharding.replicatedSharding(platform);

    log.info("Materializing buffers...", .{});
    var buffers = try model.load(arena.allocator(), io, platform, &store, &.{replicated_sharding});
    defer VNormFixture.unloadBuffers(&buffers);
    log.info("Buffers loaded (2 tensors).", .{});

    log.info("Compiling forward (reshape + rmsNorm, no mul/weight/RoPE/transpose/cache)...", .{});
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

    // === Oracle slice (used for both fixed-point blocks AND global scan). ===
    var ref_slice = try buffers.v_after_norm_oracle.toSliceAlloc(allocator, io);
    defer ref_slice.free(allocator);
    const ref_data = ref_slice.items(f32);

    // === Input slice (for the "norm actually changes V" sanity). ===
    var in_slice = try buffers.v_after_proj.toSliceAlloc(allocator, io);
    defer in_slice.free(allocator);
    const in_data = in_slice.items(f32);

    if (ref_data.len != V_NORM_FLAT_LEN or data.len != V_NORM_FLAT_LEN or in_data.len != V_NORM_FLAT_LEN) {
        log.err("length mismatch: ref={d} data={d} in={d} expected={d}", .{ ref_data.len, data.len, in_data.len, V_NORM_FLAT_LEN });
        return error.VNormLengthMismatch;
    }

    // === Sanity : la RMSNorm DOIT modifier V (garde anti-regression "V non norme"). ===
    var max_change: f32 = 0.0;
    for (data, in_data) |out, inp| {
        const c = @abs(out - inp);
        if (c > max_change) max_change = c;
    }
    log.info("sanity max|v_after_norm - v_after_proj| = {e:.6} (expected > {e:.1}, RMSNorm active)", .{ max_change, V_NORM_CHANGE_MIN });
    if (max_change < V_NORM_CHANGE_MIN) {
        log.err("BLOCK: v_norm is a no-op (output == input) — regression to 'V not normed'?", .{});
        return error.VNormNoOp;
    }

    // === Fixed-point blocks (2 x 8 valeurs) vs oracle ===
    log.info("Fixed-point blocks vs oracle v_after_norm (fp32):", .{});
    var max_block: f32 = 0.0;
    for (V_NORM_BLOCKS) |block| {
        var block_max: f32 = 0.0;
        log.info("  Block {s} (flat_offset={d}):", .{ block.label, block.flat_offset });
        var i: usize = 0;
        while (i < block.width) : (i += 1) {
            const actual = data[block.flat_offset + i];
            const expected = ref_data[block.flat_offset + i];
            const diff = @abs(actual - expected);
            if (diff > block_max) block_max = diff;
            log.info("    +{d:>3}: actual={d:.8} expected={d:.8} diff={e:.3}", .{
                i, actual, expected, diff,
            });
        }
        log.info("    block max_diff: {e:.6}", .{block_max});
        if (block_max > max_block) max_block = block_max;
    }
    log.info("  -> 2 blocks max_diff: {e:.6}", .{max_block});

    // === Scan global 1024 valeurs vs oracle ===
    log.info("Scanning full tensor ({d} fp32) vs oracle v_after_norm:", .{V_NORM_FLAT_LEN});

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
    const mean_abs = @as(f32, @floatCast(sum_abs / @as(f64, @floatFromInt(V_NORM_FLAT_LEN))));

    // Decompose max_idx into (s, kvh, d). strides : s -> NKV*D = 256, kvh -> D = 256, d -> 1.
    const stride_s: usize = @intCast(NKV * D);
    const stride_kvh: usize = @intCast(D);
    const s_idx = max_idx / stride_s;
    const kvh_idx = (max_idx % stride_s) / stride_kvh;
    const d_idx = max_idx % stride_kvh;

    log.info("  -> full tensor max_abs : {e:.6} at flat_index {d} (s={d}, kvh={d}, d={d})", .{
        max_global, max_idx, s_idx, kvh_idx, d_idx,
    });
    log.info("  -> full tensor mean_abs: {e:.6}", .{mean_abs});

    const max_diff = if (max_global > max_block) max_global else max_block;
    log.info("v_norm global max_diff: {e:.6} (tolerance {e:.1})", .{ max_diff, V_NORM_TOLERANCE });
    log.info("  Expected ~1e-6 or less : pas de matmul amont, RMSNorm pure sur entree fp32", .{});

    if (max_diff > V_NORM_TOLERANCE) {
        log.err("BLOCK: v_norm max_diff exceeds tolerance", .{});
        log.err("  suspects: (1) wrong RMSNorm axis (.hd), (2) mul(weight) ajoute par erreur,", .{});
        log.err("            (3) reshape sans withTags, (4) tags incorrects sur v_after_proj", .{});
        return error.VNormFailed;
    }
    log.info("P5.2.D.2b PASS: ZML v_norm producer layer 13 validated vs PyTorch oracle", .{});
    log.info("  (no mul/weight, no v_proj, no RoPE, no transpose, no cache, no attention)", .{});
}
