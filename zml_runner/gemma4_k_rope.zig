// P5.2.D.4 — ZML RoPE K-only, producer/writer layer 13 (sliding).
//
// Objectif : etendre P5.2.D.3 (k_proj + k_norm, PASS) avec la rotation
// positionnelle RoPE sur K. Comparer contre l'oracle PyTorch fp32
// `k_after_rope` du fixture D.0. V hors scope (non rote en Gemma 4).
//
// Pipeline ZML strict :
//   k_after_proj = hidden_input.dot(k_proj_weight, .h)              (reuse D.1)
//   k_4d         = k_after_proj.reshape({1,4,1,256})
//                    .withTags(.{.b, .s, .kvh, .hd})                (piège #1)
//                                                                   (tag .hd direct,
//                                                                    requis par zml.nn.rope)
//   k_normalized = zml.nn.rmsNorm(k_4d, .hd, RMS_EPS)
//   k_after_norm = k_normalized.mul(k_norm_weight.broad(...))       (pattern Llama)
//   k_after_rope = zml.nn.rope(k_after_norm, null, opts)            (default pos_idx = arange(0,S))
//
// RoPE opts pour Gemma 4 sliding (identique a C.3 q_rope) :
//   layout  = .sequential   (HF style, split-half)
//   scaling = .default      (attention_scaling = 1.0, partial_rotary=1.0)
//   theta   = 10_000        (sliding_attention rope_theta)
//
// Convention tags ZML rope (cf llama/model.zig L508 + C.3 q_rope) :
//   - x doit avoir .s (sequence) et .hd (head dim, even)
//   - autres axes (ici .b, .kvh) preserves
//
// Diff vs C.3 q_rope :
//   - n_kv = 1 (vs n_heads = 8) -> axe head_count taggue .kvh (vs .nh)
//   - shape sortie [1,4,1,256] (vs [1,4,8,256])
//   - K-norm pattern Llama identique a Q-norm
//
// Interdits stricts P5.2.D.4 :
//   - v_proj
//   - v_norm
//   - transpose final [.b, .kvh, .s, .hd]
//   - cache / sliding mask
//   - attention scores / matmul QK / softmax
//   - layer 14 full attention (proportional RoPE)
//
// CLI : gemma4_k_rope <path-to-p5_2_d4_k_rope_layer13.safetensors>

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

const K_ROPE_TOLERANCE: f32 = 1.0e-4;
const K_ROPE_FLAT_LEN: usize = @intCast(B * S * NKV * D); // 1024

// Fixed-point blocks (reported per-position vs oracle).
// k_after_rope[0, {0,3}, 0, :8]. flat_offset = s * NKV * D + 0.
const KRopeBlock = struct {
    label: []const u8,
    flat_offset: usize,
    width: usize,
};

const K_ROPE_BLOCKS = [_]KRopeBlock{
    // Position 0 : RoPE = identite (cos=1, sin=0) -> doit egaler k_after_norm[0,0,0,:8] bit-exact.
    .{ .label = "A [0,0,0,:8]  pos=0 RoPE identity", .flat_offset = 0, .width = 8 },
    // Position 3 : RoPE active -> different de k_after_norm[0,3,0,:8].
    .{ .label = "B [0,3,0,:8]  pos=3 RoPE active", .flat_offset = 768, .width = 8 },
};

/// Fixture D.4 — 5 tenseurs (4 consommes au forward + 1 pour sanity inline).
const KRopeFixture = struct {
    hidden_input: zml.Tensor,
    k_proj_weight: zml.Tensor,
    k_norm_weight: zml.Tensor,
    k_after_norm_oracle: zml.Tensor, // utilise UNIQUEMENT pour sanity pos0/pos3
    k_after_rope_oracle: zml.Tensor,

    pub fn init(store: zml.io.TensorStore.View) KRopeFixture {
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
            .k_after_norm_oracle = store.createTensor(
                "k_after_norm",
                .{ .b, .s, .kvh, .hd },
                null,
            ),
            .k_after_rope_oracle = store.createTensor(
                "k_after_rope",
                .{ .b, .s, .kvh, .hd },
                null,
            ),
        };
    }

    pub fn load(
        self: *const KRopeFixture,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *const zml.io.TensorStore,
        shardings: []const zml.sharding.Sharding,
    ) !zml.Bufferized(KRopeFixture) {
        return zml.io.load(KRopeFixture, self, allocator, io, platform, store, .{
            .shardings = shardings,
            .parallelism = 1,
            .dma_chunks = 1,
            .dma_chunk_size = 16 * 1024 * 1024,
        });
    }

    pub fn unloadBuffers(self: *zml.Bufferized(KRopeFixture)) void {
        self.hidden_input.deinit();
        self.k_proj_weight.deinit();
        self.k_norm_weight.deinit();
        self.k_after_norm_oracle.deinit();
        self.k_after_rope_oracle.deinit();
    }

    /// Forward D.4 : k_proj + k_norm (D.3 PASS) + RoPE K-only.
    /// Resultat : Tensor({b=1, s=4, kvh=1, hd=256, f32}).
    pub fn forward(self: KRopeFixture) zml.Tensor {
        // D.1 : projection lineaire K, reduit .h.
        const k_after_proj = self.hidden_input.dot(self.k_proj_weight, .h);

        // D.3 : reshape + re-tag (piege #1), rmsNorm, mul broad (pattern Llama).
        // On tag DIRECTEMENT .hd (vs .d en D.3) car zml.nn.rope l'exige.
        const k_4d = k_after_proj
            .reshape(.{ B, S, NKV, D })
            .withTags(.{ .b, .s, .kvh, .hd });
        const k_normalized = zml.nn.rmsNorm(k_4d, .hd, RMS_EPS);
        const k_after_norm = k_normalized.mul(
            self.k_norm_weight.broad(k_normalized.shape()),
        );

        // D.4 : RoPE K-only via helper natif ZML.
        // pos_idx = null -> default arange(0, x.dim(.s)) = [0,1,2,3] tag .s.
        // layout = .sequential match Gemma 4 apply_rotary_pos_emb (cf preuve C.3).
        // scaling = .default match Gemma 4 sliding rope_type=default (theta=10000).
        const rope_opts: zml.nn.RopeOpts = .{
            .layout = .sequential,
            .scaling = .{ .default = .{ .rope_theta = ROPE_THETA } },
        };
        return zml.nn.rope(k_after_norm, null, rope_opts);
    }
};

pub fn main(init: std.process.Init) !void {
    const arena = init.arena;
    const allocator = init.gpa;
    const io = init.io;

    const process_args = try init.minimal.args.toSlice(arena.allocator());
    if (process_args.len < 2) {
        log.err("Usage: gemma4_k_rope <path-to-p5_2_d4_k_rope_layer13.safetensors>", .{});
        return error.MissingArgument;
    }
    const fixture_path = process_args[1];

    log.info("P5.2.D.4 — ZML RoPE K-only (k_proj + k_norm + rope), producer layer 13 sliding", .{});
    log.info("  layout=.sequential  scaling=.default  rope_theta={d}", .{ROPE_THETA});
    log.info("Loading fixture from {s}", .{fixture_path});

    var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, fixture_path);
    defer registry.deinit();

    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();

    const model: KRopeFixture = .init(store.view());

    log.info("Symbolic shapes:", .{});
    log.info("  hidden_input        : {f}", .{model.hidden_input});
    log.info("  k_proj_weight       : {f}", .{model.k_proj_weight});
    log.info("  k_norm_weight       : {f}", .{model.k_norm_weight});
    log.info("  k_after_norm_oracle : {f}", .{model.k_after_norm_oracle});
    log.info("  k_after_rope_oracle : {f}", .{model.k_after_rope_oracle});

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);

    const replicated_sharding = try zml.sharding.replicatedSharding(platform);

    log.info("Materializing buffers...", .{});
    var buffers = try model.load(arena.allocator(), io, platform, &store, &.{replicated_sharding});
    defer KRopeFixture.unloadBuffers(&buffers);
    log.info("Buffers loaded (5 tensors).", .{});

    log.info("Compiling forward (k_proj + reshape + rmsNorm + mul + rope, no V/transpose/cache)...", .{});
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

    // === Oracle slices ===
    var rope_ref_slice = try buffers.k_after_rope_oracle.toSliceAlloc(allocator, io);
    defer rope_ref_slice.free(allocator);
    const rope_ref = rope_ref_slice.items(f32);

    var norm_ref_slice = try buffers.k_after_norm_oracle.toSliceAlloc(allocator, io);
    defer norm_ref_slice.free(allocator);
    const norm_ref = norm_ref_slice.items(f32);

    if (rope_ref.len != K_ROPE_FLAT_LEN or norm_ref.len != K_ROPE_FLAT_LEN or data.len != K_ROPE_FLAT_LEN) {
        log.err("length mismatch: rope_ref={d} norm_ref={d} data={d} expected={d}", .{
            rope_ref.len, norm_ref.len, data.len, K_ROPE_FLAT_LEN,
        });
        return error.KRopeLengthMismatch;
    }

    // === Sanity RoPE (computed-side : zml.nn.rope vs k_after_norm) ===
    // pos 0 (flat 0..255) : RoPE doit etre l'identite (cos=1, sin=0).
    // pos 3 (flat 768..1023) : RoPE doit etre active (delta > 1e-3).
    var pos0_zml_vs_norm: f32 = 0.0;
    {
        var i: usize = 0;
        while (i < @as(usize, @intCast(D))) : (i += 1) {
            const diff = @abs(data[i] - norm_ref[i]);
            if (diff > pos0_zml_vs_norm) pos0_zml_vs_norm = diff;
        }
    }
    var pos3_zml_vs_norm: f32 = 0.0;
    {
        var i: usize = 0;
        while (i < @as(usize, @intCast(D))) : (i += 1) {
            const diff = @abs(data[768 + i] - norm_ref[768 + i]);
            if (diff > pos3_zml_vs_norm) pos3_zml_vs_norm = diff;
        }
    }
    log.info("Sanity RoPE pos 0 |k_rope_zml - k_norm_oracle|_max : {e:.6}  (expected ~0, identity)", .{pos0_zml_vs_norm});
    log.info("Sanity RoPE pos 3 |k_rope_zml - k_norm_oracle|_max : {e:.6}  (expected > 1e-3, active)", .{pos3_zml_vs_norm});
    if (pos0_zml_vs_norm > 1.0e-4) {
        log.err("BLOCK: pos 0 should be identity but ZML produced max diff {e:.6}", .{pos0_zml_vs_norm});
        log.err("  suspect: zml.nn.rope applied a rotation at pos 0 (pos_idx start != 0 ?)", .{});
        return error.KRopePos0NotIdentity;
    }
    if (pos3_zml_vs_norm <= 1.0e-3) {
        log.err("BLOCK: pos 3 should be active but ZML produced max diff {e:.6}", .{pos3_zml_vs_norm});
        log.err("  suspect: zml.nn.rope not applied (wrong tag .hd ? wrong axis ?)", .{});
        return error.KRopePos3NotActive;
    }

    // === Fixed-point blocks (2 x 8 valeurs) vs oracle ===
    log.info("Fixed-point blocks vs oracle k_after_rope (fp32):", .{});
    var max_block: f32 = 0.0;
    for (K_ROPE_BLOCKS) |block| {
        var block_max: f32 = 0.0;
        log.info("  Block {s} (flat_offset={d}):", .{ block.label, block.flat_offset });
        var i: usize = 0;
        while (i < block.width) : (i += 1) {
            const actual = data[block.flat_offset + i];
            const expected = rope_ref[block.flat_offset + i];
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

    // === Scan global 1024 valeurs vs oracle k_after_rope ===
    log.info("Scanning full tensor ({d} fp32) vs oracle k_after_rope:", .{K_ROPE_FLAT_LEN});

    var max_global: f32 = 0.0;
    var sum_abs: f64 = 0.0;
    var max_idx: usize = 0;
    for (data, rope_ref, 0..) |actual, expected, i| {
        const diff = @abs(actual - expected);
        if (diff > max_global) {
            max_global = diff;
            max_idx = i;
        }
        sum_abs += @as(f64, diff);
    }
    const mean_abs = @as(f32, @floatCast(sum_abs / @as(f64, @floatFromInt(K_ROPE_FLAT_LEN))));

    // Decompose max_idx into (s, kvh, d). strides : s=NKV*D, kvh=D, d=1.
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
    log.info("k_rope global max_diff: {e:.6} (tolerance {e:.1})", .{ max_diff, K_ROPE_TOLERANCE });
    log.info("  Expected ~ D.3 k_norm residual ~5e-7 (RoPE orthogonale, preserve)", .{});

    if (max_diff > K_ROPE_TOLERANCE) {
        log.err("BLOCK: k_rope max_diff exceeds tolerance", .{});
        log.err("  suspects: (1) wrong layout (interleaved vs sequential),", .{});
        log.err("            (2) wrong rope_theta, (3) wrong inv_freq formula,", .{});
        log.err("            (4) wrong rotation formula sign, (5) wrong pos_idx", .{});
        return error.KRopeFailed;
    }
    log.info("P5.2.D.4 PASS: ZML RoPE K-only producer layer 13 validated vs PyTorch oracle", .{});
    log.info("  (no v_proj, no v_norm, no transpose, no cache, no attention)", .{});
}
