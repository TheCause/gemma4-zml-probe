// P5.2.E.mask — ZML masque sliding RÉEL (cas synthétique S=8, window=3).
//
// Objectif : fermer le trou de couverture E.0/E.1. À S=4 et sliding_window=512, le
// masque sliding dégénère en causal (la garde `qlen >= window_len` de causalAttnMask
// est fausse -> fenêtre non appliquée). Ici S=8 >= window=3 -> la fenêtre MORD, et on
// valide la vraie logique de fenêtrage côté ZML.
//
// Pipeline ZML strict (mask only) :
//   mask          = zml.nn.causalAttnMask(.{ .q = 8, .k = 8 }, .f32, 3)   // {.q,.k} additif (0 / finfo.min)
//   scores_masked = scores_synth.add(mask.broad(scores_synth.shape()))    // broadcast sur .b,.h
//
// Convention (cf zml/nn.zig causalAttnMask, = transformers sliding_window_overlay) :
//   visible  ⟺  (k <= q)  AND  (q < k + window)  ⟺  q - window < k <= q.
//
// Le masque ZML est construit par le helper natif, INDÉPENDAMMENT de l'oracle PyTorch
// (qui le construit from-scratch en numpy). On compare les DEUX :
//   (1) mask ZML vs `sliding_mask` oracle [8,8]      -> valide la construction
//   (2) scores_masked ZML vs oracle [1,2,8,8]        -> valide l'application + broadcast
//
// Comparaison robuste aux finfo.min (-3.4e38) : visible bit-exact, masqué < -1e30,
// structure (quelles positions masquées) strictement identique.
//
// Interdits stricts P5.2.E.mask : softmax, context, dot(V), layer 14, full attention,
// scaling, RoPE, Q/K/V proj.
//
// CLI : gemma4_sliding_mask <path-to-p5_2_emask_sliding_layer_synthetic.safetensors>

const std = @import("std");
const log = std.log;

const zml = @import("zml");
const stdx = zml.stdx;

pub const std_options: std.Options = .{
    .log_level = .info,
};

const B: i64 = 1;
const NH: i64 = 2; // n heads synthétiques (broadcast du masque)
const S: i64 = 8; // séquence (q = k)
const WINDOW: u32 = 3; // sliding_window mordant (< S)

const MASK_FLAT_LEN: usize = @intCast(S * S); // 64
const MASKED_FLAT_LEN: usize = @intCast(B * NH * S * S); // 128
const MASK_THRESHOLD: f32 = -1.0e30; // en-dessous = position masquée (finfo.min ~ -3.4e38)
const VISIBLE_TOL: f32 = 1.0e-6; // positions visibles attendues bit-exact (add de 0)
const EXPECTED_MASKED_MASK: usize = 43; // sur 64 (cf oracle)
const EXPECTED_MASKED_SCORES: usize = 86; // 43 x NH=2

/// Fixture E.mask : scores synthétiques + masque oracle + scores masqués oracle.
const SlidingMaskFixture = struct {
    scores_synth: zml.Tensor,
    sliding_mask_oracle: zml.Tensor,
    scores_masked_oracle: zml.Tensor,

    pub fn init(store: zml.io.TensorStore.View) SlidingMaskFixture {
        return .{
            .scores_synth = store.createTensor(
                "scores_synth",
                .{ .b, .h, .q, .k },
                null,
            ),
            .sliding_mask_oracle = store.createTensor(
                "sliding_mask",
                .{ .q, .k },
                null,
            ),
            .scores_masked_oracle = store.createTensor(
                "scores_masked",
                .{ .b, .h, .q, .k },
                null,
            ),
        };
    }

    pub fn load(
        self: *const SlidingMaskFixture,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *const zml.io.TensorStore,
        shardings: []const zml.sharding.Sharding,
    ) !zml.Bufferized(SlidingMaskFixture) {
        return zml.io.load(SlidingMaskFixture, self, allocator, io, platform, store, .{
            .shardings = shardings,
            .parallelism = 1,
            .dma_chunks = 1,
            .dma_chunk_size = 16 * 1024 * 1024,
        });
    }

    pub fn unloadBuffers(self: *zml.Bufferized(SlidingMaskFixture)) void {
        self.scores_synth.deinit();
        self.sliding_mask_oracle.deinit();
        self.scores_masked_oracle.deinit();
    }

    /// Forward E.mask : construit le masque sliding (helper natif) + l'applique.
    ///   mask          = causalAttnMask(.{ .q = S, .k = S }, .f32, WINDOW)   {.q,.k}
    ///   scores_masked = scores_synth.add(mask.broad(...))                   {.b,.h,.q,.k}
    /// Retourne (mask, scores_masked). Pas de softmax/context/V.
    pub fn forward(self: SlidingMaskFixture) struct { zml.Tensor, zml.Tensor } {
        const mask = zml.nn.causalAttnMask(.{ .q = S, .k = S }, .f32, WINDOW);
        const scores_masked = self.scores_synth.add(mask.broad(self.scores_synth.shape()));
        return .{ mask, scores_masked };
    }
};

/// Comparaison robuste aux finfo.min : visible bit-exact, masqué < -1e30, structure identique.
fn compareMasked(
    name: []const u8,
    data: []const f32,
    oracle: []const f32,
    expected_len: usize,
    expected_masked: usize,
) !struct { max_diff_visible: f32, masked_count: usize } {
    if (data.len != oracle.len or data.len != expected_len) {
        log.err("{s}: length mismatch data={d} oracle={d} expected={d}", .{ name, data.len, oracle.len, expected_len });
        return error.MaskLengthMismatch;
    }
    var max_diff_visible: f32 = 0.0;
    var masked_count: usize = 0;
    var struct_mismatch: usize = 0;
    for (data, oracle) |z, o| {
        if (o < MASK_THRESHOLD) {
            // Oracle masque cette position -> ZML doit aussi la masquer.
            masked_count += 1;
            if (z >= MASK_THRESHOLD) struct_mismatch += 1;
        } else {
            // Oracle visible -> ZML doit être visible ET bit-exact.
            if (z < MASK_THRESHOLD) {
                struct_mismatch += 1;
            } else {
                const diff = @abs(z - o);
                if (diff > max_diff_visible) max_diff_visible = diff;
            }
        }
    }
    log.info("{s}: masked={d} (expected {d}), visible max_diff={e:.6}, struct_mismatch={d}", .{
        name, masked_count, expected_masked, max_diff_visible, struct_mismatch,
    });
    if (struct_mismatch != 0) {
        log.err("BLOCK: {s} structure mismatch on {d} positions (mask vs oracle disagree)", .{ name, struct_mismatch });
        return error.MaskStructureMismatch;
    }
    if (masked_count != expected_masked) {
        log.err("BLOCK: {s} masked_count {d} != expected {d}", .{ name, masked_count, expected_masked });
        return error.MaskCountMismatch;
    }
    if (max_diff_visible > VISIBLE_TOL) {
        log.err("BLOCK: {s} visible positions diff {e:.6} > tol {e:.1}", .{ name, max_diff_visible, VISIBLE_TOL });
        return error.MaskVisibleDiff;
    }
    return .{ .max_diff_visible = max_diff_visible, .masked_count = masked_count };
}

pub fn main(init: std.process.Init) !void {
    const arena = init.arena;
    const allocator = init.gpa;
    const io = init.io;

    const process_args = try init.minimal.args.toSlice(arena.allocator());
    if (process_args.len < 2) {
        log.err("Usage: gemma4_sliding_mask <path-to-p5_2_emask_sliding_layer_synthetic.safetensors>", .{});
        return error.MissingArgument;
    }
    const fixture_path = process_args[1];

    log.info("P5.2.E.mask — ZML sliding window mask (S={d}, window={d}, mask only, no softmax/context)", .{ S, WINDOW });
    log.info("Loading fixture from {s}", .{fixture_path});

    var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, fixture_path);
    defer registry.deinit();

    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();

    const model: SlidingMaskFixture = .init(store.view());

    log.info("Symbolic shapes:", .{});
    log.info("  scores_synth        : {f}", .{model.scores_synth});
    log.info("  sliding_mask_oracle : {f}", .{model.sliding_mask_oracle});
    log.info("  scores_masked_oracle: {f}", .{model.scores_masked_oracle});

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);

    const replicated_sharding = try zml.sharding.replicatedSharding(platform);

    var buffers = try model.load(arena.allocator(), io, platform, &store, &.{replicated_sharding});
    defer SlidingMaskFixture.unloadBuffers(&buffers);
    log.info("Buffers loaded (3 tensors).", .{});

    log.info("Compiling forward (causalAttnMask window={d} + add broadcast)...", .{WINDOW});
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

    var mask_buf, var masked_buf = results.get(struct { zml.Buffer, zml.Buffer });
    defer mask_buf.deinit();
    defer masked_buf.deinit();

    log.info("Forward mask shape          : {f} (expected [q=8, k=8])", .{mask_buf.shape()});
    log.info("Forward scores_masked shape : {f} (expected [b=1, h=2, q=8, k=8])", .{masked_buf.shape()});

    var mask_slice = try mask_buf.toSliceAlloc(allocator, io);
    defer mask_slice.free(allocator);
    const mask_data = mask_slice.items(f32);

    var masked_slice = try masked_buf.toSliceAlloc(allocator, io);
    defer masked_slice.free(allocator);
    const masked_data = masked_slice.items(f32);

    // === Grille de visibilité reconstruite depuis le masque ZML (preuve que la fenêtre mord) ===
    log.info("ZML sliding mask grid (visible 'o' / masqué '.'), S={d} window={d}:", .{ S, WINDOW });
    {
        var q: usize = 0;
        while (q < @as(usize, @intCast(S))) : (q += 1) {
            var buf: [16]u8 = undefined;
            var k: usize = 0;
            while (k < @as(usize, @intCast(S))) : (k += 1) {
                const v = mask_data[q * @as(usize, @intCast(S)) + k];
                buf[k] = if (v < MASK_THRESHOLD) '.' else 'o';
            }
            log.info("  q={d}: [{s}]", .{ q, buf[0..@as(usize, @intCast(S))] });
        }
    }

    // === Comparaison (1) masque ZML vs oracle ===
    var ref_mask_slice = try buffers.sliding_mask_oracle.toSliceAlloc(allocator, io);
    defer ref_mask_slice.free(allocator);
    const ref_mask = ref_mask_slice.items(f32);
    const mask_stats = try compareMasked("mask", mask_data, ref_mask, MASK_FLAT_LEN, EXPECTED_MASKED_MASK);

    // === Comparaison (2) scores_masked ZML vs oracle ===
    var ref_masked_slice = try buffers.scores_masked_oracle.toSliceAlloc(allocator, io);
    defer ref_masked_slice.free(allocator);
    const ref_masked = ref_masked_slice.items(f32);
    const masked_stats = try compareMasked("scores_masked", masked_data, ref_masked, MASKED_FLAT_LEN, EXPECTED_MASKED_SCORES);

    log.info("---", .{});
    log.info("Summary :", .{});
    log.info("  mask          : masked={d}/{d}, visible max_diff={e:.6}", .{ mask_stats.masked_count, MASK_FLAT_LEN, mask_stats.max_diff_visible });
    log.info("  scores_masked : masked={d}/{d}, visible max_diff={e:.6}", .{ masked_stats.masked_count, MASKED_FLAT_LEN, masked_stats.max_diff_visible });
    log.info("P5.2.E.mask PASS: ZML sliding window mask (window={d}) validated vs PyTorch oracle", .{WINDOW});
    log.info("  (real sliding window mordant S=8 > window=3, mask only, no softmax/context/V)", .{});
}
