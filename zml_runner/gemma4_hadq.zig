// SPIKE B — derisquage ZML : Hadamard (dot constant) + nearest-centroid
// (sub + carre + argMax(-.) + gather) calcule DANS LE GRAPHE ZML.
//
// Prouve que le coeur du quantizer TurboQuant est portable en ZML :
//   yr   = y0 .dot hadamard sur .d        (Hadamard rotation)
//   sq   = (yr - codebook)^2  broadcast [.s,.e,.k]
//   idx  = argMax(-sq, .k).indices        (= argmin distance, nearest-centroid)
//   yhat = codebook.gather(.{.k = idx})   (reconstruction)
//
// Compare yhat au resultat oracle PyTorch. Ecarts residuels attendus = flips de
// centroide aux frontieres (matmul Hadamard PJRT vs BLAS ~1e-5). PASS si la
// mecanique reproduit l'oracle (mean_abs petit, flips rares).
//
// CLI : gemma4_hadq <path-to-spike_hadq.safetensors>

const std = @import("std");
const log = std.log;

const zml = @import("zml");
const stdx = zml.stdx;

pub const std_options: std.Options = .{
    .log_level = .info,
};

const S: i64 = 8;
const D: i64 = 256;
const K: i64 = 16;
const FLAT_LEN: usize = @intCast(S * D); // 2048
const FLIP_THRESHOLD: f32 = 1.0e-3; // au-dela = vrai flip de centroide (gap codebook ~0.02)
const MAX_FLIP_FRAC: f32 = 0.02; // PASS si < 2% de coords flippees (toutes ~Hadamard 1e-5)

const HadqFixture = struct {
    y0: zml.Tensor, // [.s, .d]
    hadamard: zml.Tensor, // [.e, .d]  (Pi[e,d])
    codebook: zml.Tensor, // [.k]
    y_hat_oracle: zml.Tensor, // [.s, .e]

    pub fn init(store: zml.io.TensorStore.View) HadqFixture {
        return .{
            .y0 = store.createTensor("y0", .{ .s, .d }, null),
            .hadamard = store.createTensor("hadamard", .{ .e, .d }, null),
            .codebook = store.createTensor("codebook", .{.k}, null),
            .y_hat_oracle = store.createTensor("y_hat_oracle", .{ .s, .e }, null),
        };
    }

    pub fn load(
        self: *const HadqFixture,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *const zml.io.TensorStore,
        shardings: []const zml.sharding.Sharding,
    ) !zml.Bufferized(HadqFixture) {
        return zml.io.load(HadqFixture, self, allocator, io, platform, store, .{
            .shardings = shardings,
            .parallelism = 1,
            .dma_chunks = 1,
            .dma_chunk_size = 16 * 1024 * 1024,
        });
    }

    pub fn unloadBuffers(self: *zml.Bufferized(HadqFixture)) void {
        self.y0.deinit();
        self.hadamard.deinit();
        self.codebook.deinit();
        self.y_hat_oracle.deinit();
    }

    /// Forward = Hadamard + nearest-centroid + gather.
    pub fn forward(self: HadqFixture) zml.Tensor {
        // 1. Hadamard : yr[.s,.e] = sum_d y0[.s,.d] * hadamard[.e,.d]
        const yr = self.y0.dot(self.hadamard, .d); // [.s, .e]

        // 2. broadcast yr et codebook vers [.s, .e, .k]
        const target = zml.Shape.init(.{ S, D, K }, .f32).withTags(.{ .s, .e, .k });
        const yr3 = yr.appendAxes(.{.k}).broad(target); // [.s,.e,.k=1] -> [.s,.e,.k]
        const cb3 = self.codebook.insertAxes(0, .{ .s, .e }).broad(target); // [1,1,K] -> [.s,.e,.k]

        // 3. distance carree (evite abs/sqrt)
        const diff = yr3.sub(cb3);
        const sq = diff.mul(diff); // [.s,.e,.k]

        // 4. argmin = argMax(-sq) sur .k
        const am = sq.scale(-1.0).argMax(.k);
        const idx = am.indices.squeeze(.k); // argMax garde .k=1 -> squeeze -> [.s,.e]

        // 5. gather codebook[idx]
        return self.codebook.gather(.{ .k = idx }, .{}); // [.s, .e]
    }
};

pub fn main(init: std.process.Init) !void {
    const arena = init.arena;
    const allocator = init.gpa;
    const io = init.io;

    const process_args = try init.minimal.args.toSlice(arena.allocator());
    if (process_args.len < 2) {
        log.err("Usage: gemma4_hadq <path-to-spike_hadq.safetensors>", .{});
        return error.MissingArgument;
    }
    const fixture_path = process_args[1];

    log.info("SPIKE B — Hadamard + nearest-centroid en ZML", .{});
    log.info("Loading fixture from {s}", .{fixture_path});

    var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, fixture_path);
    defer registry.deinit();

    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();

    const model: HadqFixture = .init(store.view());

    log.info("Symbolic shapes:", .{});
    log.info("  y0           : {f}", .{model.y0});
    log.info("  hadamard     : {f}", .{model.hadamard});
    log.info("  codebook     : {f}", .{model.codebook});
    log.info("  y_hat_oracle : {f}", .{model.y_hat_oracle});

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);

    const replicated_sharding = try zml.sharding.replicatedSharding(platform);

    log.info("Materializing buffers...", .{});
    var buffers = try model.load(arena.allocator(), io, platform, &store, &.{replicated_sharding});
    defer HadqFixture.unloadBuffers(&buffers);

    log.info("Compiling forward (Hadamard dot + argMax + gather)...", .{});
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

    var ref_slice = try buffers.y_hat_oracle.toSliceAlloc(allocator, io);
    defer ref_slice.free(allocator);
    const ref = ref_slice.items(f32);

    if (data.len != FLAT_LEN or ref.len != FLAT_LEN) {
        log.err("length mismatch: data={d} ref={d} expected={d}", .{ data.len, ref.len, FLAT_LEN });
        return error.LengthMismatch;
    }

    var max_abs: f32 = 0.0;
    var sum_abs: f64 = 0.0;
    var n_flip: usize = 0;
    var n_exact: usize = 0;
    for (data, ref) |a, b| {
        const diff = @abs(a - b);
        if (diff > max_abs) max_abs = diff;
        sum_abs += @as(f64, diff);
        if (diff > FLIP_THRESHOLD) n_flip += 1;
        if (diff == 0.0) n_exact += 1;
    }
    const mean_abs = @as(f32, @floatCast(sum_abs / @as(f64, @floatFromInt(FLAT_LEN))));
    const flip_frac = @as(f32, @floatFromInt(n_flip)) / @as(f32, @floatFromInt(FLAT_LEN));

    log.info("Comparaison yhat ZML vs oracle PyTorch ({d} coords) :", .{FLAT_LEN});
    log.info("  exact (diff==0) : {d}/{d} ({d:.1}%)", .{ n_exact, FLAT_LEN, 100.0 * @as(f32, @floatFromInt(n_exact)) / @as(f32, @floatFromInt(FLAT_LEN)) });
    log.info("  max_abs  : {e:.6}", .{max_abs});
    log.info("  mean_abs : {e:.6}", .{mean_abs});
    log.info("  flips (diff>{e:.1}) : {d}/{d} ({d:.2}%)", .{ FLIP_THRESHOLD, n_flip, FLAT_LEN, 100.0 * flip_frac });
    log.info("  (flips = argmin bascule a une frontiere de centroide sous bruit Hadamard ~1e-5)", .{});

    if (flip_frac > MAX_FLIP_FRAC) {
        log.err("FAIL: trop de flips ({d:.2}% > {d:.2}%) — la mecanique nearest-centroid ne reproduit pas l'oracle", .{ 100.0 * flip_frac, 100.0 * MAX_FLIP_FRAC });
        return error.TooManyFlips;
    }
    log.info("SPIKE B PASS : Hadamard + nearest-centroid + gather compilent et reproduisent l'oracle en ZML.", .{});
    log.info("  -> Le coeur du quantizer TurboQuant est portable en ZML. Risque B leve EN PRATIQUE.", .{});
}
