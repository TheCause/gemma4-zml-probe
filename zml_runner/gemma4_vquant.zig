// Q3 — MSE V-only quantizer chain en ZML (norm L2 + Hadamard + nearest-centroid + inverse).
//
// Chaine complete TurboQuantMSE (turboquant.py) DANS LE GRAPHE ZML :
//   norm = ||v||_2 par vecteur, ARRONDIE en fp16 (reproduit mse.quant)
//   u    = v / norm                                   (broadcast .hd=1)
//   y    = u @ Pi.T  (= u .dot Pi sur .hd)            [.k,.e]
//   idx  = argmin_c (y - codebook)^2                  nearest-centroid
//   yhat = codebook.gather(.{.c = idx})               [.k,.e]
//   uhat = yhat @ Pi (= yhat .dot Pi sur .e)          [.k,.hd]
//   vhat = uhat * norm                                [.k,.hd]
//
// quantizeV est une fonction libre SANS constante globale (shapes derives via .dim()),
// reutilisable en Q4 ou D=1536 existe deja. Compare vhat a v_hat_oracle (PyTorch).
//
// CLI : gemma4_vquant <path-to-spike_vquant_<d>.safetensors>

const std = @import("std");
const log = std.log;

const zml = @import("zml");
const stdx = zml.stdx;

pub const std_options: std.Options = .{
    .log_level = .info,
};

const FLIP_THRESHOLD: f32 = 1.0e-3; // au-dela = vrai flip de centroide (gap codebook ~0.02)
const MAX_FLIP_FRAC: f32 = 0.05; // PASS si < 5% de coords flippees (norm fp16 + matmul Hadamard PJRT)

/// v:[.k,.hd], cb:[.c], Pi:[.e,.hd]  ->  v_hat:[.k,.hd]
fn quantizeV(v: zml.Tensor, cb: zml.Tensor, Pi: zml.Tensor) zml.Tensor {
    // norm L2 par vecteur — MSE.quant ARRONDIT la norm en fp16 (turboquant.py) : reproduire
    const norm = v.mul(v).sum(.hd).sqrt().convert(.f16).convert(.f32); // [.k,.hd=1]
    const u = v.div(norm); // broadcast (.hd=1)
    const y = u.dot(Pi, .hd); // [.k,.e]  (= u @ Pi.T)
    const target = zml.Shape.init(.{ y.dim(.k), y.dim(.e), cb.dim(.c) }, .f32)
        .withTags(.{ .k, .e, .c });
    const yr3 = y.appendAxes(.{.c}).broad(target);
    const cb3 = cb.insertAxes(0, .{ .k, .e }).broad(target);
    const diff = yr3.sub(cb3);
    const idx = diff.mul(diff).scale(-1.0).argMax(.c).indices.squeeze(.c); // [.k,.e]
    const y_hat = cb.gather(.{ .c = idx }, .{}); // [.k,.e]
    const u_hat = y_hat.dot(Pi, .e); // [.k,.hd]  (= y_hat @ Pi)
    return u_hat.mul(norm);
}

const VquantFixture = struct {
    v: zml.Tensor, // [.k, .hd]
    hadamard: zml.Tensor, // [.e, .hd]  (Pi[e,d])
    codebook: zml.Tensor, // [.c]
    v_hat_oracle: zml.Tensor, // [.k, .hd]

    pub fn init(store: zml.io.TensorStore.View) VquantFixture {
        return .{
            .v = store.createTensor("v", .{ .k, .hd }, null),
            .hadamard = store.createTensor("hadamard", .{ .e, .hd }, null),
            .codebook = store.createTensor("codebook", .{.c}, null),
            .v_hat_oracle = store.createTensor("v_hat_oracle", .{ .k, .hd }, null),
        };
    }

    pub fn load(
        self: *const VquantFixture,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *const zml.io.TensorStore,
        shardings: []const zml.sharding.Sharding,
    ) !zml.Bufferized(VquantFixture) {
        return zml.io.load(VquantFixture, self, allocator, io, platform, store, .{
            .shardings = shardings,
            .parallelism = 1,
            .dma_chunks = 1,
            .dma_chunk_size = 16 * 1024 * 1024,
        });
    }

    pub fn unloadBuffers(self: *zml.Bufferized(VquantFixture)) void {
        self.v.deinit();
        self.hadamard.deinit();
        self.codebook.deinit();
        self.v_hat_oracle.deinit();
    }

    pub fn forward(self: VquantFixture) zml.Tensor {
        return quantizeV(self.v, self.codebook, self.hadamard);
    }
};

pub fn main(init: std.process.Init) !void {
    const arena = init.arena;
    const allocator = init.gpa;
    const io = init.io;

    const process_args = try init.minimal.args.toSlice(arena.allocator());
    if (process_args.len < 2) {
        log.err("Usage: gemma4_vquant <path-to-spike_vquant_<d>.safetensors>", .{});
        return error.MissingArgument;
    }
    const fixture_path = process_args[1];

    log.info("Q3 — MSE V-only quantizer chain en ZML", .{});
    log.info("Loading fixture from {s}", .{fixture_path});

    var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, fixture_path);
    defer registry.deinit();

    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();

    const model: VquantFixture = .init(store.view());

    log.info("Symbolic shapes:", .{});
    log.info("  v            : {f}", .{model.v});
    log.info("  hadamard     : {f}", .{model.hadamard});
    log.info("  codebook     : {f}", .{model.codebook});
    log.info("  v_hat_oracle : {f}", .{model.v_hat_oracle});

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);

    const replicated_sharding = try zml.sharding.replicatedSharding(platform);

    log.info("Materializing buffers...", .{});
    var buffers = try model.load(arena.allocator(), io, platform, &store, &.{replicated_sharding});
    defer VquantFixture.unloadBuffers(&buffers);

    log.info("Compiling forward (norm + Hadamard + nearest-centroid + inverse)...", .{});
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

    var ref_slice = try buffers.v_hat_oracle.toSliceAlloc(allocator, io);
    defer ref_slice.free(allocator);
    const ref = ref_slice.items(f32);

    // taille calculee depuis l'oracle (meme runner pour d=256 et d=512)
    const flat_len: usize = ref.len;
    if (data.len != flat_len) {
        log.err("length mismatch: data={d} ref={d}", .{ data.len, flat_len });
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
    const mean_abs = @as(f32, @floatCast(sum_abs / @as(f64, @floatFromInt(flat_len))));
    const flip_frac = @as(f32, @floatFromInt(n_flip)) / @as(f32, @floatFromInt(flat_len));

    log.info("Comparaison v_hat ZML vs oracle PyTorch ({d} coords) :", .{flat_len});
    log.info("  exact (diff==0) : {d}/{d} ({d:.1}%)", .{ n_exact, flat_len, 100.0 * @as(f32, @floatFromInt(n_exact)) / @as(f32, @floatFromInt(flat_len)) });
    log.info("  max_abs  : {e:.6}", .{max_abs});
    log.info("  mean_abs : {e:.6}", .{mean_abs});
    log.info("  flips (diff>{e:.1}) : {d}/{d} ({d:.2}%)", .{ FLIP_THRESHOLD, n_flip, flat_len, 100.0 * flip_frac });
    log.info("  (flips = argmin bascule a une frontiere de centroide sous norm fp16 + Hadamard PJRT)", .{});

    if (flip_frac > MAX_FLIP_FRAC) {
        log.err("FAIL: trop de flips ({d:.2}% > {d:.2}%) — la chaine MSE V-only ne reproduit pas l'oracle", .{ 100.0 * flip_frac, 100.0 * MAX_FLIP_FRAC });
        return error.TooManyFlips;
    }
    log.info("Q3 PASS : chaine MSE V-only (norm+Hadamard+nearest-centroid+inverse) reproduit l'oracle en ZML.", .{});
}
