// W4 — gates W1/W2/W3 de la brique poids 4-bit (plan docs/superpowers/plans/
// 2026-07-24-w4-j1-brique-e2b.md). CPU nominal (Platform.auto), pas de garde VRAM.
//
//   w1 <fixtures/w4_unpack.safetensors>                      — unpack+dequant bit-exact vs oracle
//                                                              compressed-tensors + contre-test corruption
//   w2 <weights_w4/model.safetensors> <fixtures/w4_mats.safetensors>
//                                                            — dequant des 5 modules réels, bit-exact u16
//   w3 <weights_w4/model.safetensors> <fixtures/w4_gemm.safetensors>
//                                                            — GEMM f32 x·dequant(q_proj L0) vs oracle
//
// Verdicts par erreur Zig : error.GateFailed / error.VacuousGate ; PASS -> log + exit 0.

const std = @import("std");
const log = std.log;
const zml = @import("zml");
const w4 = @import("w4.zig");

pub const std_options: std.Options = .{ .log_level = .info };

const usage = "Usage: gemma4_w4gate w1 <w4_unpack.safetensors> | w2 <model.safetensors> <w4_mats.safetensors> | w3 <model.safetensors> <w4_gemm.safetensors>";

const load_opts = .{ .parallelism = 1, .dma_chunks = 1, .dma_chunk_size = 16 * 1024 * 1024 };

// ---------------------------------------------------------------------------- W1

const W1Fix = struct {
    pk: zml.Tensor,
    sc: zml.Tensor,
    q_expected: zml.Tensor,
    deq_expected: zml.Tensor,
    pk_corrupt: zml.Tensor,

    pub fn init(v: zml.io.TensorStore.View) W1Fix {
        return .{
            .pk = v.createTensor("packed", .{ .o, .gp }, null),
            .sc = v.createTensor("scales", .{ .o, .g }, null),
            .q_expected = v.createTensor("q_expected", .{ .o, .d }, null),
            .deq_expected = v.createTensor("deq_expected", .{ .o, .d }, null),
            .pk_corrupt = v.createTensor("packed_corrupt", .{ .o, .gp }, null),
        };
    }
};

const G1 = struct {
    pub fn forward(pk: zml.Tensor, sc: zml.Tensor, pkc: zml.Tensor) struct { zml.Tensor, zml.Tensor, zml.Tensor } {
        return .{ w4.unpackW4(pk), w4.dequantW4(pk, sc), w4.unpackW4(pkc) };
    }
};

fn gateW1(allocator: std.mem.Allocator, arena: std.mem.Allocator, io: std.Io, platform: *zml.Platform, sharding: zml.sharding.Sharding, fixture_path: []const u8) !void {
    log.info("W1 — unpack/dequant vs oracle compressed-tensors ({s})", .{fixture_path});

    var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, fixture_path);
    defer registry.deinit();
    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();

    const fix: W1Fix = .init(store.view());
    const fix_buf = try zml.io.load(W1Fix, &fix, arena, io, platform, &store, .{ .shardings = &.{sharding}, .parallelism = load_opts.parallelism, .dma_chunks = load_opts.dma_chunks, .dma_chunk_size = load_opts.dma_chunk_size });

    var exe = try platform.compileFn(allocator, io, G1.forward, .{ fix.pk, fix.sc, fix.pk_corrupt }, .{ .shardings = &.{sharding} });
    defer exe.deinit();
    var args = try exe.args(allocator);
    defer args.deinit(allocator);
    var results = try exe.results(allocator);
    defer results.deinit(allocator);
    args.set(.{ fix_buf.pk, fix_buf.sc, fix_buf.pk_corrupt });
    exe.call(args, &results);
    var r_unpack, var r_deq, var r_corrupt = results.get(struct { zml.Buffer, zml.Buffer, zml.Buffer });
    defer r_unpack.deinit();
    defer r_deq.deinit();
    defer r_corrupt.deinit();

    // Références host
    var qexp_s = try fix_buf.q_expected.toSliceAlloc(allocator, io);
    defer qexp_s.free(allocator);
    const q_expected = qexp_s.items(i8);
    var dexp_s = try fix_buf.deq_expected.toSliceAlloc(allocator, io);
    defer dexp_s.free(allocator);
    const deq_expected = dexp_s.items(u16);

    // (a) unpack == q_expected (valeurs entières : unpack sort du i32, la fixture stocke i8)
    var unpack_s = try r_unpack.toSliceAlloc(allocator, io);
    defer unpack_s.free(allocator);
    const unpack_v = unpack_s.items(i32);
    if (unpack_v.len != q_expected.len) {
        log.err("W1/unpack : longueurs D2H inattendues ({d} != {d})", .{ unpack_v.len, q_expected.len });
        return error.GateFailed;
    }
    for (unpack_v, q_expected, 0..) |got, exp, i| {
        if (got != @as(i32, exp)) {
            log.err("W1/unpack : divergence à l'index {d} : got={d} expected={d}", .{ i, got, exp });
            return error.GateFailed;
        }
    }
    log.info("  unpack == q_expected : {d}/{d} valeurs entières exactes", .{ unpack_v.len, unpack_v.len });

    // (b) dequant == deq_expected, BIT-EXACT bf16 (u16 = motif de bits, comparaison canonique)
    var deq_s = try r_deq.toSliceAlloc(allocator, io);
    defer deq_s.free(allocator);
    const deq_v = deq_s.items(u16);
    if (deq_v.len != deq_expected.len) {
        log.err("W1/dequant : longueurs D2H inattendues ({d} != {d})", .{ deq_v.len, deq_expected.len });
        return error.GateFailed;
    }
    for (deq_v, deq_expected, 0..) |got, exp, i| {
        if (got != exp) {
            log.err("W1/dequant : divergence bit-exact à l'index {d} : got=0x{x:0>4} expected=0x{x:0>4}", .{ i, got, exp });
            return error.GateFailed;
        }
    }
    log.info("  dequant == deq_expected : {d}/{d} u16 bit-exact", .{ deq_v.len, deq_v.len });

    // (c) NON-VACUITÉ : unpack(packed_corrupt) DOIT diverger de q_expected (attendu : exactement 1,
    // le nibble 1 de la rangée 0). Un contre-test qui ne diverge pas = gate vide.
    var corrupt_s = try r_corrupt.toSliceAlloc(allocator, io);
    defer corrupt_s.free(allocator);
    const corrupt_v = corrupt_s.items(i32);
    if (corrupt_v.len != q_expected.len) {
        log.err("W1/corrupt : longueurs D2H inattendues ({d} != {d})", .{ corrupt_v.len, q_expected.len });
        return error.GateFailed;
    }
    var n_diverg: usize = 0;
    var first_diverg: usize = 0;
    for (corrupt_v, q_expected, 0..) |got, exp, i| {
        if (got != @as(i32, exp)) {
            if (n_diverg == 0) first_diverg = i;
            n_diverg += 1;
        }
    }
    if (n_diverg == 0) {
        log.err("W1/corrupt : unpack(packed_corrupt) == q_expected sur les {d} éléments — contre-test VIDE", .{corrupt_v.len});
        return error.VacuousGate;
    }
    log.info("  contre-test corruption : {d} élément(s) divergent(s) (attendu 1), premier à l'index {d}", .{ n_diverg, first_diverg });

    // Bonus informatif (non bloquant) : variante pk.bitCast(.u4) — l'axe inséré s'appelle .bitcast.
    // Si le backend la rejette, on logge le rejet ; le verdict du gate n'en dépend JAMAIS.
    const G1B = struct {
        pub fn forward(pk: zml.Tensor) zml.Tensor {
            const nib = pk.bitCast(.u4); // {.o, .gp, .bitcast=8} u4, ordre little-endian
            const q = nib.convert(.i32).sub(zml.Tensor.scalar(8, .i32));
            return q.merge(.{ .d = .{ .gp, .bitcast } });
        }
    };
    bitcast_blk: {
        var exe_b = platform.compileFn(allocator, io, G1B.forward, .{fix.pk}, .{ .shardings = &.{sharding} }) catch |err| {
            log.info("  [info] variante bitCast(.u4) : rejetée à la compile ({s})", .{@errorName(err)});
            break :bitcast_blk;
        };
        defer exe_b.deinit();
        var args_b = try exe_b.args(allocator);
        defer args_b.deinit(allocator);
        var results_b = try exe_b.results(allocator);
        defer results_b.deinit(allocator);
        args_b.set(.{fix_buf.pk});
        exe_b.call(args_b, &results_b);
        var r_b = results_b.get(zml.Buffer);
        defer r_b.deinit();
        var b_s = try r_b.toSliceAlloc(allocator, io);
        defer b_s.free(allocator);
        const b_v = b_s.items(i32);
        var b_mismatch: usize = 0;
        if (b_v.len == q_expected.len) {
            for (b_v, q_expected) |got, exp| {
                if (got != @as(i32, exp)) b_mismatch += 1;
            }
            log.info("  [info] variante bitCast(.u4) : compile OK, {d}/{d} mismatch vs q_expected", .{ b_mismatch, b_v.len });
        } else {
            log.info("  [info] variante bitCast(.u4) : compile OK mais longueur inattendue ({d} != {d})", .{ b_v.len, q_expected.len });
        }
    }

    log.info("PASS w1", .{});
}

// ---------------------------------------------------------------------------- W2

const G2 = struct {
    pub fn forward(pk: zml.Tensor, sc: zml.Tensor) zml.Tensor {
        return w4.dequantW4(pk, sc);
    }
};

const OneT = struct { t: zml.Tensor };

const W2Mod = struct { name: []const u8, key: []const u8 };
const W2_MODS = [_]W2Mod{
    .{ .name = "model.language_model.layers.0.self_attn.q_proj", .key = "deq_q0" },
    .{ .name = "model.language_model.layers.4.self_attn.q_proj", .key = "deq_q4" },
    .{ .name = "model.language_model.layers.20.mlp.down_proj", .key = "deq_dn20" },
    .{ .name = "model.language_model.layers.0.per_layer_projection", .key = "deq_plp0" },
    .{ .name = "model.language_model.per_layer_model_projection", .key = "deq_plmp" },
};

fn gateW2(allocator: std.mem.Allocator, arena: std.mem.Allocator, io: std.Io, platform: *zml.Platform, sharding: zml.sharding.Sharding, ckpt_path: []const u8, fixture_path: []const u8) !void {
    log.info("W2 — dequant des 5 modules réels vs fixture MATS (bit-exact u16)", .{});

    // DEUX stores : checkpoint packé + fixture (pattern gemma4_gen_long_gpu).
    var reg_ck: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, ckpt_path);
    defer reg_ck.deinit();
    var store_ck: zml.io.TensorStore = .fromRegistry(allocator, &reg_ck);
    defer store_ck.deinit();
    var reg_fx: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, fixture_path);
    defer reg_fx.deinit();
    var store_fx: zml.io.TensorStore = .fromRegistry(allocator, &reg_fx);
    defer store_fx.deinit();

    var n_pass: usize = 0;
    inline for (W2_MODS) |m| {
        log.info("  module {s} -> {s}", .{ m.name, m.key });
        const lin: w4.W4Lin = .init(store_ck.view(), m.name);
        const lin_buf = try zml.io.load(w4.W4Lin, &lin, arena, io, platform, &store_ck, .{ .shardings = &.{sharding}, .parallelism = load_opts.parallelism, .dma_chunks = load_opts.dma_chunks, .dma_chunk_size = load_opts.dma_chunk_size });
        const exp_sym: OneT = .{ .t = store_fx.view().createTensor(m.key, .{ .o, .d }, null) };
        const exp_buf = try zml.io.load(OneT, &exp_sym, arena, io, platform, &store_fx, .{ .shardings = &.{sharding}, .parallelism = load_opts.parallelism, .dma_chunks = load_opts.dma_chunks, .dma_chunk_size = load_opts.dma_chunk_size });

        // UN compileFn par module : les 5 shapes diffèrent.
        var exe = try platform.compileFn(allocator, io, G2.forward, .{ lin.pk, lin.sc }, .{ .shardings = &.{sharding} });
        defer exe.deinit();
        var args = try exe.args(allocator);
        defer args.deinit(allocator);
        var results = try exe.results(allocator);
        defer results.deinit(allocator);
        args.set(.{ lin_buf.pk, lin_buf.sc });
        exe.call(args, &results);
        var r_deq = results.get(zml.Buffer);
        defer r_deq.deinit();

        var got_s = try r_deq.toSliceAlloc(allocator, io);
        defer got_s.free(allocator);
        const got = got_s.items(u16);
        var exp_s = try exp_buf.t.toSliceAlloc(allocator, io);
        defer exp_s.free(allocator);
        const exp = exp_s.items(u16);

        if (got.len != exp.len) {
            log.err("W2/{s} : longueurs D2H inattendues ({d} != {d})", .{ m.key, got.len, exp.len });
            return error.GateFailed;
        }
        var ok = true;
        for (got, exp, 0..) |g, e, i| {
            if (g != e) {
                log.err("W2/{s} : divergence bit-exact à l'index {d} : got=0x{x:0>4} expected=0x{x:0>4}", .{ m.key, i, g, e });
                ok = false;
                break;
            }
        }
        if (!ok) return error.GateFailed;
        n_pass += 1;
        log.info("    {s} : {d} u16 bit-exact", .{ m.key, got.len });
    }

    if (n_pass != W2_MODS.len) return error.GateFailed;
    log.info("PASS w2 {d}/{d}", .{ n_pass, W2_MODS.len });
}

// ---------------------------------------------------------------------------- W3

const W3_MAX_ABS: f32 = 1.0e-4;
const W3_MEAN_ABS: f32 = 1.0e-6;
const W3_MODULE = "model.language_model.layers.0.self_attn.q_proj";

const W3Fix = struct {
    x: zml.Tensor,
    out_expected: zml.Tensor,

    pub fn init(v: zml.io.TensorStore.View) W3Fix {
        return .{
            .x = v.createTensor("x", .{ .b, .s, .d }, null),
            .out_expected = v.createTensor("out_expected", .{ .b, .s, .o }, null),
        };
    }
};

const G3 = struct {
    pub fn forward(x: zml.Tensor, pk: zml.Tensor, sc: zml.Tensor) zml.Tensor {
        return x.convert(.f32).dot(w4.dequantW4(pk, sc).convert(.f32), .d);
    }
};

fn gateW3(allocator: std.mem.Allocator, arena: std.mem.Allocator, io: std.Io, platform: *zml.Platform, sharding: zml.sharding.Sharding, ckpt_path: []const u8, fixture_path: []const u8) !void {
    log.info("W3 — GEMM f32 x·dequant({s}) vs oracle", .{W3_MODULE});

    var reg_ck: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, ckpt_path);
    defer reg_ck.deinit();
    var store_ck: zml.io.TensorStore = .fromRegistry(allocator, &reg_ck);
    defer store_ck.deinit();
    var reg_fx: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, fixture_path);
    defer reg_fx.deinit();
    var store_fx: zml.io.TensorStore = .fromRegistry(allocator, &reg_fx);
    defer store_fx.deinit();

    const lin: w4.W4Lin = .init(store_ck.view(), W3_MODULE);
    const lin_buf = try zml.io.load(w4.W4Lin, &lin, arena, io, platform, &store_ck, .{ .shardings = &.{sharding}, .parallelism = load_opts.parallelism, .dma_chunks = load_opts.dma_chunks, .dma_chunk_size = load_opts.dma_chunk_size });
    const fix: W3Fix = .init(store_fx.view());
    const fix_buf = try zml.io.load(W3Fix, &fix, arena, io, platform, &store_fx, .{ .shardings = &.{sharding}, .parallelism = load_opts.parallelism, .dma_chunks = load_opts.dma_chunks, .dma_chunk_size = load_opts.dma_chunk_size });

    var exe = try platform.compileFn(allocator, io, G3.forward, .{ fix.x, lin.pk, lin.sc }, .{ .shardings = &.{sharding} });
    defer exe.deinit();
    var args = try exe.args(allocator);
    defer args.deinit(allocator);
    var results = try exe.results(allocator);
    defer results.deinit(allocator);
    args.set(.{ fix_buf.x, lin_buf.pk, lin_buf.sc });
    exe.call(args, &results);
    var r_out = results.get(zml.Buffer);
    defer r_out.deinit();

    var got_s = try r_out.toSliceAlloc(allocator, io);
    defer got_s.free(allocator);
    const got = got_s.items(f32);
    var exp_s = try fix_buf.out_expected.toSliceAlloc(allocator, io);
    defer exp_s.free(allocator);
    const exp = exp_s.items(f32);

    if (got.len != exp.len) {
        log.err("W3 : longueurs D2H inattendues ({d} != {d})", .{ got.len, exp.len });
        return error.GateFailed;
    }
    var max_abs: f32 = 0.0;
    var max_idx: usize = 0;
    var sum_abs: f64 = 0.0;
    for (got, exp, 0..) |g, e, i| {
        const diff = @abs(g - e);
        if (diff > max_abs) {
            max_abs = diff;
            max_idx = i;
        }
        sum_abs += @as(f64, diff);
    }
    const mean_abs = @as(f32, @floatCast(sum_abs / @as(f64, @floatFromInt(got.len))));
    log.info("  max_abs {e:.6} at o={d} (got={d:.7} expected={d:.7}), mean_abs {e:.6}", .{ max_abs, max_idx, got[max_idx], exp[max_idx], mean_abs });

    // points fixes: ajoutés au câblage W3 (Task 6)

    if (max_abs > W3_MAX_ABS or mean_abs > W3_MEAN_ABS) {
        log.err("W3 : hors tolérance (max_abs {e:.6} <= {e:.1} ? mean_abs {e:.6} <= {e:.1} ?)", .{ max_abs, W3_MAX_ABS, mean_abs, W3_MEAN_ABS });
        return error.GateFailed;
    }
    log.info("PASS w3", .{});
}

// ---------------------------------------------------------------------------- main

pub fn main(init: std.process.Init) !void {
    const arena = init.arena;
    const allocator = init.gpa;
    const io = init.io;

    const process_args = try init.minimal.args.toSlice(arena.allocator());
    if (process_args.len < 3) {
        log.err("{s}", .{usage});
        return error.MissingArgument;
    }
    const mode = process_args[1];

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);
    const sharding = try zml.sharding.replicatedSharding(platform);
    log.info("W4 gates — backend = {s} (CPU nominal)", .{@tagName(platform.target)});

    if (std.mem.eql(u8, mode, "w1")) {
        try gateW1(allocator, arena.allocator(), io, platform, sharding, process_args[2]);
    } else if (std.mem.eql(u8, mode, "w2")) {
        if (process_args.len < 4) {
            log.err("{s}", .{usage});
            return error.MissingArgument;
        }
        try gateW2(allocator, arena.allocator(), io, platform, sharding, process_args[2], process_args[3]);
    } else if (std.mem.eql(u8, mode, "w3")) {
        if (process_args.len < 4) {
            log.err("{s}", .{usage});
            return error.MissingArgument;
        }
        try gateW3(allocator, arena.allocator(), io, platform, sharding, process_args[2], process_args[3]);
    } else {
        log.err("mode inconnu '{s}' — {s}", .{ mode, usage });
        return error.MissingArgument;
    }
}
