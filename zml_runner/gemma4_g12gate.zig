// G12 — gates U1b→U7 du 12B Unified (plan docs/superpowers/plans/2026-07-24-w4-j2-12b-unified.md,
// contrat docs/U_12B_CONTRACT.md). CPU nominal (Platform.auto), pas de garde VRAM.
// Nom 14c (quota comptime @typeName pjrt, piège 4). Pattern : gemma4_w4gate.zig.
//
//   w2-12b <weights_12b/model.safetensors> <fixtures/u_mats12.safetensors>
//       — U1b : dequantW4 (brique J1, réutilisée telle quelle) sur les 3 familles de shapes 12B
//         (q_proj L0 sliding 4096×3840, q_proj L5 full 8192×3840, down_proj L0 3840×15360),
//         poids lus du checkpoint PACKÉ, comparés bit-exact bf16 (u16) à la fixture du script 63
//         (--fixture-u1b). Modes u2/u3/… : Tasks 3+.
//
// Verdicts par erreur Zig : error.GateFailed / error.VacuousGate ; PASS -> log + exit 0.

const std = @import("std");
const log = std.log;
const zml = @import("zml");
const w4 = @import("w4.zig");
const g12 = @import("g12.zig");

pub const std_options: std.Options = .{ .log_level = .info };

const usage = "Usage: gemma4_g12gate w2-12b <model.safetensors> <u_mats12.safetensors>";

const load_opts = .{ .parallelism = 1, .dma_chunks = 1, .dma_chunk_size = 16 * 1024 * 1024 };

// ---------------------------------------------------------------------------- w2-12b (U1b)

const G2 = struct {
    pub fn forward(pk: zml.Tensor, sc: zml.Tensor) zml.Tensor {
        return w4.dequantW4(pk, sc);
    }
};

const OneT = struct { t: zml.Tensor };

const W2Mod = struct { name: []const u8, key: []const u8 };
// Les 3 familles de shapes du contrat U0 §3 (clés == script 63 --fixture-u1b) :
const W2_MODS = [_]W2Mod{
    .{ .name = "model.language_model.layers.0.self_attn.q_proj", .key = "deq_q0" }, // sliding, 16x256
    .{ .name = "model.language_model.layers.5.self_attn.q_proj", .key = "deq_q5" }, // full, 16x512
    .{ .name = "model.language_model.layers.0.mlp.down_proj", .key = "deq_dn0" }, // down, in large
};

fn gateW2_12b(allocator: std.mem.Allocator, arena: std.mem.Allocator, io: std.Io, platform: *zml.Platform, sharding: zml.sharding.Sharding, ckpt_path: []const u8, fixture_path: []const u8) !void {
    log.info("w2-12b (U1b) — dequant des 3 shapes 12B vs fixture u_mats12 (bit-exact u16)", .{});

    // DEUX stores : checkpoint packé 12B + fixture (pattern gemma4_w4gate.gateW2).
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

        // UN compileFn par module : les 3 shapes diffèrent.
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
            log.err("w2-12b/{s} : longueurs D2H inattendues ({d} != {d})", .{ m.key, got.len, exp.len });
            return error.GateFailed;
        }
        var ok = true;
        for (got, exp, 0..) |g, e, i| {
            if (g != e) {
                log.err("w2-12b/{s} : divergence bit-exact à l'index {d} : got=0x{x:0>4} expected=0x{x:0>4}", .{ m.key, i, g, e });
                ok = false;
                break;
            }
        }
        if (!ok) return error.GateFailed;
        n_pass += 1;
        log.info("    {s} : {d} u16 bit-exact", .{ m.key, got.len });
    }

    if (n_pass != W2_MODS.len) return error.GateFailed;
    log.info("PASS w2-12b {d}/{d}", .{ n_pass, W2_MODS.len });
}

// ---------------------------------------------------------------------------- main

pub fn main(init: std.process.Init) !void {
    const arena = init.arena;
    const allocator = init.gpa;
    const io = init.io;

    const process_args = try init.minimal.args.toSlice(arena.allocator());
    if (process_args.len < 2) {
        log.err("{s}", .{usage});
        return error.MissingArgument;
    }
    const mode = process_args[1];

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);
    const sharding = try zml.sharding.replicatedSharding(platform);
    log.info("G12 gates — backend = {s} (CPU nominal), geom 12B : {d} couches, d={d}", .{ @tagName(platform.target), g12.g12.num_layers, g12.g12.d });

    if (std.mem.eql(u8, mode, "w2-12b")) {
        if (process_args.len < 4) {
            log.err("{s}", .{usage});
            return error.MissingArgument;
        }
        try gateW2_12b(allocator, arena.allocator(), io, platform, sharding, process_args[2], process_args[3]);
    } else {
        log.err("mode inconnu '{s}' — {s}", .{ mode, usage });
        return error.MissingArgument;
    }
}
