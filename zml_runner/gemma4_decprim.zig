// P5.7.7 — PRIMITIVES decode isolées (scatterSlices + zml.nn.rope(pos)).
//
// Ferme la dette d'isolation notée par la revue adversariale de decode-1 : les 2 primitives ZML neuves
// avaient été introduites ensemble. On les teste séparément, AVANT decode-3 (35 couches, dur à debugger).
//  1) scatterSlices : cache {1,1,5,4} connu, append/override à pos=4 ET pos=2 vs oracle (copie numpy).
//     -> ciblage dynamique de la colonne + override + passthrough des axes b/h/hd.
//  2) zml.nn.rope(pos) : auto-cohérence rope(x,arange)[4] == rope(x[4:5], pos=[4]) (prouve que pos est
//     utilisé ; ≡ HF via P5.2.C.3 qui a déjà validé rope-arange vs apply_rotary_pos_emb).
//
// CLI : gemma4_decprim <p5_7_7_decode_prim.safetensors>

const std = @import("std");
const log = std.log;
const zml = @import("zml");

pub const std_options: std.Options = .{ .log_level = .info };

const ROPE_THETA: f32 = 1.0e4;

fn slidingRopeArange(x: zml.Tensor) zml.Tensor {
    return zml.nn.rope(x, null, .{ .layout = .sequential, .scaling = .{ .default = .{ .rope_theta = ROPE_THETA } } });
}
fn slidingRopePos(x: zml.Tensor, pos: zml.Tensor) zml.Tensor {
    return zml.nn.rope(x, pos, .{ .layout = .sequential, .scaling = .{ .default = .{ .rope_theta = ROPE_THETA } } });
}

const Out4 = struct { zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor };

// Module = entrées de la fixture + forward(self) sans arg runtime (idiome Gate B).
const Engine = struct {
    scat_cache: zml.Tensor, // {b,h,k=5,hd=4}
    scat_update: zml.Tensor, // {b,h,k=1,hd=4}
    scat_pos4: zml.Tensor, // {s=1} i32 = [4]
    scat_pos2: zml.Tensor, // {s=1} i32 = [2]
    rope_x: zml.Tensor, // {b,s=5,nh=1,hd=8}
    rope_pos4: zml.Tensor, // {s=1} i32 = [4]

    pub fn init(v: zml.io.TensorStore.View) Engine {
        return .{
            .scat_cache = v.createTensor("scat_cache", .{ .b, .h, .k, .hd }, null),
            .scat_update = v.createTensor("scat_update", .{ .b, .h, .k, .hd }, null),
            .scat_pos4 = v.createTensor("scat_pos4", .{.s}, null),
            .scat_pos2 = v.createTensor("scat_pos2", .{.s}, null),
            .rope_x = v.createTensor("rope_x", .{ .b, .s, .nh, .hd }, null),
            .rope_pos4 = v.createTensor("rope_pos4", .{.s}, null),
        };
    }

    pub fn load(self: *const Engine, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, store: *const zml.io.TensorStore, shardings: []const zml.sharding.Sharding) !zml.Bufferized(Engine) {
        return zml.io.load(Engine, self, allocator, io, platform, store, .{ .shardings = shardings, .parallelism = 1, .dma_chunks = 1, .dma_chunk_size = 16 * 1024 * 1024 });
    }

    pub fn forward(self: Engine) Out4 {
        const so = zml.Tensor.ScatterOpts{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override };
        // 1) scatterSlices à pos=4 et pos=2 (ciblage dynamique via index tensoriel)
        const out4 = self.scat_cache.scatterSlices(.{ .k = self.scat_pos4.squeeze(.s).convert(.u32) }, self.scat_update, so);
        const out2 = self.scat_cache.scatterSlices(.{ .k = self.scat_pos2.squeeze(.s).convert(.u32) }, self.scat_update, so);

        // 2) rope(pos) : auto-cohérence. A = rope(x, arange) ; A4 = position 4 de A.
        const a_full = slidingRopeArange(self.rope_x); // {b,s=5,nh,hd}
        const a4 = a_full.choose1d(.s, 4).reshape(.{ 1, 1, 1, 8 }).withTags(.{ .b, .s, .nh, .hd });
        // B = rope(x[4:5], pos=[4]).
        const x4 = self.rope_x.choose1d(.s, 4).reshape(.{ 1, 1, 1, 8 }).withTags(.{ .b, .s, .nh, .hd });
        const b = slidingRopePos(x4, self.rope_pos4);

        return .{ out4, out2, a4, b };
    }
};

const Oracle = struct {
    exp_pos4: zml.Tensor,
    exp_pos2: zml.Tensor,

    pub fn init(v: zml.io.TensorStore.View) Oracle {
        return .{
            .exp_pos4 = v.createTensor("scat_exp_pos4", .{ .b, .h, .k, .hd }, null),
            .exp_pos2 = v.createTensor("scat_exp_pos2", .{ .b, .h, .k, .hd }, null),
        };
    }

    pub fn load(self: *const Oracle, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, store: *const zml.io.TensorStore, shardings: []const zml.sharding.Sharding) !zml.Bufferized(Oracle) {
        return zml.io.load(Oracle, self, allocator, io, platform, store, .{ .shardings = shardings, .parallelism = 1, .dma_chunks = 1, .dma_chunk_size = 16 * 1024 * 1024 });
    }
};

fn cmp(allocator: std.mem.Allocator, io: std.Io, label: []const u8, out_buf: *zml.Buffer, ref_buf: *zml.Buffer) !bool {
    var out_s = try out_buf.toSliceAlloc(allocator, io);
    defer out_s.free(allocator);
    var ref_s = try ref_buf.toSliceAlloc(allocator, io);
    defer ref_s.free(allocator);
    const out = out_s.items(f32);
    const ref = ref_s.items(f32);
    if (out.len != ref.len) {
        log.err("  {s}: length mismatch out={d} ref={d}", .{ label, out.len, ref.len });
        return error.LengthMismatch;
    }
    var max_abs: f32 = 0.0;
    for (out, ref) |a, bb| {
        const diff = @abs(a - bb);
        if (diff > max_abs) max_abs = diff;
    }
    // Ops structurelles (copie / même calcul) -> on EXIGE l'égalité exacte (0.0).
    const pass = max_abs == 0.0;
    log.info("  {s:<28} n={d:>3} max_abs={e:.4} -> {s}", .{ label, out.len, max_abs, if (pass) "PASS" else "FAIL" });
    return pass;
}

pub fn main(init: std.process.Init) !void {
    const arena = init.arena;
    const allocator = init.gpa;
    const io = init.io;

    const process_args = try init.minimal.args.toSlice(arena.allocator());
    if (process_args.len < 2) {
        log.err("Usage: gemma4_decprim <p5_7_7_decode_prim.safetensors>", .{});
        return error.MissingArgument;
    }
    const fixture = process_args[1];
    log.info("P5.7.7 primitives decode isolées (scatterSlices + rope pos)", .{});

    var reg_fx: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, fixture);
    defer reg_fx.deinit();
    var store_fx: zml.io.TensorStore = .fromRegistry(allocator, &reg_fx);
    defer store_fx.deinit();

    const engine: Engine = .init(store_fx.view());
    const oracle: Oracle = .init(store_fx.view());

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);
    const sharding = try zml.sharding.replicatedSharding(platform);

    const eng_buf = try engine.load(arena.allocator(), io, platform, &store_fx, &.{sharding});
    var orc_buf = try oracle.load(arena.allocator(), io, platform, &store_fx, &.{sharding});

    log.info("Compiling...", .{});
    var exe = try platform.compile(allocator, io, engine, .forward, .{}, .{ .shardings = &.{sharding} });
    defer exe.deinit();
    var args = try exe.args(allocator);
    defer args.deinit(allocator);
    var results = try exe.results(allocator);
    defer results.deinit(allocator);
    args.set(.{eng_buf});
    exe.call(args, &results);

    var r0, var r1, var r2, var r3 = results.get(struct { zml.Buffer, zml.Buffer, zml.Buffer, zml.Buffer });
    defer {
        r0.deinit();
        r1.deinit();
        r2.deinit();
        r3.deinit();
    }

    log.info("Comparaison (ops structurelles -> égalité exacte exigée) :", .{});
    var all_pass = true;
    all_pass = (try cmp(allocator, io, "scatter pos=4 (override)", &r0, &orc_buf.exp_pos4)) and all_pass;
    all_pass = (try cmp(allocator, io, "scatter pos=2 (dynamique)", &r1, &orc_buf.exp_pos2)) and all_pass;
    all_pass = (try cmp(allocator, io, "rope(pos=4)==rope(arange)[4]", &r2, &r3)) and all_pass;

    if (all_pass) {
        log.info("P5.7.7 decode-prim PASS — scatterSlices (ciblage+override+passthrough) + rope(pos) isolés validés", .{});
    } else {
        log.err("P5.7.7 decode-prim : divergence", .{});
        return error.PrimMismatch;
    }
}
