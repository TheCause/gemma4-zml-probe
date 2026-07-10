// G2.3 — Sweep de sensibilité par famille d'ops (cf docs/G2_3_OP_SENSITIVITY.md + spec PrecRt,
// engine.zig §PrecRt).
//
// UN SEUL binaire pour TOUTES les configs du sweep : la précision n'est plus comptime (EngineCfg)
// mais RUNTIME (`PrecRt`, champ `model.prec`) — le traçage émet les converts d'après la valeur au
// moment du compile, donc chaque run recompile son graphe mais le binaire est unique (pas de
// N cibles Bazel, pas d'explosion du quota comptime pjrt.zig — nom 16c < plafond ~20c).
//
// Les familles à passer en bf16 arrivent par argv : "fam1,fam2,..." (noms des champs de PrecRt,
// ex "qkv_proj,mlp"), ou "none" = baseline D0 (tout f32, graphe NEUTRE == G1). ⚠ fromSpecList ne
// trim PAS les espaces : "mlp, ple" → error.UnknownFamily (l'orchestrateur génère sans espaces).
// Compute reste f32 dans tous les cas (spec §4 : « bf16 » = arrondi aux bornes de l'op).
//
// Famille kv_store (mécanisme b) : le dtype de STOCKAGE vient du header de la fixture →
// Cache.checkDtype vérifie AVANT compile la cohérence fixture↔prec (fixture standard f32 pour
// kv_store=null, variante scripts/46 --kv-dtype bf16 pour kv_store=bf16).
//
// SORTIE : les LOGITS f32 de chaque step sont dumpés en binaire brut [N_steps × VOC] (leçon
// non-vacuité : le verdict G2.3 se prend sur les logits vs l'enveloppe, PAS sur l'argmax).
// L'argmax n'est reporté qu'à titre indicatif (bifurcations ATTENDUES en bf16, cf G2.0 : HF
// lui-même fait 1016/1020) → pas d'error.GenMismatch, le runner sort 0 si l'exécution aboutit
// (même politique que le runner G2.2).
//
// --no-prealloc (mécanisme G2.1, cf gemma4_gen_long_gpu.zig) : coupe la préallocation BFC pour
// que nvidia-smi mesure la VRAM RÉELLEMENT utilisée (gate G2.3.2 kv_store) — avec preallocate=true
// (défaut, perf), BFC réserve memory_fraction×24 Go d'emblée et masque l'usage réel.
//
// CLI : gemma4_g23_sweep <model.safetensors> <fixture.safetensors> <logits_out.bin> <familles> [max_steps] [--no-prealloc]

const std = @import("std");
const log = std.log;
const zml = @import("zml");
const engine = @import("engine.zig");
const mem_probe = @import("mem_probe.zig");

pub const std_options: std.Options = .{ .log_level = .info };

const L_MAX: i64 = 1024;
// G2.3 : plus de précision dans EngineCfg — PrecRt est runtime (model.prec, posé après init).
const Model = engine.EngineModel(struct {}, .{
    .two_masks = true,
    .kmax_sliding = L_MAX,
    .kmax_full = L_MAX,
});
const PackedLong = engine.Packed(true);

const ExpW = struct {
    e: zml.Tensor,
    pub fn init(v: zml.io.TensorStore.View) ExpW {
        return .{ .e = v.createTensor("expected", .{.step}, null) };
    }
    pub fn load(self: *const ExpW, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, store: *const zml.io.TensorStore, shardings: []const zml.sharding.Sharding) !zml.Bufferized(ExpW) {
        return zml.io.load(ExpW, self, allocator, io, platform, store, .{ .shardings = shardings, .parallelism = 1, .dma_chunks = 1, .dma_chunk_size = 16 * 1024 * 1024 });
    }
};

pub fn main(init: std.process.Init) !void {
    const arena = init.arena;
    const allocator = init.gpa;
    const io = init.io;

    const process_args = try init.minimal.args.toSlice(arena.allocator());
    if (process_args.len < 5) {
        log.err("Usage: gemma4_g23_sweep <model.safetensors> <fixture.safetensors> <logits_out.bin> <familles> [max_steps] [--no-prealloc]", .{});
        return error.MissingArgument;
    }
    const ckpt = process_args[1];
    const fixture = process_args[2];
    const logits_out = process_args[3];
    const fam_arg = process_args[4];
    const max_steps: ?usize = if (process_args.len >= 6) std.fmt.parseInt(usize, process_args[5], 10) catch null else null;
    var no_prealloc = false;
    for (process_args[5..]) |a| {
        if (std.mem.eql(u8, a, "--no-prealloc")) no_prealloc = true;
    }

    // "none" = baseline D0 (PrecRt tout-null == graphe neutre f32) ; sinon parse la liste de familles.
    const prec: engine.PrecRt = if (std.mem.eql(u8, fam_arg, "none")) .{} else try engine.PrecRt.fromSpecList(fam_arg);
    // En-tête consommé par le manifest/orchestrateur — NE PAS changer le format.
    log.info("G2.3 config: familles=[{s}] compute=f32", .{fam_arg});

    const platform: *zml.Platform = blk: {
        const cuda_opts: zml.platform.CreateOptions = .{
            .cuda = .{ .allocator = .{ .bfc = .{ .preallocate = !no_prealloc, .memory_fraction = 0.90 } } },
        };
        if (zml.Platform.init(allocator, io, .cuda, cuda_opts)) |p| break :blk p else |_| {}
        log.warn("CUDA indisponible — repli sur Platform.auto (probablement CPU).", .{});
        break :blk try zml.Platform.auto(allocator, io, .{});
    };
    defer platform.deinit(allocator);
    log.info("G2.3 — backend = {s} ; PrecRt runtime (converts émis au traçage).", .{@tagName(platform.target)});
    if (no_prealloc) log.info("G2.3 — preallocate=false : nvidia-smi mesure l'usage VRAM réel (pas la réserve BFC).", .{});
    const sharding = try zml.sharding.replicatedSharding(platform);

    var reg_ck: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, ckpt);
    var store_ck: zml.io.TensorStore = .fromRegistry(allocator, &reg_ck);
    var reg_fx: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, fixture);
    defer reg_fx.deinit();
    var store_fx: zml.io.TensorStore = .fromRegistry(allocator, &reg_fx);
    defer store_fx.deinit();

    const base = store_ck.view().withPrefix("model").withPrefix("language_model");
    var model: Model = try .init(arena.allocator(), base);
    // prec AVANT compile : le traçage lit self.prec (valeur au moment du compile = graphe émis).
    model.prec = prec;
    const packed_in: PackedLong = .init(store_fx.view());
    const cache0: engine.Cache = .init(store_fx.view());
    // Garde G2.3 : cohérence dtype cache (header fixture) ↔ prec.kv_store, AVANT load/compile.
    try cache0.checkDtype(model.prec);
    const ctrl_sym: engine.Ctrl = .initSymbolic();

    log.info("Materializing weights + packed inputs + caches (L_MAX={d}) ...", .{L_MAX});
    const eng_buf = try model.load(arena.allocator(), io, platform, &store_ck, &.{sharding});
    const pk_buf = try packed_in.load(arena.allocator(), io, platform, &store_fx, &.{sharding});
    var cache_buf = try cache0.load(arena.allocator(), io, platform, &store_fx, &.{sharding});
    const exp_sym: ExpW = .init(store_fx.view());
    var exp_buf = try exp_sym.load(arena.allocator(), io, platform, &store_fx, &.{sharding});
    store_ck.deinit();
    reg_ck.deinit();
    mem_probe.logMem(io, "post-load (host RSS ; VRAM via nvidia-smi)");

    var exp_slice = try exp_buf.e.toSliceAlloc(allocator, io);
    defer exp_slice.free(allocator);
    const expected_tokens = exp_slice.items(i32);
    const total = expected_tokens.len;
    const num_steps = if (max_steps) |m| @min(m, total) else total;
    log.info("NUM_STEPS = {d} (max_steps={?d})", .{ num_steps, max_steps });

    log.info("Compiling gen step (mono-graphe 35 couches, familles=[{s}]) ...", .{fam_arg});
    const t_compile: std.Io.Timestamp = .now(io, .awake);
    var exe = try platform.compile(allocator, io, model, .forward, .{ packed_in, cache0, ctrl_sym }, .{ .shardings = &.{sharding} });
    defer exe.deinit();
    log.info("  compile: {f}", .{t_compile.untilNow(io, .awake)});

    const out_file = try std.Io.Dir.createFile(.cwd(), io, logits_out, .{});
    defer out_file.close(io);
    var write_offset: u64 = 0;

    var n_match: usize = 0;
    var first_div: i64 = -1;
    var step_idx: usize = 0;
    const t0: std.Io.Timestamp = .now(io, .awake);
    while (step_idx < num_steps) : (step_idx += 1) {
        var step_buf = try zml.Buffer.scalar(io, platform, @as(u32, @intCast(step_idx)), .u32, sharding);
        const ctrl_buf = zml.Bufferized(engine.Ctrl){ .step = step_buf };

        var args = try exe.args(allocator);
        var results = try exe.results(allocator);
        args.set(.{ eng_buf, pk_buf, cache_buf, ctrl_buf });
        exe.call(args, &results);
        var r_logits, const r_slk, const r_slv, const r_flk, const r_flv = results.get(struct {
            zml.Buffer, zml.Buffer, zml.Buffer, zml.Buffer, zml.Buffer,
        });

        // dump logits f32 (le VRAI critère G2.3) + argmax indicatif
        var s = try r_logits.toSliceAlloc(allocator, io);
        const v = s.items(f32);
        const bytes = std.mem.sliceAsBytes(v);
        try out_file.writePositionalAll(io, bytes, write_offset);
        write_offset += bytes.len;
        var best: usize = 0;
        var best_val: f32 = v[0];
        for (v, 0..) |x, idx| {
            if (x > best_val) {
                best_val = x;
                best = idx;
            }
        }
        s.free(allocator);

        const exp = @as(i64, @intCast(expected_tokens[step_idx]));
        if (@as(i64, @intCast(best)) == exp) n_match += 1 else if (first_div < 0) first_div = @intCast(step_idx);
        if ((step_idx + 1) % 256 == 0) log.info("  ... {d}/{d} steps, {d} match", .{ step_idx + 1, num_steps, n_match });

        cache_buf.sl_k.deinit();
        cache_buf.sl_v.deinit();
        cache_buf.fl_k.deinit();
        cache_buf.fl_v.deinit();
        cache_buf = zml.Bufferized(engine.Cache){ .sl_k = r_slk, .sl_v = r_slv, .fl_k = r_flk, .fl_v = r_flv };
        r_logits.deinit();
        step_buf.deinit();
        args.deinit(allocator);
        results.deinit(allocator);
    }
    const elapsed = t0.untilNow(io, .awake);
    cache_buf.sl_k.deinit();
    cache_buf.sl_v.deinit();
    cache_buf.fl_k.deinit();
    cache_buf.fl_v.deinit();

    const elapsed_ns = elapsed.toNanoseconds();
    const elapsed_s = @as(f64, @floatFromInt(elapsed_ns)) / std.time.ns_per_s;
    const tok_per_s = if (elapsed_s > 0) @as(f64, @floatFromInt(num_steps)) / elapsed_s else 0;
    log.info("G2.3 PERF : {d} tokens en {d:.2}s → {d:.1} tok/s [backend={s}, familles=[{s}], batch-1] (NB : inclut le dump logits 1 Mo/step)", .{ num_steps, elapsed_s, tok_per_s, @tagName(platform.target), fam_arg });
    mem_probe.logMem(io, "post-run (host RSS ; VRAM via nvidia-smi)");

    log.info("G2.3 argmax (INDICATIF) : {d}/{d} match vs HF-fp32 ; 1re divergence = {d}", .{ n_match, num_steps, first_div });
    log.info("Logits dumpés : {s} ({d} steps × VOC f32). Verdict → analyse offline vs enveloppe.", .{ logits_out, num_steps });
}
