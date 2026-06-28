// L2 — Inférence AUTONOME host-orchestrée (chunkée) : la boucle gather→forward→argmax→reinject en ZML.
//
// Contrairement à L1a (replay : les embeds/embptls viennent de la fixture, pré-calculés par HF), L2
// GATHER lui-même les embeddings du token produit : à chaque step, le host fait argmax(logits) → tok,
// lit la ligne `tok` de embed_tokens (+ embed_tokens_per_layer) en HOST, fabrique les buffers device
// (fromBytes) et les reinjecte. cos/sin/masques/positions restent de la fixture L1a (position-only :
// indépendants du token → valides pour la gen autonome tant que le compte de positions coïncide).
//
// Chemin chunké (forwardStageStep, cf engine.zig) : borne le pic de compilation (~33 Go sinon, thrash).
// Critère L2 : la séquence GÉNÉRÉE (autonome) == HF greedy (expected) — cf DESIGN §5.5.
//
// Coût HOST : embed_tokens + embed_tokens_per_layer lus en host (~0,8 + ~4,7 Go). Le device copie de
// embptl est libérée après lecture (la table n'est pas un poids du forward). Nécessite probablement le
// swapfile (cf conventions PLAN). Préférer un run court (3e arg max_steps, ex 64) pour valider l'autonomie.
//
// CLI : gemma4_gchunk_auto <model.safetensors> <gen_long.safetensors> [max_steps]

const std = @import("std");
const log = std.log;
const zml = @import("zml");
const engine = @import("engine.zig");
const mem_probe = @import("mem_probe.zig");

pub const std_options: std.Options = .{ .log_level = .info };

const L_MAX: i64 = 1024;
const NUM_LAYERS: usize = 35;
const CHUNK: usize = 5;
const SYNC_EVERY: usize = 1;
const B: i64 = 1;
const S: i64 = 1;
const D: i64 = 1536;
const LF: i64 = 8960; // 35 * 256 (hidden_size_per_layer_input × num_layers)

// L2 : config L1a (ring=false, cache linéaire L_MAX, masque bande). L'autonomie porte sur les embeds.
const Model = engine.EngineModel(struct {}, .{ .two_masks = true, .kmax_sliding = L_MAX, .kmax_full = L_MAX });
const PackedLong = engine.Packed(true);
const StageOut = struct { zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor };

const Stage = struct { start: usize, end: usize, first: bool, last: bool };
const N_STAGES: usize = (NUM_LAYERS + CHUNK - 1) / CHUNK;
const STAGES: [N_STAGES]Stage = blk: {
    var s: [N_STAGES]Stage = undefined;
    var i: usize = 0;
    var start: usize = 0;
    while (start < NUM_LAYERS) : (start += CHUNK) {
        const end = @min(start + CHUNK, NUM_LAYERS);
        s[i] = .{ .start = start, .end = end, .first = (start == 0), .last = (end == NUM_LAYERS) };
        i += 1;
    }
    break :blk s;
};

// Tables d'embeddings lues en host pour le gather.
const EmbPtl = struct {
    w: zml.Tensor, // {voc, lf}
    pub fn init(v: zml.io.TensorStore.View) EmbPtl {
        return .{ .w = v.createTensor("embed_tokens_per_layer.weight", .{ .voc, .lf }, null) };
    }
    pub fn load(self: *const EmbPtl, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, store: *const zml.io.TensorStore, shardings: []const zml.sharding.Sharding) !zml.Bufferized(EmbPtl) {
        return zml.io.load(EmbPtl, self, allocator, io, platform, store, .{ .shardings = shardings, .parallelism = 1, .dma_chunks = 1, .dma_chunk_size = 16 * 1024 * 1024 });
    }
};

const SeqW = struct {
    e: zml.Tensor, // expected {step}
    fed0: zml.Tensor, // fed {step} — on n'utilise que fed[0] (prefill argmax s0, déterministe)
    pub fn init(v: zml.io.TensorStore.View) SeqW {
        return .{ .e = v.createTensor("expected", .{.step}, null), .fed0 = v.createTensor("fed", .{.step}, null) };
    }
    pub fn load(self: *const SeqW, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, store: *const zml.io.TensorStore, shardings: []const zml.sharding.Sharding) !zml.Bufferized(SeqW) {
        return zml.io.load(SeqW, self, allocator, io, platform, store, .{ .shardings = shardings, .parallelism = 1, .dma_chunks = 1, .dma_chunk_size = 16 * 1024 * 1024 });
    }
};

fn argmaxOf(allocator: std.mem.Allocator, io: std.Io, logits_buf: *zml.Buffer) !i64 {
    var s = try logits_buf.toSliceAlloc(allocator, io);
    defer s.free(allocator);
    const v = s.items(f32);
    var best: usize = 0;
    var best_val: f32 = v[0];
    for (v, 0..) |x, idx| {
        if (x > best_val) {
            best_val = x;
            best = idx;
        }
    }
    return @intCast(best);
}

fn dtypeSize(dt: zml.DataType) usize {
    return @intCast(dt.sizeOf());
}

pub fn main(init: std.process.Init) !void {
    @setEvalBranchQuota(200000);
    const arena = init.arena;
    const allocator = init.gpa;
    const io = init.io;

    const process_args = try init.minimal.args.toSlice(arena.allocator());
    if (process_args.len < 3) {
        log.err("Usage: gemma4_gchunk_auto <model.safetensors> <gen_long.safetensors> [max_steps]", .{});
        return error.MissingArgument;
    }
    const ckpt = process_args[1];
    const fixture = process_args[2];
    const max_steps: ?usize = if (process_args.len >= 4) std.fmt.parseInt(usize, process_args[3], 10) catch null else null;
    log.info("L2 — génération AUTONOME host-orchestrée (chunkée) ; gather embeds/embptls host-side", .{});

    var reg_ck: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, ckpt);
    var store_ck: zml.io.TensorStore = .fromRegistry(allocator, &reg_ck);
    var reg_fx: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, fixture);
    defer reg_fx.deinit();
    var store_fx: zml.io.TensorStore = .fromRegistry(allocator, &reg_fx);
    defer store_fx.deinit();

    const base = store_ck.view().withPrefix("model").withPrefix("language_model");
    const model: Model = try .init(arena.allocator(), base);
    const packed_in: PackedLong = .init(store_fx.view());
    const cache0: engine.Cache = .init(store_fx.view());
    const ctrl_sym: engine.Ctrl = .initSymbolic();
    const hidden_sym = zml.Tensor.init(.{ B, S, D }, .f32).withTags(.{ .b, .s, .d });

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);
    const sharding = try zml.sharding.replicatedSharding(platform);

    // Poids moteur (eng_buf.embed_tokens = lm_head = table d'embeddings, déjà sur device).
    const eng_buf = try model.load(arena.allocator(), io, platform, &store_ck, &.{sharding});
    // Table embed_tokens_per_layer : chargée device puis lue host (puis libérée device — pas un poids du forward).
    const embptl_sym: EmbPtl = .init(base);
    var embptl_buf = try embptl_sym.load(arena.allocator(), io, platform, &store_ck, &.{sharding});
    // cos/sin/masques/positions + cache0 + expected/fed depuis la fixture.
    const pk_buf = try packed_in.load(arena.allocator(), io, platform, &store_fx, &.{sharding});
    var cache_buf = try cache0.load(arena.allocator(), io, platform, &store_fx, &.{sharding});
    const seq_sym: SeqW = .init(store_fx.view());
    var seq_buf = try seq_sym.load(arena.allocator(), io, platform, &store_fx, &.{sharding});

    // === Lecture HOST des deux tables d'embeddings (pour le gather) ===
    // embed_tokens : reuse eng_buf.embed_tokens (device, lm_head) → host copy.
    var emb_dev = eng_buf.embed_tokens;
    var emb_host = try emb_dev.toSliceAlloc(allocator, io);
    const emb_dtype = emb_host.dtype();
    const emb_esz = dtypeSize(emb_dtype);
    const emb_bytes = emb_host.constData(); // {voc, d} row-major
    // embed_tokens_per_layer : host copy puis libère le device (économise ~4,7 Go device).
    var eptl_dev = embptl_buf.w;
    var eptl_host = try eptl_dev.toSliceAlloc(allocator, io);
    const eptl_esz = dtypeSize(eptl_host.dtype());
    const eptl_bytes = eptl_host.constData(); // {voc, lf}
    eptl_dev.deinit(); // table libérée du device (host copy conservée) ; embptl_buf.w désormais invalide (non réutilisé)
    store_ck.deinit();
    reg_ck.deinit();
    mem_probe.logMem("post-load (tables d'embeddings en host)");

    // expected + fed0 (seed) en host.
    var exp_slice = try seq_buf.e.toSliceAlloc(allocator, io);
    defer exp_slice.free(allocator);
    const expected_tokens = exp_slice.items(i32);
    var fed0_slice = try seq_buf.fed0.toSliceAlloc(allocator, io);
    defer fed0_slice.free(allocator);
    const fed_tokens = fed0_slice.items(i32);
    const total = expected_tokens.len;
    const num_steps = if (max_steps) |m| @min(m, total) else total;

    // Symboles per-step pour forwardStageStep (dtype = native du checkpoint, agronomique pour le forward).
    const embeds_sym = zml.Tensor.init(.{ B, S, D }, emb_dtype).withTags(.{ .b, .s, .d });
    const embptls_sym = zml.Tensor.init(.{ B, S, LF }, eptl_host.dtype()).withTags(.{ .b, .s, .lf });

    const hidden_shape = zml.Shape.init(.{ B, S, D }, .f32).withTags(.{ .b, .s, .d });
    const zeros = try arena.allocator().alloc(u8, @intCast(B * S * D * 4));
    @memset(zeros, 0);
    var dummy_hidden = try zml.Buffer.fromBytes(io, platform, hidden_shape, sharding, zeros);
    defer dummy_hidden.deinit();

    // Scratch hosts pour les lignes gather (1 ligne à la fois : pas de copie massive).
    const emb_row_bytes: usize = @intCast(D * @as(i64, @intCast(emb_esz)));
    const eptl_row_bytes: usize = @intCast(LF * @as(i64, @intCast(eptl_esz)));
    const emb_scratch = try allocator.alloc(u8, emb_row_bytes);
    defer allocator.free(emb_scratch);
    const eptl_scratch = try allocator.alloc(u8, eptl_row_bytes);
    defer allocator.free(eptl_scratch);
    const emb_step_shape = zml.Shape.init(.{ B, S, D }, emb_dtype).withTags(.{ .b, .s, .d });
    const eptl_step_shape = zml.Shape.init(.{ B, S, LF }, eptl_host.dtype()).withTags(.{ .b, .s, .lf });

    // ===== Compile les N stages (forwardStageStep) =====
    log.info("Compiling {d} stages (forwardStageStep, autonome)...", .{N_STAGES});
    var exes: [N_STAGES]zml.Exe = undefined;
    inline for (STAGES, 0..) |stage, si| {
        const F = struct {
            fn f(m: Model, es: zml.Tensor, ep: zml.Tensor, p: PackedLong, ca: engine.Cache, h: zml.Tensor, ct: engine.Ctrl) StageOut {
                return m.forwardStageStep(stage.start, stage.end, stage.first, stage.last, es, ep, p, ca, h, ct);
            }
        }.f;
        exes[si] = try platform.compileFn(allocator, io, F, .{ model, embeds_sym, embptls_sym, packed_in, cache0, hidden_sym, ctrl_sym }, .{ .shardings = &.{sharding} });
    }
    defer for (&exes) |*e| e.deinit();
    mem_probe.logMem("post-compile (go/no-go)");

    // ===== Boucle autonome : gather(host) → fromBytes(device) → forwardStageStep → argmax → reinject =====
    var all_pass = true;
    var n_match: usize = 0;
    var first_fail: i64 = -1;
    var fed_tok: i64 = fed_tokens[0]; // seed = prefill argmax s0 (déterministe, lu de la fixture `fed`)
    log.info("L2 seed fed[0]={d} (prefill argmax) ; {d} steps", .{ fed_tok, num_steps });

    var step_idx: usize = 0;
    while (step_idx < num_steps) : (step_idx += 1) {
        // 1) gather HOST des embeddings du token à feed (fed_tok).
        const emb_off: usize = @intCast(fed_tok * @as(i64, @intCast(emb_row_bytes)));
        const eptl_off: usize = @intCast(fed_tok * @as(i64, @intCast(eptl_row_bytes)));
        @memcpy(emb_scratch, emb_bytes[emb_off .. emb_off + emb_row_bytes]);
        @memcpy(eptl_scratch, eptl_bytes[eptl_off .. eptl_off + eptl_row_bytes]);

        // 2) host → device (fromBytes) pour les entrées per-step.
        var embeds_step_buf = try zml.Buffer.fromBytes(io, platform, emb_step_shape, sharding, emb_scratch);
        var embptls_step_buf = try zml.Buffer.fromBytes(io, platform, eptl_step_shape, sharding, eptl_scratch);

        var step_buf = try zml.Buffer.scalar(io, platform, @as(u32, @intCast(step_idx)), .u32, sharding);
        const ctrl_buf = zml.Bufferized(engine.Ctrl){ .step = step_buf };

        // 3) exécute les stages en séquence (cache + hidden threadés device→device).
        var hidden_buf = dummy_hidden; // first stage : ignoré (hidden = embeds_step)
        var tok: i64 = -1;
        inline for (STAGES, 0..) |stage, si| {
            var args = try exes[si].args(allocator);
            var results = try exes[si].results(allocator);
            args.set(.{ eng_buf, embeds_step_buf, embptls_step_buf, pk_buf, cache_buf, hidden_buf, ctrl_buf });
            exes[si].call(args, &results);
            var out0, const nsl, const nsv, const nfl, const nfv = results.get(struct {
                zml.Buffer, zml.Buffer, zml.Buffer, zml.Buffer, zml.Buffer,
            });

            const do_sync = (si % SYNC_EVERY == SYNC_EVERY - 1);
            if (do_sync) {
                var s = try out0.toSliceAlloc(allocator, io);
                s.free(allocator);
            }
            var old_cache = cache_buf;
            cache_buf = zml.Bufferized(engine.Cache){ .sl_k = nsl, .sl_v = nsv, .fl_k = nfl, .fl_v = nfv };
            old_cache.sl_k.deinit();
            old_cache.sl_v.deinit();
            old_cache.fl_k.deinit();
            old_cache.fl_v.deinit();
            if (si != 0) hidden_buf.deinit();
            if (stage.last) {
                tok = try argmaxOf(allocator, io, &out0);
                out0.deinit();
            } else {
                hidden_buf = out0;
            }
            args.deinit(allocator);
            results.deinit(allocator);
        }

        // 4) le token produit devient le prochain feed (autonomie) + validation.
        const exp = @as(i64, @intCast(expected_tokens[step_idx]));
        const ok = tok == exp;
        if (ok) n_match += 1 else {
            all_pass = false;
            if (first_fail < 0) first_fail = @intCast(step_idx);
            if (first_fail >= 0 and (step_idx - @as(usize, @intCast(first_fail)) < 8)) {
                log.info("  DIVERGENCE step {d} (pos {d}) : généré={d} HF={d}", .{ step_idx, 4 + step_idx, tok, exp });
            }
        }
        if ((step_idx + 1) % 64 == 0) log.info("  ... {d}/{d} steps, {d} match", .{ step_idx + 1, num_steps, n_match });

        fed_tok = tok; // reinject (autonomie)
        embeds_step_buf.deinit();
        embptls_step_buf.deinit();
        step_buf.deinit();
    }
    cache_buf.sl_k.deinit();
    cache_buf.sl_v.deinit();
    cache_buf.fl_k.deinit();
    cache_buf.fl_v.deinit();
    emb_host.free(allocator);
    eptl_host.free(allocator);
    mem_probe.logMem("post-run");

    log.info("L2 AUTONOME : {d}/{d} tokens match (généré vs HF greedy, first_fail step {d})", .{ n_match, num_steps, first_fail });
    if (all_pass) {
        log.info("L2 PASS — génération AUTONOME == HF greedy sur {d} tokens (host gather + reinject)", .{num_steps});
    } else {
        log.warn("L2 : {d} divergences vs HF (1re au step {d}). L'autonomie reproduit HF jusqu'à cette position.", .{ num_steps - n_match, first_fail });
        // Pas d'error fatal : une divergence = info (l'autonomie peut dériver après une 1re flip, cumulativement).
    }
}
