// G12 — gates U1b→U7 du 12B Unified (plan docs/superpowers/plans/2026-07-24-w4-j2-12b-unified.md,
// contrat docs/U_12B_CONTRACT.md). CPU nominal (Platform.auto), pas de garde VRAM.
// Nom 14c (quota comptime @typeName pjrt, piège 4). Pattern : gemma4_w4gate.zig.
//
//   w2-12b <weights_12b/model.safetensors> <fixtures/u_mats12.safetensors>
//       — U1b : dequantW4 (brique J1, réutilisée telle quelle) sur les 3 familles de shapes 12B
//         (q_proj L0 sliding 4096×3840, q_proj L5 full 8192×3840, down_proj L0 3840×15360),
//         poids lus du checkpoint PACKÉ, comparés bit-exact bf16 (u16) à la fixture du script 63
//         (--fixture-u1b).
//   u2 <weights_12b/model.safetensors> <fixtures/u_embed.safetensors>
//       — U2 : gather embed_tokens (bf16 NON quantifié du PACKÉ, D9) + scale CHEMIN 12B (D12 :
//         ×62.0 constante bf16 == bf16(√3840), produit bf16×bf16 — reproduit l'ordre de casts
//         EXACT de Gemma4TextScaledWordEmbedding.forward ; jamais dans engine.zig), comparé
//         BIT-EXACT u16 (max_abs == 0) à la fixture du script 64.
//   u3 <weights_12b/model.safetensors> <fixtures/u_sliding.safetensors>
//       — U3 : étages q/k/v de l'attention sliding L0 (dequantW4 du PACKÉ via W4Lin.toW,
//         q_norm/k_norm, v_norm SANS poids, rope sliding zml.nn.rope theta 1e4 == engine
//         slidingRope), comparés f32 aux hooks du module réel (script 65). GATING au périmètre
//         PRÉ-ENREGISTRÉ du plan (§ Task 4, fixture « S=8, étages a/b/c/d ») : S=8 aux seuils
//         §3 (max_abs <= 1e-4, mean_abs <= 1e-6). Les étages S=1040 sont EN PLUS mesurés en
//         tripwire diagnostique (borne 1e-3) : le run du 24 juil a montré que zml.nn.rope
//         diverge du HF réel de 1 ULP f32 sur inv_freq (zml exp(-log θ·n/N) vs HF θ^(-n/N)),
//         amplifié LINÉAIREMENT par la position (Δangle = pos·Δinv ≈ 6.1e-5 rad à pos 708 →
//         max_abs 4.9e-4 sur l'étage a, diagnostic au chiffre près) — pas un bug de câblage ;
//         la borne 1e-3 attrape un vrai bug (GQA/layout => O(0.1)) tout en tolérant la phase
//         ULP (~5e-4 prédit). L'effet s'annule en rotation RELATIVE dans les scores QK : U4
//         (S=1040 mordant) tient 2.8e-5 << 1e-4.
//   u4 <weights_12b/model.safetensors> <fixtures/u_sliding.safetensors>
//       — U4 : attention sliding complète L0 (scores GQA groupe 2 par splitAxis — le groupe
//         dérive de la shape de K, pattern engine.zig runLayerGen ; masque ADDITIF de la
//         fixture = mécanique HF réelle ; softmax f32 ; context ; o_proj) vs la sortie du
//         module réel : max_abs <= 1e-4. Non-vacuité : le comptage masqué du masque S=1040
//         est RECOMPTÉ in-gate et doit mordre (== causal pur + 136, fenêtre 1024 < S).
//         Modes u5/… : Tasks 5+.
//
// ⚠ IMPÉRATIF modes futurs (u3-u7) : tout mode lisant le PACKÉ passe par openStores /
// registryFromFile — JAMAIS TensorRegistry.fromPath (realPath traverse les symlinks HF vers
// blobs/<sha256> sans extension .safetensors -> error.InvalidPath, bug mordu au 1er run w2-12b).
//
// Verdicts par erreur Zig : error.GateFailed / error.VacuousGate ; PASS -> log + exit 0.

const std = @import("std");
const log = std.log;
const zml = @import("zml");
const w4 = @import("w4.zig");
const g12 = @import("g12.zig");

pub const std_options: std.Options = .{ .log_level = .info };

const usage = "Usage: gemma4_g12gate <w2-12b|u2|u3|u4> <model.safetensors> <fixture.safetensors>";

const load_opts = .{ .parallelism = 1, .dma_chunks = 1, .dma_chunk_size = 16 * 1024 * 1024 };

/// `TensorRegistry.fromPath` refuse le checkpoint PACKÉ : `weights_12b/model.safetensors` est un
/// symlink vers le cache HF dont le `file.realPath` (resolveFiletype, safetensors.zig:543) se
/// résout en `blobs/<sha256>` SANS extension `.safetensors` -> `.unknown` -> error.InvalidPath
/// (bug mordu au premier run w2-12b). Contournement LOCAL au gate (zml/ intouché) : ouvrir le
/// fichier et appeler `parseSafetensors` directement — licite car le contrat U0 garantit le
/// packé MONO-fichier (jamais un index). La lecture des données passe par l'URI realpath du
/// blob, qui est un fichier régulier lisible.
fn registryFromFile(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !zml.safetensors.TensorRegistry {
    const file = try std.Io.Dir.cwd().openFile(io, path, .{ .mode = .read_only });
    var registry: zml.safetensors.TensorRegistry = .init(allocator);
    errdefer registry.deinit();
    try zml.safetensors.parseSafetensors(allocator, io, &registry, file);
    return registry;
}

/// Les DEUX stores de tout gate : checkpoint PACKÉ + fixture. Passage OBLIGATOIRE (voir ⚠ en
/// tête) : le packé passe STRUCTURELLEMENT par registryFromFile, la fixture (fichier régulier
/// .safetensors, jamais un symlink HF) par fromPath.
const Stores = struct {
    reg_ck: zml.safetensors.TensorRegistry,
    store_ck: zml.io.TensorStore,
    reg_fx: zml.safetensors.TensorRegistry,
    store_fx: zml.io.TensorStore,

    fn deinit(self: *Stores) void {
        self.store_fx.deinit();
        self.reg_fx.deinit();
        self.store_ck.deinit();
        self.reg_ck.deinit();
    }
};

/// Construit IN PLACE dans `out` : TensorStore garde un *pointeur* vers sa registry (io.zig:34)
/// — un retour par valeur déplacerait les registries et pendrait ces pointeurs.
fn openStores(out: *Stores, allocator: std.mem.Allocator, io: std.Io, ckpt_path: []const u8, fixture_path: []const u8) !void {
    out.reg_ck = try registryFromFile(allocator, io, ckpt_path);
    errdefer out.reg_ck.deinit();
    out.store_ck = .fromRegistry(allocator, &out.reg_ck);
    errdefer out.store_ck.deinit();
    out.reg_fx = try .fromPath(allocator, io, fixture_path);
    errdefer out.reg_fx.deinit();
    out.store_fx = .fromRegistry(allocator, &out.reg_fx);
}

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
    var st: Stores = undefined;
    try openStores(&st, allocator, io, ckpt_path, fixture_path);
    defer st.deinit();
    const store_ck = &st.store_ck;
    const store_fx = &st.store_fx;

    var n_pass: usize = 0;
    inline for (W2_MODS) |m| {
        log.info("  module {s} -> {s}", .{ m.name, m.key });
        const lin: w4.W4Lin = .init(store_ck.view(), m.name);
        const lin_buf = try zml.io.load(w4.W4Lin, &lin, arena, io, platform, store_ck, .{ .shardings = &.{sharding}, .parallelism = load_opts.parallelism, .dma_chunks = load_opts.dma_chunks, .dma_chunk_size = load_opts.dma_chunk_size });
        const exp_sym: OneT = .{ .t = store_fx.view().createTensor(m.key, .{ .o, .d }, null) };
        const exp_buf = try zml.io.load(OneT, &exp_sym, arena, io, platform, store_fx, .{ .shardings = &.{sharding}, .parallelism = load_opts.parallelism, .dma_chunks = load_opts.dma_chunks, .dma_chunk_size = load_opts.dma_chunk_size });

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

// ---------------------------------------------------------------------------- u2 (embed + scale)

// D12 — chemin 12B du scale d'embedding, reproduit l'ordre de casts EXACT du module HF réel
// `Gemma4TextScaledWordEmbedding.forward` : `super().forward(ids) * self.embed_scale.to(bf16)`
// = gather bf16 PUIS multiplication par la constante bf16(√3840) = 62.0 exactement (0x4278),
// produit bf16×bf16. `Tensor.scale(62.0)` construit `.scalar(62.0, self.dtype())` — dtype du
// tenseur gatheré = bf16, donc constante bf16 : PAS le chemin E2B (engine.zig:38 EMBED_SCALE
// f64 sur tenseur converti f32), qui reste intouché (neutralité prouvée par md5 HLO, gate U1).
const U2 = struct {
    pub fn forward(emb: zml.Tensor, ids: zml.Tensor) zml.Tensor {
        return emb.gather(.{ .voc = ids }, .{}).scale(62.0);
    }
};

const U2Fix = struct { ids: zml.Tensor, expected: zml.Tensor };

const EMB_KEY = "model.language_model.embed_tokens.weight"; // bf16 NON quantifié dans le PACKÉ (D9)

fn gateU2(allocator: std.mem.Allocator, arena: std.mem.Allocator, io: std.Io, platform: *zml.Platform, sharding: zml.sharding.Sharding, ckpt_path: []const u8, fixture_path: []const u8) !void {
    log.info("u2 (U2) — gather embed_tokens + scale bf16 62.0 (chemin 12B, D12) vs fixture u_embed (bit-exact u16)", .{});

    var st: Stores = undefined;
    try openStores(&st, allocator, io, ckpt_path, fixture_path);
    defer st.deinit();
    const store_ck = &st.store_ck;
    const store_fx = &st.store_fx;

    const emb_sym: OneT = .{ .t = store_ck.view().createTensor(EMB_KEY, .{ .voc, .d }, null) };
    if (emb_sym.t.dim(.voc) != 262144 or emb_sym.t.dim(.d) != g12.g12.d) {
        log.err("u2 : shape embed_tokens inattendue {d}x{d}", .{ emb_sym.t.dim(.voc), emb_sym.t.dim(.d) });
        return error.GateFailed;
    }
    const emb_buf = try zml.io.load(OneT, &emb_sym, arena, io, platform, store_ck, .{ .shardings = &.{sharding}, .parallelism = load_opts.parallelism, .dma_chunks = load_opts.dma_chunks, .dma_chunk_size = load_opts.dma_chunk_size });

    const fix_sym: U2Fix = .{
        .ids = store_fx.view().createTensor("ids", .{.s}, null),
        .expected = store_fx.view().createTensor("expected", .{ .s, .d }, null),
    };
    const n_ids = fix_sym.ids.dim(.s);
    if (n_ids == 0 or fix_sym.expected.dim(.s) != n_ids or fix_sym.expected.dim(.d) != g12.g12.d) {
        log.err("u2 : fixture incohérente (ids {d}, expected {d}x{d})", .{ n_ids, fix_sym.expected.dim(.s), fix_sym.expected.dim(.d) });
        return error.VacuousGate;
    }
    const fix_buf = try zml.io.load(U2Fix, &fix_sym, arena, io, platform, store_fx, .{ .shardings = &.{sharding}, .parallelism = load_opts.parallelism, .dma_chunks = load_opts.dma_chunks, .dma_chunk_size = load_opts.dma_chunk_size });

    var exe = try platform.compileFn(allocator, io, U2.forward, .{ emb_sym.t, fix_sym.ids }, .{ .shardings = &.{sharding} });
    defer exe.deinit();
    var args = try exe.args(allocator);
    defer args.deinit(allocator);
    var results = try exe.results(allocator);
    defer results.deinit(allocator);
    args.set(.{ emb_buf.t, fix_buf.ids });
    exe.call(args, &results);
    var r_got = results.get(zml.Buffer);
    defer r_got.deinit();

    var got_s = try r_got.toSliceAlloc(allocator, io);
    defer got_s.free(allocator);
    const got = got_s.items(u16);
    var exp_s = try fix_buf.expected.toSliceAlloc(allocator, io);
    defer exp_s.free(allocator);
    const exp = exp_s.items(u16);

    if (got.len != exp.len or got.len != @as(usize, @intCast(n_ids * g12.g12.d))) {
        log.err("u2 : longueurs D2H inattendues ({d} != {d}, attendu {d})", .{ got.len, exp.len, n_ids * g12.g12.d });
        return error.GateFailed;
    }
    var n_diff: usize = 0;
    for (got, exp, 0..) |g, e, i| {
        if (g != e) {
            if (n_diff == 0) log.err("u2 : divergence bit-exact à l'index {d} (id sonde {d}) : got=0x{x:0>4} expected=0x{x:0>4}", .{ i, i / @as(usize, @intCast(g12.g12.d)), g, e });
            n_diff += 1;
        }
    }
    if (n_diff != 0) {
        log.err("u2 : {d}/{d} u16 divergents — le critère reste BIT-EXACT (max_abs == 0), pas d'assouplissement", .{ n_diff, got.len });
        return error.GateFailed;
    }
    log.info("PASS u2 — {d} ids x {d} : {d} u16 bit-exact (max_abs == 0), scale bf16 62.0 chemin 12B", .{ n_ids, g12.g12.d, got.len });
}

// ---------------------------------------------------------------------------- u3/u4 (attention sliding L0)

// Constantes du contrat U0 §1 pour la couche 0 sliding (géométrie g12.g12) :
const ATTN_PREFIX = "model.language_model.layers.0.self_attn.";
const HD_S: i64 = @intCast(g12.g12.hd_sliding); // 256 (pattern engine.zig:102 — Geom porte des usize)
const RMS_EPS: f32 = 1.0e-6;
const SLIDING_WINDOW: i64 = 1024;
const U34_SEQ_LENS = [_]i64{ 8, 1040 }; // 1040 > fenêtre : cas mordant (témoin U4)
const U3_MAX_ABS: f64 = 1.0e-4; // §3 — U4 : max_abs seul, même seuil
const U3_MEAN_ABS: f64 = 1.0e-6;

/// Poids d'attention L0, lus du PACKÉ (D9 — dequantW4 in-graph, brique U1b bit-exacte).
const AttnW = struct {
    q: w4.W4Lin,
    k: w4.W4Lin,
    v: w4.W4Lin,
    o: w4.W4Lin,
    q_norm: zml.Tensor,
    k_norm: zml.Tensor,

    fn init(v_ck: zml.io.TensorStore.View) AttnW {
        return .{
            .q = .init(v_ck, ATTN_PREFIX ++ "q_proj"),
            .k = .init(v_ck, ATTN_PREFIX ++ "k_proj"),
            .v = .init(v_ck, ATTN_PREFIX ++ "v_proj"),
            .o = .init(v_ck, ATTN_PREFIX ++ "o_proj"),
            .q_norm = v_ck.createTensor(ATTN_PREFIX ++ "q_norm.weight", .{.hd}, null),
            .k_norm = v_ck.createTensor(ATTN_PREFIX ++ "k_norm.weight", .{.hd}, null),
        };
    }
};

// Ops miroir du chemin moteur (runLayerGen, prec fam=null == baseline f32) — mêmes primitives
// zml, mêmes ordres de casts : dot `a_f32.dot(b.convert(.f32))` (dotPrec fam=null), q/k_norm
// `rmsNorm(.hd) puis mul(poids convert f32)` (rmsScaleHdPrec), rope `zml.nn.rope sequential
// theta 1e4` (slidingRope ; pos=null => arange sur .s, positions 0..S-1 du prefill).
const U34Ops = struct {
    fn projNormRope(h: zml.Tensor, pk: zml.Tensor, sc: zml.Tensor, norm: zml.Tensor) zml.Tensor {
        const w = w4.dequantW4(pk, sc).withTags(.{ .o, .d }); // bf16 {o,d}, == export dq (U1b)
        const x = h.convert(.f32).dot(w.convert(.f32), .d); // {b,s,o} f32
        const xh = x.splitAxis(.o, .{ .nh = .auto, .hd = HD_S }); // {b,s,nh,hd}
        const normalized = zml.nn.rmsNorm(xh, .hd, RMS_EPS);
        const xn = normalized.mul(norm.convert(.f32).broad(xh.shape()));
        return zml.nn.rope(xn, null, .{ .layout = .sequential, .scaling = .{ .default = .{ .rope_theta = g12.g12.rope_theta_sliding } } });
    }

    /// Étage (c) : v_proj + v_norm SANS poids (RMSNorm with_scale=False — l'op existe, piège 9).
    fn vStage(h: zml.Tensor, pk: zml.Tensor, sc: zml.Tensor) zml.Tensor {
        const w = w4.dequantW4(pk, sc).withTags(.{ .o, .d });
        const x = h.convert(.f32).dot(w.convert(.f32), .d);
        const xh = x.splitAxis(.o, .{ .nh = .auto, .hd = HD_S });
        return zml.nn.rmsNorm(xh, .hd, RMS_EPS);
    }

    /// U4 : attention complète — GQA groupe 2 par splitAxis, le groupe DÉRIVE de la shape de K
    /// (k.dim(.h) = 8 -> hq = 16/8 = 2), pattern engine.zig runLayerGen (chemin .manual).
    fn attn(h: zml.Tensor, qpk: zml.Tensor, qsc: zml.Tensor, qn: zml.Tensor, kpk: zml.Tensor, ksc: zml.Tensor, kn: zml.Tensor, vpk: zml.Tensor, vsc: zml.Tensor, opk: zml.Tensor, osc: zml.Tensor, mask: zml.Tensor) zml.Tensor {
        const q = projNormRope(h, qpk, qsc, qn);
        const k = projNormRope(h, kpk, ksc, kn);
        const v = vStage(h, vpk, vsc);
        const q_final = q.transpose(.{ .b, .nh, .s, .hd }).rename(.{ .nh = .h, .s = .q });
        const k_new = k.transpose(.{ .b, .nh, .s, .hd }).rename(.{ .nh = .h, .s = .k });
        const v_new = v.transpose(.{ .b, .nh, .s, .hd }).rename(.{ .nh = .h, .s = .k });

        const qs = q_final.splitAxis(.h, .{ .h = k_new.dim(.h), .hq = .auto });
        var scores = qs.dot(k_new, .hd).merge(.{ .h = .{ .h, .hq } }).transpose(.{ .b, .h, .q, .k });
        scores = scores.add(mask.convert(.f32).broad(scores.shape())); // additif, scaling 1.0 (piège 10)
        const probs = scores.softmax(.k); // f32

        const ps = probs.splitAxis(.h, .{ .h = v_new.dim(.h), .hq = .auto });
        const ctx = ps.dot(v_new, .k).merge(.{ .h = .{ .h, .hq } }).transpose(.{ .b, .h, .q, .hd });
        const attn_m = ctx.transpose(.{ .b, .q, .h, .hd }).merge(.{ .m = .{ .h, .hd } });
        return attn_m.dot(opk_deq(opk, osc), .m).rename(.{ .q = .s }); // {b,s,d} f32
    }

    fn opk_deq(opk: zml.Tensor, osc: zml.Tensor) zml.Tensor {
        // o_proj [3840, 4096] = [.d out, .m in] (tags du site moteur, engine.LayerW.o_proj).
        return w4.dequantW4(opk, osc).withTags(.{ .d, .m }).convert(.f32);
    }
};

/// Comparaison f32 vs f32 aux seuils §3. Diagnostics : localisation + valeurs au pire écart
/// (piège 17 : on compare des VALEURS ; en cas d'égalité de |écart| le premier index est
/// rapporté, les suivants comptés dans n_over).
const StageVerdict = struct { max_abs: f64, mean_abs: f64 };

fn compareStage(allocator: std.mem.Allocator, io: std.Io, name: []const u8, got_buf: *zml.Buffer, exp_t: zml.Buffer, max_thr: f64, mean_thr: ?f64) !StageVerdict {
    var got_s = try got_buf.toSliceAlloc(allocator, io);
    defer got_s.free(allocator);
    const got = got_s.items(f32);
    var exp_s = try exp_t.toSliceAlloc(allocator, io);
    defer exp_s.free(allocator);
    const exp = exp_s.items(f32);
    if (got.len != exp.len) {
        log.err("{s} : longueurs D2H inattendues ({d} != {d})", .{ name, got.len, exp.len });
        return error.GateFailed;
    }
    var max_abs: f64 = 0;
    var sum_abs: f64 = 0;
    var max_i: usize = 0;
    var n_over: usize = 0;
    for (got, exp, 0..) |g, e, i| {
        const d = @abs(@as(f64, g) - @as(f64, e));
        sum_abs += d;
        if (d > max_abs) {
            max_abs = d;
            max_i = i;
        }
        if (d > max_thr) n_over += 1;
    }
    const mean_abs = sum_abs / @as(f64, @floatFromInt(got.len));
    const ok = max_abs <= max_thr and (mean_thr == null or mean_abs <= mean_thr.?);
    if (ok) {
        log.info("    {s} : max_abs={e:.3} mean_abs={e:.3} ({d} f32) OK", .{ name, max_abs, mean_abs, got.len });
    } else {
        log.err("    {s} : FAIL max_abs={e:.3} (seuil {e:.1}) mean_abs={e:.3} — pire écart à l'index {d} : got={e:.6} expected={e:.6} ; {d}/{d} au-dessus du seuil", .{ name, max_abs, max_thr, mean_abs, max_i, got[max_i], exp[max_i], n_over, got.len });
        return error.GateFailed;
    }
    return .{ .max_abs = max_abs, .mean_abs = mean_abs };
}

/// Fixture d'un cas S : clés du script 65 (layouts pré-transpose des hooks).
const U34Fix = struct {
    hidden: zml.Tensor, // {b,s,d} bf16
    q: zml.Tensor, // {b,s,nh,hd} f32 (nh=16)
    k: zml.Tensor, // {b,s,nh,hd} f32 (nh=8)
    v: zml.Tensor, // {b,s,nh,hd} f32 (nh=8)
    mask: zml.Tensor, // {b,h,q,k} f32 additive
    out: zml.Tensor, // {b,s,d} f32 (sortie o_proj du module réel)

    fn init(v_fx: zml.io.TensorStore.View, comptime sfx: []const u8) U34Fix {
        return .{
            .hidden = v_fx.createTensor("hidden" ++ sfx, .{ .b, .s, .d }, null),
            .q = v_fx.createTensor("q" ++ sfx, .{ .b, .s, .nh, .hd }, null),
            .k = v_fx.createTensor("k" ++ sfx, .{ .b, .s, .nh, .hd }, null),
            .v = v_fx.createTensor("v" ++ sfx, .{ .b, .s, .nh, .hd }, null),
            .mask = v_fx.createTensor("mask" ++ sfx, .{ .b, .h, .q, .k }, null),
            .out = v_fx.createTensor("out" ++ sfx, .{ .b, .s, .d }, null),
        };
    }

    fn check(self: U34Fix, s: i64) !void {
        const g = g12.g12;
        if (self.hidden.dim(.s) != s or self.hidden.dim(.d) != g.d or
            self.q.dim(.s) != s or self.q.dim(.nh) != g.nh or self.q.dim(.hd) != g.hd_sliding or
            self.k.dim(.s) != s or self.k.dim(.nh) != g.kvh_sliding or
            self.v.dim(.s) != s or self.v.dim(.nh) != g.kvh_sliding or
            self.mask.dim(.q) != s or self.mask.dim(.k) != s or
            self.out.dim(.s) != s or self.out.dim(.d) != g.d)
        {
            log.err("u3/u4 : fixture S={d} incohérente avec la géométrie g12", .{s});
            return error.VacuousGate;
        }
    }
};

fn gateU3(allocator: std.mem.Allocator, arena: std.mem.Allocator, io: std.Io, platform: *zml.Platform, sharding: zml.sharding.Sharding, ckpt_path: []const u8, fixture_path: []const u8) !void {
    log.info("u3 (U3) — étages q/k/v attention sliding L0 (PACKÉ dequantW4, rope theta 1e4) vs hooks module réel", .{});
    log.info("  gating §3 sur S=8 (périmètre pré-enregistré du plan) ; S=1040 = tripwire diagnostique borne 1e-3 (phase ULP inv_freq, cf en-tête)", .{});

    var st: Stores = undefined;
    try openStores(&st, allocator, io, ckpt_path, fixture_path);
    defer st.deinit();

    const aw: AttnW = .init(st.store_ck.view());
    const aw_buf = try zml.io.load(AttnW, &aw, arena, io, platform, &st.store_ck, .{ .shardings = &.{sharding}, .parallelism = load_opts.parallelism, .dma_chunks = load_opts.dma_chunks, .dma_chunk_size = load_opts.dma_chunk_size });

    var n_pass: usize = 0;
    inline for (U34_SEQ_LENS) |s| {
        const sfx = std.fmt.comptimePrint("_s{d}", .{s});
        // Seuils : S=8 = §3 (gating U3) ; S=1040 = borne tripwire 1e-3, mean libre (un bug de
        // câblage GQA/layout donnerait O(0.1) ; la phase ULP amplifiée pos<=1039 prédit ~5e-4).
        const gating = s <= 8;
        const max_thr: f64 = if (gating) U3_MAX_ABS else 1.0e-3;
        const mean_thr: ?f64 = if (gating) U3_MEAN_ABS else null;
        log.info("  cas S={d} ({s}) :", .{ s, if (gating) "gating §3" else "tripwire ULP 1e-3" });
        const fix: U34Fix = .init(st.store_fx.view(), sfx);
        try fix.check(s);
        const fix_buf = try zml.io.load(U34Fix, &fix, arena, io, platform, &st.store_fx, .{ .shardings = &.{sharding}, .parallelism = load_opts.parallelism, .dma_chunks = load_opts.dma_chunks, .dma_chunk_size = load_opts.dma_chunk_size });

        // Étage (a) Q : proj + q_norm + rope.
        {
            var exe = try platform.compileFn(allocator, io, U34Ops.projNormRope, .{ fix.hidden, aw.q.pk, aw.q.sc, aw.q_norm }, .{ .shardings = &.{sharding} });
            defer exe.deinit();
            var args = try exe.args(allocator);
            defer args.deinit(allocator);
            var results = try exe.results(allocator);
            defer results.deinit(allocator);
            args.set(.{ fix_buf.hidden, aw_buf.q.pk, aw_buf.q.sc, aw_buf.q_norm });
            exe.call(args, &results);
            var r = results.get(zml.Buffer);
            defer r.deinit();
            _ = try compareStage(allocator, io, "étage a (q post proj+norm+rope)", &r, fix_buf.q, max_thr, mean_thr);
        }
        // Étage (b) K : proj + k_norm + rope.
        {
            var exe = try platform.compileFn(allocator, io, U34Ops.projNormRope, .{ fix.hidden, aw.k.pk, aw.k.sc, aw.k_norm }, .{ .shardings = &.{sharding} });
            defer exe.deinit();
            var args = try exe.args(allocator);
            defer args.deinit(allocator);
            var results = try exe.results(allocator);
            defer results.deinit(allocator);
            args.set(.{ fix_buf.hidden, aw_buf.k.pk, aw_buf.k.sc, aw_buf.k_norm });
            exe.call(args, &results);
            var r = results.get(zml.Buffer);
            defer r.deinit();
            _ = try compareStage(allocator, io, "étage b (k post proj+norm+rope)", &r, fix_buf.k, max_thr, mean_thr);
        }
        // Étage (c) V : proj + v_norm SANS poids.
        {
            var exe = try platform.compileFn(allocator, io, U34Ops.vStage, .{ fix.hidden, aw.v.pk, aw.v.sc }, .{ .shardings = &.{sharding} });
            defer exe.deinit();
            var args = try exe.args(allocator);
            defer args.deinit(allocator);
            var results = try exe.results(allocator);
            defer results.deinit(allocator);
            args.set(.{ fix_buf.hidden, aw_buf.v.pk, aw_buf.v.sc });
            exe.call(args, &results);
            var r = results.get(zml.Buffer);
            defer r.deinit();
            _ = try compareStage(allocator, io, "étage c (v post v_norm sans poids)", &r, fix_buf.v, max_thr, mean_thr);
        }
        n_pass += 1;
    }
    if (n_pass != U34_SEQ_LENS.len) return error.GateFailed;
    log.info("PASS u3 — 3 étages S=8 aux seuils §3 (max_abs<=1e-4, mean_abs<=1e-6) + tripwire S=1040 sous borne ULP 1e-3", .{});
}

/// Non-vacuité U4 (leçon « vacuité de l'antécédent ») : le masque S>fenêtre de la fixture est
/// RECOMPTÉ ici et doit mordre — masqués == causal pur + Σ_{q>=1024}(q-1023), strictement > causal.
fn checkMaskBite(allocator: std.mem.Allocator, io: std.Io, mask_buf: zml.Buffer, s: i64) !void {
    var m_s = try mask_buf.toSliceAlloc(allocator, io);
    defer m_s.free(allocator);
    const m = m_s.items(f32);
    var n_masked: usize = 0;
    for (m) |x| {
        if (x < -1.0e30) n_masked += 1;
    }
    const su: usize = @intCast(s);
    const causal: usize = su * (su - 1) / 2;
    var bite: usize = 0;
    var qi: usize = @intCast(SLIDING_WINDOW);
    while (qi < su) : (qi += 1) bite += qi - @as(usize, @intCast(SLIDING_WINDOW - 1));
    if (n_masked != causal + bite) {
        log.err("u4 : masque S={d} — {d} masqués != causal {d} + morsure {d} (mécanique HF non reproduite ?)", .{ s, n_masked, causal, bite });
        return error.VacuousGate;
    }
    if (s > SLIDING_WINDOW) {
        if (bite == 0 or n_masked <= causal) {
            log.err("u4 : masque S={d} ne mord PAS (masqués {d} <= causal {d}) — gate vacant", .{ s, n_masked, causal });
            return error.VacuousGate;
        }
        log.info("    masque S={d} MORD : {d} masqués = causal {d} + {d} (fenêtre {d}) — témoin U4", .{ s, n_masked, causal, bite, SLIDING_WINDOW });
    } else {
        log.info("    masque S={d} : {d} masqués == causal pur {d} (fenêtre non mordante, attendu)", .{ s, n_masked, causal });
    }
}

fn gateU4(allocator: std.mem.Allocator, arena: std.mem.Allocator, io: std.Io, platform: *zml.Platform, sharding: zml.sharding.Sharding, ckpt_path: []const u8, fixture_path: []const u8) !void {
    log.info("u4 (U4) — attention sliding complète L0 (GQA groupe 2, masque HF, softmax f32, o_proj) vs module réel, S={{8,1040}}", .{});

    var st: Stores = undefined;
    try openStores(&st, allocator, io, ckpt_path, fixture_path);
    defer st.deinit();

    const aw: AttnW = .init(st.store_ck.view());
    const aw_buf = try zml.io.load(AttnW, &aw, arena, io, platform, &st.store_ck, .{ .shardings = &.{sharding}, .parallelism = load_opts.parallelism, .dma_chunks = load_opts.dma_chunks, .dma_chunk_size = load_opts.dma_chunk_size });

    var n_pass: usize = 0;
    inline for (U34_SEQ_LENS) |s| {
        const sfx = std.fmt.comptimePrint("_s{d}", .{s});
        log.info("  cas S={d} :", .{s});
        const fix: U34Fix = .init(st.store_fx.view(), sfx);
        try fix.check(s);
        const fix_buf = try zml.io.load(U34Fix, &fix, arena, io, platform, &st.store_fx, .{ .shardings = &.{sharding}, .parallelism = load_opts.parallelism, .dma_chunks = load_opts.dma_chunks, .dma_chunk_size = load_opts.dma_chunk_size });

        try checkMaskBite(allocator, io, fix_buf.mask, s);

        var exe = try platform.compileFn(allocator, io, U34Ops.attn, .{ fix.hidden, aw.q.pk, aw.q.sc, aw.q_norm, aw.k.pk, aw.k.sc, aw.k_norm, aw.v.pk, aw.v.sc, aw.o.pk, aw.o.sc, fix.mask }, .{ .shardings = &.{sharding} });
        defer exe.deinit();
        var args = try exe.args(allocator);
        defer args.deinit(allocator);
        var results = try exe.results(allocator);
        defer results.deinit(allocator);
        args.set(.{ fix_buf.hidden, aw_buf.q.pk, aw_buf.q.sc, aw_buf.q_norm, aw_buf.k.pk, aw_buf.k.sc, aw_buf.k_norm, aw_buf.v.pk, aw_buf.v.sc, aw_buf.o.pk, aw_buf.o.sc, fix_buf.mask });
        exe.call(args, &results);
        var r = results.get(zml.Buffer);
        defer r.deinit();
        _ = try compareStage(allocator, io, "étage d (attention complète -> o_proj)", &r, fix_buf.out, U3_MAX_ABS, null);
        n_pass += 1;
    }
    if (n_pass != U34_SEQ_LENS.len) return error.GateFailed;
    log.info("PASS u4 — attention complète S=8 + S=1040 (fenêtre MORDANTE recomptée) max_abs<=1e-4", .{});
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
    } else if (std.mem.eql(u8, mode, "u2")) {
        if (process_args.len < 4) {
            log.err("{s}", .{usage});
            return error.MissingArgument;
        }
        try gateU2(allocator, arena.allocator(), io, platform, sharding, process_args[2], process_args[3]);
    } else if (std.mem.eql(u8, mode, "u3")) {
        if (process_args.len < 4) {
            log.err("{s}", .{usage});
            return error.MissingArgument;
        }
        try gateU3(allocator, arena.allocator(), io, platform, sharding, process_args[2], process_args[3]);
    } else if (std.mem.eql(u8, mode, "u4")) {
        if (process_args.len < 4) {
            log.err("{s}", .{usage});
            return error.MissingArgument;
        }
        try gateU4(allocator, arena.allocator(), io, platform, sharding, process_args[2], process_args[3]);
    } else {
        log.err("mode inconnu '{s}' — {s}", .{ mode, usage });
        return error.MissingArgument;
    }
}
