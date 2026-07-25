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
//         slidingRope), comparés f32 aux hooks du module réel (script 65). Périmètre U3
//         (gating S=8, tripwire 1e-3 à S=1040 sans mean_abs) : RATIFIÉ par Régis le
//         25 juil 2026 (Amendement 2 du plan, commit 49707cd). Le
//         chemin rope à S=1040 est gaté au seuil plein par U4. Gating S=8 aux seuils §3
//         (max_abs <= 1e-4, mean_abs <= 1e-6). Les étages S=1040 sont EN PLUS mesurés en
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
//   u5 <weights_12b/model.safetensors> <fixtures/u_full.safetensors>
//       — U5 : couche 5 FULL (MQA 16Q×512 / 1KV×512 broadcast groupe 16, K=V, p-RoPE
//         proportional 512/0.25/1e6) vs hooks du module réel (script 66). Branche K=V du
//         moteur (D4) exercée par mini-graphe : V = v_norm(k_proj BRUT dequantW4 du PACKÉ,
//         avant k_norm/rope) — AUCUN poids v_proj (la couche n'en a pas au checkpoint, et
//         aucun placeholder n'est déclaré ici, a fortiori pas consommé). p-RoPE MANUELLE
//         (pattern engine.manualRope / runner w4auto ropeFull) : cos/sin HOST 512-wide,
//         384 composantes identité (partial 0.25). Périmètre Amendement 2 (U5) : gating
//         S=8 étages a/b/v (max_abs <= 1e-4, mean_abs <= 1e-6) ; sanity pos 0 == identité
//         STRICTE des cos/sin host ; tripwire positions {708, 1030} borne 1e-3 sans
//         mean_abs (cos/sin host vs HF de la fixture + étages ropés a/b restreints à ces
//         positions — même famille ULP que U3 : pow f32 host vs HF, Δangle ∝ position) ;
//         étage (c) attention complète S=8 ET S=1040 au seuil plein 1e-4 (l'output gate).
//         La discriminabilité K=V (>= 10x seuil, sinon exit 1) est CÂBLÉE dans l'oracle 66.
//   u6 <weights_12b/model.safetensors> <fixtures/u_layers.safetensors>
//       — U6 : couche décodeur COMPLÈTE (sandwich norms + attention + MLP + layer_scalar) puis
//         chaîne L0→L5 (5 sliding + 1 full) vs modules réels (script 67) — EN PASSANT PAR LE
//         MOTEUR (arbitrage contrôleur 23618bb), voir l'en-tête de la section u6 ci-dessous.
//         Seuil §3 U6 : max_abs <= 2e-4 par couche ET chaîne — resserré depuis 1e-3 au vu de
//         l'oracle (décision contrôleur 25 juil, Amendement 2 point 5) ; JAMAIS élargissable.
//   u7 <weights_12b/model.safetensors> <fixtures/u_prefill.safetensors>
//       — U7 : le forward 12B COMPLET (prefill prompt canonique + logits + softcap) —
//         généralisation du pattern u6 à Geom.g12 PLEIN (48 couches, stage-major CHUNKÉ
//         8x6 — repli prescrit du plan après 2 murs : 31e compileFn (K=1) et OOM (graphe
//         complet), cf U7Chunk), embed
//         chemin 12B du mode u2 (gather + scale bf16 62.0, D12), head du MOTEUR
//         (forwardStageGen last=true : final_norm + lm_head tied + softcapPrec tanh 30).
//         Comparaison §3-U7 sur les logits de la DERNIÈRE position : softcap exercé
//         (max|logits| <= 30 ET > 25 quelque part), top-5 en ENSEMBLE + tie rule
//         (écart de rang toléré ssi |Δlogit| < 1e-4, piège 17), max_abs documentaire
//         sous garde-fou 0.5 (Amendement 2 §U7, requalifié décision Régis 25 juil).
//         DEUX pièges de la revue U6 corrigés dans la généralisation (voir section u7).
//
// ⚠ IMPÉRATIF modes futurs (u3-u7) : tout mode lisant le PACKÉ passe par openStores /
// registryFromFile — JAMAIS TensorRegistry.fromPath (realPath traverse les symlinks HF vers
// blobs/<sha256> sans extension .safetensors -> error.InvalidPath, bug mordu au 1er run w2-12b).
//
// Verdicts par erreur Zig : error.GateFailed / error.VacuousGate ; PASS -> log + exit 0.

const std = @import("std");
const log = std.log;
const zml = @import("zml");
const engine = @import("engine.zig"); // moteur (mode u6 : forwardStageGen/Geom tronqué)
const w4 = @import("w4.zig");
const g12 = @import("g12.zig");

pub const std_options: std.Options = .{ .log_level = .info };

const usage = "Usage: gemma4_g12gate <w2-12b|u2|u3|u4|u5|u6|u7> <model.safetensors> <fixture.safetensors>";

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

/// Régime EXPLICITE par S (revue Task 4 : pas de comparaison numérique implicite `s <= 8`) :
/// gating=true => seuils §3 pleins ; gating=false => tripwire diagnostique borne 1e-3 sans
/// mean_abs (phase ULP inv_freq amplifiée par la position, cf en-tête u3).
const U3Case = struct { s: i64, gating: bool };
const U3_CASES = [_]U3Case{ .{ .s = 8, .gating = true }, .{ .s = 1040, .gating = false } };

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

/// Options de comparaison f32 (piège 17 : on compare des VALEURS ; en cas d'égalité de
/// |écart| le premier index est rapporté, les suivants comptés dans n_over).
/// `rows` != null : comparaison RESTREINTE à ces positions .s (tripwire U5 {708, 1030}) —
/// `row_elems` = éléments par position (layout s-majeur sur le reste des axes) ; mean_abs
/// sans objet dans ce régime (périmètre Amendement 2 : « informatif, sans mean_abs »).
const CmpOpts = struct {
    max_thr: f64,
    mean_thr: ?f64 = null,
    rows: ?[]const i64 = null,
    row_elems: usize = 0,
};

fn compareBuf(allocator: std.mem.Allocator, io: std.Io, name: []const u8, got_buf: *zml.Buffer, exp_t: zml.Buffer, opts: CmpOpts) !void {
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
    if (opts.rows) |rows| {
        // Régime tripwire : max_abs par position, borne opts.max_thr, valeurs consignées au log.
        for (rows) |row| {
            const base = @as(usize, @intCast(row)) * opts.row_elems;
            if (base + opts.row_elems > got.len) {
                log.err("{s} : position tripwire {d} hors du tenseur ({d} elems/row, len {d})", .{ name, row, opts.row_elems, got.len });
                return error.GateFailed;
            }
            var max_abs: f64 = 0;
            var max_i: usize = 0;
            for (got[base .. base + opts.row_elems], exp[base .. base + opts.row_elems], 0..) |g, e, i| {
                const d = @abs(@as(f64, g) - @as(f64, e));
                if (d > max_abs) {
                    max_abs = d;
                    max_i = i;
                }
            }
            if (max_abs > opts.max_thr) {
                log.err("    {s} @pos {d} : TRIPWIRE max_abs={e:.3} > borne {e:.1} — pire écart à l'offset {d} : got={e:.6} expected={e:.6}", .{ name, row, max_abs, opts.max_thr, max_i, got[base + max_i], exp[base + max_i] });
                return error.GateFailed;
            }
            log.info("    {s} @pos {d} : max_abs={e:.3} (borne {e:.1}, {d} f32) OK", .{ name, row, max_abs, opts.max_thr, opts.row_elems });
        }
        return;
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
        if (d > opts.max_thr) n_over += 1;
    }
    const mean_abs = sum_abs / @as(f64, @floatFromInt(got.len));
    const ok = max_abs <= opts.max_thr and (opts.mean_thr == null or mean_abs <= opts.mean_thr.?);
    if (ok) {
        log.info("    {s} : max_abs={e:.3} mean_abs={e:.3} ({d} f32) OK", .{ name, max_abs, mean_abs, got.len });
    } else {
        log.err("    {s} : FAIL max_abs={e:.3} (seuil {e:.1}) mean_abs={e:.3} — pire écart à l'index {d} : got={e:.6} expected={e:.6} ; {d}/{d} au-dessus du seuil", .{ name, max_abs, opts.max_thr, mean_abs, max_i, got[max_i], exp[max_i], n_over, got.len });
        return error.GateFailed;
    }
}

/// Boilerplate compile/exec/compare factorisé (dette signalée en revue Task 4, migrée sur
/// u3/u4/u5) : compile `f` sur les symboliques `syms`, exécute sur les buffers `bufs`,
/// compare le résultat f32 à `exp` selon `opts`. Un compileFn par appel (les shapes varient
/// par cas/étage — CPU, pas de cache d'exécutable nécessaire).
fn runCompare(allocator: std.mem.Allocator, io: std.Io, platform: *zml.Platform, sharding: zml.sharding.Sharding, comptime f: anytype, syms: anytype, bufs: anytype, name: []const u8, exp: zml.Buffer, opts: CmpOpts) !void {
    var exe = try platform.compileFn(allocator, io, f, syms, .{ .shardings = &.{sharding} });
    defer exe.deinit();
    var args = try exe.args(allocator);
    defer args.deinit(allocator);
    var results = try exe.results(allocator);
    defer results.deinit(allocator);
    args.set(bufs);
    exe.call(args, &results);
    var r = results.get(zml.Buffer);
    defer r.deinit();
    try compareBuf(allocator, io, name, &r, exp, opts);
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
    log.info("  gating §3 sur S=8 + tripwire 1e-3 à S=1040 (périmètre RATIFIÉ, Amendement 2) — phase ULP inv_freq, cf en-tête", .{});

    var st: Stores = undefined;
    try openStores(&st, allocator, io, ckpt_path, fixture_path);
    defer st.deinit();

    const aw: AttnW = .init(st.store_ck.view());
    const aw_buf = try zml.io.load(AttnW, &aw, arena, io, platform, &st.store_ck, .{ .shardings = &.{sharding}, .parallelism = load_opts.parallelism, .dma_chunks = load_opts.dma_chunks, .dma_chunk_size = load_opts.dma_chunk_size });

    var n_pass: usize = 0;
    inline for (U3_CASES) |case| {
        const s = case.s;
        const sfx = std.fmt.comptimePrint("_s{d}", .{s});
        // Seuils : régime EXPLICITE de la table U3_CASES — gating = §3 ; sinon tripwire 1e-3,
        // mean_abs NON gaté (abandon DÉCLARÉ, Amendement 2 — un bug de câblage GQA/layout
        // donnerait O(0.1) ; la phase ULP amplifiée pos<=1039 prédit ~5e-4).
        const opts: CmpOpts = if (case.gating)
            .{ .max_thr = U3_MAX_ABS, .mean_thr = U3_MEAN_ABS }
        else
            .{ .max_thr = 1.0e-3 };
        log.info("  cas S={d} ({s}) :", .{ s, if (case.gating) "gating §3" else "tripwire ULP 1e-3" });
        const fix: U34Fix = .init(st.store_fx.view(), sfx);
        try fix.check(s);
        const fix_buf = try zml.io.load(U34Fix, &fix, arena, io, platform, &st.store_fx, .{ .shardings = &.{sharding}, .parallelism = load_opts.parallelism, .dma_chunks = load_opts.dma_chunks, .dma_chunk_size = load_opts.dma_chunk_size });

        // Étage (a) Q : proj + q_norm + rope.
        try runCompare(allocator, io, platform, sharding, U34Ops.projNormRope, .{ fix.hidden, aw.q.pk, aw.q.sc, aw.q_norm }, .{ fix_buf.hidden, aw_buf.q.pk, aw_buf.q.sc, aw_buf.q_norm }, "étage a (q post proj+norm+rope)", fix_buf.q, opts);
        // Étage (b) K : proj + k_norm + rope.
        try runCompare(allocator, io, platform, sharding, U34Ops.projNormRope, .{ fix.hidden, aw.k.pk, aw.k.sc, aw.k_norm }, .{ fix_buf.hidden, aw_buf.k.pk, aw_buf.k.sc, aw_buf.k_norm }, "étage b (k post proj+norm+rope)", fix_buf.k, opts);
        // Étage (c) V : proj + v_norm SANS poids.
        try runCompare(allocator, io, platform, sharding, U34Ops.vStage, .{ fix.hidden, aw.v.pk, aw.v.sc }, .{ fix_buf.hidden, aw_buf.v.pk, aw_buf.v.sc }, "étage c (v post v_norm sans poids)", fix_buf.v, opts);
        n_pass += 1;
    }
    if (n_pass != U3_CASES.len) return error.GateFailed;
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

        try runCompare(allocator, io, platform, sharding, U34Ops.attn, .{ fix.hidden, aw.q.pk, aw.q.sc, aw.q_norm, aw.k.pk, aw.k.sc, aw.k_norm, aw.v.pk, aw.v.sc, aw.o.pk, aw.o.sc, fix.mask }, .{ fix_buf.hidden, aw_buf.q.pk, aw_buf.q.sc, aw_buf.q_norm, aw_buf.k.pk, aw_buf.k.sc, aw_buf.k_norm, aw_buf.v.pk, aw_buf.v.sc, aw_buf.o.pk, aw_buf.o.sc, fix_buf.mask }, "étage d (attention complète -> o_proj)", fix_buf.out, .{ .max_thr = U3_MAX_ABS });
        n_pass += 1;
    }
    if (n_pass != U34_SEQ_LENS.len) return error.GateFailed;
    log.info("PASS u4 — attention complète S=8 + S=1040 (fenêtre MORDANTE recomptée) max_abs<=1e-4", .{});
}

// ---------------------------------------------------------------------------- u5 (attention full L5, MQA, K=V, p-RoPE)

// Constantes du contrat U0 §1 pour la couche 5 full (géométrie g12.g12) :
const FULL_PREFIX = "model.language_model.layers.5.self_attn.";
const HD_F: i64 = @intCast(g12.g12.hd_full); // 512
const HALF_F: i64 = @divExact(HD_F, 2); // 256 (rotate_half par moitiés, manualRope)
const ROPE_FULL_THETA: f32 = 1_000_000.0; // rope_parameters.full_attention.rope_theta (U0)
const ROPE_FULL_ANGLES: usize = 64; // partial 0.25 : int(0.25*512 // 2)
const ROPE_FULL_HALF: usize = 256; // head_dim // 2
const U5_TRIPWIRE_POS = [_]i64{ 708, 1030 }; // périmètre Amendement 2 (708 = pire point ULP U3)
const U5_TRIP_THR: f64 = 1.0e-3;

/// Régime EXPLICITE par S (revue Task 4) : stages_gated=true => étages a/b/v aux seuils §3
/// pleins ; false => étages a/b restreints aux positions tripwire, borne 1e-3 sans mean_abs.
/// L'étage (c) est gaté au seuil plein 1e-4 aux DEUX S (l'output gate — cf. Amendement 2, périmètre U5).
const U5Case = struct { s: i64, stages_gated: bool };
const U5_CASES = [_]U5Case{ .{ .s = 8, .stages_gated = true }, .{ .s = 1040, .stages_gated = false } };

/// cos/sin HOST du chemin full — la MÊME formule que le runner (gemma4_w4auto.zig `ropeFull`
/// l.192, proportional theta 1e6 partial 0.25) : 64 fréquences actives dupliquées par moitiés
/// (layout rotate_half), 192 inv_freq nuls → 384 composantes identité (cos=1, sin=0). C'est LE
/// point de comparaison utile du tripwire : l'oracle 66 porte les cos/sin du module HF réel,
/// le gate calcule les siens comme le runner g12auto le fera (pow f32 host — famille ULP U3).
fn ropeFullHost(p: i64, cos_out: []f32, sin_out: []f32) void {
    var inv_freq: [ROPE_FULL_HALF]f32 = undefined;
    for (0..ROPE_FULL_HALF) |i| {
        if (i < ROPE_FULL_ANGLES) {
            const exp: f32 = @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(HD_F));
            inv_freq[i] = 1.0 / std.math.pow(f32, ROPE_FULL_THETA, exp);
        } else {
            inv_freq[i] = 0.0; // nope : pas de rotation (identité quelle que soit la position)
        }
    }
    const pf: f32 = @floatFromInt(p);
    for (0..ROPE_FULL_HALF) |i| {
        const angle: f32 = inv_freq[i] * pf;
        const c: f32 = @cos(angle);
        const s: f32 = @sin(angle);
        cos_out[i] = c;
        cos_out[i + ROPE_FULL_HALF] = c;
        sin_out[i] = s;
        sin_out[i + ROPE_FULL_HALF] = s;
    }
}

/// Poids d'attention L5 FULL, lus du PACKÉ (D9 — dequantW4 in-graph). PAS de v_proj : la
/// couche n'en a pas au checkpoint (attention_k_eq_v, D4) — le mini-graphe repart du k_proj
/// packé pour V. Aucun placeholder n'est déclaré ici, a fortiori pas consommé (point D4 : le
/// placeholder [1] du runner Task 8 est inconsommable par construction ; ici il n'existe pas).
const AttnWFull = struct {
    q: w4.W4Lin,
    k: w4.W4Lin,
    o: w4.W4Lin,
    q_norm: zml.Tensor,
    k_norm: zml.Tensor,

    fn init(v_ck: zml.io.TensorStore.View) AttnWFull {
        return .{
            .q = .init(v_ck, FULL_PREFIX ++ "q_proj"),
            .k = .init(v_ck, FULL_PREFIX ++ "k_proj"),
            .o = .init(v_ck, FULL_PREFIX ++ "o_proj"),
            .q_norm = v_ck.createTensor(FULL_PREFIX ++ "q_norm.weight", .{.hd}, null),
            .k_norm = v_ck.createTensor(FULL_PREFIX ++ "k_norm.weight", .{.hd}, null),
        };
    }
};

// Ops miroir du chemin moteur FULL (runLayerGen, prec fam=null == baseline f32) — mêmes
// primitives zml, mêmes ordres de casts : dot f32 (dotPrec fam=null), q/k_norm rmsScaleHdPrec,
// p-RoPE manualRope (engine.zig:181 : split moitiés + rotate_half, cos/sin FOURNIS — pas
// zml.nn.rope), v_norm sans poids sur le k_raw (branche k_eq_v_full, engine.zig:503-508).
const U5Ops = struct {
    /// manualRope engine fam=null : x*cos + rotate_half(x)*sin, moitiés de 256.
    fn ropeManual(x: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor) zml.Tensor {
        const halves = x.split(.hd, &.{ HALF_F, HALF_F });
        const rh = zml.Tensor.concatenate(&.{ halves[1].negate(), halves[0] }, .hd);
        return x.mul(cos.broad(x.shape())).add(rh.mul(sin.broad(x.shape())));
    }

    /// Projection BRUTE {b,s,nh,hd} f32 : dequantW4 du PACKÉ + dot f32 + split têtes.
    fn projRaw(h: zml.Tensor, pk: zml.Tensor, sc: zml.Tensor) zml.Tensor {
        const w = w4.dequantW4(pk, sc).withTags(.{ .o, .d }); // bf16 {o,d}, == export dq (U1b)
        return h.convert(.f32).dot(w.convert(.f32), .d).splitAxis(.o, .{ .nh = .auto, .hd = HD_F });
    }

    /// q_norm/k_norm : rmsNorm(.hd) puis mul(poids convert f32) — ordre rmsScaleHdPrec.
    fn normScale(x: zml.Tensor, norm: zml.Tensor) zml.Tensor {
        const normalized = zml.nn.rmsNorm(x, .hd, RMS_EPS);
        return normalized.mul(norm.convert(.f32).broad(x.shape()));
    }

    /// Étages (a)/(b) : proj + q/k_norm + p-RoPE manuelle (cos/sin HOST fournis).
    fn projNormRopeFull(h: zml.Tensor, pk: zml.Tensor, sc: zml.Tensor, norm: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor) zml.Tensor {
        return ropeManual(normScale(projRaw(h, pk, sc), norm), cos, sin);
    }

    /// Étage (v) — branche K=V (D4) : V = v_norm(k_proj BRUT), AVANT k_norm et AVANT rope.
    /// Seuls les poids k_proj sont consommés (aucun v_proj n'existe sur cette couche).
    fn vStageFull(h: zml.Tensor, kpk: zml.Tensor, ksc: zml.Tensor) zml.Tensor {
        return zml.nn.rmsNorm(projRaw(h, kpk, ksc), .hd, RMS_EPS);
    }

    /// Étage (c) : attention full complète — MQA broadcast groupe 16 par splitAxis (le groupe
    /// DÉRIVE de la shape de K : k.dim(.h)=1 → hq=16, pattern engine.zig runLayerGen .manual),
    /// masque additif causal, softmax f32, context, o_proj. k_raw calculé UNE fois et consommé
    /// par K (norm+rope) ET par V (v_norm seul) — exactement la branche k_eq_v_full du moteur.
    fn attnFull(h: zml.Tensor, qpk: zml.Tensor, qsc: zml.Tensor, qn: zml.Tensor, kpk: zml.Tensor, ksc: zml.Tensor, kn: zml.Tensor, opk: zml.Tensor, osc: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor, mask: zml.Tensor) zml.Tensor {
        const q = projNormRopeFull(h, qpk, qsc, qn, cos, sin);
        const k_raw = projRaw(h, kpk, ksc); // {b,s,nh=1,hd=512}
        const k = ropeManual(normScale(k_raw, kn), cos, sin);
        const v = zml.nn.rmsNorm(k_raw, .hd, RMS_EPS); // K=V : même k_raw (D4)
        const q_final = q.transpose(.{ .b, .nh, .s, .hd }).rename(.{ .nh = .h, .s = .q });
        const k_new = k.transpose(.{ .b, .nh, .s, .hd }).rename(.{ .nh = .h, .s = .k });
        const v_new = v.transpose(.{ .b, .nh, .s, .hd }).rename(.{ .nh = .h, .s = .k });

        const qs = q_final.splitAxis(.h, .{ .h = k_new.dim(.h), .hq = .auto }); // h=1, hq=16 (MQA)
        var scores = qs.dot(k_new, .hd).merge(.{ .h = .{ .h, .hq } }).transpose(.{ .b, .h, .q, .k });
        scores = scores.add(mask.convert(.f32).broad(scores.shape())); // additif, scaling 1.0 (piège 10)
        const probs = scores.softmax(.k); // f32

        const ps = probs.splitAxis(.h, .{ .h = v_new.dim(.h), .hq = .auto });
        const ctx = ps.dot(v_new, .k).merge(.{ .h = .{ .h, .hq } }).transpose(.{ .b, .h, .q, .hd });
        const attn_m = ctx.transpose(.{ .b, .q, .h, .hd }).merge(.{ .m = .{ .h, .hd } });
        // o_proj [3840, 8192] = [.d out, .m in] (tags du site moteur, engine.LayerW.o_proj).
        return attn_m.dot(w4.dequantW4(opk, osc).withTags(.{ .d, .m }).convert(.f32), .m).rename(.{ .q = .s });
    }
};

/// Fixture d'un cas S : clés du script 66 (layouts pré-transpose des hooks + cos/sin HF).
const U5Fix = struct {
    hidden: zml.Tensor, // {b,s,d} bf16
    q: zml.Tensor, // {b,s,nh,hd} f32 (nh=16, hd=512)
    k: zml.Tensor, // {b,s,nh,hd} f32 (nh=1) — key_states FINAL (post norm+rope)
    v: zml.Tensor, // {b,s,nh,hd} f32 (nh=1) — value_states == v_norm(kp brut)
    mask: zml.Tensor, // {b,h,q,k} f32 additive (causal pur)
    out: zml.Tensor, // {b,s,d} f32 (sortie o_proj du module réel)
    cos: zml.Tensor, // {s,hd} f32 du module HF réel (tripwire vs host)
    sin: zml.Tensor, // {s,hd} f32

    fn init(v_fx: zml.io.TensorStore.View, comptime sfx: []const u8) U5Fix {
        return .{
            .hidden = v_fx.createTensor("hidden" ++ sfx, .{ .b, .s, .d }, null),
            .q = v_fx.createTensor("q" ++ sfx, .{ .b, .s, .nh, .hd }, null),
            .k = v_fx.createTensor("k" ++ sfx, .{ .b, .s, .nh, .hd }, null),
            .v = v_fx.createTensor("v" ++ sfx, .{ .b, .s, .nh, .hd }, null),
            .mask = v_fx.createTensor("mask" ++ sfx, .{ .b, .h, .q, .k }, null),
            .out = v_fx.createTensor("out" ++ sfx, .{ .b, .s, .d }, null),
            .cos = v_fx.createTensor("cos" ++ sfx, .{ .s, .hd }, null),
            .sin = v_fx.createTensor("sin" ++ sfx, .{ .s, .hd }, null),
        };
    }

    fn check(self: U5Fix, s: i64) !void {
        const g = g12.g12;
        if (self.hidden.dim(.s) != s or self.hidden.dim(.d) != g.d or
            self.q.dim(.s) != s or self.q.dim(.nh) != g.nh or self.q.dim(.hd) != g.hd_full or
            self.k.dim(.s) != s or self.k.dim(.nh) != g.kvh_full or self.k.dim(.hd) != g.hd_full or
            self.v.dim(.s) != s or self.v.dim(.nh) != g.kvh_full or self.v.dim(.hd) != g.hd_full or
            self.mask.dim(.q) != s or self.mask.dim(.k) != s or
            self.out.dim(.s) != s or self.out.dim(.d) != g.d or
            self.cos.dim(.s) != s or self.cos.dim(.hd) != HD_F or
            self.sin.dim(.s) != s or self.sin.dim(.hd) != HD_F)
        {
            log.err("u5 : fixture S={d} incohérente avec la géométrie g12 full (MQA 1x512)", .{s});
            return error.VacuousGate;
        }
    }
};

/// Témoin masque couche FULL : causal PUR (pas de fenêtre) — recompté in-gate.
fn checkMaskCausal(allocator: std.mem.Allocator, io: std.Io, mask_buf: zml.Buffer, s: i64) !void {
    var m_s = try mask_buf.toSliceAlloc(allocator, io);
    defer m_s.free(allocator);
    const m = m_s.items(f32);
    var n_masked: usize = 0;
    for (m) |x| {
        if (x < -1.0e30) n_masked += 1;
    }
    const su: usize = @intCast(s);
    const causal: usize = su * (su - 1) / 2;
    if (n_masked != causal) {
        log.err("u5 : masque S={d} — {d} masqués != causal pur {d} (couche full : pas de fenêtre)", .{ s, n_masked, causal });
        return error.VacuousGate;
    }
    log.info("    masque S={d} : {d} masqués == causal pur (full, pas de fenêtre — attendu)", .{ s, n_masked });
}

fn gateU5(allocator: std.mem.Allocator, arena: std.mem.Allocator, io: std.Io, platform: *zml.Platform, sharding: zml.sharding.Sharding, ckpt_path: []const u8, fixture_path: []const u8) !void {
    log.info("u5 (U5) — attention FULL L5 (MQA 16Qx512/1KVx512, K=V branche D4, p-RoPE host 512/0.25/1e6) vs hooks module réel", .{});
    log.info("  périmètre Amendement 2 : gating S=8 étages a/b/v + sanity pos 0 stricte + tripwire {{708,1030}} 1e-3 + étage (c) S=8 ET S=1040 à 1e-4", .{});

    var st: Stores = undefined;
    try openStores(&st, allocator, io, ckpt_path, fixture_path);
    defer st.deinit();

    const aw: AttnWFull = .init(st.store_ck.view());
    const aw_buf = try zml.io.load(AttnWFull, &aw, arena, io, platform, &st.store_ck, .{ .shardings = &.{sharding}, .parallelism = load_opts.parallelism, .dma_chunks = load_opts.dma_chunks, .dma_chunk_size = load_opts.dma_chunk_size });

    var n_pass: usize = 0;
    inline for (U5_CASES) |case| {
        const s = case.s;
        const sfx = std.fmt.comptimePrint("_s{d}", .{s});
        log.info("  cas S={d} ({s}) :", .{ s, if (case.stages_gated) "gating §3 étages a/b/v + étage c" else "tripwire {708,1030} + étage c seuil plein" });
        const fix: U5Fix = .init(st.store_fx.view(), sfx);
        try fix.check(s);
        const fix_buf = try zml.io.load(U5Fix, &fix, arena, io, platform, &st.store_fx, .{ .shardings = &.{sharding}, .parallelism = load_opts.parallelism, .dma_chunks = load_opts.dma_chunks, .dma_chunk_size = load_opts.dma_chunk_size });

        // --- cos/sin HOST (formule du runner, ropeFullHost) + sanity position 0 STRICTE ---
        const su: usize = @intCast(s);
        const hdu: usize = @intCast(HD_F);
        const cos_host = try allocator.alloc(f32, su * hdu);
        defer allocator.free(cos_host);
        const sin_host = try allocator.alloc(f32, su * hdu);
        defer allocator.free(sin_host);
        for (0..su) |p| {
            ropeFullHost(@intCast(p), cos_host[p * hdu .. (p + 1) * hdu], sin_host[p * hdu .. (p + 1) * hdu]);
        }
        for (cos_host[0..hdu], sin_host[0..hdu], 0..) |c, si, i| {
            if (c != 1.0 or si != 0.0) {
                log.err("u5 : sanity p-RoPE pos 0 VIOLÉE à hd={d} : cos={e:.6} sin={e:.6} (identité STRICTE exigée)", .{ i, c, si });
                return error.GateFailed;
            }
        }
        log.info("    sanity p-RoPE host : position 0 == identité STRICTE (cos=1, sin=0 sur {d} composantes)", .{hdu});

        // --- tripwire cos/sin : host (runner) vs fixture (module HF réel), positions {708,1030},
        // borne 1e-3, sans mean_abs (famille ULP U3 : pow f32, Δangle ∝ position — informatif) ---
        if (comptime !case.stages_gated) {
            var cos_fx_s = try fix_buf.cos.toSliceAlloc(allocator, io);
            defer cos_fx_s.free(allocator);
            const cos_fx = cos_fx_s.items(f32);
            var sin_fx_s = try fix_buf.sin.toSliceAlloc(allocator, io);
            defer sin_fx_s.free(allocator);
            const sin_fx = sin_fx_s.items(f32);
            for (U5_TRIPWIRE_POS) |tp| {
                const base = @as(usize, @intCast(tp)) * hdu;
                var max_c: f64 = 0;
                var max_s: f64 = 0;
                for (0..hdu) |i| {
                    max_c = @max(max_c, @abs(@as(f64, cos_host[base + i]) - @as(f64, cos_fx[base + i])));
                    max_s = @max(max_s, @abs(@as(f64, sin_host[base + i]) - @as(f64, sin_fx[base + i])));
                }
                if (max_c > U5_TRIP_THR or max_s > U5_TRIP_THR) {
                    log.err("    tripwire cos/sin @pos {d} : max_abs cos={e:.3} sin={e:.3} > borne {e:.1} (formule host != HF au-delà de la famille ULP)", .{ tp, max_c, max_s, U5_TRIP_THR });
                    return error.GateFailed;
                }
                log.info("    tripwire cos/sin @pos {d} : max_abs cos={e:.3} sin={e:.3} (borne {e:.1}) OK", .{ tp, max_c, max_s, U5_TRIP_THR });
            }
        }

        // Buffers device des cos/sin HOST (motif w4auto : Tensor.init symbolique + fromBytes).
        const cos_sym = zml.Tensor.init(.{ s, HD_F }, .f32).withTags(.{ .s, .hd });
        const sin_sym = zml.Tensor.init(.{ s, HD_F }, .f32).withTags(.{ .s, .hd });
        var cos_buf = try zml.Buffer.fromBytes(io, platform, cos_sym.shape(), sharding, std.mem.sliceAsBytes(cos_host));
        defer cos_buf.deinit();
        var sin_buf = try zml.Buffer.fromBytes(io, platform, sin_sym.shape(), sharding, std.mem.sliceAsBytes(sin_host));
        defer sin_buf.deinit();

        // Témoin : masque causal PUR recompté (couche full — pas de fenêtre).
        try checkMaskCausal(allocator, io, fix_buf.mask, s);

        if (comptime case.stages_gated) {
            // Étage (a) Q : proj + q_norm + p-RoPE manuelle — seuils §3 pleins.
            try runCompare(allocator, io, platform, sharding, U5Ops.projNormRopeFull, .{ fix.hidden, aw.q.pk, aw.q.sc, aw.q_norm, cos_sym, sin_sym }, .{ fix_buf.hidden, aw_buf.q.pk, aw_buf.q.sc, aw_buf.q_norm, cos_buf, sin_buf }, "étage a (q post proj+norm+p-rope)", fix_buf.q, .{ .max_thr = U3_MAX_ABS, .mean_thr = U3_MEAN_ABS });
            // Étage (b) K FINAL : proj + k_norm + p-RoPE manuelle.
            try runCompare(allocator, io, platform, sharding, U5Ops.projNormRopeFull, .{ fix.hidden, aw.k.pk, aw.k.sc, aw.k_norm, cos_sym, sin_sym }, .{ fix_buf.hidden, aw_buf.k.pk, aw_buf.k.sc, aw_buf.k_norm, cos_buf, sin_buf }, "étage b (k final post proj+norm+p-rope)", fix_buf.k, .{ .max_thr = U3_MAX_ABS, .mean_thr = U3_MEAN_ABS });
            // Étage (v) K=V (D4) : v_norm(k_proj BRUT) — seuls les poids k_proj consommés.
            try runCompare(allocator, io, platform, sharding, U5Ops.vStageFull, .{ fix.hidden, aw.k.pk, aw.k.sc }, .{ fix_buf.hidden, aw_buf.k.pk, aw_buf.k.sc }, "étage v (K=V : v_norm(k_proj brut), D4)", fix_buf.v, .{ .max_thr = U3_MAX_ABS, .mean_thr = U3_MEAN_ABS });
        } else {
            // Tripwire étages ropés a/b RESTREINTS aux positions {708,1030} (borne 1e-3, sans
            // mean_abs) — la phase ULP host/HF amplifiée par la position passe par ces étages.
            const nh_elems: usize = g12.g12.nh * hdu; // éléments par position .s de l'étage a ({nh,hd})
            try runCompare(allocator, io, platform, sharding, U5Ops.projNormRopeFull, .{ fix.hidden, aw.q.pk, aw.q.sc, aw.q_norm, cos_sym, sin_sym }, .{ fix_buf.hidden, aw_buf.q.pk, aw_buf.q.sc, aw_buf.q_norm, cos_buf, sin_buf }, "tripwire étage a (q)", fix_buf.q, .{ .max_thr = U5_TRIP_THR, .rows = &U5_TRIPWIRE_POS, .row_elems = nh_elems });
            try runCompare(allocator, io, platform, sharding, U5Ops.projNormRopeFull, .{ fix.hidden, aw.k.pk, aw.k.sc, aw.k_norm, cos_sym, sin_sym }, .{ fix_buf.hidden, aw_buf.k.pk, aw_buf.k.sc, aw_buf.k_norm, cos_buf, sin_buf }, "tripwire étage b (k)", fix_buf.k, .{ .max_thr = U5_TRIP_THR, .rows = &U5_TRIPWIRE_POS, .row_elems = hdu });
        }

        // Étage (c) attention full complète : S=8 ET S=1040 au SEUIL PLEIN 1e-4 (le gating).
        try runCompare(allocator, io, platform, sharding, U5Ops.attnFull, .{ fix.hidden, aw.q.pk, aw.q.sc, aw.q_norm, aw.k.pk, aw.k.sc, aw.k_norm, aw.o.pk, aw.o.sc, cos_sym, sin_sym, fix.mask }, .{ fix_buf.hidden, aw_buf.q.pk, aw_buf.q.sc, aw_buf.q_norm, aw_buf.k.pk, aw_buf.k.sc, aw_buf.k_norm, aw_buf.o.pk, aw_buf.o.sc, cos_buf, sin_buf, fix_buf.mask }, "étage c (attention full complète -> o_proj)", fix_buf.out, .{ .max_thr = U3_MAX_ABS });
        n_pass += 1;
    }
    if (n_pass != U5_CASES.len) return error.GateFailed;
    log.info("PASS u5 — MQA K=V p-RoPE : étages a/b/v S=8 aux seuils §3, sanity pos 0 stricte, tripwire {{708,1030}} sous 1e-3, étage c S=8+S=1040 max_abs<=1e-4", .{});
}

// ---------------------------------------------------------------------------- u6 (couche complète + chaîne L0→L5 — MOTEUR)

// U6 passe par le MOTEUR (arbitrage contrôleur 23618bb : le mode u5 vérifie par ops miroir —
// la branche moteur D4 `k_eq_v_full` de runLayerGen n'était donc pas encore exercée ; U6 doit
// l'exercer via runLayerGen/Geom.g12). `runLayerGen` n'est PAS `pub` (engine.zig:463) et
// engine.zig est INTOUCHABLE ici (le md5 U1 est en jeu) — la voie DÉCLARÉE est le Geom
// TRONQUÉ : `G12U6` = g12.g12 réduit à `num_layers=6, first_kv_shared=6` (donne exactement
// L0-L5 avec la couche 5 full : (5+1)%6==0 ; aucun reader ; slots dérivés corrects — sliding
// 0..4, full 0), moteur invoqué par `forwardStageGen(i, i+1, first=false, last=false)`
// (runLayerGen partagé, engine.zig:698) : un stage par couche, 8 steps moteur par stage
// (positions 0-7, prefill S=8 step-par-step, cache mock zéro threadé — pattern P5.3 /
// w4auto HostInputs). L'ordre est stage-major : la couche j consomme aux 8 positions le
// hidden produit par la couche j-1 (chaîne bit-préservée via host, f32) et SON cache écrit
// aux steps précédents — mêmes dépendances que le decode réel.
//
// Branche D4 DANS le graphe : la couche 5 est full et G12U6.k_eq_v_full=true → runLayerGen
// émet `v_raw = k_raw` (k_proj BRUT avant k_norm/rope) + v_norm sans poids. Son
// `LayerW.v_proj` = PLACEHOLDER de shape inconsommable [1] (le tenseur layer_scalar, tags
// {.one}) — JAMAIS k_proj recyclé (D4 : un placeholder shape-compatible consommé par erreur
// donnerait v_norm(kp) correct par accident). Toute consommation accidentelle casserait au
// compile (dot sur .d impossible sur {.one}[1]) : le compile du stage 5 EST la preuve de
// non-consommation. Champs PLE (per_layer_input_gate/projection/post_per_layer_input_norm)
// et champs modèle non consommés (embed_tokens/final_norm/plmp/pl_norm — last=false,
// ple_dim=0) : même placeholder [1], branches comptime-mortes.
//
// Assemblage des engine.LayerW par VALEUR (pattern W4Step, gemma4_w4auto.zig:730-756) :
// poids packés du checkpoint → w4.W4Lin.toW(tags) in-graph, forwardStageGen inchangé.
// cos/sin full : ENTRÉES du graphe (Packed) — calculés host par ropeFullHost (formule du
// runner, même fn que le tripwire u5). Masque : two_masks=false, UNE table causale — licite
// à S=8 : l'oracle 67 ASSERTE bit-égalité masque sliding HF == masque causal HF (fenêtre
// 1024 non mordante).

const U6_MAX_ABS: f64 = 2.0e-4; // §3 U6 — resserré au vu de l'oracle (1e-3 → 2e-4), décision contrôleur 25 juil — jamais élargissable
const U6_STEPS: usize = 8; // S=8 (périmètre Amendement 2 : « S=8 explicite »), positions 0-7
const D_12B: i64 = @intCast(g12.g12.d); // 3840
const MASK_MIN: f32 = -std.math.floatMax(f32); // == torch.finfo(float32).min (masque additif HF)

/// Géométrie TRONQUÉE (voie déclarée) : g12 réduit aux couches 0-5 — seuls num_layers et
/// first_kv_shared changent, tout le reste (d, têtes, hd, kvh, k_eq_v_full, full_period 6,
/// theta, softcap) est EXACTEMENT g12.g12 — la chaîne exerce les couches réelles du 12B.
const G12U6: engine.Geom = blk: {
    var g = g12.g12;
    g.num_layers = 6;
    g.first_kv_shared = 6; // aucun reader : chaque couche écrit son slot (pattern g12, pas de YOCO)
    break :blk g;
};
const U6Model = engine.EngineModel(struct {}, .{ .geom = G12U6 });
const U6_SL_SLOTS: i64 = 5; // couches 0-4 sliding → slidingSlot 0..4
const U6_FL_SLOTS: i64 = 1; // couche 5 full → fullSlot 0

/// Table de RÉGIME EXPLICITE (pattern U3Case/U5Case) : 6 profondeurs de chaîne, TOUTES gatées
/// au seuil §3 U6 (max_abs <= U6_MAX_ABS, resserré 2e-4), S=8 seul ; la chaîne == profondeur 5 (sortie L5).
const U6Lay = struct { idx: usize, kind: []const u8 };
const U6_LAYERS = [_]U6Lay{
    .{ .idx = 0, .kind = "sliding GQA 8x256 — « couche 0 seule » (entrée = hidden oracle)" },
    .{ .idx = 1, .kind = "sliding GQA 8x256" },
    .{ .idx = 2, .kind = "sliding GQA 8x256" },
    .{ .idx = 3, .kind = "sliding GQA 8x256" },
    .{ .idx = 4, .kind = "sliding GQA 8x256" },
    .{ .idx = 5, .kind = "full MQA 1x512 K=V (branche D4 moteur, v_proj placeholder [1])" },
};

/// Poids d'UNE couche 12B au format packé — champs COMMUNS aux 6 couches (SANS v_proj : la
/// couche 5 n'en a pas au checkpoint ; pattern « deux structs distincts » w4.W4LayerW/W4KV,
/// zml.io.load exige des champs Tensor, pas d'optionnel).
const U6LayW = struct {
    input_layernorm: zml.Tensor,
    q: w4.W4Lin,
    q_norm: zml.Tensor,
    k: w4.W4Lin,
    k_norm: zml.Tensor,
    o: w4.W4Lin,
    post_attention_layernorm: zml.Tensor,
    pre_feedforward_layernorm: zml.Tensor,
    gate: w4.W4Lin,
    up: w4.W4Lin,
    down: w4.W4Lin,
    post_feedforward_layernorm: zml.Tensor,
    layer_scalar: zml.Tensor, // [1] bf16 — consommé en fin de couche ET placeholder [1] (D4/PLE)

    fn init(v: zml.io.TensorStore.View) U6LayW {
        const sa = v.withPrefix("self_attn");
        const mlp = v.withPrefix("mlp");
        return .{
            .input_layernorm = v.createTensor("input_layernorm.weight", .{.d}, null),
            .q = .init(sa, "q_proj"),
            .q_norm = sa.createTensor("q_norm.weight", .{.hd}, null),
            .k = .init(sa, "k_proj"),
            .k_norm = sa.createTensor("k_norm.weight", .{.hd}, null),
            .o = .init(sa, "o_proj"),
            .post_attention_layernorm = v.createTensor("post_attention_layernorm.weight", .{.d}, null),
            .pre_feedforward_layernorm = v.createTensor("pre_feedforward_layernorm.weight", .{.d}, null),
            .gate = .init(mlp, "gate_proj"),
            .up = .init(mlp, "up_proj"),
            .down = .init(mlp, "down_proj"),
            .post_feedforward_layernorm = v.createTensor("post_feedforward_layernorm.weight", .{.d}, null),
            .layer_scalar = v.createTensor("layer_scalar", .{.one}, null),
        };
    }
};

/// Les 6 couches + les v_proj des 5 couches sliding (miroir w4.W4Model : layers[] + kv[]).
const U6W = struct {
    layers: []U6LayW, // 6 (champs communs)
    vs: []w4.W4Lin, // 5 (v_proj des couches sliding 0-4 — la couche 5 full n'en a PAS)

    fn init(allocator: std.mem.Allocator, base: zml.io.TensorStore.View) !U6W {
        const layers = try allocator.alloc(U6LayW, G12U6.num_layers);
        const vs = try allocator.alloc(w4.W4Lin, G12U6.num_layers - 1);
        const lb = base.withPrefix("layers");
        for (layers, 0..) |*l, i| l.* = U6LayW.init(lb.withLayer(i));
        for (vs, 0..) |*x, i| x.* = .init(lb.withLayer(i).withPrefix("self_attn"), "v_proj");
        return .{ .layers = layers, .vs = vs };
    }
};

/// Assemblage par VALEUR d'un engine.LayerW (pattern W4Step / w4.toLayerW) : dequantW4
/// in-graph via toW(tags des sites moteur). Couche full (comptime) : v_proj = PLACEHOLDER [1]
/// (layer_scalar) — la branche k_eq_v_full du moteur ne le consomme pas, preuve par compile.
fn u6LayerW(w: U6W, comptime j: usize) engine.LayerW {
    const lw = w.layers[j];
    return .{
        .input_layernorm = lw.input_layernorm,
        .q_proj = lw.q.toW(.{ .o, .d }),
        .q_norm = lw.q_norm,
        .k_proj = lw.k.toW(.{ .o, .d }),
        .k_norm = lw.k_norm,
        .v_proj = if (comptime G12U6.isFull(j)) lw.layer_scalar else w.vs[j].toW(.{ .o, .d }),
        .o_proj = lw.o.toW(.{ .d, .m }),
        .post_attention_layernorm = lw.post_attention_layernorm,
        .pre_feedforward_layernorm = lw.pre_feedforward_layernorm,
        .gate_proj = lw.gate.toW(.{ .f, .d }),
        .up_proj = lw.up.toW(.{ .f, .d }),
        .down_proj = lw.down.toW(.{ .d, .f }),
        .post_feedforward_layernorm = lw.post_feedforward_layernorm,
        .per_layer_input_gate = lw.layer_scalar, // placeholders [1] — bloc PLE comptime-mort (ple_dim=0)
        .per_layer_projection = lw.layer_scalar,
        .post_per_layer_input_norm = lw.layer_scalar,
        .layer_scalar = lw.layer_scalar,
    };
}

/// Stage moteur de la couche `li` : assemble le U6Model par valeur et invoque forwardStageGen
/// (li, li+1, first=false — hidden vient de l'entrée, PAS des embeds ; last=false — pas de
/// head). Le graphe émis pour la couche li est EXACTEMENT runLayerGen (engine.zig:463).
fn U6Stage(comptime li: usize) type {
    return struct {
        pub fn forward(w: U6W, hidden_in: zml.Tensor, p: engine.Packed(false), cache: engine.Cache, ctrl: engine.Ctrl) struct { zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor } {
            var layers: [G12U6.num_layers]engine.LayerW = undefined;
            inline for (0..G12U6.num_layers) |j| layers[j] = u6LayerW(w, j);
            const m: U6Model = .{
                // Champs modèle NON consommés (last=false, ple_dim=0) : placeholder [1].
                .embed_tokens = w.layers[0].layer_scalar,
                .per_layer_model_projection = w.layers[0].layer_scalar,
                .per_layer_projection_norm = w.layers[0].layer_scalar,
                .final_norm = w.layers[0].layer_scalar,
                .layers = &layers,
                .brick = .{},
                .prec = .{}, // baseline f32 (fam=null) — le flux comparé par l'oracle 67
            };
            const out, const slk, const slv, const flk, const flv = m.forwardStageGen(li, li + 1, false, false, p, cache, hidden_in, ctrl);
            return .{ out, slk, slv, flk, flv };
        }
    };
}

/// Stat d'une profondeur de chaîne (agrégée sur les 8 steps x 3840).
const U6Stat = struct {
    max_abs: f64 = 0,
    sum_abs: f64 = 0,
    n: usize = 0,
    max_step: usize = 0,
    max_i: usize = 0,
    got: f32 = 0,
    exp: f32 = 0,
};

fn gateU6(allocator: std.mem.Allocator, arena: std.mem.Allocator, io: std.Io, platform: *zml.Platform, sharding: zml.sharding.Sharding, ckpt_path: []const u8, fixture_path: []const u8) !void {
    log.info("u6 (U6) — couche complète + chaîne L0→L5 via le MOTEUR (arbitrage 23618bb) : Geom tronqué {d} couches + forwardStageGen (runLayerGen), branche D4 k_eq_v_full DANS le graphe (L5), v_proj L5 = placeholder [1]", .{G12U6.num_layers});
    log.info("  S={d} (positions 0-{d}), seuil §3 U6 : max_abs <= {e:.1} par couche ET chaîne — resserré depuis 1e-3 au vu de l'oracle (décision contrôleur 25 juil), JAMAIS élargissable", .{ U6_STEPS, U6_STEPS - 1, U6_MAX_ABS });

    var st: Stores = undefined;
    try openStores(&st, allocator, io, ckpt_path, fixture_path);
    defer st.deinit();

    // --- Poids packés des 6 couches (checkpoint, D9) ---
    const base = st.store_ck.view().withPrefix("model").withPrefix("language_model");
    const u6w: U6W = try .init(arena, base);
    const u6w_buf = try zml.io.load(U6W, &u6w, arena, io, platform, &st.store_ck, .{ .shardings = &.{sharding}, .parallelism = load_opts.parallelism, .dma_chunks = load_opts.dma_chunks, .dma_chunk_size = load_opts.dma_chunk_size });

    // --- Fixture (script 67) : hidden bf16 + les 6 sorties de couche f32 — vers le HOST ---
    const s_i: i64 = @intCast(U6_STEPS);
    const hid_sym: OneT = .{ .t = st.store_fx.view().createTensor("hidden", .{ .b, .s, .d }, null) };
    if (hid_sym.t.dim(.b) != 1 or hid_sym.t.dim(.s) != s_i or hid_sym.t.dim(.d) != D_12B or hid_sym.t.dtype() != .bf16) {
        log.err("u6 : fixture hidden incohérente ({d}x{d}x{d} {s}, attendu 1x{d}x{d} bf16)", .{ hid_sym.t.dim(.b), hid_sym.t.dim(.s), hid_sym.t.dim(.d), @tagName(hid_sym.t.dtype()), s_i, D_12B });
        return error.VacuousGate;
    }
    const hid_buf = try zml.io.load(OneT, &hid_sym, arena, io, platform, &st.store_fx, .{ .shardings = &.{sharding}, .parallelism = load_opts.parallelism, .dma_chunks = load_opts.dma_chunks, .dma_chunk_size = load_opts.dma_chunk_size });
    var hid_bf16_s = try hid_buf.t.toSliceAlloc(allocator, io);
    defer hid_bf16_s.free(allocator);
    const hid_bf16 = hid_bf16_s.items(u16);

    const d_us: usize = @intCast(D_12B);
    const n_row = d_us; // éléments par (step) : {b=1,s=1,d}
    const n_all = U6_STEPS * n_row;

    var exp_host: [G12U6.num_layers][]f32 = undefined;
    inline for (0..G12U6.num_layers) |li| {
        const name = std.fmt.comptimePrint("out_l{d}", .{li});
        const sym: OneT = .{ .t = st.store_fx.view().createTensor(name, .{ .b, .s, .d }, null) };
        if (sym.t.dim(.s) != s_i or sym.t.dim(.d) != D_12B or sym.t.dtype() != .f32) {
            log.err("u6 : fixture {s} incohérente", .{name});
            return error.VacuousGate;
        }
        const b = try zml.io.load(OneT, &sym, arena, io, platform, &st.store_fx, .{ .shardings = &.{sharding}, .parallelism = load_opts.parallelism, .dma_chunks = load_opts.dma_chunks, .dma_chunk_size = load_opts.dma_chunk_size });
        var sl = try b.t.toSliceAlloc(allocator, io);
        defer sl.free(allocator);
        exp_host[li] = try arena.dupe(f32, sl.items(f32));
        if (exp_host[li].len != n_all) {
            log.err("u6 : {s} — {d} f32 != {d}", .{ name, exp_host[li].len, n_all });
            return error.VacuousGate;
        }
    }

    // --- hid_host : hidden f32 courant de la chaîne (init = fixture bf16 -> f32 EXACT) ---
    const hid_host = try arena.alloc(f32, n_all);
    const hid_next = try arena.alloc(f32, n_all);
    if (hid_bf16.len != n_all) {
        log.err("u6 : hidden — {d} bf16 != {d}", .{ hid_bf16.len, n_all });
        return error.VacuousGate;
    }
    for (hid_bf16, 0..) |hu, i| hid_host[i] = @bitCast(@as(u32, hu) << 16);

    // --- Tables host Packed (pattern w4auto HostInputs, tronquées à 8 steps) ---
    const hd_f_us: usize = @intCast(HD_F);
    const cos_host = try arena.alloc(f32, U6_STEPS * hd_f_us);
    const sin_host = try arena.alloc(f32, U6_STEPS * hd_f_us);
    for (0..U6_STEPS) |p| ropeFullHost(@intCast(p), cos_host[p * hd_f_us .. (p + 1) * hd_f_us], sin_host[p * hd_f_us .. (p + 1) * hd_f_us]);
    // Masque causal UNIQUE (two_masks=false) : ligne du step p = 0 pour k<=p, MASK_MIN sinon —
    // licite à S=8 : l'oracle 67 asserte bit-égalité masque sliding HF == causal HF (1024 > 8).
    const mask_host = try arena.alloc(f32, U6_STEPS * U6_STEPS);
    for (0..U6_STEPS) |p| {
        for (0..U6_STEPS) |k| mask_host[p * U6_STEPS + k] = if (k <= p) 0 else MASK_MIN;
    }
    const pos_host = try arena.alloc(i32, U6_STEPS);
    for (pos_host, 0..) |*x, p| x.* = @intCast(p);
    const emb_zero = try arena.alloc(u8, U6_STEPS * d_us * 2); // bf16 zéros — NON consommé (first=false)
    @memset(emb_zero, 0);
    const eptl_zero = try arena.alloc(u8, U6_STEPS * 2); // {8,1,1,1} bf16 — NON consommé (ple_dim=0)
    @memset(eptl_zero, 0);
    // Cache mock zéro (pattern P5.3 / w4auto) : shapes réelles Geom tronqué, KMAX=8, f32
    // (prec.kv_store=null => dtype compute — cohérence Cache.checkDtype).
    const sl_bytes = @as(usize, @intCast(U6_SL_SLOTS)) * @as(usize, @intCast(g12.g12.kvh_sliding)) * U6_STEPS * @as(usize, @intCast(HD_S)) * 4;
    const fl_bytes = @as(usize, @intCast(U6_FL_SLOTS)) * @as(usize, @intCast(g12.g12.kvh_full)) * U6_STEPS * hd_f_us * 4;
    const cache_sl_zero = try arena.alloc(u8, sl_bytes);
    @memset(cache_sl_zero, 0);
    const cache_fl_zero = try arena.alloc(u8, fl_bytes);
    @memset(cache_fl_zero, 0);

    // --- Symboliques (pattern w4auto : Tensor.init à la main, mêmes shapes que Packed/Cache) ---
    const steps_i: i64 = @intCast(U6_STEPS);
    const packed_sym = engine.Packed(false){
        .embeds = zml.Tensor.init(.{ steps_i, 1, 1, D_12B }, .bf16).withTags(.{ .step, .b, .s, .d }),
        .embptls = zml.Tensor.init(.{ steps_i, 1, 1, 1 }, .bf16).withTags(.{ .step, .b, .s, .lf }),
        .cos_full = zml.Tensor.init(.{ steps_i, 1, 1, HD_F }, .f32).withTags(.{ .step, .b, .s, .hd }),
        .sin_full = zml.Tensor.init(.{ steps_i, 1, 1, HD_F }, .f32).withTags(.{ .step, .b, .s, .hd }),
        .masks = zml.Tensor.init(.{ steps_i, 1, 1, 1, steps_i }, .f32).withTags(.{ .step, .b, .h, .q, .k }),
        .positions = zml.Tensor.init(.{steps_i}, .i32).withTags(.{.step}),
    };
    const kvh_sl: i64 = @intCast(g12.g12.kvh_sliding);
    const kvh_fl: i64 = @intCast(g12.g12.kvh_full);
    const cache_sym = engine.Cache{
        .sl_k = zml.Tensor.init(.{ U6_SL_SLOTS, 1, kvh_sl, steps_i, HD_S }, .f32).withTags(.{ .slot, .b, .h, .k, .hd }),
        .sl_v = zml.Tensor.init(.{ U6_SL_SLOTS, 1, kvh_sl, steps_i, HD_S }, .f32).withTags(.{ .slot, .b, .h, .k, .hd }),
        .fl_k = zml.Tensor.init(.{ U6_FL_SLOTS, 1, kvh_fl, steps_i, HD_F }, .f32).withTags(.{ .slot, .b, .h, .k, .hd }),
        .fl_v = zml.Tensor.init(.{ U6_FL_SLOTS, 1, kvh_fl, steps_i, HD_F }, .f32).withTags(.{ .slot, .b, .h, .k, .hd }),
    };
    const ctrl_sym: engine.Ctrl = .initSymbolic();
    const hidden_sym = zml.Tensor.init(.{ 1, 1, D_12B }, .f32).withTags(.{ .b, .s, .d });
    try cache_sym.checkDtype(.{}); // prec par défaut (kv_store=null) => cache f32 attendu — témoin

    const pk_buf = zml.Bufferized(engine.Packed(false)){
        .embeds = try zml.Buffer.fromBytes(io, platform, packed_sym.embeds.shape(), sharding, emb_zero),
        .embptls = try zml.Buffer.fromBytes(io, platform, packed_sym.embptls.shape(), sharding, eptl_zero),
        .cos_full = try zml.Buffer.fromBytes(io, platform, packed_sym.cos_full.shape(), sharding, std.mem.sliceAsBytes(cos_host)),
        .sin_full = try zml.Buffer.fromBytes(io, platform, packed_sym.sin_full.shape(), sharding, std.mem.sliceAsBytes(sin_host)),
        .masks = try zml.Buffer.fromBytes(io, platform, packed_sym.masks.shape(), sharding, std.mem.sliceAsBytes(mask_host)),
        .positions = try zml.Buffer.fromBytes(io, platform, packed_sym.positions.shape(), sharding, std.mem.sliceAsBytes(pos_host)),
    };
    var cache_buf = zml.Bufferized(engine.Cache){
        .sl_k = try zml.Buffer.fromBytes(io, platform, cache_sym.sl_k.shape(), sharding, cache_sl_zero),
        .sl_v = try zml.Buffer.fromBytes(io, platform, cache_sym.sl_v.shape(), sharding, cache_sl_zero),
        .fl_k = try zml.Buffer.fromBytes(io, platform, cache_sym.fl_k.shape(), sharding, cache_fl_zero),
        .fl_v = try zml.Buffer.fromBytes(io, platform, cache_sym.fl_v.shape(), sharding, cache_fl_zero),
    };

    // --- Chaîne stage-major : couche par couche (compile 1 stage moteur), 8 steps par stage ---
    var stats: [G12U6.num_layers]U6Stat = @splat(.{});
    inline for (0..G12U6.num_layers) |li| {
        log.info("  stage L{d} ({s}) : compile forwardStageGen({d},{d}) ...", .{ li, U6_LAYERS[li].kind, li, li + 1 });
        var exe = try platform.compileFn(allocator, io, U6Stage(li).forward, .{ u6w, hidden_sym, packed_sym, cache_sym, ctrl_sym }, .{ .shardings = &.{sharding} });
        defer exe.deinit();
        if (comptime G12U6.isFull(li)) {
            log.info("    branche k_eq_v_full ÉMISE (L{d} full) — v_proj placeholder [1] NON consommé : le compile ci-dessus est la preuve (D4)", .{li});
        }

        for (0..U6_STEPS) |step| {
            var hidden_buf = try zml.Buffer.fromBytes(io, platform, hidden_sym.shape(), sharding, std.mem.sliceAsBytes(hid_host[step * n_row .. (step + 1) * n_row]));
            var step_buf = try zml.Buffer.scalar(io, platform, @as(u32, @intCast(step)), .u32, sharding);
            const ctrl_buf = zml.Bufferized(engine.Ctrl){ .step = step_buf };

            var call_args = try exe.args(allocator);
            var call_results = try exe.results(allocator);
            call_args.set(.{ u6w_buf, hidden_buf, pk_buf, cache_buf, ctrl_buf });
            exe.call(call_args, &call_results);
            var r_hid, const r_slk, const r_slv, const r_flk, const r_flv = call_results.get(struct { zml.Buffer, zml.Buffer, zml.Buffer, zml.Buffer, zml.Buffer });

            // cache swap (motif w4auto) : le cache grandi remplace l'ancien.
            var old_cache = cache_buf;
            cache_buf = zml.Bufferized(engine.Cache){ .sl_k = r_slk, .sl_v = r_slv, .fl_k = r_flk, .fl_v = r_flv };
            old_cache.sl_k.deinit();
            old_cache.sl_v.deinit();
            old_cache.fl_k.deinit();
            old_cache.fl_v.deinit();

            // D2H : sortie de couche à cette position — comparée à l'oracle, et hidden de la
            // couche suivante (chaîne : le moteur consomme SA propre sortie, roundtrip f32 exact).
            var got_s = try r_hid.toSliceAlloc(allocator, io);
            defer got_s.free(allocator);
            const got = got_s.items(f32);
            if (got.len != n_row) {
                log.err("u6 : stage L{d} step {d} — {d} f32 != {d}", .{ li, step, got.len, n_row });
                return error.GateFailed;
            }
            const exp_row = exp_host[li][step * n_row .. (step + 1) * n_row];
            for (got, exp_row, 0..) |g, e, i| {
                const d = @abs(@as(f64, g) - @as(f64, e));
                stats[li].sum_abs += d;
                if (d > stats[li].max_abs) {
                    stats[li].max_abs = d;
                    stats[li].max_step = step;
                    stats[li].max_i = i;
                    stats[li].got = g;
                    stats[li].exp = e;
                }
            }
            stats[li].n += n_row;
            @memcpy(hid_next[step * n_row .. (step + 1) * n_row], got);

            r_hid.deinit();
            hidden_buf.deinit();
            step_buf.deinit();
            call_args.deinit(allocator);
            call_results.deinit(allocator);
        }
        @memcpy(hid_host, hid_next);
    }
    cache_buf.sl_k.deinit();
    cache_buf.sl_v.deinit();
    cache_buf.fl_k.deinit();
    cache_buf.fl_v.deinit();

    // --- Verdicts par couche (profondeur de chaîne) + chaîne (== L5) ---
    var failed = false;
    inline for (0..G12U6.num_layers) |li| {
        const st_li = stats[li];
        const mean_abs = st_li.sum_abs / @as(f64, @floatFromInt(st_li.n));
        if (st_li.max_abs <= U6_MAX_ABS) {
            log.info("    L{d} ({s}) : max_abs={e:.3} mean_abs={e:.3} ({d} f32) OK", .{ li, U6_LAYERS[li].kind, st_li.max_abs, mean_abs, st_li.n });
        } else {
            log.err("    L{d} ({s}) : FAIL max_abs={e:.3} > seuil {e:.1} (mean_abs={e:.3}) — pire écart step {d} idx {d} : got={e:.6} expected={e:.6}", .{ li, U6_LAYERS[li].kind, st_li.max_abs, U6_MAX_ABS, mean_abs, st_li.max_step, st_li.max_i, st_li.got, st_li.exp });
            failed = true;
        }
    }
    const chain = stats[G12U6.num_layers - 1];
    if (failed or chain.max_abs > U6_MAX_ABS) {
        log.err("u6 : FAIL — un dépassement du seuil {e:.1} est un FAIL, pas une renégociation (§3 U6)", .{U6_MAX_ABS});
        return error.GateFailed;
    }
    log.info("PASS u6 — couche 0 complète (layer_scalar consommé du checkpoint) + chaîne L0→L5 via le MOTEUR (forwardStageGen/Geom tronqué, branche D4 en graphe, placeholder v_proj [1] prouvé par compile) : chaîne max_abs={e:.3} <= {e:.1}", .{ chain.max_abs, U6_MAX_ABS });
}

// ---------------------------------------------------------------------------- u7 (prefill 48 couches + head softcap — MOTEUR)

// U7 — le forward 12B COMPLET : généralisation du pattern u6 à la géométrie PLEINE Geom.g12
// (48 couches), en stage-major CHUNKÉ — 8 chunks forwardStageGen de 6 couches, le dernier
// avec LAST=TRUE — VOIE DÉCLARÉE (itération 4 : le repli « forwardStageGen chunké si mur
// RAM » prescrit par le plan Task 7 ; historique des 3 runs crashés à l'en-tête de U7Chunk),
// PLUS le chemin head du MOTEUR : final_norm + lm_head tied (dot sur embed_tokens — même
// chemin que le forward E2B, engine.zig:724-727) + softcap (softcapPrec fam=null :
// raw.scale(1/30).tanh().scale(30)) — aucune op head recomposée ici.
//
// DEUX PIÈGES identifiés par la revue U6, corrigés dans CETTE généralisation (6 -> 48) :
//   (a) `vs` dimensionné num_layers - num_layers/full_period = 48 - 8 = 40 (8 couches full
//       sans v_proj) — PAS `num_layers - 1`, qui ne valait en u6 que parce que le Geom
//       tronqué n'a qu'UNE couche full ;
//   (b) l'indexation des v_proj passe par slidingSlot(j) (== nombre de couches non-full < j) —
//       les couches full s'entrelacent tous les 6 : `vs[j]` brut lirait la MAUVAISE arête EN
//       SILENCE dès la couche 6 (le v_proj « numéro 6 » appartient à la couche 7, la couche 5
//       étant full). Le remplissage de U7W.init itère les couches dans l'ordre en n'ajoutant
//       que les sliding : l'index séquentiel == slidingSlot par construction, ASSERTÉ.
//
// Embed : chemin 12B du mode u2 (D12) — gather + scale bf16 62.0 dans un graphe DÉDIÉ
// (U2.forward réutilisé tel quel), résultat bf16 -> f32 host (exact) = hidden d'entrée du
// stage 0 (first=false : le scale √3840 f64 du moteur est émis mais MORT — hidden=hidden_in,
// ple_dim=0). engine.zig intouché (D12 : jamais le downcast bf16 dans le moteur — md5 U1).
//
// Masque : two_masks=false, UNE table causale — licite au S du prompt : l'oracle 68 ASSERTE
// bit-égalité masque sliding HF == causal HF (S << fenêtre 1024, même témoin que u6).
// cos/sin full : host (ropeFullHost, même fn que u5/u6). Cache mock zéro KMAX=S, slots
// PLEINS : 40 sliding + 8 full (fullSlot/slidingSlot du Geom plein).
//
// Verdict (§3-U7, périmètre Amendement 2) sur les logits de la DERNIÈRE position :
// softcap exercé côté ZML aussi (max|logits| <= 30 ET > 25 quelque part), top-5 en ENSEMBLE
// + tie rule |Δlogit| < 1e-4 mesurée sur les logits HF de la fixture (la référence — piège
// 12 ; l'écart côté ZML est consigné au log), max_abs DOCUMENTAIRE sous garde-fou 0.5
// (Amendement 2 §U7, requalifié décision Régis 25 juil : l'oracle est le modèle bf16 RÉEL,
// enveloppe G2 0.1-0.4 — le garde-fou reste CÂBLÉ, un dépassement = FAIL,
// chiffres au log ; marges top1-top2 et rang5-rang6 lues AVANT tout verdict, piège 17).

const N12: usize = g12.g12.num_layers; // 48
const N12_FULL: usize = N12 / g12.g12.full_period; // 8
const N12_SLIDING: usize = N12 - N12_FULL; // 40 — piège revue U6 (a)
const U7Model = engine.EngineModel(struct {}, .{ .geom = g12.g12 });
const U7_MAX_ABS: f64 = 5.0e-1; // garde-fou documentaire (Amendement 2 §U7, requalifié décision Régis 25 juil : critère discriminant = top-5 ensemble+ordre+marges + softcap ; enveloppe G2 bf16-réel, mesuré 0.376 — CÂBLÉ : dépassement = FAIL)
const U7_TIE: f64 = 1.0e-4; // tie rule §3-U7 (piège 17)
const U7_SOFTCAP: f64 = 30.0;
const U7_SOFTCAP_BITE: f64 = 25.0;
const VOC_12B: i64 = 262144;

/// Poids du modèle COMPLET au format packé : 48 couches (champs communs, U6LayW réutilisé) +
/// v_proj des 40 couches sliding + embed_tokens (tied lm_head, D7) + final_norm.
const U7W = struct {
    embed_tokens: zml.Tensor, // {voc,d} bf16 NON quantifié (D9) — gather embed ET head tied
    final_norm: zml.Tensor, // {d}
    layers: []U6LayW, // 48
    vs: []w4.W4Lin, // 40 — sliding SEULEMENT, indexé slidingSlot (piège revue U6 (b))

    fn init(allocator: std.mem.Allocator, base: zml.io.TensorStore.View) !U7W {
        const layers = try allocator.alloc(U6LayW, N12);
        const vs = try allocator.alloc(w4.W4Lin, N12_SLIDING);
        const lb = base.withPrefix("layers");
        var vi: usize = 0;
        inline for (0..N12) |i| {
            layers[i] = U6LayW.init(lb.withLayer(i));
            if (comptime !g12.g12.isFull(i)) {
                // piège (b) : l'ordre d'ajout DOIT coïncider avec slidingSlot(i) — asserté.
                const slot_i = comptime blk: {
                    @setEvalBranchQuota(100_000); // slidingSlot = boucle comptime O(i) x 48 couches
                    break :blk @as(usize, @intCast(g12.g12.slidingSlot(i)));
                };
                if (vi != slot_i) return error.GateFailed;
                vs[vi] = .init(lb.withLayer(i).withPrefix("self_attn"), "v_proj");
                vi += 1;
            }
        }
        if (vi != N12_SLIDING) return error.GateFailed; // piège (a) : 40 exactement
        return .{
            .embed_tokens = base.createTensor("embed_tokens.weight", .{ .voc, .d }, null),
            .final_norm = base.createTensor("norm.weight", .{.d}, null),
            .layers = layers,
            .vs = vs,
        };
    }
};

/// Assemblage par VALEUR d'un engine.LayerW (généralise u6LayerW aux 48 couches, avec les
/// deux corrections de la revue U6) : full -> placeholder v_proj [1] (D4, inconsommable) ;
/// sliding -> vs[slidingSlot(j)] (JAMAIS vs[j] : entrelacement tous les 6, dès L6 vs[j] brut
/// lirait la mauvaise arête EN SILENCE — le v_proj « numéro 6 » appartient à la couche 7).
fn u7LayerW(w: U7W, comptime j: usize) engine.LayerW {
    const lw = w.layers[j];
    return .{
        .input_layernorm = lw.input_layernorm,
        .q_proj = lw.q.toW(.{ .o, .d }),
        .q_norm = lw.q_norm,
        .k_proj = lw.k.toW(.{ .o, .d }),
        .k_norm = lw.k_norm,
        .v_proj = if (comptime g12.g12.isFull(j))
            lw.layer_scalar // placeholder [1] (D4) — toute consommation casse au compile
        else
            w.vs[comptime blk: {
                @setEvalBranchQuota(100_000); // slidingSlot = boucle comptime O(j) x 48 couches
                break :blk @as(usize, @intCast(g12.g12.slidingSlot(j)));
            }].toW(.{ .o, .d }),
        .o_proj = lw.o.toW(.{ .d, .m }),
        .post_attention_layernorm = lw.post_attention_layernorm,
        .pre_feedforward_layernorm = lw.pre_feedforward_layernorm,
        .gate_proj = lw.gate.toW(.{ .f, .d }),
        .up_proj = lw.up.toW(.{ .f, .d }),
        .down_proj = lw.down.toW(.{ .d, .f }),
        .post_feedforward_layernorm = lw.post_feedforward_layernorm,
        .per_layer_input_gate = lw.layer_scalar, // placeholders [1] — bloc PLE comptime-mort
        .per_layer_projection = lw.layer_scalar,
        .post_per_layer_input_norm = lw.layer_scalar,
        .layer_scalar = lw.layer_scalar,
    };
}

/// VOIE D'EXÉCUTION RETENUE (itération 4 — le repli PRESCRIT PAR LE PLAN, Task 7 : « repli
/// pattern `forwardStageGen` chunké si mur RAM ») : stage-major CHUNKÉ, 8 chunks de 6
/// couches alignés full_period (chaque chunk = 5 sliding + 1 full ; L0-5, L6-11, …, L42-47),
/// soit 9 compiles au total (embed inclus). Le DERNIER chunk (last=true) émet le chemin head
/// du moteur (final_norm + lm_head tied sur embed_tokens + softcapPrec tanh 30) — sa 1re
/// sortie est alors les LOGITS {b,s,voc}.
///
/// Historique des voies (3 runs crashés, causes DIAGNOSTIQUÉES) :
///   - stage-major K=1 (48 compiles, pattern u6) : segfault REPRODUCTIBLE au 31e compileFn
///     du process (runs 1 et 3, crash à la compile du stage L29 = 31e compile embed inclus,
///     memcpy dans l'émission MLIR de manualRope) — épuisement d'une ressource interne
///     zml/pjrt par compile (PAS la RAM : swap 30 Go quasi vide au crash) ; la réduction
///     48x de la trace par compile (itération 1) n'y change rien. Run 2 : variante à frame
///     géante, stack overflow distinct (corrigé — ulimit 64 Mo conservé par prudence).
///   - graphe complet 1 compile (run 4) : compile OK et 20/25 steps OK, puis OOM-kill
///     kernel (anon-rss 23,6 Go, total-vm 114 Go) — le « mur RAM » anticipé par le plan :
///     l'exécutable 48-couches matérialise trop de dequants f32 simultanés par call.
/// Le chunké borne les DEUX ressources : 9 compiles << 31, transients par call ~6 couches.
const U7_CHUNK: usize = g12.g12.full_period; // 6 — aligné : 1 couche full par chunk
const N12_CHUNKS: usize = N12 / U7_CHUNK; // 8
fn U7Chunk(comptime ci: usize) type {
    return struct {
        pub fn forward(w: U7W, hidden_in: zml.Tensor, p: engine.Packed(false), cache: engine.Cache, ctrl: engine.Ctrl) struct { zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor } {
            @setEvalBranchQuota(1_000_000); // slidingSlot comptime x couches du chunk
            const start = ci * U7_CHUNK;
            const end = start + U7_CHUNK;
            var layers: [N12]engine.LayerW = undefined;
            // Seules les couches DU CHUNK sont assemblées (leurs dequants émis) ; les slots
            // hors chunk — JAMAIS lus par forwardStageGen(start, end) — reçoivent la LayerW
            // de la 1re couche du chunk (aucune émission supplémentaire, mêmes Tensors).
            inline for (start..end) |j| layers[j] = u7LayerW(w, j);
            inline for (0..N12) |j| {
                if (comptime (j < start or j >= end)) layers[j] = layers[start];
            }
            const m: U7Model = .{
                .embed_tokens = w.embed_tokens, // consommé par le head tied (last) SEULEMENT
                .per_layer_model_projection = w.layers[0].layer_scalar, // placeholders [1]
                .per_layer_projection_norm = w.layers[0].layer_scalar,
                .final_norm = w.final_norm,
                .layers = &layers,
                .brick = .{},
                .prec = .{}, // baseline f32 (fam=null)
            };
            const out, const slk, const slv, const flk, const flv = m.forwardStageGen(start, end, false, end == N12, p, cache, hidden_in, ctrl);
            return .{ out, slk, slv, flk, flv };
        }
    };
}

/// Top-6 host d'un vecteur de logits (tri par insertion, valeurs strictement décroissantes ;
/// à égalité de valeur le plus petit index gagne — même convention que torch.topk).
fn top6(logits: []const f32) struct { ids: [6]usize, vals: [6]f32 } {
    var ids: [6]usize = @splat(0);
    var vals: [6]f32 = @splat(-std.math.floatMax(f32));
    for (logits, 0..) |v, i| {
        if (v > vals[5]) {
            var r: usize = 5;
            while (r > 0 and v > vals[r - 1]) : (r -= 1) {}
            var m: usize = 5;
            while (m > r) : (m -= 1) {
                vals[m] = vals[m - 1];
                ids[m] = ids[m - 1];
            }
            vals[r] = v;
            ids[r] = i;
        }
    }
    return .{ .ids = ids, .vals = vals };
}

fn gateU7(allocator: std.mem.Allocator, arena: std.mem.Allocator, io: std.Io, platform: *zml.Platform, sharding: zml.sharding.Sharding, ckpt_path: []const u8, fixture_path: []const u8) !void {
    log.info("u7 (U7) — forward 12B COMPLET via le MOTEUR : embed u2 (D12) -> 48 couches (stage-major CHUNKÉ 8x6, Geom.g12 plein) -> head moteur (final_norm + lm_head tied + softcap 30)", .{});
    log.info("  pièges revue U6 corrigés : vs[{d}] (48 - {d} full) + indexation slidingSlot ; verdict §3-U7 requalifié (décision Régis 25 juil) : softcap mordant, top-5 ENSEMBLE + tie |Δ|<1e-4, max_abs documentaire garde-fou 0.5", .{ N12_SLIDING, N12_FULL });

    var st: Stores = undefined;
    try openStores(&st, allocator, io, ckpt_path, fixture_path);
    defer st.deinit();

    // --- Poids packés du modèle COMPLET (checkpoint, D9) ---
    const base = st.store_ck.view().withPrefix("model").withPrefix("language_model");
    const u7w: U7W = try .init(arena, base);
    if (u7w.embed_tokens.dim(.voc) != VOC_12B or u7w.embed_tokens.dim(.d) != D_12B or u7w.embed_tokens.dtype() != .bf16) {
        log.err("u7 : embed_tokens inattendu {d}x{d} {s}", .{ u7w.embed_tokens.dim(.voc), u7w.embed_tokens.dim(.d), @tagName(u7w.embed_tokens.dtype()) });
        return error.GateFailed;
    }
    log.info("  chargement des poids packés 48 couches + 40 v_proj + embed 2 Go + final_norm ...", .{});
    const t_all: std.Io.Timestamp = .now(io, .awake); // pattern chrono prouvé gen_auto (.untilNow)
    const u7w_buf = try zml.io.load(U7W, &u7w, arena, io, platform, &st.store_ck, .{ .shardings = &.{sharding}, .parallelism = load_opts.parallelism, .dma_chunks = load_opts.dma_chunks, .dma_chunk_size = load_opts.dma_chunk_size });
    log.info("  poids chargés en {d} ms", .{@divTrunc(t_all.untilNow(io, .awake).toNanoseconds(), std.time.ns_per_ms)});

    // --- Fixture (script 68) : ids + logits dernière position + top-5 HF ---
    const ids_sym: OneT = .{ .t = st.store_fx.view().createTensor("ids", .{.s}, null) };
    const s_i: i64 = ids_sym.t.dim(.s);
    if (s_i < 4 or s_i > 512 or ids_sym.t.dtype() != .i32) {
        log.err("u7 : fixture ids incohérente (S={d}, {s})", .{ s_i, @tagName(ids_sym.t.dtype()) });
        return error.VacuousGate;
    }
    const logits_sym: OneT = .{ .t = st.store_fx.view().createTensor("logits", .{.voc}, null) };
    const t5i_sym: OneT = .{ .t = st.store_fx.view().createTensor("top5_ids", .{.k}, null) };
    const t5v_sym: OneT = .{ .t = st.store_fx.view().createTensor("top5_vals", .{.k}, null) };
    if (logits_sym.t.dim(.voc) != VOC_12B or t5i_sym.t.dim(.k) != 5 or t5v_sym.t.dim(.k) != 5) {
        log.err("u7 : fixture logits/top5 incohérente", .{});
        return error.VacuousGate;
    }
    const ids_buf = try zml.io.load(OneT, &ids_sym, arena, io, platform, &st.store_fx, .{ .shardings = &.{sharding}, .parallelism = load_opts.parallelism, .dma_chunks = load_opts.dma_chunks, .dma_chunk_size = load_opts.dma_chunk_size });
    const logits_fx_buf = try zml.io.load(OneT, &logits_sym, arena, io, platform, &st.store_fx, .{ .shardings = &.{sharding}, .parallelism = load_opts.parallelism, .dma_chunks = load_opts.dma_chunks, .dma_chunk_size = load_opts.dma_chunk_size });
    const t5i_buf = try zml.io.load(OneT, &t5i_sym, arena, io, platform, &st.store_fx, .{ .shardings = &.{sharding}, .parallelism = load_opts.parallelism, .dma_chunks = load_opts.dma_chunks, .dma_chunk_size = load_opts.dma_chunk_size });
    const t5v_buf = try zml.io.load(OneT, &t5v_sym, arena, io, platform, &st.store_fx, .{ .shardings = &.{sharding}, .parallelism = load_opts.parallelism, .dma_chunks = load_opts.dma_chunks, .dma_chunk_size = load_opts.dma_chunk_size });

    const su: usize = @intCast(s_i);
    const d_us: usize = @intCast(D_12B);
    const voc_us: usize = @intCast(VOC_12B);
    log.info("  prompt fixture : S={d} (prompt canonique templaté, oracle 68)", .{su});

    // --- Embed : chemin 12B du mode u2 (gather + scale bf16 62.0, D12), graphe DÉDIÉ ---
    var emb_exe = try platform.compileFn(allocator, io, U2.forward, .{ u7w.embed_tokens, ids_sym.t }, .{ .shardings = &.{sharding} });
    var emb_args = try emb_exe.args(allocator);
    var emb_results = try emb_exe.results(allocator);
    emb_args.set(.{ u7w_buf.embed_tokens, ids_buf.t });
    emb_exe.call(emb_args, &emb_results);
    var r_emb = emb_results.get(zml.Buffer);
    var emb_s = try r_emb.toSliceAlloc(allocator, io);
    const emb_bf16 = emb_s.items(u16);
    const n_all = su * d_us;
    if (emb_bf16.len != n_all) {
        log.err("u7 : embed — {d} bf16 != {d}", .{ emb_bf16.len, n_all });
        return error.GateFailed;
    }
    const hid_host = try arena.alloc(f32, n_all);
    for (emb_bf16, 0..) |hu, i| hid_host[i] = @bitCast(@as(u32, hu) << 16); // bf16 -> f32 EXACT
    emb_s.free(allocator);
    r_emb.deinit();
    emb_results.deinit(allocator);
    emb_args.deinit(allocator);
    emb_exe.deinit();
    log.info("  embed 12B (gather + scale bf16 62.0) : {d} x {d} f32 (chemin u2, D12)", .{ su, d_us });

    // --- Tables host Packed (pattern u6, S runtime) ---
    const hd_f_us: usize = @intCast(HD_F);
    const cos_host = try arena.alloc(f32, su * hd_f_us);
    const sin_host = try arena.alloc(f32, su * hd_f_us);
    for (0..su) |p| ropeFullHost(@intCast(p), cos_host[p * hd_f_us .. (p + 1) * hd_f_us], sin_host[p * hd_f_us .. (p + 1) * hd_f_us]);
    // Masque causal UNIQUE (two_masks=false) — licite : l'oracle 68 ASSERTE sliding == causal à ce S.
    const mask_host = try arena.alloc(f32, su * su);
    for (0..su) |p| {
        for (0..su) |k| mask_host[p * su + k] = if (k <= p) 0 else MASK_MIN;
    }
    const pos_host = try arena.alloc(i32, su);
    for (pos_host, 0..) |*x, p| x.* = @intCast(p);
    const emb_zero = try arena.alloc(u8, su * d_us * 2); // bf16 zéros — NON consommé (first=false)
    @memset(emb_zero, 0);
    const eptl_zero = try arena.alloc(u8, su * 2); // {S,1,1,1} bf16 — NON consommé (ple_dim=0)
    @memset(eptl_zero, 0);

    // Cache mock zéro : slots PLEINS du Geom g12 (40 sliding + 8 full), KMAX=S, f32.
    const sl_slots: i64 = @intCast(N12_SLIDING);
    const fl_slots: i64 = @intCast(N12_FULL);
    const kvh_sl: i64 = @intCast(g12.g12.kvh_sliding);
    const kvh_fl: i64 = @intCast(g12.g12.kvh_full);
    const sl_bytes = @as(usize, @intCast(sl_slots)) * @as(usize, @intCast(kvh_sl)) * su * @as(usize, @intCast(HD_S)) * 4;
    const fl_bytes = @as(usize, @intCast(fl_slots)) * @as(usize, @intCast(kvh_fl)) * su * hd_f_us * 4;
    const cache_sl_zero = try arena.alloc(u8, sl_bytes);
    @memset(cache_sl_zero, 0);
    const cache_fl_zero = try arena.alloc(u8, fl_bytes);
    @memset(cache_fl_zero, 0);

    // --- Symboliques (pattern u6, S runtime) ---
    const packed_sym = engine.Packed(false){
        .embeds = zml.Tensor.init(.{ s_i, 1, 1, D_12B }, .bf16).withTags(.{ .step, .b, .s, .d }),
        .embptls = zml.Tensor.init(.{ s_i, 1, 1, 1 }, .bf16).withTags(.{ .step, .b, .s, .lf }),
        .cos_full = zml.Tensor.init(.{ s_i, 1, 1, HD_F }, .f32).withTags(.{ .step, .b, .s, .hd }),
        .sin_full = zml.Tensor.init(.{ s_i, 1, 1, HD_F }, .f32).withTags(.{ .step, .b, .s, .hd }),
        .masks = zml.Tensor.init(.{ s_i, 1, 1, 1, s_i }, .f32).withTags(.{ .step, .b, .h, .q, .k }),
        .positions = zml.Tensor.init(.{s_i}, .i32).withTags(.{.step}),
    };
    const cache_sym = engine.Cache{
        .sl_k = zml.Tensor.init(.{ sl_slots, 1, kvh_sl, s_i, HD_S }, .f32).withTags(.{ .slot, .b, .h, .k, .hd }),
        .sl_v = zml.Tensor.init(.{ sl_slots, 1, kvh_sl, s_i, HD_S }, .f32).withTags(.{ .slot, .b, .h, .k, .hd }),
        .fl_k = zml.Tensor.init(.{ fl_slots, 1, kvh_fl, s_i, HD_F }, .f32).withTags(.{ .slot, .b, .h, .k, .hd }),
        .fl_v = zml.Tensor.init(.{ fl_slots, 1, kvh_fl, s_i, HD_F }, .f32).withTags(.{ .slot, .b, .h, .k, .hd }),
    };
    const ctrl_sym: engine.Ctrl = .initSymbolic();
    const hidden_sym = zml.Tensor.init(.{ 1, 1, D_12B }, .f32).withTags(.{ .b, .s, .d });
    try cache_sym.checkDtype(.{}); // prec par défaut => cache f32 attendu — témoin

    const pk_buf = zml.Bufferized(engine.Packed(false)){
        .embeds = try zml.Buffer.fromBytes(io, platform, packed_sym.embeds.shape(), sharding, emb_zero),
        .embptls = try zml.Buffer.fromBytes(io, platform, packed_sym.embptls.shape(), sharding, eptl_zero),
        .cos_full = try zml.Buffer.fromBytes(io, platform, packed_sym.cos_full.shape(), sharding, std.mem.sliceAsBytes(cos_host)),
        .sin_full = try zml.Buffer.fromBytes(io, platform, packed_sym.sin_full.shape(), sharding, std.mem.sliceAsBytes(sin_host)),
        .masks = try zml.Buffer.fromBytes(io, platform, packed_sym.masks.shape(), sharding, std.mem.sliceAsBytes(mask_host)),
        .positions = try zml.Buffer.fromBytes(io, platform, packed_sym.positions.shape(), sharding, std.mem.sliceAsBytes(pos_host)),
    };
    var cache_buf = zml.Bufferized(engine.Cache){
        .sl_k = try zml.Buffer.fromBytes(io, platform, cache_sym.sl_k.shape(), sharding, cache_sl_zero),
        .sl_v = try zml.Buffer.fromBytes(io, platform, cache_sym.sl_v.shape(), sharding, cache_sl_zero),
        .fl_k = try zml.Buffer.fromBytes(io, platform, cache_sym.fl_k.shape(), sharding, cache_fl_zero),
        .fl_v = try zml.Buffer.fromBytes(io, platform, cache_sym.fl_v.shape(), sharding, cache_fl_zero),
    };

    // --- STAGE-MAJOR CHUNKÉ (voie retenue, itération 4 — le repli PRESCRIT PAR LE PLAN
    // « repli pattern forwardStageGen chunké si mur RAM », cf. en-tête U7Chunk pour
    // l'historique des 3 runs crashés) : 8 chunks de 6 couches, S steps par chunk, chaîne
    // hidden via host entre chunks (pattern u6), logits au (dernier chunk, dernier step). ---
    const logits_zml = try arena.alloc(f32, voc_us);
    const hid_next = try arena.alloc(f32, n_all);
    inline for (0..N12_CHUNKS) |ci| {
        const is_last_chunk = comptime (ci == N12_CHUNKS - 1);
        const t_c0: std.Io.Timestamp = .now(io, .awake);
        var exe = try platform.compileFn(allocator, io, U7Chunk(ci).forward, .{ u7w, hidden_sym, packed_sym, cache_sym, ctrl_sym }, .{ .shardings = &.{sharding} });
        defer exe.deinit();
        const t_compile_ms = @divTrunc(t_c0.untilNow(io, .awake).toNanoseconds(), std.time.ns_per_ms);
        const t_r0: std.Io.Timestamp = .now(io, .awake);
        for (0..su) |step| {
            var hidden_buf = try zml.Buffer.fromBytes(io, platform, hidden_sym.shape(), sharding, std.mem.sliceAsBytes(hid_host[step * d_us .. (step + 1) * d_us]));
            var step_buf = try zml.Buffer.scalar(io, platform, @as(u32, @intCast(step)), .u32, sharding);
            const ctrl_buf = zml.Bufferized(engine.Ctrl){ .step = step_buf };

            var call_args = try exe.args(allocator);
            var call_results = try exe.results(allocator);
            call_args.set(.{ u7w_buf, hidden_buf, pk_buf, cache_buf, ctrl_buf });
            exe.call(call_args, &call_results);
            var r_out, const r_slk, const r_slv, const r_flk, const r_flv = call_results.get(struct { zml.Buffer, zml.Buffer, zml.Buffer, zml.Buffer, zml.Buffer });

            var old_cache = cache_buf;
            cache_buf = zml.Bufferized(engine.Cache){ .sl_k = r_slk, .sl_v = r_slv, .fl_k = r_flk, .fl_v = r_flv };
            old_cache.sl_k.deinit();
            old_cache.sl_v.deinit();
            old_cache.fl_k.deinit();
            old_cache.fl_v.deinit();

            if (comptime !is_last_chunk) {
                // Sortie du chunk {1,1,d} : hidden du chunk suivant (chaîne via host, f32).
                var got_s = try r_out.toSliceAlloc(allocator, io);
                defer got_s.free(allocator);
                const got = got_s.items(f32);
                if (got.len != d_us) {
                    log.err("u7 : chunk {d} step {d} — {d} f32 != {d}", .{ ci, step, got.len, d_us });
                    return error.GateFailed;
                }
                @memcpy(hid_next[step * d_us .. (step + 1) * d_us], got);
            } else if (step == su - 1) {
                // DERNIER chunk, DERNIÈRE position : logits {1,1,voc} du chemin head moteur
                // (final_norm + lm_head tied + softcap) — la seule sortie comparée (§3-U7).
                var got_s = try r_out.toSliceAlloc(allocator, io);
                defer got_s.free(allocator);
                const got = got_s.items(f32);
                if (got.len != voc_us) {
                    log.err("u7 : logits — {d} f32 != {d}", .{ got.len, voc_us });
                    return error.GateFailed;
                }
                @memcpy(logits_zml, got);
            }

            r_out.deinit();
            hidden_buf.deinit();
            step_buf.deinit();
            call_args.deinit(allocator);
            call_results.deinit(allocator);
        }
        if (comptime !is_last_chunk) @memcpy(hid_host, hid_next);
        log.info("  chunk {d}/{d} (L{d}-L{d}, 5 sliding + 1 full K=V branche D4) : compile {d} ms, {d} steps en {d} ms{s}", .{ ci, N12_CHUNKS - 1, ci * U7_CHUNK, ci * U7_CHUNK + U7_CHUNK - 1, t_compile_ms, su, @divTrunc(t_r0.untilNow(io, .awake).toNanoseconds(), std.time.ns_per_ms), if (is_last_chunk) " — HEAD moteur (final_norm+lm_head tied+softcap)" else "" });
    }
    cache_buf.sl_k.deinit();
    cache_buf.sl_v.deinit();
    cache_buf.fl_k.deinit();
    cache_buf.fl_v.deinit();
    log.info("  forward complet : {d} chunks x {d} steps en {d} s au total depuis le chargement (stage-major chunké 6 couches, repli prescrit du plan — voie déclarée)", .{ N12_CHUNKS, su, @divTrunc(t_all.untilNow(io, .awake).toNanoseconds(), std.time.ns_per_s) });

    // --- Expected host : logits HF + top-5 HF (cohérence interne assertée) ---
    var exp_s = try logits_fx_buf.t.toSliceAlloc(allocator, io);
    defer exp_s.free(allocator);
    const logits_hf = exp_s.items(f32);
    var t5i_s = try t5i_buf.t.toSliceAlloc(allocator, io);
    defer t5i_s.free(allocator);
    const hf_top5_ids = t5i_s.items(i32);
    var t5v_s = try t5v_buf.t.toSliceAlloc(allocator, io);
    defer t5v_s.free(allocator);
    const hf_top5_vals = t5v_s.items(f32);
    if (logits_hf.len != voc_us) {
        log.err("u7 : logits HF — {d} f32 != {d}", .{ logits_hf.len, voc_us });
        return error.VacuousGate;
    }
    for (hf_top5_ids, hf_top5_vals) |ti, tv| {
        if (logits_hf[@intCast(ti)] != tv) {
            log.err("u7 : fixture incohérente — top5_vals[{d}]={e:.6} != logits[{d}]", .{ ti, tv, ti });
            return error.VacuousGate;
        }
    }

    // --- Softcap exercé côté ZML (§3-U7) : max|logits| <= 30 ET > 25 quelque part ---
    var max_abs_logit: f64 = 0;
    var n_over_25: usize = 0;
    for (logits_zml) |v| {
        const a = @abs(@as(f64, v));
        if (a > max_abs_logit) max_abs_logit = a;
        if (a > U7_SOFTCAP_BITE) n_over_25 += 1;
    }
    if (max_abs_logit > U7_SOFTCAP) {
        log.err("u7 : softcap VIOLÉ côté ZML — max|logits|={e:.4} > {d}", .{ max_abs_logit, U7_SOFTCAP });
        return error.GateFailed;
    }
    if (n_over_25 == 0) {
        log.err("u7 : softcap ne MORD pas côté ZML — aucun |logit| > {d} (max={e:.4})", .{ U7_SOFTCAP_BITE, max_abs_logit });
        return error.VacuousGate;
    }
    log.info("  softcap ZML exercé : max|logits|={d:.4} <= 30, {d} logits > 25 en |.|", .{ max_abs_logit, n_over_25 });

    // --- Marges lues AVANT tout verdict (piège 17) ---
    const tz = top6(logits_zml);
    const th = top6(logits_hf);
    log.info("  top-6 ZML : ids={any} vals={any}", .{ tz.ids, tz.vals });
    log.info("  top-6 HF  : ids={any} vals={any}", .{ th.ids, th.vals });
    log.info("  marges ZML : top1-top2={e:.6}, rang5-rang6={e:.6} ; HF : top1-top2={e:.6}, rang5-rang6={e:.6}", .{ tz.vals[0] - tz.vals[1], tz.vals[4] - tz.vals[5], th.vals[0] - th.vals[1], th.vals[4] - th.vals[5] });

    // --- max_abs logits dernière position : DOCUMENTAIRE, garde-fou 0.5 (requalifié) ---
    var max_abs: f64 = 0;
    var max_i: usize = 0;
    var sum_abs: f64 = 0;
    for (logits_zml, logits_hf, 0..) |gv, e, i| {
        const dd = @abs(@as(f64, gv) - @as(f64, e));
        sum_abs += dd;
        if (dd > max_abs) {
            max_abs = dd;
            max_i = i;
        }
    }
    const mean_abs = sum_abs / @as(f64, @floatFromInt(voc_us));
    log.info("  logits dernière position : max_abs={e:.3} (id {d} : zml={e:.6} hf={e:.6}) mean_abs={e:.3}", .{ max_abs, max_i, logits_zml[max_i], logits_hf[max_i], mean_abs });

    // --- Top-5 en ENSEMBLE + tie rule (§3-U7, piège 17) : rang par rang, un écart de rang
    // est toléré ssi |Δlogit| < 1e-4 entre les deux candidats, mesuré sur les logits HF (la
    // référence — décision d'interprétation consignée) ; l'écart côté ZML est loggé. ---
    var n_rank_diff: usize = 0;
    var tie_ok = true;
    for (0..5) |r| {
        const zi = tz.ids[r];
        const hi: usize = @intCast(hf_top5_ids[r]);
        if (zi != hi) {
            n_rank_diff += 1;
            const d_hf = @abs(@as(f64, logits_hf[hi]) - @as(f64, logits_hf[zi]));
            const d_zml = @abs(@as(f64, logits_zml[hi]) - @as(f64, logits_zml[zi]));
            if (d_hf < U7_TIE) {
                log.warn("  rang {d} : ZML id {d} != HF id {d} — TIE toléré (|Δlogit| HF={e:.3} < 1e-4 ; ZML={e:.3})", .{ r, zi, hi, d_hf, d_zml });
            } else {
                log.err("  rang {d} : ZML id {d} != HF id {d} — |Δlogit| HF={e:.3} >= 1e-4 (ZML={e:.3}) : PAS un tie", .{ r, zi, hi, d_hf, d_zml });
                tie_ok = false;
            }
        }
    }
    // Ensemble : chaque id du top-5 HF doit être dans le top-5 ZML (et réciproquement),
    // sauf tie de frontière déjà couvert par la règle rang-par-rang ci-dessus.
    var set_eq = true;
    for (0..5) |r| {
        const hi: usize = @intCast(hf_top5_ids[r]);
        var found = false;
        for (0..5) |q| {
            if (tz.ids[q] == hi) found = true;
        }
        if (!found) set_eq = false;
    }

    if (!tie_ok) {
        log.err("u7 : FAIL top-5 — écart de rang hors tie rule (§3-U7)", .{});
        return error.GateFailed;
    }
    if (max_abs > U7_MAX_ABS) {
        log.err("u7 : FAIL max_abs={e:.3} > garde-fou {e:.1} (Amendement 2 §U7 requalifié — le garde-fou documentaire reste CÂBLÉ, un dépassement = FAIL ; top-5 ensemble={}, écarts de rang={d})", .{ max_abs, U7_MAX_ABS, set_eq, n_rank_diff });
        return error.GateFailed;
    }
    log.info("PASS u7 — prefill 12B 48 couches + head moteur : softcap mordant (max|logits|={d:.4}, {d} > 25), top-5 ensemble={} (écarts de rang tolérés par tie : {d}), max_abs documentaire={e:.3} <= garde-fou 0.5 (bf16-réel requalifié, décision Régis 25 juil)", .{ max_abs_logit, n_over_25, set_eq, n_rank_diff, max_abs });
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
    } else if (std.mem.eql(u8, mode, "u5")) {
        if (process_args.len < 4) {
            log.err("{s}", .{usage});
            return error.MissingArgument;
        }
        try gateU5(allocator, arena.allocator(), io, platform, sharding, process_args[2], process_args[3]);
    } else if (std.mem.eql(u8, mode, "u6")) {
        if (process_args.len < 4) {
            log.err("{s}", .{usage});
            return error.MissingArgument;
        }
        try gateU6(allocator, arena.allocator(), io, platform, sharding, process_args[2], process_args[3]);
    } else if (std.mem.eql(u8, mode, "u7")) {
        if (process_args.len < 4) {
            log.err("{s}", .{usage});
            return error.MissingArgument;
        }
        try gateU7(allocator, arena.allocator(), io, platform, sharding, process_args[2], process_args[3]);
    } else {
        log.err("mode inconnu '{s}' — {s}", .{ mode, usage });
        return error.MissingArgument;
    }
}
