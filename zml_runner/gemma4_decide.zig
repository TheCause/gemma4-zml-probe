// SD — runner de DÉCISION typée par lecture des logits (spec
// docs/superpowers/specs/2026-09-24-decision-layer-design.md, plan docs/superpowers/plans/2026-09-24-decision-layer.md).
// Clone ciblé de gemma4_gen_auto.zig (duplication assumée, précédent gemma4_w4auto) : engine.zig et
// gemma4_gen_auto.zig INTOUCHÉS (C-SD-C). Entrée compilée : `StepDec.forward` = gather + forwardStep
// (inchangé) + topK(5) + gather des 4 candidats + logSumExp. Ids lus dans le manifest (BOS inclus :
// RIEN n'est préfixé ici). Évaluateur : sd_policy.zig (pur).
//
// CLI : gemma4_decide <model.safetensors> --manifest f --arm {letter,json} --reps R --out f.jsonl
//       [--order {fwd,both}] [--policy-selftest] [--allow-cpu] [--force-vram]
//
// Copies VERBATIM de gemma4_gen_auto.zig @ main : l.31-51 (imports, constantes, Model, PackedLong ;
// SANS BOS_ID), l.156-234 (MASK_MIN, ROPE_FULL_*, ropeFull, maskRows), l.236-327 (HostInputs),
// l.466-491 (Tabs, clés), l.664-736 (parseFreeMiB, checkVram), l.899-989 sauf 919-928 (dans main).
const std = @import("std");
const log = std.log;
const zml = @import("zml");
const engine = @import("engine.zig");
const mem_probe = @import("mem_probe.zig");

pub const std_options: std.Options = .{ .log_level = .info };

const L_MAX: i64 = 1024;
const SLIDING_WINDOW: i64 = 512;
const HD_F: i64 = 512; // dim cos/sin full (= config.global_head_dim, cf ropeFull)
const HD_S: i64 = 256; // dim cache sliding (= engine.HD_SLIDING)
const D: i64 = 1536;
const LF: i64 = 8960;
// Slots de cache producteurs (cf engine.zig: isFull(i)=(i+1)%5==0, FIRST_KV_SHARED=15) : parmi les
// 15 premières couches, 3 sont "full" (4,9,14) et 12 "sliding" (les autres) — mêmes comptes que
// SLIDING_PRODUCERS/FULL_PRODUCERS de scripts/49_gen_custom_oracle.py:41-42.
const NUM_SLIDING_SLOTS: usize = 12;
const NUM_FULL_SLOTS: usize = 3;
const Model = engine.EngineModel(struct {}, .{ .two_masks = true, .kmax_sliding = L_MAX, .kmax_full = L_MAX });
const PackedLong = engine.Packed(.tables);
const builtin = @import("builtin");
const policy = @import("sd_policy.zig");

// ============================================================================================
// Copie verbatim gemma4_gen_auto.zig:156-234.
// ============================================================================================
// Masques additifs f32 : 0 = visible, -floatMax = masqué (== torch.finfo(float32).min — même
// valeur binaire : le plus grand f32 fini, négé, des deux côtés).
const MASK_MIN: f32 = -std.math.floatMax(f32);

// Coefficients RoPE "proportional" (couches full_attention) — formule COPIÉE de
// transformers/modeling_rope_utils.py::_compute_proportional_rope_parameters (lue sur la 3090,
// 10 juil, cf commit) :
//   head_dim = config.global_head_dim = 512 (head_dim_key="global_head_dim" pour full_attention,
//     modeling_gemma4.py:1098-1099) ; base = rope_theta = 1e6 ; rope_proportion =
//     partial_rotary_factor = 0.25 ; factor = rope_parameters_dict.get("factor", 1.0) = 1.0
//     (absent de rope_scaling.full_attention, confirmé AutoConfig) ;
//   rope_angles = int(rope_proportion * head_dim // 2) = int(0.25*512 // 2) = int(128.0//2) = 64 ;
//   inv_freq_rotated[i] = 1 / base**(arange(0,2*rope_angles,2)[i] / head_dim)
//                       = 1 / base**((2*i)/head_dim)  pour i in 0..rope_angles (64 valeurs) ;
//   nope_angles = head_dim//2 - rope_angles = 256-64 = 192 ;
//   inv_freq = concat(inv_freq_rotated, zeros(nope_angles)) → 256 valeurs (head_dim//2) ;
//   inv_freq /= factor (no-op, factor=1.0).
// forward() (modeling_gemma4.py:1141-1152) : freqs[i] = inv_freq[i] * p (i in 0..256) ;
//   emb = concat(freqs, freqs) — DUPLICATION DE LA MOITIÉ (pas d'entrelacement) → 512 valeurs ;
//   cos = cos(emb) * attention_scaling ; sin = sin(emb) * attention_scaling
//   (attention_scaling = 1.0 pour "proportional" — "Unused in this type of RoPE" — no-op).
const ROPE_FULL_THETA: f32 = 1_000_000.0;
const ROPE_FULL_HEAD_DIM: f32 = 512.0; // = HD_F
const ROPE_FULL_ANGLES: usize = 64; // rope_angles
const ROPE_FULL_HALF: usize = 256; // head_dim // 2 (= HD_F / 2)

// cos/sin full pour la position p — formule ci-dessus (partial 0.25, proportional, theta 1e6).
// PRÉCISION — investigation mesurée (10 juil, fixtures courte p≤68 ET longue p≤1023, cf commits) :
// la transcription EST correcte (vérifiée valeur par valeur contre la fixture ET contre numpy),
// mais un résidu subsiste sur quelques (position, indice de fréquence) précis, QUELLE QUE SOIT la
// stratégie de calcul essayée (f64 bout-en-bout arrondi seulement à la fin ; f32 via
// `std.math.pow` ; f32 via `@exp2(exp*@log2(base))` — même ordre de grandeur les trois fois).
// Root cause : `pow()` de Zig (LLVM/libm, CPU) et de PyTorch arrondissent chacun CORRECTEMENT
// mais PAS IDENTIQUEMENT `base**exp` — 1 ULP d'écart sur certains inv_freq[i] (confirmé
// bit-à-bit : inv_freq[12]≈0.523, bits 3f05f6ee vs 3f05f6ef ; le calcul en précision arbitraire
// montre qu'aucune des deux valeurs n'est "la bonne", les deux sont à ~2 ULP du réel).
// AMPLIFICATION LINÉAIRE EN p (mesurée sur la fixture longue) : l'erreur d'angle vaut
//   Δangle ≈ Δinv_freq×p + arrondi f32 du produit inv_freq×p (±ULP à l'échelle de l'angle,
//   elle-même ∝ p puisque angle ≈ inv_freq×p) ≈ 2 ULP × p × ~6e-8 ≈ 1.2e-7 × p,
// que sin/cos propagent à pente ≤ 1. Vérifié numériquement au pire point mesuré (k=587, p=612,
// i=12, sin) : Δinv_freq = 1 ULP = 5.96e-8 → Δangle = 6.10e-5 (= 3.65e-5 de Δinv×p + arrondis de
// produit opposés, ULP(320 rad) = 3.05e-5), × pente |cos|≈0.983 → 6.00e-5 observé — REPRODUIT
// exactement par numpy f32 sur les deux angles candidats (borne 2-ULP : 7.3e-5, cohérente). Aux
// positions courtes (p≤68) le même mécanisme donnait le plancher 2^-18 = 3.81e-6. C'est un
// plancher de précision float32 inter-implémentations (même famille de piège que "pas de
// bit-à-bit inter-compiles XLA-GPU", mémoire ZML) — PAS une erreur de formule. `std.math.pow`
// est gardé (le plus standard, pas de bricolage ad hoc) ; le SELFTEST compare avec une tolérance
// DÉPENDANTE DE LA POSITION (cf `cosSinTol`), pas une constante.
fn ropeFull(p: i64, cos_out: *[HD_F]f32, sin_out: *[HD_F]f32) void {
    var inv_freq: [ROPE_FULL_HALF]f32 = undefined;
    for (0..ROPE_FULL_HALF) |i| {
        if (i < ROPE_FULL_ANGLES) {
            const exp: f32 = @as(f32, @floatFromInt(2 * i)) / ROPE_FULL_HEAD_DIM;
            inv_freq[i] = 1.0 / std.math.pow(f32, ROPE_FULL_THETA, exp);
        } else {
            inv_freq[i] = 0.0; // nope_angles : pas de rotation (angle constant nul quel que soit p)
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

// Masques additifs f32 : 0 = visible, -floatMax = masqué (== torch.finfo(float32).min).
fn maskRows(p: i64, sliding_out: []f32, full_out: []f32) void {
    const lo = @max(0, p - (SLIDING_WINDOW - 1));
    for (0..@intCast(L_MAX)) |j| {
        const ji: i64 = @intCast(j);
        sliding_out[j] = if (ji > p or ji < lo) MASK_MIN else 0;
        full_out[j] = if (ji > p) MASK_MIN else 0;
    }
}

// Tables host complètes, indexées par STEP == POSITION ABSOLUE p in 0..L_MAX-1 (identité — cf
// PLAN Task 5 : `ctrl.step` vaudra la position courante, et `pickStep(p.cos_full, step)` ira
// chercher la ligne p). Conçues comme des slices host simples (pas de Buffer/Platform) : Task 5
// les enveloppera avec `zml.Buffer.fromBytes` (mêmes shapes que `engine.Packed`/`engine.Cache`).
const HostInputs = struct {
    cos_full: []f32, // {L_MAX, HD_F}
    sin_full: []f32, // {L_MAX, HD_F}
    masks_sliding: []f32, // {L_MAX, L_MAX}
    masks_full: []f32, // {L_MAX, L_MAX}
    positions: []i32, // {L_MAX} = 0..L_MAX-1
    embeds_zero: []u8, // {L_MAX, 1, 1, D} bf16, zéros — factice, non consommé par forwardStep
    embptls_zero: []u8, // {L_MAX, 1, 1, LF} bf16, zéros — idem
    cache_sl_k: []u8, // {NUM_SLIDING_SLOTS, 1, 1, L_MAX, HD_S} f32, zéros
    cache_sl_v: []u8,
    cache_fl_k: []u8, // {NUM_FULL_SLOTS, 1, 1, L_MAX, HD_F} f32, zéros
    cache_fl_v: []u8,

    fn init(allocator: std.mem.Allocator) !HostInputs {
        const l_max: usize = @intCast(L_MAX);
        const hd_f: usize = @intCast(HD_F);

        const cos_full = try allocator.alloc(f32, l_max * hd_f);
        errdefer allocator.free(cos_full);
        const sin_full = try allocator.alloc(f32, l_max * hd_f);
        errdefer allocator.free(sin_full);
        const masks_sliding = try allocator.alloc(f32, l_max * l_max);
        errdefer allocator.free(masks_sliding);
        const masks_full = try allocator.alloc(f32, l_max * l_max);
        errdefer allocator.free(masks_full);
        const positions = try allocator.alloc(i32, l_max);
        errdefer allocator.free(positions);

        var p: i64 = 0;
        while (p < L_MAX) : (p += 1) {
            const idx: usize = @intCast(p);
            positions[idx] = @intCast(p);
            var cos_row: [HD_F]f32 = undefined;
            var sin_row: [HD_F]f32 = undefined;
            ropeFull(p, &cos_row, &sin_row);
            @memcpy(cos_full[idx * hd_f .. (idx + 1) * hd_f], &cos_row);
            @memcpy(sin_full[idx * hd_f .. (idx + 1) * hd_f], &sin_row);
            maskRows(p, masks_sliding[idx * l_max .. (idx + 1) * l_max], masks_full[idx * l_max .. (idx + 1) * l_max]);
        }

        const embeds_zero = try allocator.alloc(u8, l_max * @as(usize, @intCast(D)) * 2);
        errdefer allocator.free(embeds_zero);
        @memset(embeds_zero, 0);
        const embptls_zero = try allocator.alloc(u8, l_max * @as(usize, @intCast(LF)) * 2);
        errdefer allocator.free(embptls_zero);
        @memset(embptls_zero, 0);
        const cache_sl_k = try allocator.alloc(u8, NUM_SLIDING_SLOTS * l_max * @as(usize, @intCast(HD_S)) * 4);
        errdefer allocator.free(cache_sl_k);
        @memset(cache_sl_k, 0);
        const cache_sl_v = try allocator.alloc(u8, NUM_SLIDING_SLOTS * l_max * @as(usize, @intCast(HD_S)) * 4);
        errdefer allocator.free(cache_sl_v);
        @memset(cache_sl_v, 0);
        const cache_fl_k = try allocator.alloc(u8, NUM_FULL_SLOTS * l_max * hd_f * 4);
        errdefer allocator.free(cache_fl_k);
        @memset(cache_fl_k, 0);
        const cache_fl_v = try allocator.alloc(u8, NUM_FULL_SLOTS * l_max * hd_f * 4);
        errdefer allocator.free(cache_fl_v);
        @memset(cache_fl_v, 0);

        return .{
            .cos_full = cos_full,
            .sin_full = sin_full,
            .masks_sliding = masks_sliding,
            .masks_full = masks_full,
            .positions = positions,
            .embeds_zero = embeds_zero,
            .embptls_zero = embptls_zero,
            .cache_sl_k = cache_sl_k,
            .cache_sl_v = cache_sl_v,
            .cache_fl_k = cache_fl_k,
            .cache_fl_v = cache_fl_v,
        };
    }

    fn deinit(self: *HostInputs, allocator: std.mem.Allocator) void {
        allocator.free(self.cos_full);
        allocator.free(self.sin_full);
        allocator.free(self.masks_sliding);
        allocator.free(self.masks_full);
        allocator.free(self.positions);
        allocator.free(self.embeds_zero);
        allocator.free(self.embptls_zero);
        allocator.free(self.cache_sl_k);
        allocator.free(self.cache_sl_v);
        allocator.free(self.cache_fl_k);
        allocator.free(self.cache_fl_v);
    }
};

// ============================================================================================
// L3 — table `Tabs` (spec docs/L3_INGRAPH_DESIGN.md §2.1) : embed_tokens_per_layer, chargée en
// VRAM par le même TensorStore/zml.io que le reste du modèle. `embed_tokens` est DÉJÀ
// device-résident dans `Model` (lm_head tied, engine.zig:487/507) : le gather du `StepTok`
// plus bas le réutilise, ZÉRO Go ajouté par cette table-ci — `Tabs.eptl` (~4,7 Go bf16) est la
// SEULE table ajoutée au device. `EmbedGather` (Task 4 historique : gather HOST en streaming
// direct sur le fichier checkpoint) est SUPPRIMÉ intégralement — le gather vit désormais dans le
// graphe compilé (`StepTok.forward`).
// ============================================================================================
const EMB_KEY = "model.language_model.embed_tokens.weight"; // utilisé par SgTabs (clé ABSOLUE, root view)
const EPTL_KEY = "model.language_model.embed_tokens_per_layer.weight"; // idem

// Table L3 (spec docs/L3_INGRAPH_DESIGN.md §2.1) : SEULE table ajoutée au device —
// embed_tokens est déjà résident dans Model (lm_head tied, engine.zig:487), le gather le
// réutilise. Nom court OBLIGATOIRE (piège quota comptime @typeName, cf spec §2).
const Tabs = struct {
    eptl: zml.Tensor, // {voc,lf} bf16 BRUT (scaling ×16 déjà dans forwardStep)

    fn init(base: zml.io.TensorStore.View) Tabs {
        return .{ .eptl = base.createTensor("embed_tokens_per_layer.weight", .{ .voc, .lf }, null) };
    }
    fn load(self: *const Tabs, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, store: *const zml.io.TensorStore, shardings: []const zml.sharding.Sharding) !zml.Bufferized(Tabs) {
        return zml.io.load(Tabs, self, allocator, io, platform, store, .{ .shardings = shardings, .parallelism = 1, .dma_chunks = 1, .dma_chunk_size = 16 * 1024 * 1024 });
    }
};

// Garde VRAM au lancement (docs/VRAM_CHECK_DESIGN.md) — incident du 11 juil 2026 : Ollama à
// ~22/24 Go → OOM dès la matérialisation + crash `io.zig deinit` (double-free post-OOM, bug
// d'error-path UPSTREAM ZML, cosmétique — l'OOM est la vraie erreur). Best-effort : la garde ne
// bloque JAMAIS à tort — nvidia-smi absent/cassé/illisible → warn + continue (l'OOM reste le
// filet) ; seul « VRAM libre < seuil » mesuré avec succès fait échouer le lancement.
// ============================================================================================

// Seuil requis — G3 (Step 8, amendement méthode) : `mem_probe` (ci-dessous, "post-load"/
// "post-compile") loggue de la RSS HOST, pas de la VRAM device — et `nvidia-smi` pendant un run
// normal ne montre que la RÉSERVE BFC préallouée (`0.90 × VRAM libre au lancement`), pas le
// besoin réel. Mesure réelle faite en désactivant temporairement `preallocate` (BFC alloue à la
// demande) et en échantillonnant `nvidia-smi --query-compute-apps` pendant tout le run (compile
// + prefill + génération 999 steps, fixture A2) : pic observé = 16658 MiB ≈ 16,27 GiB. Seuil
// final = ceil(pic_GiB / 0.90) + 1 = ceil(16,27 / 0,90) + 1 = ceil(18,08) + 1 = 20 GiB — couvre
// la réserve BFC réelle (0.90×) avec 1 GiB de marge. Pas de flag de réglage (YAGNI).
const MIN_FREE_VRAM_GIB: u64 = 20;

// Parse `nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits` : première ligne =
// GPU 0 (VM mono-GPU), entier en MiB. null = sortie illisible (l'appelant warn + continue).
fn parseFreeMiB(stdout: []const u8) ?u64 {
    var lines = std.mem.splitScalar(u8, stdout, '\n');
    const first = lines.next() orelse return null;
    const trimmed = std.mem.trim(u8, first, " \t\r");
    if (trimmed.len == 0) return null;
    return std.fmt.parseInt(u64, trimmed, 10) catch null;
}

fn checkVram(gpa: std.mem.Allocator, io: std.Io) !void {
    const res = std.process.run(gpa, io, .{
        .argv = &.{ "nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits" },
    }) catch |err| {
        log.warn("garde VRAM sautée : nvidia-smi indisponible ({s}) — machine sans GPU ?", .{@errorName(err)});
        return;
    };
    defer gpa.free(res.stdout);
    defer gpa.free(res.stderr);
    switch (res.term) {
        .exited => |code| if (code != 0) {
            log.warn("garde VRAM sautée : nvidia-smi exit={d}", .{code});
            return;
        },
        else => {
            log.warn("garde VRAM sautée : nvidia-smi terminé anormalement", .{});
            return;
        },
    }
    const free_mib = parseFreeMiB(res.stdout) orelse {
        log.warn("garde VRAM sautée : sortie nvidia-smi illisible", .{});
        return;
    };
    if (free_mib >= MIN_FREE_VRAM_GIB * 1024) return;

    // Une décimale en arithmétique ENTIÈRE (pas de format float : API std.fmt 0.16-dev mouvante).
    const gib10 = free_mib * 10 / 1024;
    log.err("GPU occupé — VRAM libre {d}.{d} GiB < {d} GiB requis", .{ gib10 / 10, gib10 % 10, MIN_FREE_VRAM_GIB });
    // Déviation assumée vs spec §2 : pas de `parseComputeApps` structuré — les lignes CSV brutes
    // trimées suffisent au message (PID, nom, MiB lisibles) et restent best-effort.
    if (std.process.run(gpa, io, .{
        .argv = &.{ "nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader" },
    })) |apps| {
        defer gpa.free(apps.stdout);
        defer gpa.free(apps.stderr);
        var it = std.mem.splitScalar(u8, apps.stdout, '\n');
        while (it.next()) |line| {
            const l = std.mem.trim(u8, line, " \t\r");
            if (l.len != 0) log.err("  {s}", .{l});
        }
    } else |err| {
        log.warn("liste des process compute indisponible ({s})", .{@errorName(err)});
    }
    log.err("Libérer d'abord : `ollama ps` puis `ollama stop <modèle>` (réversible), ou --force-vram pour tenter quand même", .{});
    return error.GpuBusy;
}

// ============================================================================================
// SD — arguments, manifest, entrée compilée.
// ============================================================================================
const Arm = enum { letter, json };
const Order = enum { fwd, both };

const Args = struct {
    ckpt: []const u8,
    manifest: ?[]const u8 = null,
    arm: Arm = .letter,
    reps: usize = 5,
    out: ?[]const u8 = null,
    order: Order = .fwd,
    policy_selftest: bool = false,
    allow_cpu: bool = false,
    force_vram: bool = false,
};

const usage = "Usage: gemma4_decide <model.safetensors> --manifest f --arm {letter,json} --reps R --out f.jsonl [--order {fwd,both}] [--policy-selftest] [--allow-cpu] [--force-vram]";

fn nextVal(pa: []const [:0]const u8, i: *usize, flag: []const u8) ![]const u8 {
    i.* += 1;
    if (i.* >= pa.len) {
        log.err("{s} attend une valeur", .{flag});
        return error.MissingArgument;
    }
    return pa[i.*];
}

fn parseArgs(pa: []const [:0]const u8) !Args {
    if (pa.len < 2) {
        log.err("{s}", .{usage});
        return error.MissingArgument;
    }
    var a: Args = .{ .ckpt = pa[1] };
    var i: usize = 2;
    while (i < pa.len) : (i += 1) {
        const s = pa[i];
        if (std.mem.eql(u8, s, "--manifest")) {
            a.manifest = try nextVal(pa, &i, s);
        } else if (std.mem.eql(u8, s, "--arm")) {
            a.arm = std.meta.stringToEnum(Arm, try nextVal(pa, &i, s)) orelse return error.InvalidArgument;
        } else if (std.mem.eql(u8, s, "--reps")) {
            a.reps = try std.fmt.parseInt(usize, try nextVal(pa, &i, s), 10);
        } else if (std.mem.eql(u8, s, "--out")) {
            a.out = try nextVal(pa, &i, s);
        } else if (std.mem.eql(u8, s, "--order")) {
            a.order = std.meta.stringToEnum(Order, try nextVal(pa, &i, s)) orelse return error.InvalidArgument;
        } else if (std.mem.eql(u8, s, "--policy-selftest")) {
            a.policy_selftest = true;
        } else if (std.mem.eql(u8, s, "--allow-cpu")) {
            a.allow_cpu = true;
        } else if (std.mem.eql(u8, s, "--force-vram")) {
            a.force_vram = true;
        } else {
            log.err("argument inconnu: {s}\n{s}", .{ s, usage });
            return error.InvalidArgument;
        }
    }
    if (a.reps < 2 and a.order == .both) {
        log.err("--order both exige --reps >= 2 (témoin rep1/rep2 de C-SD-D)", .{});
        return error.InvalidArgument;
    }
    return a;
}

const Case = struct {
    case_id: []const u8,
    perm: []const u8, // "orig" | "rev" | "-" (json/warmup)
    ids: []u32,
    label_ids: [policy.N]u32,
    classes: [policy.N]policy.Class,
};

const Manifest = struct { eot_id: u32, warmup: Case, cases: []Case };

// Accès typés : un champ du mauvais type JSON rend BadManifest nommé (jamais un accès d'union
// invalide, qui serait un comportement indéfini en ReleaseFast).
fn getField(o: std.json.ObjectMap, name: []const u8) !std.json.Value {
    return o.get(name) orelse return badField(name);
}
fn asObject(v: std.json.Value, name: []const u8) !std.json.ObjectMap {
    return switch (v) {
        .object => |x| x,
        else => return badField(name),
    };
}
fn asArray(v: std.json.Value, name: []const u8) !std.json.Array {
    return switch (v) {
        .array => |x| x,
        else => return badField(name),
    };
}
fn asString(v: std.json.Value, name: []const u8) ![]const u8 {
    return switch (v) {
        .string => |x| x,
        else => return badField(name),
    };
}
fn asU32(v: std.json.Value, name: []const u8) !u32 {
    return switch (v) {
        .integer => |x| std.math.cast(u32, x) orelse return badField(name),
        else => return badField(name),
    };
}

fn readManifest(allocator: std.mem.Allocator, arena: std.mem.Allocator, io: std.Io, path: []const u8, arm: Arm) !Manifest {
    var mf = std.Io.Dir.cwd().openFile(io, path, .{ .mode = .read_only }) catch |err| {
        log.err("manifest illisible ({s}) : {s}", .{ path, @errorName(err) });
        return error.BadManifest;
    };
    defer mf.close(io);
    const mlen: usize = @intCast(try mf.length(io));
    const mtext = try allocator.alloc(u8, mlen);
    defer allocator.free(mtext);
    if (try mf.readPositionalAll(io, mtext, 0) != mlen) return error.ShortRead;
    var parsed = std.json.parseFromSlice(std.json.Value, allocator, mtext, .{ .allocate = .alloc_always }) catch |err| {
        log.err("manifest illisible ({s}) : JSON invalide ({s})", .{ path, @errorName(err) });
        return error.BadManifest;
    };
    defer parsed.deinit();
    const root = try asObject(parsed.value, "(racine)");

    const eot = try asU32(try getField(root, "eot_id"), "eot_id");
    const warm_obj = try asObject(try getField(root, "warmup"), "warmup");
    const warm_ids = try asArray(try getField(warm_obj, "ids"), "warmup.ids");
    const warmup = Case{ .case_id = "warmup", .perm = "-", .ids = try idsFrom(arena, warm_ids, "warmup.ids"), .label_ids = .{ 0, 1, 2, 3 }, .classes = .{ .direct, .search, .calculate, .insufficient } };
    if (warmup.ids.len == 0) return badField("warmup.ids (vide)");

    const key = if (arm == .letter) "letter" else "json";
    const arr = try asArray(try getField(root, key), key);
    const cases = try arena.alloc(Case, arr.items.len);
    for (arr.items, 0..) |v, k| {
        const o = try asObject(v, key);
        var c = Case{
            .case_id = try arena.dupe(u8, try asString(try getField(o, "case_id"), "case_id")),
            .perm = if (o.get("perm")) |p| try arena.dupe(u8, try asString(p, "perm")) else "-",
            .ids = try idsFrom(arena, try asArray(try getField(o, "ids"), "ids"), "ids"),
            // bras json : `cand` est ignoré ; ids factices distincts, dans le vocabulaire.
            .label_ids = warmup.label_ids,
            .classes = warmup.classes,
        };
        if (arm == .letter) {
            const li = try asArray(try getField(o, "label_ids"), "label_ids");
            const cl = try asArray(try getField(o, "classes"), "classes");
            if (li.items.len != policy.N or cl.items.len != policy.N) return badField("label_ids/classes (taille ≠ 4)");
            for (0..policy.N) |j| {
                c.label_ids[j] = try asU32(li.items[j], "label_ids");
                c.classes[j] = policy.classFromStr(try asString(cl.items[j], "classes")) orelse return badField("classes (valeur inconnue)");
            }
        }
        if (c.ids.len == 0) return badField("ids (vide)");
        cases[k] = c;
    }
    // spec §9 : un prompt > L_MAX est REFUSÉ NOMMÉMENT, les autres continuent (cas absent du
    // JSONL ⇒ le dépouilleur 86 le compte : le refus est visible, pas silencieux).
    var kept: usize = 0;
    for (cases) |c| {
        if (c.ids.len > @as(usize, @intCast(L_MAX))) {
            log.err("cas {s}/{s} REFUSÉ : {d} ids > L_MAX ({d})", .{ c.case_id, c.perm, c.ids.len, L_MAX });
            continue;
        }
        cases[kept] = c;
        kept += 1;
    }
    return .{ .eot_id = eot, .warmup = warmup, .cases = cases[0..kept] };
}

fn badField(name: []const u8) error{BadManifest} {
    log.err("manifest : champ '{s}' absent ou invalide", .{name});
    return error.BadManifest;
}

fn idsFrom(arena: std.mem.Allocator, arr: std.json.Array, name: []const u8) ![]u32 {
    const out = try arena.alloc(u32, arr.items.len);
    for (arr.items, 0..) |v, k| out[k] = try asU32(v, name);
    return out;
}

// Nom court OBLIGATOIRE (piège quota comptime @typeName, cf gemma4_gen_auto.zig:739-741).
// gather + forwardStep : IDENTIQUES à StepTok (gemma4_gen_auto.zig:742-755).
const StepDec = struct {
    pub fn forward(model: Model, tabs: Tabs, tok: zml.Tensor, cand: zml.Tensor, p: PackedLong, cache: engine.Cache, ctrl: engine.Ctrl) struct { zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor } {
        const e = model.embed_tokens.gather(.{ .voc = tok }, .{}); // {b,s,d} bf16 brut
        const el = tabs.eptl.gather(.{ .voc = tok }, .{}); // {b,s,lf} bf16 brut
        const logits, const slk, const slv, const flk, const flv = model.forwardStep(e, el, p, cache, ctrl);
        const t5 = logits.topK(.{ .voc = .voc }, 5, .{}); // bras B + bras A (top1 = glouton)
        const zc = logits.gather(.{ .voc = cand }, .{}); // {b,s,c=4} f32 — bras C (ordre des lettres)
        const lse = logits.logSumExp(.voc); // {b,s,voc=1} f32 — mass_in (op ZML, tensor.zig:1399)
        return .{ t5.values, t5.indices, zc, lse, slk, slv, flk, flv };
    }
};

pub fn main(init: std.process.Init) !void {
    @setEvalBranchQuota(200000); // piège quota comptime (cf gemma4_gchunk_auto.zig:96)
    const arena = init.arena;
    const allocator = init.gpa;
    const io = init.io;
    log.info("BUILD: mode={s}", .{@tagName(builtin.mode)}); // C-SD-E (motif gemma4_g12auto.zig:2027)

    const args = try parseArgs(try init.minimal.args.toSlice(arena.allocator()));

    if (args.policy_selftest) { // C-SD-F sur le binaire LIVRÉ (mêmes cas que les tests)
        var buf: [64]u8 = undefined;
        var n_fail: usize = 0;
        for (policy.selftest_cases) |c| {
            const d = policy.evaluate(c.in);
            const got = policy.describe(d, &buf);
            const ok = std.mem.eql(u8, got, c.want) and policy.sumOk(d);
            if (!ok) n_fail += 1;
            log.info("policy-selftest {s} : got={s} want={s} {s}", .{ c.name, got, c.want, if (ok) "OK" else "FAIL" });
        }
        if (n_fail != 0) return error.PolicySelftestFailed;
        log.info("C-SD-F PASS — {d}/{d}", .{ policy.selftest_cases.len, policy.selftest_cases.len });
        return;
    }

    // Manifest lu AVANT la plateforme et les poids : un manifest invalide échoue en < 1 s (spec §9).
    const mpath = args.manifest orelse return badField("--manifest");
    const out_path = args.out orelse return badField("--out");
    const mani = try readManifest(allocator, arena.allocator(), io, mpath, args.arm);
    log.info("manifest : {d} cas ({s}), eot_id={d}, warmup={d} ids", .{ mani.cases.len, @tagName(args.arm), mani.eot_id, mani.warmup.ids.len });

    if (args.force_vram) log.warn("--force-vram : garde VRAM sautée", .{}) else try checkVram(allocator, io);

    // ---- Copie verbatim gemma4_gen_auto.zig:899-918 ----
    // === Step 5.1 : backend CUDA (+ repli auto) — copié gemma4_gen_long_gpu.zig:80-92 (sans
    // --no-prealloc : "no no-prealloc needed", mémoire large marge cf PLAN) ===
    const platform: *zml.Platform = blk: {
        const cuda_opts: zml.platform.CreateOptions = .{ .cuda = .{ .allocator = .{ .bfc = .{ .preallocate = true, .memory_fraction = 0.90 } } } };
        if (zml.Platform.init(allocator, io, .cuda, cuda_opts)) |p| break :blk p else |_| {}
        log.warn("CUDA indisponible (libpjrt_cuda absent ?) — repli sur Platform.auto (probablement CPU).", .{});
        break :blk try zml.Platform.auto(allocator, io, .{});
    };
    defer platform.deinit(allocator);
    log.info("A1 — backend = {s} (cible : cuda)", .{@tagName(platform.target)});
    // Garde CUDA DURE (leçon de l'incident du 10 juil : le warn-and-continue a produit un run CPU
    // silencieux — binaire buildé sans `--@zml//platforms:cuda=true` → libpjrt_cuda absent des
    // runfiles → repli CPU discret ; un A2 ~1000 steps non surveillé y ramperait des heures).
    // fail-fast, échappatoire explicite --allow-cpu (débogage uniquement).
    if (platform.target != .cuda and !args.allow_cpu) {
        log.err("backend = {s} ≠ cuda — repli CPU refusé (rebuilder/lancer avec --@zml//platforms:cuda=true, ou passer --allow-cpu pour du débogage)", .{@tagName(platform.target)});
        return error.CudaRequired;
    }
    const sharding = try zml.sharding.replicatedSharding(platform);

    // ---- (gemma4_gen_auto.zig:919-928, bloc --selftest-gather : NON copié) ----
    // ---- Copie verbatim gemma4_gen_auto.zig:929-989 ----

    var reg_ck: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, args.ckpt);
    var store_ck: zml.io.TensorStore = .fromRegistry(allocator, &reg_ck);
    const base = store_ck.view().withPrefix("model").withPrefix("language_model");
    const model: Model = try .init(arena.allocator(), base);

    // Symboliques construits À LA MAIN (pas de fixture de store, cf tête de section) — mêmes shapes
    // que engine.Packed(.tables)/engine.Cache.
    const tok_sym = zml.Tensor.init(.{ 1, 1 }, .u32).withTags(.{ .b, .s });
    // Repli si le gather rank-2 ne compile pas (P5.4 n'a validé que des ids 1-D) : `tok_sym` en
    // `{ .s }` shape `[1]`, puis dans StepTok.forward : `.gather(.{ .voc = tok }).reshape(.{ 1, 1, D }).withTags(.{ .b, .s, .d })` (reshape layout-preserving + re-tag, piège ZML #1 connu) — idem `el` avec LF.
    // Repli dtype : si le gather exige des indices i32, passer tok_sym/host en `.i32` (le vocab < 2^31, cast sans perte).
    // ⚠ Si le dtype/shape des indices change ICI, changer AUSSI le tok_sym de selftestGather (SG) —
    // sinon SG resterait vert en validant autre chose que ce que le runtime fait.
    const packed_sym = PackedLong{
        .embeds = zml.Tensor.init(.{ L_MAX, 1, 1, D }, .bf16).withTags(.{ .step, .b, .s, .d }),
        .embptls = zml.Tensor.init(.{ L_MAX, 1, 1, LF }, .bf16).withTags(.{ .step, .b, .s, .lf }),
        .cos_full = zml.Tensor.init(.{ L_MAX, 1, 1, HD_F }, .f32).withTags(.{ .step, .b, .s, .hd }),
        .sin_full = zml.Tensor.init(.{ L_MAX, 1, 1, HD_F }, .f32).withTags(.{ .step, .b, .s, .hd }),
        .masks_sliding = zml.Tensor.init(.{ L_MAX, 1, 1, 1, L_MAX }, .f32).withTags(.{ .step, .b, .h, .q, .k }),
        .masks_full = zml.Tensor.init(.{ L_MAX, 1, 1, 1, L_MAX }, .f32).withTags(.{ .step, .b, .h, .q, .k }),
        .positions = zml.Tensor.init(.{L_MAX}, .i32).withTags(.{.step}),
    };
    const cache_sym = engine.Cache{
        .sl_k = zml.Tensor.init(.{ NUM_SLIDING_SLOTS, 1, 1, L_MAX, HD_S }, .f32).withTags(.{ .slot, .b, .h, .k, .hd }),
        .sl_v = zml.Tensor.init(.{ NUM_SLIDING_SLOTS, 1, 1, L_MAX, HD_S }, .f32).withTags(.{ .slot, .b, .h, .k, .hd }),
        .fl_k = zml.Tensor.init(.{ NUM_FULL_SLOTS, 1, 1, L_MAX, HD_F }, .f32).withTags(.{ .slot, .b, .h, .k, .hd }),
        .fl_v = zml.Tensor.init(.{ NUM_FULL_SLOTS, 1, 1, L_MAX, HD_F }, .f32).withTags(.{ .slot, .b, .h, .k, .hd }),
    };
    const ctrl_sym: engine.Ctrl = .initSymbolic();

    log.info("Materializing weights (store_ck) + Packed/Cache (HostInputs, zéros hors positions/cos/sin/masques) ...", .{});
    const eng_buf = try model.load(arena.allocator(), io, platform, &store_ck, &.{sharding});

    // L3 (spec docs/L3_INGRAPH_DESIGN.md §2.1) : SEULE table ajoutée au device (embed_tokens_per_layer,
    // ~4,7 Go bf16) — même TensorStore/`base` que Model.init, chargée AVANT store_ck.deinit() plus bas.
    const tabs: Tabs = .init(base); // même view withPrefix que Model.init
    const tabs_buf = try tabs.load(arena.allocator(), io, platform, &store_ck, &.{sharding});

    var host = try HostInputs.init(allocator);
    defer host.deinit(allocator);
    // Bufferized(PackedLong) assemblé À LA MAIN (motif E2, gemma4_engine_e2.zig:104-111) : chaque
    // champ = zml.Buffer.fromBytes depuis les slices host de Task 3 (mêmes shapes que packed_sym).
    const pk_buf = zml.Bufferized(PackedLong){
        .embeds = try zml.Buffer.fromBytes(io, platform, packed_sym.embeds.shape(), sharding, host.embeds_zero),
        .embptls = try zml.Buffer.fromBytes(io, platform, packed_sym.embptls.shape(), sharding, host.embptls_zero),
        .cos_full = try zml.Buffer.fromBytes(io, platform, packed_sym.cos_full.shape(), sharding, std.mem.sliceAsBytes(host.cos_full)),
        .sin_full = try zml.Buffer.fromBytes(io, platform, packed_sym.sin_full.shape(), sharding, std.mem.sliceAsBytes(host.sin_full)),
        .masks_sliding = try zml.Buffer.fromBytes(io, platform, packed_sym.masks_sliding.shape(), sharding, std.mem.sliceAsBytes(host.masks_sliding)),
        .masks_full = try zml.Buffer.fromBytes(io, platform, packed_sym.masks_full.shape(), sharding, std.mem.sliceAsBytes(host.masks_full)),
        .positions = try zml.Buffer.fromBytes(io, platform, packed_sym.positions.shape(), sharding, std.mem.sliceAsBytes(host.positions)),
    };
    var cache_buf = zml.Bufferized(engine.Cache){
        .sl_k = try zml.Buffer.fromBytes(io, platform, cache_sym.sl_k.shape(), sharding, host.cache_sl_k),
        .sl_v = try zml.Buffer.fromBytes(io, platform, cache_sym.sl_v.shape(), sharding, host.cache_sl_v),
        .fl_k = try zml.Buffer.fromBytes(io, platform, cache_sym.fl_k.shape(), sharding, host.cache_fl_k),
        .fl_v = try zml.Buffer.fromBytes(io, platform, cache_sym.fl_v.shape(), sharding, host.cache_fl_v),
    };
    store_ck.deinit();
    reg_ck.deinit();
    mem_probe.logMem(io, "post-load (poids + Packed/Cache sur device)");
    // ---- fin des copies verbatim ----

    const cand_sym = zml.Tensor.init(.{policy.N}, .u32).withTags(.{.c});
    log.info("Compiling StepDec.forward (gather+forwardStep+topK+gather candidats+logSumExp) ...", .{});
    const t_compile: std.Io.Timestamp = .now(io, .awake);
    var exe = try platform.compileFn(allocator, io, StepDec.forward, .{ model, tabs, tok_sym, cand_sym, packed_sym, cache_sym, ctrl_sym }, .{ .shardings = &.{sharding} });
    defer exe.deinit();
    log.info("  compile: {f}", .{t_compile.untilNow(io, .awake)});
    mem_probe.logMem(io, "post-compile");
    // (Task 6 : boucle des cas)
    _ = .{ eng_buf, tabs_buf, pk_buf, out_path };
    cache_buf.sl_k.deinit();
    cache_buf.sl_v.deinit();
    cache_buf.fl_k.deinit();
    cache_buf.fl_v.deinit();
}
