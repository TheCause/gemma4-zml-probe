// Runner décode complet 12B Unified (jalon J2, plan docs/superpowers/plans/2026-07-24-w4-j2-12b-unified.md
// Task 8, contrat docs/U_12B_CONTRACT.md) — clone ciblé de gemma4_w4auto.zig (J1) : SEULS
// Model/G12Step/init/géométrie/chat-template changent (diff 10 points du plan), tout le reste
// (RoPE host full, tokenizer, boucle, garde VRAM 20 GiB, garde CudaRequired) est à l'IDENTIQUE.
// Le forward compilé = `G12Step.forward` : assemble PAR VALEUR un Model moteur (48 couches,
// g12.G12Model.toLayerW — dequantW4 in-graph, v_proj full = placeholder [1] D4), embed = gather
// + scale bf16 62.0 (chemin 12B, D12 — JAMAIS le scale moteur √3840), puis délègue à
// `forwardStageGen(0, 48, first=false, last=true)` + topK.
// Cache sliding LINÉAIRE .k=L_MAX (R10 : kmax_sliding sert le modulo ring ET les shapes des
// masques in-graph — un cache court ferait des scatters hors bornes SILENCIEUX à p >= 1024) ;
// la fenêtre 1024 est portée par le MASQUE seul, GÉNÉRÉ IN-GRAPH depuis positions[step] + le
// scalaire runtime `window` (engine.ingraphMaskLines, spec 2026-07-26 — les tables host
// {L_MAX,L_MAX} quadratiques sont SUPPRIMÉES ; maskRows ne sert plus qu'au selftest).
//
// CLI : gemma4_g12auto <model.safetensors (12B w4a16-ct, weights_12b)> <tokenizer.json> --prompt "..."
//       [--max-tokens N] [--oracle fixture] [--ids-only] [--allow-cpu] [--force-vram]
//       [--dump-top5] [--out-ids f] [--window-vacuity f] [--no-prealloc]
//       [--selftest-inputs f] [--selftest-gather f (mode GPU, requiert un --prompt factice)]
// Mode --oracle : loggue en plus la marge top1−top2 par step de génération (protocole de flip W4g).
// --dump-top5 : top-5 par step aussi en mode LIBRE (requis U9). --out-ids : ids générés →
// safetensors (requis U9-ii/iv). --window-vacuity : replay teacher-forcé in-process, fenêtre
// élargie par rebind du scalaire `window` ← L_MAX en DONNÉES (U9-ii adapté in-graph). --no-prealloc :
// preallocate=false pour mesure VRAM réelle (U10, mécanisme gemma4_gen_long_gpu).
// ⚠ GPU : lancer avec `--@zml//platforms:cuda=true` sinon repli CPU silencieux.
const std = @import("std");
const log = std.log;
const zml = @import("zml");
const engine = @import("engine.zig");
const mem_probe = @import("mem_probe.zig");
const g12 = @import("g12.zig"); // Geom g12 + G12Model/G12LayerW (w4.W4Lin vit derrière g12.zig)

pub const std_options: std.Options = .{ .log_level = .info };
// ── Corps générique du runner, paramétré par la borne de contexte (COMPTIME : elle fixe les
// shapes du cache, des masques in-graph {k=L_MAX} et des tables RoPE — elle change le graphe
// tracé). Pattern « paramètre comptime explicite » du couple bbs/bbatch (piège 18 : PAS de
// @import("root") — les variantes 4k/8k importent ce fichier, root refermerait la boucle).
// Défaut 1280 (U9 : 1150 gen + prompt). Variantes : gemma4_g12a4k.zig → G12Auto(4096),
// gemma4_g12a8k.zig → G12Auto(8192).
// Coût 4096 ère TABLES (probe 26 juil, == HF-fp32 4000/4000) : pic VRAM 22 234 MiB, 8,2 tok/s
// (−9 %) — les masques {L_MAX,L_MAX} quadratiques rendaient 8k infaisable ; depuis la spec
// masques in-graph (2026-07-26), TOUT est linéaire en L_MAX (chiffres à jour : résultats du
// chantier masks-ingraph).
pub fn G12Auto(comptime L_MAX: i64) type {
return struct {

// Géométrie : TOUT vient de Geom.g12 (garde anti-source-mixte, plan Task 8 point 10 — aucun
// alias e2b du moteur (num_layers/embed-scale/têtes/dims historiques) ne doit être référencé ici).
const SLIDING_WINDOW: i64 = 1024; // config.sliding_window 12B — portée par le MASQUE seul (R10)
const HD_F: i64 = @intCast(g12.g12.hd_full); // 512 — dim cos/sin full (= config.global_head_dim)
const HD_S: i64 = @intCast(g12.g12.hd_sliding); // 256 — dim cache sliding
const D: i64 = @intCast(g12.g12.d); // 3840
const LF: i64 = 1; // FACTICE (ple_dim=0 : embptls jamais consommé ; 1 et pas 0 — buffer 0 octet non garanti)
const KVH_SL: i64 = @intCast(g12.g12.kvh_sliding); // 8 (GQA groupe 2)
const KVH_FL: i64 = @intCast(g12.g12.kvh_full); // 1 (MQA)
const N12: usize = g12.g12.num_layers; // 48
// Slots de cache : SANS YOCO (first_kv_shared = 48) chaque couche écrit son slot — 40 sliding
// + 8 full (motif (i+1)%6==0), dérivés du Geom (mêmes comptes que le gate u7).
const NUM_SLIDING_SLOTS: usize = blk: {
    @setEvalBranchQuota(100_000); // slidingSlot/fullSlot = boucles comptime O(48)
    break :blk @intCast(g12.g12.slidingSlot(g12.g12.num_layers)); // 40
};
const NUM_FULL_SLOTS: usize = blk: {
    @setEvalBranchQuota(100_000);
    break :blk @intCast(g12.g12.fullSlot(g12.g12.num_layers)); // 8
};
const Model = engine.EngineModel(struct {}, .{ .geom = g12.g12, .two_masks = true, .ingraph_masks = true, .kmax_sliding = L_MAX, .kmax_full = L_MAX });
const PackedLong = engine.Packed(.ingraph);

// BOS (id 2) : PRÉFIXÉ explicitement — l'encoder ZML (iree, cf zml/tokenizer/tokenizer.zig)
// n'ajoute AUCUN token spécial (constat Task 0 : ids ZML == ids HF sans template, modulo ce préfixe).
const BOS_ID: u32 = 2;

// Chat template 12B — templates_match=FALSE au contrat (U_12B_CONTRACT §6 : sha 12B ae53464b…
// ≠ E2B 2f1b4d75…) → adapté ICI au template 12B (U0 fait foi). VÉRITÉ = rendu HF RÉEL mesuré
// (25 juil, chat_template.jinja du snapshot 1d2c2d7f, jinja2 + apply_chat_template VM g12b) :
//   '<bos><|turn>user\nPROMPT<turn|>\n<|turn>model\n<|channel>thought\n<channel|>'
// Diff vs E2B (cas single-turn user, add_generation_prompt) : le 12B AJOUTE en fin d'amorce un
// canal de pensée VIDE '<|channel>thought\n<channel|>' (enable_thinking=false, défaut du
// template — « Fixed … thinking content-ordering », en-tête du jinja). Tokens : <|channel>=100,
// 'thought'=45518, '\n'=107, <channel|>=101 (mesurés) ; ids HF complets du prompt canonique :
// [2,105,2364,107,…,106,107,105,4368,107,100,45518,107,101] (25 ids).
// PÉRIMÈTRE : cas single-turn user → assistant UNIQUEMENT (U8/U9) — les branches
// system/tools/thinking/multi-tours du jinja 12B ne sont PAS portées ici.
// ⚠ tokens de tour : <|turn> (id 105) / <turn|> (id 106) — PAS <start_of_turn>/<end_of_turn>.
// BOS (id 2) : PRÉFIXÉ en id (l'encoder ZML n'ajoute AUCUN token spécial) — le rendu texte
// commence donc APRÈS <bos>.
fn renderChatTemplate(allocator: std.mem.Allocator, prompt: []const u8) ![]u8 {
    return std.fmt.allocPrint(allocator, "<|turn>user\n{s}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>", .{prompt});
}

/// `TensorRegistry.fromPath` refuse le checkpoint PACKÉ : `weights_12b/model.safetensors` est un
/// symlink vers le cache HF dont le `file.realPath` (resolveFiletype, safetensors.zig:543) se
/// résout en `blobs/<sha256>` SANS extension `.safetensors` -> `.unknown` -> error.InvalidPath
/// (bug mordu au premier run w2-12b, re-mordu au smoke Task 8). Contournement ÉPROUVÉ repris de
/// gemma4_g12gate.zig (gates immuables — duplication assumée) : ouvrir le fichier et appeler
/// `parseSafetensors` directement — licite car le contrat U0 garantit le packé MONO-fichier
/// (jamais un index). Les fixtures (--oracle/--selftest-*/--window-vacuity), fichiers réguliers
/// .safetensors, restent sur fromPath.
fn registryFromFile(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !zml.safetensors.TensorRegistry {
    const file = try std.Io.Dir.cwd().openFile(io, path, .{ .mode = .read_only });
    var registry: zml.safetensors.TensorRegistry = .init(allocator);
    errdefer registry.deinit();
    try zml.safetensors.parseSafetensors(allocator, io, &registry, file);
    return registry;
}

const Args = struct {
    ckpt: []const u8,
    tokjson_path: []const u8,
    prompt: ?[]const u8 = null,
    max_tokens: ?usize = null,
    oracle_path: ?[]const u8 = null,
    ids_only: bool = false,
    allow_cpu: bool = false,
    selftest_inputs: ?[]const u8 = null,
    selftest_gather: ?[]const u8 = null,
    force_vram: bool = false,
    // Nouveaux flags 12B (plan Task 8 point 8 — AUCUN n'existait dans la base clonée w4auto) :
    dump_top5: bool = false, // top-5 par step en mode LIBRE (requis U9)
    out_ids: ?[]const u8 = null, // ids générés -> safetensors (requis U9-ii replay / U9-iv teacher-forcing)
    window_vacuity: ?[]const u8 = null, // U9-ii : replay teacher-forcé, masque sliding élargi rebindé en DONNÉES
    no_prealloc: bool = false, // U10 : preallocate=false, nvidia-smi mesure la VRAM réelle (mécanisme gen_long_gpu)
};

const usage =
    "Usage: gemma4_g12auto <model.safetensors (checkpoint 12B w4a16-ct, weights_12b)> <tokenizer.json> --prompt \"...\" " ++
    "[--max-tokens N] [--oracle fixture] [--ids-only] [--allow-cpu (débogage uniquement)] " ++
    "[--force-vram] [--dump-top5] [--out-ids f] [--window-vacuity ids.safetensors] [--no-prealloc] " ++
    "[--selftest-inputs f] [--selftest-gather f (requiert un --prompt factice)]";

// Parsing à la main (comme les runners existants, ex. gemma4_gen_long_gpu.zig --no-prealloc) :
// pas de lib de flags ici, juste un balayage séquentiel des positionnels puis des --flags.
// Type EXACT du retour de std.process.Args.toSlice (cf lib/std/process/Args.zig) : une slice
// d'éléments sentinelle-terminés — chaque élément coerce vers []const u8 mais la slice ENTIÈRE
// ne coerce PAS vers []const []const u8 (piège de typage, d'où la signature précise ici).
fn parseArgs(process_args: []const [:0]const u8) !Args {
    if (process_args.len < 3) {
        log.err("{s}", .{usage});
        return error.MissingArgument;
    }
    var args: Args = .{ .ckpt = process_args[1], .tokjson_path = process_args[2] };

    var i: usize = 3;
    while (i < process_args.len) : (i += 1) {
        const a = process_args[i];
        if (std.mem.eql(u8, a, "--prompt")) {
            i += 1;
            if (i >= process_args.len) {
                log.err("--prompt attend une valeur", .{});
                return error.MissingArgument;
            }
            args.prompt = process_args[i];
        } else if (std.mem.eql(u8, a, "--max-tokens")) {
            i += 1;
            if (i >= process_args.len) {
                log.err("--max-tokens attend une valeur", .{});
                return error.MissingArgument;
            }
            args.max_tokens = std.fmt.parseInt(usize, process_args[i], 10) catch |err| {
                log.err("--max-tokens: valeur invalide '{s}' ({s})", .{ process_args[i], @errorName(err) });
                return err;
            };
        } else if (std.mem.eql(u8, a, "--oracle")) {
            i += 1;
            if (i >= process_args.len) {
                log.err("--oracle attend une valeur", .{});
                return error.MissingArgument;
            }
            args.oracle_path = process_args[i];
        } else if (std.mem.eql(u8, a, "--ids-only")) {
            args.ids_only = true;
        } else if (std.mem.eql(u8, a, "--allow-cpu")) {
            args.allow_cpu = true;
        } else if (std.mem.eql(u8, a, "--force-vram")) {
            args.force_vram = true;
        } else if (std.mem.eql(u8, a, "--dump-top5")) {
            args.dump_top5 = true;
        } else if (std.mem.eql(u8, a, "--no-prealloc")) {
            args.no_prealloc = true;
        } else if (std.mem.eql(u8, a, "--out-ids")) {
            i += 1;
            if (i >= process_args.len) {
                log.err("--out-ids attend une valeur", .{});
                return error.MissingArgument;
            }
            args.out_ids = process_args[i];
        } else if (std.mem.eql(u8, a, "--window-vacuity")) {
            i += 1;
            if (i >= process_args.len) {
                log.err("--window-vacuity attend une valeur", .{});
                return error.MissingArgument;
            }
            args.window_vacuity = process_args[i];
        } else if (std.mem.eql(u8, a, "--selftest-inputs")) {
            i += 1;
            if (i >= process_args.len) {
                log.err("--selftest-inputs attend une valeur", .{});
                return error.MissingArgument;
            }
            args.selftest_inputs = process_args[i];
        } else if (std.mem.eql(u8, a, "--selftest-gather")) {
            i += 1;
            if (i >= process_args.len) {
                log.err("--selftest-gather attend une valeur", .{});
                return error.MissingArgument;
            }
            args.selftest_gather = process_args[i];
        } else {
            log.err("argument inconnu: {s}\n{s}", .{ a, usage });
            return error.InvalidArgument;
        }
    }
    return args;
}

// ============================================================================================
// Task 3 — inputs host : cos/sin RoPE full, masques additifs, positions, tables {L_MAX,…}.
// ============================================================================================

// Masques additifs f32 : 0 = visible, -floatMax = masqué (== torch.finfo(float32).min — même
// valeur binaire : le plus grand f32 fini, négé, des deux côtés). Source unique : engine.MASK_MIN
// (les masques du RUNTIME sont générés in-graph par le moteur ; maskRows ci-dessous ne sert plus
// que de RÉFÉRENCE host au selftest — spec masques in-graph 2026-07-26 §4.5).
const MASK_MIN: f32 = engine.MASK_MIN;

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
    // (masks_sliding/masks_full {L_MAX,L_MAX} SUPPRIMÉS — masques générés in-graph, la fenêtre
    //  1024 est portée par le scalaire runtime `window` du Packed(.ingraph). Gain : 2×O(L²) f32
    //  host+device. Spec 2026-07-26 §4.4.)
    positions: []i32, // {L_MAX} = 0..L_MAX-1
    embeds_zero: []u8, // {L_MAX, 1, 1, D} bf16, zéros — factice, non consommé (first=false)
    embptls_zero: []u8, // {L_MAX, 1, 1, LF=1} bf16, zéros — FACTICE (ple_dim=0, ~2,6 Ko)
    cache_sl_k: []u8, // {NUM_SLIDING_SLOTS=40, 1, KVH_SL=8, L_MAX, HD_S} f32, zéros — cache LINÉAIRE .k=L_MAX (R10)
    cache_sl_v: []u8,
    cache_fl_k: []u8, // {NUM_FULL_SLOTS=8, 1, KVH_FL=1, L_MAX, HD_F} f32, zéros
    cache_fl_v: []u8,

    fn init(allocator: std.mem.Allocator) !HostInputs {
        const l_max: usize = @intCast(L_MAX);
        const hd_f: usize = @intCast(HD_F);

        const cos_full = try allocator.alloc(f32, l_max * hd_f);
        errdefer allocator.free(cos_full);
        const sin_full = try allocator.alloc(f32, l_max * hd_f);
        errdefer allocator.free(sin_full);
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
        }

        const embeds_zero = try allocator.alloc(u8, l_max * @as(usize, @intCast(D)) * 2);
        errdefer allocator.free(embeds_zero);
        @memset(embeds_zero, 0);
        const embptls_zero = try allocator.alloc(u8, l_max * @as(usize, @intCast(LF)) * 2);
        errdefer allocator.free(embptls_zero);
        @memset(embptls_zero, 0);
        // GQA 12B : le cache sliding porte kvh=8 têtes KV (D5) — facteur KVH_SL dans la taille.
        const sl_bytes = NUM_SLIDING_SLOTS * @as(usize, @intCast(KVH_SL)) * l_max * @as(usize, @intCast(HD_S)) * 4;
        const fl_bytes = NUM_FULL_SLOTS * @as(usize, @intCast(KVH_FL)) * l_max * hd_f * 4;
        const cache_sl_k = try allocator.alloc(u8, sl_bytes);
        errdefer allocator.free(cache_sl_k);
        @memset(cache_sl_k, 0);
        const cache_sl_v = try allocator.alloc(u8, sl_bytes);
        errdefer allocator.free(cache_sl_v);
        @memset(cache_sl_v, 0);
        const cache_fl_k = try allocator.alloc(u8, fl_bytes);
        errdefer allocator.free(cache_fl_k);
        @memset(cache_fl_k, 0);
        const cache_fl_v = try allocator.alloc(u8, fl_bytes);
        errdefer allocator.free(cache_fl_v);
        @memset(cache_fl_v, 0);

        return .{
            .cos_full = cos_full,
            .sin_full = sin_full,
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
        allocator.free(self.positions);
        allocator.free(self.embeds_zero);
        allocator.free(self.embptls_zero);
        allocator.free(self.cache_sl_k);
        allocator.free(self.cache_sl_v);
        allocator.free(self.cache_fl_k);
        allocator.free(self.cache_fl_v);
    }
};

// Lit un tenseur ENTIER de la fixture, host-side, SANS Platform : lecture positionnelle directe
// dans le fichier à `tensor.offset` (octets absolus, cf gemma4_gchunk_auto.zig:220-223) sur
// `tensor.byteSize()` octets. Durci : dtype du header vérifié AVANT lecture (une fixture au
// mauvais dtype serait sinon réinterprétée silencieusement), compte d'octets lus vérifié APRÈS
// (fichier tronqué → error.ShortRead, pas des zéros silencieux).
fn readFixtureAlloc(comptime T: type, comptime want_dtype: zml.DataType, allocator: std.mem.Allocator, io: std.Io, reg: *const zml.safetensors.TensorRegistry, file: *std.Io.File, name: []const u8) ![]T {
    const t = reg.tensors.get(name) orelse {
        log.err("tensor introuvable dans la fixture: {s}", .{name});
        return error.MissingTensor;
    };
    const dt = t.shape.dtype();
    if (dt != want_dtype) {
        log.err("{s}: dtype fixture = {s} ≠ attendu = {s}", .{ name, @tagName(dt), @tagName(want_dtype) });
        return error.DtypeMismatch;
    }
    const size: usize = @intCast(t.byteSize());
    const out = try allocator.alloc(T, size / @sizeOf(T));
    errdefer allocator.free(out);
    const got = try file.readPositionalAll(io, std.mem.sliceAsBytes(out), t.offset);
    if (got != size) {
        log.err("{s}: lecture courte — {d}/{d} octets (fixture tronquée ?)", .{ name, got, size });
        return error.ShortRead;
    }
    return out;
}

// Tolérance cos/sin DÉPENDANTE DE LA POSITION (dérivation complète : note `ropeFull`) — PAS une
// constante : l'erreur d'angle inter-implémentations croît linéairement, Δangle ≲ 2 ULP × p ×
// ULP(inv_freq≈0.5)=6e-8 ≈ 1.2e-7×p, propagée par sin/cos à pente ≤ 1. tol(p) = 1e-5 + 1.5e-7×p
// couvre cette enveloppe avec marge (mesuré : 3.81e-6 @ p≤68 vs tol 2.0e-5 ; 6.00e-5 @ p=612 vs
// tol 1.02e-4 ; borne théorique ~1.2e-4 @ p=1011 vs tol 1.6e-4) sans masquer une VRAIE régression
// de formule (qui produirait des écarts de plusieurs ordres de grandeur, pas ~2 ULP).
fn cosSinTol(p: i32) f32 {
    return 1e-5 + 1.5e-7 * @as(f32, @floatFromInt(p));
}

// --selftest-inputs <fixture> : charge une fixture d'inputs (clés positions/cos_full/sin_full/
// masks_sliding/masks_full, format hérité J1 — à produire côté oracle 12B au besoin) et
// compare, pour chaque step k, la position p = positions[k] LUE DE LA FIXTURE :
//   - cos/sin : lignes de la table `HostInputs` à l'index p (donc `ropeFull` ET la construction
//     de la table {L_MAX,…} sont toutes les deux exercées) vs cos_full/sin_full[k] — écart par
//     step ≤ cosSinTol(p) (tolérance position-dépendante, cf dérivation sur `cosSinTol`).
//   - masques : lignes RECALCULÉES à la volée par maskRows(p) (2 scratch L_MAX — plus de table
//     O(L²) : le runtime les génère in-graph, maskRows est la référence host de la SPEC du masque ;
//     l'équivalence du GRAPHE est prouvée par les gates M1/M2) vs masks_sliding/masks_full[k] —
//     égalité BIT-EXACTE (valeurs ∈ {0, MASK_MIN}).
//   - positions : continuité (positions[k] == positions[0] + k, i.e. p = seq_len + k avec
//     seq_len = positions[0], lu de la fixture — pas besoin du manifest JSON séparé).
fn selftestInputs(allocator: std.mem.Allocator, io: std.Io, fixture_path: []const u8) !void {
    var reg: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, fixture_path);
    defer reg.deinit();
    var file = try std.Io.Dir.cwd().openFile(io, fixture_path, .{ .mode = .read_only });
    defer file.close(io);

    const positions_fx = try readFixtureAlloc(i32, .i32, allocator, io, &reg, &file, "positions");
    defer allocator.free(positions_fx);
    const cos_fx = try readFixtureAlloc(f32, .f32, allocator, io, &reg, &file, "cos_full");
    defer allocator.free(cos_fx);
    const sin_fx = try readFixtureAlloc(f32, .f32, allocator, io, &reg, &file, "sin_full");
    defer allocator.free(sin_fx);
    const masks_sliding_fx = try readFixtureAlloc(f32, .f32, allocator, io, &reg, &file, "masks_sliding");
    defer allocator.free(masks_sliding_fx);
    const masks_full_fx = try readFixtureAlloc(f32, .f32, allocator, io, &reg, &file, "masks_full");
    defer allocator.free(masks_full_fx);

    const n_decode: usize = positions_fx.len;
    const hd_f: usize = @intCast(HD_F);
    const l_max: usize = @intCast(L_MAX);
    if (cos_fx.len != n_decode * hd_f or sin_fx.len != n_decode * hd_f) {
        log.err("SELFTEST INPUTS : shape cos/sin inattendue (cos.len={d} sin.len={d} n_decode*HD_F={d})", .{ cos_fx.len, sin_fx.len, n_decode * hd_f });
        return error.UnexpectedShape;
    }
    if (masks_sliding_fx.len != n_decode * l_max or masks_full_fx.len != n_decode * l_max) {
        log.err("SELFTEST INPUTS : shape masques inattendue (sliding.len={d} full.len={d} n_decode*L_MAX={d})", .{ masks_sliding_fx.len, masks_full_fx.len, n_decode * l_max });
        return error.UnexpectedShape;
    }

    var host = try HostInputs.init(allocator);
    defer host.deinit(allocator);
    const scratch_sl = try allocator.alloc(f32, l_max);
    defer allocator.free(scratch_sl);
    const scratch_fl = try allocator.alloc(f32, l_max);
    defer allocator.free(scratch_fl);

    var max_abs: f32 = 0;
    var max_ratio: f32 = 0; // max sur les steps de (écart / cosSinTol(p)) — critère de PASS : ≤ 1
    var max_at: struct { k: usize, i: usize, kind: u8, host: f32, fx: f32, tol: f32 } = .{ .k = 0, .i = 0, .kind = 'c', .host = 0, .fx = 0, .tol = 0 };
    var masks_bitexact = true;
    var positions_ok = true;
    const seq_len = positions_fx[0];

    for (0..n_decode) |k| {
        const p = positions_fx[k];
        if (p != seq_len + @as(i32, @intCast(k))) positions_ok = false;
        if (p < 0 or p >= L_MAX) {
            log.err("SELFTEST INPUTS : position hors table à step {d} (p={d})", .{ k, p });
            return error.PositionOutOfRange;
        }
        const pi: usize = @intCast(p);
        const tol = cosSinTol(p);

        const cos_row = host.cos_full[pi * hd_f .. (pi + 1) * hd_f];
        const sin_row = host.sin_full[pi * hd_f .. (pi + 1) * hd_f];
        const cos_fx_row = cos_fx[k * hd_f .. (k + 1) * hd_f];
        const sin_fx_row = sin_fx[k * hd_f .. (k + 1) * hd_f];
        for (0..hd_f) |i| {
            const dc = @abs(cos_row[i] - cos_fx_row[i]);
            const ds = @abs(sin_row[i] - sin_fx_row[i]);
            if (dc > max_abs) max_abs = dc;
            if (ds > max_abs) max_abs = ds;
            if (dc / tol > max_ratio) {
                max_ratio = dc / tol;
                max_at = .{ .k = k, .i = i, .kind = 'c', .host = cos_row[i], .fx = cos_fx_row[i], .tol = tol };
            }
            if (ds / tol > max_ratio) {
                max_ratio = ds / tol;
                max_at = .{ .k = k, .i = i, .kind = 's', .host = sin_row[i], .fx = sin_fx_row[i], .tol = tol };
            }
        }

        maskRows(p, scratch_sl, scratch_fl); // référence host recalculée à la volée (spec §4.5)
        const sl_fx_row = masks_sliding_fx[k * l_max .. (k + 1) * l_max];
        const fl_fx_row = masks_full_fx[k * l_max .. (k + 1) * l_max];
        for (0..l_max) |j| {
            if (scratch_sl[j] != sl_fx_row[j]) masks_bitexact = false;
            if (scratch_fl[j] != fl_fx_row[j]) masks_bitexact = false;
        }
    }

    const cos_sin_ok = max_ratio <= 1.0;
    if (cos_sin_ok and masks_bitexact and positions_ok) {
        log.info("SELFTEST INPUTS PASS ({d} steps, cos/sin max_abs={e} max_ratio={d:.3} de tol(p), masks bit-exact, positions ==)", .{ n_decode, max_abs, max_ratio });
    } else {
        log.err("SELFTEST INPUTS FAIL — cos/sin max_abs={e} max_ratio={d:.3} (ok={}) masks_bitexact={} positions_ok={}", .{ max_abs, max_ratio, cos_sin_ok, masks_bitexact, positions_ok });
        if (!cos_sin_ok) {
            const p0: usize = @intCast(positions_fx[0]);
            log.err("  1er step : p={d} cos_host[0..8]={any} cos_fx[0..8]={any}", .{ p0, host.cos_full[p0 * hd_f .. p0 * hd_f + 8], cos_fx[0..8] });
            log.err("  1er step : sin_host[0..8]={any} sin_fx[0..8]={any}", .{ host.sin_full[p0 * hd_f .. p0 * hd_f + 8], sin_fx[0..8] });
            log.err("  pire ratio à step k={d} (p={d}) index i={d} kind={c} : host={d} fx={d} écart={e} tol(p)={e}", .{ max_at.k, positions_fx[max_at.k], max_at.i, max_at.kind, max_at.host, max_at.fx, @abs(max_at.host - max_at.fx), max_at.tol });
        }
        return error.SelftestInputsFailed;
    }
}

// ============================================================================================
// 12B (plan Task 8 point 2) : `Tabs` (embed_tokens_per_layer, L3 E2B) est SUPPRIMÉ — la clé
// `embed_tokens_per_layer.weight` N'EXISTE PAS au checkpoint 12B (ple_dim=0), createTensor
// crasherait (précédent w4.zig:56-58). Le selftest-gather est adapté EMB-ONLY pour la même
// raison (l'eptl du SgTabs E2B référençait cette clé absente).
// ============================================================================================
const EMB_KEY = "model.language_model.embed_tokens.weight"; // utilisé par SgTabs (clé ABSOLUE, root view)

// SG : struct à 1 champ (12B : emb seul — pas d'eptl au checkpoint), dédié au mini-graphe
// gather-only du selftest — indépendant de `Model` (pas de forward complet).
const SgTabs = struct {
    emb: zml.Tensor, // {voc,d} bf16 BRUT (embed_tokens)

    fn init(base: zml.io.TensorStore.View) SgTabs {
        return .{ .emb = base.createTensor(EMB_KEY, .{ .voc, .d }, null) };
    }
    fn load(self: *const SgTabs, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, store: *const zml.io.TensorStore, shardings: []const zml.sharding.Sharding) !zml.Bufferized(SgTabs) {
        return zml.io.load(SgTabs, self, allocator, io, platform, store, .{ .shardings = shardings, .parallelism = 1, .dma_chunks = 1, .dma_chunk_size = 16 * 1024 * 1024 });
    }
};

// SG : mini-graphe gather-only — mêmes primitives que `G12Step` (gather + GatherOpts `.{}`
// OBLIGATOIRE, cf G12Step plus bas), sans forward/topK (pas besoin du modèle complet).
const SgFwd = struct {
    pub fn forward(emb: zml.Tensor, tok: zml.Tensor) zml.Tensor {
        return emb.gather(.{ .voc = tok }, .{});
    }
};

// --selftest-gather <fixture> : pour chaque step k de la fixture (fed[k] = token FED à ce
// step), gather(fed[k]) sur le CHECKPOINT doit être BIT-EXACT à embeds[k] de la fixture —
// la même ligne bf16 brute NON re-scalée. Comparaison en u16 bruts (bf16 = 2 octets, pas de
// tolérance) — ne JAMAIS l'affaiblir en tolérance (piège relevé en revue J1).
// 12B : EMB-ONLY (pas d'embptls — la clé eptl n'existe pas au checkpoint, cf SgTabs).
// Mode GPU (mini-graphe compilé `SgFwd.forward`) — la garde VRAM s'applique (câblage dans
// `main`, dispatché après Platform.init/garde CUDA/sharding). Charge SEULEMENT la table emb
// (SgTabs) via TensorStore, PAS le `Model` complet.
fn selftestGather(allocator: std.mem.Allocator, io: std.Io, platform: *zml.Platform, sharding: zml.sharding.Sharding, ckpt_path: []const u8, fixture_path: []const u8) !void {
    // Fixture d'abord (host-only, rapide) : fail-fast si la fixture est cassée, avant tout travail
    // GPU (registry + tenseurs, mêmes helpers que --selftest-inputs/--oracle).
    var reg_fx: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, fixture_path);
    defer reg_fx.deinit();
    var file_fx = try std.Io.Dir.cwd().openFile(io, fixture_path, .{ .mode = .read_only });
    defer file_fx.close(io);

    const fed_fx = try readFixtureAlloc(i32, .i32, allocator, io, &reg_fx, &file_fx, "fed");
    defer allocator.free(fed_fx);
    const embeds_fx = try readFixtureAlloc(u16, .bf16, allocator, io, &reg_fx, &file_fx, "embeds");
    defer allocator.free(embeds_fx);

    const n: usize = fed_fx.len;
    if (n == 0) {
        log.err("--selftest-gather : fixture 'fed' vide — un PASS à 0 step serait vacueux", .{});
        return error.EmptyFixture;
    }
    const d_u: usize = @intCast(D);
    if (embeds_fx.len != n * d_u) {
        log.err("SG : shape fixture inattendue (embeds.len={d}, attendu {d}x{d}={d})", .{ embeds_fx.len, n, d_u, n * d_u });
        return error.UnexpectedShape;
    }

    // Checkpoint : SEULE la table emb (12B) — pas de Model.init (le forward complet n'est pas
    // nécessaire au selftest). Root view (pas de withPrefix) : EMB_KEY est déjà la clé absolue.
    var reg_ck = try registryFromFile(allocator, io, ckpt_path); // JAMAIS fromPath sur le packé (symlink HF)
    defer reg_ck.deinit();
    var store_ck: zml.io.TensorStore = .fromRegistry(allocator, &reg_ck);
    defer store_ck.deinit();
    const base = store_ck.view();

    const sg_tabs: SgTabs = .init(base);
    const sg_buf = try sg_tabs.load(allocator, io, platform, &store_ck, &.{sharding});

    const tok_sym = zml.Tensor.init(.{ 1, 1 }, .u32).withTags(.{ .b, .s });
    var exe = try platform.compileFn(allocator, io, SgFwd.forward, .{ sg_tabs.emb, tok_sym }, .{ .shardings = &.{sharding} });
    defer exe.deinit();

    var first_fail: ?struct { step: usize, idx: usize, host: u16, fx: u16 } = null;

    for (0..n) |k| {
        // Bits du token PRÉSERVÉS (pas @intCast) : un `fed` négatif dans une fixture corrompue
        // (cf G1v) ne doit jamais déclencher un piège d'intCast — reinterprétation brute, le
        // gather XLA (ou le mismatch qui suit) qualifiera l'anomalie, pas un crash de cast.
        var tok_host = [1]u32{@bitCast(fed_fx[k])};
        var tok_buf = try zml.Buffer.fromBytes(io, platform, tok_sym.shape(), sharding, std.mem.sliceAsBytes(&tok_host));

        var call_args = try exe.args(allocator);
        var call_results = try exe.results(allocator);
        call_args.set(.{ sg_buf.emb, tok_buf });
        exe.call(call_args, &call_results);
        var r_emb = call_results.get(zml.Buffer);

        var emb_s = try r_emb.toSliceAlloc(allocator, io);
        defer emb_s.free(allocator);
        const emb_bits = emb_s.items(u16);
        // Garde longueur = shape ET dtype d'un coup (même standard que l'assert i32 du chemin
        // réel) : un gather upcasté bf16→f32 doublerait len et produirait un « mismatch »
        // trompeur au lieu d'une erreur qualifiée.
        if (emb_bits.len != d_u) {
            log.err("SG : longueur D2H inattendue (emb={d}≠{d}) — dtype/shape du gather a dérivé ?", .{ emb_bits.len, d_u });
            return error.UnexpectedShape;
        }

        const emb_fx_row = embeds_fx[k * d_u .. (k + 1) * d_u];
        if (first_fail == null) {
            for (0..d_u) |i| {
                if (emb_bits[i] != emb_fx_row[i]) {
                    first_fail = .{ .step = k, .idx = i, .host = emb_bits[i], .fx = emb_fx_row[i] };
                    break;
                }
            }
        }

        r_emb.deinit();
        tok_buf.deinit();
        call_args.deinit(allocator);
        call_results.deinit(allocator);

        if (first_fail != null) break; // 1ère divergence suffit au diagnostic — pas la peine de continuer
    }

    if (first_fail) |ff| {
        log.err("SG FAIL — step={d} (fed={d}) 1ère divergence idx={d} : host=0x{x} fixture=0x{x}", .{ ff.step, fed_fx[ff.step], ff.idx, ff.host, ff.fx });
        return error.SgGatherMismatch;
    }
    log.info("SG PASS — {d} steps × table emb bit-exact (gather in-graph, 12B emb-only)", .{n});
}

// ============================================================================================
// Boucle autonome prefill-par-decode (G12Step = gather + embed D12 + forwardStageGen + topK
// IN-GRAPH, buffers device per-step) — mécanique héritée du clone w4auto (L3).
// ============================================================================================
//
// Compile UNE FOIS le mono-graphe `G12Step.forward`, qui compose : gather (`m12.embed_tokens`)
// → scale bf16 62.0 (chemin 12B, D12/mode u2) → `Model.forwardStageGen(0, 48, first=false,
// last=true)` INCHANGÉ (engine.zig:698 — runLayerGen partagé, branche D4 K=V dans le graphe)
// → `topK(.voc, 5)`. Packed(.ingraph)/Cache SYMBOLIQUES construits À LA MAIN :
//   - cos_full/sin_full/positions : RÉELLEMENT consommés (indexés par `ctrl.step` == position
//     absolue p, cf pickStep) — remplis depuis HostInputs. `window` : scalaire {} i32 = 1024
//     (les masques sont GÉNÉRÉS in-graph depuis positions[step] + window, engine.ingraphMaskLines).
//   - embeds/embptls du Packed symbolique : déclarés (le type Packed(.ingraph) a 6 champs) mais
//     MORTS au graphe (first=false → hidden = hidden_in ; ple_dim=0 → PLE comptime-mort) —
//     remplis avec les tables zéro de HostInputs (embptls LF=1 factice), jamais consommés.
//   - Cache initial : zéro (Bufferized construit par zml.Buffer.fromBytes depuis les zéros host).
//
// R7 (MAJ post-U7) : le graphe complet 48 couches a COMPILÉ OK en CPU (u7 run 4 — c'est
// l'EXÉCUTION CPU qui a OOM en RAM, anon-rss 23,6 Go) ; sur GPU l'exécution vit en VRAM
// (~10-12 Go projetés sur 24) — le mono-compile est la voie nominale. Si un mur GPU apparaît :
// NE PAS improviser le chunk-majeur U7Chunk (il est prefill-only) — NEEDS_DECISION.

// Top5 : idx/val remplis DEPUIS LE DEVICE (topK in-graph, `G12Step` plus bas) — plus de scan host
// (`top5Of` SUPPRIMÉ, Task 4 historique). top1 = next token ; top5 entier = diagnostic --oracle
// (spec docs/L3_INGRAPH_DESIGN.md §4, vigilance ties d'argmax). Struct inchangée : même usage par
// le reste de la boucle (`gen_top5`, diagnostic FAIL Step 5.3).
const Top5 = struct { idx: [5]usize, val: [5]f32 };

// ============================================================================================
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
// ⚠ Seuil hérité de w4auto (J1), PAS re-mesuré pour G12 — dette U10.
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

// G12 : compose gather in-graph + scale bf16 62.0 (D12) + dequant W4 in-graph + forwardStageGen
// (engine INTACT) + topK. top1 du topK == argmax (tri descendant `sort`, cf tensor.zig:3096) ;
// top5 = diagnostic --oracle/--dump-top5 ; logits = sortie SUPPLÉMENTAIRE (vs w4auto) pour
// --window-vacuity (U9-ii : comparaison de logits par position — D2H seulement quand lue).
// Nom court OBLIGATOIRE (piège quota comptime @typeName sur pjrt.zig structSize).
//
// ⚠ D12 (décision d'interprétation, consignée au rapport Task 8) : `forwardStep` applique le
// scale moteur `√3840 f64` (engine.zig:754, geom.embedScale()) — INCOMPATIBLE avec D12 (le 12B
// doit reproduire l'arrondi bf16 RÉEL de HF : 62.0 exactement, U_12B_CONTRACT §7, jamais dans
// engine.zig). La voie moteur-intact est `forwardStageGen(0, 48, first=false, last=true)` :
// hidden_in = embed 12B (gather + scale bf16 62.0 + convert f32 EXACT), le chemin embed interne
// du moteur (√3840, first=false) est émis mais MORT (DCE) — précédent éprouvé : gate u7 (PASS),
// même chemin par chunks. `el = e` du plan est sans objet ici (forwardStageGen n'a pas de
// paramètre embed ; embptls du Packed reste factice, PLE comptime-mort).
const G12Step = struct {
    pub fn forward(m12: g12.G12Model, tok: zml.Tensor, p: PackedLong, cache: engine.Cache, ctrl: engine.Ctrl) struct { zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor } {
        @setEvalBranchQuota(1_000_000); // slidingSlot comptime O(j) x 48 couches (pattern U7Chunk)
        // Assemblage par VALEUR d'un Model moteur dont les poids linéaires sont des sous-graphes
        // dequantW4 — toLayerW(j) : v_proj full = placeholder [1] (D4), sliding = vs[slidingSlot].
        var layers: [N12]engine.LayerW = undefined;
        inline for (0..N12) |j| layers[j] = m12.toLayerW(j);
        const m: Model = .{
            .embed_tokens = m12.embed_tokens, // consommé par le gather ci-dessous ET le head tied (D7)
            .per_layer_model_projection = m12.layers[0].layer_scalar, // placeholders [1] — PLE mort (ple_dim=0)
            .per_layer_projection_norm = m12.layers[0].layer_scalar,
            .final_norm = m12.final_norm,
            .layers = &layers,
            .brick = .{},
            .prec = .{},
        };
        const e = m.embed_tokens.gather(.{ .voc = tok }, .{}); // {b,s,d} bf16 brut
        // Chemin 12B du scale (D12/mode u2) : ×62.0 constante bf16 == bf16(√3840), produit
        // bf16×bf16 (`scale` émet sa constante au dtype du tensor), puis bf16 -> f32 EXACT.
        const h0 = e.scale(62.0).convert(.f32);
        const logits, const slk, const slv, const flk, const flv = m.forwardStageGen(0, N12, false, true, p, cache, h0, ctrl);
        // Forme struct à un champ EXIGÉE par `Tensor.topK` (cf zml/nn.zig:1558, seul site d'appel
        // réel dans les sources ZML : `logits.topK(.{ .voc = .voc }, k, .{})`).
        const t5 = logits.topK(.{ .voc = .voc }, 5, .{});
        return .{ t5.values, t5.indices, logits, slk, slv, flk, flv };
    }
};

// --out-ids (U9-ii/iv) : écrit les ids générés en safetensors MINIMAL (une clé "ids", i32 {n}) —
// format relisible par readFixtureAlloc (--window-vacuity) ET par safetensors Python (oracles
// 69/70). Header JSON + longueur u64 LE (spec safetensors) ; écriture directe (pattern
// gemma4_g23_sweep : createFile + writePositionalAll), pas de writer zml nécessaire.
fn writeIdsSafetensors(allocator: std.mem.Allocator, io: std.Io, path: []const u8, ids_i64: []const i64) !void {
    const n = ids_i64.len;
    const data = try allocator.alloc(i32, n);
    defer allocator.free(data);
    for (ids_i64, 0..) |t, k| data[k] = @intCast(t); // ids < vocab 262144 : cast sans perte
    const header = try std.fmt.allocPrint(allocator, "{{\"ids\":{{\"dtype\":\"I32\",\"shape\":[{d}],\"data_offsets\":[0,{d}]}}}}", .{ n, n * 4 });
    defer allocator.free(header);
    var len_le: [8]u8 = undefined;
    std.mem.writeInt(u64, &len_le, header.len, .little);
    const f = try std.Io.Dir.createFile(.cwd(), io, path, .{});
    defer f.close(io);
    try f.writePositionalAll(io, &len_le, 0);
    try f.writePositionalAll(io, header, 8);
    try f.writePositionalAll(io, std.mem.sliceAsBytes(data), 8 + header.len);
}

pub fn run(init: std.process.Init) !void {
    @setEvalBranchQuota(200000); // piège quota comptime (cf gemma4_gchunk_auto.zig:96)
    const arena = init.arena;
    const allocator = init.gpa;
    const io = init.io;

    const process_args = try init.minimal.args.toSlice(arena.allocator());
    const args = try parseArgs(process_args);

    // === Task 3 : --selftest-inputs — indépendant du prompt/tokenizer/poids (fixture only) ===
    if (args.selftest_inputs) |fixture_path| {
        try selftestInputs(allocator, io, fixture_path);
        return;
    }

    const prompt_text = args.prompt orelse {
        log.err("--prompt est requis\n{s}", .{usage});
        return error.MissingArgument;
    };

    // === Gate A0 : tokenizer ZML natif + chat template Zig ===
    var tokenizer = try zml.tokenizer.Tokenizer.fromFile(allocator, io, args.tokjson_path);
    defer tokenizer.deinit();
    var encoder = try tokenizer.encoder();
    defer encoder.deinit();

    // EOT_ID — MESURÉ depuis le tokenizer (spec §3.4), JAMAIS hardcodé : encode "<turn|>" (le token
    // de fin de tour, cf renderChatTemplate) et exige EXACTEMENT 1 id. Un compte ≠ 1 signalerait un
    // tokenizer/template différent de celui mesuré (10 juil, id=106) — BLOCKED plutôt qu'un repli
    // silencieux sur une valeur hardcodée.
    var eot_tok = try encoder.encodeAlloc(allocator, "<turn|>");
    defer eot_tok.deinit(allocator);
    if (eot_tok.items.len != 1) {
        log.err("EOT: '<turn|>' encode en {d} tokens (attendu 1) — ids={any}", .{ eot_tok.items.len, eot_tok.items });
        return error.EotNotSingleToken;
    }
    const eot_id: u32 = eot_tok.items[0];
    log.info("EOT_ID = {d} (mesuré depuis le tokenizer)", .{eot_id});
    // reset() avant réutilisation : l'encoder iree est un automate à état (cf round-trip --ids-only).
    encoder.reset();

    const rendered = try renderChatTemplate(arena.allocator(), prompt_text);
    var prompt_tok = try encoder.encodeAlloc(allocator, rendered);
    defer prompt_tok.deinit(allocator);

    var ids: std.ArrayList(u32) = try .initCapacity(allocator, prompt_tok.items.len + 1);
    defer ids.deinit(allocator);
    try ids.append(allocator, BOS_ID);
    try ids.appendSlice(allocator, prompt_tok.items);

    if (args.ids_only) {
        log.info("ids = {any}", .{ids.items});

        // Round-trip détok (Step 2.4) : decode les ids APRÈS bos (= prompt_tok.items, la partie
        // produite par l'encoder, gabarit de chat INCLUS — pas seulement le texte user) puis
        // re-encode. Plus fort que le PLAN Step 2.4 (qui n'exigeait que le prompt hors template) :
        // on assume la déviation, le round-trip couvre aussi les tokens de tour <|turn>/<turn|>.
        var decoder = try tokenizer.decoder();
        defer decoder.deinit();
        var text_rt = try decoder.decodeAlloc(allocator, ids.items[1..]);
        defer text_rt.deinit(allocator);

        // reset() avant réutilisation : l'encoder iree est un automate à état (encode_state_t) ;
        // finalize() ne remet pas AT_INPUT_START, réutiliser encoder sans reset risquerait de
        // faire fuiter l'état du 1er encodage dans le round-trip.
        encoder.reset();
        var reenc = try encoder.encodeAlloc(allocator, text_rt.items);
        defer reenc.deinit(allocator);

        const round_trip_ok = std.mem.eql(u32, reenc.items, prompt_tok.items);
        if (round_trip_ok) {
            log.info("round-trip détok : PASS (decode -> re-encode == ids)", .{});
        } else {
            log.err("round-trip détok : FAIL — got={any} want={any}", .{ reenc.items, prompt_tok.items });
            return error.RoundTripFailed;
        }
        return;
    }

    // === Garde VRAM (docs/VRAM_CHECK_DESIGN.md) — avant tout travail GPU. Les modes host-only
    // (--selftest-inputs/--ids-only) ont déjà early-return au-dessus. `--selftest-gather` N'EST
    // PLUS host-only depuis L3 (spec [it.4]) : il compile un mini-graphe GPU (`SgFwd`) et passe
    // désormais PAR cette garde, comme le run normal (dispatché plus bas, après Platform.init).
    // Tourne AUSSI en --allow-cpu : ce flag ne force pas le CPU (l'init .cuda est tentée d'abord,
    // --allow-cpu ne tolère que le repli) — sur machine sans GPU, nvidia-smi absent → warn +
    // continue, donc pas de blocage à tort. Seul --force-vram saute la garde. ===
    if (args.force_vram) {
        log.warn("--force-vram : garde VRAM sautée (OOM possible en aval, assumé)", .{});
    } else {
        try checkVram(allocator, io);
    }

    // === --oracle : lit la fixture AVANT tout (positions[0] = seq_len attendu == ids.len ; fed =
    // la séquence de référence [s0,t1,…] à comparer à `generated`, cf note d'alignement en tête de
    // fichier). positions[0] == ids.len parce que le 1er step de génération de l'oracle FEED s0 à la
    // position ABSOLUE ids.len (s0 a été PRODUIT à la position ids.len-1, dernier token du prompt).
    var oracle_ids: ?[]i32 = null;
    defer if (oracle_ids) |fx| allocator.free(fx);
    if (args.oracle_path) |fixture_path| {
        var reg: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, fixture_path);
        defer reg.deinit();
        var file = try std.Io.Dir.cwd().openFile(io, fixture_path, .{ .mode = .read_only });
        defer file.close(io);
        const positions_fx = try readFixtureAlloc(i32, .i32, allocator, io, &reg, &file, "positions");
        defer allocator.free(positions_fx);
        if (positions_fx.len == 0) {
            log.err("--oracle : fixture 'positions' vide", .{});
            return error.EmptyFixture;
        }
        // Déviation assumée (longueur seule) : les prompt_ids complets ne vivent que dans le
        // manifest sidecar JSON — positions[0]==ids.len est le check le plus fort possible sur la
        // fixture seule ; un prompt FAUX de même longueur échouerait bruyamment au compare step 0.
        if (positions_fx[0] != @as(i32, @intCast(ids.items.len))) {
            log.err("--oracle : positions[0]={d} (seq_len fixture) != ids.len={d} (prompt rendu) — mismatch prompt/fixture", .{ positions_fx[0], ids.items.len });
            return error.OraclePromptMismatch;
        }
        const fed_fx = try readFixtureAlloc(i32, .i32, allocator, io, &reg, &file, "fed");
        if (fed_fx.len == 0) {
            allocator.free(fed_fx);
            log.err("--oracle : fixture 'fed' vide — un PASS à 0 step serait vacueux", .{});
            return error.EmptyFixture;
        }
        oracle_ids = fed_fx;
        log.info("--oracle : {d} steps de génération attendus (fed.len), prompt vérifié (ids.len={d} == positions[0])", .{ fed_fx.len, ids.items.len });
    }

    const max_tokens: usize = args.max_tokens orelse 200;
    const limit: usize = if (oracle_ids) |fx| fx.len else max_tokens;
    if (oracle_ids != null and args.max_tokens != null) {
        log.warn("--oracle actif : --max-tokens={d} ignoré (limite = fed.len = {d})", .{ args.max_tokens.?, limit });
    }

    // Garde-fous de lancement (hérités du clone J1 — mêmes asserts que les oracles historiques).
    if (ids.items.len + limit > @as(usize, @intCast(L_MAX))) {
        log.err("garde-fou : ids.len({d}) + limit({d}) > L_MAX({d})", .{ ids.items.len, limit, L_MAX });
        return error.SequenceTooLong;
    }
    if (ids.items.len >= @as(usize, @intCast(SLIDING_WINDOW))) {
        log.err("garde-fou : ids.len({d}) >= SLIDING_WINDOW({d})", .{ ids.items.len, SLIDING_WINDOW });
        return error.PromptTooLong;
    }

    // === Backend CUDA (+ repli auto) — copié gemma4_gen_long_gpu.zig:80-92, AVEC --no-prealloc
    // (mécanisme gen_long_gpu G2.1) : preallocate=false → BFC alloue à la demande, nvidia-smi
    // mesure la VRAM RÉELLEMENT utilisée (pas la réserve 0.90×libre) — requis U10 (piège 14). ===
    const platform: *zml.Platform = blk: {
        const cuda_opts: zml.platform.CreateOptions = .{ .cuda = .{ .allocator = .{ .bfc = .{ .preallocate = !args.no_prealloc, .memory_fraction = 0.90 } } } };
        if (zml.Platform.init(allocator, io, .cuda, cuda_opts)) |p| break :blk p else |_| {}
        log.warn("CUDA indisponible (libpjrt_cuda absent ?) — repli sur Platform.auto (probablement CPU).", .{});
        break :blk try zml.Platform.auto(allocator, io, .{});
    };
    defer platform.deinit(allocator);
    if (args.no_prealloc) log.info("--no-prealloc : preallocate=false — nvidia-smi mesure l'usage VRAM réel (pas la réserve BFC).", .{});
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

    // === Task 4 (plan L3) : --selftest-gather — mode GPU désormais (gather in-graph, spec
    // docs/L3_INGRAPH_DESIGN.md §5 SG) : dispatché ICI, APRÈS la garde VRAM + Platform.init +
    // garde CUDA dure + sharding, AVANT le chargement du modèle complet (SG ne charge que
    // la table emb via SgTabs, pas `Model`). Conséquence mécanique du déplacement : ce point est
    // en aval du check --prompt (cf `prompt_text` plus haut) — un --prompt factice est donc REQUIS (cf `usage`), et
    // --allow-cpu/--force-vram s'appliquent à SG exactement comme au run normal (aucun cas spécial).
    if (args.selftest_gather) |fixture_path| {
        try selftestGather(allocator, io, platform, sharding, args.ckpt, fixture_path);
        return;
    }

    var reg_ck = try registryFromFile(allocator, io, args.ckpt); // JAMAIS fromPath sur le packé (symlink HF -> blobs)
    var store_ck: zml.io.TensorStore = .fromRegistry(allocator, &reg_ck);
    const base = store_ck.view().withPrefix("model").withPrefix("language_model");
    // 12B : même `base` — le checkpoint w4a16-ct garde le préfixe `model.language_model.`.
    // G12Model = deux slices (48 G12LayerW + 40 v_proj indexés slidingSlot, asserté à l'init).
    const model: g12.G12Model = try .init(arena.allocator(), base);

    // Symboliques construits À LA MAIN (pas de fixture de store, cf tête de section) — mêmes shapes
    // que engine.Packed(.tables)/engine.Cache.
    const tok_sym = zml.Tensor.init(.{ 1, 1 }, .u32).withTags(.{ .b, .s });
    // Repli si le gather rank-2 ne compile pas (P5.4 n'a validé que des ids 1-D) : `tok_sym` en
    // `{ .s }` shape `[1]`, puis dans G12Step.forward : `.gather(.{ .voc = tok }).reshape(.{ 1, 1, D }).withTags(.{ .b, .s, .d })` (reshape layout-preserving + re-tag, piège ZML #1 connu).
    // Repli dtype : si le gather exige des indices i32, passer tok_sym/host en `.i32` (le vocab < 2^31, cast sans perte).
    // ⚠ Si le dtype/shape des indices change ICI, changer AUSSI le tok_sym de selftestGather (SG) —
    // sinon SG resterait vert en validant autre chose que ce que le runtime fait.
    const packed_sym = PackedLong{
        .embeds = zml.Tensor.init(.{ L_MAX, 1, 1, D }, .bf16).withTags(.{ .step, .b, .s, .d }),
        .embptls = zml.Tensor.init(.{ L_MAX, 1, 1, LF }, .bf16).withTags(.{ .step, .b, .s, .lf }),
        .cos_full = zml.Tensor.init(.{ L_MAX, 1, 1, HD_F }, .f32).withTags(.{ .step, .b, .s, .hd }),
        .sin_full = zml.Tensor.init(.{ L_MAX, 1, 1, HD_F }, .f32).withTags(.{ .step, .b, .s, .hd }),
        .positions = zml.Tensor.init(.{L_MAX}, .i32).withTags(.{.step}),
        // Fenêtre glissante en DONNÉE runtime ({} i32) — les masques sont générés in-graph par le
        // moteur (ingraphMaskLines) ; rebindable (contre-test de vacuité : window ← L_MAX).
        .window = zml.Tensor.init(.{}, .i32),
    };
    // GQA 12B (D5) : le cache sliding porte kvh=8 têtes KV (h=KVH_SL) ; full MQA h=KVH_FL=1.
    // Cache LINÉAIRE .k=L_MAX (R10) — la fenêtre 1024 est portée par le masque, PAS par la dim.
    const cache_sym = engine.Cache{
        .sl_k = zml.Tensor.init(.{ NUM_SLIDING_SLOTS, 1, KVH_SL, L_MAX, HD_S }, .f32).withTags(.{ .slot, .b, .h, .k, .hd }),
        .sl_v = zml.Tensor.init(.{ NUM_SLIDING_SLOTS, 1, KVH_SL, L_MAX, HD_S }, .f32).withTags(.{ .slot, .b, .h, .k, .hd }),
        .fl_k = zml.Tensor.init(.{ NUM_FULL_SLOTS, 1, KVH_FL, L_MAX, HD_F }, .f32).withTags(.{ .slot, .b, .h, .k, .hd }),
        .fl_v = zml.Tensor.init(.{ NUM_FULL_SLOTS, 1, KVH_FL, L_MAX, HD_F }, .f32).withTags(.{ .slot, .b, .h, .k, .hd }),
    };
    const ctrl_sym: engine.Ctrl = .initSymbolic();

    log.info("Materializing weights (store_ck, 48 couches + 40 v_proj + embed 2 Go) + Packed/Cache (HostInputs, zéros hors positions/cos/sin/masques) ...", .{});
    const eng_buf = try model.load(arena.allocator(), io, platform, &store_ck, &.{sharding});
    // 12B : pas de `Tabs` (embed_tokens_per_layer ABSENT du checkpoint, ple_dim=0 — plan Task 8 point 2).

    var host = try HostInputs.init(allocator);
    defer host.deinit(allocator);
    // Bufferized(PackedLong) assemblé À LA MAIN (motif E2, gemma4_engine_e2.zig:104-111) : chaque
    // champ = zml.Buffer.fromBytes depuis les slices host de Task 3 (mêmes shapes que packed_sym).
    const pk_buf = zml.Bufferized(PackedLong){
        .embeds = try zml.Buffer.fromBytes(io, platform, packed_sym.embeds.shape(), sharding, host.embeds_zero),
        .embptls = try zml.Buffer.fromBytes(io, platform, packed_sym.embptls.shape(), sharding, host.embptls_zero),
        .cos_full = try zml.Buffer.fromBytes(io, platform, packed_sym.cos_full.shape(), sharding, std.mem.sliceAsBytes(host.cos_full)),
        .sin_full = try zml.Buffer.fromBytes(io, platform, packed_sym.sin_full.shape(), sharding, std.mem.sliceAsBytes(host.sin_full)),
        .positions = try zml.Buffer.fromBytes(io, platform, packed_sym.positions.shape(), sharding, std.mem.sliceAsBytes(host.positions)),
        .window = try zml.Buffer.scalar(io, platform, @as(i32, @intCast(SLIDING_WINDOW)), .i32, sharding),
    };
    comptime std.debug.assert(SLIDING_WINDOW > 0); // garde runtime-window (spec §4.1)
    var cache_buf = zml.Bufferized(engine.Cache){
        .sl_k = try zml.Buffer.fromBytes(io, platform, cache_sym.sl_k.shape(), sharding, host.cache_sl_k),
        .sl_v = try zml.Buffer.fromBytes(io, platform, cache_sym.sl_v.shape(), sharding, host.cache_sl_v),
        .fl_k = try zml.Buffer.fromBytes(io, platform, cache_sym.fl_k.shape(), sharding, host.cache_fl_k),
        .fl_v = try zml.Buffer.fromBytes(io, platform, cache_sym.fl_v.shape(), sharding, host.cache_fl_v),
    };
    store_ck.deinit();
    reg_ck.deinit();
    mem_probe.logMem(io, "post-load (poids + Packed/Cache sur device)");

    // === Compile G12Step.forward (mono-graphe 48 couches — voie nominale R7 MAJ, cf tête de section) ===
    log.info("Compiling G12Step.forward (gather+scale 62.0+dequant W4+forwardStageGen 48 couches+topK, mono-graphe) ...", .{});
    const t_compile: std.Io.Timestamp = .now(io, .awake);
    var exe = try platform.compileFn(allocator, io, G12Step.forward, .{ model, tok_sym, packed_sym, cache_sym, ctrl_sym }, .{ .shardings = &.{sharding} });
    defer exe.deinit();
    log.info("  compile: {f}", .{t_compile.untilNow(io, .awake)});
    mem_probe.logMem(io, "post-compile (go/no-go)");

    // === --window-vacuity (U9-ii, pattern gemma4_vacuity_logits) : replay teacher-forcé
    // in-process des ids d'une passe 1 (--out-ids), MÊME executable, UNE compile — passe 2 avec
    // le masque sliding ÉLARGI rebindé en DONNÉES (masks_sliding <- contenu de masks_full,
    // causal plein : la fenêtre 1024 ne mord plus). Compare les logits par position (bits f32),
    // rapporte la première divergence. Attendu D10 quand S > 1024 : bit-identiques q <= 1023,
    // première divergence exactement à q = 1024. Le VERDICT du gate = Task 10 (ce mode RAPPORTE). ===
    if (args.window_vacuity) |replay_path| {
        if (oracle_ids != null) log.warn("--window-vacuity actif : --oracle ignoré (pas de décode libre dans ce mode)", .{});
        var reg_wv: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, replay_path);
        defer reg_wv.deinit();
        var file_wv = try std.Io.Dir.cwd().openFile(io, replay_path, .{ .mode = .read_only });
        defer file_wv.close(io);
        const replay_fx = try readFixtureAlloc(i32, .i32, allocator, io, &reg_wv, &file_wv, "ids");
        defer allocator.free(replay_fx);
        if (replay_fx.len == 0) {
            log.err("--window-vacuity : fixture 'ids' vide — un rapport à 0 step serait vacueux", .{});
            return error.EmptyFixture;
        }
        const s_total: usize = ids.items.len + replay_fx.len;
        if (s_total > @as(usize, @intCast(L_MAX))) {
            log.err("--window-vacuity : prompt({d}) + replay({d}) > L_MAX({d})", .{ ids.items.len, replay_fx.len, L_MAX });
            return error.SequenceTooLong;
        }
        // Séquence FED complète (teacher-forcée) : prompt puis ids rejoués — l'argmax est ignoré.
        const vocab_wv = model.embed_tokens.dim(.voc); // bounds-check (gather XLA CLAMPE en silence)
        const fed_seq = try allocator.alloc(u32, s_total);
        defer allocator.free(fed_seq);
        for (ids.items, 0..) |t, k| fed_seq[k] = t;
        for (replay_fx, 0..) |t, k| {
            if (t < 0 or t >= vocab_wv) {
                log.err("--window-vacuity : id rejoué hors vocab au step {d} : {d}", .{ k, t });
                return error.TokenOutOfRange;
            }
            fed_seq[ids.items.len + k] = @intCast(t);
        }
        // Fenêtre élargie rebindée en DONNÉES : seul le scalaire `window` change (← L_MAX : la
        // condition basse j >= p-(window-1) devient toujours vraie → sliding dégénère en causal
        // plein, exactement l'effet de l'ancien rebind masks_sliding ← masks_full). MÊME
        // executable, UNE compile — mécanisme U9-ii adapté in-graph (spec 2026-07-26 §4.5).
        const pk_wide = zml.Bufferized(PackedLong){
            .embeds = pk_buf.embeds,
            .embptls = pk_buf.embptls,
            .cos_full = pk_buf.cos_full,
            .sin_full = pk_buf.sin_full,
            .positions = pk_buf.positions,
            .window = try zml.Buffer.scalar(io, platform, @as(i32, @intCast(L_MAX)), .i32, sharding),
        };
        const voc_us: usize = @intCast(vocab_wv);
        const logits_p1 = try allocator.alloc(u32, s_total * voc_us); // bits f32 (compare BIT, pas de tolérance)
        defer allocator.free(logits_p1);
        var first_div: ?usize = null;
        var div_max_abs: f64 = 0;
        var n_ident: usize = 0;
        for (0..2) |pass| {
            log.info("WV passe {d}/2 : {d} steps teacher-forcés, masque sliding {s}", .{ pass + 1, s_total, if (pass == 0) "NORMAL (fenêtre 1024)" else "ÉLARGI (causal plein, rebind données)" });
            var cache_wv = zml.Bufferized(engine.Cache){
                .sl_k = try zml.Buffer.fromBytes(io, platform, cache_sym.sl_k.shape(), sharding, host.cache_sl_k),
                .sl_v = try zml.Buffer.fromBytes(io, platform, cache_sym.sl_v.shape(), sharding, host.cache_sl_v),
                .fl_k = try zml.Buffer.fromBytes(io, platform, cache_sym.fl_k.shape(), sharding, host.cache_fl_k),
                .fl_v = try zml.Buffer.fromBytes(io, platform, cache_sym.fl_v.shape(), sharding, host.cache_fl_v),
            };
            for (0..s_total) |step| {
                var tok_host = [1]u32{fed_seq[step]};
                var tok_buf = try zml.Buffer.fromBytes(io, platform, tok_sym.shape(), sharding, std.mem.sliceAsBytes(&tok_host));
                var step_buf = try zml.Buffer.scalar(io, platform, @as(u32, @intCast(step)), .u32, sharding);
                const ctrl_buf = zml.Bufferized(engine.Ctrl){ .step = step_buf };
                var call_args = try exe.args(allocator);
                var call_results = try exe.results(allocator);
                call_args.set(.{ eng_buf, tok_buf, if (pass == 0) pk_buf else pk_wide, cache_wv, ctrl_buf });
                exe.call(call_args, &call_results);
                var r_t5v, var r_t5i, var r_logits, const r_slk, const r_slv, const r_flk, const r_flv = call_results.get(struct {
                    zml.Buffer, zml.Buffer, zml.Buffer, zml.Buffer, zml.Buffer, zml.Buffer, zml.Buffer,
                });
                var lg_s = try r_logits.toSliceAlloc(allocator, io);
                defer lg_s.free(allocator);
                const lg = lg_s.items(f32);
                if (lg.len != voc_us) {
                    log.err("WV : logits — {d} f32 != {d}", .{ lg.len, voc_us });
                    return error.UnexpectedShape;
                }
                if (pass == 0) {
                    for (lg, 0..) |v, i| logits_p1[step * voc_us + i] = @bitCast(v);
                } else {
                    var same = true;
                    var step_max: f64 = 0;
                    for (lg, 0..) |v, i| {
                        const b1 = logits_p1[step * voc_us + i];
                        if (@as(u32, @bitCast(v)) != b1) {
                            same = false;
                            const d_abs = @abs(@as(f64, v) - @as(f64, @as(f32, @bitCast(b1))));
                            if (d_abs > step_max) step_max = d_abs;
                        }
                    }
                    if (same) {
                        n_ident += 1;
                    } else if (first_div == null) {
                        first_div = step;
                        div_max_abs = step_max;
                    }
                }
                var old_cache = cache_wv;
                cache_wv = zml.Bufferized(engine.Cache){ .sl_k = r_slk, .sl_v = r_slv, .fl_k = r_flk, .fl_v = r_flv };
                old_cache.sl_k.deinit();
                old_cache.sl_v.deinit();
                old_cache.fl_k.deinit();
                old_cache.fl_v.deinit();
                r_t5v.deinit();
                r_t5i.deinit();
                r_logits.deinit();
                tok_buf.deinit();
                step_buf.deinit();
                call_args.deinit(allocator);
                call_results.deinit(allocator);
                if ((step + 1) % 256 == 0) log.info("  WV passe {d} ... step {d}/{d}", .{ pass + 1, step + 1, s_total });
            }
            cache_wv.sl_k.deinit();
            cache_wv.sl_v.deinit();
            cache_wv.fl_k.deinit();
            cache_wv.fl_v.deinit();
        }
        if (first_div) |q| {
            log.info("WINDOW-VACUITY : logits bit-identiques sur {d} positions, PREMIÈRE DIVERGENCE à q={d} (max_abs à q : {e:.3}) — attendu D10 : q == 1024 exactement (verdict = Task 10/U9-ii)", .{ n_ident, q, div_max_abs });
        } else {
            log.warn("WINDOW-VACUITY : AUCUNE divergence sur {d} positions — la fenêtre n'a pas mordu (S <= 1024 ?) : rapport VACUEUX pour U9-ii si S > 1024 attendu", .{s_total});
        }
        return;
    }

    // === Step 5.2 → L3 : boucle prefill-par-decode + topK in-graph + arrêt ===
    var generated: std.ArrayList(i64) = .empty;
    defer generated.deinit(allocator);
    var gen_top5: std.ArrayList(Top5) = .empty; // parallèle à `generated` (diagnostic FAIL, Step 5.3)
    defer gen_top5.deinit(allocator);

    log.info("Boucle autonome : {d} steps de prefill, puis génération (limite {d}{s})", .{ ids.items.len, limit, if (oracle_ids != null) " = fed.len, oracle" else " = max_tokens" });

    // Raison d'arrêt (A3) : capturée DANS la boucle (pas reconstruite après coup) — le strip EOT
    // du détok et le verdict A3 en dépendent. `.oracle` = sortie par compte fed.len (mode --oracle).
    const StopReason = enum { oracle, eot, max_tokens, l_max };
    var stop_reason: StopReason = .oracle;

    var fed: i64 = @intCast(ids.items[0]);
    var step: usize = 0;
    // Bounds-check (fix revue) : lu UNE FOIS avant la boucle — reprend l'invariant de feu
    // `EmbedGather` : XLA `gather` CLAMPE silencieusement les indices hors-borne (une divergence
    // de logits plausible mais fausse, pas un crash) et `@intCast(fed)` ci-dessous vers u32 est UB
    // en ReleaseFast si `fed` sort de la plage. Portée : couvre le chemin `fed` (host→device) ;
    // `t5i` issu de topK/arange est ≥ 0 par construction (pas re-borné au cast usize).
    const vocab = model.embed_tokens.dim(.voc);
    const t0: std.Io.Timestamp = .now(io, .awake);
    // Step 2.7 (spec [it.6]) : capturé au dernier step de prefill (cf plus bas), initialisé à t0
    // par sûreté (jamais réellement lu à cette valeur — le prefill compte toujours ≥1 step).
    var t_prefill_end: std.Io.Timestamp = t0;
    while (true) : (step += 1) {
        // L3 (spec §2.2) : le host ne thread plus qu'un scalaire u32 (le token à feeder) — gather
        // + forwardStageGen + topK composés IN-GRAPH par G12Step (plus de Buffer.fromBytes embeds/
        // embptls host, plus de EmbedGather).
        if (fed < 0 or fed >= vocab) {
            log.err("token hors vocab: {d} (vocab={d})", .{ fed, vocab });
            return error.TokenOutOfRange;
        }
        var tok_host = [1]u32{@intCast(fed)};
        var tok_buf = try zml.Buffer.fromBytes(io, platform, tok_sym.shape(), sharding, std.mem.sliceAsBytes(&tok_host));
        var step_buf = try zml.Buffer.scalar(io, platform, @as(u32, @intCast(step)), .u32, sharding);
        const ctrl_buf = zml.Bufferized(engine.Ctrl){ .step = step_buf };

        var call_args = try exe.args(allocator);
        var call_results = try exe.results(allocator);
        call_args.set(.{ eng_buf, tok_buf, pk_buf, cache_buf, ctrl_buf });
        exe.call(call_args, &call_results);
        // r_logits : sortie supplémentaire (--window-vacuity) — NON lue ici (pas de D2H), deinit direct.
        var r_t5v, var r_t5i, var r_logits, const r_slk, const r_slv, const r_flk, const r_flv = call_results.get(struct {
            zml.Buffer, zml.Buffer, zml.Buffer, zml.Buffer, zml.Buffer, zml.Buffer, zml.Buffer,
        });

        // top5 : TOUJOURS calculé in-graph (cheap, cf PLAN), ignoré tant qu'on est en prefill (sauf
        // le dernier prefill step, qui produit s0 — cf `in_gen_phase` ci-dessous).
        const in_gen_phase = step + 1 >= ids.items.len;

        // Top5 depuis le device (~48 octets D2H) — top1 = next token (spec §2, §4 ties d'argmax).
        var t5v_s = try r_t5v.toSliceAlloc(allocator, io);
        defer t5v_s.free(allocator);
        var t5i_s = try r_t5i.toSliceAlloc(allocator, io);
        defer t5i_s.free(allocator);
        // dtype confirmé : `topK` délègue à `sort` (tensor.zig:3096), dont les indices sont
        // produits par `Tensor.arange(…, .i32)` (tensor.zig:2977) — vérifié À CHAQUE step (coût
        // nul : compare d'enum) plutôt que supposé silencieusement.
        if (t5i_s.dtype() != .i32) {
            log.err("t5.indices : dtype={s} ≠ i32 attendu (topK/sort, tensor.zig:2977)", .{@tagName(t5i_s.dtype())});
            return error.UnexpectedDtype;
        }
        const t5i = t5i_s.items(i32);
        const t5v = t5v_s.items(f32);
        var top5: Top5 = undefined;
        for (0..5) |j| {
            top5.idx[j] = @intCast(t5i[j]);
            top5.val[j] = t5v[j];
        }
        const tok: i64 = @intCast(top5.idx[0]);
        if (in_gen_phase) try gen_top5.append(allocator, top5);
        // W4g (protocole de flip) : marge top1−top2 par step de génération, mode oracle seulement.
        if (in_gen_phase and oracle_ids != null) log.info("  marge top1-top2 @ gen={d} : {d:.6} (top1={d} top2={d})", .{ gen_top5.items.len - 1, top5.val[0] - top5.val[1], top5.idx[0], top5.idx[1] });
        // --dump-top5 (U9) : top-5 par step aussi en mode LIBRE (w4auto ne le loggait qu'en --oracle).
        if (in_gen_phase and args.dump_top5) log.info("  top5 @ gen={d} : idx={any} val={any}", .{ gen_top5.items.len - 1, top5.idx, top5.val });

        // cache swap (motif gemma4_gen_long_gpu.zig:139-168) : deinit l'ancien, adopte le nouveau.
        var old_cache = cache_buf;
        cache_buf = zml.Bufferized(engine.Cache){ .sl_k = r_slk, .sl_v = r_slv, .fl_k = r_flk, .fl_v = r_flv };
        old_cache.sl_k.deinit();
        old_cache.sl_v.deinit();
        old_cache.fl_k.deinit();
        old_cache.fl_v.deinit();

        r_t5v.deinit();
        r_t5i.deinit();
        r_logits.deinit();
        tok_buf.deinit();
        step_buf.deinit();
        call_args.deinit(allocator);
        call_results.deinit(allocator);

        // Progression périodique (motif gemma4_gen_long_gpu.zig:158) — premier signe humain d'une
        // anomalie pendant un run long (A2) : silence prolongé = suspect.
        if ((step + 1) % 256 == 0) log.info("  ... step {d} ({d} générés)", .{ step + 1, generated.items.len });

        // Step 2.7 : fin du DERNIER step de prefill (nuance de mesure, cf log L3 PERF plus bas).
        if (step + 1 == ids.items.len) t_prefill_end = .now(io, .awake);

        if (step + 1 < ids.items.len) {
            // Phase 1 (prefill) : argmax ci-dessus IGNORÉ (pas le dernier token du prompt).
            fed = @intCast(ids.items[step + 1]);
            continue;
        }
        // Phase 2 (génération, s0 INCLUS dès le 1er passage ici — dernier step de prefill).
        try generated.append(allocator, tok);
        if (oracle_ids) |fx| {
            if (generated.items.len >= fx.len) break; // stop_reason reste .oracle
        } else {
            if (tok == @as(i64, @intCast(eot_id))) {
                stop_reason = .eot;
                break;
            }
            if (generated.items.len >= max_tokens) {
                stop_reason = .max_tokens;
                break;
            }
        }
        if (step + 1 >= @as(usize, @intCast(L_MAX))) {
            stop_reason = .l_max;
            log.warn("garde L_MAX atteinte (step={d}) — arrêt forcé", .{step});
            break;
        }
        fed = tok;
    }
    const elapsed = t0.untilNow(io, .awake);
    // Step 2.7 (spec [it.6], fix revue) : `gen_elapsed` échantillonné ICI, IMMÉDIATEMENT après
    // `elapsed` — AVANT les 4 deinit de cache ci-dessous. Si on le capturait après, leur coût
    // (libération device, potentiellement non négligeable) polluerait la fenêtre gen_s et
    // fausserait le tok/s de génération à la marge. Les deux `.untilNow` sont ainsi resamplés à
    // quelques ns d'écart l'un de l'autre (back-to-back, aucun travail entre les deux) —
    // négligeable à cette échelle.
    const gen_elapsed = t_prefill_end.untilNow(io, .awake);
    cache_buf.sl_k.deinit();
    cache_buf.sl_v.deinit();
    cache_buf.fl_k.deinit();
    cache_buf.fl_v.deinit();

    const elapsed_ns = elapsed.toNanoseconds();
    const elapsed_s = @as(f64, @floatFromInt(elapsed_ns)) / std.time.ns_per_s;

    // Step 2.7 (spec [it.6]) : perf prefill/génération séparées.
    // Nuance de mesure assumée : s0 est produit par le DERNIER call de prefill mais compté dans
    // `generated` — le 1er token de gén coûte ~0 s dans la fenêtre gen_s. Négligeable dès ~48
    // steps ; ne pas s'en étonner en comparant B0↔G3.
    // API : `.untilNow` (seul mécanisme de mesure déjà utilisé/validé dans ce fichier, cf t0/
    // t_compile ci-dessus) appliqué à `t_prefill_end` donne DIRECTEMENT la durée écoulée depuis la
    // fin du prefill jusqu'à MAINTENANT (== gen_s) ; pf_s se déduit par soustraction. Choix
    // délibéré vs le libellé `.since()` du plan L3_INGRAPH_PLAN.md Step 2.7 : aucune méthode de
    // différence entre deux Timestamp PASSÉS n'est visible dans les sources locales (build
    // distant, non vérifiable ici) — on reste sur l'API prouvée plutôt que de parier sur un nom
    // de méthode non confirmé (cf revue Step 2.9).
    const gen_s = @as(f64, @floatFromInt(gen_elapsed.toNanoseconds())) / std.time.ns_per_s;
    const pf_s = elapsed_s - gen_s;
    const pf_rate = if (pf_s > 0) @as(f64, @floatFromInt(ids.items.len)) / pf_s else 0;
    const gen_rate = if (gen_s > 0) @as(f64, @floatFromInt(generated.items.len)) / gen_s else 0;
    log.info("PERF : prefill {d} steps en {d:.3}s ({d:.1} tok/s) ; génération {d} tokens en {d:.3}s ({d:.1} tok/s)", .{ ids.items.len, pf_s, pf_rate, generated.items.len, gen_s, gen_rate });

    // === --out-ids (U9) : ids générés -> safetensors (clé "ids", i32) — AVANT le verdict oracle
    // (un A1Mismatch ne doit pas perdre la trace des ids produits, utile au diagnostic). ===
    if (args.out_ids) |out_path| {
        try writeIdsSafetensors(allocator, io, out_path, generated.items);
        log.info("--out-ids : {d} ids écrits -> {s}", .{ generated.items.len, out_path });
    }

    // === Gate oracle (A1, hérité w4auto — U8 l'utilisera avec la fixture u8_gen48) ===
    if (oracle_ids) |fx| {
        var n_match: usize = 0;
        var first_fail: ?usize = null;
        const n = @min(generated.items.len, fx.len);
        for (0..n) |k| {
            if (generated.items[k] == @as(i64, @intCast(fx[k]))) {
                n_match += 1;
            } else if (first_fail == null) {
                first_fail = k;
            }
        }
        const len_ok = generated.items.len == fx.len;
        if (first_fail == null and len_ok) {
            log.info("A1 PASS — {d}/{d} argmax-match (autonome complet, zéro input fixture)", .{ n_match, fx.len });
        } else {
            const ff = first_fail orelse n;
            log.err("A1 FAIL — {d}/{d} match, 1er mismatch au step gen={d}{s}", .{ n_match, fx.len, ff, if (!len_ok) " (ou longueurs différentes)" else "" });
            if (ff < fx.len) {
                const got: i64 = if (ff < generated.items.len) generated.items[ff] else -1;
                log.err("  step gen={d} : généré={d} attendu(fed)={d}", .{ ff, got, fx[ff] });
            }
            if (ff < gen_top5.items.len) {
                const t5 = gen_top5.items[ff];
                log.err("  diagnostic LOGITS (méthodo : argmax trop grossier pour diagnostiquer) — top-5 @ step gen={d} : idx={any} val={any}", .{ ff, t5.idx, t5.val });
            }
            return error.A1Mismatch;
        }
    } else {
        // === Task 7 (gate A3) : mode libre — ids générés → décodeur ZML → texte stdout (spec §2) ===
        log.info("mode libre : {d} tokens générés (EOT_ID={d}, max_tokens={d})", .{ generated.items.len, eot_id, max_tokens });
        log.info("generated = {any}", .{generated.items});
        switch (stop_reason) {
            .eot => log.info("arrêt : early-stop EOT", .{}),
            .max_tokens => log.info("arrêt : max-tokens ({d})", .{max_tokens}),
            .l_max => log.info("arrêt : garde L_MAX", .{}),
            .oracle => unreachable, // .oracle n'est atteignable qu'avec --oracle (branche du dessus)
        }

        // Détok : strip du EOT FINAL si l'arrêt vient de l'EOS (le texte de la réponse ne contient
        // pas le token de fin de tour) ; sinon tout `generated` est du texte.
        const n_text = if (stop_reason == .eot) generated.items.len - 1 else generated.items.len;
        // Conversion i64→u32 EXPLICITE à la frontière du décodeur (piège de revue : pas de
        // reinterprétation de slice — boucle élément par élément, @intCast borné par le vocab).
        const ids_u32 = try allocator.alloc(u32, n_text);
        defer allocator.free(ids_u32);
        for (generated.items[0..n_text], 0..) |t, k| ids_u32[k] = @intCast(t);
        // Décodeur FRAIS pour ce décodage final (piège de revue : l'automate iree est à état —
        // ne pas réutiliser un décodeur partiellement consommé ; NB reset() retourne !void).
        var decoder = try tokenizer.decoder();
        defer decoder.deinit();
        var text = try decoder.decodeAlloc(allocator, ids_u32);
        defer text.deinit(allocator);

        // Texte final sur STDOUT (les logs vont sur stderr) — dernier maillon du pipeline spec §2.
        var stdout_w = std.Io.File.stdout().writer(io, &.{});
        try stdout_w.interface.print("réponse : \"{s}\"\n", .{text.items});
        try stdout_w.interface.flush();

        // Verdict A3 (le critère numérique N == index_EOT_expected + 2 est vérifié côté contrôleur
        // contre la fixture ; ici on rend N et la raison d'arrêt VISIBLES).
        if (stop_reason == .eot) {
            log.info("A3 : stop early-EOT après {d} tokens (dernier = EOT_ID={d})", .{ generated.items.len, eot_id });
        }
    }
}

};
}

pub fn main(init: std.process.Init) !void {
    return G12Auto(1280).run(init);
}
