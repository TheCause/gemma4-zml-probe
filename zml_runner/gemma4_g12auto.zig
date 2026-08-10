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
//       [--max-tokens N] [--oracle fixture] [--ids-only] [--ids-only-turn2] [--allow-cpu] [--force-vram]
//       [--dump-top5] [--out-ids f] [--window-vacuity f] [--no-prealloc]
//       [--selftest-inputs f] [--selftest-gather f (mode GPU, requiert un --prompt factice)]
//       [--selftest-gencfg f (GC1 : politique generation_config, host-only, sans GPU ni tokenizer)]
//       [--selftest-sampling f (S2-U : warpers top_k/top_p/temperature, host-only)]
//       [--selftest-draw f] [--selftest-alloc-count (S-AC, host-only)] [--no-pin]
//       [--gen-config FICHIER] [--no-gen-config]
//       [--temperature F] [--top-k N] [--top-p F] [--min-tokens-to-keep N] [--seed N]
// Politique de décodage (spec 2026-07-28) : `suppress_tokens` + EOS multiples de
// generation_config.json, appliqués HOST-SIDE sur le top-5 rapatrié. Découverte automatique à
// côté du checkpoint (1 hop de symlink) ; `--gen-config` force le fichier, `--no-gen-config` la
// désactive. Avec le sampling phase 2 (chemin B host-side : top_k/top_p/temperature + seed),
// 6 des 8 clés sont appliquées — restent bos/pad, sans objet au décodage (SAMPLING_RESULTS §4).
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
// Politique de décodage `generation_config.json` (suppress_tokens + EOS multiples), HOST-SIDE :
// spec 2026-07-28-generation-config-design §4.2 — `engine.zig` reste à 0 octet, le graphe ne
// bouge pas (GC0 le prouve). Le module porte AUSSI `TOP_K`, déclaration unique de la constante 5.
const gencfg = @import("gencfg.zig");
// Warpers de sampling (spec 2026-07-29 phase 2) — fonctions pures, host-side, hors du graphe.
const sampling = @import("sampling.zig");
const sampling_ref = @import("sampling_ref.zig");
// D10 : compteur d'allocations (spec 2026-07-30 zero-alloc, C1) — wrapper std-only de init.gpa,
// toujours actif ; porteur des gates AL-0/AL-VAC/AL-BASE. `builtin.mode` : bannière BUILD (§8).
const alloc_count = @import("alloc_count.zig");
// Dump/restore du KV-cache (spec 2026-08-09) — module std-only : writer/reader safetensors
// généralisé, manifest `__metadata__`, checksums xxh64. Aucun octet de graphe (DC0).
const kvdump = @import("kvdump.zig");
const builtin = @import("builtin");

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

// Prompt texte → ids [BOS ++ template rendu] (spec repl-mode §1). Extraction du bloc inline
// historique de run() pour être appelable PAR PROMPT en mode résident. `rendered` est alloué
// sur `allocator` + free (fix revue : l'ancien passage par l'arena de run ne se vidait jamais
// → croissance host par prompt en résident). L'encoder iree est un automate à ÉTAT : reset()
// systématique avant encode (invariant historique, cf round-trip --ids-only).
fn promptToIds(allocator: std.mem.Allocator, encoder: anytype, prompt_text: []const u8) !std.ArrayList(u32) {
    encoder.reset();
    const rendered = try renderChatTemplate(allocator, prompt_text);
    defer allocator.free(rendered);
    var prompt_tok = try encoder.encodeAlloc(allocator, rendered);
    defer prompt_tok.deinit(allocator);
    var ids: std.ArrayList(u32) = try .initCapacity(allocator, prompt_tok.items.len + 1);
    errdefer ids.deinit(allocator);
    try ids.append(allocator, BOS_ID);
    try ids.appendSlice(allocator, prompt_tok.items);
    return ids;
}

// K5 — rendu du TOUR 2 (spec 2026-08-10 §4.4) : le suffixe canonique post-clôture.
//
// VÉRITÉ = la MESURE HF de la Task 2 (docs/evidence/k5/rendu_tour2_hf.json), pas ce commentaire :
// apply_chat_template sur [user1, assistant1, user2] avec generation prompt, moins le préfixe
// commun avec [user1, assistant1] SANS generation prompt. Résultat mesuré (template sha
// ae53464b…, transformers 5.14) : le suffixe vaut 17 ids
//   [105,2364,107, <user2>, 106,107,105,4368,107,100,45518,107,101]
// — c'est LITTÉRALEMENT le rendu du tour 1 privé de son BOS. D'où la délégation ci-dessous :
// dupliquer le littéral le ferait diverger en silence le jour où le tour 1 change.
//
// ⚠ CE QUE LE PLAN PRÉVOYAIT ET QUE LA MESURE A CORRIGÉ : le gabarit planifié commençait par un
// '\n'. FAUX — la mesure donne closure_tail_ids = [106, 107] : le '\n' (107) qui suit le
// `<turn|>` appartient à la CLÔTURE DU TOUR 1, pas au tour 2. Il est injecté dans ids_full
// (D-K5-5), et PF6 compare le suffixe SEUL. Le laisser ici aurait fait échouer PF6 sur son id
// de tête — et le plan interdit de « corriger » un tel FAIL en retranchant l'id : il se
// diagnostique contre closure_tail_ids, ce qui a été fait AVANT d'écrire cette ligne.
fn renderChatTemplateTurn2(allocator: std.mem.Allocator, prompt: []const u8) ![]u8 {
    return renderChatTemplate(allocator, prompt);
}

// Tour 2 → ids, SANS BOS (le BOS est le préfixe du SEUL tour 1, `:95`). Même hygiène que
// promptToIds : reset() avant encode, l'encoder iree est un automate à état.
fn promptToIdsTurn2(allocator: std.mem.Allocator, encoder: anytype, prompt_text: []const u8) !std.ArrayList(u32) {
    encoder.reset();
    const rendered = try renderChatTemplateTurn2(allocator, prompt_text);
    defer allocator.free(rendered);
    var prompt_tok = try encoder.encodeAlloc(allocator, rendered);
    defer prompt_tok.deinit(allocator);
    var ids: std.ArrayList(u32) = try .initCapacity(allocator, prompt_tok.items.len);
    errdefer ids.deinit(allocator);
    try ids.appendSlice(allocator, prompt_tok.items);
    return ids;
}

// K5/D-K5-5 — CLÔTURE du tour 1, tokenisée et non devinée. Le rendu HF ferme toujours le tour
// assistant avant le suivant, et la mesure dit avec quoi : `<turn|>` (106) PUIS '\n' (107).
// `fed_next` déjà EOS ⇒ seul le '\n' reste à poser. Les longueurs attendues sont GARDÉES : un
// tokenizer qui découperait autrement doit BLOQUER, pas produire un contexte silencieusement
// différent de celui que HF verrait (même politique que l'eot_id mesuré, `:2058`).
fn closureToIds(allocator: std.mem.Allocator, encoder: anytype, fed_next_is_eos: bool) !std.ArrayList(u32) {
    encoder.reset();
    const text: []const u8 = if (fed_next_is_eos) "\n" else "<turn|>\n";
    const want: usize = if (fed_next_is_eos) 1 else 2;
    var tok = try encoder.encodeAlloc(allocator, text);
    defer tok.deinit(allocator);
    if (tok.items.len != want) {
        log.err("K5 : clôture '{s}' encode en {d} ids (attendu {d}) — ids={any} ; tokenizer différent de celui mesuré (docs/evidence/k5/rendu_tour2_hf.json)", .{ text, tok.items.len, want, tok.items });
        return error.ClosureNotExpectedLength;
    }
    var ids: std.ArrayList(u32) = try .initCapacity(allocator, tok.items.len);
    errdefer ids.deinit(allocator);
    try ids.appendSlice(allocator, tok.items);
    return ids;
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
    // K5/PF6 : rendu du TOUR 2 en ids, host-only. Un mode à part et non une option de --ids-only :
    // le tour 2 n'a de sens que par contraste avec le tour 1 (pas de BOS, clôture non incluse), et
    // le gate doit pouvoir le lire SANS dump ni GPU.
    ids_only_turn2: bool = false,
    allow_cpu: bool = false,
    selftest_inputs: ?[]const u8 = null,
    selftest_gather: ?[]const u8 = null,
    selftest_gencfg: ?[]const u8 = null, // GC1 : politique de décodage, host-only (spec §4.5bis)
    selftest_sampling: ?[]const u8 = null, // S2-U : warpers de sampling, host-only (spec phase 2)
    selftest_draw: ?[]const u8 = null, // S2-D : tirage sur logits FIGÉS en fixture, host-only
    selftest_penalty: ?[]const u8 = null, // RP1 : penalty comparée 0 ULP au processor HF, host-only
    selftest_alloc_count: bool = false, // S-AC (D10) : le compteur compte, host-only
    selftest_kvdump_io: ?[]const u8 = null, // DC1 : round-trip fichier + mutant, host-only (dir existant)
    selftest_kvdump_eq: ?[]const u8 = null, // DC2 : équivalence intra-process (GPU) ; valeur = fichier de dump de travail
    draws: usize = 10000,
    // Sampling phase 2 — défauts NEUTRES : sans eux, le chemin B n'est pas armé et le code
    // d'avant le chantier est strictement inchangé.
    temperature: f32 = 1.0,
    top_k: u32 = 0, // 0 = désactivé (convention HF)
    top_p: f32 = 1.0,
    min_tokens_to_keep: u32 = 1,
    seed: ?u64 = null, // null, PAS 0 : 0 est une graine légitime, pas un sentinel
    // Phase 1 (repetition penalty) — défauts NEUTRES, même discipline que la phase 2 : à 1.0 le
    // chemin est un no-op par construction et le code d'avant le chantier est inchangé (gate RP2).
    repetition_penalty: f32 = 1.0,
    ignore_prompt: bool = false, // pénaliser les seuls tokens GÉNÉRÉS (HF pénalise aussi le prompt)
    gate_d1d2: bool = false, // gates G-D1/G-D2 : pont in-process contre une référence indépendante
    // Politique de décodage (spec 2026-07-28) : chemin EXPLICITE d'un generation_config.json —
    // un FICHIER, jamais un répertoire. Sert D11, dont le checkpoint corrompu est écrit à plat
    // (sans snapshot ni symlink) : sans ce flag, la découverte échouerait et un gate historique
    // mourrait en silence.
    gen_config: ?[]const u8 = null,
    // Échappatoire explicite : restaure EXACTEMENT le comportement d'avant le chantier. Ce n'est
    // pas une commodité, c'est l'instrument du contre-test de non-vacuité GC4(a) — même binaire,
    // une donnée de moins.
    no_gen_config: bool = false,
    force_vram: bool = false,
    // Nouveaux flags 12B (plan Task 8 point 8 — AUCUN n'existait dans la base clonée w4auto) :
    dump_top5: bool = false, // top-5 par step en mode LIBRE (requis U9)
    out_ids: ?[]const u8 = null, // ids générés -> safetensors (requis U9-ii replay / U9-iv teacher-forcing)
    window_vacuity: ?[]const u8 = null, // U9-ii : replay teacher-forcé, masque sliding élargi rebindé en DONNÉES
    no_prealloc: bool = false, // U10 : preallocate=false, nvidia-smi mesure la VRAM réelle (mécanisme gen_long_gpu)
    no_pin: bool = false, // D10 (C7) : désactive l'allocation DMA (pinned) de work — A/B de M-PIN
    repl: bool = false, // mode RÉSIDENT (spec 2026-07-26 repl-mode) : compile une fois, prompts en boucle sur stdin
    // Dump/restore du KV-cache (spec 2026-08-09) — l'état E1-E4 d'une génération dans UN
    // safetensors auto-décrivant ; le restore repart sans re-prefill.
    dump_cache: ?[]const u8 = null, // --dump-cache <fichier> : état E1-E4 en fin de generateOnce
    load_cache: ?[]const u8 = null, // --load-cache <fichier> : reprise sans prefill
};

const usage =
    "Usage: gemma4_g12auto <model.safetensors (checkpoint 12B w4a16-ct, weights_12b)> <tokenizer.json> --prompt \"...\" " ++
    "[--max-tokens N] [--oracle fixture] [--ids-only] [--allow-cpu (débogage uniquement)] " ++
    "[--force-vram] [--dump-top5] [--out-ids f] [--window-vacuity ids.safetensors] [--no-prealloc] " ++
    "[--selftest-inputs f] [--selftest-gather f (requiert un --prompt factice)] " ++
    "[--selftest-gencfg f (GC1 : fixture + sidecar .manifest.json ; host-only)] " ++
    "[--selftest-sampling f (S2-U : fixture warpers + sidecar ; host-only)] " ++
    "[--selftest-draw f --draws N --seed S (S2-D : tirage sur logits figés ; host-only)] " ++
    "[--selftest-penalty f (RP1 : penalty vs processor HF, 0 ULP ; host-only)] " ++
    "[--selftest-alloc-count (S-AC : compteur d'allocations ; host-only)] " ++
    "[--selftest-kvdump-io DIR (DC1 : round-trip kvdump + mutant ; host-only ; DIR doit exister)] " ++
    "[--selftest-kvdump-eq F (DC2 : équivalence intra-process du restore ; GPU ; requiert --prompt)] " ++
    "[--no-pin (désactive l'alloc DMA pinned de work — A/B M-PIN)] " ++
    "[--temperature F] [--top-k N] [--top-p F] [--min-tokens-to-keep N] [--seed N] " ++
    "(sampling phase 2 ; sans --seed la sélection reste un argmax) " ++
    "[--repetition-penalty F (phase 1 ; > 0 fini ; 1.0 = neutre)] " ++
    "[--ignore-prompt (ne pénaliser que les tokens GÉNÉRÉS ; HF pénalise aussi le prompt ; " ++
    "exclut --load-cache — garde CONSERVÉE par K5)] " ++
    "[--gate-d1d2 (G-D1/G-D2 : applyTopP comparé à une référence descendante f64 + mutants " ++
    "température ; exige un régime ARMÉ ; invalide la mesure M-COUT du même run)] " ++
    "[--gen-config FICHIER (generation_config.json explicite — un fichier, pas un répertoire)] " ++
    "[--no-gen-config (désactive la politique de décodage : comportement d'avant le chantier)] " ++
    "[--repl (résident : prompts en boucle sur stdin ; --prompt devient optionnel = 1er prompt ; " ++
    "exclusif de --oracle/--window-vacuity/--out-ids/--ids-only/--selftest-*)] " ++
    "[--dump-cache F (état KV+ids en fin de génération -> safetensors ; exclut --repl et --seed)] " ++
    "[--load-cache F (reprise sans prefill ; avec --max-tokens ou --oracle ; exclut --repl) " ++
    "[--prompt \"tour 2\"] = PREFILL PARTIEL (K5) : le contexte vient du dump, le prompt neuf est " ++
    "absorbé comme tour suivant] " ++
    "[--ids-only-turn2 (K5/PF6 : rendu du tour 2 en ids, host-only ; exige --prompt)]";

// Parsing à la main (comme les runners existants, ex. gemma4_gen_long_gpu.zig --no-prealloc) :
// pas de lib de flags ici, juste un balayage séquentiel des positionnels puis des --flags.
// Type EXACT du retour de std.process.Args.toSlice (cf lib/std/process/Args.zig) : une slice
// d'éléments sentinelle-terminés — chaque élément coerce vers []const u8 mais la slice ENTIÈRE
// ne coerce PAS vers []const []const u8 (piège de typage, d'où la signature précise ici).
/// Validation UNIQUE de la repetition penalty — partagée par le flag CLI et la directive `:penalty`
/// du repl. Deux implémentations dériveraient : la CLI refuserait `nan` et le repl l'accepterait,
/// et c'est précisément par le repl que l'utilisateur explore les valeurs.
///
/// ⚠ Garde en ACCEPTATION, même raison qu'à `--temperature` : la forme « p <= 0 → rejet »
/// laisserait passer NaN (toute comparaison avec NaN est fausse). Avec p = NaN, `p != 1.0` est
/// VRAI, la penalty s'armerait, chaque logit pénalisé deviendrait NaN, et l'argmax rendrait un
/// token arbitraire SANS erreur.
fn parsePenalty(s: []const u8) ?f32 {
    const v = std.fmt.parseFloat(f32, s) catch return null;
    if (!(v > 0 and std.math.isFinite(v))) return null;
    return v;
}

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
        } else if (std.mem.eql(u8, a, "--ids-only-turn2")) {
            args.ids_only_turn2 = true;
        } else if (std.mem.eql(u8, a, "--allow-cpu")) {
            args.allow_cpu = true;
        } else if (std.mem.eql(u8, a, "--force-vram")) {
            args.force_vram = true;
        } else if (std.mem.eql(u8, a, "--dump-top5")) {
            args.dump_top5 = true;
        } else if (std.mem.eql(u8, a, "--no-prealloc")) {
            args.no_prealloc = true;
        } else if (std.mem.eql(u8, a, "--repl")) {
            args.repl = true;
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
        } else if (std.mem.eql(u8, a, "--selftest-gencfg")) {
            i += 1;
            if (i >= process_args.len) {
                log.err("--selftest-gencfg attend une valeur (fixture .safetensors ; le sidecar <fixture>.manifest.json est requis à côté)", .{});
                return error.MissingArgument;
            }
            args.selftest_gencfg = process_args[i];
        } else if (std.mem.eql(u8, a, "--selftest-sampling")) {
            i += 1;
            if (i >= process_args.len) {
                log.err("--selftest-sampling attend une valeur (fixture .safetensors ; le sidecar <fixture>.manifest.json est requis à côté)", .{});
                return error.MissingArgument;
            }
            args.selftest_sampling = process_args[i];
        } else if (std.mem.eql(u8, a, "--selftest-draw")) {
            i += 1;
            if (i >= process_args.len) return error.MissingArgument;
            args.selftest_draw = process_args[i];
        } else if (std.mem.eql(u8, a, "--selftest-penalty")) {
            i += 1;
            if (i >= process_args.len) {
                log.err("--selftest-penalty attend une valeur (fixture .safetensors produite par scripts/76_penalty_vectors.py)", .{});
                return error.MissingArgument;
            }
            args.selftest_penalty = process_args[i];
        } else if (std.mem.eql(u8, a, "--selftest-alloc-count")) {
            args.selftest_alloc_count = true;
        } else if (std.mem.eql(u8, a, "--selftest-kvdump-eq")) {
            i += 1;
            if (i >= process_args.len) {
                log.err("--selftest-kvdump-eq attend une valeur (chemin du dump de travail à écrire)", .{});
                return error.MissingArgument;
            }
            args.selftest_kvdump_eq = process_args[i];
        } else if (std.mem.eql(u8, a, "--selftest-kvdump-io")) {
            i += 1;
            if (i >= process_args.len) {
                log.err("--selftest-kvdump-io attend une valeur (répertoire de travail EXISTANT)", .{});
                return error.MissingArgument;
            }
            args.selftest_kvdump_io = process_args[i];
        } else if (std.mem.eql(u8, a, "--no-pin")) {
            args.no_pin = true;
        } else if (std.mem.eql(u8, a, "--draws")) {
            i += 1;
            if (i >= process_args.len) return error.MissingArgument;
            args.draws = std.fmt.parseInt(usize, process_args[i], 10) catch return error.InvalidArgument;
        } else if (std.mem.eql(u8, a, "--temperature")) {
            i += 1;
            if (i >= process_args.len) return error.MissingArgument;
            const v = std.fmt.parseFloat(f32, process_args[i]) catch return error.InvalidTemperature;
            // ⚠ Garde en ACCEPTATION. `t <= 0 → rejet` laisserait passer NaN (toute comparaison
            // avec NaN est fausse) : `--temperature nan` empoisonnerait tous les logits et
            // l'argmax rendrait un token arbitraire SANS ERREUR.
            // T_MIN : divergence DÉLIBÉRÉE avec HF, qui accepte 1e-45 et produit des NaN
            // (inf - inf dans le softmax). Logits bornés par le softcap 30 ⇒ 1e-30 suffit.
            if (!(v >= T_MIN and std.math.isFinite(v))) {
                log.err("--temperature {s} refusée : attendu un réel fini >= {e}. Pour du greedy déterministe, utiliser --top-k 1.", .{ process_args[i], T_MIN });
                return error.InvalidTemperature;
            }
            args.temperature = v;
        } else if (std.mem.eql(u8, a, "--top-k")) {
            i += 1;
            if (i >= process_args.len) return error.MissingArgument;
            const v = std.fmt.parseInt(u32, process_args[i], 10) catch return error.InvalidTopK;
            if (!(v <= VOCAB_CONTRACT)) {
                log.err("--top-k {d} > vocab {d} : refus explicite, jamais un clamp silencieux", .{ v, VOCAB_CONTRACT });
                return error.TopKOutOfRange;
            }
            args.top_k = v;
        } else if (std.mem.eql(u8, a, "--top-p")) {
            i += 1;
            if (i >= process_args.len) return error.MissingArgument;
            const v = std.fmt.parseFloat(f32, process_args[i]) catch return error.InvalidTopP;
            if (!(v > 0 and v <= 1)) {
                log.err("--top-p {s} refusée : attendu 0 < p <= 1 (NaN et inf exclus par la forme). Pour ne garder qu'un token, utiliser --top-k 1.", .{process_args[i]});
                return error.InvalidTopP;
            }
            args.top_p = v;
        } else if (std.mem.eql(u8, a, "--min-tokens-to-keep")) {
            i += 1;
            if (i >= process_args.len) return error.MissingArgument;
            const v = std.fmt.parseInt(u32, process_args[i], 10) catch return error.InvalidMinTokens;
            if (!(v >= 1 and v <= VOCAB_CONTRACT)) {
                log.err("--min-tokens-to-keep {d} refusé : attendu 1..{d} (0 ferait paniquer argmax sur une slice vide)", .{ v, VOCAB_CONTRACT });
                return error.InvalidMinTokens;
            }
            args.min_tokens_to_keep = v;
        } else if (std.mem.eql(u8, a, "--seed")) {
            i += 1;
            if (i >= process_args.len) return error.MissingArgument;
            args.seed = std.fmt.parseInt(u64, process_args[i], 10) catch return error.InvalidSeed;
        } else if (std.mem.eql(u8, a, "--repetition-penalty")) {
            i += 1;
            if (i >= process_args.len) return error.MissingArgument;
            args.repetition_penalty = parsePenalty(process_args[i]) orelse {
                log.err("--repetition-penalty {s} refusée : attendu un réel fini > 0 (1.0 = neutre ; NaN et inf exclus par la forme)", .{process_args[i]});
                return error.InvalidRepetitionPenalty;
            };
        } else if (std.mem.eql(u8, a, "--ignore-prompt")) {
            args.ignore_prompt = true;
        } else if (std.mem.eql(u8, a, "--gate-d1d2")) {
            args.gate_d1d2 = true;
        } else if (std.mem.eql(u8, a, "--gen-config")) {
            i += 1;
            if (i >= process_args.len) {
                log.err("--gen-config attend une valeur : un CHEMIN DE FICHIER (…/generation_config.json), pas un répertoire", .{});
                return error.MissingArgument;
            }
            args.gen_config = process_args[i];
        } else if (std.mem.eql(u8, a, "--dump-cache")) {
            i += 1;
            if (i >= process_args.len) {
                log.err("--dump-cache attend une valeur (chemin du fichier .kvdump à écrire)", .{});
                return error.MissingArgument;
            }
            args.dump_cache = process_args[i];
        } else if (std.mem.eql(u8, a, "--load-cache")) {
            i += 1;
            if (i >= process_args.len) {
                log.err("--load-cache attend une valeur (chemin d'un fichier .kvdump)", .{});
                return error.MissingArgument;
            }
            args.load_cache = process_args[i];
        } else if (std.mem.eql(u8, a, "--no-gen-config")) {
            args.no_gen_config = true;
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
// GC1 — selftest de la politique de décodage `generation_config.json`, SANS GPU ni tokenizer
// (spec docs/superpowers/specs/2026-07-28-generation-config-design.md §4.5bis).
//
// TROIS familles de choses sont exercées, donc TROIS véhicules :
//   1. SÉLECTION   — fixture safetensors (`top5_idx` {N,5} i32, `top5_val` {N,5} f32,
//      `expect_tok` {N} i32, `expect_rank` {N} i32), produite par `scripts/71_gc1_fixture.py`
//      depuis le VRAI `SuppressTokensLogitsProcessor` de transformers : `expect_tok` est
//      l'argmax du vecteur COMPLET post-suppression (262 144 logits). Le cas confronte donc
//      notre sélection sur top-5 pré-trié à la sémantique HF — c'est précisément ce qui rend
//      la claim C2 testable au lieu d'être postulée (§4.2).
//   2. VALIDATIONS — sidecar `<fixture>.manifest.json`, liste `validation_cases` :
//      {nom, `eot_id`, `content` = le JSON LITTÉRAL du generation_config à parser,
//      `expect_error` = @errorName attendu (ou "" si le cas doit être ACCEPTÉ)}.
//      ⚠ `content` est une CHAÎNE, pas un objet : le selftest doit exercer le parser sur du
//      texte réel (y compris malformé), pas sur un objet déjà re-sérialisé par nos soins.
//      ⚠ L'`eot_id` est porté par la donnée : le tokenizer n'est PAS chargé sur ce chemin
//      (early-return avant `:899`), donc le contrôle croisé `EotNotInEosList` ne peut pas le
//      mesurer (plan Task 2 point 2.4).
//   3. DÉCOUVERTE  — liste `discovery_cases` du même sidecar : {nom, `ckpt`, et soit
//      `expect_path`, soit `expect_error`}. Les topologies (symlink ABSOLU et RELATIF, cas
//      introuvable) sont fabriquées par l'étape shell du plan — aucun selftest du repo ne crée
//      de symlink, on ne commence pas ici.
//
// PASS = 100 % des cas de sélection ET des cas de validation ET des cas de découverte, ET les
// SIX compteurs de non-vacuité tous non nuls. Un chemin qu'un selftest n'exerce pas, il ne l'a
// pas validé : chaque compteur à zéro est un FAIL, pas un warning (leçon « test à l'antécédent
// vide », feedback_test_vacuite_antecedent).
// ============================================================================================
fn selftestGencfg(allocator: std.mem.Allocator, io: std.Io, fixture_path: []const u8) !void {
    // ---- Véhicule 2/3 : le sidecar (lu d'abord — il porte aussi la politique de sélection) ----
    const manifest_path = try std.fmt.allocPrint(allocator, "{s}.manifest.json", .{fixture_path});
    defer allocator.free(manifest_path);
    var mf = std.Io.Dir.cwd().openFile(io, manifest_path, .{ .mode = .read_only }) catch |err| {
        log.err("--selftest-gencfg : sidecar illisible ({s}) : {s} — requis (porte les cas de validation et de découverte)", .{ manifest_path, @errorName(err) });
        return error.MissingManifest;
    };
    defer mf.close(io);
    const mlen: usize = @intCast(try mf.length(io));
    const mtext = try allocator.alloc(u8, mlen);
    defer allocator.free(mtext);
    if (try mf.readPositionalAll(io, mtext, 0) != mlen) return error.ShortRead;

    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, mtext, .{ .allocate = .alloc_always });
    defer parsed.deinit();
    const root = parsed.value.object;

    const sel_obj = (root.get("selection") orelse {
        log.err("--selftest-gencfg : clé 'selection' absente de {s}", .{manifest_path});
        return error.MissingManifest;
    }).object;
    const vocab_size: u32 = @intCast((sel_obj.get("vocab_size") orelse return error.MissingManifest).integer);
    const sel_eot_id: u32 = @intCast((sel_obj.get("eot_id") orelse return error.MissingManifest).integer);
    const sup_arr = (sel_obj.get("suppress_tokens") orelse return error.MissingManifest).array;
    const eos_arr = (sel_obj.get("eos_token_id") orelse return error.MissingManifest).array;

    const sup_raw = try allocator.alloc(u32, sup_arr.items.len);
    defer allocator.free(sup_raw);
    for (sup_arr.items, 0..) |v, i| sup_raw[i] = @intCast(v.integer);
    const eos_raw = try allocator.alloc(u32, eos_arr.items.len);
    defer allocator.free(eos_raw);
    for (eos_arr.items, 0..) |v, i| eos_raw[i] = @intCast(v.integer);

    // La politique de sélection passe par le MÊME constructeur validant que le runner : un
    // selftest qui fabriquerait sa GenCfg à la main testerait une copie, pas le code livré.
    var policy = try gencfg.fromLists(allocator, manifest_path, sup_raw, eos_raw, &.{}, .{
        .vocab_size = vocab_size,
        .eot_id = sel_eot_id,
    });
    defer policy.deinit(allocator);

    // ---- Véhicule 1 : les cas de SÉLECTION ----
    var reg: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, fixture_path);
    defer reg.deinit();
    var file = try std.Io.Dir.cwd().openFile(io, fixture_path, .{ .mode = .read_only });
    defer file.close(io);

    const top5_idx = try readFixtureAlloc(i32, .i32, allocator, io, &reg, &file, "top5_idx");
    defer allocator.free(top5_idx);
    const top5_val = try readFixtureAlloc(f32, .f32, allocator, io, &reg, &file, "top5_val");
    defer allocator.free(top5_val);
    const expect_tok = try readFixtureAlloc(i32, .i32, allocator, io, &reg, &file, "expect_tok");
    defer allocator.free(expect_tok);
    const expect_rank = try readFixtureAlloc(i32, .i32, allocator, io, &reg, &file, "expect_rank");
    defer allocator.free(expect_rank);

    const k: usize = gencfg.TOP_K;
    const n_cases: usize = expect_tok.len;
    if (n_cases == 0) {
        log.err("--selftest-gencfg : 0 cas de sélection — un PASS vacueux ({s})", .{fixture_path});
        return error.EmptyFixture;
    }
    if (top5_idx.len != n_cases * k or top5_val.len != n_cases * k or expect_rank.len != n_cases) {
        log.err("--selftest-gencfg : formes incohérentes — top5_idx={d} top5_val={d} expect_tok={d} expect_rank={d} (N={d}, TOP_K={d})", .{ top5_idx.len, top5_val.len, n_cases, expect_rank.len, n_cases, k });
        return error.ShapeMismatch;
    }

    var n_sel_ok: usize = 0;
    var n_bit_top1: usize = 0; // compteur 1/6 : cas où la suppression a réellement mordu
    var n_ties: usize = 0; // C2 : compter les égalités exactes, la réserve du §4.2
    for (0..n_cases) |c| {
        var idx: [gencfg.TOP_K]usize = undefined;
        for (0..k) |j| idx[j] = @intCast(top5_idx[c * k + j]);

        const sel = policy.select(&idx) catch |err| {
            log.err("cas de sélection {d} : {s} (top5_idx={any})", .{ c, @errorName(err), idx });
            return err;
        };
        const want_tok: usize = @intCast(expect_tok[c]);
        const want_rank: usize = @intCast(expect_rank[c]);
        if (sel.tok == want_tok and sel.rank == want_rank) {
            n_sel_ok += 1;
        } else {
            log.err("cas de sélection {d} FAIL — got tok={d} rank={d}, want tok={d} rank={d} (top5_idx={any})", .{ c, sel.tok, sel.rank, want_tok, want_rank, idx });
        }
        if (want_rank != 0) n_bit_top1 += 1;
        for (1..k) |j| {
            if (top5_val[c * k + j] == top5_val[c * k + j - 1]) n_ties += 1;
        }
    }

    // ---- Véhicule 2 : les cas de VALIDATION (§4.1) ----
    var n_val_ok: usize = 0;
    var n_val_total: usize = 0;
    var n_eot_not_in_eos: usize = 0; // compteur 2/6
    var n_out_of_range: usize = 0; // compteur 3/6
    var n_begin_suppress: usize = 0; // compteur 4/6
    var n_eos_empty: usize = 0; // compteur 5/6
    var n_dedup: usize = 0; // compteur 6/6

    if (root.get("validation_cases")) |vc| {
        for (vc.array.items) |case_v| {
            const case = case_v.object;
            const name = (case.get("name") orelse return error.MissingManifest).string;
            const content = (case.get("content") orelse return error.MissingManifest).string;
            const eot_id: u32 = @intCast((case.get("eot_id") orelse return error.MissingManifest).integer);
            const want_err = if (case.get("expect_error")) |e| e.string else "";
            n_val_total += 1;

            var cfg = gencfg.parseFromSlice(allocator, content, name, .{
                .vocab_size = vocab_size,
                .eot_id = eot_id,
            }) catch |err| {
                const got = @errorName(err);
                if (std.mem.eql(u8, got, want_err)) {
                    n_val_ok += 1;
                    if (std.mem.eql(u8, got, "EotNotInEosList")) n_eot_not_in_eos += 1;
                    if (std.mem.eql(u8, got, "SuppressIdOutOfRange")) n_out_of_range += 1;
                    if (std.mem.eql(u8, got, "BeginSuppressUnsupported")) n_begin_suppress += 1;
                    if (std.mem.eql(u8, got, "EosListEmpty")) n_eos_empty += 1;
                } else {
                    log.err("cas de validation '{s}' FAIL — erreur {s}, attendu {s}", .{ name, got, if (want_err.len == 0) "ACCEPTÉ" else want_err });
                }
                continue;
            };
            defer cfg.deinit(allocator);

            if (want_err.len != 0) {
                log.err("cas de validation '{s}' FAIL — ACCEPTÉ, alors que {s} était attendu", .{ name, want_err });
                continue;
            }
            // Cas accepté : si le manifest annonce une déduplication, elle doit avoir eu lieu.
            if (case.get("expect_suppress_len")) |want_len| {
                const want: usize = @intCast(want_len.integer);
                if (cfg.suppress.len != want) {
                    log.err("cas de validation '{s}' FAIL — suppress.len={d}, attendu {d} (déduplication)", .{ name, cfg.suppress.len, want });
                    continue;
                }
                if (case.get("is_dedup_case")) |b| {
                    if (b.bool) n_dedup += 1;
                }
            }
            n_val_ok += 1;
        }
    }

    // ---- Véhicule 3 : la DÉCOUVERTE 1-hop (§4.1) ----
    var n_disc_ok: usize = 0;
    var n_disc_total: usize = 0;
    if (root.get("discovery_cases")) |dc| {
        for (dc.array.items) |case_v| {
            const case = case_v.object;
            const name = (case.get("name") orelse return error.MissingManifest).string;
            const ckpt = (case.get("ckpt") orelse return error.MissingManifest).string;
            const want_err = if (case.get("expect_error")) |e| e.string else "";
            n_disc_total += 1;

            const got_path = gencfg.discoverAlloc(allocator, io, null, ckpt) catch |err| {
                const got = @errorName(err);
                if (std.mem.eql(u8, got, want_err)) {
                    n_disc_ok += 1;
                } else {
                    log.err("cas de découverte '{s}' FAIL — erreur {s}, attendu {s}", .{ name, got, if (want_err.len == 0) "un chemin" else want_err });
                }
                continue;
            };
            defer allocator.free(got_path);

            if (want_err.len != 0) {
                log.err("cas de découverte '{s}' FAIL — a résolu {s}, alors que {s} était attendu", .{ name, got_path, want_err });
                continue;
            }
            const want_path = (case.get("expect_path") orelse return error.MissingManifest).string;
            if (std.mem.eql(u8, got_path, want_path)) {
                n_disc_ok += 1;
            } else {
                log.err("cas de découverte '{s}' FAIL — got={s} want={s}", .{ name, got_path, want_path });
            }
        }
    }

    // ---- Verdict : exactitude ET non-vacuité ----
    const exact_ok = (n_sel_ok == n_cases) and (n_val_ok == n_val_total) and (n_disc_ok == n_disc_total);
    const vac = [_]struct { name: []const u8, n: usize }{
        .{ .name = "id supprimé top-1", .n = n_bit_top1 },
        .{ .name = "EotNotInEosList", .n = n_eot_not_in_eos },
        .{ .name = "SuppressIdOutOfRange", .n = n_out_of_range },
        .{ .name = "BeginSuppressUnsupported", .n = n_begin_suppress },
        .{ .name = "EosListEmpty", .n = n_eos_empty },
        .{ .name = "doublons dédupliqués", .n = n_dedup },
    };
    var vac_ok = true;
    for (vac) |v| {
        if (v.n == 0) {
            log.err("NON-VACUITÉ FAIL — le cas « {s} » n'a été exercé 0 fois : ce chemin n'est PAS validé", .{v.name});
            vac_ok = false;
        }
    }
    log.info("GC1 non-vacuité : top1_supprimé={d} eot_not_in_eos={d} out_of_range={d} begin_suppress={d} eos_empty={d} dedup={d} | égalités exactes rencontrées={d} (C2)", .{ n_bit_top1, n_eot_not_in_eos, n_out_of_range, n_begin_suppress, n_eos_empty, n_dedup, n_ties });

    if (exact_ok and vac_ok) {
        log.info("SELFTEST GENCFG PASS — sélection {d}/{d}, validations {d}/{d}, découverte {d}/{d}, 6/6 compteurs non nuls", .{ n_sel_ok, n_cases, n_val_ok, n_val_total, n_disc_ok, n_disc_total });
    } else {
        log.err("SELFTEST GENCFG FAIL — sélection {d}/{d}, validations {d}/{d}, découverte {d}/{d}, non-vacuité={}", .{ n_sel_ok, n_cases, n_val_ok, n_val_total, n_disc_ok, n_disc_total, vac_ok });
        return error.SelftestGencfgFailed;
    }
}

// ============================================================================================
// S2-U — selftest des warpers de sampling, HOST-ONLY (spec phase 2 rév. 3, plan Task 3).
//
// La fixture est produite par `scripts/72_sampling_fixture.py` depuis les VRAIS warpers de
// transformers : on ne compare jamais deux transcriptions du même auteur.
//
// FORMAT 1-D CONCATÉNÉ (`logits_in`, `mask_expected`, `offsets`) et non `{N,V}` : les cas ont des
// vocabulaires de tailles DIFFÉRENTES (8, 5, 262 144), et un padding serait fatal ET silencieux.
//
// ⚠ `compare_mode` PAR CAS, écrit par le producteur — jamais deviné ici :
//   `indices`     : le masque doit être identique À L'IDENTIQUE. Mode le plus fort, licite quand
//                   les logits du cas sont distincts ou que V <= 128 (`torch.sort` y est stable).
//   `equivalence` : multiset trié des logits survivants + masse de probabilité. Réservé aux cas
//                   à ex æquo au-delà de 128, où l'identité des survivants n'est PAS contractuelle.
// Comparer TOUJOURS par équivalence serait AVEUGLE au cas le plus discriminant : sur 8 logits
// égaux avec top_p=0,25, HF garde {6,7} et le naïf {0,1} — disjoints, même multiset, même masse.
// ============================================================================================
fn selftestSampling(allocator: std.mem.Allocator, io: std.Io, fixture_path: []const u8) !void {
    // ---- sidecar ----
    const manifest_path = try std.fmt.allocPrint(allocator, "{s}.manifest.json", .{fixture_path});
    defer allocator.free(manifest_path);
    var mf = std.Io.Dir.cwd().openFile(io, manifest_path, .{ .mode = .read_only }) catch |err| {
        log.err("--selftest-sampling : sidecar illisible ({s}) : {s} — requis (porte params et compare_mode)", .{ manifest_path, @errorName(err) });
        return error.MissingManifest;
    };
    defer mf.close(io);
    const mlen: usize = @intCast(try mf.length(io));
    const mtext = try allocator.alloc(u8, mlen);
    defer allocator.free(mtext);
    if (try mf.readPositionalAll(io, mtext, 0) != mlen) return error.ShortRead;
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, mtext, .{ .allocate = .alloc_always });
    defer parsed.deinit();
    const root = parsed.value.object;
    const cases = (root.get("cases") orelse return error.MissingManifest).array;

    // ---- fixture ----
    var reg: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, fixture_path);
    defer reg.deinit();
    var file = try std.Io.Dir.cwd().openFile(io, fixture_path, .{ .mode = .read_only });
    defer file.close(io);
    const logits_in = try readFixtureAlloc(f32, .f32, allocator, io, &reg, &file, "logits_in");
    defer allocator.free(logits_in);
    const mask_exp = try readFixtureAlloc(u8, .u8, allocator, io, &reg, &file, "mask_expected");
    defer allocator.free(mask_exp);
    const offsets = try readFixtureAlloc(i64, .i64, allocator, io, &reg, &file, "offsets");
    defer allocator.free(offsets);

    if (cases.items.len == 0 or offsets.len != cases.items.len + 1) {
        log.err("--selftest-sampling : {d} cas mais {d} offsets (attendu {d})", .{ cases.items.len, offsets.len, cases.items.len + 1 });
        return error.ShapeMismatch;
    }

    // scratch dimensionné au plus grand cas, alloué UNE FOIS (interdit d'allocation par step)
    var vmax: usize = 0;
    for (0..cases.items.len) |c| {
        const n: usize = @intCast(offsets[c + 1] - offsets[c]);
        if (n > vmax) vmax = n;
    }
    var scratch = try sampling.Scratch.init(allocator, vmax);
    defer scratch.deinit(allocator);
    const work = try allocator.alloc(f32, vmax);
    defer allocator.free(work);

    var n_ok: usize = 0;
    var n_indices: usize = 0;
    var n_equiv: usize = 0;
    var n_topk_deborde: usize = 0; // antécédent : top_k laisse PLUS de k survivants (F9)
    var n_vocab_reel: usize = 0; // antécédent : au moins un cas à l'échelle de production (F14)

    for (cases.items, 0..) |case_v, c| {
        const case = case_v.object;
        const name = (case.get("name") orelse return error.MissingManifest).string;
        const mode = (case.get("compare_mode") orelse return error.MissingManifest).string;
        const warper = (case.get("warper") orelse return error.MissingManifest).string;
        const params = (case.get("params") orelse return error.MissingManifest).object;

        const lo: usize = @intCast(offsets[c]);
        const hi: usize = @intCast(offsets[c + 1]);
        const n = hi - lo;
        const buf = work[0..n];
        @memcpy(buf, logits_in[lo..hi]);

        const min_keep: u32 = if (params.get("min_tokens_to_keep")) |v| @intCast(v.integer) else 1;

        // Ordre de HF (F8) : Temperature(17) → TopK(19) → TopP(20)
        if (params.get("temperature")) |v| {
            const t: f32 = @floatCast(v.float);
            if (t != 1.0) sampling.applyTemperature(buf, t);
        }
        if (params.get("top_k")) |v| {
            sampling.applyTopK(buf, @intCast(v.integer), min_keep, &scratch);
        }
        if (params.get("top_p")) |v| {
            const p: f32 = @floatCast(v.float);
            sampling.applyTopP(buf, p, min_keep, &scratch);
        }

        var n_surv: usize = 0;
        for (buf) |x| {
            if (x != sampling.FILTER) n_surv += 1;
        }
        if (params.get("top_k")) |v| {
            if (n_surv > @as(usize, @intCast(v.integer))) n_topk_deborde += 1;
        }
        if (n == VOCAB_CONTRACT) n_vocab_reel += 1;

        var ok = true;
        if (std.mem.eql(u8, mode, "indices")) {
            n_indices += 1;
            for (buf, 0..) |x, i| {
                const got: u8 = if (x != sampling.FILTER) 1 else 0;
                if (got != mask_exp[lo + i]) {
                    if (ok) log.err("cas '{s}' ({s}) FAIL — indice {d} : nous={d} HF={d}", .{ name, warper, i, got, mask_exp[lo + i] });
                    ok = false;
                }
            }
        } else {
            n_equiv += 1;
            // classe d'équivalence : même NOMBRE de survivants et même MASSE (les logits
            // survivants sont ex æquo par construction dans ce mode).
            var n_exp: usize = 0;
            var sum_got: f64 = 0;
            var sum_exp: f64 = 0;
            for (buf, 0..) |x, i| {
                if (x != sampling.FILTER) sum_got += logits_in[lo + i];
                if (mask_exp[lo + i] == 1) {
                    n_exp += 1;
                    sum_exp += logits_in[lo + i];
                }
            }
            const tol = 1e-3 * @max(@abs(sum_exp), 1.0);
            if (n_surv != n_exp or @abs(sum_got - sum_exp) > tol) {
                log.err("cas '{s}' ({s}) FAIL — survivants nous={d} HF={d}, somme nous={d:.4} HF={d:.4}", .{ name, warper, n_surv, n_exp, sum_got, sum_exp });
                ok = false;
            }
        }
        if (ok) n_ok += 1;
    }

    // ---- non-vacuité : un antécédent vide rend le cas INEXÉCUTABLE, pas PASS (patron GC1) ----
    var vac_ok = true;
    if (n_indices == 0) {
        log.err("NON-VACUITÉ FAIL — aucun cas comparé par INDICES : on retombe sur la règle aveugle", .{});
        vac_ok = false;
    }
    if (n_topk_deborde == 0) {
        log.err("NON-VACUITÉ FAIL — aucun cas où top_k laisse plus de k survivants : F9 n'est pas exercé", .{});
        vac_ok = false;
    }
    if (n_vocab_reel == 0) {
        log.err("NON-VACUITÉ FAIL — aucun cas à V={d} : le gate passe à vide (F14)", .{VOCAB_CONTRACT});
        vac_ok = false;
    }

    if (n_ok == cases.items.len and vac_ok) {
        log.info("SELFTEST SAMPLING PASS — cas {d}/{d} (indices {d}, équivalence {d}), antécédents topk_déborde={d} vocab_réel={d}", .{ n_ok, cases.items.len, n_indices, n_equiv, n_topk_deborde, n_vocab_reel });
    } else {
        log.err("SELFTEST SAMPLING FAIL — cas {d}/{d} (indices {d}, équivalence {d}), non-vacuité={}", .{ n_ok, cases.items.len, n_indices, n_equiv, vac_ok });
        return error.SelftestSamplingFailed;
    }
}

// ============================================================================================
// RP1 (repetition penalty) — `applyRepetitionPenalty` comparée **0 ULP** au VRAI processor HF,
// host-only (même patron que S2-U : aucun GPU, aucun poids, itération en secondes).
//
// Fixture produite par `scripts/76_penalty_vectors.py`, qui APPELLE
// `RepetitionPenaltyLogitsProcessor` — retranscrire sa formule des deux côtés ferait passer une
// faute commune (spec C5). La fixture n'a pas de sidecar `.manifest.json` : les 4 penalties sont
// un CONTRAT de la spec (§RP1), portées ici par `RP_CASES` ; un tenseur manquant échoue
// bruyamment à la lecture.
//
// ⚠ Les assertions de non-vacuité ne sont pas décoratives. Sans « ≥ 1 doublon dans hist », la
// déduplication n'est jamais exercée ; sans « ≥ 1 logit négatif ET ≥ 1 positif », une seule des
// deux branches de signe l'est — et une implémentation qui les échangerait passerait le gate.
// ============================================================================================
const RP_CASES = [_]struct { name: []const u8, p: f32 }{
    .{ .name = "logits_out_0.8", .p = 0.8 },
    .{ .name = "logits_out_1.0", .p = 1.0 },
    .{ .name = "logits_out_1.15", .p = 1.15 },
    .{ .name = "logits_out_1.5", .p = 1.5 },
};

fn selftestPenalty(allocator: std.mem.Allocator, io: std.Io, fixture_path: []const u8) !void {
    var reg: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, fixture_path);
    defer reg.deinit();
    var file = try std.Io.Dir.cwd().openFile(io, fixture_path, .{ .mode = .read_only });
    defer file.close(io);

    const logits_in = try readFixtureAlloc(f32, .f32, allocator, io, &reg, &file, "logits_in");
    defer allocator.free(logits_in);
    const hist_i32 = try readFixtureAlloc(i32, .i32, allocator, io, &reg, &file, "hist");
    defer allocator.free(hist_i32);
    const ties = try readFixtureAlloc(f32, .f32, allocator, io, &reg, &file, "logits_ties");
    defer allocator.free(ties);

    // `hist` est produit en i32 (convention safetensors du repo) ; la fonction prend des u32 —
    // ids de tokens, jamais négatifs. Un négatif dans la fixture est une CORRUPTION, pas un cas.
    const hist = try allocator.alloc(u32, hist_i32.len);
    defer allocator.free(hist);
    for (hist_i32, 0..) |v, i| {
        if (v < 0 or @as(usize, @intCast(v)) >= logits_in.len) {
            log.err("--selftest-penalty : hist[{d}] = {d} hors du vocab de la fixture (0..{d})", .{ i, v, logits_in.len });
            return error.CorruptFixture;
        }
        hist[i] = @intCast(v);
    }

    // Buffers de travail — alloués UNE FOIS (le selftest est host-only, mais on exerce la même
    // discipline que la boucle de génération : c'est ce code-là qui y sera appelé).
    const work = try allocator.alloc(f32, logits_in.len);
    defer allocator.free(work);
    const seen = try allocator.alloc(u64, (logits_in.len + 63) / 64);
    defer allocator.free(seen);

    // ---- antécédents de la fixture : sans eux le gate passerait à vide ----
    var n_distinct: usize = 0;
    var n_neg: usize = 0;
    var n_pos: usize = 0;
    {
        @memset(seen, 0);
        for (hist) |t| {
            const w = t >> 6;
            const mask = @as(u64, 1) << @truncate(t);
            if ((seen[w] & mask) != 0) continue;
            seen[w] |= mask;
            n_distinct += 1;
            if (logits_in[t] < 0) n_neg += 1 else n_pos += 1;
        }
    }
    var vac_ok = true;
    if (n_distinct >= hist.len) {
        log.err("NON-VACUITÉ FAIL — aucun doublon dans hist ({d} ids, {d} distincts) : la déduplication n'est pas exercée", .{ hist.len, n_distinct });
        vac_ok = false;
    }
    if (n_neg == 0) {
        log.err("NON-VACUITÉ FAIL — aucun logit NÉGATIF parmi les ids de hist : la branche ×penalty n'est pas exercée", .{});
        vac_ok = false;
    }
    if (n_pos == 0) {
        log.err("NON-VACUITÉ FAIL — aucun logit POSITIF ou nul parmi les ids de hist : la branche ÷penalty n'est pas exercée", .{});
        vac_ok = false;
    }

    // ---- comparaison 0 ULP contre HF, penalty par penalty ----
    var n_ok: usize = 0;
    for (RP_CASES) |c| {
        const expected = try readFixtureAlloc(f32, .f32, allocator, io, &reg, &file, c.name);
        defer allocator.free(expected);
        if (expected.len != logits_in.len) {
            log.err("{s} : {d} valeurs ≠ logits_in {d}", .{ c.name, expected.len, logits_in.len });
            return error.ShapeMismatch;
        }

        @memcpy(work, logits_in);
        @memset(seen, 0);
        const touched = sampling.applyRepetitionPenalty(work, hist, c.p, seen);

        var n_diff_hf: usize = 0; // logits que HF a changés
        var n_bad: usize = 0; // logits où NOUS différons de HF (bit à bit)
        var first_bad: i64 = -1;
        for (work, 0..) |got, i| {
            if (@as(u32, @bitCast(expected[i])) != @as(u32, @bitCast(logits_in[i]))) n_diff_hf += 1;
            if (@as(u32, @bitCast(got)) != @as(u32, @bitCast(expected[i]))) {
                if (first_bad < 0) first_bad = @intCast(i);
                n_bad += 1;
            }
        }

        var ok = true;
        if (n_bad != 0) {
            log.err("RP1 FAIL — penalty {d}: {d} valeurs hors 0 ULP, 1re à l'id {d} (nous={d:.9} HF={d:.9})", .{ c.p, n_bad, first_bad, work[@intCast(first_bad)], expected[@intCast(first_bad)] });
            ok = false;
        }
        // Le retour `touched` est la sonde de non-vacuité du câblage (Task 4) : il doit dire la
        // VÉRITÉ ici aussi, sinon le compteur `n_penalty_touched` mentirait en production.
        const expect_touched = (c.p != 1.0);
        if (touched != expect_touched) {
            log.err("RP1 FAIL — penalty {d}: touched={} attendu {} (sonde de non-vacuité fausse)", .{ c.p, touched, expect_touched });
            ok = false;
        }
        if (c.p == 1.0) {
            if (n_diff_hf != 0) {
                log.err("RP1 FAIL — penalty 1.0 : HF a changé {d} logits, la neutralité de la fixture est fausse", .{n_diff_hf});
                ok = false;
            }
        } else if (n_diff_hf != n_distinct) {
            // C'est LE fait qui fonde la spec : HF pénalise chaque token distinct AU PLUS UNE FOIS.
            log.err("RP1 FAIL — penalty {d}: HF a changé {d} logits, attendu {d} (= ids distincts). Si c'est hist.len={d}, HF ne déduplique pas et la spec est à revoir.", .{ c.p, n_diff_hf, n_distinct, hist.len });
            ok = false;
        }
        if (ok) n_ok += 1;
    }

    // ---- tie-break de l'argmax : PREMIER indice gagnant (critère RP1) ----
    // La référence n'est pas une constante magique : on dérive « premier indice atteignant le
    // maximum » indépendamment, et on exige que le max soit atteint ≥ 2 fois — sans quoi le
    // critère de tie-break serait vérifié sur un vecteur qui n'a pas d'ex æquo.
    var mx: f32 = ties[0];
    for (ties) |v| {
        if (v > mx) mx = v;
    }
    var n_ties: usize = 0;
    var first_max: usize = 0;
    for (ties, 0..) |v, i| {
        if (v == mx) {
            if (n_ties == 0) first_max = i;
            n_ties += 1;
        }
    }
    if (n_ties < 2) {
        log.err("NON-VACUITÉ FAIL — logits_ties n'a que {d} occurrence du maximum : le tie-break n'est pas exercé", .{n_ties});
        vac_ok = false;
    }
    const got_argmax = sampling.argmax(ties);
    var tie_ok = true;
    if (got_argmax != first_max) {
        log.err("RP1 FAIL — tie-break : argmax={d}, attendu {d} (PREMIER des {d} ex æquo)", .{ got_argmax, first_max, n_ties });
        tie_ok = false;
    }

    if (n_ok == RP_CASES.len and vac_ok and tie_ok) {
        log.info("RP1 PASS — {d}/{d} penalties bit-identiques au processor HF ({d} valeurs chacune), hist {d} ids dont {d} distincts ({d} logits <0, {d} >=0), tie-break={d} sur {d} ex æquo", .{ n_ok, RP_CASES.len, logits_in.len, hist.len, n_distinct, n_neg, n_pos, got_argmax, n_ties });
    } else {
        log.err("RP1 FAIL — {d}/{d} penalties, non-vacuité={}, tie-break={}", .{ n_ok, RP_CASES.len, vac_ok, tie_ok });
        return error.SelftestPenaltyFailed;
    }
}

// ============================================================================================
// S-AC (D10) — exerce les 4 fonctions vtable du CountingAllocator INSTALLÉ (celui de run()) et
// vérifie les deltas. On passe par le wrapper déjà en place : c'est l'instrument de production
// qu'on teste, pas une copie.
// ============================================================================================
fn selftestAllocCount(allocator: std.mem.Allocator, counter: *alloc_count.CountingAllocator) !void {
    const a0 = counter.n_alloc;
    const r0 = counter.n_resize;
    const m0 = counter.n_remap;
    const f0 = counter.n_free;
    const by0 = counter.bytes_alloc; // bytes_min en DELTA, comme les autres compteurs
    const b1 = try allocator.alloc(u8, 64);
    const b2 = try allocator.alloc(u8, 128);
    var b3 = try allocator.alloc(u8, 256);
    if (allocator.resize(b3, 128)) b3 = b3[0..128]; // compte l'APPEL ; longueur mise à jour si accepté (contrat Allocator)
    if (allocator.remap(b3, 512)) |nb| b3 = nb; // idem
    allocator.free(b1);
    allocator.free(b2);
    // b3 volontairement non libéré ici — libéré après le verdict pour ne pas fausser f_delta.
    const checks = [_]struct { name: []const u8, got: u64, want: u64 }{
        .{ .name = "alloc", .got = counter.n_alloc - a0, .want = 3 },
        .{ .name = "resize", .got = counter.n_resize - r0, .want = 1 },
        .{ .name = "remap", .got = counter.n_remap - m0, .want = 1 },
        .{ .name = "free", .got = counter.n_free - f0, .want = 2 },
        .{ .name = "calls", .got = counter.calls() - (a0 + r0 + m0), .want = 5 },
        .{ .name = "bytes_min", .got = @intFromBool(counter.bytes_alloc - by0 >= 448), .want = 1 },
    };
    var pass: usize = 0;
    for (checks) |c| {
        if (c.got == c.want) {
            pass += 1;
        } else {
            log.err("S-AC: {s} = {d}, attendu {d}", .{ c.name, c.got, c.want });
        }
    }
    allocator.free(b3);
    if (pass != checks.len) {
        log.err("SELFTEST-AC: FAIL {d}/{d}", .{ pass, checks.len });
        return error.SelftestFailed;
    }
    log.info("SELFTEST-AC: PASS {d}/{d}", .{ pass, checks.len });
}

// ============================================================================================
// S2-D — tirage multinomial sur des logits FIGÉS en fixture (spec phase 2, plan Task 5).
// Écrit un histogramme `counts` que `scripts/72_sampling_fixture.py --chi2` confronte à une
// théorique **torch** : deux implémentations INDÉPENDANTES. Si les deux partageaient le code,
// un biais commun passerait le test sans être vu.
// ============================================================================================
fn selftestDraw(allocator: std.mem.Allocator, io: std.Io, fixture_path: []const u8, draws: usize, seed: ?u64) !void {
    const s = seed orelse {
        log.err("--selftest-draw exige --seed : un tirage non reproductible n'est pas auditable (spec §5)", .{});
        return error.SeedRequired;
    };
    var reg: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, fixture_path);
    defer reg.deinit();
    var file = try std.Io.Dir.cwd().openFile(io, fixture_path, .{ .mode = .read_only });
    defer file.close(io);
    const lg = try readFixtureAlloc(f32, .f32, allocator, io, &reg, &file, "draw_logits");
    defer allocator.free(lg);

    const counts = try allocator.alloc(i64, lg.len);
    defer allocator.free(counts);
    @memset(counts, 0);

    var prng = std.Random.DefaultPrng.init(s);
    const rnd = prng.random();
    var n_filtered_drawn: usize = 0;
    for (0..draws) |_| {
        const tok = sampling.sample(lg, rnd);
        if (lg[tok] == sampling.FILTER) n_filtered_drawn += 1;
        counts[tok] += 1;
    }
    if (n_filtered_drawn != 0) {
        log.err("S2-D : {d} tirages ont rendu un token FILTRÉ — invariant du sampler violé", .{n_filtered_drawn});
        return error.SampledFilteredToken;
    }

    // n'écrire que les ids réellement tirés (le vocab entier ferait 2 Mo pour 10 valeurs utiles)
    var nz: usize = 0;
    for (counts) |c| {
        if (c != 0) nz += 1;
    }
    const out = try std.fmt.allocPrint(allocator, "{s}.draws.safetensors", .{fixture_path});
    defer allocator.free(out);
    const packed_counts = try allocator.alloc(i64, nz);
    defer allocator.free(packed_counts);
    var k: usize = 0;
    for (counts) |c| {
        if (c != 0) {
            packed_counts[k] = c;
            k += 1;
        }
    }
    // Même patron d'écriture que `writeIdsSafetensors` (:1484) — on ne réinvente pas un writer.
    const header = try std.fmt.allocPrint(allocator, "{{\"counts\":{{\"dtype\":\"I64\",\"shape\":[{d}],\"data_offsets\":[0,{d}]}}}}", .{ nz, nz * 8 });
    defer allocator.free(header);
    var len_le: [8]u8 = undefined;
    std.mem.writeInt(u64, &len_le, header.len, .little);
    const f = try std.Io.Dir.createFile(.cwd(), io, out, .{});
    defer f.close(io);
    try f.writePositionalAll(io, &len_le, 0);
    try f.writePositionalAll(io, header, 8);
    try f.writePositionalAll(io, std.mem.sliceAsBytes(packed_counts), 8 + header.len);
    log.info("SELFTEST DRAW — {d} tirages, seed {d}, {d} ids distincts touchés, 0 token filtré -> {s}", .{ draws, s, nz, out });
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
// ⚠ La taille vient de `gencfg.TOP_K` — DÉCLARATION UNIQUE (spec §4.2) : ce type, le `topK`
// in-graph et la boucle de lecture en étaient trois copies du littéral `5`, et la garde
// `suppress.len + 1 > TOP_K` de la politique de décodage raisonne sur cette valeur. Trois copies,
// c'est trois occasions qu'elles divergent en silence.
const Top5 = struct { idx: [gencfg.TOP_K]usize, val: [gencfg.TOP_K]f32 };

// Raison d'arrêt (A3) — HISSÉE au niveau du type (elle vivait dans generateOnce) : le manifest
// kvdump la publie (`stop_reason`) et le selftest DC2 en fait une PRÉ-CONDITION (un appel qui
// s'arrête avant sa borne rend le gate INEXÉCUTABLE, pas FAIL).
const StopReason = enum { oracle, eot, max_tokens, l_max };

// kvdump/DC2 : ce que le selftest d'équivalence doit récupérer d'un appel à generateOnce.
// Les copies sont faites APRÈS la boucle (hors fenêtre ALLOC-LOOP gelée) — aucune interaction
// avec DC6.
const Capture = struct {
    ids: *std.ArrayList(i64),
    top5: *std.ArrayList(Top5),
    stop: *StopReason,
};

// Vocabulaire du contrat U0 (docs/U_12B_CONTRACT.md — `tokenizer.json` du snapshot : 262 144
// entrées, dont 24 `added_tokens`). Sert UNIQUEMENT à borner les ids de suppression au moment du
// fail-fast, avant que les poids soient chargés. La valeur est CONTRÔLÉE contre
// `model.embed_tokens.dim(.voc)` dès que celui-ci existe : un checkpoint d'un autre vocab fait
// échouer le run au lieu de valider silencieusement une politique bornée de travers.
const VOCAB_CONTRACT: u32 = 262144;

// Plancher de température — DIVERGENCE DÉLIBÉRÉE ET DÉCLARÉE avec HF (spec F15). HF n'exige que
// `t > 0` et accepte donc `1e-45`, qui fait déborder la division en f32 et produit des `NaN`
// (`inf - inf` dans le softmax). Les logits sont bornés par le softcap 30 : 1e-30 laisse une
// marge de 8 ordres de grandeur avant le débordement de f32 (3,4e38).
const T_MIN: f32 = 1e-30;

// Ligne de log de la politique — UNE par run, c'est ce que les gates greppent.
// ⚠ LE FORMAT EST IMPOSÉ, PAS LAISSÉ AU `{any}` DE ZIG : `{any}` sur une slice produit
// `{ 258883, 258882 }` (accolades, espaces) là où les gates attendent `[258883,258882]`. GC2 est
// un gate à règle d'arrêt DURE : il FAIL à tort sur cette seule différence de rendu. D'où la
// boucle de formatage manuelle ci-dessous, et la chaîne exacte pré-enregistrée à la spec §4.1.
fn joinList(allocator: std.mem.Allocator, items: []const u32) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    try out.append(allocator, '[');
    for (items, 0..) |v, i| {
        if (i != 0) try out.append(allocator, ',');
        const s = try std.fmt.allocPrint(allocator, "{d}", .{v});
        defer allocator.free(s);
        try out.appendSlice(allocator, s);
    }
    try out.append(allocator, ']');
    return out.toOwnedSlice(allocator);
}

fn joinKeys(allocator: std.mem.Allocator, items: []const []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    try out.append(allocator, '[');
    for (items, 0..) |v, i| {
        if (i != 0) try out.append(allocator, ',');
        try out.appendSlice(allocator, v);
    }
    try out.append(allocator, ']');
    return out.toOwnedSlice(allocator);
}

fn logGencfg(allocator: std.mem.Allocator, policy: *const gencfg.GenCfg, is_oracle: bool) !void {
    if (!policy.enabled) {
        log.info("GENCFG: DÉSACTIVÉ (--no-gen-config) — politique de décodage non appliquée", .{});
        return;
    }
    const sup = try joinList(allocator, policy.suppress);
    defer allocator.free(sup);
    const eos = try joinList(allocator, policy.eos);
    defer allocator.free(eos);
    // `ignored` est DÉRIVÉ des clés présentes au fichier, jamais codé en dur : sans ce segment, le
    // log laisserait croire que « generation_config.json est appliqué » tout court, alors que ce
    // chantier n'en applique que 2 clés sur 8.
    const ign = try joinKeys(allocator, policy.ignored);
    defer allocator.free(ign);
    log.info("GENCFG: {s} suppress={s} eos={s} ignored={s} (mode={s})", .{
        policy.path, sup, eos, ign, if (is_oracle) "oracle" else "libre",
    });
}

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
        // `gencfg.TOP_K` est un comptime de MÊME VALEUR que le littéral `5` d'avant : le
        // StableHLO émis est identique, et GC0 le prouve (md5 du dump before_optimizations).
        const t5 = logits.topK(.{ .voc = .voc }, gencfg.TOP_K, .{});
        // DONATION des caches (spec 2026-07-26 cache-donation) : les 4 caches de sortie aliasent
        // les buffers d'ENTRÉE (reuseBuffer → input_output_alias à la compile) — supprime le
        // double-buffering (2×5,6 GiB à 8k, le mur M3). Contrat : le host ne relit JAMAIS un
        // cache d'entrée après le call (la boucle et le vacuity rebindent/recréent — vérifié).
        return .{ t5.values, t5.indices, logits,
            slk.reuseBuffer(cache.sl_k), slv.reuseBuffer(cache.sl_v),
            flk.reuseBuffer(cache.fl_k), flv.reuseBuffer(cache.fl_v) };
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

// K5 — variante DEUX clés de writeIdsSafetensors : "ids" (les générés, format historique INTACT
// pour tous les consommateurs existants) + "ctx_ids" (la séquence COMPLÈTE feedée avant
// génération, c.-à-d. ids_full). L'oracle 69 --context-ids en a besoin : sous reprise avec prompt
// neuf, le contexte n'est PAS exprimable par un --prompt templaté, il n'existe que sous forme
// d'ids. Écrite à côté de la fonction historique plutôt qu'en la modifiant : la reprise simple et
// tous les runs normaux doivent produire le même fichier qu'avant (claim C-K5-E).
fn writeIdsCtxSafetensors(allocator: std.mem.Allocator, io: std.Io, path: []const u8, ids_i64: []const i64, ctx: []const u32) !void {
    const n = ids_i64.len;
    const c = ctx.len;
    const data = try allocator.alloc(i32, n + c);
    defer allocator.free(data);
    for (ids_i64, 0..) |t, k| data[k] = @intCast(t); // ids < vocab 262144 : cast sans perte
    for (ctx, 0..) |t, k| data[n + k] = @intCast(t);
    const header = try std.fmt.allocPrint(allocator, "{{\"ids\":{{\"dtype\":\"I32\",\"shape\":[{d}],\"data_offsets\":[0,{d}]}},\"ctx_ids\":{{\"dtype\":\"I32\",\"shape\":[{d}],\"data_offsets\":[{d},{d}]}}}}", .{ n, n * 4, c, n * 4, (n + c) * 4 });
    defer allocator.free(header);
    var len_le: [8]u8 = undefined;
    std.mem.writeInt(u64, &len_le, header.len, .little);
    const f = try std.Io.Dir.createFile(.cwd(), io, path, .{});
    defer f.close(io);
    try f.writePositionalAll(io, &len_le, 0);
    try f.writePositionalAll(io, header, 8);
    try f.writePositionalAll(io, std.mem.sliceAsBytes(data), 8 + header.len);
}

// === DC1 (spec kvdump §5) : round-trip du format kvdump + MUTANT, host-only. Aucun GPU, aucun
// poids : c'est le format de fichier qu'on teste, pas le modèle. `dir` doit EXISTER (l'API
// std.Io.Dir de cette toolchain n'expose pas de makeDir — le run le crée en amont).
// Le gate ne vaut que parce qu'il contient sa propre contre-preuve : un octet de données flippé,
// manifest INTACT, DOIT rendre KvDumpChecksumMismatch. Sans ce mutant, un checksum jamais vérifié
// passerait pour vérifié (leçon feedback_invariant_tue_le_controle).
fn selftestKvdumpIo(allocator: std.mem.Allocator, io: std.Io, dir: []const u8) !void {
    // (0) Compatibilité INTER-LANGAGE du checksum, contrôlée AVANT tout le reste : le manifest est
    // écrit par Zig et relu par Python (script 71, gate DC1 volet Python). Si les deux xxh64 ne
    // sont pas la même fonction, tout le dispositif de checksums est un décor — et on ne
    // l'apprendrait qu'après un run GPU. Référence MESURÉE sur la VM (Task 0.4) :
    // `xxhash.xxh64(b"x").intdigest()` == 6665539201184043299.
    const ref_x = kvdump.xxh64("x");
    if (ref_x != 6665539201184043299) {
        log.err("KVIO: xxh64(\"x\") = {d} != 6665539201184043299 (référence Python xxhash) — les deux implémentations ne calculent pas le même hash", .{ref_x});
        return error.KvIoHashDivergence;
    }

    var a_vals: [24]f32 = undefined;
    for (0..24) |k| a_vals[k] = @floatFromInt(k);
    var b_vals: [6]i32 = .{ 7, -1, 0, 262143, 3, 12 };
    const a_bytes = std.mem.sliceAsBytes(a_vals[0..]);
    const b_bytes = std.mem.sliceAsBytes(b_vals[0..]);
    const a_h = kvdump.xxh64(a_bytes);
    const b_h = kvdump.xxh64(b_bytes);

    const path = try std.fmt.allocPrint(allocator, "{s}/self.kvdump", .{dir});
    defer allocator.free(path);

    var hx_a: [32]u8 = undefined;
    var hx_b: [32]u8 = undefined;
    const meta = [_]kvdump.MetaKV{
        .{ .k = "format", .v = kvdump.FORMAT },
        .{ .k = "l_max", .v = "9999" },
        .{ .k = "step_next", .v = "6" },
        .{ .k = "a_xxh64", .v = try std.fmt.bufPrint(&hx_a, "{x}", .{a_h}) },
        .{ .k = "b_xxh64", .v = try std.fmt.bufPrint(&hx_b, "{x}", .{b_h}) },
    };
    const sh_a = [_]i64{ 2, 3, 4 };
    const sh_b = [_]i64{6};
    const tensors = [_]kvdump.TensorOut{
        .{ .name = "a", .dtype = "F32", .shape = &sh_a, .bytes = a_bytes },
        .{ .name = "b", .dtype = "I32", .shape = &sh_b, .bytes = b_bytes },
    };
    kvdump.write(allocator, io, path, &tensors, &meta) catch |e| {
        log.err("KVIO: écriture impossible ({s}) : {s} — le répertoire existe-t-il ?", .{ path, @errorName(e) });
        return e;
    };

    // (1) relecture : header + meta + shapes + données, dans des buffers NEUFS.
    var f = try std.Io.Dir.cwd().openFile(io, path, .{ .mode = .read_write });
    defer f.close(io);
    var h = try kvdump.readHeader(allocator, io, f, path);
    defer h.deinit();
    const fmt_got = h.metaGet("format") orelse return error.KvDumpBadFormat;
    if (!std.mem.eql(u8, fmt_got, kvdump.FORMAT)) {
        log.err("KVIO: format relu = {s} != {s}", .{ fmt_got, kvdump.FORMAT });
        return error.KvDumpBadFormat;
    }
    const l_max_got = try kvdump.metaInt(&h, "l_max", 10);
    const step_next_got = try kvdump.metaInt(&h, "step_next", 10);
    if (l_max_got != 9999 or step_next_got != 6) {
        log.err("KVIO: manifest non round-trippé : l_max={d} step_next={d}", .{ l_max_got, step_next_got });
        return error.KvIoMetaRoundTrip;
    }
    try kvdump.expectShape(&h, "a", "F32", &sh_a);
    try kvdump.expectShape(&h, "b", "I32", &sh_b);

    const got_a = try allocator.alloc(u8, a_bytes.len);
    defer allocator.free(got_a);
    const got_b = try allocator.alloc(u8, b_bytes.len);
    defer allocator.free(got_b);
    try kvdump.readTensorInto(io, f, path, &h, "a", got_a, a_h);
    try kvdump.readTensorInto(io, f, path, &h, "b", got_b, b_h);
    if (!std.mem.eql(u8, got_a, a_bytes) or !std.mem.eql(u8, got_b, b_bytes)) {
        log.err("KVIO: round-trip NON bit-identique (a_ok={} b_ok={})", .{ std.mem.eql(u8, got_a, a_bytes), std.mem.eql(u8, got_b, b_bytes) });
        return error.KvIoRoundTrip;
    }

    // (2) MUTANT INTÉGRÉ : 1 octet flippé dans les DONNÉES de `a`, manifest INTACT.
    // La contre-preuve du gate : si ce flip passe, le checksum ne vérifie rien.
    const mut_off = h.data_base + 3;
    var one: [1]u8 = undefined;
    if (try f.readPositionalAll(io, &one, mut_off) != 1) return error.KvIoMutantSetup;
    one[0] ^= 0xFF;
    try f.writePositionalAll(io, &one, mut_off);
    const mut_res = kvdump.readTensorInto(io, f, path, &h, "a", got_a, a_h);
    if (mut_res) |_| {
        log.err("KVIO: MUTANT NON VU — un octet flippé a passé le checksum : le contrôle est VACUEUX", .{});
        return error.KvIoMutantNotSeen;
    } else |e| {
        if (e != error.KvDumpChecksumMismatch) {
            log.err("KVIO: mutant refusé par la mauvaise erreur : {s} (attendu KvDumpChecksumMismatch)", .{@errorName(e)});
            return error.KvIoMutantWrongError;
        }
    }

    log.info("KVIO: round-trip PASS + mutant VU (ChecksumMismatch)", .{});
}

pub fn run(init: std.process.Init) !void {
    @setEvalBranchQuota(200000); // piège quota comptime (cf gemma4_gchunk_auto.zig:96)
    const arena = init.arena;
    // D10 (C1) : point de substitution UNIQUE — tout ce qui alloue via `allocator` est compté.
    var counter = alloc_count.CountingAllocator.init(init.gpa);
    const allocator = counter.allocator();
    const io = init.io;
    // Bannière §8 : le mode est PROUVÉ dans chaque log (y compris selftests host, avant tout
    // early-return) — un log de gate sans `mode=ReleaseFast` est INEXÉCUTABLE, pas PASS.
    log.info("BUILD: mode={s}", .{@tagName(builtin.mode)});

    const process_args = try init.minimal.args.toSlice(arena.allocator());
    const args = try parseArgs(process_args);

    // === --repl (spec 2026-07-26 repl-mode) : EXCLUSIF des modes fixtures/probe — un REPL avec
    // oracle/out-ids écraserait ou accumulerait silencieusement des artefacts de gate ; refusé
    // au lancement (garde AVANT tout early-return de mode). ===
    if (args.repl and (args.oracle_path != null or args.window_vacuity != null or
        args.out_ids != null or args.ids_only or args.ids_only_turn2 or args.selftest_inputs != null or
        args.selftest_gather != null or args.selftest_gencfg != null or
        args.selftest_sampling != null or args.selftest_draw != null or
        args.selftest_alloc_count or args.selftest_kvdump_io != null or args.selftest_kvdump_eq != null))
    {
        log.err("--repl est exclusif de --oracle/--window-vacuity/--out-ids/--ids-only/--ids-only-turn2/--selftest-*\n{s}", .{usage});
        return error.ConflictingFlags;
    }

    // === kvdump (spec 2026-08-09 §4.5) : les QUATRE refus de combinaisons de flags. Placés ICI,
    // au plus tôt (~1 s, aucune compile, aucun GPU) : un flag silencieusement inopérant serait un
    // mensonge, et chacun de ces refus est VU échouer par le gate DC5. ===
    if (args.load_cache != null and args.repl) {
        log.err("--load-cache + --repl non supporté (spec §3 : sémantique multi-tour absente)", .{});
        return error.LoadCacheReplUnsupported;
    }
    if (args.dump_cache != null and args.repl) {
        log.err("--dump-cache + --repl non supporté v1 (spec §4.5 : un flag inopérant serait un mensonge)", .{});
        return error.DumpCacheReplUnsupported;
    }
    // K5 (spec 2026-08-10 §4.1) : --load-cache + --prompt est désormais le PREFILL PARTIEL — le
    // contexte vient du dump, le prompt est absorbé comme TOUR 2 aux positions step_next… .
    // L'ancienne garde LoadCacheWithPrompt (spec kvdump §3) est LEVÉE par ce chantier ; ce qui la
    // remplace n'est pas une garde mais une preuve : l'aller-retour teacher-forcé PF1/PF3.
    // Décision Régis ACTÉE (GO 9 août) : sampling armé + dump = refus bruyant. L'état du PRNG
    // Xoshiro256 n'est PAS sérialisé — le sérialiser ajouterait une claim d'équivalence
    // stochastique qu'aucun gate simple ne prouve. Dette écrite au doc de résultats.
    if (args.dump_cache != null and args.seed != null) {
        log.err("--dump-cache + sampling armé non supporté v1 (état PRNG non sérialisé — spec §3, dette)", .{});
        return error.DumpWithSamplingArmed;
    }
    // Phase 1 (penalty) : sous reprise, `ids` vaut `ids_fed` COMPLET — prompt d'origine ET tokens
    // déjà générés par le run dumpé. « Le prompt » n'y est plus une notion définie : la frontière
    // que `--ignore-prompt` doit couper n'est pas reconstructible depuis le dump. Refus bruyant
    // plutôt qu'une sémantique inventée. Dette écrite v1.
    if (args.ignore_prompt and args.load_cache != null) {
        log.err("--ignore-prompt + --load-cache non supporté v1 : sous reprise, ids = ids_fed complet (prompt + tokens générés du run dumpé) — la frontière du prompt n'existe plus", .{});
        return error.IgnorePromptWithLoadCache;
    }

    // === Task 3 : --selftest-inputs — indépendant du prompt/tokenizer/poids (fixture only) ===
    if (args.selftest_inputs) |fixture_path| {
        try selftestInputs(allocator, io, fixture_path);
        return;
    }

    // === GC1 : --selftest-gencfg — même patron d'early-return, AVANT tokenizer/VRAM/Platform.
    // Le chemin est host-only par construction : c'est ce qui rend le gate exécutable sans GPU
    // (spec §4.5bis) et ce qui interdit d'y mesurer l'eot_id (il vient de la fixture). ===
    if (args.selftest_gencfg) |fixture_path| {
        try selftestGencfg(allocator, io, fixture_path);
        return;
    }

    // === S2-U : --selftest-sampling — même patron host-only. C'est ce qui rend le gate
    // exécutable sans GPU (spec phase 2, objectif O1 : itérer en secondes). ===
    if (args.selftest_sampling) |fixture_path| {
        try selftestSampling(allocator, io, fixture_path);
        return;
    }

    // === S2-D : tirage sur logits FIGÉS. Host-only — aucun GPU, aucun modèle : c'est la
    // distribution du sampler qu'on teste, pas celle du modèle. ===
    if (args.selftest_draw) |fixture_path| {
        try selftestDraw(allocator, io, fixture_path, args.draws, args.seed);
        return;
    }

    // === RP1 (phase 1, penalty) : même patron host-only, AVANT tokenizer/VRAM/Platform/poids.
    // La fonction testée est celle que la boucle de génération appellera — pas une copie. ===
    if (args.selftest_penalty) |fixture_path| {
        try selftestPenalty(allocator, io, fixture_path);
        return;
    }

    // === S-AC (D10) : le compteur compte — host-only, aucun GPU. 3 alloc + 1 resize +
    // 1 remap + 2 free via le wrapper, compteurs comparés aux attendus (spec D10 §5). ===
    if (args.selftest_alloc_count) {
        try selftestAllocCount(allocator, &counter);
        return;
    }

    // === DC1 (kvdump) : round-trip fichier + mutant, host-only — même patron d'early-return,
    // AVANT tokenizer/VRAM/Platform/poids. C'est ce qui rend le gate exécutable en une seconde
    // et sans GPU (spec kvdump §5, livrable 1). ===
    if (args.selftest_kvdump_io) |dir| {
        try selftestKvdumpIo(allocator, io, dir);
        return;
    }

    // --repl : --prompt devient OPTIONNEL (s'il est fourni : premier prompt de la boucle).
    // --load-cache : --prompt est INTERDIT (refus ci-dessus) — les ids viennent du manifest du
    // dump. Sans cette exemption, TOUTE reprise mourrait ici, avant même la lecture du fichier
    // (finding bloquant de la revue kvdump).
    const prompt_text = args.prompt orelse blk: {
        if (args.repl or args.load_cache != null) break :blk "";
        log.err("--prompt est requis (sauf --repl et --load-cache)\n{s}", .{usage});
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

    // Tokenisation du prompt CLI via promptToIds (extraction repl-mode — même chemin qu'en
    // résident). En --repl sans --prompt : liste vide, la boucle stdin fournira les prompts.
    // --load-cache : aucune tokenisation — `ids` sera REMPLI par `ids_fed` du manifest (Task 5,
    // avant les pré-checks de run()). Le tokenizer reste chargé : le décodage de la continuation
    // en a besoin.
    var ids: std.ArrayList(u32) = if (args.load_cache != null or (args.repl and prompt_text.len == 0))
        .empty
    else
        try promptToIds(allocator, &encoder, prompt_text);
    defer ids.deinit(allocator);

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

        const round_trip_ok = std.mem.eql(u32, reenc.items, ids.items[1..]);
        if (round_trip_ok) {
            log.info("round-trip détok : PASS (decode -> re-encode == ids)", .{});
        } else {
            log.err("round-trip détok : FAIL — got={any} want={any}", .{ reenc.items, ids.items[1..] });
            return error.RoundTripFailed;
        }
        return;
    }

    // === K5/PF6 : --ids-only-turn2 — le rendu du TOUR 2 en ids, host-only (aucun dump, aucun GPU,
    // aucune politique). C'est la sortie que le gate PF6 compare littéralement au suffixe HF mesuré
    // (docs/evidence/k5/rendu_tour2_hf.json, clé suffix_ids). Même patron d'early-return que
    // --ids-only : APRÈS la tokenisation, AVANT la garde VRAM. ===
    if (args.ids_only_turn2) {
        if (prompt_text.len == 0) {
            log.err("--ids-only-turn2 exige --prompt (le texte du tour 2)", .{});
            return error.PromptTooLong;
        }
        var t2 = try promptToIdsTurn2(allocator, &encoder, prompt_text);
        defer t2.deinit(allocator);
        log.info("ids_turn2 = {any}", .{t2.items});
        return;
    }

    // === Garde VRAM (docs/VRAM_CHECK_DESIGN.md) — avant tout travail GPU. Les modes host-only
    // (--selftest-inputs/--ids-only) ont déjà early-return au-dessus. `--selftest-gather` N'EST
    // PLUS host-only depuis L3 (spec [it.4]) : il compile un mini-graphe GPU (`SgFwd`) et passe
    // désormais PAR cette garde, comme le run normal (dispatché plus bas, après Platform.init).
    // Tourne AUSSI en --allow-cpu : ce flag ne force pas le CPU (l'init .cuda est tentée d'abord,
    // --allow-cpu ne tolère que le repli) — sur machine sans GPU, nvidia-smi absent → warn +
    // continue, donc pas de blocage à tort. Seul --force-vram saute la garde. ===
    // === Politique de décodage `generation_config.json` (spec 2026-07-28 §4.1) — chargée ICI :
    // APRÈS l'early-return `--ids-only` et l'eot_id mesuré au tokenizer (le contrôle croisé
    // `EotNotInEosList` en dépend), et AVANT la garde VRAM. FAIL-FAST : un fichier de politique
    // invalide doit coûter une seconde, pas une compile GPU de 40 s.
    //
    // ⚠ `vocab_size` : le vrai vocab se mesure sur `model.embed_tokens.dim(.voc)`, qui n'existe
    // qu'APRÈS le chargement des poids — donc après ce point. On borne ici sur la constante de
    // contrat U0, et on CONTRÔLE ce choix contre la mesure dès qu'elle est disponible (recherche
    // `VOCAB_CONTRACT` plus bas) : un checkpoint dont le vocab diffère fait échouer le run, il ne
    // passe pas en silence. Un hardcode non contrôlé serait un pari ; celui-ci est falsifiable.
    if (args.gen_config != null and args.no_gen_config) {
        log.err("--gen-config et --no-gen-config sont contradictoires\n{s}", .{usage});
        return error.ConflictingFlags;
    }
    var policy = if (args.no_gen_config)
        try gencfg.disabled(allocator, eot_id)
    else
        try gencfg.load(allocator, io, args.gen_config, args.ckpt, .{
            .vocab_size = VOCAB_CONTRACT,
            .eot_id = eot_id,
        });
    defer policy.deinit(allocator);
    try logGencfg(allocator, &policy, args.oracle_path != null);

    // === Sampling phase 2 : scratch et RNG alloués UNE FOIS (interdit « aucune allocation par
    // step », spec §5). Dimensionnés au contrat U0 puis recoupés contre le vocab mesuré. ===
    var scfg: sampling.SamplingCfg = .{
        .temperature = args.temperature,
        .top_k = args.top_k,
        .top_p = args.top_p,
        .min_keep = args.min_tokens_to_keep,
        .seed = args.seed,
        .repetition_penalty = args.repetition_penalty,
        .ignore_prompt = args.ignore_prompt,
    };
    // ⚠ `chain_armed` est figé ICI, et c'est ce qui pilote allocations ET `defer` — jamais un
    // `scfg.pathArmed()` ré-évalué. Deux raisons, toutes deux mordantes depuis que le repl a des
    // directives : (1) `:penalty` peut armer le chemin B EN COURS de session, sur des buffers qui
    // n'auraient jamais été alloués (mordu au gate RP5 : « chemin B : logits 1048576 octets !=
    // work 0 octets ») ; (2) un `defer if (scfg.pathArmed())` ré-évalué à la sortie deviendrait
    // VRAI après un `:penalty` et libérerait un `scratch` jamais initialisé — un `undefined`
    // passé à `deinit`. La condition de libération doit être la MÊME EXPRESSION que celle
    // d'allocation, pas une expression qui lui ressemble.
    const chain_armed = scfg.pathArmed() or args.repl;
    // D10 (C7) : `work` n'est plus alloué ICI — il déménage APRÈS la création de la Platform
    // (DmaAllocator exige un Device vivant), avec son defer. Le scratch et le reste restent.
    if (chain_armed) {
        scfg.scratch = try sampling.Scratch.init(allocator, VOCAB_CONTRACT);
        scfg.resetPerPrompt();
        log.info("SAMPLING: T={d} top_k={d} top_p={d} min_keep={d} seed={?d} (chemin complet {s} ; tirage {s})", .{ scfg.temperature, scfg.top_k, scfg.top_p, scfg.min_keep, scfg.seed, if (scfg.pathArmed()) "ARMÉ" else "prêt (repl : armable par directive)", if (scfg.drawArmed()) "ON" else "OFF (argmax)" });
    } else {
        log.info("SAMPLING: neutre — chemin top-5 inchangé (aucun warper, aucun tirage)", .{});
    }
    defer if (chain_armed) scfg.scratch.deinit(allocator);

    // === Phase 1 (penalty) : historique et bitset alloués UNE FOIS, ici — jamais dans la boucle
    // (interdit D10, `ALLOC-LOOP: alloc=0`). Alloués SEULEMENT si la penalty est armée : à 1.0 le
    // runner ne paie ni les L_MAX×4 octets ni les 32 Kio, et n'écrit rien de plus par step — c'est
    // ce qui rend RP2 (non-régression bit-identique) vrai par construction et non par chance.
    // La borne L_MAX est celle que la garde de lancement de `generateOnce` fait déjà respecter
    // (`ids.len + limit <= L_MAX`) : l'historique ne peut donc pas déborder, et une garde de borne
    // explicite reste posée à l'append (une borne « impossible » non gardée est une UB en attente).
    // ⚠ En `--repl`, on alloue MÊME si la penalty est neutre au lancement : la directive
    // `:penalty` peut l'armer en cours de session, et sans buffers elle serait un flag inopérant
    // — exactement le mensonge que le repo refuse ailleurs. Coût du cas neutre : L_MAX×4 octets
    // + 32 Kio, une fois par session résidente, et une écriture d'id par step.
    const penalty_armed = scfg.repetition_penalty != 1.0 or args.repl;
    if (penalty_armed) {
        scfg.hist = try allocator.alloc(u32, @intCast(L_MAX));
        scfg.seen = try allocator.alloc(u64, (VOCAB_CONTRACT + 63) / 64);
        log.info("PENALTY: buffers alloués — rp={d} ignore_prompt={} (hist {d} ids max, seen {d} Kio{s})", .{ scfg.repetition_penalty, scfg.ignore_prompt, L_MAX, (scfg.seen.len * @sizeOf(u64)) / 1024, if (scfg.repetition_penalty == 1.0) " ; NEUTRE au lancement, armable par :penalty" else "" });
    } else if (scfg.ignore_prompt) {
        // Un flag inopérant est un mensonge : le dire au lieu de le laisser passer en silence.
        log.warn("--ignore-prompt sans --repetition-penalty : SANS EFFET (il n'y a pas de penalty à restreindre)", .{});
    }
    defer if (penalty_armed) {
        allocator.free(scfg.hist);
        allocator.free(scfg.seen);
    };

    // === Gates G-D1/G-D2 (dettes D1/D2) — pont in-process contre une référence indépendante.
    // Spec : docs/superpowers/specs/2026-08-10-d1d2-gpu-coverage.md
    var gate_storage: sampling.GateD1D2 = undefined;
    if (args.gate_d1d2) {
        // Refus BRUYANT plutôt qu'un gate qui compterait 0 step : sans warper armé, le chemin B
        // n'est pas pris et la comparaison n'aurait JAMAIS lieu — un PASS vide, le pire cas.
        if (!scfg.pathArmed()) {
            log.err("--gate-d1d2 exige un régime ARMÉ (--top-k / --top-p / --temperature / --seed) : sans warper, le chemin B n'est pas pris et le gate passerait À VIDE", .{});
            return error.GateD1D2NotArmed;
        }
        gate_storage = try sampling.GateD1D2.init(allocator, VOCAB_CONTRACT);
        scfg.gate = &gate_storage;
        log.info("GATE-D1D2: armé — applyTopP comparé à une référence descendante f64, mutants température (division, ordre de chaîne) actifs", .{});
    }
    defer if (args.gate_d1d2) gate_storage.deinit(allocator);

    if (args.force_vram) {
        log.warn("--force-vram : garde VRAM sautée (OOM possible en aval, assumé)", .{});
    } else {
        try checkVram(allocator, io);
    }

    // === kvdump — PHASE 1 du restore (spec §4.3) : manifest + validations de forme + ids_fed,
    // AVANT la compile ET AVANT les pré-checks de run(). `ids` reçoit ids_fed ICI : les gardes
    // qui suivent (garde oracle `positions[0] == ids.len`, garde de place `ids.len + limit >
    // L_MAX`) travaillent donc sur l'état RÉEL de la reprise, pas sur une liste vide. ===
    var mcheck: ?ManifestCheck = null;
    defer if (mcheck) |*mc| mc.deinit(allocator, io);
    if (args.load_cache) |cache_path| {
        mcheck = try loadCacheManifest(allocator, io, cache_path, args.ckpt, policy.path);
        try ids.appendSlice(allocator, mcheck.?.ids_fed);
        if (args.prompt != null) {
            // K5 : prompt VIDE gardé AVANT le rendu — le rendu émet toujours ses marqueurs de
            // tour, `n_new` ne peut donc jamais valoir 0 après lui (une garde post-rendu serait à
            // antécédent vide, feedback_test_vacuite_antecedent). C'est le cas PF4(d).
            if (prompt_text.len == 0) {
                log.err("K5 : --prompt vide sous reprise — un tour 2 sans contenu n'est pas un prefill partiel", .{});
                return error.PromptTooLong;
            }
            // K5 (spec §4.1) : ids_full = ids_fed ++ [fed_next] ++ [clôture] ++ ids_t2.
            // fed_next est TOUJOURS inclus (D-K5-1) : dernier token généré, jamais feedé — il fait
            // partie du texte produit (y compris un EOS de fin de tour).
            try ids.append(allocator, @intCast(mcheck.?.fed_next));
            // D-K5-5 : le rendu HF ferme TOUJOURS le tour assistant avant le suivant. La MESURE
            // (Task 2) donne la clôture exacte : `<turn|>` + '\n' quand fed_next n'est pas un EOS
            // (arrêt max_tokens, cas nominal d'un run A borné), '\n' seul sinon.
            const fed_is_eos = policy.isEos(mcheck.?.fed_next);
            var closure = try closureToIds(allocator, &encoder, fed_is_eos);
            defer closure.deinit(allocator);
            try ids.appendSlice(allocator, closure.items);
            log.info("K5: tour 1 clos par {any} (fed_next={d} {s} un EOS)", .{ closure.items, mcheck.?.fed_next, if (fed_is_eos) "EST" else "n'est PAS" });
            var t2 = try promptToIdsTurn2(allocator, &encoder, prompt_text);
            defer t2.deinit(allocator);
            const n_new = t2.items.len; // ids du tour 2 (le SEUL segment réellement prefillé)
            // D-K5-3 : garde fenêtre TRANSPOSÉE au prompt neuf. La garde historique (juste en
            // dessous, et sa jumelle `:3058`) est désactivée sous reprise ; sa raison d'être n'a
            // jamais été écrite (c2211c0) — on transpose la prudence au seul segment qui est
            // effectivement prefillé. La LEVER exigerait son propre gate : dette écrite.
            if (n_new >= @as(usize, @intCast(SLIDING_WINDOW))) {
                log.err("K5 : tour 2 de {d} ids >= SLIDING_WINDOW({d})", .{ n_new, SLIDING_WINDOW });
                return error.PromptTooLong;
            }
            try ids.appendSlice(allocator, t2.items);
            log.info("K5: prefill partiel — contexte {d} ids + fed_next + clôture {d} + tour2 {d} ids = {d} total", .{ mcheck.?.ids_fed.len, closure.items.len, n_new, ids.items.len });
        }
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
        // Déviation historique (longueur seule) : les prompt_ids complets ne vivaient que dans le
        // manifest sidecar JSON — positions[0]==ids.len était le check le plus fort possible sur la
        // fixture seule ; un prompt FAUX de même longueur échouerait bruyamment au compare step 0.
        if (positions_fx[0] != @as(i32, @intCast(ids.items.len))) {
            log.err("--oracle : positions[0]={d} (seq_len fixture) != ids.len={d} (prompt rendu) — mismatch prompt/fixture", .{ positions_fx[0], ids.items.len });
            return error.OraclePromptMismatch;
        }
        // Depuis la phase 1 (penalty), l'oracle exporte AUSSI `prompt_ids` en tenseur : la
        // comparaison devient LITTÉRALE et la déviation ci-dessus est soldée. Le tenseur reste
        // OPTIONNEL — les fixtures historiques du repo (u8_gen48 et sa famille) ne le portent pas,
        // et les casser pour renforcer un check serait un mauvais échange. Absent : on le DIT.
        if (reg.tensors.get("prompt_ids") != null) {
            const pids = try readFixtureAlloc(i32, .i32, allocator, io, &reg, &file, "prompt_ids");
            defer allocator.free(pids);
            if (pids.len != ids.items.len) {
                log.err("--oracle : prompt_ids de la fixture = {d} ids != prompt rendu {d}", .{ pids.len, ids.items.len });
                return error.OraclePromptMismatch;
            }
            for (pids, 0..) |p, i| {
                if (p != @as(i32, @intCast(ids.items[i]))) {
                    log.err("--oracle : prompt_ids[{d}] = {d} (fixture) != {d} (prompt rendu) — template ou prompt DIFFÉRENT ; un mismatch d'ids qui suivrait serait attribué à tort au sampling", .{ i, p, ids.items[i] });
                    return error.OraclePromptMismatch;
                }
            }
            log.info("--oracle : prompt vérifié LITTÉRALEMENT ({d} ids identiques à la fixture)", .{pids.len});
        } else {
            log.warn("--oracle : fixture sans tenseur 'prompt_ids' (antérieure à la phase 1) — vérification du prompt limitée à sa LONGUEUR", .{});
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
    // ⚠ En mode --load-cache cette garde est DÉSACTIVÉE (spec §4.3) : elle protège le PREFILL,
    // qui n'a pas lieu — un état repris dépasse légitimement la fenêtre glissante, le cache la
    // porte déjà. La garde de place ci-dessus, elle, RESTE active (ids.len == step_next par
    // l'invariant vérifié en phase 1) : c'est elle qui réalise le refus `SequenceTooLong` de la
    // validation §4.3(7) — même condition, même erreur, même moment (avant compile).
    if (args.load_cache == null and ids.items.len >= @as(usize, @intCast(SLIDING_WINDOW))) {
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

    // === D10 (C7/M-PIN) : work alloué APRÈS la Platform (DmaAllocator exige un Device vivant)
    // — le repli est ÉCRIT (ZML ne replie jamais : null → OutOfMemory, mem.zig:145-153), et
    // pin_on UNIQUE pilote l'alloc, la bannière ET le free : « libéré par l'allocateur qui a
    // alloué » est STRUCTUREL, pas puni par un panic (catch unreachable = UB en ReleaseFast). ===
    var dma = zml.mem.DmaAllocator.init(allocator, &platform.devices[0]);
    var pin_on = !args.no_pin;
    if (chain_armed) { // même condition qu'à l'allocation du scratch — cf note sur `chain_armed`
        if (pin_on) {
            scfg.work = dma.allocator().alloc(f32, VOCAB_CONTRACT) catch blk: {
                pin_on = false;
                break :blk try allocator.alloc(f32, VOCAB_CONTRACT);
            };
        } else {
            scfg.work = try allocator.alloc(f32, VOCAB_CONTRACT);
        }
        if (pin_on) {
            log.info("PIN: ON (work 1 MiB dma-mappé)", .{});
        } else if (args.no_pin) {
            log.info("PIN: OFF (--no-pin)", .{});
        } else {
            log.info("PIN: OFF (alloc DMA échouée : dmaMap indisponible ou OOM transitoire)", .{});
        }
    }
    // Ordre LIFO : ce defer est déclaré APRÈS `defer platform.deinit` → il s'exécute AVANT lui
    // (dmaUnmap exige la plateforme vivante).
    defer if (chain_armed) {
        if (pin_on) dma.allocator().free(scfg.work) else allocator.free(scfg.work);
    };

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
    // (cache_buf : construit PAR GÉNÉRATION dans generateOnce — spec repl-mode : chaque prompt
    //  résident repart d'un cache zéro, position 0.)
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
        // D10 (C6) : buffer de logits alloué UNE FOIS pour les 2 passes — hors fenêtre ALLOC-VAC.
        const wv_work = try allocator.alloc(f32, voc_us);
        defer allocator.free(wv_work);
        var first_div: ?usize = null;
        var div_max_abs: f64 = 0;
        var n_ident: usize = 0;
        // D10 (C4) : args/results du mode vacuity hissés — mêmes raisons que la boucle de gen,
        // AVANT les snapshots ALLOC-VAC (hors fenêtre).
        var wv_args = try exe.args(allocator);
        defer wv_args.deinit(allocator);
        var wv_results = try exe.results(allocator);
        defer wv_results.deinit(allocator);
        // === D10 : fenêtre ALLOC-VAC — les boucles de steps SEULES (le buffer logits_p1 et les
        // init hissés C4/C6 sont HORS fenêtre, spec §3/C6). ===
        const av0_alloc = counter.n_alloc;
        const av0_resize = counter.n_resize;
        const av0_remap = counter.n_remap;
        const av0_free = counter.n_free;
        const av0_bytes = counter.bytes_alloc;
        var av_steps: usize = 0;
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
                wv_args.set(.{ eng_buf, tok_buf, if (pass == 0) pk_buf else pk_wide, cache_wv, ctrl_buf });
                exe.call(wv_args, &wv_results);
                var r_t5v, var r_t5i, var r_logits, const r_slk, const r_slv, const r_flk, const r_flv = wv_results.get(struct {
                    zml.Buffer, zml.Buffer, zml.Buffer, zml.Buffer, zml.Buffer, zml.Buffer, zml.Buffer,
                });
                // D10 (C6) : toSlice direct dans wv_work — mêmes raisons que le chemin B (C2).
                const need_wv = r_logits.shape().byteSize();
                if (need_wv != wv_work.len * @sizeOf(f32)) {
                    log.err("WV : logits {d} octets != {d}", .{ need_wv, wv_work.len * @sizeOf(f32) });
                    return error.UnexpectedShape;
                }
                try r_logits.toSlice(io, zml.Slice.init(r_logits.shape(), std.mem.sliceAsBytes(wv_work)));
                const lg = wv_work;
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
                // D10 (C4) : wv_args/wv_results hissés — deinit une fois par les defer d'entête.
                if ((step + 1) % 256 == 0) log.info("  WV passe {d} ... step {d}/{d}", .{ pass + 1, step + 1, s_total });
                av_steps += 1;
            }
            cache_wv.sl_k.deinit();
            cache_wv.sl_v.deinit();
            cache_wv.fl_k.deinit();
            cache_wv.fl_v.deinit();
        }
        // D10 : deltas de la fenêtre ALLOC-VAC (gate AL-VAC).
        log.info("ALLOC-VAC: alloc={d} resize={d} remap={d} free={d} bytes={d} steps={d}", .{
            counter.n_alloc - av0_alloc, counter.n_resize - av0_resize,
            counter.n_remap - av0_remap, counter.n_free - av0_free,
            counter.bytes_alloc - av0_bytes, av_steps,
        });
        if (first_div) |q| {
            log.info("WINDOW-VACUITY : logits bit-identiques sur {d} positions, PREMIÈRE DIVERGENCE à q={d} (max_abs à q : {e:.3}) — attendu D10 : q == 1024 exactement (verdict = Task 10/U9-ii)", .{ n_ident, q, div_max_abs });
        } else {
            log.warn("WINDOW-VACUITY : AUCUNE divergence sur {d} positions — la fenêtre n'a pas mordu (S <= 1024 ?) : rapport VACUEUX pour U9-ii si S > 1024 attendu", .{s_total});
        }
        return;
    }

    // === Dispatch one-shot / résident (spec repl-mode 2026-07-26) — même chemin de code :
    // generateOnce porte la génération complète (gardes, cache zéros, boucle steps, verdicts). ===
    const vocab = model.embed_tokens.dim(.voc);
    // Contrôle croisé de la constante utilisée pour borner la politique au fail-fast (cf
    // `VOCAB_CONTRACT`) : c'est ce qui empêche ce hardcode d'être un pari silencieux. Un
    // checkpoint d'un autre vocab s'arrête ici plutôt que de tourner avec une borne fausse.
    if (vocab != @as(i64, VOCAB_CONTRACT)) {
        log.err("vocab du checkpoint = {d} ≠ VOCAB_CONTRACT = {d} : la politique de décodage a été bornée sur une valeur qui n'est pas celle de ce modèle", .{ vocab, VOCAB_CONTRACT });
        return error.VocabContractMismatch;
    }
    // Writer stdout UNIQUE pour toute la session (fix R1 : un 2e writer sur le même fd
    // entrelace ses octets avec le premier — une seule file d'écriture).
    var stdout_w = std.Io.File.stdout().writer(io, &.{});
    // kvdump : le dump n'existe qu'en one-shot (le refus DumpCacheReplUnsupported garantit qu'on
    // n'arrive jamais ici avec le flag en --repl ; le `null` des sites REPL est un invariant).
    const dump_spec: ?DumpSpec = if (args.dump_cache) |p| .{ .path = p, .ckpt_path = args.ckpt } else null;
    // kvdump — PHASE 2 du restore : APRÈS la compile (elle est hors du chrono KVLOAD-PERF, comme
    // elle est hors du temps de calcul publié côté (i) de DC7 : l'inclure d'un seul côté serait
    // inéquitable). Les 4 caches atterrissent dans host.cache_*, que generateOnce monte en device.
    const resume_state: ?Resume = if (mcheck) |*mc| try loadCacheTensors(io, mc, &host) else null;

    // === DC2 : équivalence intra-process. Dispatché ICI (comme --window-vacuity) : il lui faut
    // l'exécutable compilé. Il appelle loadCacheManifest/loadCacheTensors DIRECTEMENT, donc le
    // refus `LoadCacheWithPrompt` ne le concerne pas — il EXIGE au contraire un --prompt. ===
    if (args.selftest_kvdump_eq) |eq_path| {
        if (ids.items.len == 0) {
            log.err("--selftest-kvdump-eq exige un --prompt (les trois appels partent du même prompt)", .{});
            return error.MissingArgument;
        }
        return selftestKvdumpEq(allocator, &counter, io, platform, sharding, &exe, &eng_buf, &pk_buf, tok_sym, cache_sym, &host, &tokenizer, &stdout_w, eot_id, &policy, &scfg, vocab, ids.items, args.ckpt, eq_path);
    }

    if (!args.repl) {
        return generateOnce(allocator, &counter, io, platform, sharding, &exe, &eng_buf, &pk_buf, tok_sym, cache_sym, &host, &tokenizer, &stdout_w, eot_id, &policy, &scfg, vocab, max_tokens, args.dump_top5, oracle_ids, args.out_ids, ids.items, dump_spec, resume_state, null);
    }

    // === Mode RÉSIDENT : load+compile payés UNE fois, prompts en boucle sur stdin. Chaque
    // prompt est INDÉPENDANT (cache zéros, position 0 — V1, pas de multi-tour). Sortie : ligne
    // vide ou EOF. Erreurs de prompt (trop long, hors vocab) : signalées, boucle continue. ===
    log.info("REPL résident : prompts en boucle (ligne vide ou EOF pour quitter, max_tokens={d})", .{max_tokens});
        if (ids.items.len > 0) { // --prompt fourni : premier prompt de la session
        try stdout_w.interface.print("prompt> {s}\n", .{prompt_text});
        try stdout_w.interface.flush();
        generateOnce(allocator, &counter, io, platform, sharding, &exe, &eng_buf, &pk_buf, tok_sym, cache_sym, &host, &tokenizer, &stdout_w, eot_id, &policy, &scfg, vocab, max_tokens, args.dump_top5, null, null, ids.items, null, null, null) catch |e| switch (e) {
            error.SequenceTooLong, error.PromptTooLong, error.TokenOutOfRange => log.err("prompt refusé ({s}) — prompt suivant", .{@errorName(e)}),
            else => return e,
        };
    }
    var line_buf: [16384]u8 = undefined;
    var stdin_r = std.Io.File.stdin().reader(io, &line_buf);
    while (true) {
        try stdout_w.interface.print("prompt> ", .{});
        try stdout_w.interface.flush();
        // takeDelimiter (PAS takeDelimiterExclusive) : consomme AUSSI le '\n' (Reader.zig:895
        // « advancing the seek position past the delimiter ») — l'Exclusive laisse le \n en tête
        // et le tour suivant lit une ligne vide → sortie prématurée (bug mordu au gate R1).
        // null = fin de flux propre (EOF).
        const line_opt = stdin_r.interface.takeDelimiter('\n') catch |e| switch (e) {
            error.StreamTooLong => {
                log.err("prompt refusé (ligne > {d} octets) — prompt suivant", .{line_buf.len});
                continue;
            },
            else => return e,
        };
        const line_raw = line_opt orelse break;
        const line = std.mem.trim(u8, line_raw, " \t\r");
        if (line.len == 0) break;
        // === Directives (spec §3.4) : une ligne commençant par ':' n'est JAMAIS un prompt. ===
        // Une valeur invalide affiche un message et la session CONTINUE — un repl qui meurt sur
        // une faute de frappe perdrait la compile qu'il est justement là pour amortir.
        if (line[0] == ':') {
            const sp = std.mem.indexOfScalar(u8, line, ' ');
            const cmd = if (sp) |at| line[0..at] else line;
            const arg = if (sp) |at| std.mem.trim(u8, line[at + 1 ..], " \t") else "";
            if (std.mem.eql(u8, cmd, ":penalty")) {
                if (parsePenalty(arg)) |v| {
                    scfg.repetition_penalty = v;
                    try stdout_w.interface.print(":penalty = {d} (appliquée au prompt SUIVANT)\n", .{v});
                } else {
                    try stdout_w.interface.print(":penalty : valeur invalide '{s}' — attendu un réel fini > 0 (1.0 = neutre). Inchangée : {d}\n", .{ arg, scfg.repetition_penalty });
                }
            } else if (std.mem.eql(u8, cmd, ":ignore-prompt")) {
                if (std.mem.eql(u8, arg, "on")) {
                    scfg.ignore_prompt = true;
                    try stdout_w.interface.print(":ignore-prompt = on (seuls les tokens générés sont pénalisés)\n", .{});
                } else if (std.mem.eql(u8, arg, "off")) {
                    scfg.ignore_prompt = false;
                    try stdout_w.interface.print(":ignore-prompt = off (défaut HF : prompt ++ généré)\n", .{});
                } else {
                    try stdout_w.interface.print(":ignore-prompt : attendu 'on' ou 'off', reçu '{s}'. Inchangé : {}\n", .{ arg, scfg.ignore_prompt });
                }
            } else if (std.mem.eql(u8, cmd, ":params")) {
                try stdout_w.interface.print("params: rp={d} ignore_prompt={} T={d} top_k={d} top_p={d} min_keep={d} seed={?d} max_tokens={d}\n", .{ scfg.repetition_penalty, scfg.ignore_prompt, scfg.temperature, scfg.top_k, scfg.top_p, scfg.min_keep, scfg.seed, max_tokens });
            } else if (std.mem.eql(u8, cmd, ":help")) {
                try stdout_w.interface.print(
                    "directives : :penalty <f>  :ignore-prompt on|off  :params  :help\n" ++
                        "             (ligne vide ou EOF pour quitter ; une ligne commençant par ':' n'est jamais un prompt)\n",
                    .{},
                );
            } else {
                try stdout_w.interface.print("directive inconnue '{s}' — :help pour la liste. La ligne n'a PAS été traitée comme un prompt.\n", .{cmd});
            }
            try stdout_w.interface.flush();
            continue;
        }
        var pids = promptToIds(allocator, &encoder, line) catch |e| {
            log.err("prompt refusé (tokenisation : {s}) — prompt suivant", .{@errorName(e)});
            continue;
        };
        defer pids.deinit(allocator); // scope = l'itération : libéré à chaque tour de boucle
        generateOnce(allocator, &counter, io, platform, sharding, &exe, &eng_buf, &pk_buf, tok_sym, cache_sym, &host, &tokenizer, &stdout_w, eot_id, &policy, &scfg, vocab, max_tokens, args.dump_top5, null, null, pids.items, null, null, null) catch |e| switch (e) {
            error.SequenceTooLong, error.PromptTooLong, error.TokenOutOfRange => log.err("prompt refusé ({s}) — prompt suivant", .{@errorName(e)}),
            else => return e,
        };
    }
    log.info("REPL : sortie propre.", .{});
}

// Une génération complète — extraction de l'ex-corps de run() (spec repl-mode §1) : gardes de
// longueur, cache ZÉROS (chaque appel repart de la position 0), boucle prefill-par-decode + topK
// in-graph, verdicts (--oracle/--out-ids en one-shot ; détok stdout en libre). Types opaques
// (exe compilé, Bufferized du modèle, tokenizer) passés en anytype POINTEURS — un seul chemin de
// code pour one-shot ET résident. AUCUN état ne survit entre deux appels (R1 le vérifie).
// `policy` est passée PAR POINTEUR (spec §4.2bis) et non copiée : les trois sites d'appel
// ci-dessous (one-shot, 1er prompt du REPL, prompts suivants du REPL) doivent partager LA même
// politique. Un oubli sur l'un des trois la rendrait silencieusement inopérante dans ce mode —
// c'est exactement ce que le gate GC10 vérifie.
// kvdump (spec 2026-08-09 §4.2) : ce qu'il faut pour écrire le manifest — le chemin de sortie et
// le checkpoint à empreinter. Le chemin gencfg vient de `policy.path`, déjà passé à generateOnce.
const DumpSpec = struct { path: []const u8, ckpt_path: []const u8 };

// Shapes des 4 caches — DÉCLARÉES UNE FOIS : le dump les écrit, le restore les exige. Deux
// listes séparées auraient pu diverger en silence, et `KvDumpShapeMismatch` n'aurait plus
// discriminé qu'entre deux erreurs de frappe.
const SL_SHAPE = [_]i64{ @intCast(NUM_SLIDING_SLOTS), 1, KVH_SL, L_MAX, HD_S };
const FL_SHAPE = [_]i64{ @intCast(NUM_FULL_SLOTS), 1, KVH_FL, L_MAX, HD_F };

// === DC2 (spec kvdump §5) — ÉQUIVALENCE INTRA-PROCESS, BIT-EXACTE. Trois appels à
// `generateOnce` dans LE MÊME processus, donc le MÊME exécutable compilé/autotuné : insensible à
// la bistabilité PAR CONSTRUCTION (l'argument de S2-PONT). C'est le seul niveau où une
// équivalence bit-à-bit est légitimement exigible (inter-process : DC3, borné par la marge).
//   (1) génère K=16 tokens et DUMPE ; (2) génère K+M=48 tokens depuis zéro = RÉFÉRENCE ;
//   (3) RESTAURE le dump de (1) et génère M=32 -> doit reproduire les steps K..K+M de (2).
// Deux pré-conditions AUTO-VÉRIFIÉES, dont l'échec rend le gate INEXÉCUTABLE (jamais FAIL) :
//   A. les deux appels atteignent leur borne (`stop_reason == .max_tokens`) — un EOS précoce
//      ne prouve rien ; aucun précédent de tenue en mode libre n'est invocable (les longs runs
//      historiques étaient en mode ORACLE, où l'EOS est désactivé) ;
//   B. les K premiers ids de (1) et (2) sont identiques — c'est le canari du déterminisme
//      intra-process ET de l'alignement ids<->top5 (si l'alignement était faux, le 16/16
//      échouerait déjà ici).
fn selftestKvdumpEq(allocator: std.mem.Allocator, counter: *alloc_count.CountingAllocator, io: std.Io, platform: *zml.Platform, sharding: zml.sharding.Sharding, exe: anytype, eng_buf: anytype, pk_buf: anytype, tok_sym: zml.Tensor, cache_sym: engine.Cache, host: anytype, tokenizer: anytype, stdout_w: anytype, eot_id: u32, policy: *const gencfg.GenCfg, scfg: *sampling.SamplingCfg, vocab: i64, ids: []const u32, ckpt_path: []const u8, dump_path: []const u8) !void {
    const K: usize = 16; // tokens avant le dump
    const M: usize = 32; // tokens de continuation comparés

    var c1_ids: std.ArrayList(i64) = .empty;
    defer c1_ids.deinit(allocator);
    var c1_t5: std.ArrayList(Top5) = .empty;
    defer c1_t5.deinit(allocator);
    var s1: StopReason = .oracle;
    log.info("KVEQ: appel (1) — {d} tokens puis dump -> {s}", .{ K, dump_path });
    try generateOnce(allocator, counter, io, platform, sharding, exe, eng_buf, pk_buf, tok_sym, cache_sym, host, tokenizer, stdout_w, eot_id, policy, scfg, vocab, K, false, null, null, ids, .{ .path = dump_path, .ckpt_path = ckpt_path }, null, .{ .ids = &c1_ids, .top5 = &c1_t5, .stop = &s1 });

    var c2_ids: std.ArrayList(i64) = .empty;
    defer c2_ids.deinit(allocator);
    var c2_t5: std.ArrayList(Top5) = .empty;
    defer c2_t5.deinit(allocator);
    var s2: StopReason = .oracle;
    log.info("KVEQ: appel (2) — {d} tokens depuis zéro (référence)", .{K + M});
    try generateOnce(allocator, counter, io, platform, sharding, exe, eng_buf, pk_buf, tok_sym, cache_sym, host, tokenizer, stdout_w, eot_id, policy, scfg, vocab, K + M, false, null, null, ids, null, null, .{ .ids = &c2_ids, .top5 = &c2_t5, .stop = &s2 });

    // Pré-condition A — un arrêt prématuré ne FAIL pas : il rend l'antécédent irréalisable.
    if (s1 != .max_tokens or s2 != .max_tokens) {
        log.err("KVEQ: INEXECUTABLE — arrêt prématuré (stop1={s} stop2={s})", .{ @tagName(s1), @tagName(s2) });
        std.process.exit(3);
    }
    if (c1_ids.items.len != K or c2_ids.items.len != K + M) {
        log.err("KVEQ: INEXECUTABLE — comptes inattendus ({d} et {d})", .{ c1_ids.items.len, c2_ids.items.len });
        std.process.exit(3);
    }
    // Pré-condition B — déterminisme intra-process ET alignement ids<->top5.
    for (0..K) |j| {
        if (c1_ids.items[j] != c2_ids.items[j]) {
            log.err("KVEQ: INEXECUTABLE — déterminisme intra-process non vérifié (@gen={d} : {d} vs {d})", .{ j, c1_ids.items[j], c2_ids.items[j] });
            std.process.exit(3);
        }
    }

    // (3) restore + continuation. `ids` de cet appel = ids_fed du dump (la séquence feedée).
    var mc = try loadCacheManifest(allocator, io, dump_path, ckpt_path, policy.path);
    defer mc.deinit(allocator, io);
    const rs = try loadCacheTensors(io, &mc, host);
    var c3_ids: std.ArrayList(i64) = .empty;
    defer c3_ids.deinit(allocator);
    var c3_t5: std.ArrayList(Top5) = .empty;
    defer c3_t5.deinit(allocator);
    var s3: StopReason = .oracle;
    log.info("KVEQ: appel (3) — restore @step={d} puis {d} tokens", .{ rs.step_next, M });
    try generateOnce(allocator, counter, io, platform, sharding, exe, eng_buf, pk_buf, tok_sym, cache_sym, host, tokenizer, stdout_w, eot_id, policy, scfg, vocab, M, false, null, null, mc.ids_fed, null, rs, .{ .ids = &c3_ids, .top5 = &c3_t5, .stop = &s3 });
    if (c3_ids.items.len != M) {
        log.err("KVEQ: INEXECUTABLE — la continuation a produit {d} tokens au lieu de {d} (stop={s})", .{ c3_ids.items.len, M, @tagName(s3) });
        std.process.exit(3);
    }

    // Verdict : ids ET top-5 (indices ET BITS des valeurs) — 32/32 ou FAIL nommé.
    for (0..M) |j| {
        const ref_i = c2_ids.items[K + j];
        const got_i = c3_ids.items[j];
        const ref5 = c2_t5.items[K + j];
        const got5 = c3_t5.items[j];
        if (ref_i != got_i) {
            log.err("KVEQ: FAIL @gen={d} ids ref={d} got={d}", .{ j, ref_i, got_i });
            return error.KvEqMismatch;
        }
        for (0..gencfg.TOP_K) |r| {
            const rb: u32 = @bitCast(ref5.val[r]);
            const gb: u32 = @bitCast(got5.val[r]);
            if (ref5.idx[r] != got5.idx[r] or rb != gb) {
                log.err("KVEQ: FAIL @gen={d} rang={d} ref=({d},0x{x}) got=({d},0x{x})", .{ j, r, ref5.idx[r], rb, got5.idx[r], gb });
                return error.KvEqMismatch;
            }
        }
    }
    log.info("KVEQ: {d}/{d} bit-identiques -> PASS", .{ M, M });

    // Référence pour DC3/DC4 : ids + marge DÉCISIONNELLE de chaque step (val du rang retenu −
    // val du rang non supprimé suivant). ⚠ L'instrument logué du runner n'existe qu'en mode
    // --oracle : ici la marge est CALCULÉE depuis les top-5 capturés + la politique.
    // Formatage manuel (jamais `{any}` : ce fichier est lu par les gates Python).
    const ref_path = try std.fmt.allocPrint(allocator, "{s}.ref.json", .{dump_path});
    defer allocator.free(ref_path);
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(allocator);
    try out.appendSlice(allocator, "{\"step_next\":");
    try appendNum(allocator, &out, "{d}", .{rs.step_next});
    try out.appendSlice(allocator, ",\"k\":");
    try appendNum(allocator, &out, "{d}", .{K});
    try out.appendSlice(allocator, ",\"ids\":[");
    for (0..M) |j| {
        if (j != 0) try out.append(allocator, ',');
        try appendNum(allocator, &out, "{d}", .{c2_ids.items[K + j]});
    }
    try out.appendSlice(allocator, "],\"marges\":[");
    for (0..M) |j| {
        if (j != 0) try out.append(allocator, ',');
        const t5 = c2_t5.items[K + j];
        const sel = try policy.select(&t5.idx);
        var next_free: ?usize = null;
        for (sel.rank + 1..gencfg.TOP_K) |r| {
            if (!policy.isSuppressed(t5.idx[r])) {
                next_free = r;
                break;
            }
        }
        if (next_free) |r| {
            try appendNum(allocator, &out, "{d:.9}", .{t5.val[sel.rank] - t5.val[r]});
        } else {
            try out.appendSlice(allocator, "null");
        }
    }
    try out.appendSlice(allocator, "]}\n");
    const rf = try std.Io.Dir.createFile(.cwd(), io, ref_path, .{});
    defer rf.close(io);
    try rf.writePositionalAll(io, out.items, 0);
    log.info("KVEQ: référence écrite -> {s} ({d} ids, {d} marges)", .{ ref_path, M, M });
}

/// Petit utilitaire de formatage sans `{any}` : un fragment formaté, appendé, libéré.
fn appendNum(allocator: std.mem.Allocator, out: *std.ArrayList(u8), comptime fmt: []const u8, args: anytype) !void {
    const s = try std.fmt.allocPrint(allocator, fmt, args);
    defer allocator.free(s);
    try out.appendSlice(allocator, s);
}

// === RESTORE (spec §4.3) — EN DEUX PHASES. La compile ne doit être NI dans le chrono de C-D,
// NI avant les validations de forme :
//   phase 1 `loadCacheManifest` : AVANT la compile, ne lit que le header (quelques Ko) + le
//     petit tenseur `ids_fed` — c'est lui qui alimente les pré-checks de run() (garde oracle
//     `positions[0] == ids.len`, garde de place) ;
//   phase 2 `loadCacheTensors` : APRÈS la compile, démarre le chrono KVLOAD-PERF et lit les
//     4 caches (les GiB) directement dans host.cache_* — le `@memset(0)` de HostInputs.init est
//     ainsi remplacé de fait, et le graphe ne voit AUCUNE différence.
const Resume = struct { step_next: usize, fed_next: i64, t_load0: std.Io.Timestamp };

const ManifestCheck = struct {
    file: std.Io.File,
    header: kvdump.Header,
    path: []const u8,
    step_next: usize,
    fed_next: i64,
    ids_fed: []u32, // possédé par CE struct

    fn deinit(self: *ManifestCheck, allocator: std.mem.Allocator, io: std.Io) void {
        allocator.free(self.ids_fed);
        self.header.deinit();
        self.file.close(io);
    }
};

/// Phase 1 : header + validations de forme + `ids_fed`. AUCUN cache n'est lu ici.
/// Chaque écart est un refus BRUYANT et nommé (spec §4.5) — jamais un restore silencieusement faux.
fn loadCacheManifest(allocator: std.mem.Allocator, io: std.Io, path: []const u8, ckpt_path: []const u8, gencfg_path: []const u8) !ManifestCheck {
    var file = std.Io.Dir.cwd().openFile(io, path, .{ .mode = .read_only }) catch |e| {
        log.err("--load-cache : {s} illisible ({s})", .{ path, @errorName(e) });
        return e;
    };
    errdefer file.close(io);
    var header = kvdump.readHeader(allocator, io, file, path) catch |e| {
        log.err("--load-cache : header illisible ({s}) : {s}", .{ path, @errorName(e) });
        return e;
    };
    errdefer header.deinit();

    // (2) format
    const fmt_got = header.metaGet("format") orelse {
        log.err("--load-cache : clé `format` absente du manifest ({s})", .{path});
        return error.KvDumpBadFormat;
    };
    if (!std.mem.eql(u8, fmt_got, kvdump.FORMAT)) {
        log.err("--load-cache : format `{s}` != `{s}` attendu", .{ fmt_got, kvdump.FORMAT });
        return error.KvDumpBadFormat;
    }
    // (3) variante : le cache sliding est LINÉAIRE .k=L_MAX (R10) — un dump d'une autre borne
    // se réimplanterait à des positions FAUSSES. Refus, jamais une transposition implicite.
    const l_max_got = try kvdump.metaInt(&header, "l_max", 10);
    if (l_max_got != @as(u64, @intCast(L_MAX))) {
        log.err("--load-cache : l_max du dump = {d} != L_MAX du binaire = {d} (variante incompatible)", .{ l_max_got, L_MAX });
        return error.KvDumpVariantMismatch;
    }
    // (4) shapes/dtype des 5 tenseurs == shapes compilées
    const step_next: usize = @intCast(try kvdump.metaInt(&header, "step_next", 10));
    const fed_next_u = try kvdump.metaInt(&header, "fed_next", 10);
    const ids_shape = [_]i64{@intCast(step_next)};
    kvdump.expectShape(&header, "sl_k", "F32", &SL_SHAPE) catch |e| {
        log.err("--load-cache : shape/dtype de sl_k incompatible", .{});
        return e;
    };
    try kvdump.expectShape(&header, "sl_v", "F32", &SL_SHAPE);
    try kvdump.expectShape(&header, "fl_k", "F32", &FL_SHAPE);
    try kvdump.expectShape(&header, "fl_v", "F32", &FL_SHAPE);
    // (6) invariant d'état : la shape déclarée de ids_fed EST step_next. Un manifest forgé
    // (step_next mentí) meurt ici, avant toute lecture de données.
    kvdump.expectShape(&header, "ids_fed", "I32", &ids_shape) catch {
        const declared = kvdump.entryBytes(&header, "ids_fed") catch 0;
        log.err("--load-cache : ids_fed ({d} octets déclarés) incohérent avec step_next={d} (attendu {d} octets)", .{ declared, step_next, step_next * 4 });
        return error.KvDumpInconsistentState;
    };
    if (step_next == 0) {
        log.err("--load-cache : step_next=0 — un état sans aucun token feedé n'est pas un état de reprise", .{});
        return error.KvDumpInconsistentState;
    }
    // (5) fingerprint du checkpoint, par CONTENU (taille + xxh64 du header) — jamais par chemin :
    // un même checkpoint atteint par un autre symlink reste valide, un autre checkpoint de même
    // taille est arrêté par le hash du header (noms/shapes/offsets de tous ses tenseurs).
    const fp = try kvdump.ckptFingerprint(allocator, io, ckpt_path);
    const want_bytes = try kvdump.metaInt(&header, "ckpt_bytes", 10);
    const want_hdr = try kvdump.metaInt(&header, "ckpt_hdr_xxh64", 16);
    if (fp.bytes != want_bytes or fp.hdr_xxh64 != want_hdr) {
        log.err("--load-cache : checkpoint DIFFÉRENT de celui du dump (bytes {d} vs {d}, hdr_xxh64 {x} vs {x})", .{ fp.bytes, want_bytes, fp.hdr_xxh64, want_hdr });
        return error.KvDumpCheckpointMismatch;
    }
    // (8) gencfg : la politique est re-dérivée des fichiers COURANTS — c'est voulu, mais l'écart
    // doit être VISIBLE. WARN, pas un refus (spec §4.5, dernière ligne).
    if (header.metaGet("gencfg_path")) |gp| {
        if (!std.mem.eql(u8, gp, gencfg_path)) {
            log.warn("--load-cache : gencfg du dump `{s}` != courant `{s}` — la politique appliquée est la COURANTE", .{ gp, gencfg_path });
        }
    }
    if (header.metaGet("sampling")) |sp| {
        if (!std.mem.eql(u8, sp, "off")) log.warn("--load-cache : le dump portait sampling={s} — un restore avec d'autres warpers diverge légitimement", .{sp});
    }

    // `ids_fed` : petit (step_next x 4 octets), lu ICI parce que les pré-checks de run() en
    // dépendent (la garde oracle compare positions[0] à ids.len). Checksum vérifié.
    const raw = try allocator.alloc(i32, step_next);
    defer allocator.free(raw);
    try kvdump.readTensorInto(io, file, path, &header, "ids_fed", std.mem.sliceAsBytes(raw), try kvdump.metaInt(&header, "ids_fed_xxh64", 16));
    const ids_fed = try allocator.alloc(u32, step_next);
    errdefer allocator.free(ids_fed);
    for (raw, 0..) |t, k| {
        if (t < 0 or t >= VOCAB_CONTRACT) {
            log.err("--load-cache : ids_fed[{d}] = {d} hors vocab [0,{d})", .{ k, t, VOCAB_CONTRACT });
            return error.TokenOutOfRange;
        }
        ids_fed[k] = @intCast(t);
    }
    if (fed_next_u >= VOCAB_CONTRACT) {
        log.err("--load-cache : fed_next = {d} hors vocab [0,{d})", .{ fed_next_u, VOCAB_CONTRACT });
        return error.TokenOutOfRange;
    }
    return .{
        .file = file,
        .header = header,
        .path = path,
        .step_next = step_next,
        .fed_next = @intCast(fed_next_u),
        .ids_fed = ids_fed,
    };
}

/// Phase 2 : les GiB. Le chrono de C-D démarre à la PREMIÈRE ligne — la lecture des 2,6 GiB est
/// DANS la fenêtre (un chrono qui l'exclurait serait biaisé vers le PASS) ; la compile est
/// dehors, des deux côtés de la comparaison.
fn loadCacheTensors(io: std.Io, mc: *const ManifestCheck, host: anytype) !Resume {
    const t_load0: std.Io.Timestamp = .now(io, .awake);
    try kvdump.readTensorInto(io, mc.file, mc.path, &mc.header, "sl_k", host.cache_sl_k, try kvdump.metaInt(&mc.header, "sl_k_xxh64", 16));
    try kvdump.readTensorInto(io, mc.file, mc.path, &mc.header, "sl_v", host.cache_sl_v, try kvdump.metaInt(&mc.header, "sl_v_xxh64", 16));
    try kvdump.readTensorInto(io, mc.file, mc.path, &mc.header, "fl_k", host.cache_fl_k, try kvdump.metaInt(&mc.header, "fl_k_xxh64", 16));
    try kvdump.readTensorInto(io, mc.file, mc.path, &mc.header, "fl_v", host.cache_fl_v, try kvdump.metaInt(&mc.header, "fl_v_xxh64", 16));
    log.info("KVLOAD: {s} l_max={d} step_next={d} fed_next={d} ids={d} (reprise sans prefill)", .{ mc.path, L_MAX, mc.step_next, mc.fed_next, mc.ids_fed.len });
    log.info("KVLOAD: contexte de {d} tokens (non réaffiché)", .{mc.ids_fed.len});
    return .{ .step_next = mc.step_next, .fed_next = mc.fed_next, .t_load0 = t_load0 };
}

/// Écrit l'état E1-E4 (spec §4.1) : 4 caches f32 + ids_fed i32 + manifest auto-décrivant.
/// Le fingerprint du checkpoint est par CONTENU (taille + xxh64 du header), jamais par chemin.
fn dumpCacheFile(allocator: std.mem.Allocator, io: std.Io, ds: DumpSpec, host: anytype, ids_fed: []const i32, step_next: usize, fed_next: i64, stop_reason: anytype, policy: *const gencfg.GenCfg, scfg: *const sampling.SamplingCfg) !void {
    const ids_shape = [_]i64{@intCast(ids_fed.len)};
    const ids_bytes = std.mem.sliceAsBytes(ids_fed);
    const fp = try kvdump.ckptFingerprint(allocator, io, ds.ckpt_path);
    var buf: [12][96]u8 = undefined;
    // `sampling` est une trace INFORMATIVE : un restore avec d'autres warpers diverge
    // légitimement, mais l'écart doit être VISIBLE (spec §4.1). Le dump avec sampling ARMÉ +
    // seed est refusé en amont (DumpWithSamplingArmed) — ici on trace ce qui était demandé.
    // ⚠ `rp` est DANS cette trace : le code exige lui-même qu'un écart de warpers au restore soit
    // VISIBLE (spec kvdump §4.1), et la penalty change les ids produits autant qu'un top_p. Sans
    // cette extension, elle serait le SEUL réglage de la chaîne invisible au manifest.
    const sampling_str = if (scfg.pathArmed())
        try std.fmt.bufPrint(&buf[10], "T={d},top_k={d},top_p={d},rp={d},ignore_prompt={}", .{ scfg.temperature, scfg.top_k, scfg.top_p, scfg.repetition_penalty, scfg.ignore_prompt })
    else
        "off";
    const meta = [_]kvdump.MetaKV{
        .{ .k = "format", .v = kvdump.FORMAT },
        .{ .k = "l_max", .v = try std.fmt.bufPrint(&buf[0], "{d}", .{L_MAX}) },
        .{ .k = "step_next", .v = try std.fmt.bufPrint(&buf[1], "{d}", .{step_next}) },
        .{ .k = "fed_next", .v = try std.fmt.bufPrint(&buf[2], "{d}", .{fed_next}) },
        .{ .k = "stop_reason", .v = @tagName(stop_reason) },
        .{ .k = "ckpt_bytes", .v = try std.fmt.bufPrint(&buf[3], "{d}", .{fp.bytes}) },
        .{ .k = "ckpt_hdr_xxh64", .v = try std.fmt.bufPrint(&buf[4], "{x}", .{fp.hdr_xxh64}) },
        .{ .k = "gencfg_path", .v = policy.path },
        .{ .k = "build_mode", .v = @tagName(builtin.mode) },
        .{ .k = "sampling", .v = sampling_str },
        .{ .k = "sl_k_xxh64", .v = try std.fmt.bufPrint(&buf[5], "{x}", .{kvdump.xxh64(host.cache_sl_k)}) },
        .{ .k = "sl_v_xxh64", .v = try std.fmt.bufPrint(&buf[6], "{x}", .{kvdump.xxh64(host.cache_sl_v)}) },
        .{ .k = "fl_k_xxh64", .v = try std.fmt.bufPrint(&buf[7], "{x}", .{kvdump.xxh64(host.cache_fl_k)}) },
        .{ .k = "fl_v_xxh64", .v = try std.fmt.bufPrint(&buf[8], "{x}", .{kvdump.xxh64(host.cache_fl_v)}) },
        .{ .k = "ids_fed_xxh64", .v = try std.fmt.bufPrint(&buf[9], "{x}", .{kvdump.xxh64(ids_bytes)}) },
    };
    const tensors = [_]kvdump.TensorOut{
        .{ .name = "sl_k", .dtype = "F32", .shape = &SL_SHAPE, .bytes = host.cache_sl_k },
        .{ .name = "sl_v", .dtype = "F32", .shape = &SL_SHAPE, .bytes = host.cache_sl_v },
        .{ .name = "fl_k", .dtype = "F32", .shape = &FL_SHAPE, .bytes = host.cache_fl_k },
        .{ .name = "fl_v", .dtype = "F32", .shape = &FL_SHAPE, .bytes = host.cache_fl_v },
        .{ .name = "ids_fed", .dtype = "I32", .shape = &ids_shape, .bytes = ids_bytes },
    };
    try kvdump.write(allocator, io, ds.path, &tensors, &meta);
    const total = 2 * host.cache_sl_k.len + 2 * host.cache_fl_k.len + ids_bytes.len;
    log.info("KVDUMP: {s} l_max={d} step_next={d} fed_next={d} ids={d} octets={d} xxh64_ok", .{ ds.path, L_MAX, step_next, fed_next, ids_fed.len, total });
}

fn generateOnce(allocator: std.mem.Allocator, counter: *alloc_count.CountingAllocator, io: std.Io, platform: *zml.Platform, sharding: zml.sharding.Sharding, exe: anytype, eng_buf: anytype, pk_buf: anytype, tok_sym: zml.Tensor, cache_sym: engine.Cache, host: anytype, tokenizer: anytype, stdout_w: anytype, eot_id: u32, policy: *const gencfg.GenCfg, scfg: *sampling.SamplingCfg, vocab: i64, max_tokens: usize, dump_top5: bool, oracle_ids: ?[]const i32, out_ids_path: ?[]const u8, ids: []const u32, dump_spec: ?DumpSpec, resume_state: ?Resume, capture: ?Capture) !void {
    const limit: usize = if (oracle_ids) |fx| fx.len else max_tokens;
    if (ids.len == 0) {
        log.err("prompt vide (0 ids)", .{});
        return error.PromptTooLong;
    }
    // Gardes de lancement (déplacées de run — refaites PAR PROMPT en résident ; run les
    // pré-vérifie encore en one-shot pour le fail-fast avant compile).
    if (ids.len + limit > @as(usize, @intCast(L_MAX))) {
        log.err("garde-fou : ids.len({d}) + limit({d}) > L_MAX({d})", .{ ids.len, limit, L_MAX });
        return error.SequenceTooLong;
    }
    // kvdump (spec §4.3) : en mode resume cette garde est REMPLACÉE par `step_next + limit <=
    // L_MAX` (assurée par la garde de place ci-dessus, ids.len == step_next). Un état repris
    // dépasse légitimement la fenêtre glissante : elle protégeait le prefill, qui n'a pas lieu.
    if (resume_state == null and ids.len >= @as(usize, @intCast(SLIDING_WINDOW))) {
        log.err("garde-fou : ids.len({d}) >= SLIDING_WINDOW({d})", .{ ids.len, SLIDING_WINDOW });
        return error.PromptTooLong;
    }

    // === Phase 1 (penalty) : SEED de l'historique, ici et pas ailleurs ===
    // Contrat que ce seed établit : au moment de sélectionner le token de génération k,
    // `hist[0..hist_len]` vaut prompt ++ tokens générés avant k — exactement l'`input_ids` que
    // HF passe à son processor au même point.
    //
    // ⚠ POURQUOI EN TÊTE DE CETTE FONCTION, et pas au site d'allocation. Le site d'allocation ne
    // s'exécute qu'UNE FOIS par process, alors que `generateOnce` a SIX sites d'appel : one-shot,
    // les deux entrées de la boucle `--repl` (un appel PAR prompt), et les trois de kvdump-eq.
    // Seul un seed en tête de fonction les couvre structurellement — c'est aussi ce qui donne
    // gratuitement le RE-SEED par prompt du repl (RP5) et la reprise `--load-cache`, où `ids`
    // vaut `ids_fed` complet et où la boucle entre DIRECTEMENT en phase de génération : les
    // tokens repris n'y sont jamais re-feedés, un historique reparti de zéro ignorerait tout le
    // contexte de la reprise.
    if (scfg.hist.len > 0) {
        if (ids.len > scfg.hist.len) {
            log.err("penalty : prompt {d} ids > capacité de l'historique {d} (L_MAX)", .{ ids.len, scfg.hist.len });
            return error.SequenceTooLong;
        }
        @memcpy(scfg.hist[0..ids.len], ids);
        scfg.hist_len = ids.len;
        scfg.prompt_len = ids.len;
        // Compteurs de non-vacuité remis à zéro PAR PROMPT : en `--repl`, un `n_penalty_touched`
        // cumulé depuis le prompt précédent ferait passer l'exigence de la passe courante sans
        // qu'elle ait rien touché.
        scfg.n_penalty_touched = 0;
        scfg.n_penalty_empty_hist = 0;
    }
    // Cache ZÉROS par génération (les slices host vivent dans run pour toute la session).
    var cache_buf = zml.Bufferized(engine.Cache){
        .sl_k = try zml.Buffer.fromBytes(io, platform, cache_sym.sl_k.shape(), sharding, host.cache_sl_k),
        .sl_v = try zml.Buffer.fromBytes(io, platform, cache_sym.sl_v.shape(), sharding, host.cache_sl_v),
        .fl_k = try zml.Buffer.fromBytes(io, platform, cache_sym.fl_k.shape(), sharding, host.cache_fl_k),
        .fl_v = try zml.Buffer.fromBytes(io, platform, cache_sym.fl_v.shape(), sharding, host.cache_fl_v),
    };

    // === Step 5.2 → L3 : boucle prefill-par-decode + topK in-graph + arrêt ===
    var generated: std.ArrayList(i64) = .empty;
    defer generated.deinit(allocator);
    var gen_top5: std.ArrayList(Top5) = .empty; // parallèle à `generated` (diagnostic FAIL, Step 5.3)
    defer gen_top5.deinit(allocator);
    // D10 (C5) : capacité réservée UNE FOIS — borne = limit, qui couvre --oracle (fx.len) ET le
    // mode libre (max_tokens). Une borne ids.len+max_tokens aurait débordé sur les fixtures
    // oracle historiques (1150 ids > max_tokens 200) — passe 1 de revue.
    // ⚠ ÉCART ASSUMÉ à la spec §3 C5 (« ids.len + limit ») : les listes ne reçoivent QUE des
    // tokens de la phase gen — borne resserrée à limit (+1 pour gen_top5, qui reçoit s0 avant
    // generated), déclaré ici et dans le message de commit.
    try generated.ensureTotalCapacity(allocator, limit);
    try gen_top5.ensureTotalCapacity(allocator, limit + 1);

    log.info("Boucle autonome : {d} steps de prefill, puis génération (limite {d}{s})", .{ ids.len, limit, if (oracle_ids != null) " = fed.len, oracle" else " = max_tokens" });

    // Raison d'arrêt (A3) : capturée DANS la boucle (pas reconstruite après coup) — le strip EOT
    // du détok et le verdict A3 en dépendent. `.oracle` = sortie par compte fed.len (mode --oracle).
    // (Le type StopReason est déclaré au niveau du fichier : le manifest kvdump le publie.)
    var stop_reason: StopReason = .oracle;
    // Id de l'EOS qui a réellement arrêté la génération. Avec TROIS EOS possibles, « arrêt : EOT »
    // ne dit plus lequel : GC6 exige que le log NOMME l'id.
    var stop_eos_id: i64 = -1;
    // Nombre de steps DE GÉNÉRATION où la suppression a changé le token choisi. Déclaré en tête de
    // `generateOnce` : c'est ce qui réalise la remise à zéro par prompt en `--repl` (R12) et le
    // rend vérifiable par simple lecture. C'est le détecteur de vacuité intégré du chantier — un
    // gate de mordant qui rapporte 0 n'a rien prouvé.
    var n_suppress_hits: usize = 0;

    // D10 (C8/AL-RSS) : VmRSS échantillonné aux tokens GÉNÉRÉS 20 et 200 (prefill exclu).
    var rss_t20: ?u64 = null;
    var rss_t200: ?u64 = null;

    // D10 (C4) : args/results créés UNE FOIS — set()/call()/get() n'allouent rien (exe.zig:129-146,
    // spec F5) ; les init faisaient 6 allocs/step dont le dupe de 1006 Shapes (~293 KiB).
    // ⚠ AVANT les snapshots ALLOC-LOOP : les init sont HORS fenêtre (spec §3/C6).
    var call_args = try exe.args(allocator);
    defer call_args.deinit(allocator);
    var call_results = try exe.results(allocator);
    defer call_results.deinit(allocator);

    // === D10 : fenêtre ALLOC-LOOP — deltas du compteur autour de la boucle de steps SEULE
    // (les init hissés et les buffers pré-alloués sont hors fenêtre, spec §3 Publication). ===
    const al0_alloc = counter.n_alloc;
    const al0_resize = counter.n_resize;
    const al0_remap = counter.n_remap;
    const al0_free = counter.n_free;
    const al0_bytes = counter.bytes_alloc;

    var fed: i64 = @intCast(ids[0]);
    var step: usize = 0;
    // kvdump : reprise — `step` et `fed` viennent du dump. En reprise SIMPLE, la boucle entre
    // DIRECTEMENT en phase de génération (`in_gen_phase = step + 1 >= ids.len` est vrai par
    // l'invariant ids_fed.len == step_next). Rien d'autre ne change : mêmes call_args, même
    // exécutable, mêmes compteurs (D10).
    //
    // K5 : sous reprise AVEC prompt neuf, `ids` vaut ids_full (contexte ++ fed_next ++ clôture ++
    // tour 2), donc ids.len > step_next et `in_gen_phase` est FAUX au premier step — la boucle
    // absorbe le tour 2 en prefill-par-decode, par construction et sans branche dédiée. C'est tout
    // le chantier : le graphe ne distingue pas prefill et génération (position ≡ ctrl.step).
    //
    // ⚠ Une assertion `ids[step_next] == fed_next` a été envisagée ici puis RETIRÉE en revue de
    // spec : les deux membres dérivent du même champ du manifest, elle ne peut pas échouer
    // (feedback_controle_qui_ne_peut_pas_reussir). La limite réelle — un `fed_next` FORGÉ fabrique
    // un contexte que le host ne peut pas contredire (le cache lui est opaque) — est documentée
    // spec §4.6 et n'est visible QUE de l'aller-retour teacher-forcé : c'est ce que PF2 démontre.
    if (resume_state) |rs| {
        step = rs.step_next;
        fed = rs.fed_next;
    }
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

        call_args.set(.{ eng_buf.*, tok_buf, pk_buf.*, cache_buf, ctrl_buf });
        exe.call(call_args, &call_results);
        // r_logits : lue par le chemin B (toSlice → work) quand armé ; deinit par step.
        var r_t5v, var r_t5i, var r_logits, const r_slk, const r_slv, const r_flk, const r_flv = call_results.get(struct {
            zml.Buffer, zml.Buffer, zml.Buffer, zml.Buffer, zml.Buffer, zml.Buffer, zml.Buffer,
        });

        // D10 (DA-3) : shards de r_logits captés au 1er step (le Buffer est deinit par step).
        if (step == 0) {
            var sh_it = r_logits.shards();
            scfg.n_shards = @intCast(sh_it.remaining());
        }

        // top5 : TOUJOURS calculé in-graph (cheap, cf PLAN), ignoré tant qu'on est en prefill (sauf
        // le dernier prefill step, qui produit s0 — cf `in_gen_phase` ci-dessous).
        const in_gen_phase = step + 1 >= ids.len;

        // Top5 depuis le device (~40 octets D2H, PILE — D10/C3 : getValue délègue à toSlice,
        // zéro allocation). Gardes dtype AVANT le D2H, en acceptation ; f32 est désormais
        // vérifié aussi (il était supposé). dtype i32 : `topK` délègue à `sort` (tensor.zig:3096),
        // indices produits par `Tensor.arange(…, .i32)` (tensor.zig:2977) — vérifié À CHAQUE step
        // (coût nul : compare d'enum) plutôt que supposé silencieusement.
        if (r_t5v.shape().dtype() != .f32) {
            log.err("t5.values : dtype={s} ≠ f32 attendu", .{@tagName(r_t5v.shape().dtype())});
            return error.UnexpectedDtype;
        }
        if (r_t5i.shape().dtype() != .i32) {
            log.err("t5.indices : dtype={s} ≠ i32 attendu (topK/sort, tensor.zig:2977)", .{@tagName(r_t5i.shape().dtype())});
            return error.UnexpectedDtype;
        }
        const t5v = try r_t5v.getValue([gencfg.TOP_K]f32, io);
        const t5i = try r_t5i.getValue([gencfg.TOP_K]i32, io);
        var top5: Top5 = undefined;
        for (0..gencfg.TOP_K) |j| {
            top5.idx[j] = @intCast(t5i[j]);
            top5.val[j] = t5v[j];
        }
        // === Politique de décodage (spec §4.2) — REMPLACE l'ancien `top5.idx[0]` nu. La
        // sélection est INCONDITIONNELLE : la garder sous `in_gen_phase` serait un vrai bug, car
        // s0 (produit au dernier step de prefill) ne serait plus filtré. Seul le COMPTEUR est
        // gardé, parce qu'en prefill le token est jeté.
        const sel = policy.select(&top5.idx) catch |e| {
            log.err("step {d} : {s} — les {d} ids du top-5 sont tous supprimés (top5={any})", .{ step, @errorName(e), gencfg.TOP_K, top5.idx });
            return e;
        };
        var tok: i64 = @intCast(sel.tok);
        if (in_gen_phase and sel.rank != 0) n_suppress_hits += 1;

        // === CHEMIN B (spec phase 2 §4.2) — armé seulement si un warper ou un tirage l'exige.
        // Si rien n'est armé, le chemin A ci-dessus reste STRICTEMENT le code d'avant : mêmes
        // ~48 octets de D2H, aucune lecture du vecteur complet. ===
        if (scfg.pathArmed()) {
            const t_b0: std.Io.Timestamp = .now(io, .awake); // M-COUT : début du bloc mesuré
            // D10 (C2) : D2H DIRECT dans work (persistant) — 0 allocation, 0 copie. toSliceAlloc
            // faisait 2 allocs d'1 MiB + 1 memcpy interne + un @memcpy runner : tout disparaît.
            // Garde en ACCEPTATION avant l'assert de Slice.init (erreur propre, pas un panic).
            const need = r_logits.shape().byteSize();
            if (need != scfg.work.len * @sizeOf(f32)) {
                log.err("chemin B : logits {d} octets != work {d} octets", .{ need, scfg.work.len * @sizeOf(f32) });
                return error.UnexpectedShape;
            }
            try r_logits.toSlice(io, zml.Slice.init(r_logits.shape(), std.mem.sliceAsBytes(scfg.work)));
            const t_d2h: u64 = @intCast(t_b0.untilNow(io, .awake).toNanoseconds()); // D2H seul
            const t_w0: std.Io.Timestamp = .now(io, .awake);

            // Ordre de HF, mesuré (F8) : Penalty(4) → Suppress(15) → Temperature(17) →
            // TopK(19) → TopP(20). La penalty (PHASE 1) est donc EN TÊTE, avant la suppression.
            if (scfg.repetition_penalty != 1.0) {
                // Sous --ignore-prompt, `h` est VIDE pendant tout le prefill (hist_len ==
                // prompt_len) : la garde `h.len == 0` est OBLIGATOIRE — en ReleaseFast, `h[0]`
                // sur une slice vide est une UB SILENCIEUSE, pas un panic. (`@min` borne le bas
                // par défense, au cas où prompt_len dépasserait hist_len.)
                const lo = if (scfg.ignore_prompt) @min(scfg.prompt_len, scfg.hist_len) else 0;
                const h = scfg.hist[lo..scfg.hist_len];
                if (h.len > 0) {
                    @memset(scfg.seen, 0); // dans le `if` : 32 Kio de memset inutiles quand h est vide
                    if (sampling.applyRepetitionPenalty(scfg.work, h, scfg.repetition_penalty, scfg.seen))
                        scfg.n_penalty_touched += 1;
                } else {
                    scfg.n_penalty_empty_hist += 1; // cf exemption de PenaltyInert, fin de run
                }
            }
            sampling.applySuppression(scfg.work, policy);
            // Gate D1/D2 : `pre_temp` est l'entrée COMMUNE de la chaîne nominale et de la chaîne
            // mutée — les deux doivent partir du même vecteur, sinon le mutant ne compare rien.
            if (scfg.gate) |g| @memcpy(g.pre_temp, scfg.work);
            if (scfg.temperature != 1.0) {
                sampling.applyTemperature(scfg.work, scfg.temperature);
                if (scfg.gate) |g| {
                    g.n_temp_applied += 1; // G-D2 (i) : la ligne s'exécute enfin sur GPU
                    g.n_temp_mul_diffs += sampling_ref.tempDivVsMulDiffs(g.pre_temp, scfg.work, scfg.temperature);
                }
            }
            sampling.applyTopK(scfg.work, scfg.top_k, scfg.min_keep, &scfg.scratch);
            // G-D1 : la référence est calculée sur l'entrée EXACTE de `applyTopP`, AVANT que
            // celle-ci ne mute `work` en place. `refTopPKeep` ne mute pas son entrée.
            if (scfg.gate) |g| {
                const v = sampling_ref.refTopPKeep(scfg.work, scfg.top_p, scfg.min_keep, g);
                g.n_cut_total += v.n_cut;
                if (v.n_cut > 0) g.n_steps_with_cut += 1; // ANTÉCÉDENT du gate
                g.n_boundary_tight += v.n_boundary_tight;
                g.n_boundary_ties += v.n_boundary_ties;
            }
            sampling.applyTopP(scfg.work, scfg.top_p, scfg.min_keep, &scfg.scratch);
            if (scfg.gate) |g| {
                g.n_steps += 1;
                var first_bad: i64 = -1;
                const bad = sampling_ref.compareKeep(scfg.work, g, &first_bad);
                if (bad > 0) {
                    g.n_topp_disagree += bad;
                    if (g.first_bad_id < 0) g.first_bad_id = first_bad;
                    log.err("G-D1 désaccord @step {d} : {d} id(s) ; 1er id={d} — impl {s}, réf {s}", .{
                        step, bad, first_bad,
                        if (g.keep_impl[@intCast(first_bad)]) "GARDE" else "retire",
                        if (g.keep[@intCast(first_bad)]) "GARDE" else "retire",
                    });
                }
                // Mutant (b) — l'ORDRE de la chaîne HF. Réutilise `scfg.scratch` séquentiellement
                // (la chaîne nominale en a fini) : aucune allocation.
                const od = sampling_ref.orderMutantDiffs(g.pre_temp, scfg.work, scfg.temperature, scfg.top_k, scfg.top_p, scfg.min_keep, g, &scfg.scratch);
                g.n_order_diffs += od;
                if (od > 0) g.n_steps_with_order_diff += 1;
            }

            const tok_b: u32 = if (scfg.drawArmed())
                sampling.sample(scfg.work, scfg.prng.random())
            else
                sampling.argmax(scfg.work);

            // Invariant le plus discriminant du sampler : on ne tire JAMAIS un token filtré.
            if (scfg.work[tok_b] == sampling.FILTER) {
                log.err("chemin B : token {d} retenu alors qu'il est FILTRÉ (step {d})", .{ tok_b, step });
                return error.SampledFilteredToken;
            }

            // === S2-PONT — les DEUX sélecteurs sur le MÊME vecteur, au MÊME step, dans le MÊME
            // processus. Insensible au non-déterminisme PAR CONSTRUCTION : il n'y a ni second
            // forward, ni seconde compile, ni témoin stocké. ===
            scfg.n_steps_compared += 1;
            if (tok_b != sel.tok) {
                // Une égalité exacte au sommet est un VERDICT DISTINCT, pas un FAIL : les deux
                // tie-breaks (argmax host « premier indice » vs ordre du topK in-graph) ne sont
                // pas prouvés équivalents (gencfg.zig:21-25, dette D8).
                if (scfg.work[tok_b] == scfg.work[sel.tok]) {
                    scfg.n_exact_top_ties += 1;
                } else {
                    scfg.n_disagree += 1;
                    // ⚠ Sous penalty ARMÉE, le désaccord est ATTENDU et non un défaut : le chemin A
                    // est un topK in-graph sur les logits NUS, le chemin B décide après penalty.
                    // Les deux DOIVENT diverger dès que la penalty mord — c'est même la preuve
                    // qu'elle mord. Le compteur reste publié (ligne S2-PONT), mais l'`err` par
                    // step est rétrogradé : sinon un run sain crache 40+ lignes d'erreur et le
                    // lecteur apprend à les ignorer, y compris le jour où elles disent vrai.
                    if (scfg.repetition_penalty == 1.0) {
                        log.err("S2-PONT désaccord @step {d} : A={d} (val {d:.6}) B={d} (val {d:.6})", .{ step, sel.tok, scfg.work[sel.tok], tok_b, scfg.work[tok_b] });
                    }
                }
            }
            tok = @intCast(tok_b);

            // M-COUT — le bloc entier {D2H + warpers + sélection}, seul endroit où le surcoût
            // vit réellement. Mesurer le tok/s global le noierait dans le bruit inter-compiles.
            scfg.d2h_ns_total += t_d2h;
            scfg.warp_ns_total += @intCast(t_w0.untilNow(io, .awake).toNanoseconds());
            const dt: u64 = @intCast(t_b0.untilNow(io, .awake).toNanoseconds());
            scfg.cout_ns_total += dt;
            if (dt > scfg.cout_ns_max) scfg.cout_ns_max = dt;
            scfg.n_cout_samples += 1;
        }
        if (in_gen_phase) try gen_top5.appendBounded(top5); // D10 (C5) : garde ACTIVE tous modes (appendAssumeCapacity = UB en ReleaseFast)
        // W4g (protocole de flip) : marge top1−top2 par step de génération, mode oracle seulement.
        // ⚠ La marge BRUTE ne suffit plus dès que la suppression mord : elle parle de deux tokens
        // dont le premier n'est pas celui qu'on a retenu. Cette marge est l'instrument de
        // requalification pré-enregistré des gates U8/W4g — on publie donc AUSSI la marge
        // DÉCISIONNELLE (entre le rang retenu et le rang non supprimé suivant), seule grandeur
        // qui parle du choix réellement fait. C'est également l'unique instrument de
        // l'histogramme `rank_used` exigé par la claim C2 : sans lui, C2 n'a pas de mesure.
        if (in_gen_phase and oracle_ids != null) {
            var next_free: ?usize = null;
            for (sel.rank + 1..gencfg.TOP_K) |j| {
                if (!policy.isSuppressed(top5.idx[j])) {
                    next_free = j;
                    break;
                }
            }
            const marge_dec: f32 = if (next_free) |j| top5.val[sel.rank] - top5.val[j] else std.math.nan(f32);
            log.info("  marge top1-top2 @ gen={d} : {d:.6} (top1={d} top2={d}) rank_used={d} chosen={d} marge_decisionnelle={d:.6}", .{ gen_top5.items.len - 1, top5.val[0] - top5.val[1], top5.idx[0], top5.idx[1], sel.rank, sel.tok, marge_dec });
        }
        // --dump-top5 (U9) : top-5 par step aussi en mode LIBRE (w4auto ne le loggait qu'en --oracle).
        if (in_gen_phase and dump_top5) log.info("  top5 @ gen={d} : idx={any} val={any} rank_used={d} chosen={d}", .{ gen_top5.items.len - 1, top5.idx, top5.val, sel.rank, sel.tok });
        // K5/PF1 : les positions du PREFILL DE REPRISE sont teacher-forcées par construction (le
        // token feedé est imposé par ids_full, aucun effet boule de neige) — leur top-5 est LA
        // sortie que l'oracle compare, et il est BRUT (la politique s'applique après, host-side).
        // Émis SEULEMENT sous reprise : le prefill d'un tour 1 n'intéresse aucun gate, et mille
        // lignes pollueraient les logs des runs longs.
        if (!in_gen_phase and dump_top5 and resume_state != null) log.info("  top5 @ ctx={d} : idx={any} val={any}", .{ step, top5.idx, top5.val });

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
        // D10 (C4) : call_args/call_results hissés — deinit UNE FOIS par les defer d'entête.

        // Progression périodique (motif gemma4_gen_long_gpu.zig:158) — premier signe humain d'une
        // anomalie pendant un run long (A2) : silence prolongé = suspect.
        if ((step + 1) % 256 == 0) log.info("  ... step {d} ({d} générés)", .{ step + 1, generated.items.len });

        // Step 2.7 : fin du DERNIER step de prefill (nuance de mesure, cf log L3 PERF plus bas).
        if (step + 1 == ids.len) t_prefill_end = .now(io, .awake);

        if (step + 1 < ids.len) {
            // Phase 1 (prefill) : argmax ci-dessus IGNORÉ (pas le dernier token du prompt).
            fed = @intCast(ids[step + 1]);
            continue;
        }
        // Phase 2 (génération, s0 INCLUS dès le 1er passage ici — dernier step de prefill).
        try generated.appendBounded(tok); // D10 (C5) : idem — error.OutOfMemory si la borne était fausse, jamais une UB
        // Phase 1 (penalty) : l'historique suit le token ICI, où il est ACTÉ — et surtout PAS en
        // fin d'itération (`fed = tok`), qui vient APRÈS les trois `break` (borne oracle, EOT,
        // max_tokens) et perdrait donc le DERNIER token généré. Écriture directe, zéro allocation.
        // ⚠ Ne PAS appender `fed` en tête d'itération : avec le seed, le prompt y serait compté
        // deux fois. (Le piège off-by-one de la spec rév. 4-1 visait le câblage SANS seed —
        // avec seed, c'est l'append de `fed` qui devient le bug.)
        if (scfg.hist.len > 0) {
            if (scfg.hist_len >= scfg.hist.len) {
                log.err("penalty : historique plein ({d} ids, capacité {d}) au step {d}", .{ scfg.hist_len, scfg.hist.len, step });
                return error.SequenceTooLong;
            }
            scfg.hist[scfg.hist_len] = @intCast(tok);
            scfg.hist_len += 1;
        }
        // kvdump / C-D : l'instrument du gain. Fenêtre = du DÉBUT de la lecture des tenseurs
        // (phase 2, post-compile) au PREMIER token produit — les GiB relus sont DEDANS.
        if (resume_state) |rs| {
            if (generated.items.len == 1) {
                const dt_s = @as(f64, @floatFromInt(rs.t_load0.untilNow(io, .awake).toNanoseconds())) / std.time.ns_per_s;
                log.info("KVLOAD-PERF: chargement+h2d+reprise -> 1er token en {d:.3}s", .{dt_s});
            }
        }
        // D10 (C8) : sonde VmRSS — tokens générés 20 et 200, buffers de pile (zéro alloc Zig).
        if (generated.items.len == 20) rss_t20 = mem_probe.rssKb(io);
        if (generated.items.len == 200) rss_t200 = mem_probe.rssKb(io);
        if (oracle_ids) |fx| {
            if (generated.items.len >= fx.len) break; // stop_reason reste .oracle
        } else {
            // Arrêt multi-EOS (spec §4.3) : sémantique HF « any of », sans priorité, le token
            // étant concaténé PUIS testé (il fait donc partie de la sortie — c'est déjà ce que
            // fait le runner, seul l'élargissement 1 → 3 était à faire).
            // ⚠ Hors mode `--oracle` uniquement (décision Régis n°3) : là-bas c'est la fixture qui
            // borne le nombre de positions comparées, un arrêt n'y a pas de sens — tandis que la
            // SUPPRESSION, elle, doit être exercée des deux côtés, sans quoi l'angle mort du
            // finding §5 persisterait.
            if (policy.isEos(tok)) {
                stop_reason = .eot;
                stop_eos_id = tok;
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

    // D10 : deltas ALLOC-LOOP FIGÉS ICI, avant tout travail post-boucle (le dump alloue via
    // l'allocateur compté ; sans ce gel, la ligne ALLOC-LOOP l'imputerait à la boucle et
    // DC6 échouerait par construction — finding bloquant de revue kvdump).
    const al_alloc = counter.n_alloc - al0_alloc;
    const al_resize = counter.n_resize - al0_resize;
    const al_remap = counter.n_remap - al0_remap;
    const al_free = counter.n_free - al0_free;
    const al_bytes = counter.bytes_alloc - al0_bytes;

    // === kvdump (spec §4.2) : POINT DE DUMP UNIQUE — après la boucle et le gel des deltas,
    // AVANT les deinit du cache. Les 4 buffers lus sont les SORTIES du dernier step (le swap
    // précède tous les `break`) : jamais un buffer DONNÉ, le contrat de donation tient. ===
    if (dump_spec) |ds| {
        // d2h des 4 buffers FINAUX vers les slices host EXISTANTES (zéro alloc de 2,6 GiB,
        // plafond B10 intact — spec §4.2). toSlice : le mécanisme D2H prouvé (D10/C2).
        try cache_buf.sl_k.toSlice(io, zml.Slice.init(cache_buf.sl_k.shape(), host.cache_sl_k));
        try cache_buf.sl_v.toSlice(io, zml.Slice.init(cache_buf.sl_v.shape(), host.cache_sl_v));
        try cache_buf.fl_k.toSlice(io, zml.Slice.init(cache_buf.fl_k.shape(), host.cache_fl_k));
        try cache_buf.fl_v.toSlice(io, zml.Slice.init(cache_buf.fl_v.shape(), host.cache_fl_v));

        if (generated.items.len == 0) {
            log.err("KVDUMP: aucun token généré — état sans fed_next, dump refusé", .{});
            return error.KvDumpInconsistentState;
        }
        const step_next: usize = step + 1;
        // ids_fed = ids ++ generated[0..len-1] (le dernier généré n'a PAS été feedé).
        // Invariant spec §4.2 : ids_fed.len == step_next — assertion dure, jamais silencieuse.
        const n_gen_fed = generated.items.len - 1;
        if (ids.len + n_gen_fed != step_next) {
            log.err("KVDUMP: invariant cassé ids({d})+gen_fed({d}) != step_next({d})", .{ ids.len, n_gen_fed, step_next });
            return error.KvDumpInconsistentState;
        }
        const ids_fed = try allocator.alloc(i32, step_next);
        defer allocator.free(ids_fed);
        for (ids, 0..) |t, k| ids_fed[k] = @intCast(t);
        for (0..n_gen_fed) |k| ids_fed[ids.len + k] = @intCast(generated.items[k]);
        const fed_next: i64 = generated.items[generated.items.len - 1];

        try dumpCacheFile(allocator, io, ds, host, ids_fed, step_next, fed_next, stop_reason, policy, scfg);

        // Contrat « cache ZÉROS par génération » (:2162) restauré pour l'appel suivant.
        @memset(host.cache_sl_k, 0);
        @memset(host.cache_sl_v, 0);
        @memset(host.cache_fl_k, 0);
        @memset(host.cache_fl_v, 0);
    }

    cache_buf.sl_k.deinit();
    cache_buf.sl_v.deinit();
    cache_buf.fl_k.deinit();
    cache_buf.fl_v.deinit();

    // kvdump/DC2 : captures pour l'orchestrateur — APRÈS la boucle et APRÈS le gel des deltas,
    // donc strictement hors de la fenêtre ALLOC-LOOP (aucune interaction avec DC6).
    if (capture) |cap| {
        try cap.ids.appendSlice(allocator, generated.items);
        try cap.top5.appendSlice(allocator, gen_top5.items);
        cap.stop.* = stop_reason;
    }

    // Détecteur de vacuité du chantier (§4.2bis). Dénominateur = les tokens GÉNÉRÉS, jamais les
    // steps : le prefill (27-28 steps sur les témoins) diluerait la mesure au point de la rendre
    // illisible. Un gate de mordant qui lit `0 fois` n'a rien prouvé — c'est le prompt qu'il faut
    // changer, pas le critère.
    log.info("GENCFG: suppress a mordu {d} fois sur {d} tokens générés (prefill exclu)", .{ n_suppress_hits, generated.items.len });
    // D10 : deltas de la fenêtre ALLOC-LOOP (gate AL-0/AL-BASE) + totaux process (non-vacuité).
    log.info("ALLOC-LOOP: alloc={d} resize={d} remap={d} free={d} bytes={d} steps={d}", .{
        al_alloc, al_resize, al_remap, al_free, al_bytes, step + 1,
    });
    log.info("ALLOC-TOTAL: alloc={d} free={d} bytes={d} shards={d}", .{
        counter.n_alloc, counter.n_free, counter.bytes_alloc, scfg.n_shards,
    });
    // === Phase 1 (penalty) : publication + non-vacuité ===
    // L'invariant `hist_len == prompt_len + generated` est ce qui prouve, à l'exécution, que le
    // seed de reprise a bien eu lieu (écart 9 du plan) : sous `--load-cache`, `prompt_len` vaut
    // `step_next` — un historique reparti de zéro rendrait `hist_len == generated` seul.
    if (scfg.repetition_penalty != 1.0) {
        log.info("PENALTY: rp={d} ignore_prompt={} hist_len={d} prompt_len={d} générés={d} n_penalty_touched={d} n_penalty_empty_hist={d}", .{
            scfg.repetition_penalty, scfg.ignore_prompt, scfg.hist_len, scfg.prompt_len, generated.items.len, scfg.n_penalty_touched, scfg.n_penalty_empty_hist,
        });
        if (scfg.hist_len != scfg.prompt_len + generated.items.len) {
            log.err("PENALTY: invariant rompu — hist_len={d} != prompt_len={d} + générés={d}", .{ scfg.hist_len, scfg.prompt_len, generated.items.len });
            return error.PenaltyHistoryInvariant;
        }
        if (scfg.n_penalty_touched == 0) {
            // Exemption : si l'historique était vide à CHAQUE step, HF n'aurait rien touché non
            // plus (cas légitime `--ignore-prompt --max-tokens 1`). Ailleurs, une penalty armée
            // qui ne change JAMAIS un logit est un paramètre non propagé, pas un run réussi.
            if (scfg.n_penalty_empty_hist == scfg.n_steps_compared and scfg.n_penalty_empty_hist > 0) {
                log.warn("PENALTY: inerte mais LÉGITIME — historique vide aux {d} steps (--ignore-prompt sans token généré avant la sélection)", .{scfg.n_penalty_empty_hist});
            } else {
                log.err("PENALTY: INERTE — rp={d} armée mais aucun logit changé sur {d} steps : paramètre non propagé", .{ scfg.repetition_penalty, scfg.n_steps_compared });
                return error.PenaltyInert;
            }
        }
    }
    if (scfg.pathArmed()) {
        log.info("S2-PONT: steps_comparés={d} désaccords={d} égalités_exactes={d} (chemin B armé)", .{ scfg.n_steps_compared, scfg.n_disagree, scfg.n_exact_top_ties });
        if (scfg.repetition_penalty != 1.0) log.info("S2-PONT: les désaccords ci-dessus sont ATTENDUS sous penalty armée (chemin A = topK in-graph sur logits NUS) — ce n'est pas un FAIL", .{});
        if (scfg.n_cout_samples > 0) {
            const moy_us = @as(f64, @floatFromInt(scfg.cout_ns_total)) / @as(f64, @floatFromInt(scfg.n_cout_samples)) / 1000.0;
            const max_us = @as(f64, @floatFromInt(scfg.cout_ns_max)) / 1000.0;
            const n_f: f64 = @floatFromInt(scfg.n_cout_samples);
            const d2h_us = @as(f64, @floatFromInt(scfg.d2h_ns_total)) / n_f / 1000.0;
            const warp_us = @as(f64, @floatFromInt(scfg.warp_ns_total)) / n_f / 1000.0;
            log.info("M-COUT: bloc chemin B — moyenne {d:.1} µs/step (D2H seul {d:.1} µs, warpers {d:.1} µs), max {d:.1} µs, sur {d} steps (MESURE PUBLIÉE, pas un gate)", .{ moy_us, d2h_us, warp_us, max_us, scfg.n_cout_samples });
            // L'instrument mesure aussi celui qui le vérifie : le pont D1/D2 travaille DANS la
            // fenêtre chronométrée (2 memcpy de vocab + une référence f64 + une chaîne mutée).
            // Le chiffre ci-dessus est donc INVALIDE sous --gate-d1d2, et le dire ici vaut mieux
            // que le laisser recopier ailleurs comme s'il était comparable aux mesures M-COUT.
            if (scfg.gate != null) log.warn("M-COUT ci-dessus : NON COMPARABLE aux mesures M-COUT publiées — le pont --gate-d1d2 travaille dans la fenêtre chronométrée. Ne pas le citer comme coût des warpers.", .{});
        }
    }
    if (scfg.gate) |g| {
        // Publication BRUTE des compteurs : le verdict PASS/FAIL est rendu par le dépouilleur
        // (scripts/75_d1d2_gpu_bridge.py), pas par le binaire qui produit les chiffres.
        log.info("G-D1: steps={d} désaccords={d} 1er_id_en_désaccord={d} | ANTÉCÉDENT steps_avec_coupe={d} ids_coupés={d} | frontière_serrée={d} ex_æquo_frontière={d}", .{
            g.n_steps, g.n_topp_disagree, g.first_bad_id, g.n_steps_with_cut, g.n_cut_total, g.n_boundary_tight, g.n_boundary_ties,
        });
        log.info("G-D2: temp_appliquée={d} steps | mutant_a_division_vs_mul={d} logits | mutant_b_ordre={d} ids sur {d} steps", .{
            g.n_temp_applied, g.n_temp_mul_diffs, g.n_order_diffs, g.n_steps_with_order_diff,
        });
    }

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
    const pf_rate = if (pf_s > 0) @as(f64, @floatFromInt(ids.len)) / pf_s else 0;
    const gen_rate = if (gen_s > 0) @as(f64, @floatFromInt(generated.items.len)) / gen_s else 0;
    // kvdump : en reprise, le champ « prefill » serait un MENSONGE (aucun prefill n'a eu lieu ;
    // t_prefill_end n'est jamais réassigné, donc pf_s ne mesurerait que du bruit).
    if (resume_state) |rs| {
        const rs_rate = if (elapsed_s > 0) @as(f64, @floatFromInt(generated.items.len)) / elapsed_s else 0;
        log.info("PERF-RESUME : reprise @step={d}, {d} tokens générés en {d:.3}s ({d:.1} tok/s)", .{ rs.step_next, generated.items.len, elapsed_s, rs_rate });
    } else {
        log.info("PERF : prefill {d} steps en {d:.3}s ({d:.1} tok/s) ; génération {d} tokens en {d:.3}s ({d:.1} tok/s)", .{ ids.len, pf_s, pf_rate, generated.items.len, gen_s, gen_rate });
    }

    // D10 (C8/AL-RSS) : émis AVANT le verdict oracle A1 — un A1Mismatch n'avale pas la mesure.
    if (rss_t20 != null and rss_t200 != null) {
        log.info("RSS-DELTA: {d} KiB (t20={d} t200={d})", .{ rss_t200.? -| rss_t20.?, rss_t20.?, rss_t200.? });
    } else {
        log.info("RSS-DELTA: INEXECUTABLE (run trop court : t20={?d} t200={?d})", .{ rss_t20, rss_t200 });
    }

    // === --out-ids (U9) : ids générés -> safetensors (clé "ids", i32) — AVANT le verdict oracle
    // (un A1Mismatch ne doit pas perdre la trace des ids produits, utile au diagnostic). ===
    if (out_ids_path) |out_path| {
        // K5 : la clé ctx_ids n'apparaît QUE sous reprise AVEC prompt neuf (ids.len > step_next).
        // La reprise simple garde le format historique à une clé — sinon son log changerait et la
        // claim C-K5-E (« le chemin actuel est inchangé ») serait violée par le gate lui-même.
        const k5_resume_prompt = if (resume_state) |rs| ids.len > rs.step_next else false;
        if (k5_resume_prompt) {
            try writeIdsCtxSafetensors(allocator, io, out_path, generated.items, ids);
            log.info("--out-ids : {d} ids générés + ctx_ids {d} écrits -> {s}", .{ generated.items.len, ids.len, out_path });
        } else {
            try writeIdsSafetensors(allocator, io, out_path, generated.items);
            log.info("--out-ids : {d} ids écrits -> {s}", .{ generated.items.len, out_path });
        }
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
            .eot => log.info("arrêt : early-stop EOS id={d} (eos={any}, any-of sans priorité)", .{ stop_eos_id, policy.eos }),
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
        // Writer PARTAGÉ passé par l'appelant (fix R1 : deux writers séparés sur le même fd
        // entrelaçaient leurs octets — un seul writer = une seule file d'écriture).
        try stdout_w.interface.print("réponse : \"{s}\"\n", .{text.items});
        try stdout_w.interface.flush();

        // Verdict A3 (le critère numérique N == index_EOT_expected + 2 est vérifié côté contrôleur
        // contre la fixture ; ici on rend N et la raison d'arrêt VISIBLES).
        if (stop_reason == .eot) {
            log.info("A3 : stop early-EOS après {d} tokens (dernier = {d} ; EOT_ID mesuré = {d})", .{ generated.items.len, stop_eos_id, eot_id });
        }
    }
}

};
}

pub fn main(init: std.process.Init) !void {
    return G12Auto(1280).run(init);
}
