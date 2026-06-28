# Session du 27 juin 2026 — Rapport complet consolidé

> **Document unique et auto-suffisant** regroupant l'analyse de codebase, l'analyse des risques (point 10),
> tous les livrables de la session (avec le **code exact intégré**), et le playbook de validation sur la
> 3090. Les fichiers réels sont également écrits dans l'arbre de travail (cf §« État des fichiers ») ;
> ce document fige l'état pour archive/transmission indépendamment du working tree.
>
> **Contexte d'exécution** : environnement sandbox — réseau bloqué (SSH 3090 = `Operation not permitted`),
> `.git` read-only (commit impossible ici), pas de `weights/` local, pas de toolchain Zig/CUDA
> compilable localement. **Rien n'a été compilé ni exécuté par l'auteur.** Tous les livrables sont
> **prêts-à-valider sur la 3090** par Régis. Garantie : motifs mirrorés du code L1a prouvé, braces/parens
> équilibrés, API vérifiées dans le checkout ZML, défauts comptime NEUTRES (byte-identique).
>
> Branche active : `generation-longue`. Auteur : analyse codebase. Statut global : **livré, non validé**.

---

## Sommaire

1. Analyse de la codebase `gemma4-zml-probe` (synthèse)
2. Point 10 — Points d'attention / risques (analyse approfondie)
3. Livrables R1–R6 (mitigation des risques)
4. Livrables L1b (ring 512) + L2 (autonome host)
5. Plan GPU (référence) + livrables P-GPU-1 (G1/bench + knob PrecCfg)
6. Playbook de validation sur la 3090
7. Honnêteté / limites / non-régression
8. État des fichiers (working tree)

---

## 1. Analyse de la codebase (synthèse)

**`gemma4-zml-probe`** = portage bit-exact, op-par-op de `google/gemma-4-E2B-it` (chemin texte) vers
**ZML** (Zig + MLIR + OpenXLA + PJRT). Moteur de **recherche** (baseline fp32 prouvé vs HF), pas un runtime
de prod. ~50 gates atomiques tagués. État affiché : forward + logits + decode court (4 tokens) ==
HF ; génération longue (L1a) 1020 tokens == HF.

Particularités Gemma 4 portées : PLE (Per-Layer Embeddings), Shared KV Cache / YOCO (writers 13/14 →
readers 15-34), 2 types de couche (sliding head_dim 256 / full head_dim 512 RoPE partielle 0.25 θ=1e6),
GQA 8/1, RMSNorm Llama (`×weight`), `v_norm` sans scale, gelu_tanh, softcap 30·tanh, layer_scalar.

Méthode (la vraie valeur) : chaque op = 1 gate, oracle = source de vérité (`modeling_gemma4.py`),
multi-tap isolation, non-vacuité (corrompre l'oracle → doit FAIL), revue adversariale, contrat de
précision hybride (fp32 sauf embptl bf16). Pièges capitalisés : v_norm D.0→D.0b (oracle partagait
l'hypothèse fausse), 2 bugs oracle (layer_scalar buffer, embed_scale bf16).

Socle : `engine.zig` = `EngineModel(comptime Brick, comptime cfg)` — forward paramétré comptime,
injection de briques (TurboQuant V-only) sans copier le moteur, neutralité HLO prouvée (`diff -rq`).

Branches : `main`, `turboquant-zml-vonly`, `generation-longue` (active). Front actif : génération
longue (L1a fait, L1b/L2 restent), bloqué par le mur mémoire CPU (~33 Go compile 35 couches fp32).

---

## 2. Point 10 — Points d'attention / risques (analyse approfondie)

| # | Risque | Sév | Preuve concrète (codebase) |
|---|---|---|---|
| R1 | Dette mémoire + fuite swap + pic non mesuré | 🔴 | `gemma4_gchunk.zig:9-11` « best-effort (petite fuite tolérée) » + log `=== POST-COMPILE ===` sans capture ; ENGINE_LOG 7 juin swap 2.8→4.1 Go ; swapfile `/swapfile_xla` non pérennisé ; CHUNKING_DESIGN §5 « go/no-go » jamais réellement mesuré |
| R2 | Non-vacuité L1a absente | 🔴 | ENGINE_LOG:101 « contre-test non-vacuité L1a… partiellement couvert par L1b à venir » ; aucun tag `L1a-*` ; le 1020/1020 est affirmé, non réfuté adversarialement |
| R3 | L1b/L2/2048 non démarrés | 🟠 | `gemma4_gen_long_ring.zig`/`_auto.zig` absents ; `L_MAX` abaissé 2048→1024 partout (`46`, `gen_long`, `gchunk`) |
| R4 | Perf 55 min, 7 syncs/step | 🟠 | ENGINE_LOG « vitesse 55 min… dominée par 7 syncs host/step » ; DESIGN §7 « cache 30 MB négligeable » contredit (pic = poids f32) |
| R5 | Repro hors-infra | 🟡 | Aucun test auto, fixtures gitignorées, dépendance 3090/SSH |
| R6 | Docs spec ≠ vérité | 🟡 | README « port complete » sans qualifier decode 4-t vs gen-longue 1020 ; DESIGN D3 « 2048 » vs impl 1024 ; DESIGN §7 mémoire inexacte |

**Point-clé** : R1+R2 ne sont pas des risques « à surveiller » mais des **conditions non satisfaites du
gate L1a lui-même** (design §6 = pic post-compile go/no-go ; méthode = non-vacuité obligatoire). L1a est
donc, en rigueur, *provisional*.

---

## 3. Livrables R1–R6 (mitigation)

### R2 — Contre-test non-vacuité L1a : `zml_runner/gemma4_gchunk_vacuity.zig`

```zig
// L1a — CONTRE-TEST DE NON-VACUITÉ (R2, cf analyse point 10).
//
// But : prouver que le masque bande `masks_sliding` est RÉELLEMENT consommé par l'attention sliding
// (réfuter l'aliasing / un PASS trompeur où le masque serait ignoré). C'est la contrepartie obligatoire
// du gate L1a (méthode du projet : non-vacuité, cf docs/GENERATION_LONGUE_PLAN.md Step 4 + DESIGN §7).
//
// Corruption : on rebind le buffer `masks_sliding` sur le buffer `masks_full` (causal plein [0,p]) au
// lieu de la bande [max(0,p-511), p]. Effet : les couches sliding voient TOUT le passé (fenêtre 512
// DÉSACTIVÉE) au lieu des 512 dernières positions.
//   - p < 511 : bande ≡ causal (lo=0) → tokens IDENTIQUES à HF (pas de divergence avant ~p=511).
//   - p > 511 (p>=512) : la bande HF tronque (lo=p-511>0) à [p-511,p], notre version voit [0,p] → attention différente →
//     logits différents → argmax différent → DIVERGENCE.
//
// Critère INVERSÉ (PASS = divergence observée) :
//   - Si argmax diverge d'`expected` sur >= 1 position (typiquement à partir de p~512, step ~508) →
//     le masque est bien consommé → NON-VACUITÉ PROUVÉE → on log "VACUITY-OK" (le test « réussit » en
//     échouant à reproduire HF).
//   - Si argmax == expected sur TOUTES les positions → le masque est ignoré (aliasing/vacuité) →
//     BUG → on return error.Vacuity (le test « échoue » = alerte rouge).
//
// Diagnostic : on rapporte `first_fail` (position de 1re divergence, attendue ~512/step ~508) et le compte de
// divergences. Une divergence bien AVANT 511 est suspecte (les masques bande/causal coïncident pour
// p<511 → une divergence précoce indiquerait une autre cause, à investiguer).
//
// On réutilise l'exécution chunkée (== gchunk, chemin L1a prouvé) pour rester fidèle au gate qu'on
// contre-teste. max_steps (3e arg optionnel) permet un run court (ex: 600) capturant la divergence
// au p~512 sans attendre les 1020 steps complets.
//
// CLI : gemma4_gchunk_vacuity <model.safetensors> <gen_long.safetensors> [max_steps]

const std = @import("std");
const log = std.log;
const zml = @import("zml");
const engine = @import("engine.zig");
const mem_probe = @import("mem_probe.zig");

pub const std_options: std.Options = .{ .log_level = .info };

const L_MAX: i64 = 1024;
const NUM_LAYERS: usize = 35;
const CHUNK: usize = 5;
const SYNC_EVERY: usize = 1; // == gchunk L1a (chemin contre-testé)
const B: i64 = 1;
const S: i64 = 1;
const D: i64 = 1536;
const SLIDING_WINDOW: i64 = 512;
const SEQ_LEN: i64 = 4; // positions décodées = SEQ_LEN + step_idx → p=511 ≈ step 507

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

const ExpW = struct {
    e: zml.Tensor,
    pub fn init(v: zml.io.TensorStore.View) ExpW {
        return .{ .e = v.createTensor("expected", .{.step}, null) };
    }
    pub fn load(self: *const ExpW, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, store: *const zml.io.TensorStore, shardings: []const zml.sharding.Sharding) !zml.Bufferized(ExpW) {
        return zml.io.load(ExpW, self, allocator, io, platform, store, .{ .shardings = shardings, .parallelism = 1, .dma_chunks = 1, .dma_chunk_size = 16 * 1024 * 1024 });
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

pub fn main(init: std.process.Init) !void {
    @setEvalBranchQuota(200000);
    const arena = init.arena;
    const allocator = init.gpa;
    const io = init.io;

    const process_args = try init.minimal.args.toSlice(arena.allocator());
    if (process_args.len < 3) {
        log.err("Usage: gemma4_gchunk_vacuity <model.safetensors> <gen_long.safetensors> [max_steps]", .{});
        return error.MissingArgument;
    }
    const ckpt = process_args[1];
    const fixture = process_args[2];
    const max_steps: ?usize = if (process_args.len >= 4)
        std.fmt.parseInt(usize, process_args[3], 10) catch null
    else
        null;
    log.info("L1a NON-VACUITÉ — masks_sliding corrompu en causal (fenêtre 512 OFF), attend 1re divergence ~p={d} (step ~508)", .{SLIDING_WINDOW});

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

    const eng_buf = try model.load(arena.allocator(), io, platform, &store_ck, &.{sharding});
    const pk_buf = try packed_in.load(arena.allocator(), io, platform, &store_fx, &.{sharding});
    var cache_buf = try cache0.load(arena.allocator(), io, platform, &store_fx, &.{sharding});
    const exp_sym: ExpW = .init(store_fx.view());
    var exp_buf = try exp_sym.load(arena.allocator(), io, platform, &store_fx, &.{sharding});
    store_ck.deinit();
    reg_ck.deinit();

    var exp_slice = try exp_buf.e.toSliceAlloc(allocator, io);
    defer exp_slice.free(allocator);
    const expected_tokens = exp_slice.items(i32);
    const total = expected_tokens.len;
    const num_steps = if (max_steps) |m| @min(m, total) else total;

    // ===== CORRUPTION : rebind masks_sliding <- masks_full (fenêtre OFF, causal [0,p]) =====
    // Bufferized(PackedLong) = struct de Buffers (handles device). On construit une copie où le champ
    // masks_sliding pointe sur le buffer masks_full. Les shapes .k=L_MAX coïncident (fixture L0). Les
    // buffers sont en lecture seule (inputs) → l'aliasing d'un même buffer pour 2 champs est sûr.
    const pk_corrupt = zml.Bufferized(PackedLong){
        .embeds = pk_buf.embeds,
        .embptls = pk_buf.embptls,
        .cos_full = pk_buf.cos_full,
        .sin_full = pk_buf.sin_full,
        .masks_sliding = pk_buf.masks_full, // ← CORRUPTION : bande -> causal (fenêtre OFF)
        .masks_full = pk_buf.masks_full,
        .positions = pk_buf.positions,
    };

    const hidden_shape = zml.Shape.init(.{ B, S, D }, .f32).withTags(.{ .b, .s, .d });
    const zeros = try arena.allocator().alloc(u8, @intCast(B * S * D * 4));
    @memset(zeros, 0);
    var dummy_hidden = try zml.Buffer.fromBytes(io, platform, hidden_shape, sharding, zeros);
    defer dummy_hidden.deinit();

    log.info("Compiling {d} stages (chemin == gchunk L1a)...", .{N_STAGES});
    var exes: [N_STAGES]zml.Exe = undefined;
    inline for (STAGES, 0..) |stage, si| {
        const F = struct {
            fn f(m: Model, p: PackedLong, ca: engine.Cache, h: zml.Tensor, ct: engine.Ctrl) StageOut {
                return m.forwardStageGen(stage.start, stage.end, stage.first, stage.last, p, ca, h, ct);
            }
        }.f;
        exes[si] = try platform.compileFn(allocator, io, F, .{ model, packed_in, cache0, hidden_sym, ctrl_sym }, .{ .shardings = &.{sharding} });
    }
    defer for (&exes) |*e| e.deinit();
    mem_probe.logMem("post-compile");

    // ===== Boucle : on attend DIVERGENCE (critère inversé) =====
    var n_match: usize = 0;
    var n_diverge: usize = 0;
    var first_fail: i64 = -1;
    var step_idx: usize = 0;
    while (step_idx < num_steps) : (step_idx += 1) {
        var step_buf = try zml.Buffer.scalar(io, platform, @as(u32, @intCast(step_idx)), .u32, sharding);
        const ctrl_buf = zml.Bufferized(engine.Ctrl){ .step = step_buf };

        var hidden_buf = dummy_hidden;
        var tok: i64 = -1;
        inline for (STAGES, 0..) |stage, si| {
            var args = try exes[si].args(allocator);
            var results = try exes[si].results(allocator);
            args.set(.{ eng_buf, pk_corrupt, cache_buf, hidden_buf, ctrl_buf }); // ← pk_corrupt
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

        const exp = @as(i64, @intCast(expected_tokens[step_idx]));
        const pos = SEQ_LEN + @as(i64, @intCast(step_idx));
        if (tok == exp) {
            n_match += 1;
        } else {
            n_diverge += 1;
            if (first_fail < 0) first_fail = @intCast(step_idx);
            if (n_diverge <= 8 or pos >= SLIDING_WINDOW - 8) {
                log.info("  DIVERGENCE step {d} (pos {d}) : ZML(corrompu)={d} HF={d}", .{ step_idx, pos, tok, exp });
            }
        }
        step_buf.deinit();
    }
    cache_buf.sl_k.deinit();
    cache_buf.sl_v.deinit();
    cache_buf.fl_k.deinit();
    cache_buf.fl_v.deinit();

    log.info("NON-VACUITÉ : {d}/{d} divergent, {d} match (first_fail step {d})", .{ n_diverge, num_steps, n_match, first_fail });

    // Critère inversé : divergence observée => masque consommé => non-vacuité PROUVÉE.
    if (n_diverge > 0) {
        const ff_pos = if (first_fail >= 0) SEQ_LEN + first_fail else -1;
        log.info("VACUITY-OK (non-vacuité prouvée) — le masque bande est bien consommé ; 1re divergence au pos {d}", .{ff_pos});
        if (ff_pos >= 0 and ff_pos < SLIDING_WINDOW - 4) {
            log.warn("  1re divergence précoce (pos {d} < {d}-4) : suspect (bande≡causal pour p<511) → investiguer", .{ ff_pos, SLIDING_WINDOW });
        }
    } else {
        // Aucune divergence malgré la fenêtre désactivée → le masque est ignoré → VACUITÉ (bug).
        log.err("VACUITY-FAIL : {d}/{d} match malgré masks_sliding corrompu — le masque bande N'EST PAS consommé (aliasing/vacuité) !", .{ n_match, num_steps });
        return error.Vacuity;
    }
}

```
### R1 — Instrumentation mémoire : `zml_runner/mem_probe.zig`

```zig
// Helper mémoire — lecture RSS / swap depuis /proc/self/status (Linux), no-op ailleurs.
//
// But : instrumenter le pic mémoire post-compile (go/no-go du chunking, cf
// docs/GENERATION_LONGUE_CHUNKING_DESIGN.md §6) et caractériser la fuite résiduelle (swap 2.8→4.1 Go
// observée, ENGINE_LOG 7 juin). Sur la 3090 (Linux) : lit VmRSS/VmSwap en Ko. Sur un build non-Linux
// (ex: Mac local pour syntaxe) : renvoie null (les log* sont comptime-morts → pas d'output parasite).
//
// Unité : kilo-octets (Ko) — cohérent avec /proc (kB = page-size-agnostic, base 1024).

const std = @import("std");
const builtin = @import("builtin");
const log = std.log;

/// RSS du process en Ko, ou null si /proc indisponible (non-Linux / lecture échouée).
pub fn rssKb() ?u64 {
    if (builtin.os.tag != .linux) return null;
    return readField("VmRSS:");
}

/// Swap du process en Ko, ou null.
pub fn swapKb() ?u64 {
    if (builtin.os.tag != .linux) return null;
    return readField("VmSwap:");
}

fn readField(prefix: []const u8) ?u64 {
    var file = std.fs.openFileAbsolute("/proc/self/status", .{}) catch return null;
    defer file.close();
    var buf: [4096]u8 = undefined;
    const n = file.read(&buf) catch return null;
    const content = buf[0..n];
    var it = std.mem.splitScalar(u8, content, '\n');
    while (it.next()) |line| {
        if (std.mem.startsWith(u8, line, prefix)) {
            // "VmRSS:    12345 kB"
            const rest = line[prefix.len..];
            const trimmed = std.mem.trim(u8, rest, " \t");
            var num_end: usize = 0;
            for (trimmed) |ch| {
                if (ch < '0' or ch > '9') break;
                num_end += 1;
            }
            if (num_end == 0) return null;
            return std.fmt.parseInt(u64, trimmed[0..num_end], 10) catch null;
        }
    }
    return null;
}

/// Log formaté RSS+swap avec un tag. No-op total hors-Linux (branches comptime-mortes).
/// GiB arrondi à l'entier (u64, spec `{d}`) — évite tout spec de précision float non validable hors-3090.
pub fn logMem(tag: []const u8) void {
    if (builtin.os.tag != .linux) return;
    const rss = rssKb() orelse return;
    const rss_gib: u64 = rss / 1048576;
    const swp = swapKb();
    if (swp) |s| {
        const s_gib: u64 = s / 1048576;
        log.info("[mem] {s}: RSS={d} KiB (~{d} GiB) swap={d} KiB (~{d} GiB)", .{ tag, rss, rss_gib, s, s_gib });
    } else {
        log.info("[mem] {s}: RSS={d} KiB (~{d} GiB) swap=n/a", .{ tag, rss, rss_gib });
    }
}

```
### R1/R4 — `gemma4_gchunk.zig` modifié (instrumentation RSS + knob `SYNC_EVERY`)

```zig
// L1a CHUNKÉ (perf) — decode découpé en stages compilés séparément pour borner le pic mémoire.
//
// Adapte le mode `chain` du prefill au decode (cf docs/GENERATION_LONGUE_CHUNKING_DESIGN.md) :
// les 35 couches sont découpées en stages de CHUNK couches, chacun compilé via `compileFn` (fn-factory
// comptime). À chaque step on exécute les stages en séquence, threadant hidden + cache device→device,
// avec sync (toSliceAlloc) après chaque pour libérer le working set. Calcul == forward mono (runLayerGen
// partagé) → mêmes tokens, mais pic mémoire borné (moins de poids f32 coexistant).
//
// GATE 0 (cette version) : compile N stages, mesure le pic POST-COMPILE (les N exe résidents = go/no-go),
// puis exécute NUM_STEPS_GATE0 steps pour vérifier l'équivalence (tokens == expected). Gestion mémoire
// best-effort (petite fuite tolérée sur peu de steps ; le pic post-compile est mesuré AVANT les steps).
//
// CHANGEMENTS vs v1 (cf analyse point 10, R1/R4) :
//   - Instrumentation RSS/swap réelle (mem_probe.zig) : pic post-compile (le go/no-go annoncé, enfin
//     MESURÉ au lieu d'être seulement logué) + RSS par fenêtre de steps (caractérise la fuite résiduelle
//     swap 2.8→4.1 Go observée). No-op hors-Linux.
//   - Knob comptime SYNC_EVERY : fréquence de sync entre stages. Défaut 1 = sync après chaque stage =
//     comportement L1a exact (non-régression préservée). >1 = moins de round-trips host mais working
//     sets qui s'accumulent (trade-off perf/mémoire à caractériser via scripts/sweep_perf.sh).
//
// CLI : gemma4_gen_long_chunked <model.safetensors> <gen_long.safetensors>

const std = @import("std");
const log = std.log;
const zml = @import("zml");
const engine = @import("engine.zig");
const mem_probe = @import("mem_probe.zig");

pub const std_options: std.Options = .{ .log_level = .info };

const L_MAX: i64 = 1024;
const NUM_LAYERS: usize = 35;
const CHUNK: usize = 5; // couches/stage (divise 15 → pas de stage mixte producer/reader)
const SYNC_EVERY: usize = 1; // sync après chaque (SYNC_EVERY)-ième stage. Défaut 1 = L1a exact.
const NUM_STEPS_GATE0: usize = 4; // gate 0 : équivalence sur quelques steps (fuite cache tolérée)
const RSS_EVERY: usize = 64; // log RSS/swap tous les RSS_EVERY steps (caractérisation fuite)
const B: i64 = 1;
const S: i64 = 1;
const D: i64 = 1536;

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

const ExpW = struct {
    e: zml.Tensor,
    pub fn init(v: zml.io.TensorStore.View) ExpW {
        return .{ .e = v.createTensor("expected", .{.step}, null) };
    }
    pub fn load(self: *const ExpW, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, store: *const zml.io.TensorStore, shardings: []const zml.sharding.Sharding) !zml.Bufferized(ExpW) {
        return zml.io.load(ExpW, self, allocator, io, platform, store, .{ .shardings = shardings, .parallelism = 1, .dma_chunks = 1, .dma_chunk_size = 16 * 1024 * 1024 });
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

pub fn main(init: std.process.Init) !void {
    @setEvalBranchQuota(200000); // inline for sur N_STAGES × compileFn générique déborde le quota par défaut
    const arena = init.arena;
    const allocator = init.gpa;
    const io = init.io;

    const process_args = try init.minimal.args.toSlice(arena.allocator());
    if (process_args.len < 3) {
        log.err("Usage: gemma4_gen_long_chunked <model.safetensors> <gen_long.safetensors>", .{});
        return error.MissingArgument;
    }
    const ckpt = process_args[1];
    const fixture = process_args[2];
    log.info("L1a CHUNKÉ — {d} couches en {d} stages de {d} (L_MAX={d}, SYNC_EVERY={d})", .{ NUM_LAYERS, N_STAGES, CHUNK, L_MAX, SYNC_EVERY });

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

    log.info("Materializing weights + packed + cache0...", .{});
    const eng_buf = try model.load(arena.allocator(), io, platform, &store_ck, &.{sharding});
    const pk_buf = try packed_in.load(arena.allocator(), io, platform, &store_fx, &.{sharding});
    var cache_buf = try cache0.load(arena.allocator(), io, platform, &store_fx, &.{sharding});
    const exp_sym: ExpW = .init(store_fx.view());
    var exp_buf = try exp_sym.load(arena.allocator(), io, platform, &store_fx, &.{sharding});
    store_ck.deinit();
    reg_ck.deinit();
    mem_probe.logMem("post-load (poids+packed+cache)");

    var exp_slice = try exp_buf.e.toSliceAlloc(allocator, io);
    defer exp_slice.free(allocator);
    const expected_tokens = exp_slice.items(i32);
    const num_steps = expected_tokens.len; // run complet (équivalence sur toute la séquence)

    // dummy hidden {b,s,d} (entrée ignorée du first stage).
    const hidden_shape = zml.Shape.init(.{ B, S, D }, .f32).withTags(.{ .b, .s, .d });
    const zeros = try arena.allocator().alloc(u8, @intCast(B * S * D * 4));
    @memset(zeros, 0);
    var dummy_hidden = try zml.Buffer.fromBytes(io, platform, hidden_shape, sharding, zeros);
    defer dummy_hidden.deinit();

    // ===== Compile les N stages (fn-factory comptime) =====
    log.info("Compiling {d} stages...", .{N_STAGES});
    var exes: [N_STAGES]zml.Exe = undefined;
    inline for (STAGES, 0..) |stage, si| {
        const F = struct {
            fn f(m: Model, p: PackedLong, ca: engine.Cache, h: zml.Tensor, ct: engine.Ctrl) StageOut {
                return m.forwardStageGen(stage.start, stage.end, stage.first, stage.last, p, ca, h, ct);
            }
        }.f;
        exes[si] = try platform.compileFn(allocator, io, F, .{ model, packed_in, cache0, hidden_sym, ctrl_sym }, .{ .shardings = &.{sharding} });
        log.info("  stage {d} [{d},{d}) first={} last={} compilé", .{ si, stage.start, stage.end, stage.first, stage.last });
    }
    defer for (&exes) |*e| e.deinit();
    log.info("=== POST-COMPILE : {d} stages résidents ===", .{N_STAGES});
    mem_probe.logMem("post-compile (go/no-go : pic des N exe résidents)"); // ← la mesure annoncée, enfin capturée

    // ===== Boucle steps (run complet : équivalence sur num_steps) =====
    var all_pass = true;
    var n_match: usize = 0;
    var step_idx: usize = 0;
    const rss0 = mem_probe.rssKb();
    while (step_idx < num_steps) : (step_idx += 1) {
        var step_buf = try zml.Buffer.scalar(io, platform, @as(u32, @intCast(step_idx)), .u32, sharding);
        const ctrl_buf = zml.Bufferized(engine.Ctrl){ .step = step_buf };

        var hidden_buf = dummy_hidden; // first stage : ignoré
        var tok: i64 = -1;
        inline for (STAGES, 0..) |stage, si| {
            var args = try exes[si].args(allocator);
            var results = try exes[si].results(allocator);
            args.set(.{ eng_buf, pk_buf, cache_buf, hidden_buf, ctrl_buf });
            exes[si].call(args, &results);
            var out0, const nsl, const nsv, const nfl, const nfv = results.get(struct {
                zml.Buffer, zml.Buffer, zml.Buffer, zml.Buffer, zml.Buffer,
            });

            // sync out0 (matérialise → libère le working set du stage). SYNC_EVERY borne la fréquence :
            // défaut 1 = sync après chaque stage (== L1a, mémoire bornée). >1 = moins de round-trips host
            // mais working sets accumulés (trade-off perf/mémoire, cf sweep_perf.sh). Le dernier stage est
            // toujours matérialisé par l'argmax qui suit (toSliceAlloc) → son working set est libéré de toute façon.
            const do_sync = (si % SYNC_EVERY == SYNC_EVERY - 1);
            if (do_sync) {
                var s = try out0.toSliceAlloc(allocator, io);
                s.free(allocator);
            }

            // thread cache : deinit l'ancien (pattern e1 — les buffers d'entrée ne sont pas « donnés » par
            // call, donc deinitables après ; les reader-stages retournent une copie du cache, pas un alias).
            var old_cache = cache_buf;
            cache_buf = zml.Bufferized(engine.Cache){ .sl_k = nsl, .sl_v = nsv, .fl_k = nfl, .fl_v = nfv };
            old_cache.sl_k.deinit();
            old_cache.sl_v.deinit();
            old_cache.fl_k.deinit();
            old_cache.fl_v.deinit();

            // thread hidden
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

        const exp = @as(i64, @intCast(expected_tokens[step_idx]));
        const ok = tok == exp;
        if (ok) n_match += 1 else all_pass = false;
        if (!ok) log.err("  FAIL step {d} (pos {d}) : argmax ZML={d} HF={d}", .{ step_idx, 4 + step_idx, tok, exp });
        if ((step_idx + 1) % 256 == 0) {
            log.info("  ... {d}/{d} steps, {d} match", .{ step_idx + 1, num_steps, n_match });
        }
        // caractérisation fuite : RSS/swap tous les RSS_EVERY steps (delta vs post-compile).
        if ((step_idx % RSS_EVERY == RSS_EVERY - 1) and (rss0 != null)) {
            var tag_buf: [32]u8 = undefined;
            const tag = std.fmt.bufPrint(&tag_buf, "step {d}", .{step_idx}) catch "step";
            mem_probe.logMem(tag);
        }
        step_buf.deinit();
    }
    mem_probe.logMem("post-run (final)");

    log.info("L1a CHUNKÉ : {d}/{d} tokens match", .{ n_match, num_steps });
    if (all_pass) {
        log.info("L1a CHUNKÉ PASS — moteur chunké == HF greedy sur {d} tokens (== mono, exécution borné mémoire)", .{num_steps});
    } else {
        log.err("L1a CHUNKÉ : divergence vs expected", .{});
        return error.GenMismatch;
    }
}

```
### R4 — Harness de sweep perf : `scripts/sweep_perf.sh`

```bash
#!/usr/bin/env bash
# sweep_perf.sh — Caractérisation du trade-off perf/mémoire du decode chunké (R4, cf analyse point 10).
#
# Le decode chunké a 2 leviers comptime antagonistes (cf docs/GENERATION_LONGUE_CHUNKING_DESIGN.md §5) :
#   - CHUNK       : couches/stage. Plus grand = moins de stages = moins de syncs host/step MAIS pic
#                  compilation/stage plus gros (plus de poids f32 coexistant).
#   - SYNC_EVERY  : fréquence de sync (toSliceAlloc) entre stages. Défaut 1 = sync après chaque stage
#                  (mémoire bornée). Plus grand = moins de round-trips host MAIS working sets qui
#                  s'accumulent (le risque que le design §5 point 2 a flaggé comme « inconnue centrale »).
#
# CHUNK est NON-MONOTONE (design §5 point 2) : trop petit = pic/stage faible MAIS plus d'exe résidents
# → peut AGGRAVER le pic dominant. Ce sweep cherche l'optimum (pic <23 Go ET run rapide).
#
# Méthode : pour chaque (CHUNK, SYNC_EVERY) on patch les 2 consts comptime du runner, on build+run
# sur la 3090, on capture : temps total, RSS post-compile (go/no-go), RSS/swap finaux, match count.
# On restore le fichier après chaque config (le runner de référence doit rester CHUNK=5/SYNC_EVERY=1).
#
# ATTENTION : ce script PATCH gemma4_gchunk.zig en place (sur la 3090). Il le restore en finally.
# Les configs sont hardcodées (petite grille) — édite CONFIGS pour balayer plus large.
#
# Usage (sur la 3090, depuis le workspace ZML) :
#   cd /data/rqz_workspace/zml
#   bash /data/gemma4-zml-probe/scripts/sweep_perf.sh <model.safetensors> <gen_long.safetensors> [max_steps]
# Exemple : bash .../sweep_perf.sh weights/model.safetensors gen_long.safetensors 64   # grille courte
set -euo pipefail

CKPT="${1:-/data/gemma4-zml-probe/weights/model.safetensors}"
FIXTURE="${2:-/data/gemma4-zml-probe/gen_long.safetensors}"
MAXSTEPS="${3:-}"   # optionnel : cappe le run (ex: 64) pour itérer vite sur la grille

# Grille (CHUNK SYNC_EVERY). Garder petite : chaque config = 1 build (~min) + 1 run.
# CHUNK doit diviser 15 (pour ne pas couper un producer/reader au milieu) → 3,5,15. 7 ne divise pas 15.
CONFIGS=(
  "5 1"     # référence L1a (baseline)
  "3 1"     # +petit CHUNK : moins de pic/stage, +d'exe résidents (test non-monotonie)
  "7 1"     # +grand CHUNK : -de syncs, +gros pic/stage (7 ne divise pas 15 → stage mixte, à observer)
  "15 1"    # CHUNK=15 : 1 stage producteur + ... (très gros pic, borne haute)
  "5 2"     # SYNC_EVERY=2 : -de syncs, working sets s'accumulent (l'inconnue §5.2)
  "5 7"     # SYNC_EVERY=7 : sync seulement au dernier stage = max mémoire, min round-trips
)

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUNNER="$ROOT/zml_runner/gemma4_gchunk.zig"
ZML_WS="${ZML_WS:-/data/rqz_workspace/zml}"
BAZEL="./bazel.sh"
# les sources sont déployées dans examples/rqz/ (cf deploy_to_3090.sh) — on patch la copie déployée
DEPLOYED="${ZML_WS}/examples/rqz/gemma4_gchunk.zig"
[ -f "$DEPLOYED" ] || { echo "ERR: runner déployé introuvable: $DEPLOYED (lance deploy_to_3090.sh d'abord)"; exit 1; }

# Restaure toujours la copie déployée à la fin (le git local reste intact : on patch $DEPLOYED, pas $RUNNER).
cleanup() { cp "$RUNNER" "$DEPLOYED" 2>/dev/null || true; }
trap cleanup EXIT

run_args="$CKPT $FIXTURE"
[ -n "$MAXSTEPS" ] && : # gchunk ne supporte pas max_steps natif (run = len expected) ; pour un run court
                        # on tronque `expected` côté fixture. Ici on run complet — voir note ci-dessous.

printf "%-12s %-10s %-14s %-14s %-14s %-10s\n" "CHUNK" "SYNC_EVERY" "RSS_postcomp_GiB" "swap_final_GiB" "RSS_final_GiB" "match"
echo "----------------------------------------------------------------------------------------"

for cfg in "${CONFIGS[@]}"; do
  set -- $cfg; CHUNK=$1; SE=$2
  # patch les 2 consts comptime (lignes `const CHUNK: usize = ...` / `const SYNC_EVERY: usize = ...`)
  sed -E -i.bak \
    -e "s/^const CHUNK: usize = [0-9]+;/const CHUNK: usize = ${CHUNK};/" \
    -e "s/^const SYNC_EVERY: usize = [0-9]+;/const SYNC_EVERY: usize = ${SE};/" \
    "$DEPLOYED"
  rm -f "${DEPLOYED}.bak"

  echo ">>> build CHUNK=$CHUNK SYNC_EVERY=$SE ..."
  ( cd "$ZML_WS" && $BAZEL build //examples/rqz:gemma4_gchunk >/dev/null 2>&1 ) || {
    echo "    BUILD FAIL (CHUNK=$CHUNK SE=$SE) — stage mixte (7 ne divise pas 15) ?"; 
    printf "%-12s %-10s %-14s %-14s %-14s %-10s\n" "$CHUNK" "$SE" "BUILD_FAIL" "-" "-" "-"
    cp "$RUNNER" "$DEPLOYED"; continue
  }

  LOG=$(mktemp)
  t0=$(date +%s)
  ( cd "$ZML_WS" && $BAZEL-bin/examples/rqz/gemma4_gchunk $run_args ) >"$LOG" 2>&1 || true
  t1=$(date +%s)
  dt=$((t1 - t0))

  # parse la sortie instrumentée (mem_probe logs : "[mem] tag: RSS=... KiB (... GiB) swap=... KiB (... GiB)")
  postcomp=$(grep -m1 '\[mem\] post-compile' "$LOG" | grep -oE 'RSS=[0-9]+ KiB \([0-9.]+ GiB\)' | grep -oE '\([0-9.]+ GiB' | tr -d '( ' || echo "?")
  final=$(grep -m1 '\[mem\] post-run' "$LOG" | grep -oE 'RSS=[0-9]+ KiB \([0-9.]+ GiB\)' | grep -oE '\([0-9.]+ GiB' | tr -d '( ' || echo "?")
  swapf=$(grep -m1 '\[mem\] post-run' "$LOG" | grep -oE 'swap=[0-9]+ KiB \([0-9.]+ GiB\)' | grep -oE '\([0-9.]+ GiB' | tr -d '( ' || echo "?")
  match=$(grep -oE 'L1a CHUNKÉ : [0-9]+/[0-9]+ tokens match' "$LOG" | grep -oE '[0-9]+/[0-9]+' || echo "?")
  verdict=$(grep -m1 'L1a CHUNKÉ PASS\|divergence vs expected' "$LOG" | head -c 60 || echo "?")

  printf "%-12s %-10s %-14s %-14s %-14s %-10s  (%ds, %s)\n" "$CHUNK" "$SE" "${postcomp}GiB" "${swapf}GiB" "${final}GiB" "$match" "$dt" "$verdict"
  cp "$RUNNER" "$DEPLOYED"   # restore pour la prochaine config
done

echo "----------------------------------------------------------------------------------------"
echo "Lecture : RSS_postcomp = pic go/no-go (doit <23 GiB). match doit être N/N (équivalence préservée)."
echo "          CHUNK=5/SYNC_EVERY=1 = baseline L1a. swap_final > 0 = fuite résiduelle (R1) ; compare les configs."
echo "Note : run complet 1020 steps (~55 min baseline). Pour itérer vite, tronque la fixture `expected`."

```
### R5 — Orchestrateur de fixtures : `scripts/regen_fixtures.sh`

```bash
#!/usr/bin/env bash
# regen_fixtures.sh — Régénère TOUTES les fixtures (gitignorées) depuis les oracles Python, dans l'ordre
# des dépendances des gates (R5, cf analyse point 10).
#
# Les fixtures (.safetensors/.npy/.pt) sont gitignorées (régénérables) : un clone frais ne peut rien
# exécuter sans elles. Ce script orchestre la chaîne P-1 → P5.7.8 + TurboQuant + génération longue,
# dans l'ordre imposé par les gates (chacun consomme la sortie du précédent).
#
# Cible : la 3090 (venv gemma4-probe, weights/, HF cache). Lance depuis le repo sur la 3090.
#
# Usage :
#   bash scripts/regen_fixtures.sh             # tout, dans l'ordre
#   bash scripts/regen_fixtures.sh ple          # une phase : ple|yoco|p52|p54-p56|p57|decode|tq|genlong
#   bash scripts/regen_fixtures.sh p52 p57      # plusieurs phases
#
# Prérequis : venv actif (transformers 5.9.0, torch 2.12.0), HF_TOKEN ou hf login, weights/model.safetensors,
#            HF_HOME=/data/hf_cache. Les scripts GPU (46 gen_long, 33 gen_vq_measure) basculent en cuda auto.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python3}"
export HF_HOME="${HF_HOME:-/data/hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

ok=0; fail=0; failed_list=()
run() {  # run <label> <script>
  local label="$1" script="$2"
  if [ ! -f "scripts/$script" ]; then echo "  [SKIP] $label : $script absent"; return; fi
  printf "  [%-34s] %s ... " "$label" "$script"
  if $PY "scripts/$script" >/tmp/regen_$$.log 2>&1; then
    echo "PASS"; ok=$((ok+1))
  else
    echo "FAIL (cf /tmp/regen_$$.log)"; fail=$((fail+1)); failed_list+=("$script")
  fi
}

# Phases (ordre des dépendances). Les labels font foi dans PLANNING/ENGINE_LOG.
phase_ple() {
  run "P4-prep contract PLE"        02_contract_ple.py
  run "P3 PLE reference"            03_ple_reference.py
  run "P4-prep PLE fixture export"  06_export_ple_fixture.py
  run "P4.4.0 safetensors fixture"   08_export_safetensors_fixture.py
  run "P4-prep PLE raw pytorch"     05_ple_raw_pytorch.py
  run "P4.3 selfcheck"              07_fixture_selfcheck.py
}
phase_yoco() {
  run "P5.0 YOCO config map"        09_yoco_config_map.py
  run "P5.0 YOCO weight map"        10_yoco_weight_map.py
  run "P5.1 YOCO policy table"      13_yoco_policy_table.py
}
phase_p52() {
  run "P5.2.D0 KV oracle L13"       14_kv_oracle_layer13.py
  run "P5.2.D0 q-only reader"      14_q_only_reader_oracle.py
  for s in 15_p5_2_d1_export_fixture.py 16_p5_2_d2_export_fixture.py 17_p5_2_d3_export_fixture.py \
           18_p5_2_d4_export_fixture.py 19_p5_2_d5_export_fixture.py 20_p5_2_d2b_export_fixture.py \
           21_attention_oracle_layer15_reader_kv13.py 22_p5_2_e1_export_fixture.py 23_p5_2_emask_oracle.py \
           24_p5_2_esoftmax_export_fixture.py 25_p5_2_econtext_export_fixture.py 26_p5_2_f_oproj_oracle.py \
           27_p5_2_g_attn_residual_oracle.py 28_p5_2_h_mlp_oracle.py; do
    run "P5.2" "$(basename "$s")"
  done
}
phase_p54_p56() {
  run "P5.4 embed oracle"           30_p5_4_embed_oracle.py
  run "P5.5 head oracle"           31_p5_5_head_oracle.py
  run "P5.3 layer oracle"          32_p5_3_layer_oracle.py
  run "P5.6 full qrope oracle"     29_p5_6_full_qrope_oracle.py
  run "P5.6.K full krope oracle"   33_p5_6k_full_krope_oracle.py
}
phase_p57() {
  run "P5.7.0 loader manifest"     34_p5_7_0_loader_manifest.py
  run "P5.7.1 load ref"            35_p5_7_1_load_ref.py
  run "P5.7.3 runtime plan"        36_p5_7_3_runtime_plan.py
  run "P5.7.4 full layer oracle"   37_p5_7_4_full_layer_oracle.py
  run "P5.7.5 prefill oracle"      38_p5_7_5_prefill_oracle.py
  run "P5.7.5 prefill oracle HYBRIDE" 39_p5_7_5_prefill_oracle_hybrid.py
}
phase_decode() {
  run "P5.7.7 decode pilot oracle" 40_p5_7_7_decode_pilot_oracle.py
  run "P5.7.7 decode prim oracle"  41_p5_7_7_decode_prim_oracle.py
  run "P5.7.7 decode2 oracle"      42_p5_7_7_decode2_oracle.py
  run "P5.7.7 decode3 oracle"     43_p5_7_7_decode3_oracle.py
  run "P5.7.8 gen oracle"         44_p5_7_8_gen_oracle.py
}
phase_tq() {
  run "TQ Task0 export constants" 30_export_turboquant_constants.py
  run "TQ Q3 vquant oracle"       31_vquant_oracle.py
  run "TQ Q4 decode_vq oracle"    32_decode_vq_oracle.py
  run "TQ Q5 cost measure (GPU)"  33_gen_vq_measure.py
  run "TQ Q5 gen_vq oracle"       45_gen_vq_oracle.py
}
phase_genlong() {
  run "GEN-LONG L0 oracle (GPU)"  46_gen_long_oracle.py
}

ALL="ple yoco p52 p54-p56 p57 decode tq genlong"
phases=("$@"); [ ${#phases[@]} -eq 0 ] && phases=($ALL)

for ph in "${phases[@]}"; do
  echo "=== Phase : $ph ==="
  case "$ph" in
    ple)      phase_ple;;
    yoco)     phase_yoco;;
    p52)      phase_p52;;
    p54-p56)  phase_p54_p56;;
    p57)      phase_p57;;
    decode)   phase_decode;;
    tq)       phase_tq;;
    genlong)  phase_genlong;;
    *) echo "  phase inconnue: $ph (valides: $ALL)";;
  esac
done

echo "==============================="
echo "PASS=$ok  FAIL=$fail"
[ $fail -gt 0 ] && { echo "Échecs : ${failed_list[*]}"; exit 1; }
echo "Toutes les fixtures régénérées."

```
### R5 — Smoke test (compile-only) : `scripts/smoke.sh`

```bash
#!/usr/bin/env bash
# smoke.sh — Test de fumée : compile (sans exécuter) les runners ZML clés sur la 3090.
#
# But (R5) : vérifier rapidement que la toolchain ZML + les sources compilent, SANS avoir besoin des
# weights ni des fixtures ni d'un run (le mur mémoire n'est pas touché en build-only). C'est la
# vérification minimale de reproductibilité après un changement de source (ex: les livrables R1/R2).
#
# Inclut les runners clés : non-régression (E1/E2), gen-long (gchunk/vacuity/ring/auto), GPU (G1/bench).
#   - gemma4_engine_e1   (socle mono, config par défaut — la base de preuve HLO)
#   - gemma4_engine_e2   (brique TurboQuant)
#   - gemma4_gchunk      (L1a chunké + instrumentation mémoire R1)
#   - gemma4_gchunk_vacuity (contre-test non-vacuité R2)
#
# Usage (sur la 3090, depuis le workspace ZML) :
#   bash /data/gemma4-zml-probe/scripts/smoke.sh
#   SMOKE_TARGETS="gemma4_engine_e1 gemma4_gchunk_vacuity" bash scripts/smoke.sh   # sous-ensemble
set -uo pipefail

ZML_WS="${ZML_WS:-/data/rqz_workspace/zml}"
BAZEL="./bazel.sh"
TARGETS=${SMOKE_TARGETS:-"gemma4_engine_e1 gemma4_engine_e2 gemma4_gchunk gemma4_gchunk_vacuity gemma4_gchunk_ring gemma4_gchunk_auto gemma4_gen_long_gpu gemma4_bench"}

cd "$ZML_WS" || { echo "ERR: workspace ZML introuvable: $ZML_WS"; exit 1; }
# Prérequis swap (OOM compile, cf GENERATION_LONGUE_PLAN conventions) :
if swapon --show 2>/dev/null | grep -q .; then :; else
  echo "WARN: aucun swap actif — le compile XLA-CPU peut OOM-killer (exit 255). Vérifier /swapfile_xla."
fi

ok=0; fail=0
for t in $TARGETS; do
  printf "  [build %-26s] " "//examples/rqz:$t"
  if $BAZEL build "//examples/rqz:$t" >/tmp/smoke_$t.log 2>&1; then
    echo "OK"; ok=$((ok+1))
  else
    echo "FAIL (cf /tmp/smoke_$t.log)"; fail=$((fail+1))
  fi
done
echo "==============================="
echo "BUILD OK=$ok  FAIL=$fail"
[ $fail -eq 0 ] && echo "Smoke OK — sources + toolchain compilent (runners non exécutés)."
exit $fail

```
### R6 — Réconciliation docs

- `README.md` : statut clarifié (decode 4-tokens vs gen-longue 1020) + Limites (L_MAX 1024, swapfile, perf 55 min, non-vacuité pending).
- `docs/GENERATION_LONGUE_DESIGN.md` : D3 (2048 cible / 1024 implémenté) + §7 mémoire corrigé (pic = poids f32, pas le cache).
- `docs/GENERATION_LONGUE_PLAN.md` : bandeau STATUT + checklist (Task 0/L0/L1a cochés par commit ; non-vacuité L1a ouverte).
- `docs/ENGINE_LOG.md` : append « Travaux préparatoires R1-R6 » (prêt-à-valider).

---

## 4. Livrables L1b (ring 512) + L2 (autonome host)

### L1b — Oracle ring + masque circulaire : `scripts/47_gen_long_ring_oracle.py`

```python
"""L1b — Oracle ring-buffer 512 + masque circulaire (fixture, SANS ré-exécuter HF).

Construit la fixture L1b (`gen_long_ring.safetensors`) en RÉUTILISANT la fixture L0 (`gen_long.safetensors`)
pour tout ce qui ne dépend pas du token (expected, fed, embeds, embptls, cos_full, sin_full, positions,
cache_fl_*, cache0_full) et en RECONSTRUISANT seulement ce qui change en ring 512 :
  - `masks_sliding` : CIRCULAIRE, shape {N,1,1,1,KMAX_SLIDING=512} (au lieu de .k=L_MAX linéaire).
  - `cache_sl_k/v`  : re-packed à .k=512 (prefill 0..3 aux slots 0..3, reste 0).

La séquence greedy HF (expected/fed) est IDENTIQUE à L0/L1a : le ring 512 est un encodage mémoire du
MÊME attention (les 512 dernières positions visibles), donc tokens bit-identiques. L1b valide que le
moteur ZML reproduit HF avec le vrai ring + masque circulaire (gate L1b, cf GENERATION_LONGUE_PLAN).

Émet AUSSI `gen_long_ring_naive.safetensors` (mêmes tensors sauf `masks_sliding` NAIVE = bande non
remappée sur le ring) pour le CONTRE-TEST de non-vacuité L1b (PLAN step 277) : le runner L1b sur cette
fixture NAIVE doit DIVERGER à partir de p≈512 (la bande dépasse le ring → slots masqués à tort).

CLI : python3 scripts/47_gen_long_ring_oracle.py   (3090 ; lit gen_long.safetensors, n'a PAS besoin de GPU/HF)
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
L0 = ROOT / "gen_long.safetensors"
OUT_RING = ROOT / "gen_long_ring.safetensors"
OUT_NAIVE = ROOT / "gen_long_ring_naive.safetensors"
MANIFEST = ROOT / "gen_long_ring_manifest.json"

SEQ_LEN = 4
L_MAX = 1024
SLIDING_WINDOW = 512
KMAX_SLIDING = 512  # ring
N_DECODE = L_MAX - SEQ_LEN  # 1020
MIN = torch.finfo(torch.float32).min
HD_S = 256


def j_max(s: int, p: int) -> int | None:
    """Position la plus récente écrite au slot s du ring (mod 512) pour la position courante p, ou None."""
    if s > p:  # slot jamais écrit (aucune position ≤ p congrue à s mod 512 car s≤511 et s>p)
        return None
    k = (p - s) // 512
    return s + 512 * k


def circular_mask(p: int) -> torch.Tensor:
    """Masque circulaire {KMAX_SLIDING} : slot s visible ssa la position qu'il contient est dans la bande."""
    m = torch.full((KMAX_SLIDING,), MIN, dtype=torch.float32)
    lo = max(0, p - (SLIDING_WINDOW - 1))
    for s in range(KMAX_SLIDING):
        j = j_max(s, p)
        if j is not None and lo <= j <= p:
            m[s] = 0.0
    return m


def naive_mask(p: int) -> torch.Tensor:
    """Contre-test : bande non-remappée (slot index ≡ position). Diverge pour p≥512 (bande déborde le ring)."""
    m = torch.full((KMAX_SLIDING,), MIN, dtype=torch.float32)
    lo = max(0, p - (SLIDING_WINDOW - 1))
    hi = min(p, KMAX_SLIDING - 1)
    for s in range(KMAX_SLIDING):
        if lo <= s <= hi:
            m[s] = 0.0
    return m


def main() -> None:
    assert L0.exists(), f"L0 fixture manquante : {L0} (lancer scripts/46_gen_long_oracle.py d'abord)"
    print(f"Lecture L0 : {L0}")

    # Reuse tout sauf masks_sliding + cache_sl_k/v (reconstruits).
    reuse_names = [
        "embeds", "embptls", "cos_full", "sin_full", "positions",
        "masks_full", "cache_fl_k", "cache_fl_v", "expected", "fed",
    ]
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(L0), framework="pt") as s:
        # cache_sl originals (linéaire .k=L_MAX) pour re-pack
        sl_k_lin = s.get_tensor("cache_sl_k")  # [n_slots,1,1,L_MAX,HD_S]
        sl_v_lin = s.get_tensor("cache_sl_v")
        for n in reuse_names:
            tensors[n] = s.get_tensor(n).clone()

    n = tensors["expected"].shape[0]
    assert n == N_DECODE, f"expected len {n} != {N_DECODE}"
    positions = tensors["positions"]
    print(f"N={n}, positions {positions[0].item()}..{positions[-1].item()}, KMAX_SLIDING={KMAX_SLIDING}")

    # masks_sliding circulaire (correct) + naive (contre-test), shape {N,1,1,1,KMAX_SLIDING}.
    masks_circ = torch.zeros(n, 1, 1, 1, KMAX_SLIDING, dtype=torch.float32)
    masks_naiv = torch.zeros(n, 1, 1, 1, KMAX_SLIDING, dtype=torch.float32)
    for k in range(n):
        p = int(positions[k].item())
        masks_circ[k, 0, 0, 0, :] = circular_mask(p)
        masks_naiv[k, 0, 0, 0, :] = naive_mask(p)

    # cache_sl re-packed ring .k=512 : prefill positions 0..SEQ_LEN-1 aux slots 0..3, reste 0.
    n_slots = sl_k_lin.shape[0]
    cache_sl_k = torch.zeros(n_slots, 1, 1, KMAX_SLIDING, HD_S, dtype=torch.float32)
    cache_sl_v = torch.zeros_like(cache_sl_k)
    cache_sl_k[:, :, :, :SEQ_LEN, :] = sl_k_lin[:, :, :, :SEQ_LEN, :].float()
    cache_sl_v[:, :, :, :SEQ_LEN, :] = sl_v_lin[:, :, :, :SEQ_LEN, :].float()

    # Vérif : pour p<512, circulaire == naive (sanity) ; pour p>=512, ils diffèrent.
    for k in (0, 5, 100):
        p = int(positions[k].item())
        assert torch.equal(masks_circ[k], masks_naiv[k]), f"p<{SLIDING_WINDOW} devraient coïncider (p={p})"
    k_512 = int((512 - SEQ_LEN))  # p=512
    assert not torch.equal(masks_circ[k_512], masks_naiv[k_512]), "p=512 : circulaire vs naive doivent différer"

    # === Fixture L1b (correcte) ===
    tensors_ring = dict(tensors)
    tensors_ring["masks_sliding"] = masks_circ.contiguous()
    tensors_ring["cache_sl_k"] = cache_sl_k.contiguous()
    tensors_ring["cache_sl_v"] = cache_sl_v.contiguous()
    for k, t in tensors_ring.items():
        assert not torch.isnan(t.float()).any(), f"NaN dans {k}"
    save_file(tensors_ring, str(OUT_RING))
    print("wrote", OUT_RING)

    # === Fixture contre-test (naive) : mêmes tensors sauf masks_sliding naive ===
    tensors_naive = dict(tensors)
    tensors_naive["masks_sliding"] = masks_naiv.contiguous()
    tensors_naive["cache_sl_k"] = cache_sl_k.contiguous()
    tensors_naive["cache_sl_v"] = cache_sl_v.contiguous()
    save_file(tensors_naive, str(OUT_NAIVE))
    print("wrote", OUT_NAIVE, "(contre-test : masque non-remappé)")

    manifest = {
        "source": "L1b oracle ring 512 + masque circulaire (reuse L0, rebuild masks_sliding + cache_sl ring)",
        "l0_fixture": str(L0.name), "seq_len": SEQ_LEN, "n_decode": N_DECODE, "l_max": L_MAX,
        "kmax_sliding": KMAX_SLIDING, "kmax_full": L_MAX, "sliding_window": SLIDING_WINDOW,
        "fixtures": {
            "gen_long_ring.safetensors": "L1b correct (masque circulaire) — runner attend PASS argmax==HF",
            "gen_long_ring_naive.safetensors": "contre-test non-vacuité (masque non-remappé) — runner attend DIVERGENCE ~p=512",
        },
        "tensors": {n_: {"shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", "")}
                    for n_, t in tensors_ring.items()},
        "pass_criterion_L1b": "argmax ZML[k] == expected[k] pour tout k (ring 512 + masque circulaire)",
        "counter_test_L1b": "runner sur gen_long_ring_naive.safetensors doit DIVERGER à partir de p≈512",
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print("wrote", MANIFEST, "\nL1b oracle ring OK.")


if __name__ == "__main__":
    main()

```
### L1b — Runner ring (chunké) : `zml_runner/gemma4_gchunk_ring.zig`

```zig
// L1b — Replay génération longue : VRAI ring-buffer 512 + masque CIRCULAIRE (replay, chunké).
//
// Configuration : EngineModel(struct{}, .{ .ring=true, .two_masks=true, .kmax_sliding=512,
//                                        .kmax_full=L_MAX }). Différences vs L1a (gemma4_gchunk) :
//   - ring=true  → scatter sliding CIRCULAIRE à `pos % 512` (au lieu de `pos` linéaire).
//   - kmax_sliding=512 → cache sliding `.k=512` (anneau), masque `masks_sliding` CIRCULAIRE (.k=512).
//   - kmax_full=L_MAX   → cache full reste LINÉAIRE `.k=L_MAX` (les couches full ne sont JAMAIS fenêtrées).
//
// La séquence greedy HF est IDENTIQUE à L1a (le ring 512 encode le même attention des 512 dernières
// positions) → PASS = argmax == expected sur les N tokens, y compris APRÈS le wrap (pos ≥ 512).
//
// Contre-test de non-vacuité (PLAN L1b step 277) : lancer ce runner sur `gen_long_ring_naive.safetensors`
// (masque non-remappé) → doit DIVERGER à partir de p≈512 (la bande déborde le ring, slots masqués à tort).
//
// Chemin d'exécution chunké (== gchunk) pour éviter le thrash mémoire du mono-graphe à `.k≥512`.
// Inclut l'instrumentation RSS (R1, mem_probe.zig).
//
// CLI : gemma4_gchunk_ring <model.safetensors> <gen_long_ring.safetensors|gen_long_ring_naive.safetensors>

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
const RSS_EVERY: usize = 64;
const B: i64 = 1;
const S: i64 = 1;
const D: i64 = 1536;

// L1b : ring 512 + masque circulaire. kmax_full reste L_MAX (full jamais fenêtré).
const Model = engine.EngineModel(struct {}, .{ .ring = true, .two_masks = true, .kmax_sliding = 512, .kmax_full = L_MAX });
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

const ExpW = struct {
    e: zml.Tensor,
    pub fn init(v: zml.io.TensorStore.View) ExpW {
        return .{ .e = v.createTensor("expected", .{.step}, null) };
    }
    pub fn load(self: *const ExpW, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, store: *const zml.io.TensorStore, shardings: []const zml.sharding.Sharding) !zml.Bufferized(ExpW) {
        return zml.io.load(ExpW, self, allocator, io, platform, store, .{ .shardings = shardings, .parallelism = 1, .dma_chunks = 1, .dma_chunk_size = 16 * 1024 * 1024 });
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

pub fn main(init: std.process.Init) !void {
    @setEvalBranchQuota(200000);
    const arena = init.arena;
    const allocator = init.gpa;
    const io = init.io;

    const process_args = try init.minimal.args.toSlice(arena.allocator());
    if (process_args.len < 3) {
        log.err("Usage: gemma4_gchunk_ring <model.safetensors> <gen_long_ring.safetensors|..._naive.safetensors>", .{});
        return error.MissingArgument;
    }
    const ckpt = process_args[1];
    const fixture = process_args[2];
    const is_naive = std.mem.indexOf(u8, fixture, "naive") != null;
    log.info("L1b — ring 512 + masque circulaire (chunké) ; fixture={s}{s}", .{ fixture, if (is_naive) " [NAIVE → attend divergence ~p=512]" else "" });

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

    const eng_buf = try model.load(arena.allocator(), io, platform, &store_ck, &.{sharding});
    const pk_buf = try packed_in.load(arena.allocator(), io, platform, &store_fx, &.{sharding});
    var cache_buf = try cache0.load(arena.allocator(), io, platform, &store_fx, &.{sharding});
    const exp_sym: ExpW = .init(store_fx.view());
    var exp_buf = try exp_sym.load(arena.allocator(), io, platform, &store_fx, &.{sharding});
    store_ck.deinit();
    reg_ck.deinit();
    mem_probe.logMem("post-load");

    var exp_slice = try exp_buf.e.toSliceAlloc(allocator, io);
    defer exp_slice.free(allocator);
    const expected_tokens = exp_slice.items(i32);
    const num_steps = expected_tokens.len;

    const hidden_shape = zml.Shape.init(.{ B, S, D }, .f32).withTags(.{ .b, .s, .d });
    const zeros = try arena.allocator().alloc(u8, @intCast(B * S * D * 4));
    @memset(zeros, 0);
    var dummy_hidden = try zml.Buffer.fromBytes(io, platform, hidden_shape, sharding, zeros);
    defer dummy_hidden.deinit();

    log.info("Compiling {d} stages (ring=true, kmax_sliding=512)...", .{N_STAGES});
    var exes: [N_STAGES]zml.Exe = undefined;
    inline for (STAGES, 0..) |stage, si| {
        const F = struct {
            fn f(m: Model, p: PackedLong, ca: engine.Cache, h: zml.Tensor, ct: engine.Ctrl) StageOut {
                return m.forwardStageGen(stage.start, stage.end, stage.first, stage.last, p, ca, h, ct);
            }
        }.f;
        exes[si] = try platform.compileFn(allocator, io, F, .{ model, packed_in, cache0, hidden_sym, ctrl_sym }, .{ .shardings = &.{sharding} });
    }
    defer for (&exes) |*e| e.deinit();
    mem_probe.logMem("post-compile (go/no-go)");

    var all_pass = true;
    var n_match: usize = 0;
    var first_fail: i64 = -1;
    var step_idx: usize = 0;
    const rss0 = mem_probe.rssKb();
    while (step_idx < num_steps) : (step_idx += 1) {
        var step_buf = try zml.Buffer.scalar(io, platform, @as(u32, @intCast(step_idx)), .u32, sharding);
        const ctrl_buf = zml.Bufferized(engine.Ctrl){ .step = step_buf };

        var hidden_buf = dummy_hidden;
        var tok: i64 = -1;
        inline for (STAGES, 0..) |stage, si| {
            var args = try exes[si].args(allocator);
            var results = try exes[si].results(allocator);
            args.set(.{ eng_buf, pk_buf, cache_buf, hidden_buf, ctrl_buf });
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

        const exp = @as(i64, @intCast(expected_tokens[step_idx]));
        const ok = tok == exp;
        if (ok) n_match += 1 else {
            all_pass = false;
            if (first_fail < 0) first_fail = @intCast(step_idx);
            if (first_fail >= 0 and (step_idx - @as(usize, @intCast(first_fail)) < 8)) {
                log.info("  DIVERGENCE step {d} (pos {d}) : ZML={d} HF={d}", .{ step_idx, 4 + step_idx, tok, exp });
            }
        }
        if ((step_idx + 1) % 256 == 0) log.info("  ... {d}/{d} steps, {d} match", .{ step_idx + 1, num_steps, n_match });
        if ((step_idx % RSS_EVERY == RSS_EVERY - 1) and (rss0 != null)) {
            var tag_buf: [32]u8 = undefined;
            const tag = std.fmt.bufPrint(&tag_buf, "step {d}", .{step_idx}) catch "step";
            mem_probe.logMem(tag);
        }
        step_buf.deinit();
    }
    cache_buf.sl_k.deinit();
    cache_buf.sl_v.deinit();
    cache_buf.fl_k.deinit();
    cache_buf.fl_v.deinit();
    mem_probe.logMem("post-run");

    log.info("L1b RING : {d}/{d} tokens match (first_fail step {d})", .{ n_match, num_steps, first_fail });
    if (is_naive) {
        // Contre-test : le masque non-remappé doit DIVERGER à partir de p≈512 (pos 512 = step ~508).
        const ff_pos: i64 = if (first_fail >= 0) 4 + first_fail else -1;
        if (n_match < num_steps) {
            log.info("L1b NON-VACUITÉ RING OK — divergence (1re au pos {d}) prouve le wrap circulaire consommé", .{ff_pos});
            if (ff_pos >= 0 and ff_pos < 508) log.warn("  1re divergence précoce (pos {d}<508) : investiguer", .{ff_pos});
        } else {
            log.err("L1b NON-VACUITÉ RING FAIL : aucune divergence malgré masque non-remappé (wrap non consommé !)", .{});
            return error.Vacuity;
        }
    } else {
        if (all_pass) {
            log.info("L1b RING PASS — {d} tokens == HF greedy (ring 512 + masque circulaire, wrap franchi)", .{num_steps});
        } else {
            log.err("L1b RING : divergence vs expected (1re au step {d}) — ring/masque circulaire à investiguer", .{first_fail});
            return error.GenMismatch;
        }
    }
}

```
### L2 — engine.zig : `forwardStep` + `forwardStageStep` (entrées autonomes)

Ajoutés à `EngineModel` (les 4 forward existants INTACTS → E1/E2/L1a inchangés). Ci-dessous l'extrait
ajouté (les deux nouvelles méthodes) :

```zig
        /// L2 — forward 1-step AUTONOME (host-orchestré) : les embeds/embptls viennent d'un gather HOST
        /// du token produit (token-dépendant), tandis que cos/sin/masques/positions viennent de `p` (la
        /// fixture L1a : position-only, INDÉPENDANTS du token → valides pour la génération autonome tant
        /// que les positions coïncident, i.e. même prompt+compte). `forward` mono (E1/E2) et `forwardStageGen`
        /// (chunké) sont INTACTS → preuve HLO et L1a inchangés ; cette méthode est une NOUVELLE entrée.
        ///
        /// `embeds_step` : {b,s,d} bf16 — embed_tokens[fed_tok] host-gathered (AVANT scale √1536, brut).
        /// `embptls_step` : {b,s,lf} bf16 — embed_tokens_per_layer[fed_tok] host-gathered.
        /// Retourne {logits, sl_k, sl_v, fl_k, fl_v} (== `forward` mono, op-pour-op identique hormis la
        /// source des embeds/embptls). Permet la boucle autonome : argmax → gather host → reinject.
        pub fn forwardStep(self: Self, embeds_step: zml.Tensor, embptls_step: zml.Tensor, p: Packed(cfg.two_masks), cache_in: Cache, ctrl: Ctrl) struct { zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor } {
            const c = struct { fn call(t: zml.Tensor) zml.Tensor { return t.convert(prec.compute); } }.call;
            const step = ctrl.step;
            const cos = pickStep(p.cos_full, step);
            const sin = pickStep(p.sin_full, step);
            const mask_single = if (cfg.two_masks) {} else pickStep(p.masks, step);
            const mask_sliding = if (cfg.two_masks) pickStep(p.masks_sliding, step) else {};
            const mask_full = if (cfg.two_masks) pickStep(p.masks_full, step) else {};
            const pos_i = pickStep(p.positions, step);
            const pos_s = pos_i.reshape(.{1}).withTags(.{.s});
            const pos_u = pos_i.convert(.u32);

            const embeds = embeds_step.convert(.f32).scale(EMBED_SCALE);
            const ple = self.perLayerInputs(embptls_step, embeds);
            var hidden = embeds;
            var cache = cache_in;
            inline for (0..NUM_LAYERS) |i| {
                const ple_i = ple.choose1d(.layer, @as(i64, @intCast(i)));
                const mask = if (cfg.two_masks)
                    (if (comptime isFull(i)) mask_full else mask_sliding)
                else
                    mask_single;
                hidden = runLayerGen(self.layers[i], i, cfg, prec, hidden, ple_i, cos, sin, mask, pos_s, pos_u, &cache, self.brick);
            }

            const last_hidden = rmsScaleD(hidden, c(self.final_norm));
            const raw = last_hidden.dot(c(self.embed_tokens), .d);
            const logits = raw.scale(INV_SOFTCAP).tanh().scale(SOFTCAP);
            return .{ logits, cache.sl_k, cache.sl_v, cache.fl_k, cache.fl_v };
        }

        /// L2 CHUNKÉ — variante stage de `forwardStep` (autonome host-orchestré, chunké pour la mémoire).
        /// Comme `forwardStageGen` MAIS embeds/embptls viennent d'un gather HOST per-step (token-dépendant)
        /// au lieu de `pickStep(p.embeds/embptls)`. cos/sin/masques/positions restent de `p` (position-only).
        /// `first` → hidden = embeds_step ; `last` → final norm + lm_head + softcap. Même `runLayerGen`.
        /// Nécessaire car le mono `forwardStep` compile le graphe 35-couches (~33 Go, thrash) : le chunké
        /// borne le pic (cf GENERATION_LONGUE_CHUNKING_DESIGN). `forward`/`forwardStageGen`/`forwardStep`
        /// (E1/E2/L1a) sont INTACTS.
        pub fn forwardStageStep(self: Self, comptime start: usize, comptime end: usize, comptime first: bool, comptime last: bool, embeds_step: zml.Tensor, embptls_step: zml.Tensor, p: Packed(cfg.two_masks), cache_in: Cache, hidden_in: zml.Tensor, ctrl: Ctrl) struct { zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor } {
            const c = struct { fn call(t: zml.Tensor) zml.Tensor { return t.convert(prec.compute); } }.call;
            const step = ctrl.step;
            const cos = pickStep(p.cos_full, step);
            const sin = pickStep(p.sin_full, step);
            const mask_single = if (cfg.two_masks) {} else pickStep(p.masks, step);
            const mask_sliding = if (cfg.two_masks) pickStep(p.masks_sliding, step) else {};
            const mask_full = if (cfg.two_masks) pickStep(p.masks_full, step) else {};
            const pos_i = pickStep(p.positions, step);
            const pos_s = pos_i.reshape(.{1}).withTags(.{.s});
            const pos_u = pos_i.convert(.u32);

            const embeds = embeds_step.convert(.f32).scale(EMBED_SCALE);
            const ple = self.perLayerInputs(embptls_step, embeds);
            var hidden = if (first) embeds else hidden_in;
            var cache = cache_in;
            inline for (start..end) |i| {
                const ple_i = ple.choose1d(.layer, @as(i64, @intCast(i)));
                const mask = if (cfg.two_masks)
                    (if (comptime isFull(i)) mask_full else mask_sliding)
                else
                    mask_single;
                hidden = runLayerGen(self.layers[i], i, cfg, prec, hidden, ple_i, cos, sin, mask, pos_s, pos_u, &cache, self.brick);
            }
            const out_first = if (last) blk: {
                const last_hidden = rmsScaleD(hidden, c(self.final_norm));
                const raw = last_hidden.dot(c(self.embed_tokens), .d);
                break :blk raw.scale(INV_SOFTCAP).tanh().scale(SOFTCAP);
            } else hidden;
            return .{ out_first, cache.sl_k, cache.sl_v, cache.fl_k, cache.fl_v };
        }
    };
}
```
### L2 — Runner autonome host-orchestré : `zml_runner/gemma4_gchunk_auto.zig`

```zig
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

```
### engine.zig — knob précision `PrecCfg` (neutre, préparation G2)

Extrait des ajouts (struct PrecCfg + signature EngineModel + local c prec-aware + threading) :

```zig
/// Config de précision comptime (GPU). Défaut `.f32` strictement == comportement actuel (fp32 bit-exact
/// baseline) : `c()` upcast en `prec.compute` (défaut .f32 = today) → graphe HLO byte-identique (E1/E2/L1a
/// inchangés, preuve `diff -rq` préservée). G2 activera `.bf16` pour les GEMM (refactor à part : insérer des
/// `.convert(prec.gemm)` aux bornes des dot, garder norm/softmax/rope/softcap en `prec.compute`).
/// Champs `weight`/`kv` réservés (le load-dtype via createTensor n'expose pas de arg dtype ; G2 utilisera
/// une conversion post-load). NEUTRALITÉ : tout champ non default doit rester inerte en config défaut.
pub const PrecCfg = struct {
    compute: zml.DataType = .f32, // cible d'upcast de c() (norm/softmax/rope/softcap et entrées GEMM actuelles)
    weight: ?zml.DataType = null, // réservé G2 (dtype de load des poids) ; null = infer (today)
    kv: ?zml.DataType = null, // réservé G2 (dtype du cache KV) ; null = infer (today)
};


```
```zig
// signature : 3e param comptime à défaut .{} → instantiations 2-arg existantes inchangées
pub fn EngineModel(comptime Brick: type, comptime cfg: EngineCfg, comptime prec: PrecCfg = .{}) type { ... }
// dans runLayerGen (+ 4 call sites passent `prec`) :
fn runLayerGen(layer, comptime i, comptime cfg, comptime prec: PrecCfg, hidden, ...) {
    const c = struct { fn call(t: zml.Tensor) zml.Tensor { return t.convert(prec.compute); } }.call;
    ...
}
// (idem local `c` dans perLayerInputs + forward/forwardStageGen/forwardStep/forwardStageStep)
```
Neutralité : défaut `compute=.f32` → `t.convert(.f32)` == `c(t)` d'aujourd'hui → HLO byte-identique.

---

## 5. Plan GPU + livrables P-GPU-1

### `docs/GPU_PORT_PLAN.md` (434 lignes, 16 sections) — référencé, non reproduit ici

Insight central : **le portage GPU n'est pas une réécriture** — `engine.zig` compile en graphe XLA
device-agnostic, `Platform.auto` sélectionne déjà CUDA avant CPU, les op sont standard. Le chantier =
backend + précision + perf + re-validation (gates G0–G8, ~12–20 j-h). E2B tient sur 1×3090 24 Go en bf16
(~10 Go poids + ~40 Mo KV @2048) → le mur CPU ~33 Go disparaît. Cibles : 3090 ≥30 tok/s, A100 ≥150 tok/s.

### P-GPU-1 — Runner G1 (baseline fp32 GPU) : `zml_runner/gemma4_gen_long_gpu.zig`

```zig
// G1 — Baseline fp32 sur GPU (P-GPU-1, cf docs/GPU_PORT_PLAN.md §10).
//
// Le moteur `engine.zig` est device-agnostic : le MÊME graphe XLA tourne sur CPU ou GPU. Ce runner force
// le backend CUDA (avec fallback auto) et AJOUTE un timer tok/s + logging platform/RSS. Le calcul est
// STRICTEMENT identique à `gemma4_gen_long.zig` (L1a mono) : `EngineModel(struct{}, .{...})` SANS prec
// (PrecCfg par défaut = fp32 = today). → G1 = "le moteur L1a, mais sur GPU", pour mesurer le gain brut du
// backend natif et valider que l'argmax == HF tient en fp32-CUDA (drift Eigen→CUDA caractérisé au G1).
//
// Critère G1 : argmax == HF sur les N tokens (séquence == L1a CPU == HF greedy) ; drift logits vs
// baseline CPU-L1a à reporter. Perf : tok/s (decode batch-1). Le chunking n'est PAS utilisé (le mur
// mémoire CPU ~33 Go disparaît sur GPU, cf GPU_PORT_PLAN §6) → mono-graphe direct.
//
// CLI : gemma4_gen_long_gpu <model.safetensors> <gen_long.safetensors> [max_steps]
// Prérequis : libpjrt_cuda linké (cf GPU_PORT_PLAN §12) ; `nvidia-smi` pour la VRAM.

const std = @import("std");
const log = std.log;
const zml = @import("zml");
const engine = @import("engine.zig");
const mem_probe = @import("mem_probe.zig");

pub const std_options: std.Options = .{ .log_level = .info };

const L_MAX: i64 = 1024;
const B: i64 = 1;
const S: i64 = 1;
const D: i64 = 1536;
// G1 : fp32 (PrecCfg défaut) — on n'active PAS le bf16 (c'est G2).
const Model = engine.EngineModel(struct {}, .{ .two_masks = true, .kmax_sliding = L_MAX, .kmax_full = L_MAX });
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

pub fn main(init: std.process.Init) !void {
    const arena = init.arena;
    const allocator = init.gpa;
    const io = init.io;

    const process_args = try init.minimal.args.toSlice(arena.allocator());
    if (process_args.len < 3) {
        log.err("Usage: gemma4_gen_long_gpu <model.safetensors> <gen_long.safetensors> [max_steps]", .{});
        return error.MissingArgument;
    }
    const ckpt = process_args[1];
    const fixture = process_args[2];
    const max_steps: ?usize = if (process_args.len >= 4) std.fmt.parseInt(usize, process_args[3], 10) catch null else null;

    // === Backend : force CUDA (memory_fraction 0.90), fallback auto (CPU) si CUDA indisponible. ===
    const platform: *zml.Platform = blk: {
        const cuda_opts: zml.Platform.CreateOptions = .{
            .cuda = .{ .allocator = .{ .bfc = .{ .preallocate = true, .memory_fraction = 0.90 } } },
        };
        if (zml.Platform.init(allocator, io, .cuda, cuda_opts)) |p| break :blk p else |_| {}
        log.warn("CUDA indisponible (libpjrt_cuda absent ?) — repli sur Platform.auto (probablement CPU).", .{});
        break :blk try zml.Platform.auto(allocator, io, .{});
    };
    defer platform.deinit(allocator);
    log.info("G1 — backend = {s} (cible : cuda). Prérequis : libpjrt_cuda linké ; VRAM via nvidia-smi.", .{@tagName(platform.target)});
    const sharding = try zml.sharding.replicatedSharding(platform);

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

    log.info("Materializing weights + packed inputs + caches (L_MAX={d}) ...", .{L_MAX});
    const eng_buf = try model.load(arena.allocator(), io, platform, &store_ck, &.{sharding});
    const pk_buf = try packed_in.load(arena.allocator(), io, platform, &store_fx, &.{sharding});
    var cache_buf = try cache0.load(arena.allocator(), io, platform, &store_fx, &.{sharding});
    const exp_sym: ExpW = .init(store_fx.view());
    var exp_buf = try exp_sym.load(arena.allocator(), io, platform, &store_fx, &.{sharding});
    store_ck.deinit();
    reg_ck.deinit();
    mem_probe.logMem("post-load (host RSS ; VRAM via nvidia-smi)");

    var exp_slice = try exp_buf.e.toSliceAlloc(allocator, io);
    defer exp_slice.free(allocator);
    const expected_tokens = exp_slice.items(i32);
    const total = expected_tokens.len;
    const num_steps = if (max_steps) |m| @min(m, total) else total;
    log.info("NUM_STEPS = {d} (max_steps={?d})", .{ num_steps, max_steps });

    log.info("Compiling gen step (mono-graphe 35 couches, fp32) ...", .{});
    const t_compile: std.Io.Timestamp = .now(io, .awake);
    var exe = try platform.compile(allocator, io, model, .forward, .{ packed_in, cache0, ctrl_sym }, .{ .shardings = &.{sharding} });
    defer exe.deinit();
    log.info("  compile: {f}", .{t_compile.untilNow(io, .awake)});
    mem_probe.logMem("post-compile (host RSS ; go/no-go GPU = nvidia-smi)");

    var all_pass = true;
    var n_match: usize = 0;
    var first_fail: i64 = -1;
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

        const tok = try argmaxOf(allocator, io, &r_logits);
        const exp = @as(i64, @intCast(expected_tokens[step_idx]));
        const ok = tok == exp;
        if (ok) n_match += 1 else {
            all_pass = false;
            if (first_fail < 0) first_fail = @intCast(step_idx);
            if (step_idx - @as(usize, @intCast(@max(first_fail, 0))) < 8) {
                log.err("  FAIL step {d} (pos {d}) : argmax ZML={d} HF={d}", .{ step_idx, 4 + step_idx, tok, exp });
            }
        }
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
    const ms_per_tok = if (num_steps > 0) @as(f64, @floatFromInt(elapsed_ns)) / 1e6 / @as(f64, @floatFromInt(num_steps)) else 0;
    log.info("G1 PERF : {d} tokens en {d:.2}s → {d:.1} tok/s ({d:.1} ms/tok) [backend={s}, fp32, batch-1, mono-graphe]", .{ num_steps, elapsed_s, tok_per_s, ms_per_tok, @tagName(platform.target) });
    mem_probe.logMem("post-run (host RSS ; VRAM GPU via nvidia-smi)");

    log.info("G1 : {d}/{d} tokens argmax-match (vs HF)", .{ n_match, num_steps });
    if (all_pass) {
        log.info("G1 PASS — fp32-{s} reproduit HF greedy ({d} tokens) ; baseline GPU établi.", .{ @tagName(platform.target), num_steps });
    } else {
        log.err("G1 : divergence (1er fail step {d}, {d} match) — drift fp32-CUDA > tol ? (cf GPU_PORT_PLAN §10.1)", .{ first_fail, n_match });
        return error.GenMismatch;
    }
}

```
### P-GPU-1 — Benchmark decode : `zml_runner/gemma4_bench.zig`

```zig
// gemma4_bench — Benchmark decode GPU (P-GPU-1/G8, cf docs/GPU_PORT_PLAN.md §9.7).
//
// Mesure le débit decode batch-1 (tok/s) du moteur ZML sur le backend sélectionné (CUDA si dispo).
// Réutilise la fixture L1a (gen_long.safetensors) : les cos/sin/masques/positions/expected sont
// position-only (indépendants du backend) → le bench est reproductible et compare CPU vs GPU à
// calcul identique. Warmup (1 step, cold-start CUDA/cuBLAS écarté) puis timing sur N steps.
//
// Métriques : compile (ms), warmup, tok/s, ms/tok, pic RSS host, platform. VRAM GPU via `nvidia-smi`
// (le moteur n'expose pas un compteur VRAM portable ; TODO G8 : lire via PJRT device memory).
// Sanity : argmax == HF (si divergence, le bench mesure du bruit → alerte).
//
// CLI : gemma4_bench <model.safetensors> <gen_long.safetensors> [max_steps]
// Ex : bazel run //examples/rqz:gemma4_bench -- <ckpt> gen_long.safetensors 256

const std = @import("std");
const log = std.log;
const zml = @import("zml");
const engine = @import("engine.zig");
const mem_probe = @import("mem_probe.zig");

pub const std_options: std.Options = .{ .log_level = .info };

const L_MAX: i64 = 1024;
const B: i64 = 1;
const S: i64 = 1;
const D: i64 = 1536;
const Model = engine.EngineModel(struct {}, .{ .two_masks = true, .kmax_sliding = L_MAX, .kmax_full = L_MAX });
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

pub fn main(init: std.process.Init) !void {
    const arena = init.arena;
    const allocator = init.gpa;
    const io = init.io;

    const process_args = try init.minimal.args.toSlice(arena.allocator());
    if (process_args.len < 3) {
        log.err("Usage: gemma4_bench <model.safetensors> <gen_long.safetensors> [max_steps]", .{});
        return error.MissingArgument;
    }
    const ckpt = process_args[1];
    const fixture = process_args[2];
    const max_steps: ?usize = if (process_args.len >= 4) std.fmt.parseInt(usize, process_args[3], 10) catch null else null;

    const platform: *zml.Platform = blk: {
        const cuda_opts: zml.Platform.CreateOptions = .{
            .cuda = .{ .allocator = .{ .bfc = .{ .preallocate = true, .memory_fraction = 0.90 } } },
        };
        if (zml.Platform.init(allocator, io, .cuda, cuda_opts)) |p| break :blk p else |_| {}
        log.warn("CUDA indisponible — repli sur Platform.auto.", .{});
        break :blk try zml.Platform.auto(allocator, io, .{});
    };
    defer platform.deinit(allocator);
    const sharding = try zml.sharding.replicatedSharding(platform);

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

    const eng_buf = try model.load(arena.allocator(), io, platform, &store_ck, &.{sharding});
    const pk_buf = try packed_in.load(arena.allocator(), io, platform, &store_fx, &.{sharding});
    var cache_buf = try cache0.load(arena.allocator(), io, platform, &store_fx, &.{sharding});
    const exp_sym: ExpW = .init(store_fx.view());
    var exp_buf = try exp_sym.load(arena.allocator(), io, platform, &store_fx, &.{sharding});
    store_ck.deinit();
    reg_ck.deinit();

    var exp_slice = try exp_buf.e.toSliceAlloc(allocator, io);
    defer exp_slice.free(allocator);
    const expected_tokens = exp_slice.items(i32);
    const total = expected_tokens.len;
    const num_steps = if (max_steps) |m| @min(m, total) else total;
    if (num_steps < 2) { log.err("num_steps < 2 : pas assez pour warmup+mesure", .{}); return error.MissingArgument; }

    log.info("BENCH — backend={s}, fp32, batch-1, mono-graphe ; warmup=1 + measure={d} steps", .{ @tagName(platform.target), num_steps - 1 });
    const t_compile: std.Io.Timestamp = .now(io, .awake);
    var exe = try platform.compile(allocator, io, model, .forward, .{ packed_in, cache0, ctrl_sym }, .{ .shardings = &.{sharding} });
    defer exe.deinit();
    log.info("  compile: {f}", .{t_compile.untilNow(io, .awake)});
    mem_probe.logMem("post-compile");

    // run 1 step (idx 0) en warmup : cold-start cuBLAS/autotune écarté ; son cache est jeté (on recommence).
    var all_pass = true;
    var n_match: usize = 0;
    var step_idx: usize = 0;

    // helper inline pour 1 step (retourne le token, met à jour cache_buf).
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
        const tok = try argmaxOf(allocator, io, &r_logits);
        const exp = @as(i64, @intCast(expected_tokens[step_idx]));
        if (tok == exp) n_match += 1 else all_pass = false;
        cache_buf.sl_k.deinit();
        cache_buf.sl_v.deinit();
        cache_buf.fl_k.deinit();
        cache_buf.fl_v.deinit();
        cache_buf = zml.Bufferized(engine.Cache){ .sl_k = r_slk, .sl_v = r_slv, .fl_k = r_flk, .fl_v = r_flv };
        r_logits.deinit();
        step_buf.deinit();
        args.deinit(allocator);
        results.deinit(allocator);
        if (step_idx == 0) {
            // warmup done ; reset du timer pour la mesure (le cache accumule normalement).
            break;
        }
    }
    const measure_steps = num_steps - 1;
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
        const tok = try argmaxOf(allocator, io, &r_logits);
        const exp = @as(i64, @intCast(expected_tokens[step_idx]));
        if (tok == exp) n_match += 1 else all_pass = false;
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
    const tok_per_s = if (elapsed_s > 0) @as(f64, @floatFromInt(measure_steps)) / elapsed_s else 0;
    const ms_per_tok = if (measure_steps > 0) @as(f64, @floatFromInt(elapsed_ns)) / 1e6 / @as(f64, @floatFromInt(measure_steps)) else 0;
    mem_probe.logMem("post-run (host RSS ; VRAM GPU via nvidia-smi)");

    log.info("========== gemma4_bench ==========", .{});
    log.info("  backend      : {s}", .{@tagName(platform.target)});
    log.info("  precision    : fp32 (G1 baseline ; bf16 = G2/G3)", .{});
    log.info("  config       : batch-1, mono-graphe 35 couches, L_MAX={d}", .{L_MAX});
    log.info("  warmup       : 1 step (écarté)", .{});
    log.info("  measured     : {d} steps", .{measure_steps});
    log.info("  throughput   : {d:.1} tok/s  ({d:.2} ms/tok)", .{ tok_per_s, ms_per_tok });
    log.info("  total decode : {d:.2}s", .{elapsed_s});
    log.info("  sanity       : argmax==HF {d}/{d} ({s})", .{ n_match, num_steps, if (all_pass) "OK" else "DIVERGENCE — bench mesure du bruit, investiguer drift" });
    log.info("  VRAM (GPU)   : lancer `nvidia-smi` (pic) ; TODO G8 : read PJRT device memory.", .{});
    log.info("===================================", .{});
    if (!all_pass) log.warn("sanity KO : le bench mesure des tokens divergents — valider le drift fp32-{s} avant de trust les tok/s.", .{@tagName(platform.target)});
}

```
### BUILD.bazel — ajout (cibles nouvelles)

```python
# L1a — CONTRE-TEST DE NON-VACUITÉ (R2) : rebind masks_sliding <- masks_full (fenêtre OFF) → attend
# divergence ~p=512 (prouve que le masque bande est consommé). PASS = divergence (critère inversé).
# Nom court (gchunk_vac) : quota comptime pjrt structSize (cf 31 mai / ENGINE_LOG).
zig_binary(
    name = "gemma4_gchunk_vacuity",
    main = "gemma4_gchunk_vacuity.zig",
    srcs = ["engine.zig", "mem_probe.zig"],
    visibility = ["//visibility:public"],
    deps = ["//bazel", "//zml"],
)

# L1b — replay ring 512 + masque circulaire (chunké). Sur *_naive.safetensors = contre-test non-vacuité.
zig_binary(
    name = "gemma4_gchunk_ring",
    main = "gemma4_gchunk_ring.zig",
    srcs = ["engine.zig", "mem_probe.zig"],
    visibility = ["//visibility:public"],
    deps = ["//bazel", "//zml"],
)

# L2 — génération AUTONOME host-orchestrée (gather host → forwardStageStep → argmax → reinject).
zig_binary(
    name = "gemma4_gchunk_auto",
    main = "gemma4_gchunk_auto.zig",
    srcs = ["engine.zig", "mem_probe.zig"],
    visibility = ["//visibility:public"],
    deps = ["//bazel", "//zml"],
)

# G1 (P-GPU-1) — baseline fp32 sur GPU : force CUDA + timer tok/s + RSS. Calcul == L1a mono (PrecCfg défaut).
zig_binary(
    name = "gemma4_gen_long_gpu",
    main = "gemma4_gen_long_gpu.zig",
    srcs = ["engine.zig", "mem_probe.zig"],
    visibility = ["//visibility:public"],
    deps = ["//bazel", "//zml"],
)

# G8 — benchmark decode GPU (warmup + tok/s + sanity argmax==HF). VRAM via nvidia-smi.
zig_binary(
    name = "gemma4_bench",
    main = "gemma4_bench.zig",
    srcs = ["engine.zig", "mem_probe.zig"],
    visibility = ["//visibility:public"],
    deps = ["//bazel", "//zml"],
)

```
---

## 6. Playbook de validation sur la 3090

> Prérequis : `libpjrt_cuda` linké dans le workspace ZML (`--config=cuda`) ; `nvidia-smi` ;
> swapfile `/swapfile_xla` (CPU) — inutile sur GPU. Deploy : `ZML_REMOTE=user@gpu-host
> ZML_JUMP=macmini ZML_DST=/data/rqz_workspace/zml/examples/rqz zml_runner/deploy_to_3090.sh`.

### 6.1 Smoke (compile-only, sans weights/run) — confirme que TOUT compile (dont le knob PrecCfg)
```bash
cd /data/rqz_workspace/zml && bash /data/gemma4-zml-probe/scripts/smoke.sh
# attendu BUILD OK=8 : gemma4_engine_e1, _e2, gchunk, _vacuity, _ring, _auto, _gen_long_gpu, _bench
```

### 6.2 Non-régression (le seul vrai risque du knob PrecCfg) — E1/E2 + preuve HLO
```bash
cd /data/rqz_workspace/zml
./bazel.sh run //examples/rqz:gemma4_engine_e1 -- /data/gemma4-zml-probe/weights/model.safetensors <p5_7_8_gen.safetensors>   # PASS 4 tokens
./bazel.sh run //examples/rqz:gemma4_engine_e2 -- /data/gemma4-zml-probe/weights/model.safetensors <decode_vq_gen.safetensors> # PASS 4 tokens
# Preuve HLO (la SEULE vérif non-faisable hors 3090) : PrecCfg{} byte-identique au baseline
#   dumper --xla_dump_to avant/après et diff -rq : 0 diff sur les fichiers HLO.
```

### 6.3 R1 — Mesurer le go/no-go mémoire enfin capturé
```bash
./bazel.sh run //examples/rqz:gemma4_gchunk -- <ckpt> gen_long.safetensors
# lire : [mem] post-compile (go/no-go): RSS=... (~?? GiB)  → doit <23 Go ; pente [mem] step N = fuite
```

### 6.4 R2 — Contre-test non-vacuité L1a (clôture le gate L1a)
```bash
./bazel.sh run //examples/rqz:gemma4_gchunk_vacuity -- <ckpt> gen_long.safetensors 600
# attendu : VACUITY-OK (divergence ~step 508 / pos 512). Si VACUITY-FAIL (1020/1020 match) → masque non consommé = bug.
```

### 6.5 L1b — ring 512 + masque circulaire
```bash
python3 /data/gemma4-zml-probe/scripts/47_gen_long_ring_oracle.py   # produit gen_long_ring.safetensors + _naive
./bazel.sh run //examples/rqz:gemma4_gchunk_ring -- <ckpt> gen_long_ring.safetensors        # attendu L1b RING PASS 1020/1020
./bazel.sh run //examples/rqz:gemma4_gchunk_ring -- <ckpt> gen_long_ring_naive.safetensors # attendu VACUITY-OK (divergence ~pos 512)
```

### 6.6 L2 — autonomie host-orchestrée
```bash
./bazel.sh run //examples/rqz:gemma4_gchunk_auto -- <ckpt> gen_long.safetensors 64
# attendu L2 PASS : séquence GÉNÉRÉE == HF greedy (gather host + reinject). Coût host ~5,5 Go.
```

### 6.7 R4 — sweep perf (trade-off CHUNK × SYNC_EVERY)
```bash
bash /data/gemma4-zml-probe/scripts/sweep_perf.sh <ckpt> gen_long.safetensors
# table CHUNK×SYNC_EVERY → RSS post-compile / swap final / match / temps ; confirme la non-monotonie
```

### 6.8 P-GPU-1 — G1 baseline fp32 GPU + bench
```bash
./bazel.sh run //examples/rqz:gemma4_gen_long_gpu -- <ckpt> gen_long.safetensors 64   # G1 PASS + tok/s (vs ~55 min CPU)
./bazel.sh run //examples/rqz:gemma4_bench -- <ckpt> gen_long.safetensors 256          # tok/s GPU + sanity argmax==HF
nvidia-smi   # VRAM pic (TODO G8 : read PJRT device memory)
```

---

## 7. Honnêteté / limites / non-régression

- **Non validé** : sandbox bloque réseau + `.git` read-only + pas de weights local → **rien compilé/exécuté**.
  Braces/parens équilibrés, API timing vérifiée (`std.Io.Timestamp`/`untilNow`, cf `examples/benchmark`),
  motifs mirrorés du runner L1a prouvé, `zml.DataType`/`Platform.init`/`Buffer.fromBytes`/`toSliceAlloc`/`Slice.constData`
  vérifiés dans le checkout ZML.
- **Non-régression** : `engine.zig` — `forward`/`forwardStageGen` intacts ; 2 nouvelles méthodes
  (`forwardStep`/`forwardStageStep`) appelées par AUCUN runner existant ; `PrecCfg` défaut byte-identique.
  Le seul risque non-éliminable à distance = la **preuve HLO `diff -rq`** (à faire sur 3090).
- **Risque compile mineur** : pattern `const c = struct { fn call(t) ... }.call;` (capture comptime
  `prec` dans un struct fn imbriqué) — si Zig le rejette, correctif mécanique (inliner les `convert`).
- **L2 dtype** : `embed_tokens`/`embed_tokens_per_layer` lus en host au dtype natif du checkpoint ;
  critère L2 = argmax==HF greedy (pas bit-exact vs L1a) → tolère bf16/fp32.
- **Commit** : `.git` read-only → Régis commite. Suggéré :
  `git add -A && git commit -m "feat(gen-long+gpu): R1-R6 + L1b(ring) + L2(autonome) + PrecCfg + G1/bench + plan GPU (prêt-à-valider 3090)"`

---

## 8. État des fichiers (working tree, 27 juin 2026)

```
Modifiés (M) :
  README.md
  docs/ENGINE_LOG.md
  docs/GENERATION_LONGUE_DESIGN.md
  docs/GENERATION_LONGUE_PLAN.md
  zml_runner/BUILD.bazel
  zml_runner/engine.zig              (+PrecCfg, +forwardStep, +forwardStageStep)
  zml_runner/gemma4_gchunk.zig       (instrumentation RSS + SYNC_EVERY)
Nouveaux (??) :
  docs/GPU_PORT_PLAN.md              (plan GPU, 434 lignes — référencé, non reproduit)
  scripts/47_gen_long_ring_oracle.py (L1b oracle)
  scripts/regen_fixtures.sh           (R5 orchestrateur)
  scripts/smoke.sh                    (R5 compile-check)
  scripts/sweep_perf.sh               (R4 sweep)
  zml_runner/gemma4_bench.zig         (G8 bench)
  zml_runner/gemma4_gchunk_auto.zig   (L2 autonome)
  zml_runner/gemma4_gchunk_ring.zig   (L1b ring)
  zml_runner/gemma4_gchunk_vacuity.zig(R2 non-vacuité)
  zml_runner/gemma4_gen_long_gpu.zig  (G1 baseline GPU)
  zml_runner/mem_probe.zig            (R1 RSS/swap)
```

---

> **Fin du rapport consolidé.** Ce document + `docs/GPU_PORT_PLAN.md` + l'arbre de travail constituent
> l'état complet de la session. Tout est prêt pour validation sur la 3090 par Régis.
