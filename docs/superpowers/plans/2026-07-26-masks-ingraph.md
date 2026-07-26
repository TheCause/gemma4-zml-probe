# Masques in-graph — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Générer les masques d'attention du 12B dans le graphe (depuis `positions[step]` + scalaire runtime `window`) → mémoire linéaire en L_MAX, équivalence stricte prouvée (M1/M2), sonde 8k au verdict mesuré (M3).

**Architecture:** `engine.zig` : `Packed` passe de `(two_masks: bool)` à `(mode: MaskMode)` avec un 3ᵉ variant `.ingraph` (sans tables masques, avec scalaire `window`) ; `EngineCfg.ingraph_masks` (défaut neutre) fait générer les deux lignes de masque par `iota`+`cmp`+`select` dans `forward`/`forwardStageGen`. Le runner `gemma4_g12auto` bascule (packed_sym/pk_buf/vacuity/selftest), nouvelle cible `gemma4_g12a8k`.

**Tech Stack:** Zig/ZML/Bazel sur la 3090 (`/data/rqz_workspace/zml/examples/rqz`, deploy `zml_runner/deploy_to_3090.sh`, env `ZML_REMOTE`/`ZML_DST` OBLIGATOIRES — piège deploy silencieux). Build GPU : `--@zml//platforms:cuda=true` sur CHAQUE `bazel.sh run`.

**Spec:** `docs/superpowers/specs/2026-07-26-masques-ingraph-design.md` (approuvée revue 2).

**Référence gates :** M0 = mécanisme U1 (md5 HLO `before_optimizations`, module principal pré-désigné, comparaison exit code — recette dans le plan J2 Step 2.1 et l'historique 3090). M1/M2 = critères PR #13. Vacuité = mécanisme U9-ii adapté (rebind `window`).

---

## Task 0 : Témoins (AVANT tout édit moteur)

Le code de la branche == main (seuls des docs ont été committés). Les témoins se génèrent MAINTENANT.

**Files:** aucun édit — runs 3090, artefacts dans `logs/`.

- [ ] **0.1** Vérifier la 3090 libre : `nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv` (garde-fou contention Ollama).
- [ ] **0.2** Deploy propre : `ZML_REMOTE=<user@gpu-host> ZML_DST=/data/rqz_workspace/zml/examples/rqz zml_runner/deploy_to_3090.sh` — vérifier la sortie rsync NON vide (piège deploy silencieux).
- [ ] **0.3** Dump HLO témoin M0 (recette U1/ENGINE_LOG:92 — dumps `--xla_dump_to`, `diff -rq`, 2 diffs bénins tolérés : `debug_options` + noms SSA `.ll` alpha-équivalents) pour les DEUX chemins migrés : `gemma4_engine_e1`/`gemma4_engine_e2` (**`.single`** — mécanisme U1 historique, CPU) et `gemma4_g12auto` + `gemma4_gen_auto` (**`.tables`** — gen_auto:51 est `Packed(true)`, PAS `.single`). Dumps archivés `/data/gemma4-zml-probe/m0_hlo_before/`.
- [ ] **0.4** Run témoin M1 : `gemma4_g12auto` free-run 48 tokens, MÊME protocole que le run « défaut 1280 bit-identique » de PR #13 (relire le message du commit `85ed329` + ENGINE_LOG pour prompt/flags exacts), `--out-ids /data/gemma4-zml-probe/mi_witness_1280_ids.safetensors`.
- [ ] **0.5** Run témoin M2 : `gemma4_g12a4k`, préfixe 300 tokens (protocole PR #13 variante), `--out-ids /data/gemma4-zml-probe/mi_witness_4k_ids.safetensors`.
- [ ] **0.6** Rapatrier les 2 fichiers ids + md5 dans `logs/` (M1, non versionné : `logs/` est l'archive locale du repo).

## Task 1 : engine.zig — MaskMode + Packed(mode) + migration call-sites

**Files:**
- Modify: `zml_runner/engine.zig:307-363` (Packed), `:382-397` (EngineCfg), signatures `:655,698,742,780`
- Modify (migration mécanique `.single`) : `gemma4_engine_e1.zig:70`, `gemma4_engine_e2.zig:84`, `gemma4_g12gate.zig` (×6 : 1027/1141/1161/1401/1546/1564)
- Modify (migration mécanique `.tables`) : `gemma4_bench.zig:28`, `gemma4_bbatch.zig:52`, `gemma4_g23_sweep.zig:45`, `gemma4_g12auto.zig:62`, `gemma4_gchunk_auto.zig:37`, `gemma4_gchunk.zig:42`, `gemma4_gchunk_vacuity.zig:50`, `gemma4_gchunk_ring.zig:39`, `gemma4_gen_long.zig:24`, `gemma4_gen_long_gpu.zig:35`, `gemma4_gen_auto.zig:51`, `gemma4_vacuity_logits.zig:44`, `gemma4_w4auto.zig:39`

- [ ] **1.1** `engine.zig` : ajouter `pub const MaskMode = enum { single, tables, ingraph };` et `pub const MASK_MIN: f32 = -std.math.floatMax(f32);`.
- [ ] **1.2** `Packed(comptime mode: MaskMode)` : `.single` = ancien `Packed(false)` inchangé ; `.tables` = ancien `Packed(true)` inchangé ; `.ingraph` = `.tables` sans `masks_sliding`/`masks_full`, avec `window: zml.Tensor` (`{}` i32) — `init` : `.window = v.createTensor("window", .{}, null)` (même si les runners 12B assemblent à la main, garder la symétrie), `load` identique.
- [ ] **1.3** `EngineCfg` : `ingraph_masks: bool = false` + helper `pub fn maskMode(comptime cfg: EngineCfg) MaskMode` (`ingraph_masks` → `.ingraph`, sinon `two_masks` → `.tables`, sinon `.single`) + `@compileError` comptime si `ingraph_masks and !two_masks`.
- [ ] **1.4** Remplacer `Packed(cfg.two_masks)` par `Packed(cfg.maskMode())` dans les 4 signatures forward (`:655,698,742,780`) et adapter les extractions `mask_single`/`mask_sliding`/`mask_full` en branches comptime à 3 cas (le cas `.ingraph` renvoie `{}` pour l'instant — génération en Task 2).
- [ ] **1.5** Migration call-sites : `Packed(false)` → `Packed(.single)` (e1, e2, g12gate), `Packed(true)` → `Packed(.tables)` (bench, bbatch, g23_sweep, g12auto). Vérif : `grep -rn "Packed(true)\|Packed(false)" zml_runner/` = 0.
- [ ] **1.6** `gemma4_g12auto.zig` : remplacer le `MASK_MIN` local par `engine.MASK_MIN` (une source).
- [ ] **1.7** Commit : `feat(engine): Packed(MaskMode) — 3e variant .ingraph (window scalaire runtime), migration mécanique des call-sites`.

## Task 2 : engine.zig — génération in-graph + gardes

**Files:**
- Modify: `zml_runner/engine.zig` (forward `:655+`, forwardStageGen `:698+`, forwardStep `:742+`, forwardStageStep `:780+`)

- [ ] **2.1** Helper privé dans engine.zig (à côté de `pickStep`) :

```zig
// Masques in-graph (spec 2026-07-26 §4.3) : lignes générées depuis positions[step] et le
// scalaire runtime window. Valeurs STRICTEMENT {0, MASK_MIN} (pas d'addition de masques :
// -floatMax + -floatMax = -inf ≠ MASK_MIN). q=1 (décode) : shape {b,h,q,k}.
fn ingraphMaskLines(comptime kmax_sl: i64, comptime kmax_fl: i64, positions: zml.Tensor, window: zml.Tensor, step: zml.Tensor) struct { zml.Tensor, zml.Tensor } {
    const pos = pickStep(positions, step); // scalaire i32
    // full {k=kmax_fl} : j <= p
    const iota_f = zml.Tensor.iota(zml.Shape.init(.{ .k = kmax_fl }, .i32), .k);
    const le_f = iota_f.cmp(.LE, pos.broad(iota_f.shape()));
    const zero_f = zml.Tensor.scalar(0, .f32).broad(le_f.shape().withDtype(.f32));
    const min_f = zml.Tensor.scalar(MASK_MIN, .f32).broad(le_f.shape().withDtype(.f32));
    const full_line = le_f.select(zero_f, min_f);
    // sliding {k=kmax_sl} : j <= p ET j >= p - (window - 1)
    const iota_s = zml.Tensor.iota(zml.Shape.init(.{ .k = kmax_sl }, .i32), .k);
    const pos_s = pos.broad(iota_s.shape());
    const le_s = iota_s.cmp(.LE, pos_s);
    const lo = pos_s.sub(window.broad(iota_s.shape())).addConstant(1); // p - window + 1
    const ge_s = iota_s.cmp(.GE, lo);
    const zero_s = zml.Tensor.scalar(0, .f32).broad(le_s.shape().withDtype(.f32));
    const min_s = zml.Tensor.scalar(MASK_MIN, .f32).broad(le_s.shape().withDtype(.f32));
    // AND booléen : select imbriqué (repli robuste ; si Tensor.logical(.AND) existe sur le
    // workspace, équivalent — les VALEURS produites sont identiques dans les deux cas)
    const sliding_line = le_s.select(ge_s.select(zero_s, min_s), min_s);
    // {k} → {b=1,h=1,q=1,k} : reshape layout-preserving + re-tag (pièges ZML #1/#2)
    return .{
        sliding_line.reshape(.{ 1, 1, 1, kmax_sl }).withTags(.{ .b, .h, .q, .k }),
        full_line.reshape(.{ 1, 1, 1, kmax_fl }).withTags(.{ .b, .h, .q, .k }),
    };
}
```

  NB syntaxe exacte (`Shape.init` avec tag, `cmp` broadcast, `withDtype`, `addConstant`) : à
  ajuster au compil sur le workspace 3090 — les ops existent (tensor.zig : iota:2014,
  cmp:3700, select:3869, scalar:2061, logical:1031), la sémantique du helper est le contrat.
- [ ] **2.2** `forward` et `forwardStageGen` : cas `.ingraph` des branches comptime →
  `const sl, const fl = ingraphMaskLines(cfg.kmax_sliding, cfg.kmax_full, p.positions, p.window, step);` (les champs `kmax_*` existent — g12auto les règle déjà à L_MAX). Le reste (sélection par `cfg.geom.isFull(i)`, `runLayerGen`) inchangé.
- [ ] **2.3** `forwardStep` et `forwardStageStep` : `if (cfg.ingraph_masks) @compileError("ingraph_masks non câblé sur forwardStep/forwardStageStep — runners 12B = forwardStageGen seul");`.
- [ ] **2.4** Commit : `feat(engine): génération in-graph des masques (iota+cmp+select, valeurs {0, MASK_MIN} exactes) derrière EngineCfg.ingraph_masks`.

## Task 3 : Gate M0 — neutralité (3090)

- [ ] **3.1** Deploy (mêmes précautions que 0.2), rebuild les 4 cibles témoins (e1, e2 `.single` ; g12auto, gen_auto `.tables`) avec dump HLO → `m0_hlo_after/` ; comparaison `diff -rq` before/after par cible — seuls les 2 diffs bénins documentés tolérés (ENGINE_LOG:92). Tout fichier HLO qui bouge = FAIL M0 → STOP (règle d'arrêt spec §5).
- [ ] **3.2** Builds verts de **TOUS** les runners migrés (R3 spec : « tous les runners two_masks avant merge ») : `gemma4_engine_e1`, `gemma4_engine_e2`, `gemma4_g12gate`, `gemma4_bench`, `gemma4_bbatch`, `gemma4_g23_sweep`, `gemma4_g12auto`, `gemma4_g12a4k`, `gemma4_gen_auto`, `gemma4_gchunk_auto`, `gemma4_gchunk`, `gemma4_gchunk_vacuity`, `gemma4_gchunk_ring`, `gemma4_gen_long`, `gemma4_gen_long_gpu`, `gemma4_vacuity_logits`, `gemma4_w4auto` (cuda=true pour les cibles GPU).
- [ ] **3.2b** Runs bon marché du chemin `.single` : e1 et e2 (CPU) re-PASS — valide `.single` au-delà du HLO.
- [ ] **3.3** Commit + tag `gate/mi-m0-pass` (message avec les md5).

## Task 4 : Bascule g12auto en ingraph

**Files:**
- Modify: `zml_runner/gemma4_g12auto.zig` (`:61-62` cfg/PackedLong, `:299-380` HostInputs, `:428-516` selftest, `:1011-1019` packed_sym, `:1038-1046` pk_buf, `:1100-1109` pk_wide vacuity — boucle 2-passes `if (pass == 0) pk_buf else pk_wide` à `:1132`)

- [ ] **4.1** `const Model = engine.EngineModel(struct {}, .{ .geom = g12.g12, .two_masks = true, .ingraph_masks = true, .kmax_sliding = L_MAX, .kmax_full = L_MAX });` et `const PackedLong = engine.Packed(.ingraph);`.
- [ ] **4.2** `HostInputs` : supprimer les allocs/remplissages `masks_sliding`/`masks_full` (et leurs `free`). `maskRows` RESTE (selftest 4.4).
- [ ] **4.3** `packed_sym` : retirer les 2 champs masques ; ajouter `.window = zml.Tensor.init(.{}, .i32)` (scalaire sans tags). `pk_buf` : retirer les 2 `fromBytes` masques ; ajouter `.window = try zml.Buffer.scalar(io, platform, @as(i32, @intCast(SLIDING_WINDOW)), .i32, sharding)` (pattern `step_buf:1127`). Garde : `comptime std.debug.assert(SLIDING_WINDOW > 0);`.
- [ ] **4.4** Selftest inputs : la comparaison masques vs fixture se fait désormais en recalculant chaque ligne à la volée via `maskRows(p, scratch_sl, scratch_fl)` (2 lignes de scratch L_MAX, plus de table O(L²)) — prouve que la SPEC du masque n'a pas dérivé ; l'équivalence du GRAPHE est M1/M2.
- [ ] **4.5** `--window-vacuity` : `pk_wide` = `pk_buf` avec `.window = try zml.Buffer.scalar(io, platform, @as(i32, @intCast(L_MAX)), .i32, sharding)` (fenêtre non mordante) — supprimer le rebind buffer masks. Mettre à jour les logs/commentaires du mode (mention « rebind window », attendu D10 inchangé : divergence exactement à q = 1024).
- [ ] **4.6** Ajuster les commentaires de tête (`:9-37`) : le coût quadratique documenté est LEVÉ, pointer la spec.
- [ ] **4.7** Commit : `feat(g12auto): bascule masques in-graph — window scalaire runtime, vacuity par rebind window, selftest ligne-à-la-volée`.

## Task 5 : Gate M1 — équivalence 1280

- [ ] **5.1** Deploy + build + run `gemma4_g12auto` MÊME protocole que 0.4, `--out-ids mi_candidate_1280_ids.safetensors`.
- [ ] **5.2** Comparaison ids témoin/candidat (48/48 requis — script python trivial ou `cmp` des fichiers si le format est stable). FAIL → STOP.
- [ ] **5.3** Vacuity 1280 : `--window-vacuity` (replay du témoin) — divergence attendue exactement à q = 1024.
- [ ] **5.4** Commit + tag `gate/mi-m1-pass` (chiffres dans le message).

## Task 6 : Gate M2 — équivalence 4096

- [ ] **6.1** Idem M1 sur `gemma4_g12a4k`, préfixe 300 : 300/300 requis. FAIL → STOP.
- [ ] **6.2** Commit + tag `gate/mi-m2-pass`.

## Task 7 : Cible 8k

**Files:**
- Create: `zml_runner/gemma4_g12a8k.zig` (copie de `gemma4_g12a4k.zig`, L_MAX=8192 — piège 18 : PAS de `@import("root")`)
- Modify: `zml_runner/BUILD.bazel` (cible `gemma4_g12a8k`, calquée sur `gemma4_g12a4k`)

- [ ] **7.1** Créer le fichier + la cible. Commit : `feat(ctx-long): cible gemma4_g12a8k (G12Auto(8192))`.

## Task 8 : Gate M3 — sonde 8k (verdict mesuré, pas de FAIL possible)

- [ ] **8.1** `nvidia-smi` garde-fou, puis build `gemma4_g12a8k` cuda=true — **chronométrer la compile** (donnée publiée). Si quota comptime pjrt déborde : parade patch `@setEvalBranchQuota` (piège workspace, vérifier qu'il est en place).
- [ ] **8.2** Run libre long `--no-prealloc` en **nohup + log distant + stdin fermé** (R6, leçon session U7) ; surveiller par polls du log (filtres sur états TERMINAUX seulement). Relever : OOM ou pas, pic VRAM `nvidia-smi`, tok/s, nombre de tokens stables.
- [ ] **8.3** Si le run tient : vacuity 8k (rebind window, divergence attendue à q = 1024).
- [ ] **8.4** Publier le verdict dans le doc de résultats (Task 9) : PASS technique {pic, tok/s, compile} OU mur chiffré {OOM à quel poste, décomposition, pointeur backlog arène XLA}. Commit + tag `gate/mi-m3-verdict`.

## Task 9 : Docs + clôture

- [ ] **9.1** `docs/MASKS_INGRAPH_RESULTS.md` : gates M0-M3, chiffres, verdict 8k, pièges neufs éventuels.
- [ ] **9.2** `PLANNING.md` : item backlog « masques in-graph » → clos avec pointeur ; README si le bandeau mentionne la limite 4k.
- [ ] **9.3** Anonymisation : `git diff main...HEAD | grep -cE '192\.168\.|10\.0\.|<user>@|/Users/<user>|/home/<user>|<alias-ssh-perso>'` = 0 (pattern complet §5.4 DOCUMENTATION.md avec les vrais littéraux).
- [ ] **9.4** Push branche + PR vers main (corps : spec, gates, verdict M3).
