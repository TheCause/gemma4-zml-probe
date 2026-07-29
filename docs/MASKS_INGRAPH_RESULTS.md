# Masques in-graph — Résultats (chantier du 26 juillet 2026)

> **⚠ Portée de la claim « == HF »** (passe de nuance, chantier `generation_config` du 29 juil 2026).
> Partout dans ce document, « == HF » signifie **même argmax sur les logits bruts** — un critère
> plus strict que comparer deux `generate()`, mais **pas** le même énoncé. Jusqu'au 29 juil le
> portage n'appliquait **pas** `generation_config.json` (`suppress_tokens`, EOS multiples) : la
> lecture « reproduit ce que `generate()` produirait » était **fausse**. Elle est devenue vraie
> **pour le 12B en mode libre** et reste **fausse pour les runners E2B**.
> Détail et chiffres : `docs/GENERATION_CONFIG_RESULTS.md` · `docs/FINDING_GENERATION_CONFIG.md`.

> Spec : `docs/superpowers/specs/2026-07-26-masques-ingraph-design.md` (approuvée revue 2)
> Plan : `docs/superpowers/plans/2026-07-26-masks-ingraph.md` (approuvé revue 2)
> Niveau de travail : léger. Branche `masks-ingraph`.

## Verdict en une ligne

**Le seul terme quadratique du design 12B (tables de masques `{L_MAX,L_MAX}`) est supprimé** :
les lignes de masque sont générées **dans le graphe** depuis `positions[step]` et un scalaire
runtime `window` — équivalence stricte prouvée aux deux bornes existantes (M1/M2), neutralité
HLO byte-identique sur les chemins non touchés (M0), sonde 8k : **voir §M3**.

## Ce qui a changé

| Composant | Avant | Après |
|---|---|---|
| `engine.Packed` | `Packed(two_masks: bool)`, 2 variants | `Packed(mode: MaskMode)`, 3 variants — `.ingraph` = 6 champs, **sans** tables masques, **avec** `window {} i32` |
| Masques du décode 12B | 2 tables host f32 `{step=L_MAX,…,k=L_MAX}` (12,5 MiB à 1280, 128 MiB à 4k, 512 MiB à 8k — host ET device) + 1 `dynamicSlice` par masque par step | `engine.ingraphMaskLines` : `iota {k=L_MAX}` + `cmp` + `select`, valeurs strictement `{0, MASK_MIN}`, bornes identiques à `maskRows` |
| Fenêtre glissante | cuite dans les tables (SW=1024) | **donnée runtime** (`window`, 12B : 1024) — rebindable |
| Contre-test de vacuité (U9-ii) | rebind buffer `masks_sliding` ← `masks_full` | rebind scalaire `window` ← L_MAX (même executable, une compile — sémantique identique, vérifiée : divergence exactement à q=1024) |
| `EngineCfg` | — | `ingraph_masks: bool = false` (+ helper comptime `maskMode()`, gardes `@compileError` : two_masks requis, forwardStep/forwardStageStep non câblés) |
| Call-sites | `Packed(true/false)` | migration mécanique 17 fichiers (`.tables`/`.single`) |
| Cibles 12B | `gemma4_g12auto` (1280), `gemma4_g12a4k` (4096) | les mêmes, **basculées ingraph**, + `gemma4_g12a8k` (8192, sonde) |

## Gates

| Gate | Critère pré-enregistré | Mesure | Verdict |
|---|---|---|---|
| **M0 neutralité** | HLO `before_optimizations` byte-identique, option off (méthode U1/ENGINE_LOG:92) | **4/4 témoins byte-identiques** : e1 + e2 (`.single`), gen_auto + g12auto (`.tables`, moteur édité + runner pré-bascule) ; E1 4/4 re-PASS, E2 4/4 re-PASS, gén gen_auto identique ; **18 builds verts** (17 runners migrés + g12a8k). Diffs résiduels = post-autotune GPU (`thunk_*`, `after_optimizations`, buffer-assignment) + codegen `.ll`/`.ptx` + `debug_options`/`config` (chemin de dump) — le graphe tracé est byte-identique | **PASS** (tag `gate/mi-m0-pass`) |
| **M1 équivalence 1280** | 48/48 ids free-run identiques au témoin `.tables` (protocole PR #13) | `cmp` des out-ids **vide** (48 ids, prompt « fenêtre glissante », arrêt max-tokens) ; + vacuity candidat : logits **bit-identiques sur 1024 positions**, première divergence **exactement à q=1024** (replay 1178 positions teacher-forcées → **la zone mordante de la fenêtre est couverte**, renfort au-delà du critère minimal) — même rapport que le témoin `.tables` (max_abs à q : 1,78e-2 candidat vs 3,07e-2 témoin, valeur descriptive : non-déterminisme inter-compiles XLA-GPU documenté, le critère D10 est la position) | **PASS** (tag `gate/mi-m1-pass`) |
| **M2 équivalence 4096** | préfixe identique au témoin `gemma4_g12a4k` `.tables` | **124/124 ids identiques** (`cmp` vide) + early-stop EOT au même step (le protocole libre s'arrête à 124 < 300 — comparaison complète, pas tronquée) ; 8,1 tok/s (≈ témoin 8,3) | **PASS** (tag `gate/mi-m2-pass`) |
| **M3 sonde 8k** | verdict mesuré publié (pas de FAIL possible) | — À REMPLIR — | — |

Témoins archivés : `logs/mi_witness_1280_ids.safetensors` (+ `.log`), `logs/mi_witness_4k_ids.safetensors`
(+ `.log`), `logs/mi_witness_wv.log` (rapport vacuity de référence). Artefacts candidat et dumps HLO
before/after/candidate sur la 3090 (`m0_hlo_*`, `m1_candidate_*`, `m2_candidate_*`).

## §M3 — Sonde 8k (verdict mesuré)

Protocole : `gemma4_g12a8k` (= `G12Auto(8192)`), `--no-prealloc` (mesure VRAM réelle, mécanisme
U10), `--window-vacuity` sur une fixture replay de **8000 ids** (tuilage des 1150 ids
teacher-forcés du gate U9) → 28 (prompt) + 8000 = **8028 positions exercées**, 2 passes
(fenêtre 1024 puis window←8192), poll `nvidia-smi` 10 s.

- **Compile : PASS en 38,9 s** (≈ la 4k : 37,9 s) — l'arène de compile XLA n'est PAS le mur 8k.
- **Exécution : OOM au premier step** — `ResourceExhausted Out of memory while trying to
  allocate 2.50GiB` (PJRT), VRAM à 22 216 MiB / 24 576 au moment de l'alloc.
- **2,50 GiB = exactement UN cache sliding** (40 slots × 8 KVH × 8192 × 256 × 4 o =
  2 684 354 560 o) : le mur suivant est le **double-buffering des caches KV** — le step
  retourne les caches en sortie sans donation des buffers d'entrée, XLA alloue donc les
  caches EN DOUBLE (2 × 5,6 GiB à 8k ; à 4k, 2 × 2,8 GiB rentrait encore).
- Décomposition statique 8k : poids w4+embed ~11,6 GiB + caches K+V 5,6 GiB + tables
  linéaires ~0,1 GiB ≈ 17,3 GiB ; avec le double-buffering ≈ 22,9 GiB + activations > 24 GiB.
- Vacuité de fenêtre à 8k : non exercée (le run n'atteint pas le premier step).

**Verdict M3 : mur suivant chiffré.** Le terme quadratique est levé (les masques in-graph
compilent et le graphe 8k se trace en 39 s) ; **8k reste infaisable sur 24 Go avec le cache
actuel**, non plus à cause des masques mais du **double-buffering des caches KV linéaires**.
Deux pistes backlog, par ordre de levier :
1. **Donation des buffers de cache** (input-output aliasing PJRT) : le runner rebinde déjà
   les caches de sortie sur l'entrée du step suivant — la donation éliminerait le ×2
   (→ statique 8k ~17,3 GiB, marge ~6 GiB : 8k passerait vraisemblablement).
2. **Ring buffer sliding 1024** (`ring=true` + `kmax_sliding=1024`, mécanisme déjà dans le
   moteur) : cache sliding 5,4 GiB → 0,67 GiB quel que soit L_MAX — mais change les scatters
   (chantier à part, gates dédiés).

## Fidélité — périmètre de la claim (inchangé)

La claim « == HF-fp32 STRICT » reste celle de PR #13 : **jusqu'à 4041 positions** (4000/4000
teacher-forcé fp32). Le chemin in-graph en hérite par l'équivalence M1/M2. **Aucune claim de
fidélité HF au-delà de 4041 positions** — la sonde 8k est technique (stabilité, VRAM, débit,
fenêtre), par décision pré-enregistrée (spec §2, pas d'oracle 8k).

## Leçons / notes d'exécution

- **Le byte-identique HLO interdit de réordonner l'émission des ops des modes existants** :
  `ingraphMaskLines` refait son propre `pickStep(positions)` plutôt que de réutiliser le `pos_i`
  extrait plus bas — XLA dédupliquera ; déplacer l'extraction aurait cassé M0.
- **`cmp` ZML broadcaste les scalaires rank-0 nativement** (tensor.zig) — pas de `broad`
  explicite pour `pos`/`window` ; `Shape.init(.{ .k = n }, dt)` porte déjà le tag (le
  `.withTags` sur une Shape ne compile pas — attrapé avant build).
- **AND booléen sans op logique** : `le.select(ge.select(zero, min), min)` — valeurs strictement
  `{0, MASK_MIN}` ; ne JAMAIS additionner deux masques (−floatMax + −floatMax = −inf ≠ MASK_MIN).
- Séquencement des témoins : le témoin M0 `.tables` du runner basculé exige un état hybride
  (moteur édité + runner pré-bascule) — worktree au commit moteur, `checkout <pré-bascule> --
  <runner>`, deploy, run. Les témoins ids/vacuity se génèrent AVANT tout deploy.
