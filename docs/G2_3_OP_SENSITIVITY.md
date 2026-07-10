# G2.3 — Cartographie de sensibilité bf16 par-op (protocole PRÉ-ENREGISTRÉ)

> **Gate `G2.3`** — protocole committé le 10 juillet 2026, **AVANT tout run de sweep** (discipline
> G2 : les métriques, seuils, départages et procédures d'échec de ce document sont figés ici ;
> tout écart post-hoc serait du cherry-picking et invaliderait le verdict).
> **Spec source** : `docs/superpowers/specs/2026-07-10-g2-3-op-sensitivity-design.md` (prime en cas
> de doute). **Doc parent** : `docs/G2_BF16_FIDELITY.md` (méthode de l'enveloppe, bras A/B, G2.2).
> **Question falsifiable** : *quelles familles d'ops de Gemma 4 tolèrent l'arrondi bf16, lesquelles
> sont sensibles, et quelle config bf16 « max sûre » tient ≤ 2× l'enveloppe HF ?*

---

## 1. But et préséance

G2.2 a prouvé que ZML en gemm-bf16 global reste 2 à 5× **sous** l'enveloppe de bruit que HF-bf16
s'autorise vs HF-fp32. G2.3 décompose ce résultat en un **instrument** : classement de sensibilité
par famille d'ops (12 familles, balayage one-hot) + config combinée validée vs l'enveloppe.
Consommateurs : TurboQuant (où la basse précision casse) et alambic (choix de précision
distillation).

**Préséance** : ce protocole **supersede le croquis G2.3 de `G2_BF16_FIDELITY.md` §3**, qui
balayait dans le sens inverse (tout-bf16 moins une famille f32). Le sens retenu — **one-hot bf16
sur base f32** — est justifié par la double référence D0 (§2) : chaque famille est mesurée
isolément contre une baseline ZML-fp32 propre, sans que l'effet des 11 autres familles ne
contamine la mesure.

## 2. Bras et références

| Bras | Contenu | Format EXACT | Rôle |
|---|---|---|---|
| **A** | logits HF-fp32, teacher-forcing S46 (dump G2.0, script 50) | `/data/gemma4-zml-probe/g2_logits_a_f32.npy` — **`.npy` memmap** [1020, 262144] f32, lu par `np.load(..., mmap_mode="r")` | vérité de référence — **verdicts** |
| **B** | logits HF-bf16 (enveloppe G2.0) | `/data/gemma4-zml-probe/g2_logits_b_bf16u16.npy` — **bit-patterns bf16 stockés en uint16** ; réinterprétation OBLIGATOIRE `torch.from_numpy(u16).view(torch.bfloat16).float()`, jamais lu comme entiers | échelle relative des seuils |
| **D0** | logits ZML fp32 (run tout-null du binaire sweep) | `.bin` **f32 brut** [steps × 262144], `np.memmap(dtype=np.float32)` | baseline ZML — **classement** + non-vacuité |
| **Dᵢ** | logits ZML, famille i en bf16 (one-hot) | `.bin` f32 brut, idem D0 | mesures du sweep |
| **D\*** | logits ZML, config combinée | `.bin` f32 brut | verdict final G2.3.2 |

⚠ Piège de format : traiter le `.npy` du bras A comme du brut décalerait toutes les valeurs du
header numpy (métriques poubelle) ; lire le bras B comme des entiers u16 est la même classe de
piège. Le loader du script 52 choisit par extension et réinterprète B explicitement.

**Double référence** : le **classement** de sensibilité se calcule **vs D0** (isole l'arrondi bf16
pur, sans le drift fp32 ZML-vs-HF ~1e-5) ; les **verdicts** (buckets, PASS combiné) se calculent
**vs A** rapportés à l'enveloppe B (comparabilité G2.0/G2.2). Nommage manifest : `*_vs_D0` /
`*_vs_A`.

**Chaîne de custody** : md5 + paramètres de génération (script, args, SHA repo) de A, B et D0
consignés dans `fixtures/g2_3_manifest.json` ; le script 52 **refuse** (exit 2) toute référence au
checksum non conforme. Règle : **au moindre doute, régénérer** (script 50 pour A/B, run `none`
pour D0) plutôt que réutiliser.

## 3. Les 12 familles

Sémantique opérationnelle contractuelle : **« bf16 » = arrondi des opérandes aux bornes de l'op
(convert in → op → convert out vers f32) ; le calcul interne reste au gré d'XLA.** On ne claim
jamais « calculé en bf16 », seulement « arrondi aux bornes ». NB assumé : `zml.nn.rmsNorm` et
`Tensor.softmax` upcastent EUX-MÊMES leur calcul interne en f32 (nn.zig, tensor.zig) — c'est
conforme au contrat (bornes arrondies), et c'est compté dans les converts attendus.

Comptes détaillés + dérivations : **`fixtures/g2_3_expected_converts.json`** (oracle du check
anti-câblage-croisé §5.3, dérivé de `engine.zig` @ `1230175`). Résumé :

| # | Famille | Sites (graphe `.forward`, seul compilé) | Delta converts vs D0 |
|---|---|---|---|
| 1 | `qkv_proj` | 35 q + 15 k + 15 v (producers 0–14, YOCO) | **+65** (+1 net/site) |
| 2 | `qk_scores` | 35 | **+105** (+3/site, cache f32 en one-hot) |
| 3 | `pv_ctx` | 35 | **+105** (+3/site) |
| 4 | `o_proj` | 35 | **+35** (+1 net/site) |
| 5 | `mlp` | 3×35 | **+105** (+1 net/site) |
| 6 | `ple` | 2×35 + 1 frontend | **+71** (+1 net/site) |
| 7 | `head` | 1 | **+1** |
| 8 | `norms` | 176 rmsScaleD + 50 q/k_norm + 15 v_norm | **+1190** (+5/site pondéré, +4 v_norm) — *uncertain* |
| 9 | `softmax` | 35 | **+140** (+4/site) — *uncertain* |
| 10 | `rope` | 10 manualRope (full) + 40 slidingRope | **+200** (+4/site) — *uncertain* |
| 11 | `softcap` | 1 | **+2** |
| 12 | `kv_store` | 30 writes + 70 reads | **+100** (caveat fixture variante) |

Les trois comptes *uncertain* (émissions internes zml vérifiées sur le miroir M1, pas sur la rev
du workspace 3090) sont **tranchés par le premier run one-hot** de leur famille et le JSON est
mis à jour (commit) **avant le sweep complet** — les alternatives chiffrées sont dans le JSON.

**Décisions actées** (questions ouvertes de la spec §4, tranchées ici) :

- **`norms` = entrée ET poids arrondis** (pas seulement l'entrée), sortie re-upcastée. Le poids
  checkpoint étant bf16, la chaîne émise au poids est bf16→f32→bf16 (le convert f32 d'origine
  reste la base, ordre d'émission préservé — cf commentaires `rmsScaleDPrec`/`rmsScaleHdPrec`).
  `v_norm` (producers seulement) n'a pas de poids : entrée seule. **Limitation documentée** :
  `per_layer_projection_norm` (rmsNorm du frontend PLE) est **HORS famille `norms`** — elle
  n'apparaît pas dans l'énumération de la spec §4 et reste f32 dans toutes les configs. La carto
  ne dit donc rien de la sensibilité de cette norme-là.
- **`rope` = x ET cos/sin arrondis.** Pour les couches full (`manualRope`) : les trois entrées
  sont explicitement encadrées (cos/sin fixture = f32, l'arrondi est une vraie op). Pour les
  couches sliding (`zml.nn.rope` natif) : les cos/sin internes sont générés en f32 puis
  **convertis au dtype de x** (prouvé `nn.zig` l.286-287) → arrondir x arrondit AUSSI cos/sin,
  sans patcher zml.
- **`kv_store` = mécanisme (b), fixture variante bf16** (`gen_long_kvbf16.safetensors`, générée
  par `scripts/46_gen_long_oracle.py --kv-dtype bf16`). Le dtype de STOCKAGE vient du header de
  la fixture ; writes arrondis avant scatter, reads re-upcastés. **L'état prefill initial est
  arrondi bf16 aussi** (le contrat s'applique au prefill, pas seulement au decode). Baseline
  one-hot = stockage f32 SANS arrondi à la lecture (≠ G2.2 qui arrondissait à la lecture).
  **Run HORS sweep par défaut** (lancé séparément avec `FIXTURE=<gen_long_kvbf16>`) ; garde :
  `Cache.checkDtype` refuse avant compile toute incohérence fixture↔prec. Seule famille qui
  change une **empreinte mémoire** → outcome secondaire VRAM (§8.4).

## 4. Métriques et verdicts (PRÉ-ENREGISTRÉS)

Par run, chacune **vs D0 ET vs A** : `max_abs(logits)` p50/p95/max, KL p50/p95/max,
mismatches argmax /1020, step de première bifurcation. **Toutes publiées** dans le manifest, quel
que soit le verdict — engagement anti-cherry-picking : aucune métrique ne sera ajoutée, retirée ou
re-pondérée après le premier run ; un résultat gênant se publie tel quel.

**Direction des KL (pré-enregistrée, convention du script 51 qui fait foi — ligne 65 :
`(lsa.exp() * (lsa - lsd)).sum()` = KL(référence‖run))** : toutes les KL de ce protocole sont
**KL(ref‖run)** avec ref ∈ {A, D0} selon la métrique — `kl_vs_A` = KL(A‖Dᵢ), `kl_vs_D0` =
KL(D0‖Dᵢ). Les ratios restent ainsi directement comparables aux ratios publiés de G2.2.

**Source normative des seuils** : les ratios se calculent sur les **valeurs NON-ARRONDIES** du
manifest `fixtures/g2_envelope_manifest.json` (chargé par le script 52 via `--envelope`) — c'est
LUI qui fait foi. Les valeurs chiffrées ci-dessous et dans le tableau sont **indicatives**
(arrondies) ; en cas d'écart à la marge, le manifest gagne. Règle identique pour max_abs et
toute métrique rapportée à l'enveloppe.

- **Métrique primaire de VERDICT** : `KL p50 vs A`, en **ratio de l'enveloppe B** (la plus
  discriminante en G2.2). Enveloppe B (G2.0, §7.1 du doc parent), valeurs indicatives :
  KL p50 ≈ 1.0e-4, max_abs p50 ≈ 0.425.
- **Buckets** :

| Bucket | Critère (ratio vs enveloppe B, **manifest non-arrondi = normatif**) | Indicatif KL p50 vs A | Indicatif max_abs p50 vs A |
|---|---|---|---|
| `SAFE` | ≤ 1× B | ≈ ≤ 1.0e-4 | ≈ ≤ 0.425 |
| `TOLERABLE` | ≤ 2× B | ≈ ≤ 2.1e-4 | ≈ ≤ 0.85 |
| `SENSITIVE` | > 2× B | ≈ > 2.1e-4 | ≈ > 0.85 |

- **Départage pré-enregistré** : si `KL p50` et `max_abs p50` donnent des buckets différents,
  **le pire l'emporte**.
- **Métrique primaire de CLASSEMENT** : `KL p50 vs D0` (ordre de sensibilité, diagnostic).
- Autres verdicts possibles par run : `FAIL-SANITY` (§5.1), `VACUOUS` (§5.2), `INVALID` (§5.3) —
  distincts des buckets, publiés au même titre.

## 5. Gates par run

Ordre : sanité → non-vacuité → anti-câblage-croisé → métriques. Un échec en amont court-circuite
l'aval mais **s'écrit au manifest** (le verdict EST le résultat).

### 5.1 Sanité (verdict `FAIL-SANITY`)

Avant toute métrique : scan NaN/Inf sur les 1020×262k logits ; détection de dégénérescence —
entropie moyenne effondrée vs A, répétition de token anormale. **Seuils : calibrés via
`52_g2_3_analyze.py --calibrate-sanity` sur les memmaps A/B de la 3090 AVANT le sweep, valeurs
consignées ICI avant le premier run one-hot** (le `g2_envelope_metrics.npz` ne contient pas
d'entropie — la calibration lit les memmaps) :

| Seuil de sanité | Valeur |
|---|---|
| NaN/Inf | 0 toléré (tout NaN/Inf ⇒ FAIL-SANITY) |
| Entropie moyenne minimale | **À CALIBRER** |
| Répétition argmax maximale (run de tokens identiques) | **À CALIBRER** |

Échec ⇒ verdict `FAIL-SANITY`, distinct de `SENSITIVE`, publié, et **famille exclue du combiné**
quels que soient ses ratios.

### 5.2 Non-vacuité (verdict `VACUOUS`)

`max_abs(Dᵢ vs D0) > 0` requis. Si bit-identique à D0 ⇒ verdict `VACUOUS` (l'arrondi n'a rien
arrondi) ⇒ investigation, jamais un `SAFE` silencieux.

### 5.3 Anti-câblage-croisé (verdict `INVALID`)

Le graphe HLO du run doit **différer** de celui de D0, et le **delta de converts** (recensement
converts du run − recensement D0, sur le dump **PRÉ-optimisation** `*before_optimizations*`) doit
être **== à l'attendu** de `fixtures/g2_3_expected_converts.json` (script 53). Mismatch ⇒ run
`INVALID`, à corriger avant de continuer. **Multi-familles** (D\*, S49 combiné) : attendu =
**somme des deltas** des familles actives (additivité stricte : chaque famille re-upcaste f32 à
ses bornes, les chaînes bf16→f32→bf16 sont émises telles quelles) ; si la somme s'avère non
vérifiable en pratique, 53 dégrade en `differs_from_d0` seul pour les multi-familles et le
consigne au manifest.

## 6. Procédure combinée (G2.3.2)

- **Config initiale** = toutes les familles `SAFE` du classement G2.3.1.
- **Critère PASS** : **≤ 2× l'enveloppe B appliqué aux métriques de verdict du §4** (KL p50 vs A
  ≤ 2.1e-4, départage max_abs p50 ≤ 0.85) — et **non** la table multi-percentile complète de
  G2.2 §7.1, qui est publiée à titre informatif.
- **Procédure d'échec pré-enregistrée** : retrait glouton de la famille au **pire `KL p50 vs
  D0`**, **max 3 essais**, **chaque essai publié** au manifest (aucun essai silencieux). Si 3
  échecs ⇒ verdict publié : « pas de config combinée SAFE au critère ≤ 2× » — null result assumé.
- **Interaction** (les one-hot sont diagnostics, non prédictifs — seul le run combiné fait
  verdict) : publier `KL p50(D*) / max_i KL p50(Dᵢ actives)` **ET** le ratio à la **somme** des
  KL p50 actives. L'écart à l'additivité est un résultat, pas une hypothèse cachée.

## 7. Stabilité S49

La claim est scopée S46 (une séquence, greedy). Test minimal : rejouer sur S49 (« capital of
France », 48 tokens, `scripts/49_gen_custom_oracle.py`) la config **combinée** + les **2
familles** les plus proches d'une frontière de bucket.

- **Sémantique pré-enregistrée** : les runs S49 sont **vs D0-S49 uniquement** (nouveau run `none`
  sur la fixture S49) — jamais vs A/B, qui sont des trajectoires S46 (les mélanger donnerait des
  métriques poubelle). `REF_A=none` dans l'orchestrateur.
- **Critère de concordance** = **ordre relatif des rangs** : les familles testées gardent-elles
  leur rang de `KL p50 vs D0` relatif l'une à l'autre et au combiné ? Concordance = claim
  renforcée ; discordance = publiée telle quelle.
- **Sanité informative-only** (les seuils §5.1 sont calibrés sur S46) : pas de FAIL-SANITY
  bloquant sur le diagnostic de stabilité.
- Runs **namespacés `RUN_PREFIX=g2_3_s49`** : D0-S46 et son md5 de custody sont intouchés.

## 8. Exécution

### 8.1 Binaire unique et provenance

**Un seul build pour tout le sweep** : `g2_3_sweep.sh` calcule le sha256 du binaire au départ et
**refuse** de continuer s'il change ; chaque entrée du manifest porte hash binaire + rev git du
workspace ZML (`/data/rqz_workspace/zml`) + rev du repo. Prérequis build : swap actif sur la VM
(check `smoke.sh`), patch `pjrt.zig` (`@setEvalBranchQuota`) en place.

### 8.2 Déterminisme et purge

- **Déterminisme** : 1 config répétée (ex `mlp`), md5 du dump comparé au premier passage
  (l'orchestrateur md5-somme chaque dump **avant** purge, `logs/g2_3_md5.log`). Attendu :
  identique. Greedy ⇒ pas de seed.
- **Purge au fil de l'eau** (1 dump ≈ 1,07 Go, ~17 runs > espace libre VM) : run → analyse (52) →
  consignation → purge du dump et du dump HLO. On conserve : **D0** (jamais purgé), **D\*** final
  (`KEEP=1`), manifest, npz de métriques par run. Check d'espace avant chaque run, abort propre
  sinon. Conséquence assumée : pas de ré-analyse post-hoc avec une métrique nouvelle (elles sont
  pré-enregistrées §4 ; re-run possible).
- **Fiabilité des artefacts** : l'orchestrateur se fie au **code de sortie** du runner **et à la
  fraîcheur du fichier logits** (mtime postérieur au lancement), **jamais à sa seule existence**
  — le runner ne crée le fichier qu'**après** le compile (un fichier stale d'un run antérieur
  peut exister alors que le run courant a échoué au compile).

### 8.3 Transport 3090

⚠ **Le `/data/gemma4-zml-probe` de la VM n'est PAS un clone git** — la synchronisation des
scripts 52/53, du JSON des converts attendus et de ce doc se fait **par rsync depuis M1**
(constaté au smoke T6). `deploy_to_3090.sh` ne pousse que `zml_runner/` vers le workspace ZML.

### 8.4 Protocole VRAM (`kv_store`)

Leçon G2.1 (mesure contaminée, invalidée) : GPU **vierge** (aucun process concurrent),
`--no-prealloc` (sinon nvidia-smi montre la réserve BFC, pas l'usage), pic scopé au **PID du
run**, mesure répétée **2×**, conditions consignées au manifest. Référence de comparaison :
**8 494 MiB** (fp32-store, G2.1).

## 9. Résultats

*(à remplir gate par gate — les FAIL/null results se documentent au même titre que les PASS)*

| Gate | Date | Verdict | Mesures clés |
|---|---|---|---|
| G2.3.0 (neutralité + pré-enregistrement) | — | — | — |
| G2.3.1 (sweep one-hot) | — | — | — |
| G2.3.2 (combiné + stabilité + VRAM) | — | — | — |

### 9.1 G2.3.0 — Neutralité du refactor

| Preuve | Attendu | Observé |
|---|---|---|
| HLO tout-null vs baseline PRÉ-refactor (gold) | byte-identique (méthode E1) | — |
| Repli (si diff cosmétique) : G1 64/64 + E1 4/4 + replay 1020/1020 | cumulatif complet | — |
| Non-régression G2.2 (`qkv_proj,qk_scores,pv_ctx,o_proj,mlp,ple,head`) | ratios 0.44×/0.28×, argmax 1016/1020, bifurcation 96 | — |
| Custody A / D0 (md5) | enregistrés au manifest | — |
| Comptes *uncertain* du JSON tranchés (norms/softmax/rope) | JSON mis à jour + committé avant sweep | — |
| Seuils de sanité calibrés (§5.1) | valeurs consignées §5.1 | — |

### 9.2 G2.3.1 — Classement de sensibilité (12 familles)

Classé par `KL p50 vs D0` décroissant :

| Rang | Famille | KL p50 vs D0 | KL p50 vs A (ratio B) | max_abs p50 vs A (ratio B) | argmax /1020 | 1re bifurcation | Converts Δ att./obs. | Verdict |
|---|---|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — | — | — |

### 9.3 G2.3.2 — Config combinée

| Essai | Familles actives | KL p50 vs A (ratio B) | max_abs p50 vs A (ratio B) | Interaction (D\*/max ; D\*/somme) | Verdict |
|---|---|---|---|---|---|
| 1 | — | — | — | — | — |

### 9.4 Stabilité S49

| Run (S49, vs D0-S49) | KL p50 vs D0-S49 | Rang S46 | Rang S49 | Concordance |
|---|---|---|---|---|
| — | — | — | — | — |

### 9.5 VRAM `kv_store`

| Mesure | Pic PID (MiB) | Référence fp32 (G2.1) | Delta |
|---|---|---|---|
| 1 | — | 8 494 | — |
| 2 | — | 8 494 | — |
