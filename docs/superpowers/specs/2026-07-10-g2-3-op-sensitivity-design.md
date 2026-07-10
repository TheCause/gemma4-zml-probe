# G2.3 — Cartographie de sensibilité bf16 par-op (design)

> **Statut** : design validé par Régis (10 juil 2026), durci par 2 cycles /iterate (14 propositions
> intégrées). Critère directeur : **rigueur falsifiable** — le verdict doit tenir face à une review
> adversariale.
> **Prérequis de lecture** : `docs/G2_BF16_FIDELITY.md` (méthode de l'enveloppe, bras A/B, critères G2.2).
> **Branche** : `g2.3-op-sensitivity` (depuis `main`, post-merge PR #3).

## 1. Contexte et objectif

G2.2 a prouvé que ZML en **gemm-bf16 global** reste 2 à 5× sous l'enveloppe de bruit que HF-bf16
s'autorise vs HF-fp32. G2.3 décompose ce résultat : **quelles familles d'ops de Gemma 4 tolèrent
bf16, lesquelles sont sensibles, et quelle est la config bf16 « max sûre » ?**

Consommateurs du livrable : **TurboQuant** (savoir où la basse précision casse — le cache KV en
premier lieu) et **alambic** (choix de précision pour la distillation). Le livrable est un
**instrument** (classement chiffré + runner rejouable), pas une optimisation de perf.

**Livrable double** (décision Régis) :
1. **Carto** : classement de sensibilité par famille d'ops (balayage one-hot).
2. **Config combinée** : les familles tolérantes activées ensemble, validée vs l'enveloppe.

## 2. Périmètre / non-objectifs

- **Dans le périmètre** : les 12 familles du §4, séquence de référence S46 (le `fed` de
  `46_gen_long_oracle.py`), teacher-forcing 1020 steps, GPU 3090, critères relatifs à l'enveloppe G2.0.
- **Hors périmètre** : fp16/fp8/int8 (bf16 seulement) ; perf/débit comme objectif (mesuré en
  passant, non optimisé) ; batching, L3, runtime autonome ; multimodal.
- **Claim scopée** : la carto est mesurée **sur une séquence** (S46, greedy). La généralisation est
  testée a minima (§5.6) et la claim publiée est bornée en conséquence.

## 3. Architecture (approche B — un binaire, config runtime)

### 3.1 Refactor `PrecCfg` comptime → `PrecRt` runtime

Constat de code (`engine.zig`) : `dotPrec(prec, a, b, axis)` est appelé sur tous les sites GEMM
avec un `prec.gemm` **global**, et `prec` est comptime (replié depuis `cfg.prec`, membre du type du
modèle → allonge `@typeName`, cause du piège quota comptime `pjrt.zig structSize`).

Refactor :
- Nouveau struct **`PrecRt`** : un champ `?zml.DataType` **par famille** (§4), `null` = f32.
  Porté par le modèle comme **champ runtime** (`self.prec`), renseigné depuis argv **avant**
  `platform.compile`. Le traçage ZML exécute du code Zig ordinaire : un branchement sur une valeur
  runtime suffit à émettre (ou non) les converts — le graphe compilé diffère par run, le binaire
  est unique.
- `dotPrec` prend le champ de **sa** famille (`prec.qk_scores`, `prec.mlp`, …) au lieu du global.
- **Familles non-GEMM (8–12)** : elles ne passent pas par `dotPrec` — le refactor inclut des
  wrappers convert-in/convert-out à leurs sites (`rmsNorm`, `softmax`, application rope, softcap),
  plus le changement de dtype des buffers de cache pour `kv_store` (§4). Le plan doit budgéter ces
  wrappers, pas seulement le chemin GEMM.
- `EngineCfg.prec` (comptime) est **retiré**. Le runner G2.2 (`gemma4_gen_long_gpu_bf16`) est
  réexprimé comme un cas du nouveau mécanisme (toutes les familles GEMM à bf16) — non-régression §6.1.
- Bénéfice collatéral : la cfg de précision sort du type → `@typeName` raccourcit, on cesse
  d'aggraver le piège comptime (le patch workspace `@setEvalBranchQuota` reste requis par ailleurs).

### 3.2 Composants

| Composant | Rôle |
|---|---|
| `zml_runner/gemma4_g23_sweep.zig` | Runner unique : `--bf16 <fam>[,<fam>...]` (vide = baseline D0), teacher-forcing S46, dump logits f32, dump HLO |
| `scripts/g2_3_sweep.sh` (3090) | Orchestration : build unique → boucle configs → run → analyse → purge dump (§7) |
| `scripts/52_g2_3_analyze.py` | Métriques par run vs D0 et vs A, ratios vs enveloppe B, sanité, non-vacuité logits, manifest |
| `scripts/53_g2_3_hlo_check.py` (ou équiv.) | Check HLO par run : diff vs baseline + comptage converts attendus/observés (§5.3) |
| `fixtures/g2_3_manifest.json` | Manifest versionné : configs, provenance, métriques, verdicts |
| `docs/G2_3_OP_SENSITIVITY.md` | Protocole **pré-enregistré committé avant tout run** + résultats |

## 4. Les 12 familles

Sites relevés dans `engine.zig` (`runLayerGen` + frontend/head). Sémantique opérationnelle
contractuelle : **« bf16 » = arrondi des opérandes aux bornes de l'op (convert in → op → convert
out vers f32) ; le calcul interne reste au gré d'XLA.** On ne claim jamais « calculé en bf16 »,
seulement « arrondi aux bornes ». (Durcissement C1-i6.)

| # | Famille | Sites (engine.zig) | Converts attendus (à dériver précisément en G2.3.0) |
|---|---|---|---|
| 1 | `qkv_proj` | q_proj (35) ; k_proj/v_proj (producers 0–14 seulement, YOCO) | 35 + 2×15 sites |
| 2 | `qk_scores` | dot Q·Kᵀ | 35 |
| 3 | `pv_ctx` | dot probs·V | 35 |
| 4 | `o_proj` | projection sortie attention | 35 |
| 5 | `mlp` | gate/up/down | 3×35 |
| 6 | `ple` | per_layer_input_gate, per_layer_projection (35) + projection frontend (1) | 2×35 + 1 |
| 7 | `head` | dot lm_head (embed_tokens tied) | 1 |
| 8 | `norms` | rmsNorm : input/post-attn/pre-ff/post-ff + q/k/v_norm + norm finale | à dériver |
| 9 | `softmax` | softmax attention | 35 |
| 10 | `rope` | application cos/sin (q, k) | à dériver |
| 11 | `softcap` | tanh final | 1 |
| 12 | `kv_store` | **stockage** du cache K/V en bf16 | 2 writes + reads |

**Contrat `kv_store`** (durcissement C1-i6) : baseline one-hot = stockage f32 **sans** arrondi à la
lecture (≠ G2.2 qui arrondissait à la lecture avec stockage f32) ; le run `kv_store` = buffers de
cache créés en bf16, convert à l'écriture, upcast f32 à la lecture. C'est la seule famille qui
change une **empreinte mémoire**, pas seulement le compute → outcome secondaire VRAM (§7.3).

Les comptes de converts attendus par famille sont **dérivés et committés en G2.3.0** (avant le
sweep) — ils servent d'oracle au check anti-câblage-croisé (§5.3). YOCO impose la minutie : les
readers (15–34) n'ont pas de sites K/V.

**Questions ouvertes tranchées en G2.3.0** (avec la dérivation des comptes) : pour `norms`,
l'arrondi porte-t-il aussi sur le **poids** de la norme (ou seulement l'entrée) ? Pour `rope`,
les tables cos/sin sont-elles arrondies ? La réponse choisie est pré-enregistrée dans
`G2_3_OP_SENSITIVITY.md` avant le sweep.

**Note de préséance** : cette spec **supersede le croquis G2.3 de `G2_BF16_FIDELITY.md` §3**, qui
balayait dans le sens inverse (tout-bf16 moins une famille). Le sens one-hot-bf16 sur base f32 est
justifié par la double référence D0 (§5.1) : chaque famille est mesurée isolément contre une
baseline propre.

## 5. Protocole de mesure (pré-enregistré)

Tout ce §5 est figé dans `docs/G2_3_OP_SENSITIVITY.md` **committé avant le premier run de sweep**
(discipline G2). (Durcissement C1-i5.)

### 5.1 Références et bras

| Bras | Contenu | Rôle |
|---|---|---|
| **A** | logits HF-fp32, teacher-forcing S46 (dump G2.0) | vérité de référence — **verdicts** |
| **B** | enveloppe HF-bf16 vs A (métriques G2.0) | échelle relative des seuils |
| **D0** | logits ZML fp32 (run tout-null du nouveau binaire) | baseline ZML — **classement diagnostic** + non-vacuité |
| **Dᵢ** | logits ZML, famille i en bf16 (one-hot) | mesures du sweep |
| **D\*** | logits ZML, config combinée | verdict final |

**Double référence** (durcissement C1-i2) : le **classement** de sensibilité se calcule **vs D0**
(isole l'effet de l'arrondi bf16 pur, sans le drift fp32 ZML-vs-HF ~1e-5) ; les **verdicts**
(buckets et PASS combiné) se calculent **vs A** rapportés à l'enveloppe B (comparabilité
G2.0/G2.2). Nommage explicite dans le manifest : `*_vs_D0` / `*_vs_A`.

**Chaîne de custody** (durcissement C2-i5) : md5 + paramètres de génération (script, args, SHA
repo) de A, B et D0 consignés dans le manifest ; `52_g2_3_analyze.py` **refuse** une référence au
checksum non conforme. Au moindre doute → régénérer plutôt que réutiliser.

### 5.2 Métriques et verdicts

Par run : max_abs(logits) p50/p95/max, KL p50/p95/max, mismatches argmax /1020, step de 1re
bifurcation — chacune vs D0 et vs A. **Toutes publiées** dans le manifest, quel que soit le verdict.

- **Métrique primaire de verdict** : `KL p50 vs A`, en ratio de l'enveloppe B (la plus
  discriminante en G2.2).
- **Buckets** : `SAFE` ≤ 1× B · `TOLERABLE` ≤ 2× B · `SENSITIVE` > 2× B.
- **Départage pré-enregistré** : si `KL p50` et `max_abs p50` donnent des buckets différents, **le
  pire l'emporte**.
- **Métrique primaire de classement** : `KL p50 vs D0`.

### 5.3 Non-vacuité et anti-câblage-croisé (par run)

(Durcissements C1-i1 + C2-i1.)
- **HLO** : le graphe du run doit **différer** de celui de D0 ; comptage des ops `convert`
  **attendu vs observé** par famille (oracle = table §4 dérivée en G2.3.0). Mismatch → run
  `INVALID`, à corriger avant de continuer.
- **Logits** : `max_abs(Dᵢ vs D0) > 0` requis. Si bit-identique → verdict `VACUOUS` (l'arrondi n'a
  rien arrondi) → investigation, jamais un `SAFE` silencieux.

### 5.4 Sanité (par run)

(Durcissement C2-i3.) Avant toute métrique : scan NaN/Inf sur les 1020×262k logits ; détection de
dégénérescence (entropie moyenne effondrée vs A, répétition de token anormale — seuils calibrés
sur les bras A/B existants et pré-enregistrés). Échec → verdict **`FAIL-SANITY`**, distinct de
`SENSITIVE`, publié, et famille exclue du combiné quels que soient ses ratios.

### 5.5 Interaction (combiné)

(Durcissement C1-i3.) Les one-hot sont **diagnostics, non prédictifs** — seul le run combiné fait
verdict. Publier la métrique d'interaction : `KL p50(D*) / max_i KL p50(Dᵢ actives)` et le ratio à
la somme. L'écart à l'additivité est un résultat, pas une hypothèse cachée.

### 5.6 Stabilité de séquence

(Durcissement C1-i4.) La claim est scopée S46. Test de stabilité minimal : rejouer sur la séquence
S49 (« capital of France », 48 tokens — courte, étiquetée « stabilité faible ») : la config
**combinée** + les **2 familles** les plus proches d'une frontière de bucket. Concordance des
verdicts = claim renforcée ; discordance = publiée telle quelle.

## 6. Gates

### 6.1 G2.3.0 — Neutralité du refactor (bloquant : aucun sweep avant PASS)

- `PrecRt` tout-null doit reproduire la baseline. **Hiérarchie de preuve pré-enregistrée**
  (durcissement C2-i2) :
  - **Gold** : HLO byte-identique au graphe baseline (méthode E1).
  - **Repli** (si diff cosmétique MLIR) : publier le diff catégorisé **ET** exiger cumulativement
    G1 64/64 == HF, E1 4/4, replay complet 1020/1020 == HF. En dessous → refactor **rejeté**.
- Non-régression G2.2 : l'ex-config « tout gemm bf16 » exprimée en `PrecRt` reproduit les métriques
  G2.2 (mêmes ratios vs B, tolérance de lecture near-identique — l'ancien runner comptime est
  retiré après ce check).
- Dérivation + commit de la table des converts attendus par famille (§4).
- Livrable : doc protocole `G2_3_OP_SENSITIVITY.md` committé (pré-enregistrement complet §5).

### 6.2 G2.3.1 — Sweep one-hot

12 runs + D0. Chaque run : sanité (§5.4) → non-vacuité/câblage (§5.3) → métriques (§5.2).
Livrable : classement committé (manifest + doc), toutes familles avec verdict
`SAFE/TOLERABLE/SENSITIVE/FAIL-SANITY/VACUOUS`.

### 6.3 G2.3.2 — Config combinée

- Config initiale = toutes les familles `SAFE`. Critère PASS : **≤ 2× l'enveloppe B appliqué aux
  métriques de verdict du §5.2** (KL p50 vs A, départage max_abs p50) — et non la table
  multi-percentile complète de G2.2 §7.1, qui reste publiée à titre informatif.
- **Procédure d'échec pré-enregistrée** (durcissement C1-i7) : retrait glouton de la famille au
  pire `KL p50 vs D0`, **max 3 essais**, **chaque essai publié** dans le manifest. Si 3 échecs →
  verdict publié : « pas de config combinée SAFE au critère ≤ 2× » — null result assumé.
- Stabilité S49 (§5.6) sur la config finale.

## 7. Contraintes d'exécution

### 7.1 Binaire unique et provenance

(Durcissement C2-i4.) **Un seul build pour tout le sweep** : `g2_3_sweep.sh` calcule le hash du
binaire au départ et **refuse** de continuer s'il change ; hash binaire + rev git du workspace ZML
+ rev du repo dans chaque entrée du manifest. Prérequis build : swap actif sur la VM (check
`smoke.sh`), patch `pjrt.zig` en place.

### 7.2 Budget disque — analyse au fil de l'eau

(Durcissement C2-i6.) Un dump logits f32 ≈ **1,07 Go** ; ~17 runs ≈ 18 Go > ~25 Go libres sur la
VM → risque d'échec nocturne. La boucle d'orchestration fait : run → analyse (52) → consignation →
**purge du dump**. On ne conserve que D0, D\* final, manifest et npz de métriques (~3 Go bornés).
Check d'espace avant chaque run, abort propre sinon. Conséquence assumée : pas de ré-analyse
post-hoc avec une métrique nouvelle (les métriques sont pré-enregistrées ; re-run possible).

### 7.3 Protocole VRAM (`kv_store`)

(Durcissement C2-i7 — leçon G2.1, erreur interne documentée.) GPU **vierge** (aucun process
concurrent), `--no-prealloc`, pic scopé au PID du run, mesure répétée 2×, conditions consignées
dans le manifest. Référence de comparaison : 8 494 MiB (fp32-store, G2.1).

### 7.4 Déterminisme

Vérification run-to-run bitwise sur 1 config répétée (comme G2.0). Greedy → pas de seed.

## 8. Artefacts et critères de succès

**Artefacts versionnés** : `G2_3_OP_SENSITIVITY.md` (protocole AVANT runs + résultats APRÈS),
`fixtures/g2_3_manifest.json`, logs rapatriés M1, tags `gate/G2.3.0-*`, `gate/G2.3.1-*`,
`gate/G2.3.2-*`.

**Succès de G2.3** =
1. G2.3.0 PASS (neutralité prouvée au niveau gold ou repli cumulatif complet) ;
2. 12 familles avec un verdict non-`INVALID` chacune, classement publié ;
3. un verdict combiné publié (PASS **ou** null result) ;
4. manifest auditable de bout en bout (provenance, custody, tous les essais).

Un sweep qui révèle « tout est SENSITIVE sauf X » est un **succès** de l'instrument (c'est une
carto), pas un échec du projet.

## 9. Risques et mitigations

| Risque | Mitigation |
|---|---|
| Refactor runtime casse la neutralité | Gate G2.3.0 bloquante, hiérarchie de preuve pré-enregistrée (§6.1) |
| Flag câblé au mauvais site | Comptage converts attendu/observé (§5.3) |
| Run vacuux classé SAFE | Non-vacuité logits + HLO par run (§5.3) |
| Casse déguisée en bruit médian | Gates de sanité, verdict FAIL-SANITY (§5.4) |
| Références périmées sur disque | Custody md5 + refus au mismatch (§5.1) |
| Disque plein à mi-sweep | Analyse au fil de l'eau + purge + check espace (§7.2) |
| Mesure VRAM contaminée (précédent G2.1) | Protocole VRAM pré-enregistré (§7.3) |
| Cherry-picking métrique post-hoc | Pré-enregistrement métrique/seuils/départage + publication de tout (§5.2) |
| Conclusion combinée non portée par les one-hot | One-hot = diagnostic ; métrique d'interaction publiée (§5.5) |
| Artefact de séquence | Claim scopée + stabilité S49 (§5.6) |
| Binaires hétérogènes dans le sweep | Build unique + hash dans le manifest (§7.1) |
| OOM compile XLA | Swap vérifié avant build (leçon G2, `smoke.sh`) |

## 10. Hors scope / suites possibles

- Étendre à fp8/int8 (le mécanisme `PrecRt` le permettrait — YAGNI ici).
- Run exploratoire « combiné + TOLERABLE » (étiqueté exploratoire) si le combiné PASS avec marge —
  décision au moment des résultats, hors verdict confirmatoire.
- Transfert du classement vers TurboQuant (quantization agressive des familles SAFE) et alambic
  (choix de précision distillation) — hors de ce chantier.
