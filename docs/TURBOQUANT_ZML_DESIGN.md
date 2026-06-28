# TurboQuant × ZML — Design : quantizer KV V-only en ZML (POC)

**Date** : 2026-06-04
**Statut** : design validé (brainstorming + creusage A/B/C), prêt pour plan d'implémentation
**Repo** : gemma4-zml-probe (moteur decode ZML de Gemma-4-E2B-it)

---

## 1. Objectif

Brancher la quantification KV-cache TurboQuant sur le moteur decode ZML (Gemma-4-E2B-it,
forward+decode+génération == HuggingFace prouvés). **Mesurer le coût/gain sur de la vraie
génération**, la fidélité ZML==HF servant de garde-fou.

Ce POC porte le **quantizer MSE V-only** dans `gemma4_decode3.zig`, validé gate-par-gate
contre un oracle PyTorch, et mesure la divergence de génération.

## 2. Pourquoi ce scope précis (findings du creusage 2026-06-04)

Trois mesures empiriques (détail dans `turboquant.md`) cadrent ce design :

- **Caractérisation K/V** : var K layer 0 = 0.016, var V = 1.0 (normalisé par `v_norm`).
  Distribution ultra-saine (mécanisme = `k_norm`/`v_norm` RMSNorm scalé). head_dim 256 et
  512 sont **pow2** → Hadamard direct, sans split. n_kv=1 → pas de per-head.
- **Coût génération (test A)** : **V-only à 4 bits préserve la génération** (KL 0.02,
  argmax-match 0.95). Quantifier K coûte cher *malgré* sa variance basse (K est
  softmax-sensible). **MSE simple ≥ PROD (QJL)** à budget égal → le QJL et le split sont
  superflus. Le **quantizer minimal MSE est optimal**. (Ce résultat **supersède** la
  conclusion « terrain dégagé pour K+V » de la 1ʳᵉ caractérisation : la variance K basse ne
  suffit pas — ne pas ressusciter la voie K+V-direct sur la seule base de la variance.)
- **Spike B (ZML)** : `Hadamard + nearest-centroid + gather` compile et est **bit-exact**
  en ZML (2048/2048 coords, 0 flip). Le portage est faisable en pratique.

**Décisions de scope** :
- Cible = **V-only**, quantizer = **`TurboQuantMSE` Hadamard** (pas PROD, pas split, pas per-head).
- K+V asymétrique (K à plus de bits) = **spec suivant**, hors de ce POC.
- POC = **fake-quant** (distorsion `quantize→dequantize` en fp32 dans le graphe). Le gain
  compression est **théorique** (b bits/élément + 1 fp16 norm par vecteur). Le **packing
  mémoire réel** (gain effectif) = extension ultérieure (moteur déployable).

## 3. Architecture

Trois unités à frontières nettes :

```
[PyTorch offline]            [.safetensors]              [ZML / Zig]
 constantes MSE      ──►   codebook[K], Pi[d,d]    ──►   quantizer V-only
 (pas de calib data)        (par head_dim 256/512)        inséré au point cache
                                                          de gemma4_decode3.zig
                                                                  │
                                              [mesure] ◄──────────┘
                              génération ZML-quant vs HF-quant (portage)
                                          ZML-quant vs baseline (coût)
```

### 3.1 Constantes (offline, PyTorch)

`TurboQuantMSE(d, b, rotation='hadamard')` ne nécessite **aucune calibration données**
(codebook fitté Lloyd-Max sur N(0,1/d) synthétique ; normalisation par vecteur intrinsèque).
Le codebook (échelle ∝ 1/√d) ET Pi (taille d) **dépendent de `d`** → on exporte **deux jeux
distincts**, un pour chaque head_dim :
- `codebook_256`, `hadamard_256` (sliding) et `codebook_512`, `hadamard_512` (full)
- `codebook_*` : `[K=2^b]` f32 (centroïdes scalaires Lloyd-Max)
- `hadamard_*` : `[d, d]` f32 (Pi Sylvester/√d, orthogonale, Pi⁻¹ = Pi.T)

**Sélection runtime** par le flag `full` du producer (le dispatch `isFull(i)` déjà présent
`gemma4_decode3.zig:174`, qui branche déjà HD_SLIDING/HD_FULL) : `{cb_256,Pi_256}` vs
`{cb_512,Pi_512}`.

Format : `.safetensors` (consommé nativement par `zml.safetensors.TensorRegistry.fromPath`,
pattern déjà rodé dans le projet). Script `scripts/NN_export_turboquant_constants.py`.

### 3.2 Quantizer ZML V-only (le cœur)

Chaîne d'ops, appliquée à V (`[.k positions, .hd]`) — le spike a validé Hadamard + nearest-centroid ;
restent norm/scale et inverse-Hadamard.

**Convention d'axes** : Pi est tagué `[.e, .hd]` ; le codebook porte `.c` (centroïdes). Le flux
d'axes est `.hd → .e → (recherche sur .c) → .e → .hd`. Le `.c` du codebook est **distinct** de
`.k` (axe-position du cache) — voir Q3 (rename obligatoire).

```
norm = sqrt(sum(v*v, .hd))                 # L2 par vecteur (sum tensor.zig:1410 + sqrt:1954)
u    = v.div(norm)                          # broadcast (div:984, pattern x.div(x.sum(.a)))
y    = u.dot(Pi, .hd)                       # Hadamard -> axe .e   [spike: bit-exact]
sq   = (y_b - cb_b)^2   (broadcast .c)      # distance² sur [.k,.e,.c]   [spike]
idx  = argMax(sq.scale(-1), .c).indices.squeeze(.c)   # argmin -> [.k,.e]   [spike]
y_hat= cb.gather(.{.c = idx}, .{})          # reconstruction -> [.k,.e]   [spike]
u_hat= y_hat.dot(Pi, .e)                    # inverse-Hadamard (Pi orthogonale) -> .hd
v_hat= u_hat.mul(norm)                      # ×norm
```

Pièges ZML capitalisés (spike B) : `gather` exige `squeeze` de l'axe réduit d'`argMax` ;
broadcast rang-N via `appendAxes`/`insertAxes` + `Shape.init(...).withTags(...)` + `.broad` ;
distance² évite `abs`/`sqrt`.

### 3.3 Insertion dans le moteur decode

Point d'insertion V (cf cartographie `gemma4_decode3.zig`, producers 0-14) :
- **write** : après `v_norm`, avant `scatterSlices` au cache (~L230) → on stocke `v_hat`.
- (V-only : pas de quantization de K ; K reste fp32.)

Pour un fake-quant, remplacer `v` par `v_hat = dequant(quant(v))` au write suffit : le cache
contient les valeurs déquantifiées, exactement ce qu'un cache quantifié restituerait.

## 4. Plan de validation gate-par-gate

Discipline oracle/gate du projet (chaque gate : oracle PyTorch → ZML → compare → commit+tag).

| Gate | Contenu | Oracle | Critère |
|---|---|---|---|
| **Q1** | Hadamard `u.dot(Pi)` | numpy/torch `u @ Pi.T` | bit-near (~1e-5) ✅ *(spike)* |
| **Q2** | nearest-centroid (argmin+gather) | `cb[argmin (y-cb)²]` | bit-exact ✅ *(spike)* |
| **Q3** | chaîne MSE complète `v→v_hat` | `MSE.dequant(MSE.quant(v))` | max_abs ≤ 1e-4 |
| **Q4** | insertion 1-step decode (V quant) | oracle decode HF **V-quant** | `last_hidden` max_abs ≤ 1e-2 **ET** mean_abs ≤ 1e-4 |
| **Q5** | génération N tokens | baseline + HF-quant | voir §5 |

Q1+Q2 sont déjà couverts par le spike `gemma4_hadq.zig`. Q3 ajoute norm/scale + inverse-Hadamard.

**Note Q3 (rename d'axe, load-bearing)** : dans le decode, V porte déjà `.k` comme axe-position
du cache (`gemma4_decode3.zig:222` rename `.s→.k`, scatter sur `.k` L230). Le codebook **doit**
donc être tagué `.c` (et non `.k` comme dans le spike isolé) : sans ce rename, le `gather` panique
« axis appears more than twice » (cf piège #2 du spike). C'est un rename obligatoire, pas cosmétique.

**Note Q4 (tolérance & référence)** : l'oracle decode du projet exige `max_abs ≤ 1e-2 ET
mean_abs ≤ 1e-4` (`gemma4_decode3.zig:354` ; un seul `max_abs` = WARN). Cette barre **bit-near**
s'applique à la comparaison **ZML-quant vs HF-quant à constantes identiques** (l'erreur mesurée =
le portage seul, qui ne doit rien ajouter → bit-near attendu), **PAS** à ZML-quant vs baseline
(là, l'écart = la distorsion de quant, légitimement plus grand — c'est le COÛT mesuré en §5).

## 5. Critère de succès (double référence)

Deux comparaisons à ne pas confondre (cf creusage C) :

- **Validation du PORTAGE** (ZML-quant vs HF-quant, mêmes constantes) — le garde-fou :
  - `last_hidden` (1 step decode) : max_abs ≤ **1e-2** **ET** mean_abs ≤ **1e-4**
    (les deux bornes de l'oracle projet `gemma4_decode3.zig:354` ; un seul max_abs = WARN).
    Bit-near attendu car le portage ne fait que reproduire la *même* quant que HF.
  - séquence greedy N tokens : argmax ZML == argmax HF-quant sur ≥ **95 %** des tokens.
  - *Le portage ne doit ajouter aucune erreur au-delà de la quantification elle-même.*
- **Mesure du COÛT de quant** (ZML-quant vs baseline non-quant) — l'information cherchée :
  - divergence-point moyen, KL moyen, top5-overlap, argmax-match (métriques de C).
  - **PAS de ROUGE-1** (piège verbatim documenté), pas de MSE-attention seule.

Protocole : greedy, N ≥ 48 tokens, ≥ 8 prompts variés (élargir l'échantillon vs le test A
à 4 prompts). Gain compression rapporté en théorique (b bits + 1 fp16 norm / vecteur de d).

## 6. Périmètre — dans / hors

**Dans** : V-only, MSE Hadamard, fake-quant (distorsion), gates Q1-Q5, mesure génération.
**Hors (specs suivants)** : quantization de K, budget asymétrique K/V, packing mémoire réel
(sous-byte), contextes longs, QJL/split/per-head (prouvés superflus ici).

## 7. Risques & limites

- **K non quantifié** : le gain mémoire V-only est modeste (~½ du cache). Assumé pour le POC ;
  l'intérêt est de prouver qu'un quantizer vit dans le moteur ZML, base pour K+V.
- **bf16 runtime vs fp32 mesure** : le moteur tourne bf16 ; la distorsion mesurée reflète le
  runtime réel. Cohérent avec le projet.
- **Échantillon génération** : élargir à ≥ 8 prompts (le test A à 4 prompts montrait une
  variance inter-prompt forte).
- **norm/inverse-Hadamard non encore compilés** : Q3 les ajoute ; risque faible (ops simples,
  `dot`/`mul`/reduce déjà éprouvés dans le moteur).

## 8. Artefacts

- Spike : `gemma4_hadq.zig` (+ cible bazel), oracle `spike_hadq_oracle.py`, fixture
  `spike_hadq.safetensors` (3090 `/data/rqz_workspace/zml/examples/rqz/` et `/data/gemma4-zml-probe/`).
- Tests de creusage : `measure_k_distribution_gemma4.py`, `test_kv_quant_generation.py` (+ JSON).
- Tout sur la 3090 + copies M1 `/tmp/`. Repo local-only sauf décision de push.
