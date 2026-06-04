# TurboQuant × ZML V-only — Résultats Q5 (génération)

Gate **Q5** du POC « quantification KV TurboQuant MSE V-only dans le moteur decode ZML de
Gemma-4-E2B-it ». Deux volets :

- **Volet A — COÛT** (HuggingFace, Python) : que coûte la quant V-only sur la génération greedy ?
- **Volet B — PORTAGE** (ZML) : le moteur ZML-V-quant reproduit-il fidèlement la séquence HF-V-quant ?

Méthode : V quantifié 4 bits (`TurboQuantMSE`, rotation Hadamard, nearest-centroid sur N(0,1/d), 1 norm
fp16 par vecteur), K en fp32/fp16. Insertion au point canonique du KV-cache (post-`v_norm`, avant le
transpose/cache). Constantes (codebook + Hadamard, par head_dim 256/512) déterministes et partagées entre
oracle HF et runner ZML (`turboquant_constants.safetensors`, Task 0). Greedy. **Pas de ROUGE-1** (piège
verbatim), pas de MSE-attention.

Tout mesuré sur la RTX 3090. Scripts : `scripts/33_gen_vq_measure.py` (coût),
`scripts/45_gen_vq_oracle.py` (fixture portage), `zml_runner/gemma4_gen_vq.zig` (runner ZML).

---

## Volet A — Coût V-only (HF, 10 prompts, N=48, greedy)

Modèle `google/gemma-4-E2B-it`, bf16, CUDA. `vonly_mse4` (V 4 bits, K fp32) vs `baseline` (KV non quantifié).
Métriques sur le contexte identique avant la première divergence (distributions strictement comparables) :

- **divergence_point** : premier token généré où l'argmax diffère de baseline (48 = jamais divergé).
- **KL** = KL(baseline ‖ V-quant) moyen, **top5** = recouvrement top-5 moyen, **argmatch** = taux d'accord argmax.

| # | divergence_point /48 | top5-overlap | KL moyen | argmax-match | prompt |
|---|---|---|---|---|---|
| 0 | 41 | 0.910 | 0.0099 | 0.976 | Explain why the sky appears blue |
| 1 | 19 | 0.920 | 0.0405 | 0.950 | It is a truth universally acknowledged… |
| 2 | 22 | 0.904 | 0.0143 | 0.957 | Write a Fibonacci function |
| 3 | 14 | 0.920 | 0.0200 | 0.933 | Three tips for improving focus |
| 4 | 30 | 0.916 | 0.0148 | 0.968 | Causes of the French Revolution |
| 5 | 27 | 0.921 | 0.0048 | 0.964 | Translate sentence into French |
| 6 | 48 | 0.913 | 0.0102 | 1.000 | Train speed (reasoning) |
| 7 | 12 | 0.908 | 0.0227 | 0.923 | Five European garden birds |
| 8 | 30 | 0.916 | 0.0050 | 0.968 | How photosynthesis works |
| 9 | 13 | 0.957 | 0.0093 | 0.929 | Steps to make a cup of tea |

**Synthèse (moyenne sur 10 prompts, ≥ 8 requis) :**

| métrique | valeur |
|---|---|
| effective bits (V) | 4.0 |
| divergence_point moyen | **25.6 / 48** (min 12, max 48) |
| prompts ayant divergé | 9 / 10 |
| top5-overlap moyen | **0.918** |
| KL moyen | **0.0152** |
| argmax-match moyen | **0.957** |

**Lecture.** La quant V-only MSE 4 bits est **peu coûteuse mais non gratuite** : la génération reste
identique en moyenne 25 tokens avant de bifurquer, le top-5 est conservé à ~92 %, le KL moyen est faible
(~0.015 nat) et 96 % des argmax coïncident. Un seul prompt (train/raisonnement) ne diverge jamais sur 48
tokens ; les prompts type listes (oiseaux, focus, thé) divergent plus tôt (12-14), cohérent avec une
sensibilité accrue aux petites perturbations là où plusieurs continuations sont quasi-équiprobables. JSON
brut : `/data/gemma4-zml-probe/gen_vq_measure.json`.

---

## Volet B — Portage ZML (fidélité du moteur de génération)

Le moteur de génération ZML `gemma4_gen_vq.zig` est une **copie de `gemma4_decode4.zig`** (boucle decode N
tokens, cache empaqueté threadé, scatter `(.slot,.k=pos)`, réinjection — plomberie prouvée au gate P5.7.8)
avec **un seul ajout** : l'insertion de `quantizeV` au point `v_norm` (branche producer), exactement comme
au gate Q4 (`gemma4_decode_vq.zig`), plus le chargement des constantes codebook/Hadamard. La plomberie
cache/boucle n'a pas été réécrite.

Validation teacher-forcing : la fixture (`scripts/45_gen_vq_oracle.py`) feed les tokens HF-V-quant
pré-gatherés et l'oracle HF lui-même quantifie V au même point. Critère : argmax ZML-V-quant == token
suivant HF-V-quant à chaque step.

| step | position | argmax ZML-V-quant | argmax HF-V-quant | verdict |
|---|---|---|---|---|
| 0 | 4 | 107 | 107 | PASS |
| 1 | 5 | 1 | 1 | PASS |
| 2 | 6 | 106 | 106 | PASS |
| 3 | 7 | 1 | 1 | PASS |

**Portage V-quant : 4/4 tokens argmax-match — séquence ZML-V-quant == HF-V-quant greedy.**

Le portage n'ajoute donc aucune erreur au-delà de la quantification elle-même : le quantizer ZML (prouvé
bit-exact en Q3, inséré sans régression en Q4) reproduit fidèlement la génération HF sous V-quant. Fixture :
`/data/gemma4-zml-probe/decode_vq_gen.safetensors` ; manifeste : `decode_vq_gen_manifest.json`.

---

## Gain de compression (théorique)

V-only : chaque vecteur de `head_dim` éléments est encodé sur **4 bits/élément + 1 scalaire de norme fp16**
(16 bits par vecteur). Référence fp16 = 16 bits/élément.

| head_dim | original (fp16) | V-quant (4 b + 1 norm fp16) | ratio |
|---|---|---|---|
| 256 (sliding) | 256 × 16 = 4096 bits | 256 × 4 + 16 = 1040 bits | **3.94×** |
| 512 (full) | 512 × 16 = 8192 bits | 512 × 4 + 16 = 2064 bits | **3.97×** |

Asymptotiquement le ratio tend vers 4× (le surcoût d'une norme fp16 par vecteur est négligeable dès
head_dim ≥ 256). **C'est un gain sur la moitié V du KV-cache uniquement** (K reste fp16) : sur un cache où
K et V occupent autant de place, la réduction du cache total est ≈ (1 + 1/3.95) / 2 ≈ **0.63×**, soit ~37 %
d'économie sur le KV-cache. Quantifier aussi K (hors périmètre de ce POC) rapprocherait le cache complet
du 4×.

---

## Limites (périmètre du POC)

- **V-only.** K n'est pas quantifié (fp16/fp32). La mesure de coût et le gain ne valent que pour la branche
  V du cache ; le coût d'une quant K+V (étudiée en amont mais hors gate Q5) n'est pas mesuré ici.
- **Précision.** Génération HF en bf16 ; chaîne ZML en fp32 (poids convertis), norm arrondie en fp16 pour
  matcher `turboquant.py`. Le portage est validé argmax-exact, pas bit-exact sur les logits (logits
  amplifiés par le lm_head ; les critères liants restent argmax + last_hidden bit-near, cf gate Q4).
- **NUM_STEPS modeste.** Le moteur de génération ZML (hérité de decode4/P5.7.8) tourne sur **4 tokens** à
  partir d'un prompt synthétique court (`[2,105,2048,4095]`, kmax=8). Le but du volet B est la **fidélité du
  portage** (argmax ZML == argmax HF-V-quant), pas la longueur ni la qualité du texte ; le coût/qualité réel
  de la génération est mesuré côté HF (volet A, N=48, 10 prompts naturels). Étendre la boucle ZML à N=48
  demanderait d'agrandir kmax et la fixture (caches paddés), sans changer la mécanique d'insertion.
- **Échantillon coût.** 10 prompts (≥ 8 requis), N=48, greedy uniquement (pas de sampling). Suffisant pour
  un signal de coût agrégé, pas pour une évaluation statistique fine par domaine.

## Conclusion

Le quantizer TurboQuant MSE V-only est **porté avec succès dans le moteur de génération decode ZML** de
Gemma-4-E2B-it : génération ZML-V-quant == HF-V-quant (4/4 argmax), et le coût HF mesuré est faible
(divergence ~25 tokens en moyenne, top5 ~0.92, KL ~0.015, argmax-match ~0.96) pour un gain de compression
de ~3.95× sur la branche V du KV-cache. Le POC TurboQuant × ZML V-only (gates Q1→Q5) est complet.
