# G2 — Fidélité bf16 (ZML vs HF en régime de production)

> **Gate `G2`** — 4 juillet 2026. Suite directe de `docs/GPU_PORT_PLAN.md` §5 (« drift bf16 vs
> baseline fp32, **à caractériser au G2** ») et du contrat de précision `docs/P5_7_5_precision_contract.md`
> (signatures drift-vs-bug §6, réutilisées telles quelles ici).
>
> **Question falsifiable** : *ZML reste-t-il dans l'enveloppe de bruit que HF s'autorise à lui-même
> en bf16 — et sinon, quelle opération casse en premier ?*

---

## 1. Pourquoi ce gate (recadrage)

Le résultat central du projet (« 1020/1020 == HF », tags `gen-long-validated-3090` /
`gpu-baseline-validated-3090`) est établi **en fp32** : oracle hybride 46/49 (poids bf16 disque
upcastés fp32) vs moteur ZML fp32. Or fp32 n'est le régime de production de personne — le
checkpoint est bf16, l'inférence réelle est bf16. Le bit-exact fp32 ne dit **rien** sur bf16 :
c'est précisément en basse précision que l'ordre des réductions, le softcapping et les RMSNorm
divergent.

G2 n'est donc **pas** une optimisation VRAM (20→10 Go, bénéfice secondaire) : c'est le test qui
décide si la claim de fidélité du projet est **robuste** ou un **artefact fp32**.

**Insight de design central** : en bf16, « == bit-à-bit » n'existe plus (kernels cuBLAS, ordre de
réduction non spécifié). Le critère de succès doit être **relatif** : on mesure d'abord combien
*HF-bf16 diverge de HF-fp32* (le bruit que l'implémentation de référence s'autorise à elle-même),
puis on exige que ZML-bf16 ne fasse pas pire. Sans cette enveloppe, aucun verdict n'est
interprétable.

---

## 2. Bras expérimentaux

| Bras | Poids | Compute | Implémentation | Rôle |
|---|---|---|---|---|
| **A** | fp32 (upcast) | fp32 | HF hybride (script 46) | vérité de référence — **déjà fait** (G1) |
| **B** | bf16 | bf16 natif | HF sans upcast (script 50) | **enveloppe de bruit** de l'oracle lui-même |
| **C** | bf16 | f32 (`PrecCfg` défaut) | ZML `engine.zig` | le « G2 VRAM » (conversion bf16→f32 exacte) |
| **D** | bf16 | bf16 (`PrecCfg.compute=.bf16`) | ZML `engine.zig` | le vrai régime de production |

Tous les bras sont **teacher-forcés sur la trajectoire de A** (tenseur `fed` de
`gen_long.safetensors`) : on compare des logits sur une trajectoire commune, pas des trajectoires
libres qui bifurquent trivialement. C'est le harnais existant de 46 — aucun changement de format.

---

## 3. Gates et critères (définis AVANT tout run)

### G2.0 — Oracle enveloppe HF-bf16 (script 50)

Mesure `div(B, A)` par step : `max_abs(logits)`, `KL(A‖B)`, position `p0` de première bifurcation
argmax, marge top1−top2 de A (contexte du risque de bifurcation).

- **Résultat en soi** : l'enveloppe. Si HF-bf16 bifurque de HF-fp32 à `p0`, personne ne peut
  exiger mieux de ZML.
- **Sanity requis** : le re-run du bras A (teacher-forcé) reproduit `expected` 1020/1020
  (cohérence avec le fixture 46). FAIL ⇒ environnement changé, stop.
- **Déterminisme requis** : deux passes B consécutives (B1, B2) **bitwise identiques** (logits
  bf16 comparés en u16). FAIL ⇒ figer les kernels (flags cuBLAS/torch) avant de continuer.

### G2.1 — ZML poids bf16, compute f32 (bras C)

bf16→f32 est une conversion **exacte** (élargissement IEEE-754) : les GEMM voient les mêmes
valeurs qu'en G1.

- **PASS** : 1020/1020 == `expected`, logits comparables à G1 (bande de drift P5.7.5 §5),
  VRAM ~÷2.
- **FAIL si un seul token diffère** ⇒ **bug de plomberie** (conversion, layout, gather), pas de la
  physique numérique. Première marche volontairement sûre.

### G2.2 — ZML compute bf16 (bras D) — *l'expérience*

Teacher-forcing sur `fed`. Critères, par step :

- **PASS** : `div(D, A) ≤ 2 × div(B, A)` sur les courbes `max_abs` et `KL` (comparaison
  d'enveloppes, pas point à point : percentiles p50/p95/max), **ET** première bifurcation argmax
  de D pas significativement plus tôt que `p0` (ordre de grandeur : ≥ p0/2).
- **WARN** : ratio dans ]2×, 5×] → diagnostic P5.7.5 §6 (drift concentré et lisse vs marche) avant
  verdict.
- **FAIL** : ratio > 5×, NaN/Inf, ou bifurcation quasi immédiate (p ≪ p0) ⇒ G2.3 obligatoire.

### G2.3 — Cartographie de la casse (si G2.2 non-PASS, ou en bonus)

Balayage `PrecCfg` par bloc : softcap seul en f32, normes seules en f32, softmax seul en f32,
RoPE seul en f32… → localiser **quelle op** rompt la fidélité en premier. Produit le savoir
transférable (quelles ops de Gemma 4 tolèrent la basse précision — alimente TurboQuant/alambic).
Le tri a priori de `GPU_PORT_PLAN.md` §5.2 (normes/softmax/RoPE sensibles, gelu tolérant) est
l'hypothèse à confirmer/infirmer.

---

## 4. Métriques — sémantique

- **Logits, pas argmax** (leçon capitalisée de la non-vacuité, cf mémoire projet : l'argmax greedy
  est trop robuste pour détecter une divergence). L'argmax n'est qu'un indicateur secondaire
  (position de bifurcation).
- `max_abs` : max |logits_X − logits_A| après softcap, en fp32.
- `KL(A‖B)` : sur log_softmax fp32 des logits complets (262 144 classes).
- `margin_A[k]` = top1−top2 des logits A au step k : contextualise les bifurcations (une
  bifurcation sur marge ~0 est attendue ; sur marge large c'est un signal).
- Signatures drift-vs-bug : **réutiliser P5.7.5 §6 verbatim** (monotone/concentré = drift ;
  marche/diffus = bug).

## 5. Ce que ce protocole ne couvre pas

- **Prompts multiples** : G2.0–G2.2 tournent sur le prompt du banc 46 (`[2,105,2048,4095]`,
  1020 tokens). Extension à 2 prompts contrastés (type « Paris » court via 49, prompt code/math)
  = raffinement post-G2.2, même harnais.
- Le non-déterminisme **inter-GPU** (3090 vs autre carte) : hors scope, on caractérise sur la
  3090 uniquement.
- fp16 (mentionné GPU_PORT_PLAN §5) : bf16 d'abord, fp16 éventuel après.

## 6. Artefacts

| Artefact | Rôle | Localisation |
|---|---|---|
| `scripts/50_bf16_envelope_oracle.py` | G2.0 (bras A re-run + B + B2) | repo (versionné) |
| `<ROOT>/g2_logits_a_f32.npy` | logits A memmap [1020, 262144] fp32 (~1,07 Go) | 3090, régénérable |
| `<ROOT>/g2_logits_b_bf16u16.npy` | logits B1 memmap bf16-as-u16 (~0,54 Go) | 3090, régénérable |
| `fixtures/g2_envelope_metrics.npz` | courbes par step (max_abs, kl, match, margin_a) | gitignoré, rapatrié M1 |
| `fixtures/g2_envelope_manifest.json` | summary + percentiles + p0 + verdict déterminisme | versionnable |
| `logs/50_bf16_envelope.log` | sortie console | gitignoré, rapatrié M1 |
| `docs/G2_BF16_FIDELITY.md` §7 | résultats + verdicts (échecs documentés au même titre) | ce fichier |

**Coût** : Tier 1. Pass A ≈ pass 46 (minutes, fp32 GPU) ; B/B2 plus rapides (bf16). Dev = script 50
(fait) + plumbing `PrecCfg` dans `gemma4_gen_long_gpu` (G2.1/G2.2).

---

## 7. Résultats

*(à remplir gate par gate — les FAIL se documentent au même titre que les PASS)*

| Gate | Date | Verdict | Mesures clés |
|---|---|---|---|
| G2.0 | 2026-07-04 | **PASS** (déterminisme + sanity) — enveloppe mesurée | cf §7.1 |
| G2.1 | — | — | — |
| G2.2 | — | — | — |
| G2.3 | — | — | — |

### 7.1 G2.0 — enveloppe HF-bf16 mesurée (3090, run 205s+151s+42s)

- **Sanity pass A** : 1020/1020 argmax == `expected` (l'env reproduit 46). PASS.
- **Déterminisme B1==B2** : PASS **bitwise** (u16, 1020×262144). Les kernels bf16 de la 3090
  sont run-to-run déterministes dans un même process — pas de flag à figer.
- **Enveloppe div(HF-bf16, HF-fp32)** :
  - argmax : **1016/1020** — HF-bf16 ne se reproduit pas lui-même. Première bifurcation
    **p0 = step 21** (pos 25), sur une marge A de 0.137 (pas une marge dégénérée : les 5 marges
    les plus basses de la trajectoire vont de 0.030 à 0.082).
  - `max_abs(logits)` : p50 **0.425**, p95 0.661, p99 0.772, max **1.546**.
  - `KL(A‖B)` : p50 **1.0e-4**, p95 1.7e-3, p99 3.8e-3, max 1.3e-2.
- **Lecture** : le bruit bf16 légitime sur les logits est ~**4 ordres de grandeur** au-dessus du
  drift fp32 du projet (~1e-5 Eigen-vs-BLAS). Malgré ça, l'argmax ne flippe que 4× sur 1020
  (~0,4 %/step) — nouvelle confirmation que l'argmax est un détecteur trop robuste (leçon
  non-vacuité). Le « 1020/1020 » exigible en fp32 n'est PAS un critère exigible en bf16 :
  l'implémentation de référence elle-même ne le tient pas.

**Critères G2.2 désormais chiffrés** (2× l'enveloppe) :

| Métrique | Seuil PASS (D vs A) |
|---|---|
| max_abs p50 / p95 / max | ≤ 0.85 / 1.32 / 3.09 |
| KL p50 / p95 / max | ≤ 2.1e-4 / 3.5e-3 / 2.6e-2 |
| mismatches argmax | ≤ 8 / 1020 (2× les 4 de B) |
| première bifurcation | ≥ step ~10 (p0/2) |

Artefacts : `fixtures/g2_envelope_manifest.json` (versionné), `g2_envelope_metrics.npz` +
`logs/50_bf16_envelope.log` (rapatriés M1), memmaps logits sur 3090 (régénérables).
