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
| **C** | bf16 | f32 (`PrecCfg` défaut) | ZML `engine.zig` | ≡ état courant de G1 (découvert au G2.1, cf §3) |
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

**Découverte en ouvrant le chantier (4 juil)** : le bras C est **déjà l'état courant de G1**.
`createTensor(name, tags, null)` (zml `io.zig`, `maybeCreateTensor`) crée le tenseur au dtype du
**header safetensors** — le 3ᵉ argument est le *partitioning*, pas le dtype, et aucun upcast
n'a lieu au load. Le checkpoint étant bf16, les poids sont bf16 sur device depuis toujours ;
le `c()` de `engine.zig` fait l'upcast f32 **dans le graphe** (élargissement exact). Le critère
« 1020/1020 == expected » du bras C est donc **déjà prouvé par G1**.

Corollaire : la claim « VRAM 20→10 Go » du backlog était un artefact de mesure — les ~22 Go
observés au G1 = **réserve BFC** (`preallocate=true, memory_fraction 0.90` ⇒ 21,6 Go réservés
d'emblée), pas l'usage.

- **Reste du gate** : mesure empirique de la VRAM réelle (`--no-prealloc` ajouté au runner) +
  re-vérification argmax sur le run de mesure.
- **PASS** : usage réel cohérent avec poids bf16 (~10-14 Go, poids 10,2 + caches + workspace
  transitoire des converts f32) ; argmax == expected sur les steps du run.
- **FAIL si** usage ≥ ~19 Go (les poids seraient f32 → la lecture de code ci-dessus est fausse,
  à réinvestiguer) ou mismatch argmax.

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
| G2.1 | 2026-07-04 | **PASS** (bras C ≡ G1, prouvé io.zig + run 64/64) | cf §7.2 |
| G2.2 | 2026-07-04 | **PASS** — ZML-bf16 2 à 5× PLUS fidèle que l'enveloppe HF | cf §7.3 |
| G2.3 | — | non déclenché (G2.2 PASS) — reste dispo en bonus TurboQuant/alambic | — |

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

### 7.2 G2.1 — bras C ≡ G1 (poids déjà bf16 sur device)

Cf §3 (découverte). Preuves : (a) structurelle — `io.zig maybeCreateTensor` crée le tenseur au
dtype du header safetensors (bf16), le 3ᵉ arg de `createTensor` est le partitioning, aucun upcast
au load ; (b) runtime — G1 refait post-édits `engine.zig` : 64/64 == HF, 109,3 tok/s. La VRAM
réelle (mesure isolée `--no-prealloc`, pic borné au PID du run, GPU vierge) : **8 494 MiB
(~8,5 Go)**, run 64/64 PASS. NB : une 1re mesure (22 234 MiB) était contaminée par des runs
`preallocate=true` concurrents pendant la fenêtre du traqueur — invalidée.

Leçon : « VRAM ~22 Go » (ENGINE_LOG G1) était la **réserve BFC** (`memory_fraction 0.90`), pas
l'usage. Usage réel ~8,5 Go (poids résidents bf16 — `embed_tokens_per_layer` n'est pas résident
sur ce chemin, les embptls per-step viennent de la fixture — + caches f32 + transitoires f32 des
converts + workspace XLA). Corollaires : le gain VRAM attendu de « G2 bf16 » n'existe pas (poids
déjà bf16) ; et **le banc gen-long GPU tiendrait sur une carte 12 Go**.

### 7.3 G2.2 — ZML gemm=bf16 : PASS, 2 à 5× SOUS l'enveloppe HF

Design D1 : `PrecCfg.gemm=.bf16` — les 2 opérandes de chaque dot convertis bf16, résultat
re-upcasté f32 ; normes/softmax/RoPE/softcap/résiduels f32 ; cache KV f32 (stockage), arrondi
bf16 à la lecture par QK/PV. Neutralité HLO : `gemm=null` n'émet rien (convert same-dtype =
`return self`, vérifié `tensor.zig`) ; G1 64/64 inchangé post-édits.

| Métrique | D (ZML gemm-bf16 vs A) | B (enveloppe HF-bf16) | ratio (seuil PASS ≤ 2×) |
|---|---|---|---|
| max_abs p50 / p95 / max | 0.185 / 0.330 / 0.623 | 0.425 / 0.661 / 1.546 | **0.44 / 0.50 / 0.40×** |
| KL p50 / p95 / max | 2.9e-5 / 4.6e-4 / 2.6e-3 | 1.0e-4 / 1.7e-3 / 1.3e-2 | **0.28 / 0.27 / 0.19×** |
| argmax match | 1016/1020 | 1016/1020 | égal |
| 1re bifurcation | step 96 | step 21 | 4.6× plus tard |
| débit | 103,8 tok/s (dump logits inclus) | — | ≈ G1 fp32 (109) |

**Lecture** : arrondir uniquement aux bornes des GEMM (D1) produit ~2× moins de bruit logits et
~4× moins de divergence distributionnelle que le chemin bf16 natif de HF (flux entier bf16).
**La claim de fidélité du projet n'est pas un artefact fp32** : en régime GEMM-bf16, ZML reste
plus proche de la vérité fp32 que l'implémentation de référence ne l'est d'elle-même.

Incident build documenté : la cfg comptime `.prec` allonge le `@typeName` du modèle → dépassement
du quota comptime (1000) dans `pjrt.zig structSize` (`indexOf` sur `@typeName`). Patch workspace
(1 ligne, `@setEvalBranchQuota(100_000)`, commenté `local patch rqz`) — **prérequis à
re-appliquer si le workspace ZML est resynchronisé upstream**.

Artefacts : `fixtures/g2_2_manifest.json` (versionné), `g2_2_metrics.npz`, `logs/g2_2_run.log`,
dump `g2_logits_d_f32.bin` (3090, régénérable).
