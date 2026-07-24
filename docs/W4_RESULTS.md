# Poids 4-bit w4a16 (compressed-tensors) dans ZML — résultats du jalon J1 (E2B)

> Spec : `docs/superpowers/specs/2026-07-18-w4-poids-4bit-12b-design.md`
> Plan exécuté : `docs/superpowers/plans/2026-07-24-w4-j1-brique-e2b.md`
> Machine : VM 3090 (24 GiB). Exécution : 24 juillet 2026, en une session, subagent-driven,
> aucun échec de build (3 builds verts du premier coup).
> Convention du repo : la doc porte les résultats (logs/ gitignorés) ; FAIL/null publiés comme les PASS.

## Objet

Brique **`dequantW4`** : des poids **int4 groupés (format w4a16 compressed-tensors)** dépliés en
bf16 **dans le graphe compilé**, prouvée de bout en bout sur **Gemma-4-E2B** — jalon **J1** de la
spec W4 (cible finale : Gemma 4 **12B** sur la 3090, jalon J2, au tiroir). Le moteur partagé
**`engine.zig` n'est pas modifié d'un octet** (prouvé au gate WN) : la brique s'assemble en amont
(pattern wrapper `W4Step`, précédent `StepTok`/L3) et délègue à `forwardStep` inchangé.
Résultat central : **décode GPU 48/48 == HF-même-checkpoint** à 40,9 tok/s, pic VRAM réel
**10 524 MiB (−37 % vs bf16)**.

## Le format w4a16 compressed-tensors (vérifié sur le checkpoint Google 12B)

Par Linear `[out, in]` :

| Tenseur | Dtype / shape | Sémantique |
|---|---|---|
| `weight_packed` | I32 `[out, in/8]` | 8 nibbles par mot i32, **little-endian** : le nibble j du mot w = colonne `8w+j` ; chaque nibble stocke **q+8** (non signé) |
| `weight_scale` | BF16 `[out, in/32]` | un scale par **groupe de 32 le long de l'entrée** |
| `weight_shape` | I64 | shape d'origine (ignoré par ZML) |

- **Dequant** : `(nibble − 8) × scale`, en bf16.
- **Quantize** (recette) : `scale = max_abs/7.5` ; `q = round-half-even(clamp(x/scale, −8, 7))`.

## Les 6 gates — tous PASS (24 juillet 2026)

| Gate | Verdict | Chiffres | Commit | Tag |
|---|---|---|---|---|
| W0 — checkpoint E2B-W4 validé | **PASS** | 276 linears auto-vérifiés ; round-trip HF `run_compressed=False` ; décode 48 tok CPU sur w4dq → « Paris », détok round-trip 48/48 | `536f1b9`+`75a5a80` | `gate/w4-w0-pass` |
| W1 — unpack ZML bit-exact | **PASS** | 64/64 entiers, 64/64 u16 dequant ; contre-test corruption : **exactement 1 élément divergent à l'index prédit** ; build vert du 1er coup (145 s) | `e5f48c4` | `gate/w4-w1-pass` |
| W2 — dequant checkpoint réel | **PASS** | 5/5 familles de shape, **bit-exact bf16**, ~42,5 M u16 comparés | `e08cf76` | `gate/w4-w2-pass` |
| W3 — premier GEMM (q_proj L0) | **PASS** | max_abs **7.75e-7** (~130× sous le seuil 1e-4), mean_abs 9.47e-8, 8/8 points fixes | `262581e` | `gate/w4-w3-pass` |
| W4g — décode GPU complet | **PASS** | **48/48 == HF-même-checkpoint** ; 40,9 tok/s ; pic VRAM réel **10 524 MiB** (−37 % vs 16 658 bf16) | `195e32e` | `gate/w4-w4g-pass` |
| WN — non-régression moteur | **PASS** | `engine.zig` diff main = **0 ligne** ; témoin `gen_auto` re-PASS 48/48 (91,7 tok/s) ; smoke 4/4 builds | `cb930ee` | `gate/w4-wn-pass` |

### W0 — checkpoint validé (recette Google rejouée sur E2B)

Google ne publie pas de checkpoint w4a16 pour l'E2B : la recette du 12B a été **rejouée**
(llm-compressor 0.12.0, oneshot **data-free**, group 32 symétrique, observer
`memoryless_minmax`, ignore lm_head + embeddings + tours vision/audio) → `weights_w4/`
(7,8 Go, mono-fichier). **276 linears** packés — l'E2B est YOCO : 15 producers ×9 +
20 readers ×7 (sans k/v/k_norm) + `per_layer_model_projection`. Vérifications : auto-vérifs
des 276 linears, round-trip HF (`run_compressed=False`), et un **décode 48 tokens CPU sur la
décompression de référence `weights_w4dq/`** (bf16, 1951 tenseurs = 2011 − 60 vestigiaux) →
« Paris », détok round-trip 48/48.

### W1/W2 — unpack et dequant bit-exact vs la référence compressed-tensors

`unpackW4` en **extraction logique** (`shiftRightLogical` + AND 0xF, puis −8) ; dequant par
`splitAxis` en groupes `g/gi=32` + mul du scale broadcasté. Référence :
`pack_to_int32`/`unpack_from_int32` de compressed-tensors. W1 prouve le bit-exact sur fixture
(+ contre-test de corruption à divergence localisée) ; W2 le prouve sur **les 5 familles de
shape du checkpoint réel** (~42,5 M u16, 5/5). Déviation déclarée vs spec §3.5 (« entier ») :
couverture **par famille de shape + e2e**, pas l'intégralité des 276 linears un à un.
Variante `bitCast(.u4)` : compile et matche sur CPU — piste consignée, **non retenue**.

### W3 — premier GEMM

q_proj L0, flux bf16→f32 (= `dotPrec` fam=null, le flux du moteur) : max_abs **7.75e-7**
vs oracle torch fp32.

### W4g — décode GPU complet

`gemma4_w4auto` (clone ciblé de `gen_auto`, **32 lignes de code de diff** : `W4Step` assemble
un `Model` par valeur avec les poids `dequantW4` et délègue à `forwardStep` **inchangé**) :

- **48/48 == HF-même-checkpoint** (oracle `scripts/56` sur le checkpoint W4) ; marge top1−top2
  min **0.003031** (step 24) ; le run `--no-prealloc` est aussi 48/48.
- **40,9 tok/s** en génération.
- Pic VRAM **réel** : **10 524 MiB** (`--no-prealloc`, protocole G3) vs 16 658 MiB bf16 → **−37 %**.
- Démo FR cohérente hors fixture (définition de la quantization 4-bit, 38,2 tok/s, early-EOT).

### WN — le moteur n'a pas bougé

`git diff main -- engine.zig` = vide (0 octet) ; le témoin `gemma4_gen_auto` re-PASS 48/48
dans la même session (91,7 tok/s) ; smoke 4/4 builds. Dans `w4.zig`, les readers YOCO portent
un **placeholder inerte prouvé mort au traçage** (`engine.zig:415-423` ne consomme pas k/v/k_norm
des readers).

---

## Finding — l'invariance d'échelle des norms rend le contre-test scale aveugle

Le contre-test de non-vacuité prévu (corrompre `weight_scale` et exiger un FAIL) a d'abord
**passé à vide** : une corruption **UNIFORME** des scales de q_proj (×4 puis ×100) **ne flippe
pas l'argmax**. Cause structurelle, pas numérique : `q_norm` (RMSNorm, `engine.zig:408`) annule
tout facteur uniforme (**RMSNorm(c·x) == RMSNorm(x)**). Idem k/v (k_norm/v_norm) et
o/down/up (sandwich norms).

> **10 des 11 familles de linears de Gemma 4 sont scale-invariantes par construction** —
> seule **`gate_proj`** traverse une non-linéarité (gelu) avant toute norm.

Contre-test retenu : **gate_proj L17 ×100** → divergence au step gen=4, erreur levée par le
banc avec diagnostic logits. NB : l'erreur s'appelle **`A1Mismatch`** (héritée du témoin
`gen_auto`), pas `GenMismatch` comme le plan l'écrivait.

**Corollaires** :
- (a) re-confirmation de la leçon du 28 juin : **l'argmax greedy est trop robuste pour prouver
  la non-vacuité** — et ici même les *logits* d'un chemin normé sont insensibles à un facteur
  uniforme ; il faut viser la seule famille qui traverse une non-linéarité.
- (b) éclairage sur la **robustesse du 4-bit** : les erreurs de scale par-rangée sont absorbées
  par les norms — une partie du bruit de quantization est structurellement invisible en sortie.

Logs de preuve (gitignorés, `logs/`) : `w4g_corrupt.log` (q_proj ×4), `w4g_corrupt100.log`
(q_proj ×100), `w4g_corrupt_gate.log` (gate_proj — le retenu).

---

## Observations

- **Coût du dequant par step : ×2,2** — 40,9 tok/s (W4) vs 91,7 tok/s (témoin bf16, même
  session). R4 assumé : **correctness first**, la perf n'a jamais été un critère du jalon.
- **Le gain VRAM E2B est borné par les embeddings bf16** : `embed_tokens_per_layer` (4,7 G) +
  `embed_tokens` (0,8 G) restent en bf16 — c'est **ATTENDU** : la brique vise le 12B, où les
  linears dominent le budget.
- **Variante `bitCast(.u4)`** : compile et matche sur CPU — piste plus directe que l'extraction
  logique, non retenue pour J1 (l'extraction logique est celle prouvée sur GPU).
- **Aucun échec de build** sur tout le chantier (3 builds verts du premier coup) — le pattern
  « spec revue + plan revu + subagents » tient.

## Environnement et reproduction

Chemins VM anonymisés (`user@gpu-host`) — voir `deploy_to_3090.sh` pour le déploiement.

- **venv quantization/fixtures** : `/data/venvs/w4quant` — llm-compressor 0.12.0,
  compressed-tensors 0.17.1, transformers 5.9.0, torch 2.12.0.
- **venv oracles de génération** : `gemma4-probe` (+ compressed-tensors 0.17.1).
- **Référence de packing** :
  `from compressed_tensors.compressors.pack_quantized import pack_to_int32, unpack_from_int32`.
- **Artefacts VM (hors git)** : `weights_w4/` (checkpoint packé 7,8 Go), `weights_w4dq/`
  (décompression de référence bf16), fixtures `w4_unpack`/`w4_mats`/`w4_gemm` + `w4_gen48`.

```bash
# 1. Quantizer l'E2B (recette Google 12B rejouée) puis produire la référence dequant
python scripts/54_w4_quantize.py          # → weights_w4/
python scripts/55_w4_dequant_export.py    # → weights_w4dq/ (référence bf16)

# 2. Oracle de génération HF sur le checkpoint W4 (fork du 49)
python scripts/56_w4_gen_oracle.py        # → fixture w4_gen48

# 3. Fixtures unitaires + gates W1/W2/W3
python scripts/57_w4_unpack_fixture.py && python scripts/58_w4_gemm_oracle.py
./bazel.sh run //examples/rqz:gemma4_w4gate -- ... --gate w1|w2|w3

# 4. Décode GPU complet (W4g) — et contre-test gate_proj (59)
./bazel.sh run --@zml//platforms:cuda=true //examples/rqz:gemma4_w4auto -- \
  /data/gemma4-zml-probe/weights_w4/model.safetensors <tokenizer.json> --oracle w4_gen48.safetensors
python scripts/59_w4_corrupt_ckpt.py      # corruption ciblée (non-vacuité)
```

Logs de preuve dans `logs/` (gitignoré) : `w4g.log`, `w4g_corrupt.log`, `w4g_corrupt100.log`,
`w4g_corrupt_gate.log`, `w4_demo.log`, `w4g_noprealloc.log`, `w4_vram.log`, `w4_wn.log`.

---

## État J2 (12B) — au tiroir, GO = décision Régis

Conformément à la **règle d'arrêt de la spec**, J1 s'arrête ici ; le passage à J2
(`Gemma4Unified`, 12B sur la 3090) est une décision de Régis. Ce qui est **déjà en poche**
(manifest du checkpoint vérifié le 24 juillet 2026) :

- `google/gemma-4-12B-it-qat-w4a16-ct` : **328 linears packés** (g32 sym), mono-fichier
  **9,56 GiB** ; ~6 Go de linears packés + embeddings bf16 → **VRAM projetée ~10-12 Go sur 24**.
- ⚠ Les **8 couches full du 12B n'ont PAS de v_proj** (`attention_k_eq_v=true`) — à porter à
  la carte des diffs U0.
- Le **tokenizer est identique octet à octet** à celui de l'E2B.
