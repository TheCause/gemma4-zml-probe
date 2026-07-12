# gemma4-zml-probe — Documentation complète

> Documentation de référence du projet : ce qu'il est, ce qu'il sait faire, comment l'utiliser.
> Rédigée le 9 juillet 2026 (état = fin du chantier G2, PR `generation-longue` → `main`).
> Entrée rapide en anglais : [`README.md`](../README.md). Carte visuelle : [`CARTOGRAPHIE_portage.md`](CARTOGRAPHIE_portage.md).

---

## 1. Vue d'ensemble

**gemma4-zml-probe** est un portage **op-par-op, prouvé bit-près**, du chemin texte de
**`google/gemma-4-E2B-it`** vers **[ZML](https://github.com/zml/zml)** (Zig + MLIR + OpenXLA/PJRT),
validé en permanence contre HuggingFace Transformers pris comme oracle.

Le but n'est **pas** « faire tourner Gemma 4 » (Ollama, llama.cpp, vLLM le font déjà). Le but est un
**moteur de référence contrôlé au niveau de l'opération** :

- il reproduit le modèle **bit-près vs PyTorch** — une baseline fp32 prouvée contre laquelle mesurer ;
- c'est un substrat propre pour **expérimenter au niveau du graphe** (quantization custom, tricks de
  KV-cache, recherche d'architecture) — ce que les runtimes clef-en-main n'exposent pas ;
- il ajoute **Gemma au catalogue ZML** (l'upstream ne livre que Llama / Qwen / LFM).

C'est une **baseline de recherche**, pas un moteur de production : fp32 (avec un régime bf16 validé),
mono-séquence, sans sampling ni fast-prefill.

### Fiche d'identité

| | |
|---|---|
| Modèle source | `google/gemma-4-E2B-it` (texte seul ; vision/audio hors scope) |
| Cible | ZML — forward compilé AOT en binaire natif, zéro Python au runtime |
| Fidélité | **1020/1020 tokens == HF greedy** (CPU chunké ET GPU fp32) ; bf16 : 2–5× sous l'enveloppe HF |
| Perf GPU | **109 tok/s** (RTX 3090, fp32, mono-graphe) ; VRAM réelle **~8,5 Go** (tiendrait sur 12 Go) |
| Méthode | ~50 gates atomiques, chacune committée + taggée, oracle PyTorch indépendant |
| Licence | Apache-2.0 (comme ZML et Gemma) · © 2026 Régis Rigaud / TheCause |
| Repo | https://github.com/TheCause/gemma4-zml-probe |

---

## 2. Capacités (ce que le projet sait faire aujourd'hui)

### 2.1 Reproduction fidèle de bout en bout

```
prefill (last_hidden ~1e-5 vs HF) → logits (tokens == HF, 0 flip)
  → decode 1 token (last_hidden + logits + argmax == HF)
  → generate 4 tokens (== HF greedy)
  → generate 1020 tokens (== HF greedy, sliding window 512)
       L1a replay linéaire · L1b ring-buffer 512 (wrap franchi) · L2 autonome
  → GPU CUDA : 1020/1020 == HF à 109 tok/s
```

- **Génération longue CPU (chunkée)** : le decode 35 couches fp32 dépasse la RAM hôte en mono-graphe ;
  le moteur chunké (`engine.zig`) borne le pic (~19 Gio mesurés via `mem_probe.zig`) et reproduit HF
  1020/1020 sur les trois variantes L1a / L1b (ring) / L2 (autonome, gather→réinjection host,
  embeddings lus en streaming depuis le safetensors).
- **Génération longue GPU (mono-graphe)** : le mur mémoire CPU disparaît en VRAM ; `gemma4_gen_long_gpu`
  reproduit HF 1020/1020 à 109 tok/s (~350× le CPU). Le chunking est inutile sur GPU.
- **Non-vacuité prouvée** : un contre-test par **logits** (`gemma4_vacuity_logits`) prouve que le masque
  sliding est réellement consommé (corruption du masque → logits identiques pour p<512, divergents dès
  p=512). Leçon : l'argmax greedy est trop robuste pour détecter ça — comparer les logits.

### 2.2 Inférence sur prompt libre (pipeline texte → texte)

Chaîne complète démontrée : prompt texte → chat template Gemma → génération → texte vérifié.

- `scripts/49_gen_custom_oracle.py` : oracle HF paramétré par `--prompt` (chat template, `SEQ_LEN`
  variable, `--n-tokens`) — produit une fixture au format standard.
- `gemma4_gen_long_gpu` : ZML reproduit la séquence token par token (doit être == HF).
- `scripts/48_detokenize.py` : détokenisation + **gate de round-trip** `tokenize(detokenize(ids)) == ids`
  (le texte affiché est prouvé fidèle aux tokens).

Démonstration : « What is the capital of France? Answer in one word. » → ZML GPU **48/48 == HF**
(108 tok/s) → texte **« Paris »** → round-trip **48/48 PASS**.

**Runtime 100 % autonome (livré 10-11 juil 2026, gates A0-A3)** — l'ex-« limite assumée »
(banc validé contre l'oracle HF) est levée : `gemma4_gen_auto` est un binaire texte→texte
qui ne dépend plus que des poids et du `tokenizer.json` (tokenizer ZML natif + chat template
Zig + prefill-par-decode + early-stop EOS + détok). Usage :

```bash
cd /data/rqz_workspace/zml && ./bazel.sh run --@zml//platforms:cuda=true \
  //examples/rqz:gemma4_gen_auto -- \
  /data/gemma4-zml-probe/weights/model.safetensors \
  <chemin>/tokenizer.json \
  --prompt "What is the capital of France? Answer in one word." [--max-tokens 200]
# stdout : réponse : "Paris"
```

⚠ Le flag `--@zml//platforms:cuda=true` est OBLIGATOIRE (sinon repli CPU silencieux —
désormais refusé en dur par `error.CudaRequired`, échappatoire `--allow-cpu` débogage).
⚠ **VRAM** : le GPU peut être occupé par un autre service local (ex. Ollama, ~22 Go).
`gemma4_gen_auto` refuse alors de démarrer : garde intégrée au lancement (`error.GpuBusy`
si VRAM libre < **20 GiB** — seuil relevé post-L3, cf plus bas — process occupants listés,
cf `docs/VRAM_CHECK_DESIGN.md`), échappatoire `--force-vram`.
La garde tourne aussi en `--allow-cpu` (ce flag ne force pas le CPU, l'init `.cuda` est
tentée d'abord). Libérer : `ollama ps` puis `ollama stop <modèle>` (réversible — rechargé
à la demande). Garde best-effort (nvidia-smi absent/cassé → warn + continue) et propre à
`gemma4_gen_auto` — pour les AUTRES runners GPU, vérifier à la main.

**L3 in-graph (livré 12 juil 2026, branche `l3-ingraph`, spec `L3_INGRAPH_DESIGN.md`)** — le
forward devient token → token : gather des embeddings (`embed_tokens` déjà device, table
`embed_tokens_per_layer` ajoutée via `TensorStore`) et `topK` (next token + top5 diagnostic)
sont désormais **dans le graphe compilé** — le host ne thread plus qu'un scalaire u32 par
step (au lieu des 2 lectures d'embeddings + D2H des logits complets de l'ancienne boucle).
`engine.zig` reste intact d'un octet. Perf mesurée séparément par phase : prefill ~71,5 tok/s
(gate G1, 21 steps), génération **~110-113 tok/s** (≥ 109 tok/s replay, ≥ B0 pré-L3), compile
~17 s. VRAM : le
gather in-graph ajoute la table `embed_tokens_per_layer` en résidence device → pic mesuré
**~16,3 Go** (16 658 MiB), d'où la garde relevée à **20 GiB** (détail des gates : cf
`L3_INGRAPH_DESIGN.md` § Résultats). `--selftest-gather` bascule en mode **GPU** (mini-graphe
compilé, la garde VRAM s'y applique désormais) — il requiert un `--prompt` factice pour passer
la validation d'arguments.

**Validation réelle (11 juil 2026)** : prompt libre en français hors de toute fixture
(« Explique-moi la fenêtre glissante d'attention en trois phrases ») → 110 tokens,
early-stop EOT naturel, réponse correcte sur stdout, 54,5 tok/s moyenne prefill inclus
(mesure pré-L3 ; post-L3 la génération seule atteint ~110 tok/s, cf ci-dessus). Validation :
A1 48/48 == HF autonome complet ; A2 différentiel (autonome ≥ replay, même bifurcation de
marge fine au step ~590 — le N/N n'est pas une propriété garantie de toute séquence, cf
`GEN_AUTONOME_DESIGN.md` § Résultats) ; A3 early-stop EOS + « Paris » sur stdout. Modes de
banc : `--oracle <fixture>` (comparaison à `fed`), `--ids-only`, `--selftest-inputs`,
`--selftest-gather` (mode GPU depuis L3, cf plus haut).

### 2.2 bis Génération BATCHÉE multi-prompts (chantier batching, 8 gates PASS le 12 juillet 2026)

`gemma4_bbatch` : le même runtime autonome, mais **B séquences en parallèle** dans un seul graphe.
Le moteur est devenu **shape-polymorphe** (les 5 reshapes dérivent B/S des shapes d'entrée), donc
**un binaire unique sert tous les B** — sans changer d'un octet le HLO à B=1 (gate T0, md5
byte-identique).

```bash
# banc multi-prompts (B = nombre de lignes du fichier)
gemma4_bbatch <model.safetensors> <tokenizer.json> --prompts prompts.txt --max-tokens 48
# fidélité par lane vs oracles HF (fixtures appariées par index)
gemma4_bbatch ... --prompts fixtures/bench_prompts_b4.txt --oracles f0.st,f1.st,f2.st,f3.st
# constitution du jeu (les lanes doivent avoir la MÊME longueur tokenisée)
gemma4_bbatch ... --prompts candidats.txt --ids-only
```

**Débit mesuré (fp32, 3090)** — le pic VRAM ne bouge pas (16 670 MiB à tous les B : il est atteint
pendant la **compilation**, le cache KV par lane ~38 Mo y est noyé) :

| B | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| agrégé (tok/s) | 113,6 | 211,6 | 401,9 | 718,5 | **1 203** | 1 734 | **2 106** |
| par lane (tok/s) | 113,6 | 105,8 | 100,5 | 89,8 | 75,2 | 54,2 | 32,9 |

**Le plafond n'est pas la VRAM, c'est le compute** — il n'y a pas de mur, mais un rendement
décroissant. **Point d'exploitation utile : B=8-16** (×6 à ×10 de débit agrégé en gardant
75-90 tok/s par séquence). Au-delà : throughput pur (corpus, distillation), pas d'interactif.
Non-régression vérifiée : bbatch à B=1 == `gen_auto` (ratio 0,999, runs appariés).
Contraintes V1 : prompts de **même longueur tokenisée** (le moteur partage un `ctrl.step` et une
position de scatter), greedy (sampling host-side optionnel hors gates).
Détails et findings : `docs/BATCHING_RESULTS.md`.

### 2.3 Fidélité en bf16 (chantier G2, PASS 4 juillet 2026)

La claim « == HF » n'est **pas un artefact fp32**. Protocole complet : [`G2_BF16_FIDELITY.md`](G2_BF16_FIDELITY.md).

- **Méthode de l'enveloppe** : on mesure d'abord combien HF-bf16 diverge de HF-fp32 (l'enveloppe),
  puis on exige de ZML ≤ 2× ce bruit. « == bit-à-bit » n'existe pas en bf16 : **HF-bf16 ne se
  reproduit pas lui-même** (1016/1020 vs fp32, 1re bifurcation au step 21) — tout verdict est relatif.
- **Résultat (G2.2)** : ZML en gemm-bf16 (`PrecCfg.gemm=.bf16` : opérandes des GEMM arrondis bf16,
  reste du flux f32) est **2 à 5× PLUS fidèle au fp32 que HF-bf16** — max_abs p50 0.185 vs 0.425,
  KL 0.28×, 1re bifurcation au step 96 vs 21, 103,8 tok/s.
- **Découverte (G2.1)** : les poids sont **déjà bf16 sur device** (ZML charge au dtype du header
  safetensors) ; la VRAM réelle est **8 494 MiB** — les ~22 Go vus auparavant étaient la réserve BFC
  (`memory_fraction 0.90`), pas l'usage. Corollaire : le banc GPU tiendrait sur une carte 12 Go.

### 2.4 Briques de recherche (au-delà de Gemma)

- **Socle moteur modulaire** (`engine.zig` + `gemma4_engine_e1/e2`) : moteur de decode + briques
  comptime interchangeables, non-régression prouvée par comparaison de graphes HLO (E1 : 1037 fichiers,
  1 diff = le path). Design : [`ZML_MODULAR_ENGINE_DESIGN.md`](ZML_MODULAR_ENGINE_DESIGN.md).
- **POC TurboQuant V-only** (`brick_turboquant.zig`, `gemma4_vquant/decode_vq/gen_vq`, scripts 30–33/45) :
  quantization du cache V branchée comme brique dans le moteur. Résultats :
  [`TURBOQUANT_ZML_RESULTS.md`](TURBOQUANT_ZML_RESULTS.md).
- **PrecCfg** : configuration de précision comptime par famille d'ops (gemm/normes/softmax/...) —
  l'outil qui a permis G2.2, réutilisé pour cartographier la sensibilité par-op — **G2.3, PASS
  10 juil 2026** (cf [`G2_3_OP_SENSITIVITY.md`](G2_3_OP_SENSITIVITY.md)) : 12/12 familles SAFE,
  config combinée 12-familles SAFE à 0.486× l'enveloppe, kv_store bf16 quasi-gratuit.

---

## 3. Le modèle porté — spécificités Gemma 4 E2B

E2B = « Effective 2B » : 2,3 Md de paramètres effectifs, 5,1 Md avec embeddings. Dense (pas de MoE),
35 couches, hidden 1536, vocab 262 144. Les pièges qui font l'intérêt du portage :

| Spécificité | Détail | Traduction ZML |
|---|---|---|
| **PLE** (Per-Layer Embeddings) | 2ᵉ table d'embeddings `[262144, 8960]` injectant un résiduel par couche ; scalings **implicites** dans Transformers (`×√256` PLE, `×√1536` embed) à réappliquer explicitement depuis les poids bruts | gather + scale + fusion `(identity + context)/√2` |
| **Shared KV / YOCO** | 15 producers (couches 0–14), 2 writers (13 sliding / 14 full), 20 readers (15–34, Q-only, pas de modules K/V au runtime) | `scatterSlices(.slot,.k)` + `choose1d(.slot)` ; table de routage statique 35 entrées |
| **Deux types de couches** | sliding : head_dim 256, RoPE θ=1e4, fenêtre 512, MLP 6144 · full (couches 4,9,14,19,24,29,34) : head_dim **512**, **RoPE partielle 0.25** (128/512 dims tournent), θ=1e6 « proportional », MLP double-wide **12288** | RoPE sliding = `zml.nn.rope` natif ; RoPE full = manuelle (split/negate/concat + cos/sin oracle) |
| **GQA** | 8 têtes Q / 1 tête KV | `splitAxis(.h)` des têtes Q (pas de repeat_kv matérialisé) |
| **RMSNorm** | pattern **Llama** `normalized × weight` (≠ Qwen `×(1+weight)`) ; `q_norm`/`k_norm` avec scale, **`v_norm` SANS scale** (normalise quand même !) | `zml.nn.rmsNorm` (neutre) + `.mul(weight)` explicite |
| Activation / tête | `gelu_pytorch_tanh` ; lm_head **tied** avec embed_tokens ; softcap final `30·tanh(x/30)` ; `layer_scalar` par couche | `Tensor.gelu` ; `scale·tanh·scale` |
| Attention | scaling **1.0** (pas √head_dim — la norme passe par q_norm), masque **additif**, softmax fp32, sliding window par **masque** (pas troncation du cache) | `zml.nn.causalAttnMask(..., window)` |
| Checkpoint | multimodal : poids texte préfixés `model.language_model.*` (600 clés) ; K/V des readers présents sur disque mais **ignorés** au runtime | manifest loader (script 34) : 540 tenseurs chargés, 60 ignorés |

Le flux de données complet (frontend PLE → boucle 35 couches → head softcapé) est décrit dans
[`CARTOGRAPHIE_portage.md`](CARTOGRAPHIE_portage.md).

---

## 4. La méthode (la discipline qui fait la valeur du repo)

Chaque opération est une **gate** :

1. **Lire `modeling_gemma4.py`** (la source de vérité) — ne rien supposer.
2. **Oracle PyTorch** indépendant (modules réels quand possible, jamais une ré-dérivation à la main).
3. **Fixture** (`.safetensors`/`.pt` + manifest JSON versionné).
4. **Runner ZML** minimal (une complexité nouvelle par gate).
5. **Comparaison** : points fixes + scan global, tolérance de référence 1e-4 (fp32).
6. **Commit + tag git** (`gate/...`, `p5.x-...-pass`).

Règles capitalisées par l'expérience :

- **Indépendance de l'oracle** : un oracle qui partage une hypothèse avec le code testé donne un PASS
  trompeur (bug `v_norm` : oracle et ZML omettaient tous deux la norm → accord à ~5e-6, faux).
- **Non-vacuité** : prouver qu'un mécanisme est consommé en le corrompant et en observant les **logits**
  (l'argmax greedy est trop robuste). Drapeau jaune : `max_abs = 0` sur un matmul = suspect.
- **Résidu numérique attendu** : ~1e-5 = bruit matmul PJRT-CPU (Eigen-like) vs PyTorch BLAS — pas un bug.
- **Le compilateur est irremplaçable** : un audit multi-agents minutieux (17 agents, session 28/06)
  a raté 3 erreurs de syntaxe/API Zig que le premier build a révélées.
- **Critère fp32 ≠ critère bf16** : le « 1020/1020 » est exigible en fp32 seulement ; en bf16 le
  critère est l'enveloppe chiffrée (G2, §2.3).

---

## 5. Structure du repo

```
scripts/       Oracles Python (PyTorch/HF) + exporteurs de fixtures (00 → 51)
zml_runner/    Runners ZML (.zig) + engine.zig + BUILD.bazel + deploy_to_3090.sh
docs/          Notes par gate, contrats, plans, logs de preuve, cette doc
fixtures/      Manifests versionnés (les .npy/.pt/.safetensors sont régénérables, gitignorés)
logs/          Logs d'exécution rapatriés (preuves)
gemma4-e2b-it-meta/  Métadonnées du modèle (config, pas les poids)
```

### 5.1 Scripts Python (oracles et outils)

| Plage | Rôle |
|---|---|
| `00`–`04` | Sonde initiale : env check, métadonnées, contrat PLE, référence PLE, run_all |
| `05`–`08` | PLE raw PyTorch + export fixtures (`.npy` → selfcheck → safetensors) |
| `09`–`13` | Cartographie YOCO : config map, weight map, table de routage (policy table) |
| `14`–`29` | Oracles attention par gate : Q/K/V proj+norm, RoPE (sliding et full partielle), QK, masque sliding, softmax, context, o_proj, résiduels, MLP |
| `30`–`33` (`p5_x`) | Embedding gather, head (norm+lm_head+softcap), couche décodeur complète, K-rope full |
| `34`–`39` | Runtime 35 couches : loader manifest, load ref, runtime plan, full layer, prefill (oracle hybride fp32/bf16) |
| `40`–`44` | Decode incrémental : pilote, primitives, decode 2/3, boucle de génération (4 tokens) |
| `46`–`47` | **Génération longue** : oracle 1020 tokens (linéaire, puis ring) |
| `48` | **Détokenisation + gate round-trip** (ids → texte prouvé fidèle) |
| `49` | **Oracle prompt custom** (chat template, `--prompt`, `--n-tokens`) |
| `50`–`51` | **G2 bf16** : oracle enveloppe (teacher-forcing HF-bf16 vs fp32) + analyse métriques |
| `30`–`33`, `45`, `spike_hadq`, `measure_k_distribution`, `test_kv_quant_generation` | Piste **TurboQuant** (quantization V, Hadamard) |
| `smoke.sh` | Build-only des runners clés (toolchain OK sans weights ni RAM) |
| `regen_fixtures.sh`, `sweep_perf.sh` | Régénération des fixtures ; sweep de perf (CHUNK) |

### 5.2 Runners ZML (`zml_runner/`)

**Moteur et génération** (le cœur vivant) :

| Runner | Rôle |
|---|---|
| `engine.zig` | Moteur de decode **chunké** partagé (stages compilés, cache threadé step-à-step) |
| `gemma4_prefill.zig` | Prefill 35 couches |
| `gemma4_logits.zig` | Head + logits |
| `gemma4_decode1..4.zig`, `gemma4_decprim.zig` | Decode incrémental (pilote sliding → full → e2e → boucle) |
| `gemma4_gen_long.zig` | Génération longue mono-graphe (CPU, réf. historique) |
| `gemma4_gchunk.zig` / `_ring` / `_auto` / `_vacuity` | L1a chunké / L1b ring-buffer 512 / L2 autonome / contre-test non-vacuité |
| `gemma4_vacuity_logits.zig` | Contre-test non-vacuité par **logits** |
| `gemma4_gen_long_gpu.zig` | **Génération longue GPU fp32** (mono-graphe CUDA, 109 tok/s) |
| `gemma4_g23_sweep.zig` | Sweep bf16 par familles (G2.2/G2.3), moteur `PrecRt` runtime — CLI `<model> <fixture> <logits_out> <familles> [max_steps] [--no-prealloc]`, familles = champs PrecRt ou `none` ; config G2.2 = `qkv_proj,qk_scores,pv_ctx,o_proj,mlp,ple,head` |
| `gemma4_engine_e1.zig` / `_e2.zig` | Socle modulaire : non-régression HLO / brique TurboQuant |
| `gemma4_bench.zig`, `mem_probe.zig` | Bench débit ; instrumentation mémoire |

**Gates unitaires** (une op chacun, conservés comme suite de preuve) : `gemma4_ple_fixture`,
`gemma4_embed`, `gemma4_q/k/v_proj`, `gemma4_q/k/v_norm`, `gemma4_q/k_rope`, `gemma4_full_qrope/krope`,
`gemma4_qk_scores`, `gemma4_sliding_mask`, `gemma4_softmax`, `gemma4_context`, `gemma4_oproj`,
`gemma4_attn_resid`, `gemma4_mlp`, `gemma4_layer`, `gemma4_full_layer`, `gemma4_head`,
`gemma4_kv_slot`, `gemma4_policy_lookup`, `gemma4_routing_mock`, `gemma4_load_check/_all`.

**TurboQuant** : `brick_turboquant.zig`, `gemma4_vquant`, `gemma4_decode_vq`, `gemma4_gen_vq`, `gemma4_hadq`.

### 5.3 Documentation (`docs/`)

| Doc | Contenu |
|---|---|
| `CARTOGRAPHIE_portage.md` | **Carte visuelle** de tout le portage (source de vérité de la sketchnote) |
| `ENGINE_LOG.md` | Journal de preuve du moteur (verdicts réels, mesures) |
| `G2_BF16_FIDELITY.md` | Protocole + résultats fidélité bf16 (enveloppe, G2.0/2.1/2.2) |
| `GENERATION_LONGUE_{PLAN,DESIGN,CHUNKING_DESIGN}.md` | Plan et design de la génération longue |
| `GPU_PORT_PLAN.md` | Plan du portage GPU (finalement trivial : 1 flag) |
| `ROADMAP_to_full_forward.md` | Roadmap historique vers le forward complet |
| `P5_*.md` | Notes par phase : cartographie YOCO, policy table, attention, closeout, contrat de précision, decode |
| `ZML_MODULAR_ENGINE_{DESIGN,PLAN}.md` | Socle moteur modulaire (briques comptime) |
| `TURBOQUANT_ZML_{DESIGN,PLAN,RESULTS}.md` | POC quantization V |
| `SESSION_2026-06-27_RAPPORT.md` | Rapport de la session « écrite sans compiler » + audit |

---

## 6. Utilisation

### 6.1 Prérequis

- **Compte HuggingFace avec la licence Gemma acceptée** (`huggingface-cli login`).
- **Python** : voir `requirements.txt` — testé avec transformers 5.9.0, torch 2.12.0.
- **Un checkout ZML** (Bazel) sur la machine de compute. Testé CPU (`libpjrt_cpu`) et CUDA.
- **Les poids** `google/gemma-4-E2B-it` à `weights/model.safetensors` (~10 Go, non inclus).
- GPU : une carte ≥ 12 Go de VRAM suffit (mesure réelle 8,5 Go).

### 6.2 Topologie et déploiement

Le repo vit sur la machine de dev ; les runners se compilent **dans un workspace ZML** sur l'hôte de
compute (les sources sont copiées dans `examples/rqz/` du workspace) :

```bash
# Copier zml_runner/ vers le workspace ZML distant (rsync, jump host optionnel)
ZML_REMOTE=user@gpu-host ZML_JUMP=bastion ZML_DST=/data/zml/examples/rqz \
  zml_runner/deploy_to_3090.sh

# Sur l'hôte de compute, depuis le workspace ZML :
./bazel.sh build //examples/rqz:<cible>          # CPU
./bazel.sh build //examples/rqz:<cible> --@zml//platforms:cuda=true   # GPU
```

**Vérification rapide de la toolchain** (build-only, sans weights ni fixtures ni RAM) :

```bash
bash scripts/smoke.sh                             # runners clés
SMOKE_TARGETS="gemma4_engine_e1" bash scripts/smoke.sh   # sous-ensemble
```

### 6.3 Rejouer une gate (le geste de base)

```bash
# 1. L'oracle PyTorch produit une fixture sous fixtures/
python scripts/40_p5_7_7_decode_pilot_oracle.py

# 2. Build + run du runner ZML correspondant
./bazel.sh build //examples/rqz:gemma4_decode1
./bazel-bin/examples/rqz/gemma4_decode1 weights/model.safetensors fixtures/p5_7_7_decode1.safetensors
```

Chaque runner imprime `max_abs` / `mean_abs` vs l'oracle et un verdict PASS/FAIL.
Les fixtures gitignorées se régénèrent via `scripts/regen_fixtures.sh` (ou le script oracle individuel).

### 6.4 Génération longue

```bash
# Oracle HF : séquence de référence 1020 tokens (greedy, sliding window 512)
python scripts/46_gen_long_oracle.py            # linéaire  → gen_long.safetensors
python scripts/47_gen_long_ring_oracle.py       # variante ring

# CPU chunké (borne le pic mémoire ; ~55 min/1020 steps)
./bazel.sh run //examples/rqz:gemma4_gchunk      -- weights/model.safetensors gen_long.safetensors
./bazel.sh run //examples/rqz:gemma4_gchunk_ring -- ...   # ring-buffer 512 (wrap réel)
./bazel.sh run //examples/rqz:gemma4_gchunk_auto -- ...   # L2 autonome (gather→réinjection)

# GPU fp32 (mono-graphe, 109 tok/s)
./bazel.sh run //examples/rqz:gemma4_gen_long_gpu --@zml//platforms:cuda=true -- \
  weights/model.safetensors gen_long.safetensors [max_steps]
```

Attendu : `1020/1020 == HF`. Le contre-test de non-vacuité se rejoue via `gemma4_vacuity_logits`.

### 6.5 Inférence sur un prompt libre (end-to-end)

```bash
# 1. Oracle HF : chat template + tokenisation + génération de référence
python scripts/49_gen_custom_oracle.py \
  --prompt "What is the capital of France? Answer in one word." --n-tokens 48

# 2. ZML reproduit sur GPU (doit être == HF token par token)
./bazel.sh run //examples/rqz:gemma4_gen_long_gpu --@zml//platforms:cuda=true -- \
  weights/model.safetensors gen_custom.safetensors 48

# 3. Détokenisation + validation round-trip
python scripts/48_detokenize.py gen_custom.safetensors
```

### 6.6 Régime bf16 (G2)

```bash
# Enveloppe HF-bf16 vs HF-fp32 (teacher-forcing sur la séquence de 46)
python scripts/50_bf16_envelope_oracle.py

# Run ZML gemm-bf16 (dump logits) puis analyse vs l'enveloppe
./bazel.sh run //examples/rqz:gemma4_g23_sweep --@zml//platforms:cuda=true -- \
  weights/model.safetensors gen_long.safetensors g2_2_logits_d.bin \
  qkv_proj,qk_scores,pv_ctx,o_proj,mlp,ple,head
python scripts/51_g2_2_analyze.py
```

Critères PASS (≤ 2× l'enveloppe) chiffrés dans [`G2_BF16_FIDELITY.md`](G2_BF16_FIDELITY.md) §7.1.

### 6.7 Mesures de performance et mémoire

```bash
./bazel.sh run //examples/rqz:gemma4_bench --@zml//platforms:cuda=true -- ...  # débit
# VRAM réelle : lancer avec --no-prealloc (sinon on mesure la réserve BFC, pas l'usage)
# RAM CPU : mem_probe.zig instrumente le pic post-compile
bash scripts/sweep_perf.sh    # sweep CHUNK (CPU)
```

---

## 7. Résultats numériques de référence

### 7.1 Fidélité fp32 (vs HF fp32)

| Étape | Résultat |
|---|---|
| PLE end-to-end (P4.4.2, gates A→J) | scan global max_abs **1.5e-5** (= bruit matmul PJRT vs BLAS), marge ~6500× |
| Toutes les ops du forward (P5.2–P5.6) | chacune ≤ ~7e-5, la plupart bit-exact ou ~1e-7 |
| Couche décodeur complète (P5.3) | max_abs 6.7e-5 |
| Prefill + logits + decode + gen 4 tokens | tokens **== HF, 0 flip** |
| Génération longue CPU (L1a/L1b/L2) | **1020/1020 == HF** chacune |
| Génération longue GPU fp32 | **1020/1020 == HF**, 109 tok/s, VRAM réelle 8,5 Go |

### 7.2 Fidélité bf16 (G2, vs HF fp32 — ratios vs l'enveloppe HF-bf16)

| Métrique | ZML gemm-bf16 (D) | Enveloppe HF-bf16 (B) | Ratio (PASS ≤ 2×) |
|---|---|---|---|
| max_abs logits p50 / p95 / max | 0.185 / 0.330 / 0.623 | 0.425 / 0.661 / 1.546 | **0.44 / 0.50 / 0.40×** |
| KL p50 / p95 / max | 2.9e-5 / 4.6e-4 / 2.6e-3 | 1.0e-4 / 1.7e-3 / 1.3e-2 | **0.28 / 0.27 / 0.19×** |
| Argmax match | 1016/1020 | 1016/1020 | égal |
| 1re bifurcation | step 96 | step 21 | 4,6× plus tard |

**Lecture** : arrondir uniquement aux bornes des GEMM bruite moins que le flux tout-bf16 natif de HF.
En régime bf16, ZML est plus proche de la vérité fp32 que l'implémentation de référence ne l'est
d'elle-même.

---

## 8. Pièges et garde-fous (capitalisés — à lire avant de toucher au code)

**ZML :**

1. **`reshape` perd les tags** — chaîner `.withTags(.{...})` immédiatement après tout reshape suivi
   d'une op qui cible un axe par tag (`rmsNorm`, `mul`/`add` cross-taggés).
2. **`mul`/`add` ne broadcastent pas** — expliciter `weight.broad(other.shape())` (pattern Llama).
3. **`zml.nn.rmsNorm` est neutre** (pas de poids). Gemma 4 = pattern Llama `.mul(weight)`. **Ne pas**
   réutiliser le wrapper Qwen3.5 (`×(1+weight)`).
4. **Quota comptime `pjrt.zig structSize`** : un `@typeName` long (nom de runner > ~20c, ou cfg
   comptime non-défaut comme `.prec`) déborde le quota de 1000 branches. Patch workspace 1 ligne
   `@setEvalBranchQuota(100_000)` (commenté `local patch rqz`) — **à réappliquer si le workspace ZML
   est resynchronisé upstream**.
5. **CUDA** : builder avec `--@zml//platforms:cuda=true` sinon `Platform.init(.cuda)` retombe en CPU
   silencieusement. **Ne pas** mettre `/usr/local/cuda/lib64` dans `LD_LIBRARY_PATH` (ZML gère CUDA
   via runfiles).
6. **Compile XLA-CPU du forward 35 couches** : gourmand en RAM — un swap actif est requis sur l'hôte
   (sinon OOM-kill exit 255 au compile). `smoke.sh` vérifie et avertit.
7. **Le traçage ZML DÉDUPLIQUE les nœuds identiques** (règle d'émission découverte en G2.3, non
   documentée upstream) : émettre deux fois la même op sur les MÊMES opérandes/attributs produit
   **UN seul nœud** dans le HLO `before_optimizations` — pas deux. Vérifiée sur `convert`,
   `choose1d` et les constantes, y compris **à travers des invocations de fonction distinctes**
   (`zml.nn.rope` appelé 40× → UN `idx.convert(f32)` émis). Cinq preuves empiriques (10 juil
   2026, workspace 3090) :
   - `xff.convert(bf16)` partagé par gate_proj et up_proj → 1 convert/couche au lieu de 2
     (one-hot `mlp` : Δ70 observé, pas 105) ;
   - `h0.convert(bf16)` partagé par q/k/v_proj → 1/couche (one-hot `qkv_proj` : Δ35, pas 65) ;
   - les `choose1d` des 20 readers YOCO ≡ celui de leur writer (mêmes opérandes) → leurs converts
     fusionnent (run 7-familles : Δ382 exact) ;
   - recensement D0 réconcilié **à l'unité** : 542 converts (sans dédup on en prédirait 581) ;
   - une dédup **inter-features** : un convert `bf16[1,1,1536]` d'opérande `%multiply` partagé
     entre les familles norms et ple (localisée par bisection + diff de signatures HLO).
   **Conséquences pratiques** : (a) tout oracle de comptage d'ops doit compter les nœuds ÉMIS
   après déduplication, jamais les appels dans le code ; (b) les comptes de deux features ne
   s'additionnent que MOINS leurs nœuds partagés nommés (cf `_interfamily_dedups` de
   `fixtures/g2_3_expected_converts.json`) ; (c) propriété utile : une sous-expression répétée ne
   coûte rien de plus dans le graphe. **Portée** : établie empiriquement (mécanisme présumé =
   value-uniquing du builder MLIR, non tracé dans les sources) — à re-vérifier si le workspace
   ZML est resynchronisé upstream. Dossier de preuve : `docs/G2_3_OP_SENSITIVITY.md` §5.3/§9 +
   `fixtures/g2_3_expected_converts.json` `_meta`.

**Gemma 4 :**

8. **Scalings d'embeddings implicites** : `×√1536` (embed) et `×√256=16` (PLE) sont inclus dans les
   modules Transformers mais PAS dans les poids bruts — à réappliquer explicitement.
9. **`with_scale=False` ≠ pas de normalisation** : `v_norm` normalise V (division RMS) sans poids
   appris. « Pas de poids au checkpoint » ne signifie pas « pas d'op ».
10. **Attention scaling = 1.0** (pas 1/√head_dim), pas de softcap d'attention (seulement le softcap
    final 30), masque additif, softmax fp32.
11. **Checkpoint multimodal** : les poids texte sont sous `model.language_model.*` ; les K/V des
    readers existent sur disque mais sont ignorés au runtime (YOCO).

**Méthode :**

12. **Oracle = source de vérité** : dériver chaque oracle de `modeling_gemma4.py`, jamais d'une
    hypothèse ré-encodée (sinon PASS trompeur par hypothèse partagée).
13. **Comparer les logits, pas l'argmax**, pour tout contre-test de non-vacuité.
14. **VRAM : mesurer avec `--no-prealloc`** — sinon on lit la réserve BFC, pas l'usage réel.
15. **Pas de bit-à-bit entre deux compiles XLA-GPU** (autotuning : le `before_optimizations` est
    stable, le post-opt non) — comparer des MÉTRIQUES, quantifier le bruit compile-à-compile
    (~2 % sur un grand effet, jusqu'à ~16 % sur un petit — G2.3 §9.2/§9.4).
16. **Un banc doit distinguer « le test a échoué » de « le test n'a pas tourné »** (chantier
    batching) : le 1er sweep a rapporté un « B=4 FAIL » qui était en réalité un run **refusé par
    la garde de contention** (VRAM du run précédent non libérée). Corollaire : **verrou
    d'instance unique** (`flock`) + fichiers temporaires uniques (`mktemp`) — deux sweeps
    concurrents (un « tué » côté client survivait sur la VM) se partageaient GPU et `/tmp`, et
    produisaient des chiffres croisés parfaitement plausibles.
17. **Le batching n'introduit pas d'erreur, il EXPOSE la fragilité des ties** : à B>1, l'ordre de
    réduction GEMM change → un tie à ~1e-4 sur les logits bascule. Le même run B=4 donne 4/4 en
    isolation et 3/4 dans le sweep (**même binaire, même prompts**) : c'est le piège 15 qui se
    manifeste sur l'argmax. Les lanes restent bit-identiques **entre elles** (aucune
    contamination). Ne jamais lire une bifurcation de lane comme un bug de batching sans avoir
    regardé la **marge top1−top2**.
18. **`@import("root")` referme une boucle de dépendance** quand le fichier root importe le
    module qui le lit (`gemma4_bbs` → `gemma4_bbatch` → root) : pour une variante comptime entre
    deux binaires partageant des sources, passer un **paramètre comptime explicite**
    (`runWith(comptime attn, init)`), pas `@import("root")`.
19. **`zml.nn.sdpa` scale K par 1/√hd par DÉFAUT** alors que Gemma 4 a un scaling de 1.0 (la
    normalisation passe par `q_norm`) → `.scale = Tensor.scalar(1.0)` **obligatoire**, sinon les
    scores sont silencieusement divisés par 16.

---

## 9. Limites actuelles et travaux futurs

**Limites assumées** (baseline de recherche, pas moteur de prod) :

- Mono-séquence : pas de batching, pas de sampling (greedy only), pas de fast-prefill.
- Multimodal (vision/audio) hors scope — chemin texte uniquement.
- `L_MAX` plafonné à 1024 sur CPU (le compile XLA-CPU à k=2048 dépasse l'hôte ~23 Go) ; la fenêtre 512
  est quand même franchie ~2×. Sur GPU la limite est différente (VRAM du graphe).
- Le banc dépend de l'oracle HF pour tokeniser et fixer la longueur (pas de runtime autonome).

**Backlog** (ordre du planning courant, cf [`../PLANNING.md`](../PLANNING.md)) :

1. Batching / flash-attention (perf GPU au-delà du mono-séquence).
2. ~~L3 in-graph (boucle de décode dans le graphe, réduire les 7 syncs host/step du CPU)~~ —
   **fait 12 juil 2026** (cf [`L3_INGRAPH_DESIGN.md`](L3_INGRAPH_DESIGN.md) § Résultats).
3. Runtime 100 % autonome (tokenizer intégré + early-stop EOS).
4. ~~G2.3 bonus : cartographie de sensibilité bf16 par-op~~ — **PASS 10 juil 2026**
   (cf [`G2_3_OP_SENSITIVITY.md`](G2_3_OP_SENSITIVITY.md)) : 12/12 familles SAFE, config combinée
   12-familles SAFE à 0.486× l'enveloppe, kv_store bf16 quasi-gratuit.

---

## 10. Historique des phases (résumé)

| Phase | Contenu | État |
|---|---|---|
| P-1 → P3 | Sonde PLE : contrat des poids, référence Transformers, reproduction raw-PyTorch | ✅ mai 2026 |
| P4 | Mini-runner ZML PLE-only (gates A→J, bit-exact → 1.5e-5) | ✅ 28 mai |
| P5.0 → P5.2 | Cartographie YOCO, policy table, attention op-par-op (Q/K/V, RoPE, QK, masque, softmax, context, o_proj, résiduels, MLP) | ✅ 28 mai – 2 juin |
| P5.3 → P5.6 | Couche décodeur complète, embedding, head, RoPE full partielle, audit closeout | ✅ 31 mai – 2 juin |
| P5.7 | Runtime 35 couches : loader, prefill, logits, decode incrémental, génération 4 tokens | ✅ juin |
| Gén. longue | Moteur chunké L1a/L1b/L2 (CPU) + GPU CUDA — 1020/1020 == HF | ✅ 28 juin |
| Pipeline e2e | Scripts 48/49 : prompt libre → texte vérifié | ✅ 28 juin |
| G2 bf16 | Enveloppe + gemm-bf16 : fidélité tenue en basse précision | ✅ 4 juil |
| PR → main | Consolidation de la branche `generation-longue` | 🔄 9 juil (PR #3) |

Chaque gate a son tag git (`git tag -l 'gate/*' 'p5.*' '*-pass' '*validated*'`) et sa note dans `docs/`.

---

## 11. Licence et attribution

Code : **Apache-2.0** ([`LICENSE`](../LICENSE)) — même licence que ZML et Gemma.
© 2026 Régis Rigaud / TheCause. Les poids Gemma 4 sont distribués par Google sous les
[termes Gemma / Apache-2.0](https://huggingface.co/google/gemma-4-E2B-it) — non inclus dans ce repo.
