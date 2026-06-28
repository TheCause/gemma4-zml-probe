# Plan de portage GPU — moteur d'inférence Gemma-4-E2B-it en ZML

> Document complet et exhaustif des modifications et améliorations pour faire tourner le moteur ZML
> (actuellement CPU fp32, `libpjrt_cpu`) sur GPU (CUDA / ROCm), du baseline bit-exact au moteur de
> génération performant. Branche suggérée : `gpu-port` (depuis `generation-longue`).
>
> **Insight central (lire en premier) : le portage GPU n'est PAS une réécriture.** Le moteur `engine.zig`
> compile en un graphe XLA/HLO **device-agnostic** ; les op utilisées (`dot`, `rmsNorm`, `rope`, `softmax`,
> `gelu`, `scatterSlices`, `dynamicSlice`, `choose1d`, `convert`, `scale`, `tanh`…) sont des op XLA
> standard que le backend GPU abaisse en noyaux CUDA natifs. Le code applicatif est donc **déjà portable**.
> Le chantier GPU est : **(1) disponibilité du backend CUDA/ROCm, (2) stratégie de précision, (3) knobs
> de perf (batching, attention, fusion), (4) re-validation contre l'oracle fp32 + HF** — avec la même
> discipline de gates que le portage CPU.
>
> Specs sœurs : `ZML_MODULAR_ENGINE_DESIGN.md`, `GENERATION_LONGUE_DESIGN.md`, `P5_7_5_precision_contract.md`.
> Auteur : analyse codebase 27 juin 2026. Statut : **proposition** (non implémentée).

---

## 0. TL;DR exécutif

| Axe | CPU (actuel) | GPU (cible) | Effort |
|---|---|---|---|
| Backend | `libpjrt_cpu`, `Platform.auto` tombe sur CPU (VM sans PJRT CUDA) | `libpjrt_cuda` chargé → `Platform.auto` sélectionne CUDA | Faible (infra) |
| Précision | fp32 partout (baseline bit-exact) | bf16/fp16 poids + accumul fp32 ; fp16 où sûr | Moyen (contrat) |
| Mémoire | **Mur à ~33 Go** (35 couches fp32 déroulées, swap thrash, chunking obligatoire) | ~10 Go poids bf16 + ~40 Mo KV @2048 → **tient sur 1×24 Go**, chunking devient optionnel | — (le mur disparaît) |
| Perf decode | ~55 min / 1020 tokens (7 syncs host/step) | **cible 30–80 tok/s** batch-1 (3090), >150 tok/s (A100) | Moyen (kernels + bench) |
| Batching | batch=1 | batch ≥ 1 (continuous batching optionnel) | Moyen |
| Prefill | S=4 fixture | fast-path long-contexte (attention paged/flash) | Moyen |
| Validation | argmax == HF, tol 1e-4 | argmax == HF + drift borné vs baseline fp32-CPU + HF | Méthode conservée |

**Le premier deliverable (G0–G1) est quasi-gratuit** : sur une machine avec CUDA, `Platform.auto(.{})`
sélectionne déjà le GPU avant le CPU (ordre ZML : `tpu, neuron, rocm, cuda, cpu`). Le moteur actuel,
recompilé avec `libpjrt_cuda` disponible, tourne **tel quel en fp32 sur GPU** et doit reproduire HF
(dans une tolérance plus large : CUDA vs BLAS). Tout le reste est optimisation + précision.

---

## 1. État actuel et pourquoi CPU-only

- **Code** : `zml_runner/*.zig` appelle `zml.Platform.auto(allocator, io, .{})` (cf `gemma4_gen_long.zig`,
  `gemma4_gchunk.zig`, …). `Platform.auto` essaie `tpu→neuron→rocm→cuda→cpu` et garde le 1er qui
  s'initialise (`platform.zig:275`). **Aucune ligne ne force le CPU.**
- **Pourquoi CPU en pratique** : la VM 3090 de dev n'expose pas `libpjrt_cuda` (ou le plugin CUDA PJRT
  n'est pas buildé dans le workspace ZML local). `auto` tombe donc sur `cpu`. Le README documente
  `libpjrt_cpu` comme backend — c'est un constat d'environnement, pas un verrou code.
- **Pourquoi fp32** : choix méthodologique (baseline bit-near vs HF, contrat `P5_7_5_precision_contract.md`).
  Le fp32 + 35 couches déroulées = ~22–33 Go → **le mur mémoire CPU** qui a justifié le chunking
  (`GENERATION_LONGUE_CHUNKING_DESIGN.md`). Sur GPU, ce mur n'existe pas (cf §6).

**Conséquence** : le moteur est GPU-ready par construction. Le travail est de **rendre le backend
disponible, choisir la précision, et prouver** — pas de réécrire `runLayerGen`.

---

## 2. Faits ZML pertinents pour le GPU (vérifiés dans le checkout ZML)

- `Platform.auto(.{})` — sélection auto, **cuda avant cpu**. Aucun changement de code requis pour
  pointer le GPU ; il faut juste que le plugin CUDA PJRT soit chargé.
- `CreateOptions` (`platform.zig:~505`) :
  - `cuda: Cuda = .{ .allocator = .{ .bfc = .{ .preallocate = true, .memory_fraction = 0.85 } } }` —
    BFC (Best-Fit with Coalescing), pré-alloue 85 % de la VRAM par défaut. Tunable (cf §8).
  - `cpu: Cpu = .{ .device_count = 4 }` — nb threads CPU (utile si host-fallback).
  - `rocm: struct{} = .{}` — backend AMD (idem CUDA, plugin PJRT ROCm).
- **Le graphe compilé est le même** quel que soit le device : `platform.compile`/`compileFn` produisent
  du HLO abstrait ; le backend l'abaisse (CPU = Eigen, GPU = cuDNN/cublas/flash-attention-ABI XLA).
  → **les preuves HLO `diff -rq` du projet restent valides** (le graphe ne change pas, seul l'abaissement).
- `zml.sharding.replicatedSharding(platform)` — sharding utilisé par les runners ; sur mono-GPU = tout
  répliqué sur 1 device. Multi-GPU = `sharding` par tenseur (cf §11).
- Les **examples ZML** (`examples/llm/{llama,lfm,qwen}`) tournent déjà sur GPU via le même
  `Platform.auto` — le pattern est éprouvé. Gemma4 suit le même chemin.

---

## 3. Matériel cible et sizing

| Cible | VRAM | Cas | Verdict |
|---|---|---|---|
| **RTX 3090** (existant) | 24 Go | Dev/validation, bf16 batch-1 | ✅ tient largement (cf §6) |
| RTX 4090 | 24 Go | Dev + ~2× perf | ✅ |
| A6000 / L40S | 48 Go | Batch + long-contexte | ✅ confortable |
| **A100 / H100** | 80 Go | Prod, batching dense, 8K+ contexte | ✅ cible perf |
| 2–8× GPU | ×VRAM | TP (tensor parallel) gros modèle | E2B tient sur 1 GPU → TP non requis (cf §11) |

**Gemma-4-E2B tient sur UNE 3090** : 5,1 Md params × 2 octets (bf16) ≈ **10,2 Go** + KV cache négligeable
(~40 Mo @ 2048, cf §6) + activations (< 1 Go batch-1) → ~12 Go << 24 Go. La marge permet le batching
et le long-contexte sans tensor-parallel. C'est un avantage structurel du modèle "Effective 2B".

---

## 4. Budget mémoire GPU (le mur CPU disparaît)

| Composant | CPU fp32 (actuel) | GPU bf16 (cible) |
|---|---|---|
| Poids texte (35 couches + frontend + norm) | ~14,8 Go fp32 résidents (load) | **~5,1 Go** bf16 (2,3 Md effectifs) |
| `embed_tokens` [262144,1536] (tied lm_head) | 1,6 Go fp32 | 0,8 Go bf16 |
| `embed_tokens_per_layer` [262144,8960] | 9,4 Go (bf16 sur disque, upcast perdant) | 4,7 Go bf16 |
| **Total poids résidents** | ~18,5 Go fp32 / 13,8 Go hybride | **~10,2 Go bf16** |
| KV cache YOCO @ L=2048 | ~80 Mo fp32 | ~40 Mo bf16 (sliding 12×2048×256 + full 3×2048×512) |
| Activations decode batch-1 | ~1 Go | ~0,5 Go |
| Workspace XLA / cuBLAS | — | ~1–2 Go |
| **Pic compile mono-graphe 35 couches** | **~33 Go → swap thrash (fatal)** | **< 16 Go** (abaissement GPU + réutil buffers) |

→ **Le chunking du decode (`gemma4_gchunk*.zig`) devient OPTIONAL sur GPU** : le mono-graphe 35-couches
compile et tourne sans thrash. On garde le chunking pour le **prefill long-contexte** (cf §9.4) et comme
repli mémoire, pas pour le decode batch-1. C'est le gain structurel le plus important.

> **Note embeddings** : `embed_tokens_per_layer` (4,7 Go bf16) est le plus gros tenseur. Sur GPU on le
> garde bf16 sur device pour le gather L2 (ou on lit les lignes à la demande, cf §9.6). Pour le prefill,
> le PLE ne charge que la projection `per_layer_model_projection` [8960,1536] (5,5 Mo), pas la table
> entière — la table n'est lue que pour gather les tokens, comme aujourd'hui.

---

## 5. Stratégie de précision (le cœur du chantier)

### 5.1 Principe : préserver l'argmax, pas le bit-exact

Le contrat CPU (bit-near ~1e-5 vs HF) ne tient pas en bf16. L'objectif GPU : **argmax(tokens) == HF
greedy + drift borné vs le baseline fp32-CPU prouvé**. La métrique de gate devient :
- **PASS primaire** : `argmax == HF` (0 flip sur la séquence de validation) ;
- **PASS secondaire** : `max_abs(logits) ≤ seuil_gpu` (drift bf16 vs baseline fp32, à caractériser au G2) ;
- **Non-régression vs CPU** : la séquence GPU == la séquence CPU-L1a (qui == HF).

### 5.2 Quoi garder en fp32, quoi passer en bf16/fp16

| Op | Précision recommandée | Raison |
|---|---|---|
| Poids `q/k/v/o_proj`, `gate/up/down`, PLE proj | **bf16** (stockage) / **fp32 accumulate** (matmul) | matmul GEMM : cuBLAS accumulate fp32 sur entrées bf16 |
| `rmsNorm` (x/√mean) | **fp32** (upcast avant) | sensible : la division RMS amplifie le bruit bf16 |
| `q_norm`/`k_norm`/`v_norm` × weight | fp32 pour la norm, mul en fp32 | idem |
| `softmax` (scores → probs) | **fp32** (softmax fp32 natif XLA) | classique : softmax bf16 perd les queues |
| `RoPE` (sliding `zml.nn.rope`, full `manualRope`) | **fp32** (cos/sin fp32) | RoPE bf16 dérive sur θ=1e6 (full) |
| `gelu_pytorch_tanh` | bf16 ok (ou fp32) | peu sensible |
| `softcap` final `30·tanh(x/30)` | **fp32** (tanh + scale) | borne les logits, sensible près de ±30 |
| `lm_head` dot [hidden]×[vocab 262144] | bf16 entrées / **fp32 accumulate** | gros GEMM, l'accumulate fp32 fixe l'argmax |
| `layer_scalar` × | fp32 (scalaire) | valeurs petites (0.0178…0.5) |
| KV cache | **bf16** (stockage) | tolère bf16 (cf TurboQuant V-only 4 bits déjà validé ~0.92 top-5) |
| `embed_tokens` gather | bf16 (table) → fp32 après ×√1536 | le scale √1536=39.19 en fp32 |

**Règle pragmatique** : `c(t) = t.convert(.f32)` (déjà partout dans `engine.zig`) devient le point
d'upcast fp32 avant les norm/softmax/rope/softcap. Les GEMM restent en bf16-entrées/fp32-accumulate
(natif cuBLAS sur `dot`). C'est la configuration "mixed bf16" standard des LLM en prod.

### 5.3 Implémentation du knob précision

`engine.zig` gagne une config comptime (même pattern que `EngineCfg`, neutralité préservée) :

```zig
pub const PrecCfg = struct {
    compute: zml.DataType = .f32,   // dtype des accumulations sensibles (norm/softmax/rope/softcap)
    weight: zml.DataType = .f32,    // dtype de chargement des poids (f32=baseline, bf16=GPU prod)
    kv: zml.DataType = .f32,        // dtype du cache KV
};
// EngineModel(comptime Brick, comptime cfg: EngineCfg, comptime prec: PrecCfg)
```

- `PrecCfg{}` (défaut fp32) → **byte-identique à aujourd'hui** (E1/E2/L1a inchangés, preuve HLO OK).
- `PrecCfg{ .weight = .bf16, .kv = .bf16 }` (compute reste f32) → config GPU prod.
- Les `c(layer.*)` deviennent `layer.convert(prec.compute)` ; les `createTensor(..., null)` passent
  `prec.weight` comme dtype attendu → `zml.io.load` charge/convertit à ce dtype.
- Les `Cache` tensors prennent `prec.kv`.

> Le poids "compute f32" est le levier de sécurité : si l'argmax dérive en full-bf16, on garde les
> norms/softmax/softcap en fp32 (coût perf minime, ces ops sont petites) et on ne bf16-ise que les GEMM.

---

## 6. Le mur mémoire CPU → pourquoi il s'effond sur GPU

Le mur actuel (cf `GENERATION_LONGUE_CHUNKING_DESIGN.md` §1) : le compile XLA-**CPU** du graphe 35-couches
fp32 matérialise les **conversions f32 des poids** de toutes les couches simultanément (~33 Go) car
Eigen ne réutilise pas les buffers跨-couches comme le fait le GPU. Sur GPU :
- l'abaissement XLA-GPU **réutilise les buffers** (fusion + in-place) → le working set = 1 couche + GEMM ;
- les poids bf16 sont 2× plus petits ;
- la VRAM 24 Go ≥ ~16 Go de pic compile.
→ Le **mono-graphe 35-couches compile et tourne sans chunking**. `gemma4_gen_long.zig` (mono) devient
le runner GPU principal (rapide, simple) ; `gemma4_gchunk*.zig` reste pertinent pour le **prefill**
(S>1, activations grosses) et comme repli.

**Conséquence pour L2** : le `forwardStep` mono (que j'ai ajouté) devient exécutable sur GPU (sans le
thrash qui l'empêchait en CPU) → la version la plus simple de l'autonomie tourne. Le chunké
`forwardStageStep` reste pour la mémoire maîtrisée / long-contexte.

---

## 7. Modifications de code — fichier par fichier

### 7.1 `zml_runner/engine.zig` (socle)
1. **Knob précision `PrecCfg`** (§5.3) — paramètre comptime de `EngineModel`, défaut neutre.
2. **Dimension batch `B`** → rendre `B` paramètre comptime (aujourd'hui `B=1` const). Permet le batching
   (§9.2). Les `reshape(.{B,S,...})`, `scatterSlices`, masques deviennent batch-aware (le masque
   causal/bande est `[B,H,Q,K]` ou broadcast `[1,H,Q,K]`→`[B,...]`).
3. **KV cache layout** : passer de `{slot,b,h,k,hd}` (slot-major, adapté au `choose1d` CPU) à un layout
   GPU-friendly (cf §9.3) — optionnel, le layout actuel fonctionne (scatter est supporté), mais un layout
   `[slot,k,h,hd]` contiguë par tête aide les noyaux d'attention fusionnés.
4. **Aucun changement** à `runLayerGen` (déjà device-agnostic). Les `convert(.f32)` existants deviennent
   `convert(prec.compute)`.

### 7.2 Runners (`gemma4_gen_long.zig`, `gemma4_gchunk*.zig`, `gemma4_engine_e1/e2.zig`)
- Instancier `EngineModel(_, cfg, prec)` avec `prec` GPU (`.bf16` poids/kv).
- `Platform.auto(.{})` **inchangé** (sélectionne CUDA si dispo). Option : forcer `.cuda` via
  `Platform.init(allocator, io, .cuda, .{ .cuda = .{ .allocator = .{ .bfc = .{ .memory_fraction = 0.9 } } } })`
  pour debug/perf.
- Charger les poids au dtype `prec.weight` (le `createTensor` propage).
- Ajouter un timer `std.time` autour de la boucle + débit tok/s (cf §9.7).

### 7.3 `zml_runner/BUILD.bazel` + workspace ZML
- **Dépendre du plugin CUDA PJRT** : s'assurer que le build ZML produit/linke `libpjrt_cuda.so`
  (côté workspace ZML : `--config=cuda` ou la cible PJRT CUDA). Vérifier `ldd` du binaire.
- Aucune nouvelle cible runner requise pour G0–G3 (les runners existants, recompilés, tournent sur GPU).
- Nouvelles cibles pour les gates perf : `gemma4_bench`, `gemma4_batch` (cf §9).

### 7.4 `zml_runner/deploy_to_3090.sh`
- Déjà paramétrable (`ZML_REMOTE`, `ZML_DST`). Ajouter un check `nvidia-smi` + `ldd libpjrt_cuda` côté
  remote dans le smoke (cf §11).

### 7.5 Oracles Python (`scripts/46`, `47`)
- **Inchangés** : l'oracle HF (GPU CUDA PyTorch) produit déjà les fixtures (le L0 tourne sur GPU !).
  L'oracle reste la source de vérité indépendante du device ZML.

---

## 8. Configuration GPU (VRAM, allocator, flags)

```zig
const platform: *zml.Platform = try .init(allocator, io, .cuda, .{
    .cuda = .{ .allocator = .{ .bfc = .{
        .preallocate = true,
        .memory_fraction = 0.90,   // 90% des 24 Go = 21,6 Go pour le modèle (laisse 2,4 Go driver/workspace)
    } } },
});
```

- `memory_fraction` : 0.85 (défaut ZML) → 0.90 sur la 3090 (10 Go de modèle laisse de la marge).
- `XLA_FLAGS` : `--xla_gpu_cuda_data_dir`, `--xla_gpu_ftz` (flush-to-zero, attention NaN), et surtout
  **désactiver les contournements CPU** (`--xla_cpu_multi_thread_eigen` n'a plus de sens).
- `TF_CPP_MIN_LOG_LEVEL=2` pour réduire le bruit XLA.
- Pour le debug de perf : `--xla_dump_to` (HLO), `--xla_gpu_autotune_level=4` (autotune cuBLAS/cuDNN).

---

## 9. Optimisations de performance (par ordre de gain)

### 9.1 Backend natif (gain immédiat, ~1000×)
Recompiler avec `libpjrt_cuda` : le même graphe tourne sur cuBLAS/cuDNN. Decode batch-1 passe de
"heures/55min" à **secondes**. **C'est le gain gratuit du portage** — mesurer au G1.

### 9.2 Batching (gain ~linéaire jusqu'à saturer les SM)
- `B` paramètre comptime (§7.1). Decode multi-sequences : le cache KV devient `[B, slot, h, k, hd]`,
  le scatter par `(b, slot, pos)`, le masque par batch.
- Le PLE per-layer `choose1d(.layer, i)` est partagé (indépendant de B) → pas de surcoût.
- Continuous batching (ajout dynamique de séquences au cache) : gate ultérieur (G7), nécessite un
  scheduler host (slots libres, mask par séquence active). Optionnel pour un moteur de recherche.

### 9.3 Attention — sliding window + YOCO + GQA (le cœur perf)
Gemma4 a 3 particularités à abaisser efficacement :
- **GQA 8Q/1KV** : déjà géré par `splitAxis(.h)` → le backend voit un GEMM `[B,Q,8,hd]×[B,K,1,hd]`
  broadcast. XLA-GPU abaisse en `cublasGemmStridedBatched`. OK.
- **Sliding window 512** : actuellement géré par un **masque additif** (`scores.add(mask)`) sur tout le
  cache `.k`. Sur GPU long-contexte, cela alloue `[B,H,Q,K]` complet (O(K²)) — gaspillage. **Optim :**
  - decode (Q=1) : le masque est trivial (1 ligne) → coût O(K), négligeable. Pas d'optim urgente.
  - prefill (Q=S grand) : utiliser un **noyau d'attention sliding-window fusionné** (flash-attention à
    fenêtre) → O(S·W) au lieu de O(S·K). XLA expose des flash-attention via `sdax`/custom-call ;
    ZML `zml.nn.sdpa` existe mais **ne donne pas les scores bruts** (le projet l'avait écarté pour ça).
    En mode "produit" (pas besoin des scores bruts), `sdpa` avec `window=512` est le bon chemin.
- **YOCO shared KV** : les readers 15-34 réutilisent le KV des writers 13/14 via `choose1d(.slot)`.
  Sur GPU, c'est un slice/zero-copy du tenseur cache → gratuit. La shared-KV est **un avantage perf**
  (20 couches Q-only, pas de K/V proj) déjà capté par l'archi.
- **Layout cache** : `[slot,k,h,hd]` contiguë (têtes groupées) aide cuDNN/flash. Le layout actuel
  `{slot,b,h,k,hd}` fonctionne ; un re-layout est une optim de second ordre.

### 9.4 Prefill fast-path (long contexte)
- Le prefill actuel est S=4 (fixture). Un vrai moteur gère S jusqu'à 8K+.
- **Chunked prefill** : découper le prompt en blocs (le mode `chain` du prefill, `ENGINE_LOG` 2 juin,
  est déjà ce pattern — le réutiliser). Sur GPU, blocs plus gros (1024–4096 tokens).
- **Attention flash/paged** sur les blocs (§9.3). Le masque causal + sliding window gérés dans le noyau.
- Le prefill produit le `cache0` (KV writers 13/14 + producers 0-14) qui nourrit le decode.

### 9.5 Fusion de noyaux (leverage XLA)
XLA-GPU fusionne automatiquement les op élément-wise (rmsNorm, gelu, scale, tanh, softcap, residuel).
Les patterns `rmsScaleD(x,w) = rmsNorm(x).mul(w)` et `softcap = scale·tanh·scale` se fusionnent en un
noyau → pas de travail manuel, juste **vérifier le dump HLO** que la fusion a lieu (XLA le fait sur GPU
bien mieux que sur CPU). C'est un gain gratuit du backend.

### 9.6 Gather des embeddings (L2) sur GPU
- **Actuel (L2 CPU)** : lecture host des tables (~5,5 Go) + `fromBytes` par step. Sur GPU, on garde les
  tables en **VRAM bf16** (0,8 + 4,7 = 5,5 Go, tient) et on fait le **gather dans le graphe** (le vrai L3
  du design) : passer `tok` (scalaire u32) au forward, `embed_tokens.gather(.{.voc=tok})` → pas de
  round-trip host/device par step. C'est l'optim L3 — sur GPU elle est **naturelle et rapide** (le gather
  est une op XLA). → L2-sur-GPU converge vers L3 gratuitement.
- `embed_tokens_per_layer` (4,7 Go) en VRAM : OK sur 24 Go (modèle ~10 + table 4,7 = 15 Go < 21,6).
  Repli si VRAM tendue : la garder en host pinned-memory + transfert asynchrone d'1 ligne (18 Ko/step).

### 9.7 Benchmarks (mesurer, pas présupposer)
- Runner `gemma4_bench.zig` : time-to-first-token (prefill), decode tok/s (batch 1/4/8/16), pic VRAM,
  pour N=128/512/2048 tokens. Comparer CPU-baseline vs GPU-fp32 vs GPU-bf16.
- Cibles (à confirmer au G8) : 3090 batch-1 decode **≥ 30 tok/s** (gemma-4-E2B ~2.3B bf16, plausible vs
  vLLM/llama.cpp qui font 40–80 tok/s sur 3090 pour ~2B) ; A100 ≥ 150 tok/s ; TTFT ≤ 200 ms @ S=512.

### 9.8 Sampling (hors-scope greedy → optionnel)
Le moteur est greedy (argmax). Pour un moteur utilisable : ajouter temperature/top-k/top-p (host, post-
logits) + le softcap déjà présent. Faible effort, gate G8-bis optionnel.

---

## 10. Méthodologie de validation — gates G0 → G8

On conserve la discipline du projet (oracle = source de vérité, gate atomique, commit+tag, non-vacuité).
**L'oracle de référence devient double** : (a) HF (PyTorch GPU) — la vérité absolue ; (b) le **baseline
fp32-CPU prouvé** (`gemma4_gen_long` 1020/1020) — la régression interne. Le fp32-CPU n'est plus la cible
de prod mais reste l'**étalon de non-régression**.

| Gate | Contenu | Critère PASS |
|---|---|---|
| **G0** | Backend CUDA PJRT disponible : `nvidia-smi` + `ldd libpjrt_cuda` + `Platform.init(.cuda)` s'initialise | platform.target == .cuda |
| **G1** | **fp32 sur GPU** : recompiler `gemma4_gen_long` tel quel (prec fp32) → replay 1020 tokens | argmax == HF ; `max_abs(logits)` vs CPU-baseline caractérisé (tol GPU plus large, ~1e-2 attendu Eigen→CUDA) |
| **G2** | **Mixed bf16** (poids bf16, compute f32) : `PrecCfg{.weight=.bf16, .kv=.bf16, .compute=.f32}` | argmax == HF 1020/1020 ; `max_abs` vs CPU < seuil G2 (à fixer, ~1e-1 logits / 0 flip argmax) |
| **G3** | **Full bf16 compute** (compute .bf16 où sûr, f32 pour norm/softmax/rope/softcap) | argmax == HF ; perf mesurée ; drift acceptable |
| **G4** | **Mono-graphe sans chunking** : `gemma4_gen_long` (mono) sur GPU, mesurer pic VRAM + tok/s | tient en VRAM < 21 Go ; perf G1× (backend natif) ; tokens == G1 |
| **G5** | **Attention fast-path** : `sdpa` (window=512) pour le chemin produit (pas de scores bruts) ; garder le chemin manuel pour la validation | argmax == HF ; perf decode × |
| **G6** | **Batching** : `B` comptime > 1, multi-sequences, perf scaling | argmax == HF par batch ; débit batch × |
| **G7** | **Prefill long-contexte** : chunked prefill (mode `chain`) + flash/paged attention, S=512/2048 | last_hidden == HF (tol large) ; TTFT mesuré |
| **G8** | **Benchmarks** : `gemma4_bench`, cibles tok/s, VRAM, TTFT ; doc résultats | cibles §9.7 atteintes |
| (G9) | **Continuous batching** + sampling (optionnel) | non-régression + fonctionnel |

**Non-vacuité** par gate : corrompre un masque / un poids → le gate doit FAIL (le projet a déjà
`gemma4_gchunk_vacuity.zig` — le réutiliser sur GPU). **Non-régression CPU** : E1/E2 + L1a CPU restent
verts à chaque commit (la config défaut `PrecCfg{}` est byte-identique → preuve HLO `diff -rq`).

### 10.1 Le piège précision (oracle-independence)
Ne pas "valider bf16 contre un oracle bf16 recodé" (le travers v_norm D.0→D.0b). L'oracle reste
**HF (PyTorch) dans sa précision native** + le baseline fp32-CPU. Si le bf16-GPU dérive, on localise par
tap (les 8 taps aux frontières 0/4/13/14/15/19/33 + last_hidden) comme aujourd'hui.

---

## 11. Multi-GPU, tensor/pipeline parallel (hors périmètre E2B, mais documenté)

E2B tient sur **1 GPU 24 Go** → le TP/PP n'est **pas requis**. Documenté pour complétude :
- **Tensor parallel** (TP) : sharder `q_proj`/`k_proj`/`v_proj`/`o_proj` par têtes, MLP par dimension
  intermédiaire, via `zml.sharding` (mesh `mesh:tp`). NCCL all-reduce après attn/MLP. ZML supporte le
  sharding natif. Utile seulement si >E2B ou batching massif.
- **Pipeline parallel** (PP) : les 35 couches sur N GPU, micro-batches. Le chunking par couches
  (`forwardStageGen`) est **déjà un pipeline potentiel** — chaque stage sur un device.
- **DP (data parallel)** : le plus simple pour le batching — répliquer le modèle,split le batch. ZML
  `replicatedSharding` + mesh DP.

→ Recommandation : **mono-GPU pour E2B**. TP/PP = chantier séparé si besoin de batches denses ou de
modèles plus gros (Gemma-4 plus large).

---

## 12. Infra / build / CI

- **Workspace ZML** : configurer le build CUDA (toolchain nvcc, cuDNN, cuBLAS, `libpjrt_cuda`).
  Vérifier `./bazel.sh build //examples/rqz:gemma4_gen_long --config=cuda` (ou équivalent).
- **Container** : image CUDA (nvidia/cuda:12.x + cudnn) pour reproductibilité ; montée sur le workspace.
- **CI** : gate G0 (smoke build CUDA) + G1 (fp32-GPU 4 tokens) en smoke ; G2/G4 en nightly (GPU requis).
- **Smoke** : étendre `scripts/smoke.sh` (déjà livré) avec un check `nvidia-smi` + `ldd` du plugin CUDA
  + un mini-run GPU 4-tokens.
- **Swapfile** : sur GPU, **plus besoin du swapfile CPU** (le mur disparaît). À retirer du runbook GPU.

---

## 13. Risques et mitigations

| Risque | Impact | Mitigation |
|---|---|---|
| Drift bf16 → flip d'argmax | tokens != HF | Knob `compute=.f32` (norm/softmax/rope/softcap en f32) ; ne bf16-iser que les GEMM ; caractériser au G2 |
| `softcap 30·tanh` sensible en bf16 près de ±30 | logits saturés mal arrondis | softcap **toujours fp32** (§5.2) |
| RoPE full θ=1e6 "proportional" partial 0.25 en bf16 | dérive positionnelle | cos/sin **fp32** (déjà le cas, `manualRope` reçoit cos/sin oracle fp32) |
| Sliding window masqué par masque additif O(K²) sur long prefill | perf / VRAM prefill | flash-attention window (G5/G7) ; decode Q=1 non concerné |
| `libpjrt_cuda` absent / version mismatch | G0 fail | container CUDA fixe ; `ldd` check au smoke |
| YOCO shared KV + scatter sur GPU (cohérence) | bug silencieux | non-régression vs CPU-L1a (qui valide YOCO) ; non-vacuité |
| `embed_tokens_per_layer` 4,7 Go en VRAM | VRAM tendue batch dense | pinned-memory + transfert async 1 ligne, ou TP/DP |
| Fusion XLA moins agressive qu'attendu | perf < cible | dump HLO (`--xla_dump_to`), autotune cuBLAS, kernels custom (custom-call) si besoin |
| "L'oracle GPU valide GPU" (piège indépendance) | faux PASS | oracle = HF natif + baseline fp32-CPU, jamais recodé bf16 (§10.1) |

---

## 14. Roadmap phasée (effort indicatif, 1 dev)

| Phase | Gates | Livrable | Effort |
|---|---|---|---|
| **P-GPU-1 : Backend + baseline** | G0, G1 | fp32-GPU reproduit HF (1020 tokens), tok/s mesuré | 1–2 j (surtout infra CUDA) |
| **P-GPU-2 : Précision prod** | G2, G3 | mixed-bf16 reproduit HF, perf ×, contrat précision GPU figé | 2–4 j |
| **P-GPU-3 : Perf decode** | G4, G5 | mono-graphe (sans chunking) + attention fast-path, cible tok/s | 3–5 j |
| **P-GPU-4 : Batching + prefill** | G6, G7 | batch + prefill long-contexte, TTFT | 4–6 j |
| **P-GPU-5 : Bench + durabilisation** | G8 | `gemma4_bench`, doc résultats, container/CI | 2–3 j |
| (P-GPU-6) | G9 | continuous batching + sampling (optionnel) | 3–5 j |

**Total noyau (P1–P5) : ~12–20 j-homme.** Le modèle E2B + le socle déjà GPU-ready rendent ce chantier
beaucoup plus court qu'un portage from-scratch : l'essentiel est la précision + les kernels d'attention
+ la validation.

---

## 15. Non-goals (ce que ce plan ne fait pas)

- **Réécrire le moteur** : `engine.zig` est inchangé sémantiquement (knobs comptime en plus, défaut neutre).
- **Tensor/pipeline parallel** : E2B tient sur 1 GPU (§11). TP/PP = chantier séparé.
- **Multimodal** : toujours hors-scope (chemin texte seulement).
- **Quantization avancée** (INT8/INT4/fp8) : le projet a déjà TurboQuant V-only 4-bits (POC `gemma4_gen_vq`).
  Le port GPU réutilise la brique (`EngineModel(TurboQuantVBrick)`) — c'est un chantier complémentaire,
  pas inclus ici. L'**fp8** (H100) est une extension naturelle post-G3 (matmul fp8 natif Hopper).
- **Serving HTTP / vLLM-like** : hors-scope (moteur de recherche, pas runtime prod).

---

## 16. Checklist d'acceptation du port GPU

- [ ] **G0** : `Platform.target == .cuda` sur la 3090 ; `libpjrt_cuda` linké.
- [ ] **G1** : `gemma4_gen_long` fp32-GPU replay 1020/1020 argmax == HF ; drift vs CPU caractérisé + doc.
- [ ] **G2** : mixed-bf16 1020/1020 argmax == HF ; `max_abs(logits)` < seuil G2 doc.
- [ ] **G3** : full-bf16 (compute sélectif) argmax == HF ; perf G1×.
- [ ] **G4** : mono-graphe GPU (sans chunking) tient en VRAM ; perf decode cible.
- [ ] **G5** : `sdpa` window=512 reproduit HF ; perf ×.
- [ ] **G6** : batch B>1 argmax == HF par séquence ; débit scaling.
- [ ] **G7** : prefill S≥512 last_hidden == HF (tol) ; TTFT mesuré.
- [ ] **G8** : `gemma4_bench` doc tok/s/VRAM/TTFT ; cibles §9.7.
- [ ] **Non-régression CPU** : E1/E2 + L1a CPU toujours PASS à chaque commit (config défaut neutre).
- [ ] **Preuve HLO** : `diff -rq` des dumps config défaut == baseline (PrecCfg{} byte-identique).
- [ ] **Non-vacuité GPU** : `gemma4_gchunk_vacuity` sur GPU → divergence attendue.
- [ ] **Doc** : `ENGINE_LOG.md` GPU append + `GPU_PORT_RESULTS.md` (benchs).

---

> **Conclusion** : le portage GPU de Gemma-4-E2B-it en ZML est **faible effort, haut gain** parce que le
> socle est device-agnostic et le modèle "Effective 2B" tient sur une seule 3090 en bf16. Le chantier
> tient en ~12–20 j-homme, dominé par la **stratégie de précision** (préserver l'argmax en bf16) et les
> **kernels d'attention** (sliding window + YOCO). Le baseline fp32-CPU prouvé reste l'étalon de
> non-régression — la discipline de gates du projet s'applique à l'identique, en changeant l'oracle de
> cible (bit-exact → argmax + drift borné).
