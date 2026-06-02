# CARTOGRAPHIE — Portage du moteur d'inférence Gemma-4 → ZML

> **Source de vérité texte** d'une planche de facilitation graphique (sketchnote) résumant tout le
> portage. À recopier tel quel si la planche est ré-éditée — un re-render par image-gen corrompt
> systématiquement les termes techniques (PJRT, choose1d, v_norm sans scale, MLP 12288…) ; éditer le
> texte à la main ou via un outil vectoriel (texte éditable), jamais par régénération d'image.

**Porter un cerveau d'un monde (PyTorch) à un autre (ZML), brique par brique, en prouvant chaque brique
contre l'original — jusqu'à ce qu'il génère le même texte.**

Colonne vertébrale : **MODÈLE SOURCE → MÉTHODE → MOTEUR ZML → PREUVE (== HF)**

---

## ① LE MODÈLE SOURCE : Gemma-4-E2B-it

- E2B = Effective 2B
- 2,3 Md effectifs / **5,1 Md** avec embeddings
- dense (pas MoE) · Google · 2 avril 2026 · Apache-2.0
- 35 couches · hidden 1536 · **vocab 262144**
- PLE — Per-Layer Embeddings · 2ᵉ table d'embeddings
- embed_tokens_per_layer ×√256 = 16 · embed_tokens ×√1536 · lm_head tied
- **Sliding** (locale) : head_dim 256 · RoPE θ=1e4 · fenêtre 512 · **MLP 6144**
- **Full** (globale) : head_dim 512 · RoPE partielle 0.25 · 128/512 tournent · **θ=1e6 proportional** · **MLP 12288**
- full : 4, 9, 14, 19, 24, 29, 34
- Shared KV Cache / YOCO · writers 13/14 → readers 15-34
- GQA : 8 têtes Q / 1 tête KV
- **RMSNorm : pattern Llama ×weight ≠ Qwen ×(1+weight)**
- q_norm, k_norm, **v_norm sans scale**
- **gelu_pytorch_tanh** · Softcap final : **30·tanh(x/30)** · layer_scalar
- Multimodal hors-scope (texte seulement)

## ② LA CIBLE : ZML

- Zig + MLIR + OpenXLA + **PJRT** + Bazel
- forward compilé en binaire natif AOT · **pas de Python** · Tensor symbolique
- tags d'axes nommés : `.b .s .d .h .hd .k .slot .layer .voc` · contraction par tag
- Ops fondamentales : `dot` · `reshape` · `withTags` · `scale` · `rmsNorm` · `rope` · `splitAxis`/`merge` · `transpose`/`rename` · `softmax` · `gelu` · `concatenate` · `convert`
- Ops decode : `scatterSlices` · `dynamicSlice` · **`choose1d`**
- Buffer (device) · TensorStore / safetensors · Bufferized · `zml.io.load`
- idiome KvCache : cache `{layer/slot, k, h, hd}` + scatter à `(slot, pos)`
- Backend : **libpjrt_cpu** (CPU fp32) · extensible : CUDA / ROCm / TPU

## ③ LA MÉTHODE

- 1 op = 1 gate · 1 nouvelle complexité par gate · Oracle = source de vérité
- Multi-tap isolation · bug localisé sur un tap précis · Oracle independence · piège v_norm : D.0 → D.0b
- **Boucle** : Lire `modeling_gemma4.py` (ne rien supposer) → ORACLE PyTorch → FIXTURE (`.safetensors`) → RUNNER ZML → COMPARAISON → PASS (tol 1e-4) ? → COMMIT + TAG / **sinon : localiser**
- Contrat de précision hybride · fp32 partout sauf `embed_tokens_per_layer` en bf16
- Revue adversariale · **non-vacuité** · drapeau jaune : max_abs = 0 sur un matmul = suspect · perturbation test
- résidu ~1e-5 / 1e-6 = bruit matmul **PJRT-CPU** vs PyTorch BLAS

### MAPPING Gemma4 ↔ ZML

| Concept Gemma4 | Traduction ZML |
|---|---|
| embed ×√1536 / PLE ×√256 | gather + scale + fusion /√2 |
| RMSNorm (**Llama**) | `zml.nn.rmsNorm` + `.mul(weight)` |
| RoPE sliding | `zml.nn.rope(pos)` natif |
| RoPE full partielle | `manualRope` (split/neg/concat + cos/sin oracle) |
| GQA (8Q/1KV) | `splitAxis(.h)` des têtes Q |
| Shared KV / YOCO | `scatterSlices(.slot,.k)` + **`choose1d(.slot)`** |
| KV cache qui grandit | `scatterSlices(.k = pos)` |
| softcap 30 | `scale·tanh·scale` |
| sliding window | masque (pas de troncation cache) |

## ⑤ LE FLUX DE DONNÉES

```
tokens → embed ×√1536 + PLE frontend (token_identity + context)/√2
   ↓
BOUCLE 35 COUCHES :
   input layernorm
   → ATTENTION [ q/k/v_proj · q/k norm, v_norm (sans scale) · RoPE
                 · QK (GQA splitAxis) · + MASK · softmax · ×V (context) · o_proj ]
   → + résiduel
   → MLP [ pre_ff_norm · gate·gelu × up · down ]
   → + résiduel
   → BLOC PLE per-layer [ gate·gelu × per_layer_input · proj · norm ]
   → × layer_scalar
   ↓
final norm → lm_head (tied) → softcap 30 → LOGITS → argmax → TOKEN
                                                     (108 injecté → 1018 généré)
```

- **PREFILL** : S=4 · pos_idx = 0..3 · mask causal · writers 13/14 publient K/V
- **DECODE** : S=1 · KV cache qui **GRANDIT** · `scatterSlices` à pos · pos_idx absolu · mask incrémental · readers réutilisent le cache du writer
- **GÉNÉRATION** : decode bouclé · cache threadé · **step en step** · **pos incrémenté**
- writers 13/14 → **YOCO** → readers 15-34

## ⑥ L'INFRASTRUCTURE

- **M1** (MacBook, SOURCE canonique) : écrit oracles + runners · git commits + tags
- `rsync` / `ssh` (jumphost macmini)
- **3090** (COMPUTE) : `venv gemma4-probe` (PyTorch) · ZML workspace + bazel build · checkpoint full 9,6 Go / slim 3,6 Go · fixtures · `libpjrt_cpu`

## ④ LE PARCOURS

PLE (P4.4) → YOCO carto (P5.0/5.1) → ops attention (P5.2 A→H) → couche sliding (P5.3) → embed (P5.4) → head (P5.5) → RoPE full (P5.6) → loader/dispatch (P5.7.0-4) → **MOTEUR PREFILL 35 couches** (P5.7.5) → **logits == HF** (P5.7.6) → **DECODE** [ pilote sliding (decode-1) + primitives isolées + pilote full (decode-2) + moteur e2e (decode-3) ] → **GÉNÉRATION N tokens** (P5.7.8)

## ⑦ LA PREUVE

- prefill : `last_hidden` bit-near ~1e-5 vs HF
- logits : tokens identiques à HF (0 flip)
- decode 1 token : last_hidden + logits + argmax == HF (1018)
- decode e2e : last_hidden 2.9e-4 · logits 1.0e-4
- génération N tokens : séquence ZML == HF greedy `[1018, 6398, 25967, 53121]`
- ~50 gates tagués · dernier : `p5.7.8-generation-pass`
- 🏁 **Gemma-4-E2B tourne en ZML, sortie identique à HuggingFace**

## INTERACTIONS MAJEURES

1. Gemma4 archi lue dans `modeling_gemma4.py` → reproduite par les ops ZML
2. Oracle → Fixture → Runner ZML → refs oracle → PASS/tag
3. Prefill → KV caches → Decode → Génération
4. Writers 13/14 → Shared KV → Readers 15-34
5. Contrat hybride → contrainte mémoire → 1 process / VM 24 Go
6. Multi-tap + oracle independence + revue adversariale → évitent le faux PASS
