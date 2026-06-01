# ROADMAP — gemma4-zml-probe vers le forward end-to-end

> Établi le 1 juin 2026 (mode autonome, ultracode). Ancré dans `modeling_gemma4.py` (transformers
> 5.9.0) via workflow d'analyse 4 agents. Objectif : porter le forward complet de Gemma-4-E2B-it
> en ZML, chaque maillon validé bit-à-bit vs oracle PyTorch (tolérance 1e-4). Revue cross-LLM prévue.

## Config (text)
hidden=1536, intermediate=6144, 35 couches, 8 têtes / 1 KV, head_dim=256, vocab=262144,
eps=1e-6, activation=`gelu_pytorch_tanh`, num_kv_shared_layers=20, sliding_window=512,
final_logit_softcapping=30.0. `full_attention` aux couches 4,9,14,19,24,29,34 (reste sliding).

## Déjà validé (✅)
- **PLE** (P4.4) ; **YOCO cartographie + policy table** (P5.0/5.1).
- **Sous-couche attention sliding complète** (P5.2.A→G, pilote layer 15 reader × 13 writer) :
  `q/k/v proj → norm → RoPE → QK → masque sliding → softmax → context → o_proj → post_attn_norm → +res`.

## Maillons restants (distinct ops) et plan de gates

| Gate | Contenu | Maillon neuf | Risque | Dépend |
|---|---|---|---|---|
| ~~**P5.2.H — MLP**~~ ✅ | pre_ff_norm → gate/up → `gelu(gate)*up` → down → post_ff_norm → +res. **intermediate=12288 (double-wide, layers 15-34)** ; 6144 (layers 0-14). `Tensor.gelu`=gelu_pytorch_tanh. scan max_abs 5.34e-5. tag `p5.2-h-mlp-zml-pass` | — | LOW | G ✅ |
| **P5.3 — couche décodeur (sliding)** | assemble attention sublayer (E/F/G) + MLP (H) en 1 couche end-to-end | composition | LOW | H |
| **P5.4 — embedding + scale** | `embed_tokens[ids] * √1536` | gather/lookup ZML | MED | — |
| **P5.5 — head** | final `norm` → `lm_head` (tied, vocab 262144) → softcap `30·tanh(x/30)` | gather/big-dot + softcap (tanh) | MED | — |
| **P5.6 — layer 14 full attention** | full attn : RoPE `proportional` + `attention_scaling` (theta propre) | RoPE manuelle (cos/sin de l'oracle en fixture, `q*cos+rotate_half(q)*sin`) — PAS `zml.nn.rope` | **HIGH** | — |
| **P5.7 — assemblage multi-couches / end-to-end** | boucle 35 couches, KV-sharing YOCO câblé, embedding→couches→norm→lm_head→logits vs HF | intégration (KV cache, masques, rotary partagés) | HIGH | tous |

### Notes de faisabilité
- **gelu** : `zml/tensor.zig:1346 Tensor.gelu` = `0.5x(1+tanh(√(2/π)(x+0.044715x³)))` = `gelu_pytorch_tanh`. Direct.
- **Layer 14 RoPE** : `Gemma4TextRotaryEmbedding` calcule cos/sin par `layer_type` ; full = `proportional`
  (init fn dédiée + `attention_scaling`, `cos = emb.cos()*attention_scaling`). Parade : oracle exporte
  cos/sin, ZML applique la rotation manuellement (robuste à proportional/scaling/partial). À vérifier :
  présence d'un `partial_rotary_factor` (non confirmé — l'hypothèse 0.25 du doc D.4 reste à valider).
- **Tied weights** : `lm_head.weight` = `embed_tokens.weight` (vocab 262144 × 1536) — charger une fois.
- **embed_scale** = √1536 ≈ 39.19 (buffer non-persistant, calculé au runtime).
- **PLE** : pipeline auxiliaire (embed_tokens_per_layer + per_layer_model_projection + norm) si
  `hidden_size_per_layer_input>0` — déjà partiellement fait en P4.4, à réintégrer en P5.7.

## Discipline (inchangée)
Oracle PyTorch = source de vérité (lire `modeling_gemma4.py`, ne rien supposer). Gate atomique :
oracle 3090 → fixture → runner ZML → comparaison (fixed-points + scan global, tol 1e-4) → commit + tag.
Noms de runners ZML ≤ ~20c (quota comptime `pjrt.zig`). `max_abs=0.0` global = drapeau jaune →
test de perturbation. Fixtures binaires gitignorées (régénérables). Repo local-only.

## Ordre d'exécution autonome
H (MLP) → P5.3 (couche sliding) → P5.4 (embed) → P5.5 (head) → P5.6 (layer 14, dé-risquage) →
P5.7 (end-to-end). Mise à jour de ce fichier + PLANNING + projets.md à chaque gate fermée.
