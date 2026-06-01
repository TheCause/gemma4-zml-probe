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
| ~~**P5.6 — layer 14 full attn RoPE**~~ ✅ | RoPE manuelle partielle (Q-path). **DÉCOUVERTE : full attn = head_dim 512** (q_proj [4096,1536]), partial_rotary 0.25 (128/512 tournent), theta=1e6, scaling=1.0. RoPE manuelle (cos/sin oracle 512-wide, `rotate_half` via split/neg/concat, `q*cos+rh*sin`). scan max_abs 7.99e-6. **RISQUE LEVÉ.** tag `p5.6-full-qrope-zml-pass` | RoPE manuelle (PAS `zml.nn.rope`) | ✅ | — |
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

## Recette d'assemblage (phase intégration — toutes les ops ci-dessous sont VALIDÉES)

### P5.3 — couche décodeur sliding end-to-end (1 couche, bounded, sans KV cache/PLE)
Oracle = `Gemma4TextDecoderLayer(config, layer_idx=13)` réel (producer sliding, intermediate 6144),
appelé avec `position_embeddings=rotary(h,pos,"sliding_attention")`, `attention_mask`=sliding causal
(S=4 → causal), `shared_kv_states={}`, `past_key_values=None`. ZML compose en UN forward les
maillons déjà validés :
```
residual = layer_input                                  # synthétique pré-norm
h = rmsNorm(layer_input).mul(input_layernorm_w)         # NOUVEAU poids input_ln (même pattern C.2/G)
q = h.dot(q_proj_w).reshape.rmsNorm.mul(q_norm_w) ; zml.nn.rope(q, sliding)   # C.1/C.2/C.3
k = h.dot(k_proj_w)...rmsNorm.mul(k_norm_w) ; zml.nn.rope(k, sliding)         # D.1/D.3/D.4
v = h.dot(v_proj_w)...rmsNorm(no scale)                                        # D.2/D.2b
scores = splitAxis(q).dot(k,.hd).merge.transpose ; += causalAttnMask          # E.1/E.mask
probs = softmax(scores,.k) ; ctx = splitAxis(probs).dot(v,.k).merge.transpose # E.softmax/E.context
attn = ctx.transpose.merge(têtes).dot(o_proj_w)                               # F
h = residual + rmsNorm(attn).mul(post_attn_w)                                 # G
residual = h ; h = rmsNorm(h).mul(pre_ff_w)
h = h.dot(gate_w).gelu().mul(h.dot(up_w)).dot(down_w)                         # H
out = residual + rmsNorm(h).mul(post_ff_w)                                    # H
```
Maillons non encore isolés pour la couche RÉELLE complète :
- `input_layernorm` (= rmsNorm+mul de G/H → trivial).
- **Bloc PLE per-layer** (modeling_gemma4 L1429-1438, actif car `hidden_size_per_layer_input>0`) :
  `residual + post_per_layer_input_norm(per_layer_projection(gelu(per_layer_input_gate(h)) * per_layer_input))`
  puis `× layer_scalar`. 3 nouveaux poids (`per_layer_input_gate`, `per_layer_projection`,
  `post_per_layer_input_norm`) + `per_layer_input` (du pipeline PLE P4.4) + `layer_scalar`. Ops toutes
  de types validés (Linear=dot, gelu, mul, RMSNorm) → gate P5.3.bis avant l'assemblage couche.
Fixture = poids layer 13 + cos/sin sliding + mask + per_layer_input + layer_input + output oracle.

### P5.7 — runtime 35 couches end-to-end (productionisation)
Au-delà de la composition : (1) loader 35 couches (producers 0-14 vs readers 15-34 ; MLP 6144 vs
12288 ; sliding head_dim 256 vs full 512) ; (2) **KV cache YOCO** : producers écrivent
`shared_kv_states`, readers lisent (sliding→13, full→14) via policy table P5.1 (validée) ;
(3) **PLE** par couche (P4.4) ; (4) masques sliding S≥512 + full ; (5) rotary sliding+full pré-calc ;
(6) embedding (P5.4) → couches → norm+head (P5.5) ; (7) génération (decode incrémental) optionnelle.
Phase d'ingénierie runtime (pas validation d'op). Sous-découper : P5.7.a (loader + 1 forward prefill
complet vs HF logits) puis P5.7.b (decode/génération). Idéal en contexte frais.

## Discipline (inchangée)
Oracle PyTorch = source de vérité (lire `modeling_gemma4.py`, ne rien supposer). Gate atomique :
oracle 3090 → fixture → runner ZML → comparaison (fixed-points + scan global, tol 1e-4) → commit + tag.
Noms de runners ZML ≤ ~20c (quota comptime `pjrt.zig`). `max_abs=0.0` global = drapeau jaune →
test de perturbation. Fixtures binaires gitignorées (régénérables). Repo local-only.

## Ordre d'exécution autonome (re-priorisé par risque/information)
~~H (MLP)~~ ✅ → ~~P5.6 (layer 14 RoPE, dé-risquage)~~ ✅ → **P5.4 (embed)** → **P5.5 (head)** →
P5.3 (assemblage couche sliding e2e) → P5.7 (multi-couches). Mise à jour de ce fichier +
PLANNING + projets.md à chaque gate. Après P5.4/P5.5, **toutes les ops distinctes du forward
seront validées en ZML** (assemblage = composition mécanique).

## Journal des gates fermées (mode autonome, 1 juin 2026)
- **P5.2.H** (MLP, double-wide 12288) — `p5.2-h-mlp-zml-pass` — scan 5.34e-5.
- **P5.6** (full attn Q-rope manuelle partielle, layer 14, head_dim 512) — `p5.6-full-qrope-zml-pass` — scan 7.99e-6. **Risque levé.**
- **P5.4** (embedding gather + scale √1536, slice vocab 4096) — `p5.4-embed-zml-pass` — **bit-exact**.
- **P5.5** (head : final norm + lm_head tied + softcap 30·tanh(x/30), slice vocab 4096) — `p5.5-head-zml-pass` — scan 5.44e-5.

## ✅ TOUTES LES OPS DISTINCTES DU FORWARD VALIDÉES EN ZML (1 juin 2026)
gather+scale · rmsNorm(+scale, pattern Llama) · dot (toutes projections + lm_head) · RoPE sliding (zml.nn.rope) + RoPE full partial MANUELLE (split/neg/concat + cos/sin oracle) · QK GQA (splitAxis) · sliding mask (causalAttnMask) · softmax(.k) · context GQA · gelu (Tensor.gelu=gelu_pytorch_tanh) · residual add · softcap (tanh) · KV-sharing routing (policy). Inconnus architecturaux résolus : double-wide MLP (12288, layers 15-34), full attn head_dim 512 + partial rotary 0.25, tied lm_head, embed_scale √1536, softcap 30. **Reste = INTÉGRATION (composition d'ops validées) : P5.3 (couche e2e), P5.7 (35 couches + KV cache + PLE).**

## Découvertes architecturales (vs hypothèses initiales)
- **MLP double-wide** : layers KV-shared (15-34) ont intermediate=12288 (`use_double_wide_mlp`) ; 0-14 = 6144.
- **full_attention head_dim = 512** (global_head_dim), pas 256. q_proj [4096,1536], o_proj [1536,4096], q/k_norm [512]. Partial rotary 0.25 → 128/512 dims tournent. theta=1e6, scaling=1.0. **Pas un simple changement de theta : chemin attention parallèle à dims doublées.**
- **embed_scale** = √1536 ; **lm_head tied** à embed_tokens ; **final_logit_softcapping** = 30·tanh(x/30).
