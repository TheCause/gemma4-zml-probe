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
| ~~**P5.3 — couche décodeur (sliding)**~~ ✅ | couche RÉELLE complète (input_ln + attn + MLP + bloc PLE per-layer + layer_scalar) vs `Gemma4TextDecoderLayer` module. scan **6.72e-5** (PASS 1er essai). tag `p5.3-layer-zml-pass` | composition | ✅ | H ✅ |
| ~~**P5.4 — embedding + scale**~~ ✅ | `embed_tokens[ids] * √1536` (gather ZML). **bit-exact**. tag `p5.4-embed-zml-pass` | gather/lookup | ✅ | — |
| ~~**P5.5 — head**~~ ✅ | final `norm` → `lm_head` (tied) → softcap `30·tanh(x/30)`. scan 5.44e-5. tag `p5.5-head-zml-pass` | softcap (tanh) | ✅ | — |
| ~~**P5.6.K — K full-rope**~~ ✅ | RoPE manuelle partielle K (layer 14, gap fermé par audit closeout). scan 2.68e-7. tag `p5.6k-full-krope-zml-pass` | — (= P5.6 sur K) | ✅ | — |
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
Phase d'ingénierie runtime (pas validation d'op). **Idéal en contexte frais.**

**Sous-découpage P5.7 (un seul nouveau type de complexité par gate — proposé Régis 2 juin) :**
- ~~**P5.7.0**~~ ✅ — loader manifest only (lister/mapper les poids des 35 couches, aucun compute).
  Script `scripts/34_p5_7_0_loader_manifest.py`, manifest `fixtures/p5_7_0_loader_manifest.json`.
  Résumé : 600 clés disque attendues, 540 tenseurs runtime à charger, 60 clés K/V reader ignorées au
  runtime (YOCO), `v_norm` documenté comme op RMSNorm sans poids. Validation checkpoint optionnelle
  via `--require-weights`; sautée localement si `weights/model.safetensors` absent.
- ~~**P5.7.1**~~ ✅ — load embeddings + 1 couche depuis le checkpoint RÉEL bf16, bit-exact (tag `p5.7.1-load-zml-pass`).
- ~~**P5.7.2**~~ ✅ — load les 35 couches (597 tenseurs, []LayerW + prefix/withLayer), shapes par-couche OK (tag `p5.7.2-loadall-zml-pass`).
- ~~**P5.7.3**~~ ✅ — runtime plan 35 couches (dispatch type/reader/target/dims/rope/mask), `fixtures/p5_7_3_runtime_plan.json` (tag `p5.7.3-runtime-plan-done`).
- ~~**P5.7.4**~~ ✅ — couche FULL attention complète (layer 14, head_dim 512 + RoPE manuelle) vs module réel, scan 1.63e-5. **Les 2 types de couche validés e2e** (sliding P5.3 + full P5.7.4) (tag `p5.7.4-full-layer-zml-pass`).
- **P5.7.5 ⏳ — prefill 35 couches, no generation. DÉCISION DE DESIGN OUVERTE :**
  - **Blocage mémoire** : le `Gemma4TextModel` complet en fp32 (~17 Go, `embed_tokens_per_layer` fp32 = 9.4 Go) **ne tient pas** dans les 23 Go de la VM 3090. Oracle relancé en **bf16** (~9 Go, dtype natif checkpoint) → `fixtures/p5_7_5_prefill.safetensors` (last_hidden_state, cos/sin sliding+full, masque).
  - **Stratégie de précision à trancher** : oracle bf16 vs moteur ZML fp32 → comparaison **lâche** (drift bf16 sur 35 couches). Options : (a) moteur ZML en bf16 pour matcher HF (matmul bf16 + norm fp32) ; (b) oracle fp32 via load_state_dict(assign=True)+meta (matérialiser les buffers rotary) ou machine ≥24 Go ; (c) tolérance structurelle (~1e-1 abs) acceptant que la comparaison prouve le CÂBLAGE (35 couches + KV sharing + dispatch), pas la précision bit. **Recommandation : (b) si faisable (rigueur fp32 conservée), sinon (c) documenté.**
  - **Build moteur** (composition d'ops validées, non encore écrit) : embedding (P5.4) + PLE frontend (P4.4) → boucle 35 couches dispatchées sliding/full (P5.3/P5.7.4) avec **KV sharing YOCO** (capter K/V de layer 13 sliding + 14 full, réutilisés par readers 15-34) → final norm. Runner unrollé (idiome qwen `[]TransformerLayer`).
- **P5.7.6** — comparaison logits vs HF (le test e2e décisif) — head P5.5 sur le last_hidden de P5.7.5.
- **P5.7.7** — decode 1 token (KV cache incrémental, pos_idx).
- **P5.7.8** — decode N tokens (génération).

Prérequis avant P5.7 : audit closeout fait (`docs/P5_6_closeout.md`, base saine, 0 gap).

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
- **P5.7.0** (loader manifest only, 35 couches, no compute) — `fixtures/p5_7_0_loader_manifest.json`.
  600 clés disque attendues ; 540 tenseurs runtime à charger ; 60 K/V reader disk-only ignorés ; validation
  checkpoint prête via `python scripts/34_p5_7_0_loader_manifest.py --require-weights` quand les poids sont présents.

## ✅ COUCHE DÉCODEUR SLIDING COMPLÈTE VALIDÉE E2E (P5.3, 2 juin) + TOUTES OPS DISTINCTES (1 juin)
gather+scale · rmsNorm(+scale, pattern Llama) · dot (toutes projections + lm_head) · RoPE sliding (zml.nn.rope) + RoPE full partial MANUELLE (split/neg/concat + cos/sin oracle) · QK GQA (splitAxis) · sliding mask (causalAttnMask) · softmax(.k) · context GQA · gelu (Tensor.gelu=gelu_pytorch_tanh) · residual add · softcap (tanh) · KV-sharing routing (policy). Inconnus architecturaux résolus : double-wide MLP (12288, layers 15-34), full attn head_dim 512 + partial rotary 0.25, tied lm_head, embed_scale √1536, softcap 30. **Reste = INTÉGRATION (composition d'ops validées) : P5.3 (couche e2e), P5.7 (35 couches + KV cache + PLE).**

## Découvertes architecturales (vs hypothèses initiales)
- **MLP double-wide** : layers KV-shared (15-34) ont intermediate=12288 (`use_double_wide_mlp`) ; 0-14 = 6144.
- **full_attention head_dim = 512** (global_head_dim), pas 256. q_proj [4096,1536], o_proj [1536,4096], q/k_norm [512]. Partial rotary 0.25 → 128/512 dims tournent. theta=1e6, scaling=1.0. **Pas un simple changement de theta : chemin attention parallèle à dims doublées.**
- **embed_scale** = √1536 ; **lm_head tied** à embed_tokens ; **final_logit_softcapping** = 30·tanh(x/30).
