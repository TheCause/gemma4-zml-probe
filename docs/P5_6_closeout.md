# P5.6.closeout — Audit de complétude pré-P5.7

> 2 juin 2026. Verrouillage avant d'ouvrir le runtime 35 couches. Audit par workflow 3 agents
> adversariaux (complétude composants / tolérances-divergences / cohérence docs-mémoire) +
> fermeture du seul gap réel (K-full-rope). **Verdict : base saine pour P5.7.**

## 1. Matrice de complétude — composant forward Gemma4 → runner / tag / preuve / tolérance

Tolérance projet = **1e-4** (5e-4 pour la couche composée P5.3). Résidu = matmul PJRT-CPU (Eigen) vs
PyTorch BLAS, croît avec la longueur de réduction. `0.0` = bit-exact (réduction courte ou op exacte).

| # | Composant (modeling_gemma4) | s/f | Runner ZML | Tag | Preuve (scan max_abs) |
|---|---|---|---|---|---|
| 0a | embed_tokens × √1536 (scaled embedding) | both | gemma4_embed | p5.4-embed-zml-pass | **0.0** bit-exact |
| 0b | PLE amont : embed_tokens_per_layer + per_layer_model_projection + per_layer_projection_norm | both | gemma4_ple_fixture | P4.4.2 (A→J) | bit-exact→1.5e-5 |
| 0c | rotary cos/sin (sliding `default` θ1e4 / full `proportional` θ1e6 partial 0.25) | both | (consommé par rope runners) | p5.2-c3 / p5.6 | — |
| 0d | masks (causal / sliding window) | both | gemma4_sliding_mask | p5.2-emask | structurel 43/64 |
| 1 | input_layernorm (RMSNorm) | both | gemma4_layer | p5.3-layer-zml-pass | 6.72e-5 (composée) |
| 2 | q_proj | both | gemma4_q_proj | p5.2-c1 | 1.14e-5 |
| 3 | q_norm (RMSNorm) | both | gemma4_q_norm | p5.2-c2 | 6.68e-6 |
| 4 | RoPE Q sliding (zml.nn.rope) | s | gemma4_q_rope | p5.2-c3 | 6.68e-6 |
| 4f | RoPE Q full (manuelle partielle, head_dim 512) | f | gemma4_full_qrope | p5.6-full-qrope-zml-pass | 7.99e-6 |
| 5 | k_proj | both | gemma4_k_proj | p5.2-d1 | 5.48e-6 |
| 6 | k_norm (RMSNorm) | both | gemma4_k_norm | p5.2-d3 | 5.36e-7 |
| 7 | RoPE K sliding (zml.nn.rope) | s | gemma4_k_rope | p5.2-d4 | 5.36e-7 |
| 7f | RoPE K full (manuelle partielle, head_dim 512) | f | gemma4_full_krope | **p5.6k-full-krope-zml-pass** | 2.68e-7 |
| 8 | v_proj | both | gemma4_v_proj | p5.2-d2 | 5.25e-6 |
| 9 | v_norm (RMSNorm **with_scale=False**) | both | gemma4_v_norm / kv_slot | p5.2-d2b / p5.2-d5-kv-slot-mock-pass | 2.38e-7 / 4.17e-6 |
| 10 | QK scores (Q·Kᵀ, scaling 1.0, GQA) | both | gemma4_qk_scores | p5.2-e1 | 2.38e-6 |
| 11 | + mask additif | both | (gemma4_layer / sliding_mask) | p5.2-emask / p5.3 | structurel / 6.72e-5 |
| 12 | softmax(.k) fp32 | both | gemma4_softmax | p5.2-esoftmax | 2.98e-8 |
| 13 | context (probs·V, GQA) | both | gemma4_context | p5.2-econtext | **0.0** (non-vacuous, perturbation) |
| 14 | o_proj (concat têtes + linear) | both | gemma4_oproj | p5.2-f | 2.29e-5 |
| 15 | post_attention_layernorm + résiduel | both | gemma4_attn_resid | p5.2-g | 9.54e-7 |
| 16 | pre_feedforward_layernorm | both | gemma4_mlp | p5.2-h | 5.34e-5 |
| 17 | MLP gate/up + gelu_pytorch_tanh + down (**6144 / 12288 double-wide**) | both | gemma4_mlp | p5.2-h | 5.34e-5 |
| 18 | post_feedforward_layernorm + résiduel | both | gemma4_mlp | p5.2-h | 5.34e-5 |
| 19 | bloc PLE per-layer (gate→gelu→×per_layer_input→proj→norm→+res) | both | gemma4_layer | p5.3-layer-zml-pass | 6.72e-5 |
| 20 | × layer_scalar (0.0884) | both | gemma4_layer | p5.3-layer-zml-pass | 6.72e-5 |
| 21 | final norm (Gemma4TextModel.norm) | model | gemma4_head | p5.5-head-zml-pass | 5.44e-5 |
| 22 | lm_head (tied = embed_tokens) | model | gemma4_head | p5.5-head-zml-pass | 5.44e-5 |
| 23 | final_logit_softcapping (30·tanh(x/30)) | model | gemma4_head | p5.5-head-zml-pass | 5.44e-5 |
| — | couche décodeur sliding COMPLÈTE (1→20 composés) | s | gemma4_layer | p5.3-layer-zml-pass | 6.72e-5 |
| — | KV-sharing routing (policy table) | both | gemma4_policy_lookup / routing_mock | p5.2-a / p5.2-b | structurel 35/35 |

**Couverture : COMPLÈTE.** Aucun composant non mappé. (MOE `router`/`experts` = `enable_moe_block=False`
sur E2B → hors scope, non instancié.)

## 2. Checklist de verrouillage

- [x] **Tous les tags p5* PASS** : 30 tags (29 + p5.6k), tous PASS (cf §4).
- [x] **Chaque op forward → runner validé** : matrice §1, 0 gap (K-full-rope fermé par p5.6k).
- [x] **Chaque divergence HF/ZML expliquée** : résidu = matmul PJRT-CPU Eigen vs PyTorch BLAS,
  linéaire en longueur de réduction (cf §3). Audit tolérances : ALL_JUSTIFIED.
- [x] **Chaque tolérance justifiée** : §3. Marges 1.8×→bit-exact ; aucune injustifiée.
- [x] **Tags superseded/canonical documentés** : §4.
- [x] **ROADMAP ↔ PLANNING alignés** : table ROADMAP corrigée (P5.4/P5.5 marquées closed).
- [x] **Aucun faux invariant en mémoire** : audit cohérence = aucun. Hypothèses initiales
  (partial_rotary, double-wide, full head_dim 512, o_proj clipping) toutes corrigées/re-documentées.

## 3. Justification des tolérances (audit numérique = ALL_JUSTIFIED)

- **Linéarité résidu ∝ longueur réduction** : `.h=1536`→1.14e-5 ; même réduction output 4× plus petit→5.48e-6 ;
  `.m=2048`→2.29e-5 ; `.f=12288` (×100 magnitude)→5.34e-5. Cohérent matmul Eigen vs BLAS.
- **Atténuation RMSNorm** : C.1 1.14e-5 → C.2 6.68e-6 (÷1.7, =RMS) ; D.1 5.48e-6 → D.3 5.36e-7 (÷10, poids uniforme).
- **RoPE orthogonale** : C.3 max_abs = C.2 (rotation préserve la norme, aucun drift).
- **Bit-exact justifiés** : E.context 0.0 (réduction `.k=4` ≤4 termes, arrondi déterministe ; **non-vacuous
  prouvé par test oracle corrompu** → sortie réelle ≠ zéros, h=0..7 couverts) ; P5.4 embed 0.0 (gather = sélection
  exacte, scale = mul scalaire).
- **Gates marginales mais justifiées** (<5×) : H mlp 5.34e-5 (réduction 12288), head 5.44e-5, P5.3 couche
  6.72e-5 vs 5e-4 (cumul multi-matmul, layer_scalar 0.088 réduit le résidu ~11×).
- **Oracles indépendants** : E.context (repeat_kv PyTorch vs splitAxis ZML convergent), o_proj (einsum h-major
  vs F.linear), G/H/P5.3 (modules réels Gemma4RMSNorm/Gemma4TextDecoderLayer, pas ré-dérivation).

## 4. Tags superseded / canonical

- **p5.2-d5-kv-slot-mock-pass** = CANONICAL (V = RMSNorm sans scale, correction D.0b du bug v_norm).
- **p5.2-d5-zml-kv-slot-mock-pass** = **SUPERSEDED** (V brut, bug D.0). Conservé volontairement pour tracer
  le faux PASS « end-to-end » (oracle et ZML partageaient l'hypothèse fausse → accord trompeur ~5e-6). Ne PAS
  réutiliser. Leçon : oracle indépendant de la source de vérité (cf `feedback_oracle_independence`).

## 5. Verdict stratégique

Risque **scientifique/numérique LEVÉ** (toutes ops + chemins sliding/full + couche e2e + head validés).
Reste le risque **intégration / mémoire / orchestration / performance** = P5.7 (runtime). La question n'est
plus « est-ce possible ? » mais « comment le produire proprement ». Sous-découpage P5.7.0→P5.7.8 (un seul
nouveau type de complexité par gate) dans `ROADMAP_to_full_forward.md`.
