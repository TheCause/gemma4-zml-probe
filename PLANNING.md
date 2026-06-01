# Planning gemma4-zml-probe

> Sonde PLE puis portage ZML de `google/gemma-4-E2B-it`. Roadmap P-1 → P7 (section 10 procédure d'origine).

## Etat 31 mai 2026 (P5.2.E.mask PASS — ZML sliding mask réel S=8/window=3, bit-exact)

P-1 ✅ · P2 ✅ · P3 ✅ · P4-prep ✅ · P4.3 ✅ · P4.4.0 ✅ · P4.4.1 ✅
**P4.4.2 ✅** gates : A ✅ B ✅ C ✅ D ✅ E ✅ F ✅ G ✅ H ✅ I ✅ **J ✅**
**P5.0 ✅ · P5.1 ✅ · P5.2.A ✅ · P5.2.B ✅ · P5.2.C ✅ complet (C.0/C.1/C.2/C.3)**
**P5.2.D ✅ COMPLET (branche V réparée)** : bug `v_norm` (RMSNorm `with_scale=False` non appliqué à V) découvert le 30 mai en préparant E, corrigé en 3 sous-gates.
  - **D.0b ✅** oracle K/V corrigé (V RMSNormed sans scale) — tag `p5.2-d0b-v-norm-oracle-pass`
  - **D.2b ✅** ZML v_norm sans scale — scan global 2.384e-7 (marge ~420 000×) — tag `p5.2-d2b-zml-v-norm-pass`
  - **D.5 ✅** KV slot corrigé (`value = v_after_norm`) — K_slot 5.36e-7, V_slot 4.17e-6, sanity 0.777 — tag `p5.2-d5-kv-slot-mock-pass` (ancien V-brut `p5.2-d5-zml-kv-slot-mock-pass` superseded)
**P5.2.E ✅ COMPLET** (pilote = layer 15 reader → KV layer 13) — chaîne d'attention ZML `QK → mask → softmax → context` validée bout-en-bout vs oracles PyTorch indépendants :
  - **E.0 ✅** oracle PyTorch attention `Q15×K13ᵀ → mask → softmax → V13 → context` — scaling 1.0, GQA repeat_kv 8, masque additif, V normé (0.7768), sliding≡causal prouvé à S=4<512, Σprobs−1=1.19e-7, futur=0 strict — fixture `p5_2_e0_attention_oracle_layer15_kv13.pt` (md5 `f88ea58d…` M1≡3090) — tag `p5.2-e0-pytorch-attention-oracle-pass`
  - **E.1 ✅** ZML QK scores only — `splitAxis(Q GQA) → dot(.hd) → merge(.h,.hq) → transpose` (scaling 1.0, PAS de `1/√hd`), forward `[1,8,4,4]` ✓, scan global **max_abs 2.384e-6** (marge ~42 000× vs 1e-4) vs `scores_raw`. Helper `sdpa`/`attention` écartés (n'exposent pas les scores bruts) → dot manuel. Vérif adversariale : 2 expressions GQA (repeat_kv PyTorch vs splitAxis ZML) convergent. Runner `gemma4_qk_scores.zig` — tag `p5.2-e1-zml-qk-scores-pass`
  - **E.mask ✅** sliding mask **réel** S=8/window=3 (cas mordant) — helper natif `zml.nn.causalAttnMask(.{.q=8,.k=8}, .f32, 3)` + `add(broad)`. Grille ZML == oracle, masked **43/64** (mask) **86/128** (scores), visible **bit-exact** (0.0), struct_mismatch=0. Convention triple-validée (transformers `sliding_window_overlay` = helper ZML = table Régis). Preuve que la fenêtre mord : causal pur masquerait 28, sliding 43 (+15 anciennes). Garde `qlen>=window` = ce qui dégénérait E.0/E.1. Runner `gemma4_sliding_mask.zig` — tag `p5.2-emask-sliding-mask-pass`
  - **E.softmax ✅** ZML softmax only — `scores_masked.softmax(.k)` fp32 (conv. sdpa), forward `[1,8,4,4]` ✓, Σprobs−1=**1.19e-7**, futur masqué=**0** strict, NaN/Inf=false, 3 fixed-points **bit-exact** (max_diff 0.0), scan global **max_abs 2.98e-8** (marge ~3300× vs 1e-4, < jitter QK E.1 car softmax borne [0,1]). finfo.min géré (sub max → exp=0). **Piège build** : nom long `gemma4_attention_softmax` (24c) débordait le quota comptime 1000 branches de `pjrt.zig structSize` (`@setEvalBranchQuota` dans `main` n'atteint pas cette Sema) → renommé runner **`gemma4_softmax.zig`** (14c). Script `24_` (23 pris). — tag `p5.2-esoftmax-zml-pass`
  - **E.context ✅** ZML context dot — `probs.splitAxis(.h){.h=1,.hq=8}.dot(v_final,.k).merge(.h,.hq).transpose` (GQA par split des têtes Q, miroir E.1 ; V=`v_final` RMSNorm no-scale D.0b), forward `[1,8,4,256]` ✓, NaN/Inf=false, max|context|=6.26, 3 fixed-points (dont **h=7** → GQA correcte) **bit-exact**, scan global **max_abs 0.0 / mean_abs 0.0 bit-exact** (réduction .k=4 courte → PJRT-CPU≡BLAS). Garde anti-régression V auto-contenue : RMS(v_final,hd)≈1 (dev 4.77e-7). Design figé par **workflow 4-agents adversarial** (split PROBS pas V). Runner `gemma4_context.zig` (14c). Script `25_`. — tag `p5.2-econtext-zml-pass`
**P5.2.F ✅** ZML o_proj (projection sortie attention, layer 15) — `context.transpose({.b,.q,.h,.hd}).merge({.m={.h,.hd}}).dot(o_proj_weight,.m)` → `[1,4,1536]`. o_proj TEXTE = `nn.Linear(2048,1536,bias=False)` **sans clipping** (Gemma4ClippableLinear = attention VISION, fausse alerte écartée). Transpose(1,2) manquant dans context E.0 géré. Concat têtes h-major confirmé (einsum oracle 5.72e-6). Forward `[1,4,1536]`, 2 fixed-points max_diff 2.50e-6, scan global **max_abs 2.29e-5** (réduction .m=2048, cohérent q_proj C.1 ; résidu non-nul → réfute echo oracle). Runner `gemma4_oproj.zig`. Script `26_`. Oracle `nn.Linear` lit `...layers.15.self_attn.o_proj.weight` [1536,2048]. — tag `p5.2-f-oproj-zml-pass`
**P5.2.G ✅** ZML post_attention_layernorm + résiduel (layer 15) — `out = residual + rmsNorm(attn_output,.d).mul(pa_norm_weight)`. Ground truth `Gemma4TextDecoderLayer.forward` L1395-1406 (sandwich norm). post_attn_ln = `Gemma4RMSNorm(1536,eps=1e-6,with_scale)` pattern Llama (`*weight`, weight non-uniforme mean 0.914). Oracle = **module réel Gemma4RMSNorm** (pas ré-dérivation). residual = stand-in `hidden_input` C.0 (vrai résiduel pré-input_layernorm non modélisé par le pilote → valide l'OP, pas la sémantique e2e). Forward `[1,4,1536]`, 2 fixed-points 2.38e-7, scan global **max_abs 9.54e-7** (marge ~100 000×). Runner `gemma4_attn_resid.zig`. Script `27_`. — tag `p5.2-g-attn-resid-zml-pass`
Tag courant : `p5.2-g-attn-resid-zml-pass`. **SOUS-COUCHE ATTENTION COMPLÈTE** (q/k/v proj/norm/rope → QK → mask → softmax → context → o_proj → post_attn_norm → +residual). Reste vers couche décodeur complète : **MLP** (gate/up/down + activation + pre/post_feedforward_layernorm + résiduel), `input_layernorm` + câblage e2e résiduel, layer 14 full attention (p-RoPE proportional, patch zml.nn.rope), vrai sliding au compute (S≥512).

> **Gemma 4 E2B PLE minimal ZML validated end-to-end.**

Synthèse numérique P4.4.2 :
- A→G : tous bit-exact PJRT CPU vs numpy fp32 (`max_diff = 0.0`).
- H (rmsNorm + weight) : 1.49e-8 vs numpy fp32 — 1 ULP fp32 sur 2 blocks, bit-exact sur 2 autres.
- I (fusion add) : 1.49e-8 — heritage exact de H, l'add n'ajoute aucun drift.
- J (scale 1/√2 + comparaison fixture) :
  - 4 blocks max_diff : **1.79e-7** vs fixture fp32 (`ple_reference_final.npy`).
  - Scan global 35840 valeurs : max_abs **1.526e-5** (flat_index 10756), mean_abs **1.85e-7**.
  - Tolérance 1e-4, marge ~6500×.
  - Résidu confirmé = matmul PJRT-CPU Eigen-like vs PyTorch BLAS (P4.3 observait 1.53e-5 numpy vs PyTorch, on retrouve à l'octet près).

Le matmul Gate E est la seule source de divergence ; tout le reste de la chaîne PLE reproduit PyTorch à <2e-7. Le PLE-only ZML minimal est **validé end-to-end** pour gemma-4-E2B-it sur l'input `'ZML test prompt'`.

**Next session: P5 (YOCO / Shared KV) — débloqué, non démarré.**

### Connaissance capitalisée pour P5 et au-delà

1. **Piège ZML #1 — reshape perd les tags** : `reshape(.{...})` retourne shape anonyme. Re-tagger via `.withTags(.{ ... })` avant toute op qui cible un axe par tag (rmsNorm, mul/add cross-tagged).
2. **Piège ZML #2 — mul/add ne broadcastent pas implicitement** : `weight.broad(other.shape())` obligatoire (pattern Llama `model.zig:391`).
3. **Choix RMSNorm verrouillé** : pattern Llama `normalized.mul(weight)`. Variante Qwen3.5 `normalized.mul(1+weight)` interdite pour Gemma 4.
4. **Numérique attendu** : matmul PJRT-CPU vs PyTorch BLAS introduit ~1.5e-5 résidu. Pour valider une couche entière vs référence PyTorch, viser tolérance 1e-4 ; pour valider la couche vs numpy fp32 reproduit localement, viser bit-exact ou 1 ULP.
5. **Piège Gemma4 #1 — `with_scale=False` ≠ pas de normalisation** : `v_norm = Gemma4RMSNorm(head_dim, eps, with_scale=False)` normalise V (division RMS) **sans** poids appris. L'absence de `v_norm.weight` au checkpoint = « pas de poids », PAS « pas de norm ». V est RMSNormé sans scale ; K et Q sont RMSNormés AVEC scale (`with_scale=True`). En ZML : V = `zml.nn.rmsNorm(v_4d, .hd, eps)` **sans** `.mul(weight)`. (bug D.0→D.0b, 30 mai)
6. **Principe méthodo — l'oracle doit être indépendant du code testé** : le bug v_norm a survécu à un PASS « end-to-end » parce que l'oracle PyTorch ET l'implémentation ZML partageaient la même hypothèse fausse → ils s'accordaient à ~5e-6 (fausse confiance). Un oracle ne révèle un bug que s'il dérive de la **source de vérité** (`modeling_gemma4.py`), jamais d'une hypothèse ré-encodée à la main.
7. **Faits attention Gemma4 (pour P5.2.E)** : `Gemma4TextAttention.scaling = 1.0` (PAS √head_dim — la norm passe par q_norm), **pas de softcap d'attention** (seulement `final_logit_softcapping` en P7), GQA via `repeat_kv` (`num_key_value_groups = 8`), masque **additif**, softmax fp32.

## Planning gemma4-zml-probe

### Haute priorité

- [H] **P4.4.2 — Mini-runner ZML PLE-only**
  - Charger `fixtures/ple_fixture.safetensors` via `zml.safetensors.TensorRegistry`.
  - Reproduire le pipeline PLE :
    `lookup × √1536`, `lookup × √256`, projection `× 1/√1536`, reshape, RMSNorm Gemma 4 pure `* weight` ε=1e-6, fusion `/√2`.
  - Comparer à `ple_reference_final` (chargé depuis le même safetensors).
  - Gate fp32 : `max_abs ≤ 1e-4` + fixed point `[0,0,0,:4]` aligné.
  - Sans YOCO, sans attention, sans KV-cache.

### Medium

- [M] **P5 — Shared KV / YOCO** (`num_kv_shared_layers=20`) : inspecter forward Transformers, tracer shapes et cache lifecycle.
- [M] **P6 — Attention hybride** : pattern `layer_types` 4×sliding + 1×full × 7 (full aux couches 4, 9, 14, 19, 24, 29, 34), p-RoPE.

### Backlog

- [B] **P7 — Logits** : `final_logit_softcapping = 30.0`, top-k overlap, flip-rate temp=0.
- [B] Intégrer `05` et `06` dans `04_run_all.sh`.
- [B] Tester d'autres `input_ids` que `'ZML test prompt'` pour vérifier la généralisation du sous-graphe.

## Garde-fous

- Ne PAS écrire `gemma4.zig` complet avant que P4.4.2 mini-runner PLE-only passe.
- Ne PAS ouvrir P5 (YOCO) tant que P4.4.2 n'est pas fermé.
- Référence à viser en P4.4.2 : **tensor `ple_reference_final` dans `fixtures/ple_fixture.safetensors`** (fp32, vérité math depuis P3) — pas la version bf16 du `.pt` Transformers.
- `fixtures/ple_fixture.safetensors` est gitignored mais régénérable via `scripts/08_export_safetensors_fixture.py` (depuis les `.npy` versionnés).
- Piège RmsNorm : `zml.nn.rmsNorm` est neutre. **NE PAS** réutiliser le wrapper `RmsNorm` de `examples/llm/models/qwen3_5/model.zig` (variante `1+weight`). Suivre le pattern Llama (`examples/llm/models/llama/model.zig:391`) : `normalized.mul(weight.broad(x.shape()))` sans `.add(normalized)`.
- **Piège ZML #1 — reshape perd les tags** : `tensor.reshape(.{1,4,35,256})` retourne un `Tensor({1,4,35,256, f32})` anonyme. Pour cibler un axe par tag (`rmsNorm(x, .d, eps)`, `add` avec un tenseur taggué, etc.), il faut chaîner `.withTags(.{ .b, .s, .l, .d })` immédiatement après le reshape. Observé Gate H, valable pour Gate I et au-delà.
- **Piège ZML #2 — `mul`/`add` ne broadcastent pas implicitement** : `normalized {.b,.s,.l,.d}.mul(weight {.d})` panique `mul expects tensor shapes to match`. Il faut expliciter le broadcast : `weight.broad(normalized.shape())`. Pareil pour `add`. Pattern Llama exact : `normalized.mul(self.weight.convert(x.dtype()).withTags(.{.d}).broad(x.shape()))`. Le `.convert(dtype)` est utile en mixed-precision ; ici tout est fp32, on l'omet.
- Compute : 3090 pour Python (`/data/gemma4-zml-probe`, venv `/data/venvs/gemma4-probe`). ZML est dans `/data/rqz_workspace/zml` sur la 3090. Pour le portage Zig, machine = 3090 (Bazel + accès `examples/`).
- **Oracle = source de vérité, pas hypothèse** : avant de coder un oracle PyTorch d'une couche, LIRE le `forward` de référence dans `modeling_gemma4.py` (scaling, softcap, norms, masque). Ne jamais inférer « pas de poids ⇒ pas d'op » (cf bug v_norm D.0→D.0b). Un oracle qui partage une hypothèse avec le code ZML testé donne un PASS trompeur.

## Mémoire associée

`~/dev/Ma_MEMOIRE/memory/project_gemma4_zml_probe.md` (config invariants, piège ScaledWordEmbedding, pipeline de référence, résultats P3, fixture P4-prep).
