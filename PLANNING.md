# Planning gemma4-zml-probe

> Sonde PLE puis portage ZML de `google/gemma-4-E2B-it`. Roadmap P-1 → P7 (section 10 procédure d'origine).

## État 9-10 juillet 2026 (🏁 portage validé CPU+GPU, G2 fidélité bf16 PASS — PR generation-longue → main)

**Le portage est complet et la claim de fidélité est solide aux deux régimes de précision.**

- **Génération longue validée (28/06, 3090)** sur les DEUX backends : CPU chunké L1a/L1b(ring)/L2
  autonome **1020/1020 == HF** ; GPU CUDA fp32 mono **1020/1020 == HF à 109 tok/s** (~350× vs CPU).
  Non-vacuité du masque prouvée par contre-test LOGITS. Tags `gen-long-validated-3090` +
  `gpu-baseline-validated-3090`.
- **Pipeline end-to-end texte→texte démontré** : `scripts/49_gen_custom_oracle.py` (oracle HF,
  prompt custom via chat template) → `gemma4_gen_long_gpu` (ZML == HF) → `scripts/48_detokenize.py`
  (détok + round-trip gate). Démo « capital of France » → « Paris », 48/48 == HF.
- **G2 fidélité bf16 PASS (4 juil, tags `gate/G2.*`)** — la claim « == HF » n'est PAS un artefact
  fp32. Méthode de l'enveloppe : G2.0 mesure combien HF-bf16 diverge de HF-fp32 (il ne se reproduit
  pas lui-même : 1016/1020, bifurcation step 21) ; G2.2 exige de ZML ≤ 2× cette enveloppe → ZML
  gemm-bf16 est **2 à 5× PLUS fidèle** au fp32 que HF-bf16 (max_abs p50 0.185 vs 0.425, KL 0.28×).
  Doc : `docs/G2_BF16_FIDELITY.md`.
- **Découverte G2.1** : poids DÉJÀ bf16 sur device (dtype du header safetensors), VRAM réelle
  **8,5 Go** (les ~22 Go = réserve BFC `memory_fraction 0.90`). Le « gain VRAM bf16 » du backlog
  n'existait pas ; le banc GPU tiendrait sur une carte 12 Go.
- **E1 rerun PASS 4/4** (`f74b8df`) : neutralité des édits G2 confirmée aux 3 niveaux (source/G1/E1).
- **Cette session (9 juil)** : PR `generation-longue` → `main` + rafraîchissement de ce PLANNING.

### Planning courant

- [x] **PR `generation-longue` → `main`** — mergée le 9 juil (PR #3, `c4d483b`).
- [x] **G2.3 — cartographie de sensibilité bf16 par-op** — **LES 3 GATES PASS le 10 juil**
  (branche `g2.3-op-sensitivity`, tags `gate/G2.3.*`) : moteur `PrecRt` runtime (12 familles),
  sweep one-hot **12/12 SAFE** (classement softcap > norms > mlp > … > softmax), config combinée
  **12 familles SAFE (KL 0.486× l'enveloppe)**, interaction quasi-additive (1.06×), stabilité S49,
  VRAM kv_store −17 MiB (= ½ cache). Oracle anti-câblage-croisé 12/12 exact (2 découvertes :
  déduplication de nœuds au traçage ZML ; dédup inter-familles norms×ple nommée). Résultats :
  `docs/G2_3_OP_SENSITIVITY.md`. → PR vers main à merger.
- [M] **Batching / flash-attention** — perf GPU au-delà du mono-séquence.
- [M] **L3 in-graph** — boucle de décode dans le graphe (réduire les allers-retours host).
- [x] **Runtime 100 % autonome** — **LIVRÉ 10-11 juil 2026** (branche `gen-autonome`,
  spec `docs/GEN_AUTONOME_DESIGN.md`, plan `docs/GEN_AUTONOME_PLAN.md`) : binaire
  `gemma4_gen_auto` texte→texte (tokenizer ZML natif, chat template Zig, prefill-par-decode,
  early-stop EOS, détok stdout). Gates : A0 ids==HF bit-exact ; A1 **48/48 autonome complet**
  (75-94 tok/s) ; A2 critère N/N FAIL publié → requalifié PASS différentiel (autonome ≥
  replay, même bifurcation marge 0.006 au step ~590 — décision Régis) ; A3 early-stop
  « Paris ». Non-régression E1+replay PASS, non-vacuité template PASS. Pièges neufs :
  tokens de tour `<|turn>`/`<turn|>` (EOT=106, lookup `<end_of_turn>`→unk silencieux) ;
  **repli CPU silencieux sans `--@zml//platforms:cuda=true`** (garde dure ajoutée) ;
  tolérance cos/sin ULP linéaire en position.
- [x] **Transfert G2.3 → TurboQuant / alambic** — **FAIT le 10 juil 2026** : notes livrées aux
  deux repos consommateurs (`turboquant/transfert_g23_zml.md` : baseline kv_store bf16, anti-cible
  softcap, banc rejouable ~40 s/run + pointeur CLAUDE.md ;
  `alambic/docs/decisions/2026-07-10-transfert-g23-precision-seuils.md` : régime bf16 student,
  budget bruit teacher ~1e-4 KL p50 dans les seuils, méthode de l'enveloppe — décision Régis
  requise avant le tag `prereg-v1`, consignée au PLANNING alambic). Cross-tasks soldées.

### Garde-fous courants

- **Piège workspace ZML** : patch local 1 ligne `@setEvalBranchQuota(100_000)` dans `pjrt.zig`
  (`structSize`, commenté `local patch rqz`) — **à réappliquer si le workspace ZML de la 3090 est
  resynchronisé upstream**. Requis dès qu'un `@typeName` de type (modèle, runner) devient assez long
  pour dépasser le quota comptime 1000 (`indexOf` sur `@typeName`) — un piège général, pas propre à
  une famille de runners donnée.
- Critère « 1020/1020 » exigible en fp32 seulement — en bf16, HF lui-même ne le tient pas (G2.0) ;
  le critère bf16 est l'enveloppe chiffrée de `docs/G2_BF16_FIDELITY.md` §7.1.
- Leçons méthodo toujours actives : argmax greedy trop robuste (comparer les LOGITS) ; un audit
  multi-agents ne remplace pas le compilateur ; oracle = source de vérité (voir garde-fous
  historiques en bas de fichier).

---

## État 2 juin 2026 (P5.7.5-prep — contrat de précision verrouillé, gate docs-only avant moteur 35 couches)

**Gate courant `P5.7.5-prep` ✅** — décision Régis : **oracle HYBRIDE** (fp32 sauf `embed_tokens_per_layer`
bf16). Contrat figé dans `docs/P5_7_5_precision_contract.md`. Périmètre = docs seulement, **aucun runner**.
- **Pourquoi hybride** : modèle texte full fp32 = 18,51 Go résidents / pic chargement ≈ 27,8 Go > VM 23 Go.
  `embed_tokens_per_layer` `[262144,8960]` = 9,40 Go (50,7 % des params résidents, bf16 sur disque → upcast
  ne récupère rien). Hybride = 13,82 Go ; **bit-identique au full fp32** (gather exact + ×√256=16 puissance
  de 2 exacte), rigueur fp32 préservée sur les 35 couches.
- **Seuils** : PASS `max_abs ≤ 1e-2` **ET** `mean_abs ≤ 1e-4` ; WARN `1e-2 < max_abs ≤ 1e-1` (→ investiguer
  câblage : distribution, localisation par couche, points fixes, suspects YOCO/dispatch/MLP-width) ; FAIL
  `> 1e-1` / NaN-Inf / mismatch shape ou distribution. Drift attendu = matmul Eigen-vs-BLAS accumulé sur
  35 couches (~1e-3..1e-2), concentré (mean_abs petit). Distinguer drift vs bug = §6 du contrat.
- **Dette pour P5.7.5 (phase moteur)** : `scripts/38_p5_7_5_prefill_oracle.py` est **full bf16** → à régénérer
  en hybride ; fixture `p5_7_5_prefill.safetensors` **périmé** ; `expected_zml_max_abs_le` du manifest
  (2e-3) à aligner sur le PASS contractuel (1e-2). Options 2 (moteur ZML bf16) / 3 (tol 1e-1 critère premier)
  rejetées. **Interdit** : démarrer le moteur 35 couches tant que ce contrat n'est pas committé (✅ fait).

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
**P5.2.H ✅** ZML MLP feed-forward (layer 15) — `out = residual + post_ff_norm(down(gelu(gate(x))*up(x)))`, `x=pre_ff_norm(residual)`. **Découverte : layer 15 (reader, KV-shared) + use_double_wide_mlp → intermediate=12288** (pas 6144 ; layers 0-14=6144, 15-34=12288). `gelu_pytorch_tanh` = `Tensor.gelu` ZML (confirmé 0.0). gating `gelu(gate)*up` (pas gelu(gate*up)). Oracle = modules réels Gemma4RMSNorm+ACT2FN. Forward `[1,4,1536]`, max|out|=103.6, 2 fixed-points 1.43e-5, scan global **max_abs 5.34e-5** (marge ~1.9×, réduction .f=12288). Runner `gemma4_mlp.zig`, script `28_`. — tag `p5.2-h-mlp-zml-pass`
**P5.6 ✅** ZML full_attention Q-rope manuelle partielle (layer 14) — **RISQUE DU PROJET LEVÉ**. Découverte : full attn = **head_dim 512** (q_proj [4096,1536], pas 256), partial_rotary 0.25 (128/512 dims tournent), theta=1e6, scaling=1.0. `zml.nn.rope` ne couvre pas proportional → **RoPE manuelle** : cos/sin oracle (512-wide, 384 identité), `rotate_half` via `split(.hd,{256,256})`/`negate`/`concatenate`, `q*cos+rh*sin`. Sanity oracle : manuelle == apply_rotary_pos_emb à 0.0. Scan global **max_abs 7.99e-6** (marge ~12×). Runner `gemma4_full_qrope.zig`, script `29_`. Le reste du chemin full = identique sliding (E/F) avec dims 512. — tag `p5.6-full-qrope-zml-pass`
**P5.4 ✅** ZML embedding gather + scale √1536 (slice vocab 4096) — `weight.gather(.{.voc=ids}).scale(√1536)`. Op gather validée (P4.4 avait validé le scale, pas le gather). **bit-exact** (max_abs 0.0). Runner `gemma4_embed.zig`, script `30_`. — tag `p5.4-embed-zml-pass`
**P5.5 ✅** ZML head (final norm + lm_head tied + softcap) — `rmsNorm(hidden).mul(norm_w)` → `dot(lm_head_slice,.d)` → `scale(1/30).tanh().scale(30)`. Op neuve softcap (Tensor.tanh). lm_head tied=embed_tokens, slice vocab 4096. Forward `[1,4,4096]`, max|logits|=29.6 (borné ±30), 2 fixed-points 1.86e-5, scan **max_abs 5.44e-5**. Runner `gemma4_head.zig`, script `31_`. — tag `p5.5-head-zml-pass`
**P5.3 ✅** ZML **couche décodeur sliding COMPLÈTE** (layer 13 producer) vs module réel `Gemma4TextDecoderLayer` — compose input_ln + attention(QKV/norm/rope/QK/mask/softmax/context/o_proj) + résiduel + MLP(6144) + résiduel + bloc PLE per-layer (gate/proj/norm + per_layer_input) + layer_scalar(0.0884). Gestion tags (rename .s→.q/.k, GQA splits) OK du 1er essai. Forward `[1,4,1536]`, 2 fixed-points 6.68e-6, scan **max_abs 6.72e-5** (marge ~7.4×). Runner `gemma4_layer.zig`, script `32_`. — tag `p5.3-layer-zml-pass`
**P5.6.K ✅** ZML full_attention K-rope manuelle (layer 14) — ferme le gap K-full-rope de l'audit closeout (même technique que P5.6 sur K). scan 2.68e-7. tag `p5.6k-full-krope-zml-pass`.
**P5.6.closeout ✅ (2 juin)** — audit complétude (workflow 3 agents) : **matrice composant→runner/tag/preuve/tolérance** (`docs/P5_6_closeout.md`), 0 gap, tolérances ALL_JUSTIFIED, 0 faux invariant, tags superseded documentés. **Base saine pour P5.7.**
**P5.7.0 ✅** loader manifest only — `scripts/34_p5_7_0_loader_manifest.py` produit `fixtures/p5_7_0_loader_manifest.json` sans compute ni chargement payload. Résumé : 600 clés disque attendues, 540 tenseurs runtime à charger, 60 clés K/V reader disk-only ignorées par YOCO, `v_norm` documenté comme op sans poids (`with_scale=False`). Validation checkpoint optionnelle via `--require-weights`; sautée localement car `weights/model.safetensors` absent.
**P5.7.0 ✅ (Codex/ChatGPT, vérifié Claude 2 juin)** — loader manifest 35 couches (`scripts/34_p5_7_0_loader_manifest.py`, `fixtures/p5_7_0_loader_manifest.json`). **Vérification Claude contre le checkpoint réel sur 3090** (ce que Codex avait sauté) : `--require-weights` PASS, **600 clés / 0 manquante / 0 mismatch** ; reverse-check **600 clés `model.language_model.*` = 600 attendues, 0 oubli, 0 extra** (1411 autres = tours vision/audio hors scope). Shapes archi correctes (full head_dim 512, MLP double-wide 12288 readers, YOCO producers 0-14 / readers 15-34 → target 13 sliding / 14 full, K/V readers disque-only ignorés). Manifest committé mis à jour SKIPPED→PASS.
Tag courant : `p5.6k-full-krope-zml-pass`. **🎯 COUCHE DÉCODEUR E2E + TOUTES OPS DISTINCTES VALIDÉES + AUDIT CLOSEOUT + P5.7.0 manifest.** Reste = P5.7 runtime (P5.7.1→.8). Idéal contexte frais. **🎯 TOUTES LES OPS DISTINCTES DU FORWARD VALIDÉES EN ZML.** Reste = intégration (35 couches + KV cache + PLE). Voir ROADMAP. **MODE AUTONOME** (re-priorisé par risque) : ~~H~~ ~~P5.6~~ faits → **P5.4 (embedding+scale)** → **P5.5 (final norm+lm_head+softcap)** → P5.3 (assemblage couche e2e) → P5.7 (multi-couches). Après P5.4/5.5 : toutes ops distinctes validées. Plan = `docs/ROADMAP_to_full_forward.md`.

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

## Planning historique (état 31 mai — tout est clos depuis, voir « Planning courant » en tête)

### Haute priorité

- [x] **P4.4.2 — Mini-runner ZML PLE-only**
  - Charger `fixtures/ple_fixture.safetensors` via `zml.safetensors.TensorRegistry`.
  - Reproduire le pipeline PLE :
    `lookup × √1536`, `lookup × √256`, projection `× 1/√1536`, reshape, RMSNorm Gemma 4 pure `* weight` ε=1e-6, fusion `/√2`.
  - Comparer à `ple_reference_final` (chargé depuis le même safetensors).
  - Gate fp32 : `max_abs ≤ 1e-4` + fixed point `[0,0,0,:4]` aligné.
  - Sans YOCO, sans attention, sans KV-cache.

### Medium

- [x] **P5 — Shared KV / YOCO** (`num_kv_shared_layers=20`) : inspecter forward Transformers, tracer shapes et cache lifecycle.
- [x] **P6 — Attention hybride** : pattern `layer_types` 4×sliding + 1×full × 7 (full aux couches 4, 9, 14, 19, 24, 29, 34), p-RoPE.

### Backlog

- [x] **P7 — Logits** : `final_logit_softcapping = 30.0`, top-k overlap, flip-rate temp=0.
- [B] Intégrer `05` et `06` dans `04_run_all.sh`.
- [x] Tester d'autres `input_ids` que `'ZML test prompt'` (fait : gén longue 1020 tokens + prompts custom script 49).

## Garde-fous (historiques — pièges ZML/Gemma4 toujours valides)

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

