# P5.7.7 — Decode incrémental Gemma-4-E2B-it (ZML)

> Établi le 2 juin 2026. Le forward prefill complet (35 couches + YOCO) et les logits sont prouvés
> bout-en-bout (P5.7.5 / P5.7.6, `last_hidden` bit-near ~1e-5, tokens == HF). Le decode n'ajoute
> qu'**une** chose : l'attention ne recalcule plus tout le prompt, elle lit un **cache KV qui grandit
> d'une colonne par token**. Discipline inchangée : oracle PyTorch = vérité, gate atomique,
> 1 nouvelle complexité par gate, fixtures gitignorées, repo local-only.

## Modèle mental — la mécanique neuve du decode

| # | Nouveauté | Idiome ZML (llama/qwen `model.zig`) | Spécificité Gemma4 |
|---|-----------|--------------------------------------|--------------------|
| 1 | **Append** du K/V du nouveau token | `cache.scatterSlices(.{.k = token_index}, new_kv, .override)` | 2 caches (**sliding hd=256** / **full hd=512**), pas un seul |
| 2 | **`pos_idx`** absolu pour la RoPE | `pos = arange(s) + token_index` (decode S=1 → `[p]`) | sliding via `zml.nn.rope(pos)`, full via rope manuelle (cos/sin à pos p) |
| 3 | **Mask incrémental** S=1 | q=1 attend k=0..p | p<512 ⇒ sliding ≡ causal tout-visible (prouvé E.0) |
| 4 | **YOCO en decode** | — | writer 13/14 **append** sur son cache ; readers 15-34 **lisent le cache grandi** du writer |

### Mécanique HF confirmée (`modeling_gemma4.py`, transformers 5.9.0)
- **Cache stocke K post-rope / V post-norm** (attn L1247-50, avant `update` L1251).
- **Producers 0-14** : `past_key_values.update(k, v, layer_idx)` → cache par couche qui grandit (HF =
  `DynamicCache`, sliding géré par **masque**, pas troncation).
- **Writers 13 (sliding) / 14 (full)** (`store_full_length_kv`, L1253-54) : publient leur cache
  **complet et à jour** dans `shared_kv_states[layer_type]`.
- **Readers 15-34** (L1235-36) : lisent `shared_kv_states` — jamais `past_key_values`. `scaling=1.0`.
- **`position_ids` decode** (modèle L1659) = `arange(1) + past_seen_tokens` = `[p]`. cos/sin par
  `layer_type` (L1690). Itération croissante ⇒ writer 13 peuple `shared_kv_states` avant lecture par 15.
- **Découverte (pour decode-2/full)** : `use_alternative_attention = attention_k_eq_v and not is_sliding`
  → sur les couches **full**, V peut = K (`v_proj=None`). Le pilote sliding n'est **pas** concerné.

## Sous-gate **decode-1** (pilote — ce qu'on code en premier)

Plus petite unité qui exerce TOUTE la mécanique pertinente = **writer 13 (sliding, append) × reader 15
(reuse du cache grandi)**. Les deux sont sliding ⇒ `zml.nn.rope` natif gère `pos_idx`, **pas** de rope
manuelle (réservée au full). Le reste de la couche (input_ln, MLP, bloc PLE, `layer_scalar`) est déjà
validé (P5.3) → on **isole l'attention + cache** : on compare la sortie du module `self_attn`
(post `o_proj`), pas la couche entière.

- **Setup** : prompt S=4 (`input_ids = [2, 105, 2048, 4095]`), nouveau token à **position p=4** (=
  `argmax` du logit prefill de la dernière position), `kmax=5`.
- **Oracle** (`scripts/40_p5_7_7_decode_pilot_oracle.py`) : build hybride (streaming, réutilise 39) →
  prefill `use_cache=True, return_shared_kv_states=True` → `argmax` → 1 decode step
  (`past_key_values=pkv`, `return_shared_kv_states=True`) avec **hooks** sur `layers[13].self_attn` et
  `layers[15].self_attn`. `shared_kv_states["sliding_attention"]` donne le cache writer 13 (prefill
  `[0..3]`, decode `[0..4]`) — **pas besoin de fouiller `DynamicCache`**.
  - **Fixture** (`fixtures/p5_7_7_decode1.safetensors`) :
    - `cache13_k_prefill`, `cache13_v_prefill` : `[1,1,5,256]`, positions 0..3 réelles, **col 4 = 0**
      (buffer cache pré-rempli côté ZML, à scatter).
    - `attn_in_13`, `attn_in_15` : `[1,1,1536]` (entrée de `self_attn` = hidden **post** input_layernorm).
    - `cos_sliding`, `sin_sliding` : `[1,1,256]` (pos 4) — référence ; ZML calcule via `zml.nn.rope(pos=4)`.
    - `mask_decode` : `[1,1,1,5]` (sliding, q=1, k=5 ; tout 0 à p<512).
    - **Références** : `cache13_k_after`, `cache13_v_after` `[1,1,5,256]` (post-append) ; `k13_new`,
      `v13_new` `[1,1,1,256]` (token p) ; `attn_out_13`, `attn_out_15` `[1,1,1536]`.
- **Runner** (`zml_runner/gemma4_decode1.zig`) : charge poids couche 13 (q/k/v proj, q/k norm, o_proj) +
  couche 15 (q proj, q norm, o_proj — un reader **n'a pas** de modules k/v). Cache sliding
  `{k=5,h=1,hd=256}` pré-rempli (fixture). Couche 13 : `q_proj/q_norm/rope(pos=4)`,
  `k_proj/k_norm/rope(pos=4)`, `v_proj/v_norm`, **`scatterSlices` à `.k=4`** → cache grandi ;
  attention `q(1)×k(0..4)` + mask → context → `o_proj`. Couche 15 (reader) : `q_proj/q_norm/rope(pos=4)`,
  **réutilise le cache grandi**, attention → `o_proj`.
- **Validation** (tol 1e-4, contrat précision hybride P5.7.5 hérité) : (1) `cache13_*_after == HF`
  (le scatter) ; (2) `k13_new/v13_new == HF` (contenu du token p) ; (3) `attn_out_13 == HF` ;
  (4) `attn_out_15 == HF` (reader sur cache grandi). Fixed-points + scan global. **Revue adversariale**
  du runner (scatter/pos_idx/transpose/mask) avant PASS. Commit + tag `p5.7.7-decode1-pilot-pass`.

**Ce que le pilote prouve** : `scatterSlices` append + `pos_idx` + mask S=1 + reuse reader YOCO sur cache
grandi — *toute* la dynamique decode, sur sliding. Le reste devient de l'extension.

### Micro-gate d'isolation des primitives (dette decode-1 fermée)

La revue adversariale a noté que `scatterSlices` et `zml.nn.rope(pos)` avaient été introduites **ensemble**
dans le composite. On les a isolées rétroactivement (runner autonome `zml_runner/gemma4_decprim.zig`,
oracle déterministe `scripts/41_p5_7_7_decode_prim_oracle.py`, sans modèle) — **3/3 PASS, égalité exacte 0.0** :
- `scatterSlices` à pos=4 **et** pos=2 vs copie numpy → ciblage **dynamique** de la colonne + override +
  passthrough des axes b/h/hd.
- `rope(pos=4) == rope(arange)[4]` bit-exact → **prouve que l'argument `pos` est utilisé** (sinon position 0
  ≠ 4) ; ≡ HF via P5.2.C.3 (rope-arange déjà validé vs `apply_rotary_pos_emb`).
- Le `0.0` est ici correct (ops structurelles / calcul identique, pas un matmul) → pas de drapeau jaune.
- tag `p5.7.7-decprim-isolation-pass`. **decode-3 repose désormais sur des primitives prouvées individuellement.**

## Esquisse de la suite (à reconfirmer après le pilote)

decode-1 (pilote sliding) ✅ → **decode-2** ✅ (couche 14 **full** × reader 19, head_dim 512 + `manualRope`) →
**decode-3** ✅ (MOTEUR e2e 35 couches : 15 caches producers empaquetés en 2 tenseurs multi-slots, dispatch,
YOCO, → norm → lm_head+softcap ; **last_hidden 2.9e-4 + logits 1.0e-4 + argmax token suivant == HF=1018**,
1 process ; tag `p5.7.7-decode3-e2e-pass`) → **P5.7.8** (boucle génération N tokens, reste à faire).
**Le decode est fonctionnellement prouvé : le moteur ZML prédit le même token suivant que HuggingFace.**

- **Précision** : contrat hybride P5.7.5 inchangé (fp32 sauf `embed_tokens_per_layer` bf16).
- **Mémoire** : decode S=1 = working set ~4× plus petit qu'en prefill ; caches négligeables →
  vraisemblablement 1 process sans chaînage (à confirmer au gate 35-couches).
- **Sliding window > 512** : non couvert (prompt court) ; ring-buffer / éviction = dette explicite
  pour les longues générations, à tracer.
