# Planning gemma4-zml-probe

> Sonde PLE puis portage ZML de `google/gemma-4-E2B-it`. Roadmap P-1 → P7 (section 10 procédure d'origine).

## Etat 28 mai 2026 (P4.4.2 PLE-only CLOSED)

P-1 ✅ · P2 ✅ · P3 ✅ · P4-prep ✅ · P4.3 ✅ · P4.4.0 ✅ · P4.4.1 ✅
**P4.4.2 ✅** gates : A ✅ B ✅ C ✅ D ✅ E ✅ F ✅ G ✅ H ✅ I ✅ **J ✅**
Tag courant : `gate/P4.4.2-gate-J-pass`.

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

## Mémoire associée

`~/dev/Ma_MEMOIRE/memory/project_gemma4_zml_probe.md` (config invariants, piège ScaledWordEmbedding, pipeline de référence, résultats P3, fixture P4-prep).
