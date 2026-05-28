# Planning gemma4-zml-probe

> Sonde PLE puis portage ZML de `google/gemma-4-E2B-it`. Roadmap P-1 → P7 (section 10 procédure d'origine).

## Etat 28 mai 2026 (fin de session)

P-1 ✅ · P2 ✅ · P3 ✅ · P4-prep ✅ · P4.3 ✅ · P4.4.0 ✅ · P4.4.1 ✅
P4.4.2 gates : A ✅ B ✅ C ✅ D ✅ E ✅ F ✅ G ✅ **H ✅** (A→G bit-exact PJRT CPU vs numpy fp32, H max_diff 1.49e-8 ≪ tol 1e-4)
Tag courant : `gate/P4.4.2-gate-H-pass` (commit `a716838`).

Gate H prouve que la première réduction numérique sensible (variance → rsqrt → mul weight pur Gemma 4, pattern Llama) reproduit la référence numpy fp32 à 1 ULP. Choix Llama vs Qwen3.5 verrouillé : `normalized.mul(weight)`, jamais `normalized.mul(1+weight)`.

**Next session: Gate I only — pure add (token_identity + context_normalized), no /√2.**

Frontière Gate I : première fusion des deux branches PLE validées indépendamment (Gate D pour token_identity, Gate H pour context_normalized). Op mathématiquement triviale (`add` element-wise), mais sémantiquement le premier point où le pipeline réunit ses deux flux. À traiter seul. `/√2` final + comparaison à `ple_reference_final` reportés en Gate J.

### Contrat Gate I (préparé)

```text
input A : token_identity      [b=1, s=4, l=35, d=256]  ← branche Gate D (déjà ×√D=×16)
input B : context_normalized  [b=1, s=4, l=35, d=256]  ← branche Gate H (rmsNorm * weight)
op      : A.add(B)
output  : ple_sum             [b=1, s=4, l=35, d=256]
target  : numpy fp32 (token_identity + context_normalized) sur 4 points fixes A/B/C/D
attendu : bit-exact ou ~1 ULP fp32 (max_diff ≤ 1e-7)
tag     : gate/P4.4.2-gate-I-pass
```

Subtilité Gate I : la branche Gate D (`embed_tokens_per_layer_slice.scale(SQRT_D).reshape(.{1,4,35,256})`) produit shape anonyme → `.withTags(.{ .b, .s, .l, .d })` requis avant `add` (sinon shape mismatch comme on l'a eu sur Gate H pour `mul`).

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
