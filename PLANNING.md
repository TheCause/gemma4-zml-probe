# Planning gemma4-zml-probe

> Sonde PLE puis portage ZML de `google/gemma-4-E2B-it`. Roadmap P-1 → P7 (section 10 procédure d'origine).

## Etat 28 mai 2026 (fin de session)

P-1 ✅ · P2 ✅ · P3 ✅ · P4-prep ✅ · P4.3 ✅ · P4.4.0 ✅ · P4.4.1 ✅
P4.4.2 gates : A ✅ B ✅ C ✅ D ✅ E ✅ F ✅ G ✅ H ✅ **I ✅** (A→G bit-exact, H max_diff 1.49e-8, I max_diff 1.49e-8 — l'add n'ajoute pas de drift)
Tag courant : `gate/P4.4.2-gate-I-pass` (commit `52f70a3`).

Gate H a verrouillé la première réduction numérique sensible (variance → rsqrt → mul weight pur Gemma 4, pattern Llama vs Qwen3.5). Gate I a confirmé que la fusion des deux branches PLE (token_identity de Gate D + context_normalized de Gate H) reproduit numpy fp32 à 1 ULP — l'addition element-wise ne dégrade rien au-delà du résidu rsqrt de H. Sur 4 points fixes, blocks B/C/D bit-exact, block A à 1 ULP fp32 strictement hérité de H.

**Next session: Gate J only — `.scale(INV_SQRT_2)` final + comparaison à `ple_reference_final` chargé depuis le fixture.**

Frontière Gate J : dernier scale + premier oracle de bout en bout. Différence avec Gates B→I (référence calculée à la volée en numpy fp32) : la cible est le tenseur `ple_reference_final` déjà matérialisé dans le buffer ZML, produit en P3 via PyTorch fp32 (matmul BLAS). Risque numérique principal : matmul `[4,1536] @ [1536,8960]` peut diverger ~1e-5 entre PJRT CPU (Eigen) et PyTorch (BLAS différent) — c'est ce qu'on a observé en P4.3 (`max_abs = 1.53e-5` numpy vs PyTorch). Donc Gate J ne sera **probablement pas bit-exact** ; tolerance cible 1e-4 (cohérent avec gates précédents), attendu ~1e-5.

### Contrat Gate J (préparé)

```text
input  : sortie Gate I (token_identity + context_normalized)
op     : .scale(INV_SQRT_2) avec INV_SQRT_2 = 0.7071067811865475 (déjà défini ligne 38)
output : ple_final  [b=1, s=4, l=35, d=256]
target : fixture `ple_reference_final` (chargé en buffer, déclaré symboliquement
         dans PleFixture, jamais comparé jusqu'à présent)
mode   : comparaison globale (5 blocs flat) + comparaison tenseur entier
         via toHostBuffer + numpy.allclose (rtol/atol matchés sur diff)
attendu: max_diff ≤ 1e-4, probable 1e-5 (matmul PJRT-CPU vs PyTorch BLAS)
tag    : gate/P4.4.2-gate-J-pass
```

Choix d'implémentation pour Gate J : on a déjà `model.ple_reference_final` dans le buffer device. Deux options :
1. Charger les 35 840 valeurs du fixture en host, faire `max(abs(actual - expected))` côté Zig — simple mais log volumineux.
2. Étendre le forward à renvoyer un tuple (ple_final, ple_reference_final) et comparer en host. Plus propre, mais demande de vérifier la syntaxe ZML pour forward multi-output.

Aller en (1) en première passe — c'est le pattern déjà utilisé pour les blocks A→I, juste appliqué à tout le tenseur en plus des 5 blocs strategiques. Si la moindre divergence dépasse 1e-4, isoler le matmul (Gate E) en faisant tourner deux fois la même chaîne avec/sans Eigen, ou comparer Gate E vs PyTorch directement.

Gate J ferme P4.4.2 et débloque P5 (YOCO).

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
