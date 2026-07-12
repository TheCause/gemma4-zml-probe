# Batching statique + variante sdpa — résultats des gates

> Spec : `docs/superpowers/specs/2026-07-12-batching-flash-attn-design.md`
> Plan : `docs/superpowers/plans/2026-07-12-batching-flash-attn.md`
> Machine : VM 3090 (24 GiB, driver 580.159.03), ZML `adee932e`, fp32.
> Convention du repo : la doc porte les résultats (logs/ gitignorés) ; FAIL/null publiés comme les PASS.

## Phase 1 — Batching

| Gate | État | Verdict | Chiffres |
|---|---|---|---|
| T0 — neutralité du B shape-polymorphe | ✅ fait (12 juil) | **PASS** | md5 `before_optimizations` identique |
| B1 — selftest primitives batchées | ✅ fait (12 juil) | **PASS** | 4/4 sous-tests exacts, B=2 |
| B2 — fidélité par lane (B=2, B=4) | ⏳ | | |
| B3 — indépendance inter-lanes | ⏳ | | |
| B4 — sweep B → plafond VRAM | ⏳ | | |

## Phase 2 — sdpa

| Gate | État | Verdict | Chiffres |
|---|---|---|---|
| S1 — neutralité `attn=.manual` | ⏳ | | |
| S2 — fidélité sdpa | ⏳ | | |
| S3 — A/B perf sdpa vs manual | ⏳ | | |
| F1 — spike FA2 (optionnel) | ⏳ | | |

---

## T0 — neutralité du B shape-polymorphe — **PASS** (12 juillet 2026)

**Changement prouvé neutre** : les 5 sites de reshape d'`engine.zig` (q :395, k :412, v :417,
PLE token_identity :536, PLE context :539) ne consomment plus les constantes `B`/`S` mais
dérivent leurs dims des **shapes d'entrée** (`h0.dim(.b)`, `h0.dim(.s)`, `embptl_slice.dim(.*)`,
`embeds.dim(.*)`). Le moteur devient shape-polymorphe → **un binaire unique sert tous les B**
(custody G2.3 build-unique conservée, `@typeName` inchangé, quota comptime non touché).

**Méthode** (gold G2.3.0 — jamais le post-opt, piège 15) : dump `XLA_FLAGS=--xla_dump_to` du
**témoin `gemma4_gen_auto`** (entrées symboliques à b=1/s=1 littéraux → les 5 sites sont exercés
via `forwardStep` + `perLayerInputs`), déployé et **rebuildé** avant/après la modification.

| | md5 `module_0001.zml.before_optimizations.txt` |
|---|---|
| BEFORE (HEAD inchangé) | `ac9df2ae66da8aba65a0a606bf5947ec` |
| AFTER (5 sites convertis) | `ac9df2ae66da8aba65a0a606bf5947ec` |
| **Verdict** | **byte-identique → PASS** (repli §6 T0 non nécessaire) |

Dump : 308 fichiers. Sanity du run : mêmes tokens générés des deux côtés
(`generated = { 9259, 236888 }` → « Hello! »), backend cuda confirmé.

**Infra capturée (T0 infra, `fixtures/batch_manifest.json`)** : rev ZML 3090
`adee932e` == miroir M1 (la présomption de la spec est **confirmée**) ; patch quota comptime
présent à **`pjrt/pjrt.zig:26`** (`@setEvalBranchQuota(100_000)`) — NB le chemin réel est
`pjrt/pjrt.zig`, pas `zml/pjrt.zig` ; GPU vierge au moment de la capture (0 / 24 576 MiB).

**Chemins réels sur la VM** (non documentés jusqu'ici, à réutiliser pour tous les runs) :
- poids : `/data/gemma4-zml-probe/weights/model.safetensors`
- tokenizer : `/data/gemma4-zml-probe/gemma4-e2b-it-meta/tokenizer.json`
- ⚠ le checkpoint du cache HF (`/data/hf_cache/hub/models--google--gemma-4-E2B-it/snapshots/…`)
  est un **lien symbolique vers `blobs/`** → `error.InvalidPath` avec ZML `TensorRegistry.fromPath`.

---

## B1 — selftest des primitives batchées — **PASS 4/4** (12 juillet 2026)

`gemma4_bbatch --selftest-batch <fixture>` : mini-graphes compilés (pattern SgFwd), B=2, valeurs
exactes vs référence host. **Le runner a compilé du premier coup** sur la 3090 (14,5 s) — le
gather rank-2 `{B,1}` passe sans le repli 1-D documenté, et le layout D2H du topK est bien `{b,K}`.

| Sous-test | Verdict | Ce qui est prouvé |
|---|---|---|
| 1 — `gather` batché | **PASS** | 2 lanes × 2 tables (embed_tokens + eptl) **bit-exact u16** vs gather 1-lane ET vs la fixture (tok = {50429, 106}) |
| 2 — `scatterSlices` batché | **PASS** | **Jamais exercé dans ce repo** : `pos_u` scalaire PARTAGÉ, update `{b=2,…}` → lane0=1.0 et lane1=2.0 écrites chacune à `k=3`, zéros ailleurs. La sémantique « dim `.b` non indexée = fenêtre » est confirmée. |
| 3 — `topK` batché | **PASS** | Layout D2H `{b=2, K=5}` (stride 5 par lane), indices i32, argmax distincts par lane. Valeurs choisies sans tie possible (`v[l][j] = (7j+3l) % 16`, 7 premier avec 16 ⇒ permutation) → attendu calculable host, indépendant de la politique de tie-break XLA. |
| 4 — `broad` rank-égal | **PASS** | Le masque `{b=1,h=1,q=1,k}` est bien diffusé aux 2 lanes. **L'invariant fragile de la spec §2.8 est prouvé, pas supposé** : à rank égal `broad` diffuse par POSITIONS (pas par tags) et l'ordre des axes coïncide ici. Les tables Packed peuvent donc rester à b=1 (jamais ×B en VRAM). |

VRAM libre au lancement : 23,8 GiB — **aucun seuil appliqué** (le plafond est l'output du banc,
la garde ne mord que sur contention).
