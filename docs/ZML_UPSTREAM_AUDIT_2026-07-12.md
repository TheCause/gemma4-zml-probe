# Audit upstream ZML — adee932e → origin/master (12 juillet 2026)

> Contexte : question Régis pendant le cadrage batching (« prend-on en compte les mises à jour
> ZML ? on est en retard »). Vendored : `adee932e` (7 avril 2026, M1 `~/dev/zml` == workspace 3090
> présumé). Upstream fetché le 12 juil : **164 commits d'avance** (HEAD `ed2bd190`, 10 juil).
> Audit par 5 lecteurs parallèles (attention, breaking API, sampling, infra, coût du bump).
> **Décision : pas de bump avant ni pendant le chantier batching+sdpa ; bump = chantier dédié
> APRÈS, à ouvrir seulement sur besoin précis** (consigné spec batching §8).

## 1. Les deux verrous de la spec batching tiennent sur origin/master

- **cudnn sdpa toujours commenté** : `nn.zig:1202-1205` (« TODO(Corentin): Re-enable that »,
  `canUseCudnnSdpa` n'existe nulle part). `zml.nn.sdpa` reste une composition dot/softmax-f32/dot,
  sémantiquement identique au vendored → l'A/B sdpa (S1-S3) mesurera la même fusion XLA avec ou
  sans bump.
- **FA2/FA3 assertent toujours B==1** : `flashattn.zig:271` et `:467` (les lignes ont bougé,
  l'assert non). Tout chemin flash **batché** exige un cache paginé `{page,k_chunk,hkv,hd}` —
  incompatible avec le cache YOCO 15 slots `{slot,b,h,k,hd}` sans refonte du layout.

→ **Rien dans les 164 commits ne débloque le design du chantier.** L'attention manuelle composée
et toute la surface Tensor utilisée (dot batch dims, gather+GatherOpts, scatterSlices, topK,
rmsNorm, rope, compileFn) sont **identiques en signature** entre les deux révisions.

## 2. Ce que l'upstream apporte de neuf (pertinent pour NOUS, plus tard)

- **Triton Unified Attention** (post-vendored, `ac9bc32d` 13 avr ; défaut cuda du chemin paginé
  depuis `57d191d0`) : **B>1 natif** (block_table/seq_lens/query_start_len style vLLM varlen),
  **f32 accepté** au niveau du DSL, **scale custom** (le 1.0 de Gemma exprimable, `6ad3e267`),
  **sliding window** et is_causal supportés, heuristique dédiée head_dim ≥ 256, hd=512 non
  interdit mais **non testé upstream**. MAIS cache **paginé obligatoire** → c'est un **3e chantier**
  (« triton paged attention » : bump + migration du cache YOCO vers un layout paginé), pas une
  extension du spike F1.
- **FA2 non-paginé : `window_size_left` configurable** (`89b0908c`, 29 juin) — pertinent pour les
  couches sliding-window en B=1, MAIS contrainte FA2 upstream (fp16/bf16, head_dim ≤ 256, dans le
  `.so` prébuilt) → le lane full hd=512 fp32 du moteur très probablement incompatible, bump ou pas.
  À documenter comme limite attendue du gate F1.
- **Sampling in-graph** : `sampleTokens`/`sampleTokensDynamic`/`Tensor.Rng` existent DÉJÀ dans
  adee932e. Seul delta upstream : bugfix `73498264` (2 juil, 12 lignes — adee932e MULTIPLIE les
  logits par la température au lieu de DIVISER, dans `sampleTokensDynamic` que nous n'utilisons
  pas). Le RNG device n'est pas garanti déterministe entre backends/versions et son état global
  lie le bruit par lane à B → **conforte le choix host-side du §3.5** de la spec. Si besoin un
  jour : cherry-pick de `73498264`, pas un bump.
- RoPE `proportional` natif, `softmaxBiased`, `zml.Compiler`, fixes robustesse pjrt/io.
- **Tokenizer iree patché pour matcher la sortie HF** (`001dbbbd` #623, 2 juil) — changement de
  COMPORTEMENT : les ids produits peuvent différer. Pour un projet « == HF » c'est une correction
  bienvenue, mais après bump : re-mesurer l'EOT, re-valider BOS, **régénérer les fixtures**.

## 3. Coût du bump (quand il sera décidé) — chantier dédié

Churn global : 616 fichiers, +48 778/−21 655. Aucun CHANGELOG upstream — migrations à découvrir
par messages de commit.

**Migration code (~550-600 lignes, ~95 % sed-able, ~49 de nos ~55 runners)** :
- Refonte sharding `0979289f` #514 : `zml.sharding` module → `zml.Sharding` file-struct,
  `replicatedSharding(platform)` supprimé → const `Sharding.replicated`,
  `createTensor(..., null)` devient `@compileError` → `.replicated` (~418 sites),
  `Buffer.scalar` perd le param sharding (15 sites).
- `CreateOptions.cuda` → `.xla_gpu` (`ec776c53`) : gemma4_bench.zig:71, g23_sweep.zig:84,
  gen_auto.zig:902, gen_long_gpu.zig:83.
- `platform.deinit(allocator)` → `deinit(allocator, io)` (~15+ runners).
- Tokenizer : `hftokenizers` supprimé, `encodeAlloc` retourne `[]u32` (plus de `.items`),
  `feed_one`→`feedOne` (~8 lignes dans gen_auto).

**Toolchain** : Zig 0.16.0-dev.2722 → **0.16.0 stable** ; Bazel 8.5.1 → **9.1.1** ;
rules_zig 0.12.2 → 0.15.1 ; plugin pjrt-cuda nightly-2026-03-24 → manual-2026-07-03 (~3 mois de
XLA plus récent). CUDA toolkit INCHANGÉ (13.1.1/cuDNN 9.19.1) — driver 3090 580.159 reste bon.
Rebuild complet du dep-tree sur la VM (lourd).

**Patch local 3090** : `pjrt.zig structSize` toujours SANS `@setEvalBranchQuota` upstream —
le patch 100_000 se ré-applique proprement (contexte identique). Piège noms-courts toujours vivant.

**Preuves invalidées par le bump (à re-produire)** :
- Baselines/custody HLO G2.3 (md5, scripts/52/53) — le nouveau XLA change les dumps.
- Pic VRAM 16 658 MiB → re-mesurer, re-dériver la garde 20 GiB (formule ceil(pic/0.90)+1).
- Gates à rejouer : E1, replay 1020/1020, G1 48/48, A0 tokenizer (CRITIQUE, cf. #623),
  selftest-gather, G2b, garde VRAM. Les 113 tok/s peuvent bouger.
- Numéros de ligne ZML cités dans les docs (spec batching §2, GPU_PORT_PLAN, L3_INGRAPH_DESIGN) —
  annotés « pinned adee932e », à re-pointer.

**Note neutre** : `memory_fraction` défaut upstream 0.85 → **0.90** (`5e4d1b1d`) = notre réglage
explicite actuel, aucun changement de comportement.

## 4. Décision et déclencheurs de réouverture

**Pas de bump avant/pendant le chantier batching+sdpa** (zéro déblocage, bruit de migration +
re-gates sur cible mouvante = la pire fenêtre). Le bump devient pertinent :
1. si on ouvre le chantier « **triton paged attention** » (flash B>1 réel — impose de toute façon
   bump + refonte cache paginé) ;
2. sur besoin précis (fix XLA, feature, sécurité) ;
3. par hygiène, en chantier dédié post-batching avec re-validation complète des gates.

La spec batching consigne ce non-objectif avec renvoi ici.
