# Batching statique + variante d'attention sdpa — design

> **Statut** : spec validée par Régis le 12 juillet 2026 (cadrage par questions + choix explicites).
> **Décisions de cadrage (Régis)** : usage cible = **banc multi-prompts (mesure/recherche)** → batch
> **statique** ; le **plafond VRAM est un output du banc**, pas un input ; **greedy par défaut**
> (oracle == HF conservé), temperature/top-k **en option host-side hors gates** ; flash-attention
> **dans le même cycle, après le batching** (variante A/B mesurée par le banc) ; runner **autonome**
> (squelette gen_auto) ; critère de fidélité = **argmax 48/48 par lane + requalification
> différentielle pré-enregistrée**.
> Base : L3 in-graph mergé main (PR #7, tag `gate/l3-ingraph-pass`). Second sous-projet du backlog
> (`PLANNING.md:37`, seul item non-[x]).

## 1. But

Étendre le moteur decode au **batch statique B>1** pour instrumenter la scalabilité du runtime
autonome sur la 3090 : tok/s (agrégé et par lane) et pic VRAM en fonction de B, **plafond B
rapporté par le banc**. Ensuite, dans le même cycle, introduire une **variante d'attention
comptime `sdpa`** (`zml.nn.sdpa`) mesurée A/B par ce banc (fidélité + perf).

Doctrine perf du repo inchangée : **fidélité obligatoire, perf mesurée sans objectif dur**
(non-régression différentielle seulement) — décision de cadrage L3 reprise telle quelle.

## 2. Découvertes de lecture qui cadrent le design

Faits vérifiés dans les sources (12 juil 2026, checkout ZML M1 `adee932e`) :

1. **`B` n'apparaît qu'à 5 sites** dans `engine.zig` : reshapes q/k/v (`engine.zig:395,412,417`)
   et les 2 reshapes PLE (`engine.zig:536,539`). Tous les tenseurs (Cache `{slot,b,h,k,hd}`,
   Packed `{step,b,...}`, transposes, logits, topK) portent déjà le tag `.b`. L'axe paramétrique
   est un changement de dimension, pas de plomberie.
2. **Le verrou réel est la position partagée** : `ctrl.step` scalaire unique (`engine.zig:302-311`)
   pilote `pickStep` (une ligne de table Packed pour tout le batch) ET `pos_u` scalaire est
   l'index `.k` du `scatterSlices` du cache pour **toutes** les lanes (`engine.zig:433-444`).
   → V1 = prompts de **même longueur tokenisée** obligatoire. Longueurs hétérogènes
   (pos `[b]`, masques par lane, padding) = changement invasif, **hors périmètre**.
3. **Le chemin cudnn de `sdpa` est du code mort** (`zml/nn.zig:1085-1088`, bloc commenté
   « TODO(Corentin): Re-enable that », `canUseCudnnSdpa` n'existe nulle part) : `zml.nn.sdpa`
   dans ce vendored = composition dot/softmax-f32/dot. L'A/B sdpa mesurera la **fusion XLA**,
   pas un kernel flash.
4. **Des custom calls flash-attention réels existent et sont actifs sur cuda**
   (`zml/attention/flashattn.zig`, auto-enregistrés `platform.zig:250-259` ; 3090 CC 8.6 → FA2)
   MAIS **`fa2.attention` asserte B==1** (`flashattn.zig:280`, idem fa3 :476). Le seul chemin FA
   batché est la paged attention, au layout cache `{page,k_chunk,...}` incompatible avec les
   15 slots YOCO. → « flash + batching » simultané **infaisable sur ce vendored ZML** :
   le design séquence, FA2 réel = spike B=1 optionnel non bloquant (gate F1).
5. **La révision ZML de la 3090 n'est pinée nulle part** (deploy = rsync de `zml_runner/` seul ;
   présomption forte `/data/rqz_workspace/zml == adee932e` — les numéros de ligne cités dans
   les commentaires du code livré matchent exactement). À capturer en gate T0, avec le rappel
   du **patch local pjrt.zig** `@setEvalBranchQuota(100_000)` à réappliquer si resync.
6. `sdpa` **accepte un tag `.b` surnuméraire** : `hasTags` = présence seule, `collectDims` ne
   visite que les tags listés (le mode `.strict` n'exclut pas les tags en plus,
   `nn.zig:1687,1697`), et `Tensor.dot` promeut tout tag commun non contracté en batch dim
   (`tensor.zig:1139-1155`). Une seule variante sdpa sert B=1 et B>1.
7. `sdpa` applique par défaut un **scaling 1/√hd sur K** (`nn.zig:1099-1101`) alors que Gemma 4
   n'a **aucune op de scaling** au dot QK (`engine.zig:460`, scaling 1.0, la norme passe par
   q_norm — piège déjà documenté P5.2 : « sinon scores divisés par 16 »).
   → `opts.scale = Tensor.scalar(1.0)` obligatoire.
8. **Piège `broad`** : quand rank source == rank cible, broadcast **par positions** et non par
   tags (`tensor.zig:2183-2195`). Règle de design : masques et tables restent **sans axe `.b`**
   (ranks différents → chemin par tags).

## 3. Architecture

### 3.1 Engine — axe `.b` comptime (extension `EngineCfg`)

```zig
pub const EngineCfg = struct {
    ring: bool = false,
    two_masks: bool = false,
    kmax_sliding: i64 = 8,
    kmax_full: i64 = 8,
    b: i64 = 1,                            // NOUVEAU — axe batch
    attn: enum { manual, sdpa } = .manual, // NOUVEAU — variante d'attention (§3.4)
};
```

- Les 5 reshapes consomment `cfg.b` au lieu de la constante `B` (qui reste `pub` pour les
  runners legacy, ou est aliasée — arbitrage cosmétique au plan).
- **Contrat de neutralité** (pattern `ring`/`two_masks`, `engine.zig:318-323`) : les défauts
  `b=1, attn=.manual` doivent **élider comptime** toute op nouvelle — l'exigence est l'élision
  comptime, PAS l'identité numérique (une op émise mais neutre casse le diff,
  `GENERATION_LONGUE_DESIGN.md:41`).
- **Ordre d'émission intouché** : `rmsScaleDPrec` émet `wi` avant le rmsNorm,
  `rmsScaleHdPrec` l'inverse (`engine.zig:86-98`) — toute harmonisation casserait le byte-diff.
- **Quota comptime** : ajouter des champs à `EngineCfg` allonge `@typeName` (tension avec la
  leçon G2.3 qui avait sorti la précision du type). Mitigations connues : noms courts,
  patch pjrt.zig 100_000 (3090), build 3090 tôt dans le plan (leçon L3 [it.5]).

### 3.2 Runner `gemma4_bbatch.zig` (nouveau, autonome)

Squelette gen_auto généralisé par lane. Nom court (13c < plafond ~20c du quota comptime).

- **CLI** : `<model.safetensors> <tokenizer.json>` + `--prompts <fichier>` (un prompt par ligne)
  OU `--oracles <f1,f2,...>` (N fixtures 49) ; `--max-tokens`, `--force-vram`, `--allow-cpu`,
  `--no-prealloc` (mesure VRAM, pattern g23_sweep), `--temperature/--top-k/--seed` (mode charge,
  §3.5). B = nombre de prompts/fixtures fournis (comptime : binaire compilé par valeur de B via
  paramètre build, ou table de B supportés — arbitrage au plan, cf. risque compile §7).
- **Contrainte V1** : tous les prompts à la **même longueur tokenisée** — sinon
  `error.PromptLengthMismatch`. Gardes existantes par lane : `ids.len < SLIDING_WINDOW=512`,
  `ids.len + limit ≤ L_MAX=1024`.
- **StepTok batché** : `tok_sym {B,1}` u32 tags `{b,s}` ; gather embed_tokens + eptl inchangés
  (le gather rank-2 à B>1 est à re-valider : P5.4 n'a validé que des ids 1-D, repli
  reshape/re-tag documenté `gen_auto.zig:938-942`) ; `forwardStep` intact ; topK →
  `{b, K}` (layout D2H à valider au premier compile — gate B1).
  **Cross-ref SG↔StepTok** : si le dtype/shape des indices de gather change, le selftest gather
  doit changer À L'IDENTIQUE (piège documenté `gen_auto.zig:941-942`).
- **Boucle host** : feed d'un vecteur `[B]` u32 par step (bounds-check **par lane** — le gather
  XLA clampe silencieusement les indices hors-borne) ; prefill-par-decode partagé (même
  longueur ⇒ frontière prefill/gén commune) ; **early-stop EOT par lane** : lane finie →
  réinjection d'un token de bourrage (EOT=106, mesuré du tokenizer, borné vocab) + sortie
  masquée host + `StopReason` par lane ; le batch court jusqu'à la dernière lane active ou
  `--max-tokens`. Pas de sortie anticipée du graphe.
- **Mesure** : convention s0 de L3 reprise à l'identique (s0 produit par le dernier call de
  prefill, compté dans generated) pour rester comparable aux 110-113 tok/s B=1 ;
  `t_prefill_end` unique (frontière commune) ; tok/s **agrégé** = Σ tokens générés (lanes
  actives) / gen_s, et tok/s **par lane** ; prefill mesuré séparément.
- **Garde VRAM** : la garde 20 GiB de gen_auto est calibrée B=1 et reste **intouchée**.
  bbatch porte une **garde de contention** (refus si d'autres process compute occupent le GPU,
  pattern checkVram, échappatoire `--force-vram`) — pas de seuil fixe : le plafond est ce que
  le banc mesure.

### 3.3 Banc/sweep (script, pattern `g2_3_sweep.sh`)

- Balayage B ∈ {1, 2, 4, 8, 16, …} (liste bornée, pas linéaire — chaque B = un graphe compilé,
  compile ~17 s + risque RAM compile).
- Chaque point : build (sha256 du binaire consigné), run oracle (fidélité) puis run mesure
  `--no-prealloc` + échantillonnage `nvidia-smi --query-compute-apps` → pic(B)
  (méthode G3 — piège 14 : `mem_probe` = RSS host ; sous prealloc, nvidia-smi ne montre que la
  réserve BFC 0.90×libre).
- **Arrêt propre par projection** : pic(2B) projeté ≈ pic(B) + B×Δ (Δ ≈ 38 Mo cache/lane
  mesuré + marge activations) ; si projection > ~22 GiB → stop avant OOM. On ne compte PAS
  sur un échec propre : l'error-path OOM upstream crashe en General protection exception
  (`io.zig` deinit, bug connu `PLANNING.md:70-77`).
- Manifest JSON de custody : sha256 binaire, `zml_rev` (capturé T0), driver, date, table
  B → {tok/s agrégé, tok/s par lane, pic MiB, verdict fidélité}.
- **Avant tout run** : `nvidia-smi --query-compute-apps` (contention Ollama = piège
  opérationnel n°1) ; `deploy_to_3090.sh` rate silencieusement sans `ZML_REMOTE`/`ZML_DST` ;
  flag `--@zml//platforms:cuda=true` obligatoire (repli CPU silencieux sinon).

### 3.4 Variante d'attention `sdpa` (après le batching)

Branche comptime au site scores/softmax/PV (`engine.zig:459-467`), pattern `two_masks` :

- `.manual` (défaut) : chemin actuel, byte-identique (gate S1).
- `.sdpa` : `zml.nn.sdpa(q_final, cache_k, cache_v, .{ .attn_mask = mask, .scale = un })`
  avec `un = Tensor.scalar(1.0, dtype)` (**obligatoire**, découverte n°7) ; masques additifs
  existants `{q,k}` **sans axe `.b`** passés tels quels (broad par tags, découverte n°8) ;
  sdpa fait le même splitAxis GQA que l'engine (`nn.zig:1093` == `engine.zig:459`) et la même
  convention softmax f32 ; head_dim hétérogène (sliding 256 / full 512) : sdpa est appelé
  par couche, transparent.
- **Pas de byte-identité attendue** pour `.sdpa` (scale émis sur K même à 1.0 pré-opt, ordre
  transpose/merge différent) : c'est un **A/B mesuré et oraclé**, pas prouvé.
- **Interaction PrecRt** : en mode `.sdpa`, les familles `qk_scores`/`softmax`/`pv_ctx` ne
  s'appliquent plus (sdpa fait ses propres converts). Les gates sdpa tournent en **fp32 pur**
  (prec par défaut) ; combiner sdpa × familles bf16 G2.3 = hors périmètre.

### 3.5 Sampling optionnel (mode charge, hors gates)

- `topK` in-graph passe à K comptime paramétrable dans bbatch (défaut 64 ; gen_auto reste K=5,
  aucune contrainte d'identité entre les deux binaires — les gates de bbatch sont vs fixtures HF).
- `--temperature T` / `--top-k k ≤ K` / `--seed s` : softmax host sur les K logits rapatriés
  par lane, RNG host seedé par (seed, lane, step) — **zéro RNG device**, D2H ≈ K×8 octets/lane/step.
- **Troncature assumée et documentée** : échantillonnage restreint au top-K rapatrié
  (approximation de charge, pas une implémentation de sampling de référence).
- Greedy (`indices[0]`) reste le seul mode gaté ; aucun oracle en mode sampling.

## 4. Oracle et fidélité

- **N invocations de `scripts/49_gen_custom_oracle.py`** (une fixture safetensors + manifest
  par prompt — mécanisme existant, zéro format nouveau ; wrapper shell optionnel pour générer
  le lot). Prompts du banc choisis à même longueur tokenisée (vérifiable via `prompt_ids` du
  manifest sidecar).
- **Référence = HF mono B=1 par séquence** (jamais un run HF batché : le padding HF changerait
  la numérique de la référence elle-même).
- **Vigilance pré-enregistrée** : changer B change les shapes GEMM → kernels/ordres de
  réduction XLA différents → logits non bit-identiques au mono. Une lane peut bifurquer
  d'argmax **sans bug** (précédents : HF ne se reproduit pas lui-même 1016/1020 ; ties à marge
  0,006 ; non-déterminisme inter-compiles, piège 15).
- **Procédure d'échec pré-enregistrée** (pattern A2) : au premier mismatch d'une lane,
  diagnostic top-5 du step fautif (marge top1−top2) ; si la marge est fine (tie), le FAIL brut
  est **publié** puis requalifié en critère différentiel documenté ; sinon c'est un vrai FAIL.

## 5. Baselines (chiffres de référence, L3 mergé)

| Mesure | Valeur | Source |
|---|---|---|
| Génération B=1 (L3) | 110-113 tok/s ≥ replay 109 ≥ B0 91,4 | `L3_INGRAPH_DESIGN.md:111-120` |
| Prefill-par-decode | ~71,5 tok/s (mesuré séparément) | `DOCUMENTATION.md:106-107` |
| Compile StepTok GPU | ~17 s | `L3_INGRAPH_DESIGN.md:113` |
| Pic VRAM B=1 | 16 658 MiB ≈ 16,27 GiB (--no-prealloc) | commit `75ee030` |
| Garde gen_auto | 20 GiB = ceil(16,27/0,90)+1 (B=1, intouchée) | `gen_auto.zig:679` |
| Coût marginal/lane | ~38 Mo cache f32 (kmax 1024) + ~1 Mo logits | calcul §2 + mesure G2.3.2 |
| Bifurcation longue G2b | 960 ≥ replay 590 (non-déterminisme inter-compiles documenté) | `L3_INGRAPH_DESIGN.md:116` |
| Bruit perf inter-compiles | ~2 % (grand effet) à ~16 % (petit) | `DOCUMENTATION.md:464-466` |

## 6. Validation — gates (un commit/tag par gate, `gate/batch-*`)

| Gate | Contenu | Critère PASS |
|---|---|---|
| **T0** | Neutralité axe `.b` + `attn` (défauts) ; capture `git rev-parse` ZML 3090 | md5 `module_0001.zml.before_optimizations.txt` **identique** sur gemma4_gen_auto recompilé (méthode gold G2.3.0 — jamais le post-opt, piège 15) ; rev consignée au manifest |
| **B1** | Spike primitives batchées sur 3090 : scatterSlices update `.b>1` au même pos (jamais exercé), gather tok `{B,1}`, topK `{b,K}` layout D2H | mini-graphe type SgFwd, valeurs exactes vs référence host, B=2 |
| **B2** | Fidélité batch B=2 puis B=4 | chaque lane **48/48 == sa fixture 49** (fp32) ; procédure d'échec §4 ; **non-vacuité** : une lane corrompue (fed altéré) → FAIL obligatoire |
| **B3** | Indépendance inter-lanes | B lanes, même prompt → sorties **identiques entre lanes** (test de contamination) |
| **B4** | Sweep B ∈ {1,2,4,8,…} → plafond | table B → {tok/s agrégé, par lane, pic VRAM --no-prealloc} publiée ; plafond B rapporté ; **non-régression différentielle** : bbatch B=1 ≥ gen_auto L3 en tok/s gén (multi-runs, bruit budgété) |
| **S1** | Neutralité `attn=.manual` | md5 before_opt identique (même méthode que T0) |
| **S2** | Fidélité sdpa, B=1 et B=4 | 48/48 par lane vs mêmes fixtures ; procédure d'échec §4 |
| **S3** | Perf A/B sdpa vs manual | mesure multi-runs aux mêmes B ; verdict différentiel publié (gain OU absence de gain — le cudnn étant mort, l'A/B mesure la fusion XLA : conclusion honnête exigée) |
| **F1** *(spike optionnel, non bloquant)* | FA2 custom call à B=1 | mesure tok/s + fidélité courte ; échec/incompatibilité = résultat publiable, ne bloque pas le cycle |

Chaque gate FAIL/null se documente au même titre qu'un PASS (« la doc porte les résultats,
pas les logs » — logs/ gitignorés).

## 7. Risques & mitigations

| Risque | Mitigation |
|---|---|
| Quota comptime `@typeName` (pjrt structSize) re-débordé par les champs EngineCfg | noms courts, patch 100_000 vérifié en T0, **build 3090 tôt** (leçon L3) |
| Rev ZML 3090 ≠ adee932e (cudnn/sdpa différents, lignes citées fausses) | capture rev en T0 avant tout engagement ; patch pjrt à réappliquer si resync |
| scatter batché `.b>1` jamais exercé — divergence silencieuse (pas de crash) | gate B1 dédié avant tout portage complet |
| Mismatch argmax légitime (ties, GEMM différentes) pris pour un bug | procédure d'échec pré-enregistrée §4, top5 embarqué, critère court 48 |
| OOM pendant le sweep = crash GPE upstream | arrêt par projection §3.3, jamais « pousser jusqu'au crash » |
| Contention 3090 (Ollama ~22 Go) | garde de contention bbatch + `nvidia-smi` avant chaque point |
| A/B sdpa pollué par bruit inter-compiles (2-16 %) | multi-runs, verdict différentiel, budget bruit consigné dans la spec du banc |
| Compile RAM/temps par valeur de B (35 couches × B) | liste de B bornée, compile time consigné par point, chunking en secours documenté |
| deploy silencieusement raté → test de l'ancien binaire (vécu 11 juil) | sha256 du binaire vérifié avant chaque run (pattern g2_3_sweep) |

## 8. Non-objectifs

- Continuous batching / serveur (requêtes entrantes/sortantes en cours de génération).
- Prompts de longueurs hétérogènes : padding, positions `[b]`, masques par lane.
- Vrai prefill S>1 (option B) — reste la baseline léguée, mesurée séparément (L3 §6 [it.6]).
- Sampling in-graph (RNG device) ; sampling de référence (le mode charge est une approximation top-K).
- Flash-attention batchée (paged attention, layout cache incompatible) ; bump de révision ZML.
- Combinaison sdpa × familles de précision bf16 (G2.3).

## 9. Périmètre exact

**Fichiers touchés** :
- `zml_runner/engine.zig` — champs `b` + `attn` d'EngineCfg, 5 reshapes, branche attention.
- `zml_runner/gemma4_bbatch.zig` — **nouveau** runner.
- `zml_runner/BUILD.bazel` — cible `gemma4_bbatch` (pattern zig_binary, srcs engine+mem_probe).
- `scripts/` — script sweep (pattern g2_3_sweep.sh) + wrapper génération d'oracles en lot (optionnel).
- `docs/` — ce design + doc de résultats du banc.

**Garantis intacts** : `gemma4_gen_auto.zig` (recompilé pour T0/S1, zéro ligne modifiée),
`gemma4_decode4.zig` (oracle E1), tous les runners existants, fixtures existantes,
`deploy_to_3090.sh`.
