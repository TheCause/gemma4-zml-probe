# Batching statique + variante d'attention sdpa — design

> **Statut** : spec validée par Régis le 12 juillet 2026 (cadrage par questions + choix explicites) ;
> durcie par revue multi-lentilles (3 réviseurs : canonique, fact-check adversarial contre les
> sources, scope/pattern repo) — 4 issues majeures corrigées [rev.1-4], mineures intégrées.
> **Décisions de cadrage (Régis)** : usage cible = **banc multi-prompts (mesure/recherche)** → batch
> **statique** ; le **plafond VRAM est un output du banc**, pas un input ; **greedy par défaut**
> (oracle == HF conservé), temperature/top-k **en option host-side hors gates** ; flash-attention
> **dans le même cycle, après le batching** (variante A/B mesurée par le banc) ; runner **autonome**
> (squelette gen_auto) ; critère de fidélité = **argmax 48/48 par lane + requalification
> différentielle pré-enregistrée**.
> Base : L3 in-graph mergé main (PR #7, tag `gate/l3-ingraph-pass`). Second sous-projet du backlog
> (`PLANNING.md:37`, seul item non-[x]).
> **Structure de livraison : deux phases séquencées** [rev.2] — Phase batch (T0→B4,
> **shippable seule**) puis Phase sdpa (S1→S3, détachable sans invalider la première) ;
> sampling §3.5 = dernière tâche détachable ; F1 = spike optionnel non bloquant.

## 1. But

Étendre le moteur decode au **batch statique B>1** pour instrumenter la scalabilité du runtime
autonome sur la 3090 : tok/s (agrégé et par lane) et pic VRAM en fonction de B, **plafond B
rapporté par le banc**. Ensuite, dans le même cycle, introduire une **variante d'attention
comptime `sdpa`** (`zml.nn.sdpa`) mesurée A/B par ce banc (fidélité + perf).

Doctrine perf du repo inchangée : **fidélité obligatoire, perf mesurée sans objectif dur**
(non-régression différentielle seulement) — décision de cadrage L3 reprise telle quelle.

## 2. Découvertes de lecture qui cadrent le design

Faits vérifiés dans les sources (12 juil 2026, checkout ZML M1 `adee932e` ; re-vérifiés
un à un par le fact-check de revue) :

1. **`B` n'est consommé qu'à 5 sites** dans `engine.zig` : reshapes q/k/v (`engine.zig:395,412,417`)
   et les 2 reshapes PLE (`engine.zig:536,539`) — seule autre occurrence : la déclaration
   `pub const B` ligne 23. Tous les tenseurs (Cache `{slot,b,h,k,hd}`, Packed `{step,b,...}`,
   transposes, logits, topK) portent déjà le tag `.b`. L'axe paramétrique est un changement de
   dimension, pas de plomberie.
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
   tags (`tensor.zig:2183-2195`). Le `mask.broad(scores.shape())` existant (`engine.zig:461`)
   est rank-égal : à B>1 il broadcaste `b:1→B` positionnellement — **correct ici parce que
   l'ordre des axes coïncide**, mais c'est un invariant fragile → vérifié explicitement en B1.
   Règle de design : les tables Packed (masques, cos/sin) restent **à b=1** (jamais ×B en VRAM).

## 3. Architecture

### 3.1 Engine — B dérivé des shapes (shape-polymorphe) [rev.3]

**Décision (tranchée en spec, pas au plan)** : PAS de champ `b` dans `EngineCfg`. Les 5 sites
consommateurs dérivent la dimension batch (et seq) **des shapes d'entrée** :

```zig
// avant :  .reshape(.{ B, S, NH, hd })
// après :  .reshape(.{ x.dim(.b), x.dim(.s), NH, hd })   // x = tensor d'entrée du site
```

Justification — c'est le geste G2.3 réappliqué (« sortir la config du type ») :
- **Un binaire unique** sert tous les B → doctrine de custody G2.3 §7.1 conservée telle quelle
  (build unique pour tout le sweep, sha256 constant vérifié avant chaque run — pas de custody
  par point).
- **Zéro allongement de `@typeName`** (le quota comptime pjrt structSize n'est pas touché).
- B devient un **paramètre runtime du runner** (il construit tok `{B,1}` et caches
  `{slot,B,...}` ; le graphe est tracé/compilé au lancement comme aujourd'hui, ~17 s).
- Neutralité : avec des entrées à `b=1`, les dims émises sont identiques à l'actuel →
  preuve HLO gold (gate T0). Les constantes `pub const B/S` restent déclarées (compat
  runners existants) mais plus consommées par le moteur.
- **Ordre d'émission intouché** : `rmsScaleDPrec` émet `wi` avant le rmsNorm,
  `rmsScaleHdPrec` l'inverse (`engine.zig:86-98`) — toute harmonisation casserait le byte-diff.

### 3.2 Runner `gemma4_bbatch.zig` (nouveau, autonome)

Squelette gen_auto généralisé par lane. Nom court (13c < plafond ~20c du quota comptime).

- **CLI** : `<model.safetensors> <tokenizer.json>` + `--prompts <fichier>` (un prompt par ligne)
  OU `--oracles <f1,f2,...>` (N fixtures 49) ; `--replicate N` (duplique la liste de prompts,
  points de sweep perf) ; `--max-tokens`, `--force-vram`, `--allow-cpu`, `--no-prealloc`
  (mesure VRAM, pattern g23_sweep), `--temperature/--top-k/--seed` (mode charge, §3.5),
  `--selftest-batch <fixture>` (spike B1, §6).
  **B est implicite = nombre de prompts/fixtures fournis** (après `--replicate`) — aucun
  mismatch compile/runtime possible [rev. canonique].
- **Contrainte V1** : tous les prompts à la **même longueur tokenisée** — sinon
  `error.PromptLengthMismatch` (le message liste les longueurs par lane). Gardes existantes
  par lane : `ids.len < SLIDING_WINDOW=512`, `ids.len + limit ≤ L_MAX=1024`.
  **Opérationnalisation** [rev.4] : mode `--ids-only` de bbatch = tokenise les candidats et
  rapporte les longueurs (tri par longueur) — sert d'outil de constitution du jeu ; le fichier
  `--prompts` du banc est un **artefact versionné** (`fixtures/bench_prompts_b<N>.txt`,
  committé) ; **politique pré-enregistrée** : gates de fidélité (B2) sur prompts **distincts**
  à B=2 et B=4 ; points de sweep B≥8 : `--replicate` autorisé pour la mesure perf, avec
  fidélité spot-check (la lane 0 doit == sa fixture).
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
  actives) / gen_s, et tok/s **par lane** ; prefill mesuré séparément. Le **K du topK est
  consigné au manifest** (confound K=64 bbatch vs K=5 gen_auto — si B4 est marginal, re-run
  avec K apparié) [rev. canonique].
- **Garde VRAM** : la garde 20 GiB de gen_auto est calibrée B=1 et reste **intouchée**.
  bbatch porte une **garde de contention** (refus si d'autres process compute occupent le GPU,
  échappatoire `--force-vram`) — logique nvidia-smi **dupliquée** depuis checkVram, PAS
  extraite en module partagé (gen_auto reste intact d'un octet) [rev.5]. Pas de seuil fixe :
  le plafond est ce que le banc mesure.

### 3.3 Banc/sweep (script, pattern `g2_3_sweep.sh`)

- **Protocole pré-enregistré committé AVANT tout run** (discipline G2.3 §5) :
  `docs/BATCH_BENCH_PROTOCOL.md` fige, avant le premier point de mesure : la liste des B,
  le jeu de prompts (fichier versionné), le nombre de runs par point, les statistiques,
  le budget bruit, et le protocole B4 (§6). Tout écart = déviation documentée.
- **Un build unique pour tout le sweep** (§3.1) : sha256 du binaire consigné une fois,
  vérifié avant chaque run (le hash qui change = sweep invalide — doctrine G2.3 §7.1).
- Balayage B ∈ {1, 2, 4, 8, 16, …} (liste bornée, pas linéaire — chaque B = un graphe
  tracé/compilé au lancement, ~17 s par point, compile time consigné).
- Chaque point : run oracle/spot-check (fidélité) puis run mesure `--no-prealloc` +
  échantillonnage `nvidia-smi --query-compute-apps` → pic(B) (méthode G3 — piège 14 :
  `mem_probe` = RSS host ; sous prealloc, nvidia-smi ne montre que la réserve BFC 0.90×libre).
- **Arrêt propre par projection** : pic(2B) projeté ≈ pic(B) + B×Δ (Δ ≈ 38 Mo cache/lane
  mesuré + marge activations) ; si projection > ~22 GiB → stop avant OOM. On ne compte PAS
  sur un échec propre : l'error-path OOM upstream crashe en General protection exception
  (`io.zig` deinit, bug connu `PLANNING.md:70-77`).
- Manifest JSON de custody : sha256 binaire, `zml_rev` (capturé T0), K du topK, driver, date,
  table B → {tok/s agrégé, tok/s par lane, pic MiB, verdict fidélité, compile s}.
- **Avant tout run** : `nvidia-smi --query-compute-apps` (contention Ollama = piège
  opérationnel n°1) ; `deploy_to_3090.sh` rate silencieusement sans `ZML_REMOTE`/`ZML_DST` ;
  flag `--@zml//platforms:cuda=true` obligatoire (repli CPU silencieux sinon).

### 3.4 Variante d'attention `sdpa` (Phase 2 — livrée avec son gate) [rev.2]

**Ordre de livraison tranché** : le champ `attn` d'`EngineCfg` et la branche comptime au site
scores/softmax/PV (`engine.zig:459-467`) arrivent **ensemble, en Phase sdpa, après B4** —
jamais un champ sans branche (gate de neutralité vacueux). T0 ne revendique donc QUE la
neutralité du B shape-polymorphe ; **S1 est le gate de neutralité de `attn`**, exécuté au
moment de sa livraison.

```zig
pub const EngineCfg = struct {
    ring: bool = false,
    two_masks: bool = false,
    kmax_sliding: i64 = 8,
    kmax_full: i64 = 8,
    attn: enum { manual, sdpa } = .manual,  // Phase 2 uniquement
};
```

- `.manual` (défaut) : chemin actuel, byte-identique (gate S1 — pattern `two_masks`).
- `.sdpa` : `zml.nn.sdpa(q_final, cache_k, cache_v, .{ .attn_mask = mask, .scale = un })`
  avec `un = Tensor.scalar(1.0, dtype)` (**obligatoire**, découverte n°7) ; masques additifs
  existants passés tels quels (comportement broad vérifié en B1, découverte n°8) ;
  sdpa fait le même splitAxis GQA que l'engine (`nn.zig:1093` == `engine.zig:459`) et la même
  convention softmax f32 ; head_dim hétérogène (sliding 256 / full 512) : sdpa est appelé
  par couche, transparent.
- **Pas de byte-identité attendue** pour `.sdpa` (scale émis sur K même à 1.0 pré-opt, ordre
  transpose/merge différent) : c'est un **A/B mesuré et oraclé**, pas prouvé.
- **Interaction PrecRt** : en mode `.sdpa`, les familles `qk_scores`/`softmax`/`pv_ctx` ne
  s'appliquent plus (sdpa fait ses propres converts). Les gates sdpa tournent en **fp32 pur**
  (prec par défaut) ; combiner sdpa × familles bf16 G2.3 = hors périmètre.
- **Note quota comptime** : ce champ enum allonge `@typeName` (tension avec la leçon G2.3) —
  mitigations : noms courts, patch pjrt 100_000 vérifié en T0, build 3090 tôt (leçon L3 [it.5]).

### 3.5 Sampling optionnel (mode charge, hors gates — dernière tâche, détachable)

- `topK` in-graph passe à K paramétrable dans bbatch (défaut 64 ; gen_auto reste K=5,
  aucune contrainte d'identité entre les deux binaires — les gates de bbatch sont vs fixtures HF).
- `--temperature T` / `--top-k k ≤ K` / `--seed s` : softmax host sur les K logits rapatriés
  par lane, RNG host seedé par (seed, lane, step) — **zéro RNG device**, D2H ≈ K×8 octets/lane/step.
- **Troncature assumée et documentée** : échantillonnage restreint au top-K rapatrié
  (approximation de charge, pas une implémentation de sampling de référence).
- Greedy (`indices[0]`) reste le seul mode gaté ; aucun oracle en mode sampling.

## 4. Oracle et fidélité

- **N invocations de `scripts/49_gen_custom_oracle.py`** (une fixture safetensors + manifest
  par prompt — mécanisme existant, zéro format nouveau ; wrapper shell pour générer le lot).
  Prompts du banc choisis à même longueur tokenisée via `--ids-only` (§3.2), jeu committé.
- **Référence = HF mono B=1 par séquence** (jamais un run HF batché : le padding HF changerait
  la numérique de la référence elle-même).
- **Vigilance pré-enregistrée** : changer B change les shapes GEMM → kernels/ordres de
  réduction XLA différents → logits non bit-identiques au mono. Une lane peut bifurquer
  d'argmax **sans bug** (précédents : HF ne se reproduit pas lui-même 1016/1020 ; ties à marge
  0,006 ; non-déterminisme inter-compiles, « piège 15 » de DOCUMENTATION.md).
- **Procédure d'échec pré-enregistrée** (pattern A2) : au premier mismatch d'une lane,
  diagnostic top-5 du step fautif (marge top1−top2) ; si la marge est fine (tie), le FAIL brut
  est **publié** puis requalifié en critère différentiel documenté ; sinon c'est un vrai FAIL.

## 5. Baselines (chiffres de référence, L3 mergé)

| Mesure | Valeur | Source |
|---|---|---|
| Génération B=1 (L3) | 110-113 tok/s ≥ replay 109 ≥ B0 91,4 | `L3_INGRAPH_DESIGN.md:111-120` |
| Prefill-par-decode | ~71,5 tok/s (mesuré séparément) | `DOCUMENTATION.md` §usage |
| Compile StepTok GPU | ~17 s | `L3_INGRAPH_DESIGN.md:113` |
| Pic VRAM B=1 | 16 658 MiB ≈ 16,27 GiB (--no-prealloc) | commit `75ee030` |
| Garde gen_auto | 20 GiB = ceil(16,27/0,90)+1 (B=1, intouchée) | `gen_auto.zig:679` |
| Coût marginal/lane | ~38 Mo cache f32 (kmax 1024) + ~1 Mo logits | calcul §2 + mesure G2.3.2 |
| Bifurcation longue G2b | 960 ≥ replay 590 (non-déterminisme inter-compiles documenté) | `L3_INGRAPH_DESIGN.md:116` |
| Bruit perf inter-compiles | ~2 % (grand effet) à ~16 % (petit) | `DOCUMENTATION.md` piège 15 |

NB : les références de lignes vers `DOCUMENTATION.md`/`PLANNING.md` (documents vivants)
driftent — préférer les ancres de section (« piège 14/15 ») [rev. fact-check].

## 6. Validation — gates (un commit/tag par gate, `gate/batch-*`)

**Phase 1 — batching (shippable seule)** :

| Gate | Contenu | Critère PASS |
|---|---|---|
| **T0** | Neutralité du B shape-polymorphe (`dim(.b)/dim(.s)` aux 5 sites) ; capture `git rev-parse` ZML 3090 + vérif patch pjrt | md5 `module_0001.zml.before_optimizations.txt` **identique** sur gemma4_gen_auto recompilé (méthode gold G2.3.0 — jamais le post-opt, piège 15 ; gen_auto est le bon témoin : il instancie EngineModel avec les défauts) ; rev consignée au manifest |
| **B1** | Spike primitives batchées sur 3090, mode `--selftest-batch` de bbatch : scatterSlices update `.b=2` au même pos (jamais exercé), gather tok `{2,1}`, topK `{b,K}` layout D2H, **et broad du masque b=1→B rank-égal (§2.8)** | mini-graphe type SgFwd, valeurs exactes vs référence host, B=2 |
| **B2** | Fidélité batch, **prompts distincts**, B=2 puis B=4 | chaque lane **48/48 == sa fixture 49** (fp32) ; procédure d'échec §4 ; **non-vacuité** : une lane corrompue (fed altéré) → FAIL obligatoire |
| **B3** | Indépendance inter-lanes — **pinné [rev.6]** : B=4, même prompt ×4, 48 steps | égalité des **ids u32 par step sur les 4 lanes**, à chaque step où les lanes sont actives (même prompt ⇒ même step d'EOT attendu ; toute divergence de step d'EOT = FAIL) |
| **B4** | Sweep B ∈ {1,2,4,8,…} → plafond. **Protocole pré-enregistré** [rev.1], committé dans `BATCH_BENCH_PROTOCOL.md` AVANT tout run : comparateur = **runs frais appariés** gen_auto vs bbatch B=1 dans la même fenêtre de session (jamais les chiffres publiés seuls) ; **3 runs par bras ; statistique = médiane ; budget bruit = −5 %** (couvre le ~2 % « grand effet » du piège 15 avec marge) ; charge = fixture 49 × 48 tokens + un run long 999 (façon G3) | table B → {tok/s agrégé, par lane, pic VRAM --no-prealloc, compile s} publiée ; plafond B rapporté ; non-régression : **médiane(bbatch B=1) ≥ 0,95 × médiane(gen_auto)** en tok/s gén |

**Phase 2 — sdpa (détachable)** :

| Gate | Contenu | Critère PASS |
|---|---|---|
| **S1** | Neutralité `attn=.manual` — livré AVEC la branche comptime (§3.4, jamais un champ sans branche) | md5 before_opt identique (même méthode que T0) |
| **S2** | Fidélité sdpa, B=1 et B=4 | 48/48 par lane vs mêmes fixtures ; procédure d'échec §4 |
| **S3** | Perf A/B sdpa vs manual | même protocole que B4 (runs appariés, 3×, médiane, budget −5 %) aux mêmes B ; verdict différentiel publié (gain OU absence de gain — le cudnn étant mort, l'A/B mesure la fusion XLA : conclusion honnête exigée) |
| **F1** *(spike optionnel, non bloquant)* | FA2 custom call à B=1 | mesure tok/s + fidélité courte ; échec/incompatibilité = résultat publiable, ne bloque pas le cycle |

Chaque gate FAIL/null se documente au même titre qu'un PASS (« la doc porte les résultats,
pas les logs » — logs/ gitignorés).

## 7. Risques & mitigations

| Risque | Mitigation |
|---|---|
| Rev ZML 3090 ≠ adee932e (cudnn/sdpa différents, lignes citées fausses) | capture rev en T0 avant tout engagement ; patch pjrt à réappliquer si resync |
| scatter batché `.b>1` jamais exercé — divergence silencieuse (pas de crash) | gate B1 dédié avant tout portage complet |
| broad rank-égal par positions (masques) donne le bon résultat « par chance » (ordre des axes) | vérification explicite en B1 ; tables Packed jamais étendues à `.b>1` |
| Mismatch argmax légitime (ties, GEMM différentes) pris pour un bug | procédure d'échec pré-enregistrée §4, top5 embarqué, critère court 48 |
| OOM pendant le sweep = crash GPE upstream | arrêt par projection §3.3, jamais « pousser jusqu'au crash » |
| Contention 3090 (Ollama ~22 Go) | garde de contention bbatch + `nvidia-smi` avant chaque point |
| A/B pollué par bruit inter-compiles (2-16 %) | protocole B4/S3 pré-enregistré : runs appariés, 3×, médiane, budget −5 % |
| Confound K topK (64 bbatch vs 5 gen_auto) dans B4 | K consigné au manifest ; si résultat marginal, re-run K appariés |
| Compile RAM/temps par valeur de B (35 couches × B) | liste de B bornée, compile time consigné par point, chunking en secours documenté |
| deploy silencieusement raté → test de l'ancien binaire (vécu 11 juil) | sha256 du binaire vérifié avant chaque run (pattern g2_3_sweep) |
| Quota comptime `@typeName` (champ enum `attn`, Phase 2) | noms courts, patch 100_000 vérifié en T0, build 3090 tôt (leçon L3) |

## 8. Non-objectifs

- Continuous batching / serveur (requêtes entrantes/sortantes en cours de génération).
- Prompts de longueurs hétérogènes : padding, positions `[b]`, masques par lane.
- Vrai prefill S>1 (option B) — reste la baseline léguée, mesurée séparément (L3 §6 [it.6]).
- Sampling in-graph (RNG device) ; sampling de référence (le mode charge est une approximation top-K).
- Flash-attention batchée (paged attention, layout cache incompatible) ; bump de révision ZML.
- Combinaison sdpa × familles de précision bf16 (G2.3).

## 9. Périmètre exact

**Fichiers touchés** :
- `zml_runner/engine.zig` — Phase 1 : 5 reshapes → `dim(.b)/dim(.s)` ; Phase 2 : champ `attn`
  + branche comptime au site attention.
- `zml_runner/gemma4_bbatch.zig` — **nouveau** runner (inclut le spike B1 en mode
  `--selftest-batch` — pas de runner séparé) [rev.5].
- `zml_runner/BUILD.bazel` — cible `gemma4_bbatch` (pattern zig_binary, srcs engine+mem_probe).
- `scripts/` — script sweep (pattern g2_3_sweep.sh) + wrapper génération d'oracles en lot.
- `fixtures/bench_prompts_b<N>.txt` — jeu de prompts du banc, **versionné** [rev.4].
- `docs/BATCH_BENCH_PROTOCOL.md` — protocole pré-enregistré du banc, committé avant tout run [rev.1].
- `docs/` — ce design + doc de résultats du banc.
- `PLANNING.md` — item [M] ligne 37 (avancement) [rev.5].
- `docs/DOCUMENTATION.md` — usage bbatch + pièges nouveaux (PromptLengthMismatch, garde de
  contention, broad rank-égal) [rev.5].

**Garantis intacts** : `gemma4_gen_auto.zig` (recompilé pour T0/S1, zéro ligne modifiée —
la garde de contention de bbatch duplique la logique nvidia-smi, pas d'extraction de module),
`gemma4_decode4.zig` (oracle E1), tous les runners existants, fixtures existantes,
`deploy_to_3090.sh`.
