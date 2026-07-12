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
| B2 — fidélité par lane (B=2, B=4) | ✅ fait (12 juil) | **PASS** | 2/2 puis 4/4 lanes à 48/48 == HF ; non-vacuité OK |
| B3 — indépendance inter-lanes | ✅ fait (12 juil) | **PASS** | 4 lanes identiques, 48/48 steps |
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

---

## B2 — fidélité par lane vs HF — **PASS** (12 juillet 2026)

Chaque lane est comparée **à sa propre fixture HF mono B=1** (jamais à un run HF batché, dont
le padding changerait la numérique de la référence — spec §4). Jeu : 4 prompts **distincts** de
même longueur tokenisée (19 tokens), fixtures produites par `scripts/60_batch_oracles.sh`
(N invocations de `scripts/49`, `--n-tokens 48` figé).

| Run | Verdict | Détail |
|---|---|---|
| **B=2** (prompts distincts) | **PASS 2/2** | lanes 0 et 1 : **48/48 argmax-match** |
| **B=4** (prompts distincts) | **PASS 4/4** | lanes 0-3 : **48/48 argmax-match** |
| **Non-vacuité** (fixture corrompue) | **FAIL obligatoire obtenu** | `fed[10]` altéré (12345) sur la lane 1 → FAIL **au step gen=10 exactement** (`généré=674, attendu=12345`), diagnostic top-5 émis (marge top1−top2 = 9,48e-2) ; **lane 0 reste PASS** → le gate mord, et il mord *par lane*. |

**Aucune bifurcation d'argmax légitime observée** : la procédure d'échec pré-enregistrée (§4 —
requalification différentielle sur tie) **n'a pas eu à servir**. Le batching ne dégrade pas la
fidélité : à B=2 et B=4, chaque lane reproduit HF token pour token sur 48 steps.

## B3 — indépendance inter-lanes — **PASS** (12 juillet 2026)

`--replicate 4` sur un prompt unique, 48 steps : **les 4 lanes produisent des ids identiques à
chaque step** (48/48 comparés), et leurs textes détokenisés sont identiques — c'est d'ailleurs
exactement la réponse HF de la fixture correspondante (« ## Le Mystère de Moustache… »),
recoupement croisé gratuit avec B2. Aucune contamination : le `scatterSlices` à position
scalaire partagée n'écrit pas d'une lane sur l'autre (ce que B1/2 avait prouvé sur mini-graphe,
B3 le confirme sur le moteur complet).

**Limite honnête** : l'EOT n'arrive pas dans la fenêtre de 48 steps pour ce prompt
(`step EOT=null` sur les 4 lanes) → la clause « même step d'EOT » du critère est **vacuous ici**
(cas prévu par le plan). Le critère porteur reste l'égalité des ids par step, qui est vérifiée.

## Perf — premiers chiffres (non gatés, B4 fera la mesure protocolaire)

| B | Prefill (par lane) | Génération agrégée | Par lane | Compile |
|---|---|---|---|---|
| 2 | 65-73 tok/s | **210 tok/s** | 105 tok/s | 25,8 s |
| 4 | 69-71 tok/s | **402 tok/s** | 100 tok/s | 22,1 s |

Le débit agrégé scale **quasi linéairement** (×1,9 à B=2, ×3,6 à B=4 par rapport au mono ~110)
avec une érosion par lane très faible (110 → 105 → 100 tok/s). Chiffres indicatifs : le verdict
de non-régression B4 se fera sur **runs frais appariés, 3×, médiane** (protocole pré-enregistré).

---

## Finding — bifurcation d'argmax sur tie à B=8 (le risque pré-enregistré s'est matérialisé)

Au premier run à B=8 (`--replicate 2` sur le jeu de 4 prompts), **2 lanes sur 8 divergent de leur
fixture HF** — et l'analyse montre que **ce n'est pas un bug de batching** :

```
B2 lane 1 : FAIL — 20/48 match, 1er mismatch au step gen=18
B2 lane 5 : FAIL — 20/48 match, 1er mismatch au step gen=18     (même prompt que lane 1)
  step gen=18 : généré=1017  attendu(fed)=236764
  top-5 : idx={1017, 236764, …}  val={15.564460, 15.564353, …}
  marge top1−top2 = 1,07e-4
B3 PASS — lanes identiques 48/48 steps (4 groupes × 2 réplicas)
```

**Ce qui est établi** :
1. Les deux lanes qui divergent portent le **même prompt** (lane 5 = réplica de lane 1) et
   divergent **ensemble, au même step, vers le même token** — B3 le confirme indépendamment
   (les lanes répliquées restent bit-identiques entre elles).
   → **Aucune contamination inter-lanes** : le batching est correct.
2. Les deux logits en compétition diffèrent de **1,07e-4 sur ~15,56** (écart relatif ~7e-6) :
   c'est un **tie quasi parfait**. Changer B change les shapes GEMM, donc les kernels et l'ordre
   de réduction XLA — un tie de cette finesse bascule.
3. Les 6 autres lanes restent à **48/48**, et à B=2 et B=4 **toutes** les lanes sont à 48/48.

**Application de la procédure d'échec pré-enregistrée** (spec §4, pattern A2) : le FAIL brut est
**publié** ci-dessus ; le diagnostic top-5 montre une marge fine (tie) ; le critère est donc
**requalifié en différentiel** pour B ≥ 8 — c'est exactement ce que le protocole prévoyait
(§3 : « B ≥ 8 → spot-check lane 0 »). La leçon L3 se répète : **le critère N/N n'est pas
exigible dès que l'ordre de réduction change** (précédents : HF ne se reproduit pas lui-même
1016/1020 ; bifurcation G2b à 960 vs 590 selon la compile).

**Ce que ça dit du batching en production** : jusqu'à B=4, la fidélité est **stricte** (== HF
token pour token). Au-delà, les sorties restent parfaitement cohérentes entre lanes et
sémantiquement correctes, mais un tie occasionnel peut faire diverger une lane de la trajectoire
HF de référence — comportement attendu de tout moteur batché en float, pas un défaut du portage.

## Piège d'instrumentation (sweep v1) — corrigé

Le premier sweep a rapporté **B=4 → FAIL** alors que le même run **PASSE 4/4 en isolation**.
Cause : le script enchaînait les runs sans attendre la libération de la VRAM du précédent → la
**garde de contention du runner** (`error.GpuBusy`) refusait le run, et l'absence de « B2 PASS »
dans la sortie se lisait comme un échec de fidélité. `scripts/61_batch_sweep.sh` attend désormais
un GPU réellement libre avant chaque run et **loggue la cause** de tout FAIL.
Leçon : un banc qui ne distingue pas « le test a échoué » de « le test n'a pas tourné » produit
des faux FAIL — exactement le genre de silence que la discipline maison cherche à éliminer.

## Mesure VRAM — le pic est dominé par la compilation, pas par le cache

Mesure fine (`--no-prealloc`, échantillonnage 0,2 s, scopé au PID) :

| B | pic VRAM |
|---|---|
| 1 | 16 698 MiB |
| 8 | 16 686 MiB |

Le pic **ne croît pas avec B** dans cette gamme : il est atteint **pendant la compilation XLA**
(workspace transitoire), alors que le coût marginal par lane (~38 Mo de cache KV f32) reste noyé
dedans. Conséquence : dans la gamme testée, **le plafond de batch n'est pas gouverné par la VRAM
du cache** mais par le pic de compilation — le seuil de garde de `gemma4_gen_auto` (20 GiB,
calibré B=1) reste donc valide tel quel.
