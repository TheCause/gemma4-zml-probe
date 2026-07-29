# Spec — sampling : faire du 12B un moteur d'inférence conforme (phase 2)

> **Date** : 2026-07-29 · **Révision 2** · **Machines** : 3090 (runner) + M4 (oracle)
>
> ## Autorité — à lire avant tout
>
> Ce document **ne couvre QUE la phase 2 (sampling)** et **les points d'intégration** avec le
> chantier `generation_config` livré le 29 juil.
>
> **La phase 1 (repetition penalty) est régie intégralement par
> `2026-07-27-sampling-repetition-penalty-design.md` (rév. 4).** La rév. 1 du présent document
> avait réémis ses gates sous les mêmes noms, avec des critères **plus faibles** — dont un
> **vacueux**. Elle est réputée nulle sur la phase 1. Motif complet et 26 findings arbitrés :
> `2026-07-29-sampling-penalty-arbitrage.md`.
>
> **Règle de nommage, non négociable** : les gates de la rév. 4 gardent leurs noms
> (`RP0`…`RP7`, `SM0`…`SM3`). Les gates du présent document sont préfixés **`S2-`** (phase 2).
> Aucun tag ne peut désigner deux contenus : `gate/rp2-pass` appartient à la rév. 4 et à elle
> seule.

---

## 0. Ce qui a changé depuis la rév. 1, et pourquoi

Trois relecteurs adversariaux et un workflow de mesure ont produit **30 corrections**. Les cinq
qui changent la structure :

| # | Fait neuf | Conséquence |
|---|---|---|
| **D1** | La rév. 1 avait réécrit la phase 1 en moins bien | La phase 1 est rendue à la rév. 4. Ce document ne la touche plus. |
| **D2** | `zig` **absent du PATH**, **zéro** `zig_test` dans `BUILD.bazel` (vérifié) | Aucun `zig test`. Les selftests passent par `--selftest-*`, patron de `--selftest-gencfg` livré le 29 juil. La rév. 4 avait **déjà** tranché cela (amendement 1) ; la rév. 1 l'avait réintroduit. |
| **D3** | **Les témoins 48 et 124 sont DÉTERMINISTES** — mesuré (F17) | Le gate-pont peut s'y adosser. La bistabilité est **prompt-spécifique**, pas une propriété générale du 12B. |
| **D4** | `--temperature 0` : HF **lève une `ValueError`** (mesuré, F15) | **Divergence assumée avec la rév. 4**, qui prévoyait « cas spécial branché sur `argmax` ». La rév. 4 décidait sans connaître ce fait. On reproduit HF. |
| **D5** | Le critère de succès de la rév. 1 était **inatteignable** | « l'oracle HF produit la même séquence » est impossible avec deux RNG distincts. Réécrit en §1. |

---

## 1. Objectifs et critère de succès

**O1 — valider chaque brique en régime déterministe avant d'activer l'aléatoire.** Chaque brique
a un régime neutre où elle n'a pas le droit d'être aléatoire, et où elle doit rendre **bit-à-bit**
ce que le moteur rendait avant elle.

**O2 — moteur d'inférence conforme.** `generation_config.json` déclare
`do_sample: true, top_k: 64, top_p: 0.95, temperature: 1.0`. Le greedy est un régime que Google
**ne recommande pas**.

**Critère de succès — réécrit (D5), et atteignable :**

1. `--seed 42` deux fois ⇒ **ids identiques** (comparés sur la seule ligne `generated`, cf. V3).
2. `--repetition-penalty 1.0 --top-k 1` ⇒ ids **bit-identiques** au greedy d'avant le chantier.
3. En teacher-forcing, à chaque position, **l'ensemble des tokens survivants** après
   `penalty → suppress → T → topK → topP` est **identique** à celui de HF, et les probabilités
   renormalisées coïncident à 1e-6. *(C'est déterministe — contrairement aux séquences tirées.)*

---

## 2. Faits établis (mesurés, avec l'instrument nommé)

**F8 — ordre des processors.** `_get_logits_processor` compte **26** `processors.append(...)`.
Rangs réels : `RepetitionPenalty` **4**, `SuppressTokens` **15**, `Temperature` **17**,
`TopK` **19**, `TopP` **20**.
⇒ **Penalty → Suppress → Temperature → TopK → TopP.**
⚠ Existent entre nos étapes, **inertes ici** mais à nommer : `SuppressTokensAtBegin` (**16**) et
**`TopHLogitsWarper` (18)** — ce dernier **casserait l'adjacence Temperature→TopK** s'il était
armé.
*(La rév. 1 annonçait 4/14/16/18/19 sur 23 : sa sonde comptait les classes dédupliquées, pas les
`append`. Un fait annoncé « mesuré » l'avait été avec un instrument faux.)*

**F8a — la température s'applique AVANT top-p, et cela change le résultat.** Mesuré sur V=4096,
`top_p = 0,95` : `T=0,5 → 14` survivants, `T=1,0 → 521`, `T=2,0 → 2313`. L'ordre n'est pas
cosmétique.

**F8b — la température N'EST PAS neutre vis-à-vis de top-k en f32.** La division est monotone au
sens **large** : l'arrondi peut **fusionner** deux logits distincts en une égalité ; si elle tombe
au rang k, la règle stricte de F9 fait **survivre un token de plus**. Contre-exemple mesuré :
`[100, 12.5308094, 12.5308104, 1]`, `top_k=2` → 2 survivants ; après `/0.7` → **3**.
*(La rév. 1 affirmait l'inverse, en contradiction avec sa propre F9, et cette phrase était
destinée à être gravée en commentaire.)*

**F8c — `temperature: 1.0` ⇒ HF n'instancie PAS le warper** (`temperature is not None and
temperature != 1.0`). **Donc la température est ABSENTE de la chaîne réelle de ce modèle** : un
gate « config Google complète » ne l'exercerait **jamais**. D'où **S2-T**, gate dédié à `T ≠ 1`.

**F9 — `TopKLogitsWarper` : les ex æquo du k-ième SURVIVENT.** `scores < topk(...)[..., -1]`,
inégalité **stricte**. Mesuré : `[3,2,1,1,1]`, `k=3` → **5** survivants ; 64 logits égaux,
`k=8` → **64**. **`top_k` ne borne pas le nombre de survivants.**
Deux lignes que la rév. 1 avait coupées : `self.top_k = max(top_k, min_tokens_to_keep)` (`__init__`)
et `top_k = min(self.top_k, scores.size(-1))` (`__call__`).

**F10 — `TopPLogitsWarper` trie en ASCENDANT et retranche par le bas** (`cum <= 1 - top_p`),
`min_tokens_to_keep` protégeant la **queue** du tri ascendant (= les plus probables).
Contre-exemple **disjoint** mesuré (le plus discriminant) : 8 logits égaux, `top_p = 0,25` →
HF garde **{6,7}**, la formulation naïve descendante garde **{0,1}** — **intersection vide**.
Aucun test de cardinalité ne l'attraperait. Sans égalité exacte, les deux formulations divergent
encore **4 fois sur 2000** tirages aléatoires (V=4096) par pur arrondi de sommation.

**F11 — `filter_value = -inf`** sur les deux warpers.

**F14 — `torch.sort` n'est PAS stable par défaut, ET son comportement DÉPEND DE LA TAILLE.**
Mesuré : pour n ≤ 64 les ex æquo ressortent en ordre d'index ; à partir de n ≥ 256, **non**.
Sur le vocabulaire réel (262 144, tout égal, `top_p=0,5`) HF rend 131 072 survivants dont
l'index 0, **en ensemble non contigu**.
⇒ **Une fixture d'edge-case écrite à n ≤ 64 exerce un AUTRE chemin que la production.**
Conséquence contractuelle : l'identité exacte des survivants sur ex æquo **n'est pas garantie**.
La comparaison porte sur la **classe d'équivalence** (multiset trié des logits survivants + masse
de probabilité), et **au moins un cas de fixture est à n = 262 144**.

**F15 — `--temperature 0` : HF lève une `ValueError`, avant tout forward.** Aucune validation dans
`generation_config` (`.validate()` passe) ; le garde-fou est dans
`TemperatureLogitsWarper.__init__` (`not (temperature > 0)`), et le warper **est** construit
(`0.0 != 1.0`), donc l'exception part à la construction de la `LogitsProcessorList`.
⚠ **Piège mesuré** : `> 0` **ne suffit pas**. `T = 1e-45` (dénormal) **passe** le garde-fou et
produit `[inf, inf, inf, 0] → softmax = [nan, nan, nan, nan]` (`inf - inf`). Reproduire HF
littéralement hériterait de ce trou.

**F16 — coûts mesurés sur la cible** (VM 3090, PCIe 4.0 ×16, Ryzen 7 5800X ; ⚠ *déviation de
custody* : `gemma4_g12auto` était résident à 22 210 MiB pendant la mesure — les absolus sont à
requalifier sur GPU libre, le **delta** entre bras absorbe l'essentiel du biais) :

| opération | coût | % d'un step (9,09 ms) |
|---|---|---|
| D2H 1 MiB **pinned**, marginal | **52,2 µs** | 0,6 % |
| D2H 1 MiB **pageable** | +51 µs de plus | — |
| `partial_sort` top-64 sur 262 144 | **90,5 µs** | 1,0 % |
| softmax plein vocab | 817 µs | 9,0 % |
| **`std::sort` complet** | **16 176 µs** | **183 %** |
| bras K=5 (baseline) | 5,87 µs | — |

⇒ Le surcoût **dépend entièrement de l'algorithme host** : 1,55 % bien implémenté, **183 % avec
un tri complet**. Le seuil de la rév. 1 (« < 15 % ») ne pouvait ni distinguer 1,55 % de 10,8 %,
ni être atteint par une implémentation correcte.

**F17 — les témoins 48 et 124 sont REPRODUCTIBLES.** Campagne dédiée du 29 juil, **flux séparés**,
chaque run **recompilant à neuf** (38-40 s, aucun cache XLA) :

| témoin | runs | trajectoires distinctes | marge la plus fine | rapport au jitter |
|---|---|---|---|---|
| **48** | **23/23 identiques** | **1** | 0,0924 @ pos 43 | **~90×** son jitter |
| **124** | **13/13 identiques** | **1** | 0,0136 @ pos **120** | **~3,1×** son spread |

Les compiles **diffèrent réellement** (`max|Δ logit|` = 1,3e-2 à 1,7e-2) : la variabilité
d'autotuning accusée d'avoir fait basculer `@47` **a été exercée 20 fois**, et les témoins y ont
résisté. Plus fort encore : les témoins archivés datent du **26 juil**, produits par un binaire
**antérieur** au chantier `generation_config`, et le binaire du 29 juil les reproduit
**bit-identiques** — ils ont survécu à un rebuild **et** à un changement de source.

⇒ **La bistabilité est une propriété de la position @47 du prompt « zero story » (marge 0,004587),
pas une propriété du 12B en génération libre.** Nuance importante au finding, à propager.

⚠ **Réserve nommée sur le 124** : la marge @120 (0,0136) n'est qu'à ~3× de son spread de compile,
et toute la fragilité est concentrée dans les **15 derniers tokens** (zone de clôture avant EOS) ;
sur les indices 0..109 la marge la plus fine remonte à 0,092. **Durcissement retenu, gratuit** :
le gate porte sur les **110 premiers ids**. Si le gate échoue un jour, **regarder la position 120
avant d'accuser le code**.

---

## 3. Claims falsifiables — PRÉ-ENREGISTRÉES

**S2-C1 — le chemin complet, en régime neutre, est indiscernable du chemin greedy.**
Prédiction : `--repetition-penalty 1.0 --top-k 1` ⇒ ids **bit-identiques**, témoins 48 (48 ids) et
124 (**110 premiers ids**, F17). **Ce qui la tue** : un id différent, hors ex æquo instrumenté.

**S2-C2 — les warpers reproduisent HF, y compris aux bords ET à l'échelle réelle.**
Prédiction : selftest **100 %** sur une fixture contenant au minimum (a) le cas **disjoint** de
F10, (b) `min_tokens_to_keep` mordant, (c) l'égalité au rang k de F9, (d) **au moins un cas à
n = 262 144** (F14). **Ce qui la tue** : un seul cas de bord divergent. **Ce qui la rend vacueuse
si on l'omet** : ne tester qu'à n ≤ 64.

**S2-C3 — le tirage suit la distribution.** χ² à α = 0,01, **n = 10 000**, **k = 10 ids
distincts** (`top_k = 10` exactement — pas des bins agrégés, sinon une permutation d'ids **dans**
un bin est invisible). Distribution théorique par implémentation **indépendante** (torch).
**Non-vacuité** : injection **half-split** (+b sur k/2, −b sur k/2) à **b = 10 %** doit **FAIL**.
*Dérivation* : en half-split λ = n·b², **indépendant de k** ⇒ puissance ≥ 99,98 % à k ∈ {10,32,64}.
*(La rév. 1 exigeait 5 % en mono-catégorie : χ² = 2,78 contre 21,67 critique — elle n'aurait pas
mordu. Et en mono-catégorie il faudrait 18,7 % à k=10 mais **67,5 %** à k=64, ce qui ne serait plus
un biais mais un sampler cassé.)*

**S2-C4 — le graphe ne bouge pas.** HLO pré-optimisation **byte-identique**, avec **fraîcheur
prouvée** : répertoire de dump vidé, mtime postérieur à l'édition, nombre de fichiers **et** volume
publiés des deux côtés (patron GC0 : 510 fichiers, 1 905 860 o). **Ce qui la tue** : md5 différent
⇒ un `Tensor.Rng` a fui dans le graphe.

**S2-C5 — le surcoût du chemin complet est borné.** Prédiction : Δ médian du bloc
{D2H + sample} ≤ **250 µs/step** (≈2,8 %, soit ~1,8× le coût mesuré d'une implémentation correcte).
**Ce qui la tue** : Δ ≥ **900 µs/step** (softmax plein vocab ou pire).

---

## 4. Design — les deux chemins et le pont

### 4.1 Chemin A (inchangé) — greedy nu
`penalty == 1.0` et sampling désactivé : top-5 rapatrié (~48 o), `gencfg.select()`. Strictement le
code prouvé par les 12 gates du chantier précédent.

### 4.2 Chemin B — dès qu'une brique s'arme
Vecteur complet rapatrié (il **sort déjà** du graphe, F1) puis, dans l'ordre de F8 :
`penalty → suppress(-inf) → ÷T → topK → topP → argmax|tirage`.

⚠ **Algorithme imposé** : `partial_sort` / heap top-K, **jamais** un tri complet (F16 : 183 % d'un
step). Mémoire hôte **pinned** exigée et vérifiée (pageable = +51 µs/step pour rien).

### 4.3 `S2-PONT` — le gate-pont, refondu

**Ce que la rév. 1 faisait mal** : elle comparait un run chemin-B à un témoin chemin-A **stocké**
(deux forwards, deux compiles). Or GC2 a mesuré `n_suppress_hits = 0` sur les témoins 48/124 : le
gate tournait sur les deux seuls témoins où A et B **ne peuvent pas** diverger.

**Forme retenue** : faire tourner **les deux sélecteurs sur le MÊME vecteur de logits, au MÊME
step, dans le MÊME processus** — les deux sorties du graphe (top-5 **et** logits complets) sont
disponibles simultanément. Insensible à la bistabilité **par construction**, et exerçant **chaque**
step.

**Critères** : `n_steps_compared == n_generated` · `n_disagree == 0` · **et** les compteurs de
non-vacuité **par branche**, tous non nuls : `n_suppress_hits ≥ 1` (⇒ **utiliser un témoin où la
suppression mord** — le témoin 200 teacher-forcé mord @57), `n_exact_top_ties` publié.
*(`n_full_path > 0` seul est **trop faible** : satisfait par 1 step sur 124.)*

⚠ **Réserve de tie-break** : l'identité A↔B suppose l'accord entre l'`argmax` host (« premier
indice gagnant ») et l'ordre du `topK` in-graph, que `gencfg.zig:21-25` documente comme **non
vérifié**. Un désaccord sur un **tie exact** est un **verdict distinct**, publié, pas un STOP
silencieux.

### 4.4 La garde `suppress.len + 1 > TOP_K` — décision prise
Elle vit dans `gencfg.fromLists` (`gencfg.zig:207`) et **refuse de charger la politique**, avant
tout choix de chemin : elle est donc **globale**, pas « chemin A seulement ». Elle est **maintenue
telle quelle** (la relâcher casserait GC4(c), gate vert et taggé). Le commentaire doit dire qu'elle
borne l'**argument de rang du chemin A** tout en s'appliquant globalement — c'est une contrainte
plus forte que nécessaire, assumée.

---

## 5. `zml_runner/sampling.zig` et la surface CLI

Fonctions **pures** sur `f32` nus, sans dépendance ZML ⇒ exerçables par `--selftest-sampling`
(host-only, **pas** `zig test`, D2).

`applyRepetitionPenalty` · `applySuppression` (délègue à `gencfg.isSuppressed`, pas de seconde
implémentation) · `applyTemperature` · `applyTopK` (**doit porter `min_tokens_to_keep`** :
`k_eff = min(max(k, min_keep), vocab)`, F9) · `applyTopP` (**F10 à la lettre** : tri ascendant,
`<=`, `min_keep` en queue) · `sample` (RNG **host**) · `argmax`.

**Trois interdits, chacun adossé à un gate** : pas de `× (1/x)` au lieu d'une division (casse la
bit-exactitude) · pas de retranscription de formule (fixtures issues des **vrais** processors,
**y compris pour la penalty**) · **aucune allocation par step** (bitset et buffer alloués une fois).

### Gardes en ACCEPTATION — jamais en rejet

> Règle héritée de la rév. 4 §8 : `p <= 0 → rejet` **laisse passer `NaN`**. Toute comparaison avec
> `NaN` étant fausse, `NaN` échoue toute acceptation et **passe** tout rejet.

| Flag | Défaut neutre | Garde |
|---|---|---|
| `--repetition-penalty` | `1.0` | `!(p > 0 and isFinite(p))` ⇒ `error.InvalidPenalty` |
| `--temperature` | `1.0` | `!(t >= T_MIN and isFinite(t))` ⇒ `error.InvalidTemperature` |
| `--top-k` | `0` (désactivé, convention HF) | `parseInt(u32)`, **plus** `!(k <= vocab)` ⇒ rejet explicite, **jamais un clamp silencieux** |
| `--top-p` | `1.0` | `!(p > 0 and p <= 1)` ⇒ `error.InvalidTopP` |
| `--seed` | `null` (**pas** `0` : `0` est une graine légitime) | `parseInt(u64)` ; **absent alors que le tirage est armé ⇒ `error.SeedRequired`** — jamais d'ensemencement par l'horloge |
| `--min-tokens-to-keep` | `1` | `!(m >= 1 and m <= vocab)` ⇒ rejet (`m = 0` ferait paniquer `argmax` sur une slice vide) |

**`--temperature 0` — décision D4, divergence datée avec la rév. 4.** HF **lève une `ValueError`**
(F15) ; la rév. 4 prévoyait « brancher sur `argmax` » sans connaître ce fait. **On reproduit HF** :
rejet, avec le message renvoyant vers `--top-k 1` pour du greedy déterministe.

**`T_MIN` — divergence délibérée, déclarée.** HF accepte `T = 1e-45` et produit des `NaN` (F15).
On impose `T_MIN` tel que `max|logit| / T ≤ 3,4e38` ; logits Gemma bornés par le softcap 30
⇒ `T_MIN = 1e-30`. **Écrit comme divergence**, pas comme reproduction.

---

## 6. Gates — phase 2

| Gate | Contenu | PASS pré-enregistré |
|---|---|---|
| **S2-PONT** | Les deux sélecteurs, même vecteur, même step, même processus (§4.3) | `n_disagree == 0` · `n_steps_compared == n_generated` · `n_suppress_hits ≥ 1` · `n_exact_top_ties` publié — **FAIL ⇒ STOP** |
| **S2-U** | `--selftest-sampling` sur fixtures des **vrais** warpers | **100 %**, avec les 4 classes de bord de S2-C2 dont **un cas à n = 262 144** ; comparaison par **classe d'équivalence** (F14) ; **compteurs d'antécédent** non nuls, patron GC1 |
| **S2-T** | **Température exercée** (`T = 0,7`), teacher-forcé | ensembles de survivants == HF sur 200 positions. **Sans ce gate, `applyTemperature` ne serait validé nulle part** (F8c) |
| **S2-D** | Distributionnel (S2-C3) | χ² α=0,01, k=10 **ids distincts** ; **non-vacuité half-split b=10 % ⇒ FAIL** ; **un seul re-run, seed pré-déclarée, deux échecs = FAIL** |
| **S2-R** | Reproductibilité | même seed ⇒ **ligne `generated` identique** (V3) |
| **S2-N** | Non-vacuité du RNG | ≥ k sorties distinctes sur N seeds, **k et N calculés depuis la fixture AVANT** la mesure · `n_draws > 0` · assertion dure **`logits[token_tiré] > -inf`** à chaque step, comptée |
| **S2-E** | Équivalence bout-en-bout (D5-3) | **ensembles de survivants** identiques à HF sur 200 positions ; probabilités renormalisées à 1e-6 |
| **S2-P** | Surcoût (S2-C5) | chronomètre **in-process**, bras **alternés ABAB dans le MÊME processus**, test des signes sur 48 paires · PASS ≤ 250 µs/step · **non-vacuité obligatoire : le bras `std::sort` complet doit être exécuté une fois et FAIL** |
| **S2-G** | Graphe (S2-C4) | HLO byte-identique **avec fraîcheur prouvée** (dump vidé, mtime, nb fichiers, volume) |

**Règle d'arrêt** : S2-PONT, S2-U, S2-D et S2-G ⇒ **STOP**. Aucune requalification sans décision
écrite de Régis.

---

## 7. Vigilances

**V1 — un gate position-par-position doit être teacher-forcé**, sauf sur les témoins **dont le
déterminisme est mesuré** (F17 : 48 et 124, ce dernier borné à 110 ids). Ailleurs : critère
statistique.

**V2 — séparer stdout et stderr** dans toute capture destinée à un grep (`> out 2> err`).

**V3 — un détecteur ne doit matcher ni la ligne de configuration, ni le bruit d'environnement.**
Corollaire nouveau : S2-R et S2-N comparent la **seule ligne `generated`** — sinon l'écho
`seed=<n>` rendrait N sorties trivialement distinctes (**S2-N ne pourrait plus échouer**) et un
horodatage empêcherait S2-R de réussir. Contre-prouver **dans les deux sens**.

**V4 — `temperature: 1.0` ⇒ ne pas instancier le warper**, comme HF (F8c). *(Motif corrigé : la
rév. 1 justifiait cela par « `x / 1.0` n'est pas neutre en f32 », ce qui est **faux** — la division
par 1,0 est exacte en IEEE-754. La conduite était bonne, son motif était une supposition.)*

**V5 — la penalty comprime les marges** (÷1,15) : elle fabrique des quasi-ties. C'est pourquoi
l'instrument fp32 précède les gates de penalty. ⚠ **Précision (rév. 2)** : HF **upcaste en fp32
avant tout processor** (`_sample` : `.to(copy=True, dtype=torch.float32)`), quel que soit le dtype
du forward. Le forward fp32 de ce projet a un motif **distinct** — l'oracle bf16 fabrique des ties
artificiels (leçon J2) — et **ce motif-là reste valide**. Ne pas écrire « HF fait un forward fp32 ».

**V6 — interaction penalty × EOS, non instruite** : pénaliser un id EOS présent dans l'historique
modifie sa probabilité, **donc l'arrêt**. La dette « équivalence de l'arrêt » existe déjà ; la
penalty l'**aggrave**.

**V7 — l'état inter-prompts croît.** Ce chantier ajoute bitset, buffer de logits **et état RNG**.
`seed` est associée à **(seed, index de token DANS le prompt courant)** — remise à zéro par prompt,
comme le compteur de suppression (R12, prouvé par GC10). Sans cette décision, S2-R serait vrai en
one-shot et faux en session.

---

## 8. Hors périmètre et dettes

**Hors périmètre** (décision Régis, périmètre « conformité de génération ») : multi-tour avec
contexte accumulé · stop sequences · streaming · batching B>1 · beam search ·
`no_repeat_ngram_size` · câblage E2B.

**Dettes écrites** :
- **E2B** : ne sort pas ses logits du graphe (`gen_auto.zig:753`) ⇒ « reproduit `generate()` »
  reste **faux** pour lui.
- **Équivalence de l'arrêt** runner ↔ HF : prouvée par aucun gate (héritée), **aggravée** par V6.
- **RP7 « la récitation est-elle levée » : SUSPENDU** (décision datée du 29 juil). Le symptôme
  d'origine du chantier n'a **jamais été reproduit** (3 témoins greedy, hypothèse réfutée) : le
  gate est inexécutable faute de cas. Il se **ré-arme** si un prompt qui récite est trouvé.
- **Ancrages de ligne de la rév. 4 §3.2 périmés** depuis le merge (`in_gen_phase` `:1363`→`:1755`,
  `generateOnce` `:1296`→`:1680`, sites `:1244/:1254/:1283`→`:1624/:1634/:1663`). Ils sont
  **indicatifs** ; la règle anti-off-by-one (`hist.append(fed)` en tête d'itération) reste valide.
- **Custody de F16** : mesures prises GPU non vierge (22 210 MiB résidents). Les **absolus** sont à
  requalifier sur GPU libre ; les **deltas** entre bras portent le verdict.

---

## 9. Ordre d'exécution

1. Phase 1 **selon la rév. 4** (RP0 → RP7), sans modification.
2. `sampling.zig` + fixtures des vrais warpers + **S2-U** (host-only).
3. Chemin B câblé + **S2-PONT** (STOP).
4. **S2-T**, **S2-E** (teacher-forcés).
5. **S2-D**, **S2-R**, **S2-N** (stochastique).
6. **S2-P**, **S2-G**.
7. Documentation, passe de nuance (GC11 doit rester vert), PR.

---

## 10. Historique de révision

**Rév. 1** (29 juil) — réfutée par trois revues adversariales et un workflow de mesure : phase 1
réécrite en moins bien, `zig test` inexistant réintroduit, gate RP-3 vacueux, gate-pont posé là où
les deux chemins ne peuvent pas diverger, χ² non discriminant, seuil de débit ni atteignable ni
mordant, deux tables de gates en collision de noms.

**Rév. 2** (29 juil) — la phase 1 est rendue à la rév. 4 ; préfixe `S2-` pour éviter toute
collision ; F8/F9/F10 corrigés et complétés (F14 tri non stable, F15 `T=0` et dénormaux, F16 coûts
mesurés, F17 déterminisme des témoins) ; gate-pont refondu en comparaison intra-processus ; seuils
recalculés (χ² half-split 10 %, surcoût 250 µs/step avec non-vacuité par le bras `std::sort`) ;
gardes en acceptation pour les 6 paramètres ; `--temperature 0` aligné sur HF (divergence datée
avec la rév. 4) ; `T_MIN` déclaré comme divergence délibérée.
