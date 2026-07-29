# Spec — sampling : faire du 12B un moteur d'inférence conforme (phase 2)

> **Date** : 2026-07-29 · **Révision 3** (resserrée) · **Machines** : 3090 (runner) + M4 (oracle)

## 0. Autorité — à lire avant tout

Ce document ne couvre **que la phase 2 (sampling)** et **les points d'intégration** avec le
chantier `generation_config` livré le 29 juil.

| Domaine | Document qui fait foi |
|---|---|
| Phase 1 — repetition penalty, gates **`RP0`…`RP7`** | **`2026-07-27-…-design.md` (rév. 4)**, intégralement et sans modification |
| Phase 2 — sampling | **le présent document**, gates préfixés **`S2-`** |
| Gates **`SM0`…`SM3`** de la rév. 4 | ⚠ **SUPERSÉDÉS** par les gates `S2-` ci-dessous — **ne pas les exécuter** |

⚠ **La supersession de `SM0…SM3` est explicite et motivée** : `SM1` prescrit un χ² avec « binning
~10 catégories » et « biais injecté de 5 % », deux critères **réfutés par la mesure** (§2, C3) — un
biais de 5 % y donne χ² = 2,78 contre 21,67 critique. La rév. 2 déclarait par erreur ces gates « en
vigueur » : un implémenteur suivant sa règle de nommage aurait exécuté le test réfuté.

**Rien d'autre de la rév. 4 n'est modifié par ce document.** En particulier `RP7` (« la récitation
est-elle levée ») **reste tel qu'il y est écrit** ; la proposition de le suspendre — le phénomène
n'a jamais été reproduit, donc le gate est sans cas — est portée au §7 comme **dette à arbitrer**,
pas appliquée ici.

Historique et motifs : `2026-07-29-sampling-penalty-arbitrage.md` (42 décisions), révisions §10.

---

## 1. Objectifs et critère de succès

**O1 — valider chaque brique en régime déterministe avant d'activer l'aléatoire.**
**O2 — moteur conforme** : `do_sample: true, top_k: 64, top_p: 0.95, temperature: 1.0` est la
configuration **nominale** de Google ; le greedy est un régime qu'il ne recommande pas.

**Critère de succès** (atteignable, contrairement à celui de la rév. 1) :

1. `--seed N` deux fois ⇒ **ligne `generated` identique**.
2. `--repetition-penalty 1.0 --top-k 1` ⇒ ids **bit-identiques** au greedy d'avant le chantier.
3. Les warpers reproduisent HF **à la classe d'équivalence près** (§2, F14) sur des fixtures
   incluant l'échelle réelle.

---

## 2. Faits établis (mesurés ; instrument nommé ; chiffres re-vérifiés)

**F8 — ordre.** `_get_logits_processor` compte **26** `append`. Rangs : `RepetitionPenalty` **4**,
`SuppressTokens` **15**, `Temperature` **17**, `TopK` **19**, `TopP` **20**
⇒ **Penalty → Suppress → Temperature → TopK → TopP**. Inertes ici mais présents entre nos étapes :
`SuppressTokensAtBegin` (16), **`TopH` (18)** — ce dernier casserait l'adjacence Temperature→TopK
s'il était armé.

**F8a — la température s'applique avant top-p et change le nombre de survivants** (V=4096,
`top_p=0,95`) : `T=0,5 → 14` · `T=1 → 521` · `T=2 → 2313`.

**F8b — la température n'est PAS neutre vis-à-vis de top-k en f32.** La division est monotone au
sens **large** : l'arrondi peut fusionner deux logits en une égalité et, sous la règle stricte de
F9, faire survivre un token de plus. Mesuré : `[100, 12.5308094, 12.5308104, 1]`, `k=2` → 2
survivants ; après `/0,7` → **3**.

**F8c — `temperature: 1.0` ⇒ HF n'instancie pas le warper.** Donc **la température est absente de
la chaîne réelle de ce modèle** : aucun gate « config Google » ne l'exercerait. Couverture : par
fixture (S2-U), et **dette GPU déclarée** (§7).

**F9 — `TopKLogitsWarper` : les ex æquo du k-ième survivent** (`scores < topk(...)[-1]`, strict).
Mesuré : `[3,2,1,1,1]`, `k=3` → **5** survivants ; 64 logits égaux, `k=8` → **64**.
**`top_k` ne borne pas le nombre de survivants.** Le constructeur applique de plus
`k_eff = min(max(k, min_tokens_to_keep), vocab)`.

**F10 — `TopPLogitsWarper` trie en ASCENDANT et retranche par le bas** (`cum <= 1 − top_p`),
`min_tokens_to_keep` protégeant la **queue** du tri ascendant (= les plus probables).
Contre-exemple **disjoint** mesuré : 8 logits égaux, `top_p = 0,25` → HF garde **{6,7}**, la
formulation naïve descendante **{0,1}** — **intersection vide**. Sans aucune égalité exacte, les
deux divergent encore **4 fois sur 2000** par arrondi de sommation.
⚠ **Note d'implémentation** : le masque est ramené dans l'espace vocabulaire par un `scatter` dont
le tenseur **de base** est `sorted_indices_to_remove` lui-même. Transposer naïvement produit un
masque juste dans l'espace **trié** et faux dans l'espace **vocabulaire**.

**F11 — `filter_value = -inf`** sur les deux warpers.

**F14 — `torch.sort` n'est pas stable, et la bascule se produit entre n = 128 et n = 129.**
Jusqu'à **128** les ex æquo ressortent en ordre d'index ; à partir de **129**, non.
⇒ **Une fixture écrite sous 129 exerce un autre chemin que la production à 262 144.**
Conséquence contractuelle : l'identité exacte des survivants sur ex æquo **n'est pas garantie** ;
la comparaison porte sur la **classe d'équivalence** — multiset trié des logits survivants et masse
de probabilité — et **au moins un cas de fixture est à n = 262 144**.

**F15 — `--temperature 0` : HF lève une `ValueError` avant tout forward**
(`TemperatureLogitsWarper.__init__`, `not (temperature > 0)` ; le warper est construit puisque
`0.0 != 1.0`). ⚠ **`> 0` ne suffit pas** : `T = 1e-45` **passe** le garde-fou et produit
`[inf,…] → softmax = [nan,…]`.

**F16 — coûts mesurés**, tous rapportés à **un step de référence de 9 090 µs** (9,0 tok/s) :

| opération | coût | % d'un step |
|---|---|---|
| D2H 1 MiB pinned, marginal | 52,2 µs | 0,57 % |
| `partial_sort` top-64 sur 262 144 | 90,5 µs | 1,00 % |
| **D2H + `partial_sort`** (implémentation visée) | 142,7 µs | **1,57 %** |
| + softmax plein vocab | 959,7 µs | **10,56 %** |
| **tri complet** | 16 176 µs | **177,95 %** |

*(La rév. 2 annonçait 1,55 / 10,8 / 183 % : elle mélangeait deux bases de step. Base unique
ci-dessus.)* ⚠ **Custody** : mesures prises GPU **non vierge** (22 210 MiB résidents). Les
**absolus** sont à requalifier ; les **écarts entre bras** portent le verdict.

**F17 — les témoins 48 et 124 sont reproductibles.** Campagne du 29 juil, flux séparés, chaque run
**recompilant à neuf** :

| témoin | runs | trajectoires | marge la plus fine |
|---|---|---|---|
| **48** | **23/23 identiques** | **1** | 0,0924 @ pos 43 |
| **124** | **13/13 identiques** | **1** | 0,0136 @ pos 120 |

**36 compiles distinctes** au total (23 + 13), pas 20 : la variabilité d'autotuning **a été
exercée** — les compiles diffèrent réellement d'un run à l'autre. Plus fort : les témoins archivés
du **26 juil**, produits par un binaire **antérieur** au chantier `generation_config`, sont
reproduits **bit-identiques**.

⇒ **La bistabilité est une propriété de la position @47 du prompt « zero story » (marge 0,004587),
pas du 12B en génération libre.** Ce qui décide de la légitimité d'un gate en roue libre est **la
marge minimale du témoin rapportée au jitter de compile**.

⚠ **Réserve sur le 124** : sa fragilité est concentrée dans ses **derniers tokens** (zone de
clôture avant EOS) ; sa marge @120 n'est qu'à ~3× de son spread de compile. **Un gate qui s'y
adosse porte sur ses 110 premiers ids**, et un échec futur se diagnostique **@120 avant d'accuser
le code**.

---

## 3. Claims falsifiables — PRÉ-ENREGISTRÉES

**C1 — le chemin complet, en régime neutre, est indiscernable du chemin greedy.**
Prédiction : `--repetition-penalty 1.0 --top-k 1` ⇒ ids bit-identiques (témoin 48 ; témoin 124
borné à 110 ids). **Tue la claim** : un id différent hors égalité exacte instrumentée.

**C2 — les warpers reproduisent HF, aux bords ET à l'échelle réelle.**
Prédiction : 100 % sur une fixture contenant (a) le cas **disjoint** de F10, (b) `min_tokens_to_keep`
mordant, (c) l'égalité au rang k de F9, (d) **au moins un cas à n = 262 144**. **Tue la claim** :
un cas de bord divergent. **La rend vacueuse** : ne tester que sous n = 129 (F14).

**C3 — le tirage suit la distribution.** χ², α = 0,01, n = 10 000, **k = 10 ids distincts**
(pas des bins agrégés : une permutation d'ids **dans** un bin serait invisible), sur des **logits
figés en fixture**, distribution théorique par implémentation **indépendante**.
**Non-vacuité** : injection **half-split** (+b sur k/2, −b sur k/2) à **b = 10 %** doit **FAIL**.
*Dérivation* : λ = n·b², **indépendant de k** ⇒ puissance ≥ 99,98 % à k ∈ {10, 32, 64}. En
mono-catégorie il faudrait 18,7 % à k=10 mais **67,5 %** à k=64 — plus un biais, un sampler cassé.

**C4 — le graphe ne bouge pas.** HLO pré-optimisation byte-identique, **fraîcheur prouvée** :
répertoire vidé, mtime postérieur à l'édition, nombre de fichiers **et** volume publiés (patron
GC0 : 510 fichiers, 1 905 860 o). **Tue la claim** : un md5 différent ⇒ un `Tensor.Rng` a fui.

---

## 4. Design — deux chemins, un pont

**Chemin A (inchangé)** : `penalty == 1.0` et sampling désactivé ⇒ top-5 (~48 o),
`gencfg.select()`. Code prouvé par les 12 gates du chantier précédent.

**Chemin B** : vecteur complet (il **sort déjà** du graphe, rév. 4 F1, aujourd'hui non lu), puis
`penalty → suppress(-inf) → ÷T → topK → topP → argmax|tirage` (ordre F8).
⚠ **Algorithme imposé** : `partial_sort`/heap top-K — **jamais un tri complet** (F16 : 177,95 %
d'un step). Mémoire hôte **pinned**.

**La garde `suppress.len + 1 > TOP_K`** vit dans `gencfg.fromLists` (`gencfg.zig:207`) et **refuse
le chargement** de la politique : elle est donc **globale**, pas propre au chemin A. Elle est
**maintenue** (la relâcher casserait GC4(c), gate vert et taggé) ; le commentaire doit dire qu'elle
borne l'argument de rang du **chemin A** tout en s'appliquant partout — contrainte plus forte que
nécessaire, **assumée**.

---

## 5. `zml_runner/sampling.zig` et surface CLI

Fonctions **pures** sur `f32` nus, sans dépendance ZML ⇒ exerçables par `--selftest-sampling`
(host-only). **Aucun `zig test`** : `zig` est absent du PATH et `BUILD.bazel` ne charge que
`zig_binary` (vérifié).

`applyRepetitionPenalty` *(implémentation régie par la rév. 4)* · `applySuppression` (délègue à
`gencfg.isSuppressed`) · `applyTemperature` · `applyTopK` (**porte `min_tokens_to_keep`** :
`k_eff = min(max(k, min_keep), vocab)`) · `applyTopP` (**F10 à la lettre**, `scatter` compris) ·
`sample` (RNG **host**) · `argmax`.

**Interdits** : pas de `× (1/x)` au lieu d'une division · pas de retranscription de formule
(fixtures issues des **vrais** processors) · **aucune allocation par step**.
⚠ Ce dernier n'a **pas** de gate porteur (§7, dette D3) : il est tenu par la revue de code, et
c'est écrit plutôt que sous-entendu.

### Gardes en ACCEPTATION — jamais en rejet

> `p <= 0 → rejet` **laisse passer `NaN`** : toute comparaison avec `NaN` est fausse, donc `NaN`
> échoue toute acceptation et **passe** tout rejet.

| Flag | Défaut neutre | Garde |
|---|---|---|
| `--repetition-penalty` | `1.0` | `!(p > 0 and isFinite(p))` |
| `--temperature` | `1.0` | `!(t >= T_MIN and isFinite(t))` |
| `--top-k` | `0` (désactivé, convention HF) | `parseInt(u32)` + `!(k <= vocab)` ⇒ rejet, **jamais un clamp silencieux** |
| `--top-p` | `1.0` | `!(p > 0 and p <= 1)` |
| `--seed` | `null` (**pas** `0` : graine légitime) | `parseInt(u64)` ; **absent alors que le tirage est armé ⇒ `error.SeedRequired`** — jamais l'horloge |
| `--min-tokens-to-keep` | `1` | `!(m >= 1 and m <= vocab)` (`0` ferait paniquer `argmax` sur une slice vide) |

**`--temperature 0`** : **rejet, comme HF** (F15), message renvoyant vers `--top-k 1`.
⚠ **Divergence délibérée et déclarée** : `T_MIN = 1e-30` (logits bornés par le softcap 30), parce
que HF accepte `1e-45` et produit des `NaN`.

**État par prompt** : bitset, buffer de logits **et RNG** sont remis à zéro **à chaque prompt**
(`seed` associée à *(seed, index de token dans le prompt courant)*), comme le compteur de
suppression (R12, prouvé par GC10).

---

## 6. Gates — **cinq**, et une mesure publiée

> **Resserrement (rév. 3).** La rév. 2 en portait neuf ; la vérification a montré que **les neuf
> avaient une faille**, dont trois n'étaient pas exécutables avec l'outillage réel. Chaque gate
> ci-dessous passe **trois filtres** : exécutable · démontrablement capable d'échouer · antécédent
> non vide. Ce qui n'a pas passé les trois est une **dette écrite** (§7), pas un gate de plus.

| Gate | Contenu | PASS pré-enregistré |
|---|---|---|
| **S2-PONT** | Les **deux sélecteurs** sur le **même vecteur de logits, au même step, dans le même processus** (les deux sorties du graphe sont disponibles simultanément) | `n_disagree == 0` **hors égalités exactes** · `n_steps_compared == n_generated` · **antécédent** : `n_suppress_hits ≥ 1` (⇒ témoin où la suppression **mord** — le témoin 200 teacher-forcé mord @57) · `n_exact_top_ties` **publié** ; un désaccord **sur une égalité exacte** est un **verdict distinct**, pas un FAIL — **FAIL ⇒ STOP** |
| **S2-U** | `--selftest-sampling` sur fixtures des **vrais** warpers HF | **100 %** avec les 4 classes de C2, **dont un cas à n = 262 144** ; comparaison par **classe d'équivalence** ; **compteurs d'antécédent non nuls** (patron GC1 : un cas dont le compteur vaut 0 est **inexécutable**, pas PASS) |
| **S2-D** | Distributionnel (C3), logits **figés en fixture** | χ² α=0,01, k=10 ids distincts ; **non-vacuité : half-split b=10 % ⇒ FAIL** ; **un seul re-run, seed pré-déclarée, deux échecs = FAIL** |
| **S2-R** | Reproductibilité et non-vacuité du tirage | même seed ⇒ **ligne `generated` identique** (comparer **cette ligne seule** : l'écho `seed=` rendrait N sorties trivialement distinctes) · `n_draws > 0` · assertion dure **`logits[token_tiré] > -inf`**, comptée · N seeds ⇒ ≥ k sorties distinctes, **k et N calculés depuis la fixture AVANT** la mesure |
| **S2-G** | Graphe (C4) | HLO byte-identique **avec fraîcheur prouvée** (dump vidé, mtime, nb fichiers, volume) |

**Mesure publiée — `M-COUT`** *(ce n'est pas un gate : aucun PASS/FAIL)*. Chronomètre **in-process**
autour du bloc {D2H + sample}, bras **alternés A/B dans le même processus**, médiane par step sur
48 steps, **publiée** avec la table F16 en regard. Motif du déclassement : le surcoût visé (1,57 %)
est **sous le plancher de résolution** du protocole de débit du projet (bruit inter-compiles 2-16 %)
— un seuil PASS y serait soit inatteignable, soit incapable de distinguer 1,57 % de 10,6 %. On
**mesure et on publie** plutôt que de prétendre trancher.

**Règle d'arrêt** : S2-PONT, S2-U, S2-D, S2-G ⇒ **STOP**. Aucune requalification sans décision
écrite de Régis.

---

## 7. Dettes — ce qui n'est PAS couvert, et pourquoi

| # | Dette | Motif du non-gate |
|---|---|---|
| **D1** | **`applyTopP` n'est exercé sur GPU par aucun gate.** Sa couverture est **la fixture seule** (S2-U) | Le régime neutre du pont est `--top-k 1`, qui **court-circuite** top-p. Un régime neutre par top-p exigerait un véhicule GPU dédié, non disponible. ⚠ C'est la brique dont F10/F14 montrent qu'une formulation naïve rend un ensemble **disjoint** — la dette est donc **sérieuse et déclarée** |
| **D2** | **`applyTemperature` n'est exercé de bout en bout nulle part** | `temperature: 1.0` ⇒ HF n'instancie pas le warper (F8c) : la config Google ne l'exerce jamais. Couverture = fixture seule |
| **D3** | **L'interdit « aucune allocation par step » n'a pas de gate porteur** | Le déclasser en revue de code est honnête ; l'adosser au débit ne le serait pas (F16 : l'effet est sous le plancher de résolution) |
| **D4** | **Équivalence de l'arrêt runner ↔ HF** : prouvée par aucun gate (héritée) | **Aggravée** par ce chantier : pénaliser un id EOS présent dans l'historique modifie sa probabilité, **donc l'arrêt** |
| **D5** | **`RP7` (« la récitation est-elle levée »)** — proposition : **suspendre** | Le symptôme d'origine n'a **jamais été reproduit** (3 témoins greedy, hypothèse réfutée) : le gate est sans cas. ⚠ **RP7 appartient à la rév. 4** ; cette proposition doit y être portée par une décision datée, **elle n'est pas appliquée ici** |
| **D6** | **E2B non couvert** | Ses runners ne sortent pas les logits du graphe (`gen_auto.zig:753`) |
| **D7** | **Custody de F16** | Mesures prises GPU non vierge : absolus à requalifier |
| **D8** | **Tie-break `argmax` host vs `topK` in-graph** non vérifié (`gencfg.zig:21-25`) | S2-PONT le **publie** (`n_exact_top_ties`) au lieu de le supposer résolu |

---

## 8. Vigilances

**V1** — un gate position-par-position doit être teacher-forcé, **sauf** sur un témoin dont le
déterminisme est **mesuré** (F17). Le critère est la **marge minimale rapportée au jitter**.
**V2** — séparer stdout et stderr dans toute capture destinée à un grep.
**V3** — un détecteur ne doit matcher ni la ligne de configuration, ni le bruit d'environnement ;
contre-prouver **dans les deux sens**.
**V4** — `temperature: 1.0` ⇒ ne pas instancier le warper, comme HF. *(Motif corrigé : `x / 1.0`
**est** exact en IEEE-754 ; la rév. 1 justifiait la bonne conduite par un fait faux.)*
**V5** — HF **upcaste en fp32 avant tout processor**, quel que soit le dtype du forward. Le forward
fp32 de ce projet a un motif **distinct** — l'oracle bf16 fabrique des ties artificiels (leçon J2)
— et ce motif reste valide. **Ne pas écrire « HF fait un forward fp32 ».**

---

## 9. Ordre d'exécution

1. **Phase 1 selon la rév. 4** (`RP0` → `RP7`), sans modification.
2. `sampling.zig` + fixtures des vrais warpers + **S2-U** (host-only, aucune compile GPU).
3. Chemin B câblé + **S2-PONT** (STOP).
4. **S2-D**, **S2-R** (stochastique, fixtures figées).
5. **S2-G**, puis la mesure **M-COUT** publiée.
6. Documentation, passe de nuance (GC11 doit rester vert), PR.

---

## 10. Historique de révision

**Rév. 1** — réfutée par trois revues : phase 1 réécrite en moins bien, `zig test` inexistant,
gate `RP-3` vacueux, gate-pont posé là où les deux chemins ne peuvent pas diverger, χ² non
discriminant, seuil de débit ni atteignable ni mordant.

**Rév. 2** — phase 1 rendue à la rév. 4 ; F14-F17 ajoutés ; gate-pont refondu. **Vérifiée à son
tour** : 37 décisions sur 42 intégrées, mais **les 9 gates avaient une faille**, 3 n'étaient pas
exécutables, la règle de nommage **réinstallait** la contradiction qu'elle devait tuer
(`SM0…SM3` déclarés en vigueur alors qu'ils portent les critères réfutés), et **quatre chiffres
étaient mal transcrits** (bascule `torch.sort` 64/256 au lieu de **128/129** ; 183 % au lieu de
**177,95 %**, deux bases de step mélangées ; « 20 compiles » au lieu de **36** ; une marge attribuée
au mauvais témoin).

**Rév. 3** — **resserrement** : 9 gates → **5 gates + 1 mesure publiée**, chacun passant trois
filtres (exécutable · capable d'échouer · antécédent non vide). Supersession de `SM0…SM3` rendue
explicite. Chiffres corrigés et rapportés à une **base unique**. Ce qui ne passe pas les filtres
devient une **dette écrite** (§7, 8 dettes) — dont la plus sérieuse, **D1**, dit noir sur blanc que
`applyTopP` n'a **aucune** couverture GPU.
