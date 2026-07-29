# Arbitrage des findings — revue de la spec sampling rév. 1 (29 juil 2026)

> **Ce document existe pour qu'aucun finding ne disparaisse en silence.** Trois relecteurs
> adversariaux ont été dispatchés en parallèle sur `2026-07-29-sampling-penalty-design.md` rév. 1,
> avec des angles délibérément distincts : **fidélité à la source** (vérifier les faits contre
> transformers 5.14.1 réellement installé), **falsifiabilité des contrôles** (quels gates ne
> peuvent pas échouer, lesquels échoueraient à tort), **complétude** (trous, contradictions,
> contradictions avec le code livré).
>
> **26 findings.** Chacun est ici accepté, nuancé ou rejeté, avec le motif. Un relecteur se trompe
> aussi : trois d'entre eux sont nuancés ci-dessous, et le motif est écrit.
>
> **Verdict global : la rév. 1 est réécrite, pas corrigée.** Le motif commun à la majorité des
> findings n'est pas une erreur ponctuelle : en croyant « prolonger » la spec du 27 juil (rév. 4,
> deux tours de revue), je l'ai **réécrite en moins bien**. La correction structurelle est donc de
> lui rendre la phase 1 intégralement, et de ne garder ici que la phase 2 et les points
> d'intégration neufs.

## Décisions structurelles (prises avant le détail)

| # | Décision | Motif |
|---|---|---|
| **D1** | **La phase 1 (repetition penalty) redevient intégralement celle de la rév. 4.** Elle est retirée de la nouvelle spec. | La rév. 4 exigeait un mordant pré-calculé (`hamming ≥ 3` avant tout GPU), `divergences.len > 0`, la vérification à `penalty = 0.8`, et des contre-tests par mutation avec plancher. Ma rév. 1 avait remplacé tout cela par un critère plus faible **et vacueux** (F1). On ne remplace pas un contrôle éprouvé par un plus commode. |
| **D2** | **Table unique gate → critère → document qui fait foi**, dans la nouvelle spec. | Deux documents employaient les mêmes noms (`RP-2` désignait deux gates différents) : un tag `gate/rp2-pass` aurait été ambigu. Le projet a déjà payé une contradiction interne de spec (GC2 serait mort à tort). |
| **D3** | **Aucun `zig test`.** Les selftests passent par un drapeau `--selftest-*`. | Vérifié : `zig` absent du PATH, `zml_runner/BUILD.bazel` ne charge que `zig_binary`, **zéro** `zig_test`. La rév. 4 avait **déjà** corrigé cette erreur (amendement 1) ; la rév. 1 l'a réintroduite trois fois. Le patron qui marche est celui de `--selftest-gencfg`, livré hier. |
| **D4** | **RP7 (« la récitation est-elle levée ») : SUSPENDU, pas supprimé.** | Le symptôme d'origine du chantier n'a **jamais été reproduit** : `FINDING_GENERATION_CONFIG.md` rapporte 3 témoins greedy sans aucune boucle, et l'hypothèse « ça boucle au-delà de 200 tokens » est **réfutée**. Un gate sans cas reproductible est inexécutable. Il se ré-arme le jour où un prompt qui récite est trouvé. **Décision datée, pas disparition silencieuse.** |
| **D5** | **Le critère de succès global de §1 est réécrit.** | Il disait « l'oracle HF, à politique identique, produit la même séquence ». Avec `do_sample: true` des deux côtés, HF tire sur le RNG torch et le runner sur un RNG host : **les séquences ne peuvent jamais coïncider**. Un critère de succès qui ne peut pas être atteint n'est pas un critère. |

## Findings acceptés — faits (relecteur « fidélité à la source »)

| # | Finding | Décision |
|---|---|---|
| A1 | Rangs des processors faux : **4/15/17/19/20 sur 26**, pas 4/14/16/18/19 sur 23 | **ACCEPTÉ**, re-vérifié en propre. Cause : ma sonde comptait les *classes distinctes instanciées* (regex + déduplication) au lieu des `processors.append(...)` réels — elle avait perdu 3 entrées. **Un fait annoncé « mesuré » l'avait été avec un instrument faux.** L'ordre relatif, dont dépend le code, est confirmé. |
| A2 | « La température est sans effet sur top-k » est **faux en f32** | **ACCEPTÉ.** La division par `T` est monotone au sens **large** : l'arrondi f32 peut fusionner deux logits distincts en une égalité ; si elle tombe au rang k, la règle stricte de F9 laisse survivre un token de plus. C'était une **contradiction interne** avec F9, et la phrase était destinée à être gravée en commentaire — elle aurait installé une fausse garantie. |
| A3 | `temperature: 1.0` ⇒ HF n'instancie pas le warper ⇒ **la température est absente de la chaîne réelle de ce modèle** | **ACCEPTÉ.** Conséquence directe : le gate « config Google complète » **n'exercerait jamais** `applyTemperature`. Il faut un gate dédié à `T ≠ 1.0`, sinon la brique part en production non testée. |
| A4 | `SuppressTokensAtBegin` (#16) et `TopHLogitsWarper` (#18) existent entre nos étapes, non nommés | **ACCEPTÉ.** Inertes pour cette configuration (les deux clés sont absentes), mais `TopH` **casse l'adjacence Temperature → TopK** s'il est un jour armé. À nommer explicitement. |
| A5 | `applyTopK` : `max(top_k, min_tokens_to_keep)` et le clamp au vocab manquent à la signature | **ACCEPTÉ.** Invisible en production (`min_tokens_to_keep = 1`), donc l'écart n'apparaîtrait **que** dans la fixture — au pire endroit : l'implémenteur chercherait la cause hors de son champ de vision. |
| A6 | Le `scatter` de `TopPLogitsWarper` prend `sorted_indices_to_remove` comme tenseur de base | **ACCEPTÉ** comme note d'implémentation. Fonctionnellement neutre, mais à ne pas transposer naïvement. |

## Findings acceptés — contrôles (relecteur « falsifiabilité »)

| # | Finding | Décision |
|---|---|---|
| B1 | **RP-3 est vacueux** : son critère (`n_match == n_total − 1`, mismatch @47) est *exactement* le résultat que GC7 a mesuré hier **sans penalty dans le code** | **ACCEPTÉ — le plus grave.** Une penalty morte (flag non branché, bitset jamais peuplé) passerait le gate. C'est le motif que GC5 avait servi à éliminer, pré-enregistré comme succès. Correction : réintroduire le **mordant pré-calculé** de la rév. 4 (`hamming ≥ 3`, publié **avant** le run GPU) et un compteur `n_penalty_touched > 0`. |
| B2 | RP-3 échouerait **aussi à tort** ; l'ε de quasi-tie n'est appliqué par **aucun** critère | **ACCEPTÉ.** Le critère est positionnel : il **exige** un mismatch, alors que la variante B donne 200/200 — c'est mot pour mot la leçon écrite de GC7. Et la penalty **comprime les marges** (÷1,15), donc elle peut déplacer @47 *et* créer un quasi-tie ailleurs. Correction : critère **sur la marge, pas sur la position**, avec la liste des positions à risque pré-calculée sur l'oracle seul. |
| B3 | **SM-0 ne peut pas échouer sur ce qu'il vise** : avec `top-k 1` la distribution est une masse de Dirac | **ACCEPTÉ.** Le gate est satisfait à l'identique si la seed est ignorée, si le RNG rend toujours 0, ou si `sample()` n'est jamais appelé. Correction : compteurs `n_draws > 0` et `n_survivors` publiés, assertion dure `logits[token_tiré] > -inf`, et un régime dégénéré **non trivial** obtenu par top-p (qui exerce `applyTopP` au lieu de le court-circuiter). |
| B4 | **RP-PONT tourne là où les deux chemins ne peuvent pas diverger** | **ACCEPTÉ, et c'est le finding le plus utile.** GC2 a mesuré `n_suppress_hits = 0` sur les témoins 48 et 124 — or la suppression est *précisément* ce qui distingue les deux sélecteurs (F13). Correction adoptée : faire tourner **les deux sélecteurs sur le même vecteur de logits, au même step, dans le même processus**. Insensible à la bistabilité **par construction**, et exerçant chaque step. |
| B5 | `n_full_path > 0` est un détecteur de vacuité **trop faible** (satisfait par 1 step sur 124) | **ACCEPTÉ.** Même motif que le détecteur vacueux de GC3, d'un cran plus haut. Correction : `n_steps_compared == n_generated`, plus des compteurs par branche. |
| B6 | **SM-5 ne peut ni réussir ni échouer** | **ACCEPTÉ.** Branche 1 impossible (deux RNG distincts) ; branche 2 sans métrique ni seuil. Correction : comparer des **ensembles de survivants** (déterministe) et des probabilités renormalisées, pas des séquences. |
| B7 | **χ² : un biais de 5 % ne fait pas FAIL** | **ACCEPTÉ**, recalculé en propre : χ² = **2,78** contre 21,67 critique (df = 9, α = 0,01). Ma preuve de non-vacuité n'aurait pas mordu — et j'aurais été tenté de la requalifier après coup, exactement le mécanisme de requalification en cascade déjà payé. Le seuil réellement détectable à n = 10 000 est **~15 %**. |
| B8 | Le binning à ~10 catégories **anesthésie** le test : une permutation d'ids **dans** un bin est invisible | **ACCEPTÉ.** Or c'est le bug le plus probable de `sample()` (indice trié rendu au lieu de l'id vocabulaire). Correction : `top_k = 10` exactement, pour que les 10 catégories soient **10 ids distincts**. |
| B9 | RP-6 : seuil à 15 % alors que l'effet réel est ~1 % **et** que le bruit du projet est de 2 à 16 % | **ACCEPTÉ.** Un seuil à l'intérieur du bruit ne discrimine rien, et il passerait même si l'implémentation allouait par step — l'interdit qu'il était censé garder. Correction : design **apparié**, 3 runs par bras, médiane, seuil calibré sur l'effet ; et l'interdit d'allocation adossé à un **compteur d'allocations à 0**, pas au débit. |
| B10 | RP-5 : « RSS stable » n'est pas un critère | **ACCEPTÉ.** Aucun chiffre, aucune fenêtre : aucun résultat ne peut le faire échouer. Correction : `max_rss(token 200) − max_rss(token 20) < 5 MiB`, publié. |
| B11 | RP-4 : mutations dont l'**antécédent n'est pas mesuré** | **ACCEPTÉ.** La mutation « signe inversé » est **muette** si tous les tokens de l'historique portent un logit positif. Correction : patron GC1 — compteurs d'antécédent (`n_hist_dup`, `n_penalty_branch_pos`, `n_penalty_branch_neg`) mesurés **avant** les mutations ; une mutation dont l'antécédent est à 0 est **inexécutable**, pas PASS. |
| B12 | L'interdit « ne pas retranscrire » n'est **pas appliqué à la penalty** | **ACCEPTÉ.** `RepetitionPenaltyLogitsProcessor` est appelable comme les warpers : aucune raison de l'exempter de la fixture issue du vrai processor. |
| B13 | Le cas de bord top-p de S3 est **probablement inconstructible** tel que décrit | **ACCEPTÉ.** Discriminer `<=` de `<` exige `cumulative_probs == 1 − top_p` **exactement en f32** ; sur des logits arbitraires ça n'arrive pas, et le cas serait silencieusement remplacé par un « proche de la frontière » qui ne discrimine rien. La construction qui marche (logits **égaux**, `top_p = 0,5`) est en cours de vérification. |
| B14 | Vigilance 5 repose sur un **fait faux** : `x / 1.0` **est** exact en IEEE-754 | **ACCEPTÉ.** La conduite (ne pas instancier le warper à `T = 1.0`, comme HF) reste bonne ; son motif était une supposition non mesurée, dans un document dont l'autorité repose sur « mesuré et non supposé ». Le motif est corrigé, pas la conduite. |
| B15 | Vigilance 3 appliquée à l'envers **tue SM-3 et SM-4** | **ACCEPTÉ.** Si la comparaison porte sur la capture complète, l'écho `seed=<n>` rend les N sorties trivialement distinctes ⇒ SM-4 ne peut pas échouer ; et un horodatage ⇒ SM-3 ne peut pas réussir. Correction : comparer la **seule ligne `generated`**, et contre-prouver dans les deux sens. |

## Findings acceptés — complétude

| # | Finding | Décision |
|---|---|---|
| C1 | **`top-k 1` n'est pas déterministe sous ma propre F9** | **ACCEPTÉ, et je ne l'avais pas vu.** Avec `k = 1`, une égalité exacte au rang 1 laisse ≥ 2 survivants. Le « régime neutre » que je présentais comme parfaitement déterministe ne l'est donc pas dans le cas même que F9 déclare non hypothétique. Le compteur de ties exacts (que la rév. 4 avait et que ma rév. 1 avait supprimé) doit revenir. |
| C2 | **Aucune surface CLI ni gestion d'erreur** pour les 5 nouveaux paramètres | **ACCEPTÉ.** La rév. 4 consacrait un §8 entier à ce mode d'échec pour **un seul** paramètre, avec la règle « garde en **acceptation** (`p > 0 and isFinite(p)`), jamais en rejet, sinon `NaN` passe ». Le trou est rouvert sur `temperature`, `top_p`, `top_k`, `seed`. Point non tranché le plus coûteux : **que fait `--temperature 0`** ? |
| C3 | La garde `suppress.len + 1 > TOP_K` : **décision annoncée, pas prise** | **ACCEPTÉ**, vérifié en propre : la garde vit dans `gencfg.fromLists` (`gencfg.zig:207`) et **refuse de charger la politique**, avant tout choix de chemin. Mon §4.4 (« sans objet sur le chemin B ») est **faux**, et la rendre conditionnelle casserait GC4(c), gate vert et taggé. |
| C4 | RP-1 affaibli jusqu'à la quasi-vacuité ; son coût n'est plus consommé par personne | **ACCEPTÉ.** « Un run aboutit » ne peut guère échouer, et à 69,75 s/token un run de 48 tokens coûte ~56 min de GPU que rien en aval n'utilise (RP-3 est teacher-forcé). Soit on restaure la revalidation 48/48, soit on retire RP-1 — et on le dit. |
| C5 | `--ignore-prompt` a **disparu en silence** (champ, flag, directive, et le contre-test RP4-(c)) | **ACCEPTÉ.** L'en-tête déclarait le design de la penalty « valide sans modification » : l'implémenteur ne peut pas savoir si la fonctionnalité est dedans ou dehors. |
| C6 | Les **ancrages de ligne** de la rév. 4 §3.2 sont périmés depuis le merge | **ACCEPTÉ avec réserve** (voir nuances). |
| C7 | L'**état du RNG** et sa remise à zéro par prompt ne sont pas spécifiés | **ACCEPTÉ.** `seedé par (seed, step)` : `step` global ou par prompt ? En `--repl`, R12 impose la remise à zéro du compteur de suppression mais ne dit rien du RNG. SM-3 pourrait être vrai en one-shot et faux en session. À trancher, d'autant que ce chantier ajoute **plus** d'état inter-prompts (bitset, buffer, RNG). |

## Findings nuancés — ce que je ne prends pas au pied de la lettre

| # | Finding | Nuance |
|---|---|---|
| N1 | « HF upcaste en fp32 avant les processors ⇒ le forward fp32 de RP-1 est payé pour autre chose » | **Juste sur le fait, incomplet sur la conclusion.** Le fait est vérifié (`_sample` : `.to(copy=True, dtype=torch.float32)`). Mais le forward fp32 a dans **ce projet** un second motif, indépendant : l'oracle bf16 **fabrique des ties artificiels** qui rendent les verdicts d'argmax illisibles (leçon J2, requalifications en cascade). Les deux motifs coexistent. Ce qui doit être corrigé, c'est **mon raisonnement écrit** — « HF fait un forward fp32 » est faux — pas la décision d'utiliser fp32. |
| N2 | « HLO byte-identique est vrai par construction ⇒ invariant qui tue le contrôle » | **Juste, mais le gate se garde.** Le relecteur le concède lui-même : il détecte une fuite de RNG in-graph, ce qui est précisément le risque de la phase 2. Ce qui manque n'est pas le gate mais sa **fraîcheur** : pré-enregistrer le vidage du répertoire de dump, le mtime postérieur à l'édition, le nombre de fichiers et le volume — comme GC0 l'a fait. Sans quoi il compare le témoin à lui-même. |
| N3 | « Les ancrages de ligne de la rév. 4 sont périmés » (C6) | **Vrai, mais ce n'est pas un défaut de la spec en revue.** Corriger les ancrages d'un document daté du 27 juil reviendrait à réécrire une archive. Traitement retenu : la nouvelle spec porte une **table de correspondance des ancrages** (ancien → nouveau numéro de ligne), et déclare les numéros de la rév. 4 **indicatifs**. La règle anti-off-by-one (`hist.append(fed)` en tête d'itération), elle, reste valide et doit être **répétée** dans la nouvelle spec, parce qu'elle corrige un vrai bug. |

## Ce que les gates ne couvriront toujours pas — à écrire dans la spec

Recensé par le relecteur « falsifiabilité », accepté intégralement :

1. `applyTemperature` n'est validé de bout en bout **nulle part** (A3) — couverture = fixture host seule, sauf à ajouter un gate teacher-forcé à `T ≠ 1`.
2. `applyTopP` n'est exercé **sur GPU** par aucun gate qui puisse échouer.
3. L'**équivalence des deux mécanismes de suppression** (F13) — le risque nº 1 du design — n'était exercée par aucun gate (corrigé par B4).
4. Aucune assertion « le token tiré n'était pas filtré » (`logits[t] > -inf`) : l'invariant le plus simple et le plus discriminant du sampler.
5. **Interaction penalty × EOS** : pénaliser un id EOS présent dans l'historique modifie sa probabilité, donc **l'arrêt**. La dette « équivalence de l'arrêt » existe déjà ; la penalty l'**aggrave**, et ce n'était pas dit.
6. La **reproductibilité des témoins 48 et 124** n'avait jamais été mesurée alors que trois gates y adossaient une règle d'arrêt dure — **campagne en cours**, résultat à intégrer.
