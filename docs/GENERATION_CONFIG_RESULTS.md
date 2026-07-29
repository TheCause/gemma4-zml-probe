# `generation_config` — résultats

> **Statut : CHANTIER EN COURS.** Gates verts : GC0, GC1, GC2, GC3, GC12 (Task 0). En attente :
> GC4, GC5, GC6, GC7, GC8, GC10, GC11. Ce document n'est PAS un rapport de clôture — tout ce qui
> n'y porte pas de chiffre n'a pas été mesuré.
>
> Spec : `docs/superpowers/specs/2026-07-28-generation-config-design.md` (rév. 3) ·
> Plan : `docs/superpowers/plans/2026-07-29-generation-config.md` ·
> Finding d'origine : `docs/FINDING_GENERATION_CONFIG.md` ·
> Finding qui a réécrit le protocole : `docs/FINDING_NONDETERMINISME_TRAJECTOIRE.md`

## 1. Ce que le chantier corrige

Le portage 12B faisait un **argmax nu** et ne s'arrêtait que sur `<turn|>` (106), alors que Google
déclare `"suppress_tokens": [258883, 258882]` et **trois** `eos_token_id` `[1, 106, 50]`.
Conséquence mesurée : le runner émettait `<image|>` (258882) **en greedy**, au milieu d'un texte.

L'écart n'était pas dans le forward — l'A/B du 27 juil l'avait innocenté — mais dans la **politique
de décodage**. Et `69_u8_gen_oracle.py` partageait **exactement le même angle mort** : aucun gate
existant ne pouvait détecter l'écart, puisque l'instrument était aveugle au même endroit que son
sujet.

## 2. Périmètre de la claim — à lire avant tout chiffre

Modèle : `docs/MASKS_INGRAPH_RESULTS.md` §« périmètre ».

- **« ids == HF » reste VRAIE** au sens « même argmax sur les logits bruts » — un critère plus
  strict que comparer deux `generate()`. Les gates historiques mesurent bien ce qu'ils disent.
- **Ce chantier rend vraie la seconde lecture** — « reproduit ce que `generate()` produirait » —
  **pour le 12B en mode libre**, et **seulement** pour lui.
- **Elle reste FAUSSE pour les runners E2B** (dette §6). Écrire l'inverse est interdit.
- **Deux clés sur huit.** `do_sample: true`, `top_k: 64`, `top_p: 0.95`, `temperature: 1.0` ne sont
  **PAS** appliqués. « Le portage applique `generation_config.json` » tout court serait faux : il
  applique `suppress_tokens` et `eos_token_id`. Le log le dit à chaque run, via un segment
  `ignored=[…]` **dérivé des clés présentes** et jamais codé en dur.

## 3. Architecture retenue

La politique est **host-side**, hors du graphe. Le graphe sort déjà un top-5 **trié décroissant**
rapatrié à chaque step ; avec `|S| = 2` supprimés, l'argmax post-suppression est de **rang brut
≤ 3**, donc déjà présent dans ce top-5. La garde `suppress.len + 1 > TOP_K` rend cet argument vrai
**par construction** et non par chance.

Conséquences : `engine.zig` **0 octet**, `G12Step.forward` inchangé, **zéro D2H supplémentaire** —
et GC0 le **prouve** au lieu de le supposer.

`gencfg.TOP_K` est la **déclaration unique** de la constante `5`, qui existait en trois copies (le
type `Top5`, le `topK` in-graph, la boucle de lecture).

## 4. Verdicts mesurés

| Gate | Verdict | Chiffres |
|---|---|---|
| **GC0** — le graphe n'a pas bougé | **PASS** | md5 `module_0001.zml.before_optimizations.txt` = `297679847aa04b719942d75d093adf2b`, **identique** au témoin du 27 juil ; 1 905 860 octets ; 510 fichiers de dump des deux côtés |
| **GC1** — selftest de la politique, host-only | **PASS** | sélection **8/8**, validations **12/12**, découverte **4/4**, **6/6** compteurs de non-vacuité non nuls (top1_supprimé=4, eot_not_in_eos=1, out_of_range=2, begin_suppress=1, eos_empty=2, dedup=1) ; **2 égalités exactes** comptées (instrument de C2) ; exit 0, zéro fuite |
| **GC2** — non-régression | **PASS** | témoin **48** et témoin **124** : ids **bit-identiques**, `n_suppress_hits = 0`, ligne littérale `suppress=[258883,258882]` présente (1 occurrence par run) |
| **GC3** — mordant, critère **statistique** | **PASS** | AVANT (GC12) : `258882` dans **11/20** runs · APRÈS : **0/20** · **p = 1,159e-07** sous H0 · `n_suppress_hits ≥ 1` dans **12/20** · concordance mordant ↔ remplaçant `11814` : **12/12, zéro discordance** |
| **GC5** — l'oracle ne partage plus l'angle mort | **PASS** | deux runs du **même** script (`script_md5 = 8605874b39b5449bb98af064fbcb5567`, identique des deux côtés), **un seul** flag de différence, tous deux `--compute-fp32`. Branche **nue** : `top5_per_pos[57].ids[0] = 258882`, top-5 `[258882, 11814, 3495, 1548, 13186]` et valeurs `[19.1645, 18.9147, 18.6035, 18.0667, 17.9584]` — **reproduction exacte** des chiffres publiés le 27 juil. Branche **politique** : `top5_policy_per_pos[57].ids[0] = 11814`, `top5_per_pos[57]` **inchangé** à `258882`. `n_match` : **199 → exactement 198**, mismatches `@47 (27069 vs 5743)` et `@57 (11814 vs 258882)` |
| **GC7** — équivalence runner ↔ oracle, politique des **deux** côtés | **PASS** (au-delà du critère) | témoin produit APRÈS, teacher-forcé fp32. **Variante A** (@47=5743, suppression mordue) : `n_match` **199/200**, **unique mismatch @47** (27069 vs 5743, marge 0,004536), `n_bites=1` — exactement le critère pré-enregistré. **Variante B** (@47=27069) : `n_match` **200/200, ZÉRO mismatch**, `n_bites=0`. `script_md5` identique, transformers 5.14.1 |
| **GC4** — non-vacuité, 3 sous-tests | **PASS** | **(a)** `--no-gen-config`, N=20 : `258882` **réapparaît dans 13/20** runs (ligne de base 11/20 — dans la fluctuation binomiale attendue), log `GENCFG: DÉSACTIVÉ` présent · **(b)** `suppress_tokens: []` : `258882` dans **5/5**, les 5 runs **vont au bout** (`max-tokens (200)`), log `suppress=[] (aucune suppression)` · **(c)** 5 ids : `error.SuppressListTooLongForTopK`, avec **0 compile** — le refus est bien un *fail-fast* AVANT le GPU |
| **GC6** — multi-EOS exercé en run réel | **PASS** | arrêt à **exactement 35 tokens générés** = la position mesurée (`496` au 35ᵉ, 20/20 runs GC3) ; `stop_reason` **nomme** l'id : `early-stop EOS id=496 (eos={1,106,50,496})` ; token conservé dans la sortie (sémantique HF) ; contre-test acquis : sans le fichier, **20/20** runs vont à 200 |
| **GC10** — `--repl` applique la politique | **PASS** | ligne `GENCFG:` de découverte présente ; **4 générations** ; **chaque prompt affiche « a mordu 1 fois »** — sans remise à zéro on lirait 1, 2, 3, 4 : le compteur est bien réinitialisé par prompt (R12) ; **0** occurrence de `chosen=258882` |
| **GC11** — passe de nuance sur la claim | **PASS** | **8/8** documents de catégorie (i) portent « argmax sur les logits bruts ». Gate **scripté** (`scripts/gc11_claim_scope.sh`), contre-preuve exercée **dans les deux sens** |
| GC8 | **en cours** | test décisif de C1 — coût **mesuré** 69,75 s/token en fp32 (vs 28,57 en bf16), n=60 ≈ 1 h 15 |

**Lecture de GC10.** Le critère pré-enregistré disait « mêmes ids » — un critère
position-par-position sur trajectoire **libre**, c'est-à-dire exactement ce que le finding bistable
interdit. Il a été converti au moment de l'exécuter : ce qui est vérifié est la présence de la
ligne, l'absence de tout `chosen=258882`, et la **remise à zéro du compteur**, trois propriétés
robustes à la bistabilité. La remise à zéro est prouvée par une observation, pas par lecture du
code : quatre prompts mordant chacun une fois, jamais cumulés.

**Lecture de GC7 — pourquoi deux variantes.** Le critère pré-enregistré (« `n_match == n_total − 1`,
unique mismatch `@47` ») suppose **implicitement** que le témoin est de la variante A. Le témoin est
bistable : la variante B porte `27069` @47, qui est **précisément** ce que l'oracle HF calcule. Sur
cette variante, il n'y a **aucun** écart — 200/200. Un gate écrit à la lettre aurait donc échoué sur
le meilleur des deux cas. Les deux sont mesurés et publiés.

Conséquence de fond : **sur ce prompt, la seule divergence runner ↔ oracle est le point de
bifurcation bistable lui-même.** Quand le runner tombe du côté que HF calcule, l'équivalence est
parfaite sur 200 positions. C'est un renfort direct au verdict GC9 du 29 juil (« @47 est un point
d'instabilité numérique, le forward est hors de cause »).

**Lecture de GC5.** La branche nue reproduisant `tf200.json` au chiffre près, la garde du gate est
franchie : l'instrument n'est pas en cause, et l'écart mesuré sur la branche politique est bien
imputable à la politique. Le mismatch `@47` n'appartient pas à ce chantier — c'est le point de
bifurcation du finding bistable, et l'oracle HF y calcule `27069`, c'est-à-dire la **variante B**.

## 5. Ce que les gates ont attrapé

Un gate qui ne trouve rien n'a pas prouvé qu'il fonctionne. Ceux-ci ont trouvé :

1. **Un double-free** dans `discoverAlloc` — `errdefer` **plus** un `free` explicite sur le même
   chemin d'erreur, sur une branche qu'aucun run nominal n'emprunte. Signalé par le DebugAllocator
   au tout premier run de GC1.
2. **Un symlink relatif résolu par `join` au lieu de `resolve`**, rendant
   `…/weights_rel/../snap_rel/…` : le bon fichier, pas la bonne chaîne.
3. **Une contradiction interne de la spec.** §4.2bis annonce `suppress` **trié** ; §4.1
   pré-enregistre la chaîne de log `suppress=[258883,258882]` (l'ordre publié par Google) ; GC2
   grep cette chaîne avec **règle d'arrêt dure**. Trier produisait `[258882,258883]` — même
   ensemble, autre rendu, **gate mort à tort**. Tranché en faveur du critère **pré-enregistré** :
   déduplication à ordre préservé (le tri n'avait aucune utilité, `isSuppressed` étant linéaire sur
   deux éléments). Révélé par le log d'un run GC0, qui ne testait pas cela.
4. **Un détecteur vacueux, dans le dépouillement de GC3 lui-même.** Le premier comptage cherchait
   `258882` dans tout le fichier de log — or la ligne `suppress=[258883,258882]` **contient** cette
   chaîne : le détecteur rendait **20/20 « présent »**, c'est-à-dire qu'il ne pouvait pas échouer.
   Corrigé pour ne lire que la ligne `generated`, puis **contre-prouvé** : le remplaçant `11814`
   est trouvé dans 12/20, exactement les runs où la suppression a mordu.

## 6. Pièges d'exécution à retenir

- **`> log 2>&1` entrelace stdout et stderr et TRONQUE la ligne que les gates grep.** Première
  passe de GC2 : la ligne littérale comptée **0 fois** alors qu'elle avait bien été émise — on
  lisait `50] ignored=[…]` sans son début, la réponse du modèle ayant écrasé le reste. Les deux
  writers du runner sont corrects (fix R1 du mode REPL) ; c'est la **capture** qui mélange.
  **Règle : tout gate qui grep un log sépare les flux** (`> out 2> err`).
- **Un gate de fidélité position-par-position sur trajectoire LIBRE est interdit** (finding
  bistable). GC3 a été converti en critère statistique **avant** de coûter du GPU. GC10 l'a été
  aussi, au moment de l'exécuter : « mêmes ids » y était encore écrit.
- **Une position d'arrêt se mesure avant de servir de critère.** Pour GC6, l'id `496` est au 35ᵉ
  token généré dans **20/20** runs de GC3 : la position est en **amont** de la bifurcation @47,
  donc déterministe — et c'est mesuré, pas supposé.

## 7. Ce qui n'est PAS prouvé, et doit rester écrit

- L'arrêt sur les **vrais** EOS `1` et `50` n'est exercé par aucun corpus réel (GC6 exerce la
  mécanique multi-EOS avec un id fabriqué, pas ces ids-là).
- Le tie-break `sort`/`argmax` en cas d'**égalité exacte au sommet, non supprimée**, reste non
  vérifié. GC1 **compte** les égalités rencontrées (2) ; le cas d'un tie non supprimé au rang 0
  n'est pas exercé.
- Les argmax des steps de **prefill** ne sont écrits nulle part : on ne sait pas si un id supprimé
  y est top-1. La garde `in_gen_phase` rend la question sans effet sur les mesures publiées — elle
  ne l'instruit pas.
- **Le périmètre E2B n'est pas couvert** : les runners E2B ne sortent pas les logits de leur graphe
  (`gen_auto.zig:753`, 6 sorties), et l'E2B n'a de toute façon **pas** de `suppress_tokens` — coder
  `258882` en dur y serait faux.

## 8. Impact sur les gates existants

- **D11** (`70_u8_corrupt.py`) écrit son checkpoint corrompu **à plat**, sans snapshot ni symlink :
  la découverte automatique y échouerait et le contre-test deviendrait inexécutable. Il doit
  désormais passer **`--gen-config <chemin du dq>/generation_config.json`** — un chemin de
  **fichier**, jamais un répertoire.
- **U8/W4g** : la marge `top1−top2` reste publiée telle quelle (instrument de requalification
  pré-enregistré), mais les logs y ajoutent `rank_used`, `chosen` et la **marge décisionnelle** —
  sans quoi, dès que la suppression mord, la marge affichée parlerait de deux tokens dont le
  premier n'est pas celui qui a été retenu.
