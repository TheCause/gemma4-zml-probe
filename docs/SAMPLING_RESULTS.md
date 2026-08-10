# Sampling (phase 2) — résultats

> **Statut : LES 5 GATES DE LA PHASE 2 SONT VERTS** — `S2-U`, `S2-PONT`, `S2-D`, `S2-R`, `S2-G`, tous taggés,
> plus la mesure publiée `M-COUT`. Ce qui n'est pas couvert est écrit au §5 (9 dettes), et ce qui
> n'y porte pas de chiffre n'a pas été mesuré.
>
> **Mise à jour du 10 août 2026 — 3 gates de plus, dettes D1 et D2 SOLDÉES** : `G-D0`, `G-D1`,
> `G-D2` (couverture GPU de `applyTopP` et `applyTemperature`). Détail, contre-preuves et
> limites au **§7**.
>
> Spec : `docs/superpowers/specs/2026-07-29-sampling-penalty-design.md` (rév. 3) ·
> Plan : `docs/superpowers/plans/2026-07-29-sampling-phase2.md` (rév. 2) ·
> Arbitrage des revues : `docs/superpowers/specs/2026-07-29-sampling-penalty-arbitrage.md`

## 1. Ce que le chantier livre

Le 12B applique `top_k`, `top_p`, `temperature` et un tirage **reproductible à seed fixée**. Avec
la politique de décodage livrée le 29 juil (`suppress_tokens` + 3 EOS), il exécute désormais **les
clés de `generation_config.json` que Google publie** — `do_sample: true, top_k: 64, top_p: 0.95,
temperature: 1.0` — au lieu d'un greedy que Google ne recommande pas.

**Tout est host-side.** Le graphe sort déjà les logits complets ; le chemin B les rapatrie et
applique la chaîne dans l'ordre **mesuré** de HF :
`Penalty(4) → Suppress(15) → Temperature(17) → TopK(19) → TopP(20)`.

**Deux chemins coexistent, et c'est délibéré** :
- **chemin A** — rien d'armé : top-5 rapatrié (~48 octets), `gencfg.select()`. **Strictement le
  code d'avant le chantier.**
- **chemin B** — dès qu'un warper ou une seed est demandé : vecteur complet + chaîne HF.

**Deux conditions d'armement distinctes**, et cette distinction est nécessaire : le **chemin B**
s'arme dès qu'un warper est demandé ; le **tirage** s'arme *seulement* si `--seed` est fourni. Les
confondre aurait fait que `--top-k 1` — le régime neutre du gate-pont — n'active pas le chemin B,
et le gate n'aurait eu **rien à comparer**.

## 2. Verdicts mesurés

| Gate | Verdict | Chiffres |
|---|---|---|
| **S2-U** — warpers vs HF, host-only | **PASS** | **10/10** cas (7 par indices, 7 par équivalence : 3), antécédents `topk_déborde=3`, `vocab_réel=3`, exit 0, zéro fuite. **Aucune seconde de GPU.** |
| **S2-PONT** — les 2 sélecteurs, même vecteur, même step, même processus | **PASS** | **454 steps comparés** sur 3 runs, **0 désaccord**, **0 égalité exacte**. Antécédent non vide (`n_suppress_hits = 1`). Non-régression : témoin 48 **bit-identique**, témoin 124 identique sur ses **110 premiers ids** |
| **S2-D** — distributionnel | **PASS** | **χ² = 7,9333** contre **21,665994** critique (df=9, α=0,01, n=10 000), k = **10 ids distincts**, théorique **torch indépendante**. Non-vacuité : biais half-split 10 % ⇒ **χ² = 109,63, FAIL** |
| **S2-R** — reproductibilité et non-vacuité du RNG | **PASS** | même seed ⇒ histogramme **md5-identique** (host) et ids **identiques** (génération) · 5 seeds ⇒ **5 histogrammes distincts** · 3 seeds ⇒ **3 sorties distinctes** |
| **S2-G** — le graphe n'a pas bougé | **PASS** | md5 `297679847aa04b719942d75d093adf2b` **identique** au témoin figé AVANT la première ligne de code, tout le chemin B ayant été ajouté entre-temps. **Fraîcheur prouvée** : dump neuf, 510 fichiers, 18 956 Ko, mtime du jour |
| **M-COUT** — surcoût du chemin complet | *mesure publiée, sans PASS/FAIL* | bloc `{D2H + warpers + sélection}` chronométré **in-process** : **moyenne 3 730 µs/step**, max 5 556 µs, n = 88 ⇒ **3,5 % d'un step** (~106 000 µs) |

### Lecture de M-COUT — et une erreur qu'il a révélée dans la spec

La mesure vaut **~27× la borne basse C++** (0,13 %). **J'avais attribué l'écart au build `dbg`.
La mesure a réfuté cette hypothèse** : en `opt`, le bloc coûte **3 828,8 µs** — *autant* qu'en
`dbg` (3 730,1) — alors que le binaire passe de **451 Mo à 40 Mo**.

Décomposition (D9 résolue) : **D2H + copie 1 514 µs** · **warpers 2 268 µs** · total 3 796 µs =
**3,6 % d'un step**. Le coût est **structurel** — parcourir 262 144 logits trois fois, à ~0,96 ns
par élément-opération, ce qui est un ordre de grandeur normal — et non un artefact de compilation.

Deux leviers restent ouverts et **non faits** : `toSliceAlloc` **alloue 1 Mo à chaque step**
(l'interdit §5 que ce code viole — dette **D10**), et le D2H plafonne à **0,65 Go/s**, soit ~48×
sous le PCIe 4.0 théorique, faute de mémoire *pinned*.

⚠ **Surtout, cette mesure a révélé que la base de calcul de la spec était fausse d'un facteur 12.**
La table F16 rapportait les coûts à « un step de 9 091 µs » — ce qui correspond à **110 tok/s**,
alors que ce modèle fait **9,x tok/s** (~106 000 µs/step). L'erreur a traversé **deux révisions et
trois relecteurs** : une sonde annonçait « 110-113 tok/s », je l'ai reprise en corrigeant son
arithmétique interne (183 % → 177,95 %) **sans vérifier la plausibilité de la base**, alors que
« 9 tok/s » est écrit partout ailleurs dans ce dépôt. **Raffiner un chiffre faux ne le rend pas
juste.** Conséquence utile : un tri complet coûterait **15,2 %** d'un step et non 178 % — cher,
mais pas rédhibitoire.

⚠ **Ne pas mal lire les compteurs S2-PONT d'un run ARMÉ.** Le run de `M-COUT` affiche
**21 désaccords** : c'est **attendu**, le sampling y diverge légitimement de l'argmax
(`top_k=64, top_p=0,95, seed=42`). Le gate `S2-PONT` n'est valide qu'en **régime neutre** — c'est
là qu'il a rendu 454 steps et 0 désaccord.

## 3. Ce que les gates ont attrapé — et ce qu'ils ont corrigé chez moi

Un gate qui ne trouve rien n'a pas prouvé qu'il fonctionne. Ceux-ci ont été **vus échouer** :

1. **`S2-U` contre-prouvé par deux mutations.** `<` → `<=` dans `applyTopK` : **6/10**, et le
   détecteur de non-vacuité mord (« aucun cas où top_k laisse plus de k survivants ») — la mutation
   supprime précisément le phénomène que le gate exerce. Tri de `applyTopP` passé en descendant :
   **5 cas** tombent, dont un à **262 144 survivants au lieu de 464**.
2. **Mon « cas phare » ne discriminait pas ce que je croyais.** `topp_disjoint_8egaux` (8 logits
   égaux) **n'a pas mordu** sous la mutation du tri : sur des valeurs toutes égales, le comparateur
   rend `false` dans les deux sens et `sortUnstable` laisse le même ordre — ascendant et descendant
   y sont indiscernables. Le cas reste valide comme test de **conformité**, mais ce sont les cas à
   logits **distincts** qui discriminent le sens du tri.
3. **Le premier run de `S2-PONT` avait un antécédent VIDE** (`suppress a mordu 0 fois`) : tombé sur
   la variante B du prompt bistable, où les deux chemins **ne peuvent pas** diverger. Relancé ;
   l'antécédent est tombé au run suivant. **Un antécédent obtenu au 2ᵉ essai reste un antécédent,
   à condition de le dire.**
4. **Mon premier `S2-R` ne discriminait rien.** Sur « capital of France », le modèle produit 2
   tokens sur une distribution très piquée : deux seeds y donnent **légitimement** la même sortie,
   et ma contre-preuve échouait sur un RNG parfaitement correct. **Antécédent trop faible**, pas
   défaut du code. Re-dimensionné à 60 tokens sur un prompt ouvert.
5. **La garde `CudaRequired` a mordu** : un build intermédiaire sans `--@zml//platforms:cuda=true`
   a fait **refuser** le run plutôt que de replier sur CPU en silence — sur un gate à règle d'arrêt,
   un repli aurait produit des chiffres faux.
6. **Correction d'un énoncé à moi, par la mesure** : « un biais de 5 % ne mordrait pas » vaut pour
   l'injection **mono-catégorie** (λ divisé par k−1, χ² = 2,78). En **half-split**, 5 % mord déjà
   (χ² = 36,70) — c'est tout l'intérêt de cette forme d'injection.

## 4. Périmètre de la claim

« Le 12B applique `generation_config.json` » signifie désormais **6 clés sur 8** : `suppress_tokens`
et `eos_token_id` (chantier du 29 juil), plus `do_sample`, `top_k`, `top_p`, `temperature` (ce
chantier). Ne restent hors périmètre que `bos_token_id` et `pad_token_id`, sans objet au décodage.

⚠ La claim « == HF » garde sa portée d'origine : **même argmax sur les logits bruts**. Voir
`docs/GENERATION_CONFIG_RESULTS.md` §2.

## 5. Dettes — ce qui n'est PAS couvert

| # | Dette | Motif |
|---|---|---|
| **D1** | ~~**`applyTopP` n'a AUCUNE couverture GPU**~~ **SOLDÉE (10 août 2026)** | Gate **G-D1** vert : **386 steps GPU armés, 0 désaccord** avec une référence écrite AUTREMENT (tri descendant + cumsum `< p` en f64, contre tri ascendant + `cum <= 1-p` en f32). **Antécédent plein** : top-p a coupé à **386 steps sur 386** (23 267 ids retranchés) — le gate n'est pas passé à vide. **0 cas frontière** (`|cum−p| < 1e-9`) : l'accord est structurel, pas un coup de chance numérique. §7 ci-dessous |
| **D2** | ~~**`applyTemperature` n'est pas exercé de bout en bout**~~ **SOLDÉE (10 août 2026)** | Gate **G-D2** vert : la ligne s'exécute enfin sur GPU (**386 steps** à `T=0.7`), et **les 2 mutants MORDENT** — `x/t` diffère de `x*(1/t)` sur **9 414 388** logits, et l'ordre muté `TopK→TopP→Temp` change **584 ids sur 138 steps**. ⚠ Ce que G-D2 **ne** prouve pas : cf §7, l'égalité à une « référence indépendante » y serait tautologique |
| **D3** | ~~L'interdit « aucune allocation par step » n'a pas de gate porteur~~ **SOLDÉE (30 juil)** | Gate **AL-0/AL-VAC** (compteur toujours actif, arbitrage B9) + **AL-RSS** (B10) — `docs/D10_RESULTS.md` |
| **D4** | **Équivalence de l'arrêt runner ↔ HF** : prouvée par aucun gate | Héritée du chantier précédent, **aggravée** par la penalty (pénaliser un id EOS modifie l'arrêt) |
| **D5** | **`RP7` (« la récitation est-elle levée »)** : suspendu | Le symptôme d'origine n'a jamais été reproduit. Appartient à la spec du 27 juil : à arbitrer là-bas |
| **D6** | **E2B non couvert** | Ses runners ne sortent pas les logits du graphe |
| **D7** | **Custody des coûts F16** | Mesures prises GPU non vierge : absolus à requalifier |
| **D8** | **Tie-break `argmax` host vs `topK` in-graph** non prouvé équivalent | `S2-PONT` le **publie** (`n_exact_top_ties`, observé à **0**) au lieu de le supposer résolu |
| **D9** | ~~mesuré en `dbg`~~ RÉSOLUE — ⚠ **conclusion « structurel » RECTIFIÉE le 30 juil** | La commande du build « opt » de D9 est perdue, et `-c opt` seul laisse le frontend Zig en **debug** (mode rules_zig indépendant, défaut debug). Au mode **prouvé** `ReleaseFast` : warpers **2 268 → 261 µs** sans changement de code. Le « coût structurel » était en partie un artefact de build — `docs/D10_RESULTS.md` §5 |
| **D10** | ~~`toSliceAlloc` alloue 1 Mo par step~~ **RÉSOLUE (30 juil)** | `toSlice` direct dans `work` : **0 alloc, 0 copie**, gate AL-0 quatre zéros. Bloc chemin B **3 796 → 908,7 µs** (0,86 % d'un step). ⚠ L'hypothèse « faute de pinned » est **RÉFUTÉE par A/B** (ON 441,7 vs OFF 447,1 µs) : le « 0,65 Go/s » mesurait les allocs+copies, le transfert fait 2,37 Go/s — `docs/D10_RESULTS.md` |

## 6. Divergences délibérées avec HF, déclarées

- **`T_MIN = 1e-30`** : HF n'exige que `t > 0` et accepte donc `1e-45`, qui fait déborder la
  division en f32 et produit des **`NaN`** (`inf - inf` au softmax). Les logits étant bornés par le
  softcap 30, ce plancher laisse 8 ordres de grandeur de marge. **Divergence assumée**, pas une
  reproduction.
- **`--temperature 0`** : **rejeté comme HF** (qui lève une `ValueError`), avec un message
  renvoyant vers `--top-k 1` pour du greedy déterministe.
- **Gardes en acceptation** (`!(p > 0 and isFinite(p))`) et non en rejet : `p <= 0 → rejet`
  laisserait passer **`NaN`**, toute comparaison avec `NaN` étant fausse.

## 7. D1/D2 — couverture GPU des warpers (10 août 2026, gates G-D0/G-D1/G-D2)

> Spec pré-enregistrée : `docs/superpowers/specs/2026-08-10-d1d2-gpu-coverage.md` (committée
> **avant** la première ligne de code). Preuves : `docs/evidence/d1d2/`.

**Le trou que ce chantier bouche.** `S2-PONT` compare deux *sélecteurs* et n'est valide qu'en
**régime neutre** — or son régime neutre est `--top-k 1`, qui **court-circuite précisément
top-p**. `applyTopP` n'avait donc jamais tourné sur GPU sous contrôle, et `applyTemperature`
jamais tourné du tout (la config Google porte `temperature: 1.0`, et le runner suit HF en
n'instanciant pas le warper à `T == 1.0`).

**Méthode.** Pont in-process en régime **armé** (`--gate-d1d2`) : `applyTopP` est confrontée, sur
le **même vecteur, au même step, dans le même processus**, à une référence **écrite autrement** —
tri **descendant** + cumsum exclusive `< p` en **f64**, contre tri ascendant + `cum <= 1-p` en f32.
Insensible à la bistabilité par construction. Le f64 n'est pas un luxe : sommer des probabilités
des plus grandes vers les plus petites ne donne pas le même f32 que l'inverse, et une référence
f32 aurait produit du bruit indistinguable d'un vrai désaccord.

| Gate | Prédiction | Mesuré | Verdict |
|---|---|---|---|
| **G-D0** | md5 HLO identique au témoin pré-code + `ALLOC-LOOP: alloc=0` | md5 **`297679847aa04b719942d75d093adf2b`** avant **et** après ; `alloc=0 resize=0 remap=0 free=0 bytes=0` sur les 2 runs ; `mode=ReleaseFast` | **PASS** |
| **G-D1** | ≥ 300 steps, **0 désaccord**, antécédent ≥ 30 steps avec coupe | **386 steps, 0 désaccord**, coupe à **386/386** steps (**23 267** ids retranchés), **0 cas frontière** | **PASS** |
| **G-D2** | ≥ 300 steps avec température appliquée, **les 2 mutants mordent** | **386 steps** à `T=0.7` ; mutant (a) **9 414 388** logits où `x/t ≠ x*(1/t)` ; mutant (b) **584 ids sur 138 steps** | **PASS** |

**Ce que les contre-preuves ont établi.** Le refus a été **VU** : `--gate-d1d2` sans régime armé
rend `GateD1D2NotArmed` — le gate refuse de passer à vide plutôt que de compter 0 step. Et le
**dépouilleur lui-même a été vu condamner** : 5 mutants injectés dans les logs réels (md5 différent,
1 désaccord, antécédent vide, chacun des 2 mutants neutralisé) le font passer au rouge, tandis que
le cas nominal reste vert — la question posée dans les deux sens
(`docs/evidence/d1d2/contre_preuves_depouilleur.txt`).

**⚠ Ce que ce chantier NE prouve PAS, et qui reste écrit.**

1. **G-D2 est plus mince que G-D1, par nature.** Il n'existe qu'une façon d'écrire une division
   f32 : une « référence indépendante » y comparerait le code à lui-même. G-D2 établit que la
   ligne s'exécute, qu'elle **divise** (mutant a) et que **l'ordre de la chaîne compte**
   (mutant b) — rien de plus. C'est écrit ainsi dans la spec, avant la mesure.
2. **Deux mutants « évidents » se sont révélés VACUS à l'analyse**, et ont dû être remplacés
   avant d'être écrits dans le code : (i) déplacer la température *après top-k* ne change rien
   (diviser par `T > 0` est monotone, et le critère de top-k est un pur ordre) — d'où le mutant
   retenu, *après top-p* ; (ii) exiger que la trajectoire diffère de celle du run `T=1.0` est
   **impossible en régime argmax**, pour la même raison de monotonie. Les deux auraient produit
   un gate vert sans contenu.
3. **La mesure `M-COUT` du même run est invalide** : le pont travaille dans la fenêtre
   chronométrée. Le binaire l'annonce lui-même par un `log.warn` plutôt que de laisser le chiffre
   être recopié ailleurs comme comparable.
4. **E2B toujours hors périmètre** (dette D6), **tie-break D8** non résolu, ~~**phase 1 penalty**
   suspendue~~ → **livrée le 10 août 2026, cf §8**.

## 8. Phase 1 — repetition penalty (10 août 2026)

> Spec : `docs/superpowers/specs/2026-07-27-sampling-repetition-penalty-design.md` (rév. 4) ·
> Plan d'exécution : `docs/superpowers/plans/2026-08-10-dettes-restantes-penalty.md`
> (le plan du 27 juil prédate 5 chantiers mergés et n'était plus exécutable tel quel).

La penalty s'insère **en tête** de la chaîne host-side existante, à la place que HF lui donne
(`Penalty(4) → Suppress(15) → …`). Aucun changement de graphe, aucune allocation par step :
l'historique et le bitset de déduplication sont alloués **une fois** par run.

### 8.1 Verdicts

| Gate | Verdict | Chiffres |
|---|---|---|
| **RP1** — `applyRepetitionPenalty` vs le processor HF, host-only | **PASS** | **4/4 penalties bit-identiques à 0 ULP** (512 valeurs chacune, `{0,8 ; 1,0 ; 1,15 ; 1,5}`). Historique 9 ids dont **6 distincts** (dédup exercée), **3 logits < 0 et 3 ≥ 0** (les deux branches de signe), tie-break = **premier** des 3 ex æquo. Aucune seconde de GPU. |
| **RP0** — le graphe n'a pas bougé | **PASS** | md5 `297679847aa04b719942d75d093adf2b` **dans les deux régimes**, penalty désarmée ET `--repetition-penalty 1.15` armée. `ALLOC-LOOP: alloc=0` sous penalty (17 et 52 steps). |
| **RP2** — non-régression penalty neutre | **PASS**, référence **requalifiée** | `main` **recompilé sans une ligne du chantier** produit les mêmes ids que le code pénalisé neutre : 4 runs, 3 binaires distincts, tous `06a5953f…`. ⚠ Le témoin figé en début de chantier s'est révélé non reproductible — cf `docs/evidence/penalty/FINDING_temoin_ids_non_reproductible.md`. |
| **RP5** — état par prompt en `--repl` | **PASS** | 2 passes du même prompt sous `:penalty 1.15` ⇒ **textes identiques**, `n_penalty_touched=60` aux deux, **`hist_len=61` aux DEUX** (sans re-seed : 93). Confirmé sur **20 passes** (20 textes identiques, `hist_len=53` à la 1ʳᵉ comme à la 20ᵉ). |
| **RP6** — directives du repl | **PASS** (a/b/c/d) | (a) directives seules ⇒ **0 génération, 0 ligne `PENALTY:`** ; (b) `:penalty` agit au prompt **suivant** (1 seule ligne `PENALTY:` sur 2 générations) ; (c) le texte produit sous `:penalty 1.15` est **mot pour mot** le `reponse_hf` du manifest oracle 1.15 ; (d) `0, -1, nan, inf, abc`, valeur absente, `:ignore-prompt maybe` et directive inconnue ⇒ message par cas, **valeur inchangée**, **session vivante**. |
| **RP3** — le runner sous penalty produit **les ids de HF** | **PASS** | **48/48** pour `rp=1,15` **et** pour `rp=0,8`, prompt vérifié **littéralement** (29 ids), `n_penalty_touched = 76/76`, `alloc=0`. **Non-vacuité exercée** : la même fixture 1,15 **sans** armer la penalty ⇒ `A1 FAIL 17/48`, 1er mismatch à `gen=17` — exactement la position de 1ʳᵉ divergence du mordant. |
| **RP4** — corruptions vues FAIL | **2/3 mordent**, la 3ᵉ **VACUE et déclarée** | (a) branches de signe échangées ⇒ **FAIL 15/48** (@11) · (b) déduplication retirée ⇒ **FAIL 34/48** (@34) · (c) `ignore_prompt` inversé ⇒ **A1 PASS 48/48, ne mord pas**. Code sain restauré et re-vérifié PASS. Instruction de (c) en 8.5. |
| **RP7** — récitation | *mesure publiée, sans PASS/FAIL* (décision D4) | Sur 48 tokens HF : `rp=1,0` ⇒ plus long n-gramme répété **2**, bigrammes répétés **1**, distincts 39/48 · `rp=1,15` ⇒ **1 / 0 / 38** · `rp=0,8` ⇒ **4 / 4 / 34**. À **200 tokens et longueur égale** (runner) : OFF ⇒ **4 / 13 / 3**, distincts 123/200 · ON `rp=1,15` ⇒ **2 / 6 / 0**, distincts **135/200**. La métrique bouge **dans les deux sens** attendus. |
| **M1** — coût | *mesure publiée, sans PASS/FAIL* | Chemin B armé **des deux côtés** par `--top-k 1` (sinon `M-COUT` n'est pas publié côté OFF) : **699,7 µs/step** sans penalty vs **711,1 µs/step** avec ⇒ **+11,4 µs, +1,6 %**. L'écart est dans le **D2H** (436,3 → 453,8) ; les **warpers ne bougent pas** (248,2 → 242,8, en baisse). Le surcoût de la penalty est **sous le plancher de résolution**. |
| **Round-trip dump/restore** | **PASS** | dump 24 tokens : `hist_len=53 = 29 + 24` · restore 8 tokens : `hist_len=60 = 52 + 8`, avec **`step_next=52` lu indépendamment au manifest**. Sans le seed de reprise, `hist_len` vaudrait 8. |

⚠ **Ne pas comparer ces 699 µs aux 3 730 µs du §2.** Ce n'est pas une accélération : le régime
diffère. Le `M-COUT` historique tournait sous `top_k=64, top_p=0,95`, où `applyTopP` **trie** son
sous-ensemble ; ici `--top-k 1` seul, et `applyTopP` sort immédiatement (`p >= 1,0`). Les deux
chiffres mesurent le même bloc sur des charges différentes.

### 8.2 Mordant des fixtures oracle (l'antécédent de RP3)

Produites par le VRAI processor HF (`RepetitionPenaltyLogitsProcessor`), 48 tokens,
`--compute-fp32`, prompt de référence. `n_penalty_touched = 49` pour les deux penalties.

| Comparaison | Hamming sur `fed` | 1ʳᵉ divergence |
|---|---|---|
| `rp=1,0` vs `rp=1,15` | **31 / 48** | position 17 |
| `rp=1,0` vs `rp=0,8` | **25 / 48** | position 11 |

Le plan exigeait ≥ 3 : le gate ne peut pas passer à vide.

### 8.3 Ce que les gates ont attrapé

1. **`:penalty` sur des buffers jamais alloués.** `work`/`scratch` n'étaient alloués que si
   `pathArmed()` était vrai **au lancement** ; la directive armait le chemin B en cours de
   session ⇒ `chemin B : logits 1048576 octets != work 0 octets`. Pire : les `defer` de
   libération **ré-évaluaient** `pathArmed()` — devenu vrai après un `:penalty`, l'un d'eux
   aurait libéré un `scratch` jamais initialisé. Corrigé par une condition **figée une fois**
   qui pilote allocation *et* libération. Règle qui en sort : *la condition de libération doit
   être la MÊME EXPRESSION que celle d'allocation, pas une qui lui ressemble.*
2. **Un témoin d'ids n'est pas une référence inter-fenêtre.** Détail au §8.4.
3. **L'assert softcap et la penalty.** Côté oracle, appliquer la penalty avant les asserts
   ferait sauter `max_abs <= 30` (un logit positif divisé par 0,8 monte à 37,5) — alors que cet
   assert parle du **modèle**, pas de la chaîne. La penalty s'applique donc *après* les asserts
   sur les logits bruts et *avant* `suppress_tokens` : l'ordre de HF, obtenu sans affaiblir
   l'assert.

### 8.4 La requalification de RP2, en toutes lettres

RP2 a d'abord **échoué** : 148 ids sur 200 différaient du témoin figé quelques dizaines de
minutes plus tôt. Ce n'était pas le code. `main` recompilé, sans une ligne du chantier, produit
**exactement les mêmes ids que le code pénalisé** ; trois binaires distincts s'accordent et
diffèrent tous du témoin, à md5 HLO identique.

La cause a été établie par une mesure **indépendante** : le manifest de l'oracle `rp=1,0`
publie `min_margin = 0,004589 @ gen=47` — la marge minimale de la trajectoire tombe **au token
exact** où la divergence commence, avec deux candidats séparés de 0,0046, soit **six fois moins**
que la marge min historique du repo (0,0279). Un écart d'exécution infime suffit à faire
basculer la sélection ; tout le reste est la cascade autorégressive de ce basculement unique.

**Conséquence pour les gates du repo, au-delà de ce chantier** : un témoin d'ids issu d'un
décodage **libre** et **long** n'est pas une référence fiable d'une fenêtre d'exécution à
l'autre. RP2 se juge donc contre un run de `main` recompilé dans la **même** fenêtre — ce qui
contrôle la variable que le témoin subissait. Règle d'instrumentation qui en sort : **capturer
le md5 du binaire en même temps que tout témoin de sortie**, sans quoi on ne peut pas
distinguer « le code a changé » de « le témoin a dérivé » — et le premier réflexe est
d'accuser le code qu'on vient d'écrire.

### 8.5 Dettes ouvertes par la phase 1

- **D4 aggravée** : pénaliser un id EOS modifie l'arrêt. Écrit, non gaté ici.
- **RSS du mode résident** : sur 20 prompts, +1 268 KiB avec penalty, **+1 196 KiB sans**
  (contre-test, mêmes prompts, même binaire) — pente 66,7 vs 62,9 KiB/prompt. Le critère
  « ≤ +1 Mo » du plan **n'est pas tenu**, et l'écart imputable à la penalty est de **+72 KiB** :
  le dépassement vient d'une dérive de base du repl, **antérieure à ce chantier**.
- **`--ignore-prompt` + `--load-cache`** : refusé (`IgnorePromptWithLoadCache`), refus exercé.
  Sous reprise, `ids` vaut `ids_fed` complet et « le prompt » n'y est plus une notion définie.

### 8.6 RP4 (c) est VACUE sur la trajectoire de référence — instruit, pas accepté

La corruption (c) inverse `ignore_prompt` au point d'insertion. Sous RP3 (`ignore_prompt=false`),
elle revient à **ne pas pénaliser le prompt**. Elle a produit **`A1 PASS 48/48`** : sur cette
trajectoire, pénaliser le prompt ou non **ne change aucun des 48 tokens**. Un mordant nul se
publie et s'instruit — il ne s'accepte pas.

**Ce qui a été mesuré ensuite :**

1. **Confirmation directe** : le code **sain** lancé avec `--ignore-prompt` sur la fixture 1,15
   donne lui aussi `A1 PASS 48/48` (`n_penalty_empty_hist = 29` — les 29 steps de prefill ont
   un historique légitimement vide, exactement le cas prévu par l'exemption de `PenaltyInert`).
   La corruption n'était donc pas « trop faible » : la trajectoire est **indifférente** à cette
   frontière.
2. **Le flag est bien OPÉRANT** : recherche bornée à 3 essais d'un prompt où `--ignore-prompt`
   change les ids — **les 3 mordent** (« Banana banana banana… », « Kaleidoscope… », « Repeat
   after me: quartz… »). Ce n'est donc pas le flag qui est inerte, c'est le prompt de référence
   qui est atypique : sa génération réutilise si massivement ses propres tokens que le prompt
   n'apporte presque aucun id distinct supplémentaire à l'historique.

**Ce qui reste à prouver, et c'est écrit** : que l'historique inclut le prompt **conformément à
HF**. Les points 1-2 établissent que le flag agit, pas que le **défaut** (prompt inclus) est le
bon. Le gate qui le prouverait est une fixture oracle sur un prompt discriminant — voir la ligne
« RP4 (c) » du tableau 8.1 et l'état de cette dette ci-dessus.
