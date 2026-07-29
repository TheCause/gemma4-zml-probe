# Finding — la trajectoire libre du 12B n'est pas auto-reproductible

> **Date** : 2026-07-29 · **Statut** : établi (A/B répliqué des deux côtés), quantification en cours
> (N=20) · **Origine** : découvert en Task 0 du chantier `generation_config`, **avant toute
> modification de code**, en cherchant à transformer une prédiction dérivée en fait mesuré.
>
> **Portée** : ce finding **dépasse** le chantier `generation_config`. Il concerne toute claim de
> fidélité établie sur une **génération libre** (roue libre), et la façon dont ce projet construit
> ses A/B avant/après.

## 1. Le fait

Quatre runs du **même binaire**, sur le **même checkpoint**, avec les **mêmes arguments** :

| run | date | `--dump-top5` | id @47 | `258882` @57 | vs run1 |
|---|---|---|---|---|---|
| run1 | 27 juil | non | `5743` | présent | — (témoin `rp0_witness/witness_long_before`) |
| run2 | 29 juil | oui | **`27069`** | **absent** | **1ʳᵉ divergence @47, 149 ids/200 différents** |
| run3 | 29 juil | non | `5743` | présent | **identiques** |
| run4 | 29 juil | oui | `5743` | présent | **identiques** |

Prompt : `"Tell me the story of the number zero, from its invention to modern mathematics."`,
`--max-tokens 200`. Binaire buildé le 26 juil ; source `zml_runner/` inchangée depuis
(`git log -1 -- zml_runner/` = 26 juil 21:07 UTC ; md5 local ≡ VM sur les 4 fichiers).

**Trois runs sur quatre donnent `5743` ; un seul donne `27069`.** Quand la trajectoire bascule à la
position 47, **149 des 200 ids suivants changent** — l'erreur ne se rattrape pas, elle cascade
(greedy).

## 2. Ce que ce n'est PAS

- **Ce n'est pas une dérive temporelle** : run1 (27 juil) et run3 (29 juil) sont **identiques à
  l'octet**, à deux jours et un reboot d'écart.
- **Ce n'est pas le flag d'observation** : l'hypothèse « `--dump-top5` perturbe le sujet mesuré »
  était séduisante (elle est du type que ce projet traque), elle est **RÉFUTÉE** — run4, *avec* le
  flag, reproduit run1. L'A/B a été répliqué **des deux côtés**, ce qui est précisément ce qui
  permet de l'écarter.
- **Ce n'est pas un défaut de la politique de décodage** : aucune ligne de code n'a été modifiée
  quand ceci a été mesuré.

## 3. Ce que c'est

Un **point d'instabilité numérique**. La marge top1−top2 à la position 47 vaut **0,004587**
(mesuré, `/data/tf_probe/tf200.json`) — soit ~5× le bruit résiduel d'**un** logit (`9,365e-4`, gate
U7). À cette échelle, les variations légitimes de l'exécution GPU (ordre de réduction, choix de
kernel par l'autotuning à la compile) suffisent à faire basculer l'argmax.

Le projet documentait déjà « **pas de bit-à-bit entre deux compiles XLA-GPU** » (piège 15,
`docs/DOCUMENTATION.md`; `project_alambic.md` : « autotuning, 2-16 % selon la taille d'effet »).
Ce finding le démontre **sur le token choisi**, et non plus seulement sur les bits d'un logit.

## 4. La conséquence qui compte — la divergence @47 du finding précédent est expliquée

`docs/FINDING_GENERATION_CONFIG.md` §9 rapportait :

> « En roue libre, HF diverge de ZML **dès l'index 47** : ZML `5743 ▁zero` vs HF `27069 ▁humanity`
> — marge 0,004587. C'est du bruit au niveau du tie, mais ce n'est **pas instruit**. »

**Il est maintenant instruit** : ZML produit **lui-même** `27069` une fois sur quatre. La lecture
« ZML diverge de HF à cette position » présupposait que ZML avait *une* réponse à cette position.
**Il n'en a pas.** À cette marge, comparer un run ZML à un run HF ne mesure pas une différence entre
les deux implémentations — cela échantillonne un tirage.

⇒ Le forward n'est **pas** mis en cause par @47. C'est un point où **aucune** claim d'identité
position-par-position n'est énonçable, quelle que soit l'implémentation en face.

## 5. La conséquence méthodologique — un A/B en roue libre n'est pas valide ici

Le chantier `generation_config` avait pré-enregistré (spec rév. 2, §5) :

- **GC3** : « ids 0..56 **bit-identiques** au témoin AVANT » ;
- **GC4(a)** : « `--no-gen-config` reproduit le témoin AVANT **bit-à-bit** ».

Ces deux critères **échouent aléatoirement** — environ 1 fois sur 4 sur l'échantillon observé —
pour une raison **sans aucun rapport** avec ce qu'ils prétendent mesurer. Pire : le mordant du gate
(`258882` @57) **disparaît** quand la trajectoire bifurque en amont, puisque la position 57 n'est
plus atteinte dans le même contexte.

C'est un **contrôle qui peut échouer à tort** — le symétrique des deux leçons déjà payées par ce
projet (« un contrôle qui ne peut pas échouer », « un contrôle qui ne peut pas réussir »), et
peut-être le plus coûteux des trois : il produit des investigations sur des fantômes.

**Correction retenue** : tout gate de fidélité position-par-position doit être **teacher-forcé**
(contexte imposé), jamais posé sur une trajectoire libre. C'est d'ailleurs ce que le finding
`generation_config` avait fait pour son A/B à un seul facteur (contexte forcé = prompt ++ les 57
premiers ids) — la rigueur était dans le finding, elle avait été perdue dans la spec du chantier
qui en découlait.

## 6. Portée sur les claims existantes du projet

À instruire lors de la passe de nuance (gate GC11 du chantier `generation_config`) :

| Claim | Statut | Raison |
|---|---|---|
| U8 « 48/48 == HF-fp32 STRICT » | **non affectée** | teacher-forcée (`69 --teacher-force`), marges ≥ 0,026 |
| U9-iv « 1150/1150 STRICT » | **non affectée** | teacher-forcée, marge min 0,0279 |
| PR #13 « == HF-fp32 STRICT sur 4041 positions » | **non affectée** | teacher-forcée (4000/4000) |
| Gates d'**équivalence ZML↔ZML** (M1/M2, D1/D2, R0/R1 : « ids == témoin ») | **⚠ à ré-instruire** | ils comparent deux **runs libres**. Ils ont passé — mais la marge de sécurité n'est pas connue : rien ne dit qu'un rejeu ne bifurquerait pas |
| « 1020/1020 == HF greedy » (génération longue) | **⚠ à ré-instruire** | vérifier si le critère était teacher-forcé ou libre |

⚠ **Aucun de ces gates n'est déclaré faux ici.** Ils ont mesuré ce qu'ils disent avoir mesuré, le
jour où ils l'ont mesuré. Ce qui est en cause est leur **reproductibilité**, qui n'avait jamais été
exercée : rejouer un gate d'équivalence d'ids en roue libre est un tirage, pas une vérification.

## 7. Quantification (en cours)

N = 20 runs identiques, config nominale, aucun flag d'observation. À publier : taux de bifurcation
@47 avec intervalle, et **recensement des autres positions instables** sur les 200 (le taux de 1/4
ci-dessus est une observation sur 4 runs, **pas** une mesure).

## 8. Artefacts

- Témoin d'origine : `/data/gemma4-zml-probe/rp0_witness/witness_long_before.{safetensors,log}`
- Runs du 29 juil : `/tmp/gc_t0_w200*.log` (run2), `/tmp/gc_t0_run3.log`, `/tmp/gc_t0_run4.log`
- Campagne N=20 : `/tmp/nd20/` (`ids<N>.txt`, `driver.log`)
- Marge @47 : `/data/tf_probe/tf200.json` (`mismatches[0]`, `margin = 0.004587`)
