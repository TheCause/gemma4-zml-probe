# Mini-spec — D1/D2 : couverture GPU de `applyTopP` et `applyTemperature` (10 août 2026)

> Statut : **À VALIDER PAR RÉGIS avant la première ligne de code** (Task 4 Step 1 du plan
> `2026-08-10-dettes-techniques.md`). GO de principe donné le 10 août ; ce document fixe les
> gates et leurs prédictions falsifiables.

## 1. La dette, telle qu'elle est écrite aujourd'hui

`docs/SAMPLING_RESULTS.md` §5 :

- **D1** — `applyTopP` n'a **aucune** couverture GPU. Sa seule couverture est la fixture host
  `S2-U` (`scripts/72_sampling_fixture.py`). Motif du trou : le régime neutre de `S2-PONT` est
  `--top-k 1`, qui **court-circuite** top-p. C'est la brique dont les mesures ont montré qu'une
  formulation naïve rend un ensemble **disjoint** de HF — dette qualifiée de « sérieuse ».
- **D2** — `applyTemperature` n'est pas exercé de bout en bout : la config Google porte
  `temperature: 1.0`, et le runner suit HF en **n'instanciant pas** le warper à `T == 1.0`
  (`gemma4_g12auto.zig:2873`). La ligne n'est donc **jamais exécutée** sur GPU.

## 2. Méthode — pont in-process en régime ARMÉ

Le seul protocole insensible à la bistabilité du 12B (finding du 29 juil) est celui de `S2-PONT` :
comparer **deux implémentations sur le MÊME vecteur de logits, au MÊME step, dans le MÊME
processus**. Ni second forward, ni seconde compile, ni témoin stocké.

Différence avec `S2-PONT` : celui-ci compare deux **sélecteurs** (top-5 in-graph vs chemin B) et
n'est valide qu'en **régime neutre** (`SAMPLING_RESULTS.md` l.66-68 : en régime armé ses 21
désaccords sont *attendus*, le sampling divergeant légitimement de l'argmax). Le pont D1/D2 ne
compare **pas** des sélecteurs : il compare `applyTopP` (resp. `applyTemperature`) à une
**réimplémentation de référence indépendante**, sur le vecteur intermédiaire. Il est donc valide
**en régime armé** — c'est même le seul régime où top-p et la température font quelque chose.

Insertion (chemin B, `gemma4_g12auto.zig:2870-2875`), sans toucher au graphe :

```
applySuppression(work)          ← état commun
applyTemperature(work, T)   ──► [G-D2] compare à ref_temp(copie d'avant, T)
applyTopK(work, k, min_keep)    ← état commun (déjà couvert par S2-U)
                            ──► copie de `work` dans `ref` (buffer PRÉ-RÉSERVÉ)
applyTopP(work, p, min_keep) ─► [G-D1] compare l'ensemble des survivants à ref_topp(ref, p, …)
```

**Contrainte D10** : `ref` et les compteurs sont pré-réservés à l'init (1 MiB f32, comme `work`),
**zéro allocation dans la boucle de step** — le compteur `CountingAllocator` reste actif et le
vérifie gratuitement.

## 3. Le point de rigueur — pourquoi « comparer à une référence » ne suffit pas ici

Une référence écrite comme le code testé ne prouve rien : elle compare le code à lui-même.

- Pour **D1**, la référence est écrite **différemment de l'implémentation** : tri **descendant**
  + cumsum directe `cum >= p` sur les probabilités, formulation littérale de HF, là où
  `sampling.zig:189` trie **ascendant** avec le critère `cum <= 1 - p` et n'ordonne que les
  candidats non filtrés. Deux chemins réellement distincts qui doivent rendre le **même
  ensemble** — c'est la comparaison qui a du contenu.
- Pour **D2**, il n'existe **qu'une** façon d'écrire une division f32 : une « référence
  indépendante » serait tautologique. Le contenu de G-D2 est donc ailleurs, et il faut le dire :
  ce que le gate prouve, c'est (i) que la ligne est **réellement exécutée** sur GPU (elle ne l'a
  jamais été), (ii) qu'elle **divise** et ne multiplie pas par l'inverse, (iii) que la
  température a un **effet observable** sur la trajectoire. Le point (ii) n'est prouvé que par le
  **mutant** ci-dessous, pas par la comparaison.

## 4. Les gates, leurs prédictions et ce qui les tue

| Gate | Prédiction pré-enregistrée | Ce qui la TUE |
|---|---|---|
| **G-D0** | md5 HLO `before_optimizations` **identique** au témoin capturé avant la 1ʳᵉ ligne de code (le pont est host-side), et `ALLOC-LOOP: alloc=0 resize=0 remap=0 free=0 bytes=0` inchangé | tout md5 différent ; tout compteur d'alloc non nul |
| **G-D1** | Sur **≥ 300 steps** GPU armés (`top_k=64, top_p=0.95, T=1.0`), l'ensemble des survivants de `applyTopP` == celui de la référence descendante : **0 désaccord**. **Antécédent non vide exigé** : `n_topp_bit ≥ 30` steps où top-p retranche ≥ 1 token *après* top-k | ≥ 1 désaccord (= bug réel de `applyTopP` jamais exercé sur GPU) ; ou `n_topp_bit == 0` ⇒ gate passé **À VIDE**, changer de prompt/paramètres et le **dire** |
| **G-D2** | Avec `T=0.7` : `n_temp_applied ≥ 300` (la ligne s'exécute), écart max à `logit/T` **exactement 0 ULP**, et la trajectoire **diffère** de celle du run `T=1.0` | `n_temp_applied == 0` (vacuité) ; écart ≠ 0 ; trajectoire identique à `T=1.0` (la température ne ferait rien d'observable) |

**Contre-preuves obligatoires — chaque gate doit être VU échouer** (un gate jamais vu échouer
n'est pas un gate) :

1. **Mutant D1** : référence passée en tri **ascendant** sans adapter le critère ⇒ G-D1 doit
   FAIL avec un grand nombre de désaccords (c'est la mutation qui, sur la fixture `S2-U`, avait
   sorti « 262 144 survivants au lieu de 464 »).
2. **Mutant D2-a (la division)** : `x.* = x.* * (1.0 / t)` au lieu de la division ⇒ l'écart ULP
   doit devenir ≠ 0 sur au moins un logit. **Si ce mutant ne mord pas, G-D2 ne prouve rien sur
   (ii)** et devra être publié comme tel plutôt que maquillé.
3. **Mutant D2-b (l'ORDRE de la chaîne HF)** — demandé par Régis le 10 août pour que G-D2 ne se
   contente pas de prouver que la ligne s'exécute.

   ⚠ **Le déplacement « température APRÈS top-k » serait VACU, et ne doit pas être utilisé** :
   diviser par `T > 0` est **monotone croissante**, donc `applyTopK` — dont le critère est
   `x < kth`, un pur ordre — rend **exactement le même ensemble** avant ou après division
   (et `FILTER = -inf` divisé reste `-inf`). L'ordre nominal `Temp → TopK → TopP` et l'ordre muté
   `TopK → Temp → TopP` produisent le **même** résultat : le mutant passerait, et on conclurait à
   tort que le gate garde l'ordre. C'est un contrôle qui **ne peut pas échouer**
   (`feedback_invariant_tue_le_controle`).

   **Le mutant retenu est donc `TopK → TopP → Temp`** (température après top-p), seul déplacement
   observable : `applyTopP` calcule alors son softmax sur des logits **non divisés**, donc une
   distribution moins piquée, donc un masque différent dès que `T ≠ 1`. Prédiction : à `T = 0.7`,
   l'ensemble des survivants du mutant **diffère** du nominal sur ≥ 1 step. Si l'écart est nul sur
   tous les steps, **G-D2 ne prouve pas l'ordre** et sera publié comme tel.
4. **Vacuité** : un run à `top_p = 1.0` doit rendre `n_topp_bit == 0` et faire refuser le gate
   (l'antécédent vide est détecté, pas ignoré).

## 5. Ce que ce chantier ne prouve PAS (à garder écrit)

- Il ne couvre pas l'**E2B** (dette D6 : ses runners ne sortent pas les logits du graphe).
- Il ne prouve pas l'équivalence du **tie-break** argmax host vs topK in-graph (dette D8, publiée
  par `n_exact_top_ties`, pas résolue ici).
- Il ne dit rien de la **phase 1** (repetition penalty), suspendue et hors périmètre.
- G-D1 prouve l'égalité des **ensembles de survivants**, pas celle du token tiré : le tirage
  dépend du PRNG, hors périmètre (dette K1).

## 6. Décisions Régis — PRISES le 10 août 2026

1. **Portée de G-D2** (§3) : **G-D2 renforcé par un 2ᵉ mutant** sur l'ORDRE de la chaîne HF, en
   plus du mutant sur la division. ⚠ La forme demandée (« température après top-k ») s'est
   révélée **vacue à l'analyse** — corrigée en « température après **top-p** », seul déplacement
   observable, avec la démonstration au §4.3. Le renforcement demandé est donc appliqué **sur la
   variante qui peut réellement mordre**.
2. **Paramètres du run G-D1** : `top_k=64, top_p=0.95` (valeurs de la config Google) sur un prompt
   ouvert, ≥ 300 steps. Si l'antécédent tombe sous 30, resserrer `top_p` et **le dire** (un
   antécédent obtenu au 2ᵉ essai reste un antécédent, à condition de l'écrire).
3. Le run G-D2 à `T=0.7` **arme le sampling** : il diverge légitimement du greedy. Aucun gate
   d'équivalence « == HF » n'est concerné — sa sortie ne doit pas être lue comme une régression.
