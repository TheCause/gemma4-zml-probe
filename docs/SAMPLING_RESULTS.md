# Sampling (phase 2) — résultats

> **Statut : 4 gates sur 5 verts.** `S2-U`, `S2-PONT`, `S2-D`, `S2-R` sont franchis et taggés.
> `S2-G` et la mesure `M-COUT` restent à exécuter. Ce document n'est **pas** un rapport de
> clôture : ce qui n'y porte pas de chiffre n'a pas été mesuré.
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
| **S2-G** — le graphe n'a pas bougé | *à exécuter* | témoin figé en Task 0 : md5 `297679847aa04b719942d75d093adf2b`, 1 905 860 o |
| **M-COUT** — surcoût du chemin complet | *à exécuter* | mesure publiée, **sans PASS/FAIL** |

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
| **D1** | **`applyTopP` n'a AUCUNE couverture GPU** — sa seule couverture est la fixture `S2-U` | Le régime neutre du pont est `--top-k 1`, qui **court-circuite** top-p. Un régime neutre par top-p exigerait un véhicule GPU dédié. ⚠ C'est la brique dont les mesures montrent qu'une formulation naïve rend un ensemble **disjoint** de HF : dette **sérieuse et déclarée** |
| **D2** | **`applyTemperature` n'est pas exercé de bout en bout** | `temperature: 1.0` ⇒ HF n'instancie pas le warper : la config Google ne l'exerce jamais. Couverture = fixture seule |
| **D3** | **L'interdit « aucune allocation par step » n'a pas de gate porteur** | Tenu par la revue de code. L'adosser au débit ne serait pas honnête : l'effet est sous le plancher de résolution |
| **D4** | **Équivalence de l'arrêt runner ↔ HF** : prouvée par aucun gate | Héritée du chantier précédent, **aggravée** par la penalty (pénaliser un id EOS modifie l'arrêt) |
| **D5** | **`RP7` (« la récitation est-elle levée »)** : suspendu | Le symptôme d'origine n'a jamais été reproduit. Appartient à la spec du 27 juil : à arbitrer là-bas |
| **D6** | **E2B non couvert** | Ses runners ne sortent pas les logits du graphe |
| **D7** | **Custody des coûts F16** | Mesures prises GPU non vierge : absolus à requalifier |
| **D8** | **Tie-break `argmax` host vs `topK` in-graph** non prouvé équivalent | `S2-PONT` le **publie** (`n_exact_top_ties`, observé à **0**) au lieu de le supposer résolu |

## 6. Divergences délibérées avec HF, déclarées

- **`T_MIN = 1e-30`** : HF n'exige que `t > 0` et accepte donc `1e-45`, qui fait déborder la
  division en f32 et produit des **`NaN`** (`inf - inf` au softmax). Les logits étant bornés par le
  softcap 30, ce plancher laisse 8 ordres de grandeur de marge. **Divergence assumée**, pas une
  reproduction.
- **`--temperature 0`** : **rejeté comme HF** (qui lève une `ValueError`), avec un message
  renvoyant vers `--top-k 1` pour du greedy déterministe.
- **Gardes en acceptation** (`!(p > 0 and isFinite(p))`) et non en rejet : `p <= 0 → rejet`
  laisserait passer **`NaN`**, toute comparaison avec `NaN` étant fausse.
