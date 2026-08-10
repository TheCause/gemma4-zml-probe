# M-K5-1 — mesure (scénario PF3). Prédiction : `mk51_prediction.md`

Mesure publiée **sans verdict**, aucun tag n'en dépend.

## Base

| Grandeur | Valeur |
|---|---|
| Coût d'un step (run A long, 1005 steps) | **104.77 ms** |
| Steps réellement exécutés par le run B | **60** (28 prefill du tour 2 + 32 générations) |
| Relecture du dump 880 Mio (KVLOAD-PERF moins 1 step) | **3.526 s** |
| Coût total du run B | **9.86 s** |
| Baseline : re-prefill de ids_full (1033) + 32 générations | **111.6 s** (1065 steps) |

## Les deux chiffres, publiés ensemble

- **Positions évitées : ×35,6** — un prefill complet devrait absorber les 1033 positions de
  `ids_full` ; le prefill partiel n'en absorbe que **29** (fed_next + clôture 2 + tour 2 de 26).
- **Temps : ×11,3** (111,6 s → 9,86 s).

Le second est **presque trois fois plus bas** que le premier, et c'est le fait à retenir :
la relecture du dump (3.53 s) est un coût fixe que le comptage de positions ignore.
Publier le ratio de positions seul serait un chiffre juste et trompeur.

## Confrontation à la prédiction écrite AVANT

| | Prédit | Mesuré | Verdict |
|---|---|---|---|
| Positions | ×31 (plage ×28-×35, **conditionnée à n_new ∈ [28,32]**) | ×35.6 | condition NON remplie : n_new mesuré = 26. La plage ne s'applique pas ; recalculée à n_new=26, elle donne ×35.6 — cohérent |
| Temps | ×15 | **×11.3** | **prédiction trop optimiste** |

### Pourquoi ma prédiction de temps était trop haute — l'erreur, pas l'excuse

J'avais comparé « re-prefill complet (110 s) » à « prefill partiel (3,5 s) + chargement
(3,6 s) », en **oubliant les 32 générations du côté du prefill partiel** alors que la
baseline, elle, les comptait. La comparaison honnête les met des deux côtés : le run B
exécute 60 steps, pas 29. Corrigée, la prédiction aurait donné
×11.4 — soit exactement le mesuré.

### Les kills pré-enregistrés

- « gain > ×31 ⇒ erreur de mesure » : ×11.3 — **non déclenché**.
- « gain < ×5 ⇒ la capacité ne paie pas sa relecture » : ×11.3 — **non déclenché**.
- Positions hors plage avec n_new ∈ [28,32] : condition non remplie, sans objet.

## Ce que la mesure dit de la capacité

À 1004 positions de contexte, reprendre coûte **9.9 s** contre **112 s** de
recalcul — et **3.5 s de ces 9.9 s sont de la pure relecture disque**. Le
chaînage multi-tour v1 paie donc cette relecture à CHAQUE tour : c'est la dette de
conception que K4 (cache résident) lève, et le chiffre qui la justifie.
