# M-K5-1 — prédiction ÉCRITE AVANT LES RUNS (scénario PF3)

Committée avant le premier run long. Ce qui suit est dérivé de mesures déjà faites ce jour,
pas deviné — et les deux métriques sont publiées **ensemble** : l'une sans l'autre serait un
chiffre juste et trompeur (spec §2bis).

## Base mesurée (runs courts de ce chantier, binaire `fa2284fa…`)

| Grandeur | Valeur mesurée | Source |
|---|---|---|
| Vitesse de step (prefill) | 9,4 tok/s → **0,1064 s/step** | `pf1_runA.err.log` : 34 steps en 3,626 s |
| Vitesse de step (génération) | 10,3 tok/s | `pf1_runA.err.log` : 16 tokens en 1,549 s |
| Chargement dump 880 Mio + h2d + 1er token | **3,636 s** | `pf1_runB.err.log` : `KVLOAD-PERF` |

## Scénario PF3

`step_next ≈ 1005`, tour 2 de `n_new` ids (prompt « Describe the door of my home, and remind
me of my name. » — attendu ~28-32 ids rendus), `m = 32` générations.
`ids_full = 1005 + 1 (fed_next) + 2 (clôture) + n_new`.

## Prédiction 1 — positions évitées

Le prefill partiel n'absorbe que `1 + 2 + n_new` positions au lieu de `ids_full`.

> **Ratio prédit ≈ ×31** (pour `n_new = 30` : 1038 / 33 = 31,5).
> Plage admise selon `n_new` réel : **×28 à ×35**.

## Prédiction 2 — temps de steps

- re-prefill complet équivalent : 1038 × 0,1064 ≈ **110 s**
- prefill partiel : 33 × 0,1064 ≈ **3,5 s**, **plus** le chargement du dump ≈ 3,6 s
  (mesuré, et il ne disparaît pas : le dump 4k pèse davantage, celui-ci ~880 Mio)

> **Gain en temps prédit ≈ ×15** (110 / 7,1), soit **nettement moins que le ×31 en
> positions** — parce que la relecture du dump est un coût fixe que le comptage de positions
> ignore. Publier le ×31 seul serait le chiffre trompeur.

## Ce qui tuerait ces prédictions

- ratio de positions hors de [×28, ×35] alors que `n_new` est dans [28, 32] ⇒ le comptage
  de positions du runner est faux ;
- gain en temps > ×31 ⇒ impossible sans que le chargement du dump soit gratuit : erreur de
  mesure (chrono qui n'inclut pas la relecture) ;
- gain en temps < ×5 ⇒ la capacité ne paie pas son coût de relecture à cette longueur, et
  M-K5-1 devrait le dire tel quel.

M-K5-1 est une **mesure publiée sans verdict** : aucun tag n'en dépend.
