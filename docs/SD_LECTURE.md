# SD — Lecture des résultats (24 sept 2026)

> Chiffres : `docs/SD_RESULTS.md`, généré par `scripts/86_sd_report.py` (exit 1 : P2 réfutée) ;
> contre-épreuve `scripts/87_sd_selfproof.py` : les 5 mutants sont condamnés. Spec et prédictions
> committées avant tout run : `docs/superpowers/specs/2026-09-24-decision-layer-design.md` (`7226fc7`).
> Ce fichier est écrit à la main, à part : 86 régénère `SD_RESULTS.md` à chaque exécution.

## Ce qui tient

- **Le moteur de décision est juste.** Les logits des 4 candidats lus par ZML restent à 0,040 au
  plus de HF fp32 (seuil 0,05, C-SD-B), et le logsumexp à 0,019. L'isolement entre cas est
  bit-identique dans un même processus (C-SD-D). Le graphe de `gemma4_gen_auto` n'a pas bougé
  (empreinte HLO identique, A1 48/48, C-SD-C). L'évaluateur rend l'option attendue sur toutes les
  lignes (C-SD-F).
- **P1 confirmée** : lire 4 logits plus un logsumexp coûte autant que lire le top-5
  (0,471 contre 0,478 ms, écart de 0,007 ms).
- **P2 (a) et (c) tiennent** : le gain face à l'appel JSON est positif sur les 24 cas, et il
  représente 20,6 % de la latence du bras A en médiane (seuil de 25 %).

## Ce qui est réfuté

- **P2 (b), l'amplitude, est réfutée.** Le gain médian mesuré est de 176 ms, contre 261 ms prévus
  par la formule. L'écart médian de 83,5 ms dépasse la tolérance de 78,3 ms. La cause est
  mesurée : la formule utilisait les coûts publiés (14 ms par pas d'absorption, 9 ms par pas de
  génération), alors que ce run donne 8,77 et 8,42 ms. Le chiffre de 14 ms venait d'un run court
  juste après compilation, un risque que la spec nommait. Le seuil n'est pas requalifié.

## Ce que ni la spec ni les prédictions n'annonçaient

- **Sur ce jeu, la distribution n'apporte aucun signal d'abstention.** Les 20 erreurs du bras C
  (sur 240 décisions) ont toutes une marge d'au moins 0,82. La courbe couverture/erreur est donc
  plate : aucun seuil θ ne retire une erreur sans retirer tout le reste. La spec prédisait que
  l'apport propre de C serait « l'information » (marge, abstention). **Ce n'est pas observé ici.**
- **Le format lettre coûte en exactitude.** Le bras A (JSON) classe 24 cas sur 24. Les bras B et
  C classent 220 décisions sur 240 et font les mêmes erreurs, puisque le top-1 est toujours une
  étiquette. Les erreurs portent sur `search` et `insufficient` : 4 cas sur 24 changent de
  réponse quand on permute les lettres. C'est un biais de position ou d'étiquette, pas un défaut
  de lecture.
- **Le protocole est bien aligné** : `mass_in` vaut au moins 0,945, donc le modèle met
  l'essentiel de sa masse sur les 4 étiquettes. Le bras B ne sort jamais hors étiquette (0/240).

## Lecture d'ensemble

Sur ce moteur, décider en une étape plutôt qu'en générant un JSON fait gagner environ 20 % de
latence (≈ 176 ms sur ≈ 835 ms). Une étiquette générée (B) obtient exactement ce gain ; lire
les logits (C) n'y ajoute rien de mesurable en vitesse, et **sur ce jeu** rien en information
exploitable. En contrepartie, le format lettre s'est révélé moins exact que le JSON.

## Limites

24 requêtes écrites par l'auteur de la spec, 4 classes, un seul modèle (E2B), un seul gabarit de
prompt, greedy. Le non-déterminisme inter-processus (écart ≤ 0,022 sur les logits entre deux
compilations, déjà documenté par `docs/DOCUMENTATION.md:737`) est absorbé par C-SD-B.
Hors périmètre, non mesuré : prefill S>1 (le vrai levier de latence), questions en lot, Score et
Noul, E2B décideur devant un exécutant lourd.
