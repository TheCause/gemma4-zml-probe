# Finding — un témoin d'ids de 200 tokens n'est pas reproductible d'une fenêtre à l'autre

**Date :** 10 août 2026 · **Trouvé pendant :** gate RP2 (chantier repetition penalty, Task 4 Step 4)
**Portée :** dépasse ce chantier — concerne tout gate qui compare des ids à un témoin STOCKÉ.

## Ce qui s'est passé

RP2 devait montrer que la penalty neutre ne change rien : re-run des RUN_ARGS de référence,
ids attendus bit-identiques au témoin figé en Task 1. **Ils ont différé** — 148 positions
sur 200. Premier réflexe possible : accuser le code de la penalty. Ce serait faux.

## Les mesures

| Artefact | Heure (VM) | Code source | md5 du binaire | md5 des ids |
|---|---|---|---|---|
| `rp_witness_long` (témoin Task 1) | 09:22 | `main` (`23ebdf7`) | **non capturé** (binaire de 07:15) | `bb74f916…` |
| `rp2_a` | ~10:05 | main + penalty | `59cf380a…` | `06a5953f…` |
| `rp2_b` (avec `--repetition-penalty 1.0`) | ~10:07 | main + penalty | `59cf380a…` | `06a5953f…` |
| **`rp2_main2`** | ~10:18 | **`main` PUR, recompilé** | `158646ca…` | `06a5953f…` |
| `rp2_c` (contrôle) | ~10:35 | main + penalty | `ba51d21b…` | `06a5953f…` |

**Trois binaires distincts, dont un sans une seule ligne du chantier, s'accordent sur les
mêmes 200 ids — et diffèrent tous du témoin de 09:22.**

Autres faits mesurés :
- md5 HLO `before_optimizations` **identique** aux deux moments (`297679847aa04b71…`) : le
  graphe soumis à XLA n'a pas bougé entre le témoin et les runs récents.
- La divergence commence au **token 47** sur 200, avec des ré-alignements partiels ensuite
  (positions 48 et 50 restent égales) — signature d'un basculement entre deux candidats
  proches, pas d'une faute logique (qui divergerait dès le premier token).
- Le binaire Zig **n'est pas reproductible bit-à-bit** : le même code source recompilé donne
  `59cf380a…` puis `ba51d21b…`. Sans effet sur les ids (les deux s'accordent), ce qui confirme
  que le binaire n'est pas le canal — le graphe l'est, et il est identique.

## Ce qu'on peut conclure, et ce qu'on ne peut pas

**Établi :** le code de ce chantier ne modifie pas les ids quand la penalty est neutre. La
preuve est une comparaison à `main` recompilé **dans la même fenêtre**, ce qui est plus fort
qu'une comparaison au témoin : elle contrôle la variable « fenêtre », que le témoin subissait.

**Établi :** un témoin d'ids de 200 tokens capturé à T n'est pas garanti reproductible à T+1h
sur cette machine, à graphe, poids et prompt identiques.

**NON établi :** la cause exacte. Le candidat le plus plausible est un choix d'algorithme
(autotuning cuBLAS/XLA) sensible à l'état de la machine, stable dans une fenêtre et pas entre
deux. Il manque une mesure pour trancher, et cette mesure est manquante par NOTRE faute :
**le md5 du binaire de 07:15 n'a pas été capturé avec le témoin**. Sans lui, on ne peut pas
prouver formellement que ce binaire correspondait au code de `main` — le `24 action cache hit`
du build de la Task 1 en est un indice fort, pas une preuve.

## Pourquoi ça n'avait jamais été vu

Les gates oracle historiques du repo portent sur des trajectoires de **48 tokens**
(`u8_gen48` et sa famille). La divergence apparaît ici au token **47**. Le témoin de 200
tokens introduit par ce plan est le premier instrument du repo assez long pour l'exposer.

## Conséquences

1. **RP2 est requalifié** (1ʳᵉ requalification de ce chantier, déclarée) : sa référence n'est
   plus un témoin stocké mais **un run de `main` recompilé dans la même fenêtre**. Le gate est
   PASS sous cette définition — et il prouve exactement ce qu'il devait prouver.
2. `docs/evidence/penalty/RUN_ARGS.md` conserve le prompt et `--max-tokens 200` comme RUN_ARGS ;
   ce qui est retiré, c'est la valeur du témoin comme *référence inter-fenêtre*.
3. **Règle d'instrumentation à retenir : capturer le md5 du BINAIRE en même temps que tout
   témoin de sortie.** Sans lui, on ne peut pas distinguer « le code a changé » de « le témoin
   a dérivé » — et le premier réflexe est d'accuser le code qu'on vient d'écrire.
4. **À surveiller en RP3** (Task 5) : `--oracle` compare une trajectoire LIBRE à une fixture HF
   stockée. Il hérite du même risque dès que la fixture dépasse ~47 tokens. La spec prévoit une
   procédure de requalification (§7-3, marge min) — c'est le bon endroit pour l'appliquer si
   un mismatch apparaît sans cause côté penalty.
