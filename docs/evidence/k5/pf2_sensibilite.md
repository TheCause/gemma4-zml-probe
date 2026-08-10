# PF2 — sensibilité MESURÉE du gate PF1 au mensonge de position

Mutant `shift-fwd --n N` : le manifest déclare N tokens que le cache ne porte pas
(slots restés aux zéros). Le run forgé est dépouillé contre l'oracle NOMINAL — c'est
la situation qu'un bug de position produirait.

| N | argmax basculés / comparés | \|Δ\| max | \|Δ\| médian | verdict argmax |
|---|---|---|---|---|
| 0 | 0/19 | 0.0000 | 0.0000 | — (nominal) |
| 1 | 0/19 | 1.0688 | 0.0906 | ne mord pas |
| 2 | 1/19 | 1.2450 | 0.2469 | **MORD** |
| 4 | 1/19 | 2.3224 | 0.1646 | **MORD** |
| 16 | 3/19 | 2.8970 | 0.4591 | **MORD** |

Marge médiane du scénario (canal brut, positions ctx) : **3.7007**

## Ce que ça établit

1. La corruption d'UNE position est **opérante** : elle déplace les 19 logits comparés
   (|Δ| jusqu'à 1,07). Elle n'est pas inerte, elle est **noyée** — les marges du
   scénario valent 3,70 en médiane, soit ~3,5× le déplacement maximal qu'elle produit.
2. Le plus petit mensonge VU basculer un argmax est **N = 2**.
3. Le run nominal reproduit le témoin à |Δ| = 0,0000 : le déterminisme intra-binaire
   est vérifié au passage, la ligne N=0 n'est pas décorative.

## Pourquoi les marges sont grasses ici (et non un défaut du scénario)

Les 19 positions comparées sont celles du tour 2 : `<turn|>`, `\n`, `<|turn>`, `user`,
`What`, `is`, `my`, `name`, `?`, puis l'amorce `model` / `<|channel>thought`. Ce sont des
tokens de STRUCTURE, dont la prédiction dépend peu du contexte lointain — leurs marges
sont grasses par nature, pas par choix de prompt. Un scénario à marges fines sur ces
positions-là n'est pas construisible en changeant le prompt.
