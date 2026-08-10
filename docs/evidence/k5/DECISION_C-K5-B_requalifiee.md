# Décision — C-K5-B requalifiée (Régis, 10 août 2026)

## La claim pré-enregistrée

> **C-K5-B** — « Le gate mord : un état qui ment d'**UNE** position échoue » (non-vacuité,
> spec `2026-08-10-k5-prefill-partiel-design.md` §2bis).

## Ce que la mesure a dit

**RÉFUTÉE en argmax à N = 1.** Le mutant `shift-fwd --n 1` ne fait basculer aucun des 19
argmax comparés. Il n'est pas pour autant inerte : il **déplace les 19 logits**, de 1,0688
au maximum. Il est **noyé** — la marge médiane du scénario vaut 3,7007, soit ~3,5× son
déplacement maximal.

Courbe complète : `pf2_sensibilite.md`.

| N | argmax basculés | \|Δ\| max | verdict argmax |
|---|---|---|---|
| 0 | 0/19 | 0,0000 | nominal reproduit à l'identique |
| 1 | 0/19 | 1,0688 | ne mord pas |
| **2** | **1/19** | **1,2450** | **MORD** |
| 4 | 1/19 | 2,3224 | MORD |
| 16 | 3/19 | 2,8970 | MORD |

## La décision

**Requalification, actée par Régis le 10 août 2026** : le mutant canonique de PF2 devient
`shift-fwd --n 2` — **le plus petit mensonge VU mordre**, pas le plus commode. La claim
devient :

> **C-K5-B (rév. 2)** — le dispositif PF1/PF3 détecte un état qui ment de **≥ 2 positions**
> sur un scénario dont les marges médianes valent ~3,7. Il ne détecte **pas**, en argmax, un
> mensonge d'une seule position : celui-ci reste sous le seuil de bascule.

## Ce que cette limite implique, écrit et non tu

1. **PF1 et PF3 ne sont plus vacueux** : le contraire a été vu mordre, ce qui était la seule
   exigence de fond (`feedback_invariant_tue_le_controle`). Mais leur pouvoir de détection
   est désormais **chiffré**, pas supposé.
2. **Un bug réel de position serait vu.** Un décalage de position affecte toutes les
   positions du segment, pas une seule — la classe de défauts que le gate doit attraper vit
   au-delà de N = 1. Le cas N = 1 est le cas-limite, et il est publié comme non couvert.
3. **Le mordant tombe à la position de plus faible marge** (0,437057, le minimum du
   scénario) : la corruption bascule d'abord là où le modèle hésite le plus. C'est cohérent
   avec le mécanisme, et c'est un contrôle croisé gratuit de l'explication « noyée par les
   marges ».
4. **Les marges grasses ne sont pas un défaut de scénario.** Les 19 positions comparées sont
   les tokens de structure du tour 2 (`<turn|>`, `user`, `What`, `is`, `my`, `name`, `?`,
   `model`, `thought`). Leur prédiction dépend peu du contexte lointain. Changer le texte du
   prompt ne les rend pas fines — l'option « chercher un scénario où N=1 mord » a donc été
   écartée sur ce motif, pas par économie.

## Le fait qui dépasse le gate

Le run forgé produit **la même réponse que le nominal** — « Your name is Aldebaran. » — à
**N = 1, N = 4 ET N = 16**. En décodage libre, un mensonge de seize positions reste
invisible. L'exigence « teacher-forcé, jamais en décodage libre » (`project_gemma4_zml_probe`,
10 août) est donc vérifiée empiriquement ici, et non plus seulement argumentée.
