# K5 — Prefill partiel : résultats

> Reprendre un cache KV dumpé **et absorber un prompt neuf**. La brique conversation /
> préfixe partagé, prouvée **teacher-forcée contre HF fp32**, jamais en décodage libre.
>
> Spec : `docs/superpowers/specs/2026-08-10-k5-prefill-partiel-design.md` (rév. 3).
> Plan : `docs/superpowers/plans/2026-08-10-k5-prefill-partiel.md`.
> Preuves brutes : `docs/evidence/k5/`. Exécuté le 10 août 2026.

---

## 1. Ce que le chantier livre

`--load-cache F --prompt "tour 2"` : le contexte vient du dump, le prompt neuf est absorbé
comme **tour suivant** aux positions `step_next…`, et la génération continue.

```
ids_full = ids_fed ++ [fed_next] ++ [clôture <turn|> \n] ++ ids_tour2
```

**Le chantier est 100 % host-side** : le graphe ne distingue pas prefill et génération
(position ≡ `ctrl.step`, masques in-graph, RoPE couvrant `L_MAX`). Rien n'a été ajouté au
graphe — PF0 le prouve à l'octet.

**La démonstration en une ligne** : un run A raconte une histoire à propos d'un phare
appartenant à « Aldebaran », son état est dumpé ; un run B **dans un autre process** reprend
ce dump, reçoit `"What is my name?"` et répond **« Your name is Aldebaran. »**. Le nom ne
peut venir que du cache repris.

---

## 2. Verdicts des 7 gates

| Gate | Prouve | Mesuré | Tag |
|---|---|---|---|
| **PF0** graphe intact | C-K5-D | md5 HLO `297679847aa04b719942d75d093adf2b` **identique** au témoin capturé avant la 1ʳᵉ ligne de code ; `git diff engine.zig` **vide** | `gate/pf0-pass` |
| **PF1** équivalence à la frontière | C-K5-A | **19/19** positions ctx (canal BRUT) + gen[0] + 7 générations (canal POLICY), **0 mismatch**. Marges publiées avant verdict : ctx médiane **3,7007** (min 0,4371), gen médiane **9,0490** | `gate/pf1-pass` |
| **PF2** le mordant | C-K5-B **requalifiée** | mutant `shift-fwd --n 2` : MISMATCH `ctx[3]`, marge **0,437057** > 10× TIE. Le nominal re-passe sur le même binaire | `gate/pf2-pass` |
| **PF3** fenêtre à travers la frontière | C-K5-C | scénario long (`step_next=1004`, `ids_full=1033`, T=1064) : **28/28** ctx + **32/32** générations, **0 mismatch**, témoin fenêtre **`bites_in_prefill: true`** | `gate/pf3-pass` |
| **PF4** refus bruyants | §4.6 | **5/5** cas VUS échouer avec leur erreur **nommée**, aucun crash non qualifié | `gate/pf4-pass` |
| **PF5** non-régression | C-K5-E | `--selftest-kvdump-eq` **32/32 bit-identiques** sur 1280 **et** 4k ; reprise simple : **zéro ligne `K5:`**, `--out-ids` à **une seule clé** | `gate/pf5-pass` |
| **PF6** rendu tour 2 | C-K5-F | **17/17 ids** identiques au suffixe HF mesuré | `gate/pf6-pass` |

`ALLOC-LOOP: alloc=0 resize=0 remap=0 free=0 bytes=0` sur **les 17 runs** du chantier, de
17 à 1064 steps — l'interdit D10 tient, prefill partiel compris.

---

## 3. Les 6 claims pré-enregistrées, jugées

### C-K5-A — « Le graphe est déjà correct au-delà de `step_next` » → **CONFIRMÉE**

Prédiction : argmax ZML == argmax HF fp32, `n_ctx/n_ctx` et `1/1`, divergence libre
éventuelle tolérée à marge ≤ 1,873e-3.

Mesuré : **19/19 + 1/1** (PF1) et **28/28 + 1/1** (PF3), zéro mismatch. Et **au-delà de la
prédiction** : les 7 (PF1) puis 31 (PF3) générations libres suivantes coïncident aussi, sans
qu'aucun tie n'ait eu besoin d'être invoqué. La dérivation §1.2 — position ≡ `step`, masques
in-graph, RoPE en table jusqu'à `L_MAX` — est **vérifiée par la mesure**, non plus déduite.

### C-K5-B — « Un état qui ment d'UNE position échoue » → **RÉFUTÉE puis REQUALIFIÉE**

**Réfutée à N = 1** : le mutant `shift-fwd --n 1` ne fait basculer **aucun** des 19 argmax.
Il n'est pas inerte pour autant — il **déplace les 19 logits**, jusqu'à **1,0688**. Il est
**noyé** : les marges du scénario valent 3,7007 en médiane.

| N | argmax basculés | \|Δ\| max | verdict |
|---|---|---|---|
| 0 | 0/19 | 0,0000 | nominal reproduit à l'identique |
| **1** | **0/19** | 1,0688 | **ne mord pas** |
| 2 | 1/19 | 1,2450 | MORD |
| 4 | 1/19 | 2,3224 | MORD |
| 16 | 3/19 | 2,8970 | MORD |

**Requalification actée par Régis le 10 août 2026** (`docs/evidence/k5/DECISION_C-K5-B_requalifiee.md`) :
le mutant canonique devient `--n 2`, **le plus petit mensonge VU mordre**. La claim rév. 2 :

> Le dispositif PF1/PF3 détecte un état qui ment de **≥ 2 positions** sur un scénario à
> marges médianes ~3,7. Il ne détecte **pas**, en argmax, un mensonge d'une seule position.

Contrôle croisé gratuit : le mismatch tombe **exactement à la position de plus faible marge**
du scénario (0,437057, le minimum mesuré) — ce que l'explication « noyée par les marges »
prédit, et non une histoire racontée après coup.

### C-K5-C — « La fenêtre sliding est correcte quand elle ne commence plus à 0 » → **CONFIRMÉE**

Scénario dimensionné pour que la fenêtre morde : T = 1064 > 1024. Le témoin de l'oracle
(`sliding_mask != causal_mask`) est **actif** — sans lui le gate serait INEXÉCUTABLE, pas
PASS. L'équivalence tient sur les 60 positions rapportées.

### C-K5-D — « Le graphe n'a pas bougé » → **CONFIRMÉE**

md5 HLO identique **avant et après**, et identique à celui de GC0 (27 juil), S2-G (29 juil),
DC0 (9 août) : **continuité sur sept chantiers**. `engine.zig` : zéro octet de diff.

### C-K5-E — « La reprise simple est inchangée » → **CONFIRMÉE**

`32/32` bit-identiques sur les deux variantes. Un `--load-cache` sans `--prompt` n'émet
**aucune** ligne `K5:` et son `--out-ids` porte la seule clé `"ids"` — vérifié **sur les
octets du fichier produit**, pas par lecture du code.

### C-K5-F — « Le rendu du tour 2 est celui de HF, en ids » → **CONFIRMÉE, cas de requalification NON déclenché**

Le jinja multi-tour 12B **ne réécrit pas l'historique** (`common_bc = 48/48`,
`history_rewritten = false`). Le suffixe est une concaténation propre, et le rendu Zig lui
est identique **à 17 ids sur 17**.

Ce que la mesure a corrigé au passage — voir §7.

---

## 4. M-K5-1 — le chiffre de la capacité (mesure, sans verdict)

| | Prédit (écrit avant) | Mesuré |
|---|---|---|
| Positions évitées | ×31 | **×35,6** (1033 positions contre 29 réellement absorbées) |
| **Temps** | ×15 | **×11,3** (111,6 s → 9,86 s) |

Les deux chiffres sont publiés **ensemble** : le ratio de positions ignore la relecture du
dump (**3,53 s mesurés**), que le temps, lui, paie. Détail et confrontation :
`docs/evidence/k5/mk51_mesure.md`.

**La prédiction de temps était trop optimiste, et l'erreur est écrite** : j'avais oublié les
32 générations du côté du prefill partiel alors que la baseline les comptait. Corrigée, elle
donnait ×11,4 — le mesuré. Aucun kill pré-enregistré n'a été déclenché.

### Les grandeurs prédites par la spec §2bis, confrontées

| Grandeur | Prédit | Mesuré | |
|---|---|---|---|
| coût d'un step | n_new / 9,6 s | **9,5 tok/s** (104,77 ms/step) | ✅ |
| run A du scénario PF3 | ≈ 105 s | **105,3 s** | ✅ à 0,3 % |
| `ALLOC-LOOP` des runs K5 | identique au run nu | `alloc=0` partout | ✅ |
| oracle PF3 (M4, fp32 CPU) | **≈ 15× l'oracle court** | **×1,8** (33,0 s → 58,6 s) | ❌ **RÉFUTÉE** |

La dernière ligne mérite d'être dite : la prédiction supposait un coût **proportionnel à la
longueur du prefill** (T 76 → 1064, soit ×14). Le temps réel n'a fait que **×1,8**. Le
prefill HF fp32 CPU à ces longueurs est donc dominé par un coût **par couche**, pas par la
longueur de séquence. Publié tel quel : c'est une prédiction ratée, pas une mesure ratée.

---

## 5. Périmètre — ce que ce chantier ne dit PAS

- **Un `fed_next` forgé reste indétectable host-side.** Les checksums couvrent `ids_fed` et
  les 4 caches, jamais la cohérence `fed_next`↔cache : le cache est opaque au host. Ce n'est
  **pas** un refus, c'est une limite documentée — seul l'aller-retour teacher-forcé la voit,
  et c'est ce que PF2 démontre.
- **Un mensonge d'UNE position n'est pas vu en argmax** (C-K5-B rév. 2).
- **Pas de gate teacher-forcé sur la variante 4k** (D-K5-4) : PF5 y passe, PF1/PF3 non.
- **Multi-tour ≥ 3, résident, batching** : hors périmètre.
- **E2B non couvert**, comme pour tous les chantiers 12B depuis `generation_config`.

---

## 6. Dettes (écrites, pas tues)

| # | Dette | Pourquoi |
|---|---|---|
| K5-1 | **Garde fenêtre transposée non levée** (`n_new >= 1024` ⇒ refus) | La garde historique n'a jamais eu de justification écrite (`c2211c0`). On transpose la prudence ; la lever exigerait son propre gate (D-K5-3) |
| K5-2 | **Gate teacher-forcé 4k absent** | D-K5-4 : l'oracle fp32 CPU y coûterait des heures ; PF5 couvre la non-régression 4k |
| K5-3 | **PF1 ne voit pas un mensonge d'1 position** | C-K5-B rév. 2, chiffré §3 |
| K5-4 | **Le chaînage v1 relit 880 Mio à CHAQUE tour** (3,53 s mesurés) | C'est le coût que **K4** (cache résident) lève — et le chiffre qui le justifie |
| K5-5 | **`scripts/74_kvdump_inspect.py` ne tourne pas sur la VM** | Aucun python de la VM n'a `numpy`+`xxhash` (système et les 2 venvs ComfyUI testés). Les forges se font sur M1, dump rapatrié puis renvoyé |

**K4 est débloqué** : un 2ᵉ prompt après restore EST un prefill partiel, et il est prouvé.

---

## 7. Écarts au plan, assumés et déclarés

1. **La clôture du tour 1 vaut DEUX ids, pas un.** Le plan injectait le seul `eot_id` et
   plaçait le `\n` en tête du littéral tour 2. La mesure dit `closure_tail_ids = [106, 107]` :
   le `\n` appartient à la clôture. Les deux découpages donnent le même `ids_full`, mais un
   seul rend PF6 vrai. **Diagnostiqué avant le code** par le recoupement de frontière que la
   spec §4.4.1 exigeait — ni PF6 ni PF1 n'auraient pu le voir (PF6 ne compare que le suffixe,
   et les deux côtés de PF1 consomment le même `ctx_ids`).
2. **La clôture est tokenisée, pas devinée** (`closureToIds`, garde de longueur) : un
   tokenizer qui découperait autrement BLOQUE au lieu de fabriquer un contexte silencieusement
   différent de celui que HF verrait.
3. **`renderChatTemplateTurn2` délègue à `renderChatTemplate`** : la mesure établit que le
   suffixe du tour 2 EST le rendu du tour 1 privé de son BOS. Dupliquer le littéral l'aurait
   fait diverger en silence.
4. **`shift-fwd` prend `--n`** — ajouté **après** la mesure, pour établir la sensibilité du
   gate, jamais pour forcer un vert.
5. **Script neuf `81_selfproof_80.py`** : la contre-épreuve du juge, avec contrôle que chaque
   mutant mute réellement (§8).
6. **Le `sed '0,/re/s//repl/'` du plan est inopérant sur BSD** : remplacé par du Python.

---

## 8. Ce que l'exécution a appris

**⚡ « Un mutant qui ne mute pas accuse le juge à sa place. »** La première contre-épreuve du
dépouilleur 80 déclarait le juge **aveugle** sur un cas. C'était faux : le `sed` de mutation
n'avait rien changé au log, silencieusement. Le réflexe — « mon juge est cassé » — pointait le
mauvais coupable. D'où `81_selfproof_80.py`, qui **compare le log muté au nominal** et déclare
le cas INEXÉCUTABLE plutôt que PASS quand la mutation est vide. C'est le pendant exact de
`feedback_test_vacuite_antecedent`, appliqué au mutant lui-même.

**⚡ « Une corruption opérante peut ne rien faire basculer. »** À N=1, la corruption déplace
les 19 logits comparés et n'inverse aucun argmax. « Ça ne mord pas » et « ça n'a aucun effet »
sont deux énoncés différents ; seule la mesure des **valeurs** les sépare. Un gate à critère
d'argmax a une **sensibilité**, et tant qu'on ne l'a pas chiffrée on ne sait pas ce qu'il
prouve.

**⚡ « En décodage libre, seize positions de mensonge sont invisibles. »** Les runs forgés à
N=1, 4 **et 16** produisent tous la même réponse que le nominal — « Your name is Aldebaran. ».
L'exigence « teacher-forcé, jamais en libre » n'est plus un argument : c'est une mesure.

**⚡ « Le plan supposait un environnement, la mesure l'a démenti. »** Le plan faisait tourner
le 74 sur la VM ; aucun python de la VM n'a `numpy`. Vérifié en une commande, contourné en
deux — mais un plan revu trois fois reste faillible sur un fait qu'on peut mesurer.

---

## 9. Reproduire

```bash
# build (source unique des 2 flags de mode)
ZML_WS=<workspace> ./zml_runner/build_3090.sh

# run A : contexte + dump
gemma4_g12auto <ckpt> <tok.json> --prompt "<tour 1>" --max-tokens 16 \
  --dump-cache etat.kvdump --dump-top5

# run B : reprise + prompt neuf (LA capacité)
gemma4_g12auto <ckpt> <tok.json> --load-cache etat.kvdump --prompt "<tour 2>" \
  --max-tokens 32 --dump-top5 --out-ids outB.safetensors

# oracle teacher-forcé (M4, fp32) puis verdict machine
python3 scripts/69_u8_gen_oracle.py --weights <dq> --compute-fp32 \
  --context-ids outB.safetensors --ctx-from <step_next> --out pf1.json
python3 scripts/80_pf1_bridge.py pf1.json runB.err.log

# le mordant (le gate ne vaut que parce qu'on l'a vu échouer)
python3 scripts/74_kvdump_inspect.py shift-fwd etat.kvdump forge.kvdump --n 2
python3 scripts/80_pf1_bridge.py pf1.json runForge.err.log --expect-fail

# le rendu du tour 2, host-only
gemma4_g12auto <ckpt> <tok.json> --ids-only-turn2 --prompt "<tour 2>"
python3 scripts/79_t2_render_check.py rendu_tour2_hf.json pf6.err.log
```
