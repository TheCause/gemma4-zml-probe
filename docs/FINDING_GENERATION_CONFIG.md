# Finding — le portage n'applique pas `generation_config.json`

> **Date** : 2026-07-27 · **Statut** : **CORRIGÉ le 29 juil 2026** (12 gates verts —
> `docs/GENERATION_CONFIG_RESULTS.md`) · **Origine** : découvert incidemment en gelant les témoins
> du chantier repetition penalty (un token `<image|>` apparaissait en plein texte dans une
> génération greedy).
>
> **Ce qui a été mesuré à la correction** : `258882` passe de **11/20 runs à 0/20** (p = 1,16e-07,
> gate GC3) ; l'oracle cesse de partager l'angle mort (`n_match` 199 → 198, gate GC5) ; le graphe
> est **byte-identique** au témoin (GC0) ; et la trajectoire libre du runner corrigé est
> **60/60 identique** à un décodage HF greedy appliquant la même politique (GC8), dès lors qu'elle
> ne bascule pas sur le tie bistable de la position 47.
>
> **Décision Régis** : corriger **avant** le chantier repetition penalty — la couche de
> politique de décodage doit exister avant qu'on y greffe une penalty.

## 1. Le fait

En **greedy pur** (argmax, aucun échantillonnage), `gemma4_g12auto` émet le token d'id
**258882 = `<image|>`** au milieu d'un texte purement textuel, à une frontière où un espace est
attendu :

| Run | Position | Contexte |
|---|---|---|
| 200 tokens | 57 | `…into a mathematical<image|>concept, and finally into a number…` |
| 1150 tokens | 57 et 706 | idem, puis `…He wrote "On the<image|>    Calculation with Hindu Numerals."…` |

Soit ~1 occurrence par 500-600 tokens. Le décodage étant greedy, **c'est le logit maximal** qui
tombe sur ce token — pas un tirage improbable.

Prompt : `"Tell me the story of the number zero, from its invention to modern mathematics."`

## 2. Ce que dit le tokenizer

`tokenizer.json` du snapshot : vocab 262 144, **24 `added_tokens`**. L'id 258882 est
`<image|>`, `"special": true`, dans un îlot multimodal au milieu d'une zone `<unusedNNNN>` :

```
258880 <|image|>   258881 <|audio|>   258882 <image|>   258883 <audio|>   258884 <|video|>
258879 <unused2967>                                                       258885 <unused2968>
```

Convention du vocab (`105 <|turn>` / `106 <turn|>`) : `<|X>` ouvre, `<X|>` ferme.
**258882 est donc le marqueur de FIN d'image**, l'ouvrant étant `255999 <|image>`.

## 3. La cause

`generation_config.json` du snapshot Google (identique entre le snapshot HF d'origine et
l'export dq) :

```json
{
  "bos_token_id": 2,
  "do_sample": true,
  "eos_token_id": [1, 106, 50],
  "pad_token_id": 0,
  "suppress_tokens": [258883, 258882],
  "temperature": 1.0,
  "top_k": 64,
  "top_p": 0.95
}
```

**Google supprime explicitement `<image|>` et `<audio|>` au décodage.** Transformers câble cela
inconditionnellement (`generation/utils.py`, `SuppressTokensLogitsProcessor` → `scores = -inf`),
**en amont des warpers, donc greedy compris**.

Côté portage :

```
grep -rn "suppress\|generation_config" zml_runner/*.zig          → 0 occurrence
grep -n  "suppress\|generation_config" scripts/69_u8_gen_oracle.py → 0 occurrence
```

`gemma4_g12auto` fait un **argmax nu sur les 262 144 ids**, sa seule politique étant l'arrêt sur
`<turn|>` = 106.

## 4. La preuve — A/B à un seul facteur

Contexte forcé identique (prompt templaté S=29 ++ les 57 premiers ids produits par ZML), un pas
de `model.generate()` greedy sur le vrai 12B :

```
[A_avec_suppression]  -> next = 11814  '▁hero'      | logits à -inf : 2 | logit(258882) = None
[B_sans_suppression]  -> next = 258882 '<image|>'   | logits à -inf : 0 | logit(258882) = 19.125
```

**Le seul delta est `suppress_tokens`.** Sans lui, HF reproduit ZML exactement.

Top-5 fp32 à la position 57 (logits post-softcap) :

| rang | id | token | logit |
|---|---|---|---|
| 1 | **258882** | `<image|>` | **19.1645** |
| 2 | 11814 | `▁hero` | 18.9147 |
| 3 | 3495 | `▁concept` | 18.6035 |
| 4 | 1548 | `▁number` | 18.0667 |
| 5 | 13186 | `▁entity` | 17.9584 |

Marge top1−top2 = **0,2499** — ce n'est **pas** un tie bf16 (l'ULP bf16 à 19 vaut 0,0625).
Marge médiane du run : 4,14.

**258882 est dans le top-5 à 5 positions sur 200** (rang 2 @1, **rang 1 @57**, rang 3 @138,
rang 2 @146, rang 2 @149). Le token flotte structurellement haut : c'est exactement ce que la
suppression de Google traite.

Le teacher-forcing fp32 sur le témoin 200 ids donne **199/200** d'accord argmax ZML↔HF — la
position 57 étant un **match** : l'oracle nu émet lui aussi `<image|>`.

## 5. Le point le plus important : l'instrument partage l'angle mort du sujet

`69_u8_gen_oracle.py` fait un **argmax nu**, comme le runner. Il n'applique pas
`generation_config.json`. **Aucun gate existant ne pouvait donc détecter cet écart** : l'oracle
reproduit fidèlement le défaut qu'il est censé révéler.

C'est une variante nouvelle des leçons déjà payées sur ce projet (« l'instrument dégradé fabrique
des requalifications », « un contrôle qui ne peut pas échouer ») : ici l'instrument n'est pas
dégradé, il est **structurellement aveugle au même endroit** que le sujet mesuré.

## 6. Portée exacte de la claim du projet

La claim « les ids générés == HF » est :

- **vraie** au sens « même argmax sur les **logits bruts** » — c'est un critère plus strict et
  plus informatif que de comparer deux sorties de `generate()` ;
- **fausse** au sens « le runtime reproduit ce que `model.generate()` produirait », puisque le
  portage n'applique ni `suppress_tokens` ni les trois `eos_token_id`.

Cette nuance n'était écrite nulle part. Elle ne retire rien aux gates passés (qui mesurent bien
ce qu'ils disent mesurer) mais elle doit accompagner la claim.

## 7. Second écart, même tiroir

`eos_token_id` vaut **`[1, 106, 50]`**. Le runner ne s'arrête que sur **106**
(`gemma4_g12auto.zig` l. ~895-905 et ~1424), jamais sur `1 <eos>` ni `50 <|tool_response>`.

## 8. Troisième constat : le modèle est configuré pour être ÉCHANTILLONNÉ

`do_sample: true`, `temperature: 1.0`, `top_k: 64`, `top_p: 0.95`.

Le sampling stochastique — spécifié comme « phase 2, plus tard » dans
`docs/superpowers/specs/2026-07-27-sampling-repetition-penalty-design.md` — **est la
configuration nominale du modèle**, avec des valeurs fournies par Google. Le symptôme d'origine
du chantier penalty (« le greedy boucle sur la récitation ») est plausiblement le symptôme d'un
modèle utilisé hors de sa configuration prévue.

⚠ Élément de prudence : **la récitation n'a pas pu être reproduite**. Trois témoins greedy
(2, 200 et 1150 tokens) ne montrent aucune boucle — plus long n-gramme répété : 4 tokens sur
200, 5 tokens sur les 400 derniers de 1150 ; diversité plate (0,62-0,76) sans effondrement ; le
run de 1150 atteint sa propre conclusion. L'hypothèse « les boucles apparaissent au-delà de
200 tokens » est **réfutée**. Le prompt qui récite reste à identifier.

## 9. Divergence @47, non expliquée

En roue libre sur ce prompt, HF diverge de ZML **dès l'index 47** : ZML `5743 ▁zero`
(logit fp32 24,875866) vs HF `27069 ▁humanity` (24,880453) — **marge 0,004587**. C'est du bruit
au niveau du tie, pas nécessairement un défaut, mais ce n'est **pas instruit** : il faudrait
comparer les logits ZML et HF à cette seule position (`--dump-top5` contre l'oracle fp32).

Le gate U8 ne pouvait pas le voir : son prompt est « What is the capital of France? Answer in
one word. », qui s'arrête à 2 tokens.

## 10. Ce qui reste à établir

- **Position 706 du run 1150** non vérifiée (teacher-forcing T=1177 : ~30-50 min de prefill CPU,
  jugé redondant — même token, même signature).
- **Divergence @47** caractérisée mais pas expliquée.
- **Aucune vérification** qu'un autre token supprimé/spécial apparaît ailleurs dans les 1150 ids.

## 11. Ce qu'il faut faire (décision Régis, 27 juil)

1. **Implémenter `generation_config.json`** dans le runner : `suppress_tokens` (logits → -inf
   avant l'argmax) et les **trois** `eos_token_id`.
2. **Aligner l'oracle `69_u8_gen_oracle.py` en même temps** — sans quoi l'angle mort persiste et
   le gate ne pourra pas trancher.
3. **Reprendre des témoins** : ceux du 27 juil (`/data/gemma4-zml-probe/rp0_witness/`) figent le
   token `<image|>` et deviendront caducs.
4. **Ensuite seulement**, le chantier repetition penalty déjà spécifié — `suppress_tokens` et la
   penalty vivent dans la même couche `LogitsProcessor`.

## Artefacts

- Témoins gelés (HEAD `57e1322`/`21b7b7e`, source runner identique) :
  `/data/gemma4-zml-probe/rp0_witness/` — 3 jeux d'ids + 3 dumps HLO de 510 fichiers.
- Mesures de l'investigation : `/data/tf_probe/` (`tf200.json`, `ab_generate.json`, `ab57.json`).
