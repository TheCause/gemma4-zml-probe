# Spec — Appliquer `generation_config.json` au décodage (suppress_tokens + 3 EOS)

> **Date** : 2026-07-28 · **Niveau de travail** : meute (ultracode — décision Régis, 28 juil) ·
> **Statut** : **révision 1** (après revue adversariale 5 lentilles, 48 findings arbitrés).
> **Base** : `docs/FINDING_GENERATION_CONFIG.md` (établi 27 juil, commit `159b27a`) ; PR #17 mergée
> (`2d4cda6`) ; branche `sampling-penalty` (8 commits, docs seulement, rien de mergé).
> **Demande Régis (27 juil)** : « corriger `generation_config` AVANT la penalty, en alignant
> l'oracle dans le même mouvement, et reprendre des témoins ».
>
> **Décisions Régis (28 juil, prises sur cartographie mesurée)** :
> 1. **Périmètre = 12B seul** (`gemma4_g12auto` + wrappers 4k/8k + oracle `69`). Runners E2B : dette
>    documentée (§3).
> 2. **La passe de nuance sur la claim se fait DANS ce chantier.**
> 3. **Mode `--oracle` : suppression APPLIQUÉE, arrêt EOS DÉSACTIVÉ.**
>
> **Exigence Régis (28 juil)** : « je veux un résultat **scientifiquement falsifiable** » ⇒ §2bis
> (claims + conviction + prédiction chiffrée + ce qui tue la claim), committée **avant** la première
> mesure. Cadre : `methode_scientifique.md` §P1.
>
> **Convention de citation** : sauf mention contraire, toutes les références `transformers` sont
> sur **5.14.1** (venv `g12b`, celui qui charge le 12B). Les numéros de 5.9.0 sont signalés comme
> tels — c'est un défaut attrapé en revue : 6 ancrages de la rév. 0 étaient sur 5.9.0, dont 3
> pointaient du code sans rapport.

---

## 1. Contexte et problème

### 1.1 Le fait établi (finding, 27 juil)

En greedy pur, `gemma4_g12auto` émet `258882 <image|>` au milieu d'un texte (~1×/500-600 tokens).
L'A/B à un seul facteur sur le vrai 12B **innocente le forward ZML** à cette position : ses logits
reproduisent HF, et `<image|>` gagne de 0,2499 (pas un tie bf16). Ce qui manque est la **politique
de décodage** :

```
grep -rn "suppress\|generation_config" zml_runner/*.zig            → 0 occurrence
grep -n  "suppress\|generation_config" scripts/69_u8_gen_oracle.py → 0 occurrence
```

`generation_config.json` déclare `suppress_tokens: [258883, 258882]` et **trois**
`eos_token_id: [1, 106, 50]`. Le runner fait un argmax nu et ne s'arrête que sur `106`.

### 1.2 Ce que la cartographie et la revue ont ajouté (faits neufs, tous ancrés)

| # | Fait | Ancrage | Conséquence de design |
|---|---|---|---|
| F1 | Le topK est in-graph, **mais le host reçoit le top-5 complet** (5 `i32` + 5 `f32`) | `gemma4_g12auto.zig:825`, `:1366-1384` | Suppression implémentable **côté host, graphe intact** — §4.2 |
| F2 | **Aucun `106` en dur** : `eot_id` est mesuré en encodant `"<turn|>"`, garde `error.EotNotSingleToken` | `:899-906` | Passage scalaire → ensemble, pas un « dé-durcissement » |
| F3 | La fixture du gate U8 contient `106` **dès l'index 1** | `fixtures/u8_gen48_manifest.json` | Câbler les EOS en mode `--oracle` ferait passer U8 de 48 à 1 token ⇒ décision Régis n°3 |
| F4 | Les **6 jeux d'ids 12B inspectables** ne contiennent **aucun** de `258882/258883/1/50` | `logs/*.json` + headers safetensors | Correction **non-régressive sur tout le corpus 12B vérifiable** ; claim « 4041 positions » intacte |
| F5 | `generation_config.json` **n'est pas dans le répertoire de checkpoint** : `weights_12b/` ne contient que 2 symlinks | `ls` VM, 28 juil | Découverte par `readLink` d'**un seul hop** — §4.1 |
| **F6** | **`--oracle` n'est PAS du teacher-forcing** : `fed = tok` est **inconditionnel** ; le mode ne change que la **condition d'arrêt** (`generated.len >= fx.len`) | `:1421-1422` vs **`:1438`** | La rév. 0 justifiait « pas d'impact car teacher-forcé » : **justification fausse**. La bonne raison est F4 (aucun id supprimé dans ces corpus). Corrigé en §4.4 |
| **F7** | La fixture E2B `gen_auto_long` (999 ids, **prompt du finding**) **a été inspectée** depuis M1 : **0 occurrence** de `258882/258883/1/50` dans `fed` et `expected` | `<montage local du /data VM>/gemma4-zml-probe/gen_auto_long.safetensors`, 28 juil | La rév. 0 la déclarait « non inspectable » : **faux**, mesurable en 10 s. C5 passe de *probable* à *établie* |
| **F8** | `70_u8_corrupt.py` (contre-test **D11**) écrit son checkpoint corrompu **à plat** dans `ROOT/`, **ni snapshot ni symlink** | `scripts/70_u8_corrupt.py:30` (`DST = ROOT/weights_12b_corrupt.safetensors`) | La règle « erreur dure si introuvable » **casserait D11**. Traité en §4.1 et §8 |
| **F9** | Zig **0.16.0-dev.2722** ; le projet utilise **`std.Io.Dir`**, pas `std.fs.Dir` | `MODULE.bazel:52` ; `gemma4_g12auto.zig:118` | L'API de découverte doit être `std.Io`, signature à confirmer à l'implémentation — §4.1 |

### 1.3 L'écart qui compte le plus

`69_u8_gen_oracle.py` fait un argmax nu **à deux endroits** (`:133` dans `step_top5`, `:304` dans la
boucle teacher-force, qui **n'appelle pas** `step_top5`). Aucun gate existant ne pouvait détecter
l'écart : l'instrument n'est pas dégradé, il est **structurellement aveugle au même endroit que le
sujet**. Une correction du seul `step_top5` laisserait aveugle le mode qui a produit le « 199/200 ».

---

## 2. Objectif et critères de succès

1. **Le runner 12B applique `generation_config.json`** : `suppress_tokens` avant la sélection, arrêt
   sur les **trois** `eos_token_id` en mode libre.
2. **La politique est LUE, jamais codée en dur** (l'E2B n'a **pas** `suppress_tokens` : coder
   `258882` en dur serait faux pour lui).
3. **L'oracle `69` applique la même politique, dérivée de la source de vérité** — le **vrai**
   `SuppressTokensLogitsProcessor` instancié depuis `model.generation_config`, jamais une
   ré-implémentation à la main (règle d'or, payée au bug `v_norm` D.0→D.0b).
4. **Le graphe ne bouge pas** : HLO pré-optimisation byte-identique au témoin.
5. **Non-régression prouvée** sur les jeux d'ids mesurés inchangés (F4).
6. **Le mordant est prouvé** : `<image|>` présent AVANT, absent APRÈS, position de divergence
   publiée.
7. **Périmètre de claim** : « ids == HF » est **vraie** au sens « même argmax sur les **logits
   bruts** », **fausse** au sens « reproduit `generate()` ». Après ce chantier la seconde lecture
   devient vraie **pour le 12B en mode libre**, et reste fausse pour les runners E2B — **il est
   interdit d'écrire l'inverse**.

---

## 2bis. Claims falsifiables — prédictions PRÉ-ENREGISTRÉES

> Cadre : `methode_scientifique.md` §P1 — « chaque claim a un *what would kill this claim ?*
> explicite » ; « kill fast : les seuils sont définis AVANT de lancer ».
>
> **Le pré-enregistrement est vérifiable** : cette spec est committée **avant la première mesure**.
> Le `git log` fait foi. Toute valeur mesurée qui contredit une prédiction est publiée **telle
> quelle** ; une requalification exige une décision Régis écrite.

### Le seuil de kill, dérivé (et non deviné)

Le bruit résiduel ZML↔oracle fp32 mesuré au gate U7 est **`max_abs = 9,365e-4`** — c'est l'écart
maximal **sur UN logit** (`logs/u7_fp32_gate.log:23`, id 7001 : `zml=-1.061507e1`,
`hf=-1.061600e1`). Un **flip d'argmax** exige que **deux** logits se croisent : si chacun peut
bouger de ±δ, une inversion est attribuable au bruit dès que la marge top1−top2 ≤ **2δ**.

> **Seuil de kill retenu : `2 × 9,365e-4 = 1,873e-3` sur la MARGE.**

Précédent interne qui confirme la méthode : la spec penalty rév. 4 (`…-sampling-repetition-penalty-design.md:299`)
majore le même 9,365e-4 avant de s'en servir comme seuil. Utiliser le chiffre brut sur une marge
était une **erreur d'un facteur 2**, attrapée en revue.

### C1 — « Le forward ZML est innocenté : sur ce prompt, le seul écart à `generate()` est la politique de décodage »

- **Conviction : possible.** L'A/B du finding §4 ne l'établit qu'à **une seule position** (57).
- **Prédiction** : après correction, la séquence du runner en mode libre est **identique**, position
  par position, à celle que produirait `model.generate()` sur le même checkpoint, jusqu'à l'arrêt.
- **Ce qui tue C1** : une divergence d'id à une position de marge **> 1,873e-3**. Sous ce seuil,
  c'est un tie : publié comme tel, avec sa marge.
- **Porté par** : GC8.

⚠ **C1 EST DÉJÀ EN DIFFICULTÉ, et il faut le dire avant de mesurer.** Le finding §9 rapporte une
divergence en roue libre **dès l'index 47** (`5743 ▁zero` vs `27069 ▁humanity`), marge **0,004587**
— soit **2,4× au-dessus du seuil de kill**. Elle est antérieure à 57 et **sans rapport avec
`suppress_tokens`**. Deux lectures, non départagées :

  - (a) artefact du protocole de comparaison ⇒ C1 tient ;
  - (b) écart réel du forward ⇒ **C1 est réfutée**, et le finding n'innocente le forward que
    localement.

**Conséquence de spec** : C1 ne peut pas être énoncée tant que @47 n'est pas instruite. Soit GC9
l'instruit, soit **C1 est bornée à « jusqu'à la position 46 »**, le reste déclaré non prouvé.
**Interdit d'énoncer C1 sans l'un ou l'autre.**

### C2 — « La suppression sur le top-5 est exactement équivalente à la suppression sur le vocabulaire complet »

- **Conviction : certain** par l'argument de rang (§4.2) — **sauf** égalité exacte de logits, où le
  tie-break de `sort` (ZML) peut différer de celui de `argmax` (torch).
- **Prédiction** : `rank_used ∈ {0, 1, 2}` à **200/200** ; aucun step à `rank_used ≥ 3`.
- **Ce qui tue C2** : un step à `rank_used ≥ 3` (la théorie l'interdit avec `|S| = 2` ⇒ ce serait un
  bug d'implémentation), **ou** un step avec deux logits du top-5 égaux **au bit près** au rang
  décisif.
- **Mesure neuve exigée** (jamais faite dans ce projet) : histogramme de `rank_used` **et** nombre
  de steps présentant une égalité exacte dans le top-5. Sans elle, la réserve sur le tie-break reste
  une hypothèse commode plutôt qu'un fait.

### C3 — « Le graphe n'a pas bougé »

- **Conviction : certain** (politique purement host).
- **Prédiction** : md5 du HLO pré-optimisation **identique** au témoin, à l'octet.
- **Ce qui tue C3** : un md5 différent ⇒ le design host a fui dans le graphe, l'argument « coût D2H
  nul » tombe.

### C4 — « L'oracle ne partage plus l'angle mort »

- **Conviction : certain** si GC5 est exécuté comme écrit.
- **Prédiction, à l'id près** : au contexte forcé du finding §4, oracle **nu** → `258882` ; oracle
  **avec politique** → **`11814 ▁hero`**. Pas « un token différent » : **cet id précis** (valeur
  mesurée dans `/data/tf_probe/ab57.json`).
- **Ce qui tue C4** : les deux branches donnent le même token, **ou** la branche « politique » donne
  autre chose que `11814`.

### C5 — « La correction est non-régressive sur le corpus existant »

- **Conviction : établie** (et non plus « probable ») — les 6 jeux 12B (F4) **et** la fixture E2B
  `gen_auto_long` (F7, 999 ids sur le prompt du finding) ont été inspectés : **0 occurrence**.
- **Prédiction** : `n_suppress_hits == 0` **et** ids bit-identiques sur les témoins 48 et 124.
- **Ce qui tue C5** : `n_suppress_hits > 0` sur ces témoins, ou un id qui bouge.

### Grandeurs prédites AVANT mesure

| Grandeur | Valeur prédite | Fondement |
|---|---|---|
| `n_suppress_hits`, témoin 200 tokens | **≥ 1, dont la position 57** | le finding relève `258882` @57. ⚠ **Pas « exactement 1 »** : après correction la trajectoire diverge dès 57, les 142 positions suivantes sont un **autre texte**, sur lequel rien n'a été mesuré |
| Position de la 1ʳᵉ divergence AVANT/APRÈS, témoin 200 | **57** | greedy déterministe |
| Ids 0..56 après correction vs témoin AVANT | **bit-identiques** | toute divergence antérieure réfute l'isolation du facteur |
| `n_suppress_hits`, témoins 48 et 124 | **0** | F4 |
| `rank_used` max, tous témoins | **≤ 2** | argument de rang, \|S\| = 2 |
| GC5 : `verdict_data.n_match` | **199/200 → exactement 198/200** | `{258882,258883}` n'est rang 1 qu'à la position 57 sur les 200 ; hors 57 la marge du rang 1 non supprimé est ≥ 2,78 (insensible au dtype) ; le mismatch @47 (`27069`, non supprimé) subsiste. **Cette baisse de 1 est le résultat ATTENDU, pas une régression** — toute autre valeur arrête GC5 |
| `n_suppress_hits`, témoin 1150 tokens | **≥ 1** | même raison que le témoin 200 : au-delà de 57 la trajectoire est neuve. La position 706 du témoin AVANT n'est **pas** transposable |

**Si une seule de ces valeurs tombe à côté, le résultat est publié avec l'écart et le chantier
s'arrête pour diagnostic** — pas de « c'est probablement parce que… ».

---

## 3. Non-objectifs (et où vit la dette)

- **Le sampling stochastique** (`do_sample: true, top_k: 64, top_p: 0.95`) — chantier suivant.
- **La repetition penalty** — chantier déjà spécifié ; sa **composition** avec cette couche est
  cependant traitée ici (§4.7), parce que l'ignorer ferait revenir le bug.
- **Les runners E2B** (`gen_auto`, `w4auto`, `bbatch`) et les 5 oracles E2B (44/45/46/49/56).
  **Domicile de la dette** : une entrée dédiée au `PLANNING.md` **et** une section « Dette connue »
  de `docs/GENERATION_CONFIG_RESULTS.md`, avec le contenu factuel exact :
  - les **3 EOS manquent aussi** aux runners E2B (`gen_auto:1103`, `w4auto:1107`,
    `bbatch:1349`) — et `gemma4-e2b-it-meta/config.json:52-55` déclare bien `eos_token_id: [1, 106]` ;
  - `suppress_tokens` en revanche est **absent** du `generation_config.json` E2B **et** de
    `weights_w4/` (vérifié) : pour l'E2B, la dette est **EOS-only** ;
  - `bbatch` a un piège propre : `eot_id` y sert aussi de **token de bourrage** des lanes finies
    (`:1360`) — un ensemble d'EOS y demande **un** id de bourrage, pas l'ensemble.
- **`begin_suppress_tokens`** — absent des trois configs du parc. Non implémenté, mais **refus
  bruyant** s'il apparaît (§4.1).
- **La divergence @47** — instruite par GC9 **si** GC9 est retenu ; sinon C1 est bornée (§2bis).
- **La récitation** qui a motivé le chantier penalty — **réfutée** sur 3 témoins greedy.

---

## 4. Design

### 4.1 Source de vérité : découvrir, lire, VALIDER `generation_config.json`

**Le problème (F5)** : le runner reçoit `<model.safetensors> <tokenizer.json>` en positionnels
(`:125-142`). `dirname(ckpt)` = `weights_12b/`, qui ne contient que deux symlinks. Le fichier est
dans le snapshot pointé. Un `realpath` **complet** atterrirait dans `blobs/` (nom de hash), qui ne
le contient pas non plus.

**Règle de découverte** (miroir de `scripts/63_u_dequant_export.py:70-71`, qui traite déjà ce cas en
Python avec le commentaire « UN seul hop, `realpath` interdit ») :

```
1.  --gen-config <path> fourni   → utiliser ce chemin. Illisible ⇒ erreur dure.
2.  sinon : dirname(ckpt)/generation_config.json
3.  sinon, si ckpt est un symlink → dirname(readLink(ckpt))/generation_config.json   [UN hop]
4.  sinon                        → error.GenerationConfigNotFound, message listant LES CHEMINS ESSAYÉS
```

⚠ **API (F9)** : Zig `0.16.0-dev.2722`, le projet utilise **`std.Io.Dir`** (`:118`), pas
`std.fs.Dir`. La signature exacte de `readLink` dans cette API **doit être confirmée à
l'implémentation** — c'est une étape de plan, pas une hypothèse de spec. Le comportement attendu si
le chemin **n'est pas** un symlink doit être capturé explicitement (ce n'est pas une erreur : c'est
le cas 4).

**Pas de repli silencieux** : un runner qui ne trouve pas sa politique **refuse de tourner**.

**Échappatoire explicite** : `--no-gen-config` restaure exactement le comportement d'avant le
chantier. Ce n'est pas une commodité, **c'est l'instrument du contre-test GC4(a)** — même binaire,
une donnée de moins.

⚠ **Effet de bord mesuré sur le contre-test D11 (F8)** : `70_u8_corrupt.py` écrit
`ROOT/weights_12b_corrupt.safetensors`, **à plat, sans snapshot ni symlink** ⇒ les 4 étapes
échouent ⇒ **D11 deviendrait inexécutable**. Traitement retenu : D11 passe `--gen-config <chemin du
dq>` explicitement (le contre-test porte sur les **poids**, pas sur la politique). **À écrire dans
le doc de résultats et dans le plan** — sinon un gate historique meurt en silence au prochain usage.

**Log obligatoire, une ligne par run** (ce que les gates grepperont) :

```
GENCFG: <chemin résolu> suppress=[258883,258882] eos=[1,106,50] (mode=libre|oracle)
GENCFG: DÉSACTIVÉ (--no-gen-config) — politique de décodage non appliquée
```

**Clés lues** : `suppress_tokens` (liste, optionnelle), `eos_token_id` (scalaire **ou** liste — HF
normalise `int → [int]`, `stopping_criteria.py:544-549`).

**Validation des valeurs lues** (aucune n'était spécifiée en rév. 0) :

| Cas | Décision | Pourquoi |
|---|---|---|
| `begin_suppress_tokens` présente | `error.BeginSuppressUnsupported` | sémantique temporelle différente (`logits_process.py:1863-1871`), non implémentée |
| id de suppression **négatif** ou **≥ vocab** | `error.SuppressIdOutOfRange` | HF l'ignorerait silencieusement (`isin` sur `arange(V)`) : ici on refuse |
| doublons dans `suppress_tokens` | dédupliqués, **et logués** | sans effet sur HF, mais fausserait la garde `len + 1 > TOP_K` |
| `suppress_tokens` **liste vide** ou clé absente | **accepté**, log explicite `suppress=[] (aucune suppression)` | c'est le cas E2B légitime — mais alors la suppression est **inerte** et aucun gate de mordant ne doit être déclaré PASS |
| `eos_token_id` vide ou absent | `error.EosListEmpty` | un décodage sans aucun EOS ne s'arrêterait que sur `max_tokens` |
| **`eot_id` mesuré ∉ `eos_token_id`** | `error.EotNotInEosList` | contrôle croisé tokenizer ↔ config ; **exercé dans les deux sens** au GC1 |

### 4.2 Où s'applique la suppression : côté host, sur le top-5 — et pourquoi c'est EXACT

**Le fait (F1)** : le graphe sort déjà un top-5 trié décroissant, rapatrié à chaque step (~48 o).
`topK` délègue à `sort` descendant puis `slice1d(0..k)` (`zml/tensor.zig:3096-3110`) — le top-5
**est** les rangs 1..5 de l'ordre décroissant complet.

**L'argument d'exactitude** : `|S| = 2`. L'argmax post-suppression est le premier élément de l'ordre
décroissant hors `S` ; au plus 2 éléments sont retirés au-dessus de lui ⇒ **rang brut ≤ 3**, donc
dans le top-5 déjà rapatrié.

**Garde qui rend l'argument vrai par construction** :

```zig
if (suppress.len + 1 > TOP_K) return error.SuppressListTooLongForTopK;
```

⚠ `TOP_K` **ne doit pas devenir une quatrième copie** de la constante `5` : le `5` existe déjà dans
`G12Step.forward` (`:825`), dans la boucle de lecture (`:1380`) et dans le type `Top5`. La spec
impose **une seule** déclaration, référencée par les autres — sinon un futur passage à K=8 laisse
une garde périmée qui *valide* un design devenu faux.

**Conséquences** : `engine.zig` **0 octet** ; `G12Step.forward` **inchangé** ⇒ HLO byte-identique
(GC0) ; **zéro D2H supplémentaire**.

**La sélection** (point d'insertion : juste avant l'affectation de `tok`, `:1384` — ⚠ ne **pas**
redéclarer `tok`, qui est déjà défini là) :

```zig
var chosen: ?i32 = null;
var rank_used: usize = 0;
for (0..TOP_K) |j| {
    if (!policy.isSuppressed(top5.idx[j])) { chosen = top5.idx[j]; rank_used = j; break; }
}
const tok: i64 = @intCast(chosen orelse return error.AllTopKSuppressed);

// Compté SEULEMENT en phase de génération : en prefill le token est JETÉ
// (:1414-1418, « argmax ci-dessus IGNORÉ »). Même garde que les lignes voisines
// :1385/:1387/:1389. `in_gen_phase` (:1363) est VRAI au dernier step de prefill —
// celui qui produit s0 — donc la politique s'applique bien au 1er token généré.
if (in_gen_phase and rank_used != 0) n_suppress_hits += 1;
```

⚠ **La sélection reste inconditionnelle** ; seul le **compteur** est gardé par `in_gen_phase`.
Garder la sélection sous `in_gen_phase` serait un vrai bug : `s0` ne serait plus filtré.

`error.AllTopKSuppressed` est inatteignable sous la garde ci-dessus, mais existe pour qu'un futur
élargissement échoue **bruyamment**.

**Compteur `n_suppress_hits`** = nombre de steps **de génération** où la suppression a réellement
changé le token choisi. Dénominateur : **les tokens générés** (`generated.items.len`), pas les
steps — le prefill (27-28 steps sur les témoins) ne doit pas diluer la mesure :

```
GENCFG: suppress a mordu {N} fois sur {generated.items.len} tokens générés (prefill exclu)
```

C'est **le détecteur de vacuité intégré** : un gate de mordant qui rapporte `N = 0` n'a rien prouvé.

⚠ **Ce que ce design NE reproduit PAS** : HF écrit `-float("inf")` puis fait `argmax` sur le vecteur
complet (`logits_process.py:1906-1911`, `utils.py:2938`). Nous sélectionnons dans un top-5 pré-trié.
Les deux coïncident **sauf en cas d'égalité exacte**, où le tie-break de `sort` peut différer de
celui d'`argmax` (**piège 15** connu ; tie-break de `torch.argmax` **non vérifié**). C2 exige de
**compter** ces égalités, pour savoir si la réserve est théorique ou mordante.

### 4.3 Les trois EOS

**Sémantique HF, source primaire 5.14.1** :
- `EosTokenCriteria` (`stopping_criteria.py:534`, `__init__` 544-549, `isin` 579-581) =
  `isin(dernier_token, [1,106,50])` → **« any of »**, sans priorité ;
- combinaison inter-critères = **OU** (`StoppingCriteriaList:605`, OU 608-611) ;
- le token est **concaténé PUIS testé** (`utils.py:2945` puis `:2949`) ⇒ **l'EOS fait partie de la
  sortie**.

**Le runner fait déjà la bonne chose sur la conservation** (token gardé, strippé à la détok,
`:1519-1521`). **Seul l'élargissement 1 → 3 est à faire**, sur le test `:1424`.

**À corriger en même temps** : le log de fin affiche `EOT_ID={d}` (`:1542-1543`) — il afficherait
une valeur fausse si l'arrêt venait de `1` ou `50`. Il doit consigner **quel** EOS a arrêté.

**Le problème d'exerçabilité** (leçon `feedback_controle_qui_ne_peut_pas_reussir`) : aucun corpus
mesuré ne contient `1` ni `50` (F4). Un gate posé sur les prompts existants serait un **contrôle qui
ne peut pas réussir**.

**Réponse — GC6** : exercer le chemin multi-EOS via un `generation_config` **fabriqué**. Contenu
minimal imposé, sinon il se ferait refuser par les validations de §4.1 :

```json
{ "eos_token_id": [1, 106, 50, <ID_FRÉQUENT>], "suppress_tokens": [258883, 258882] }
```

⚠ **Le `106` doit rester dans la liste** : sinon `error.EotNotInEosList` (contrôle croisé §4.1)
refuse le fichier — la rév. 0 se mordait la queue ici. `<ID_FRÉQUENT>` est **mesuré sur le témoin
AVANT** (un id présent tôt dans la séquence), pas choisi au jugé. Le run doit s'arrêter
**exactement** à sa première occurrence, `stop_reason` **nommant** cet id.

### 4.4 Les trois modes — corrigé (F6)

| Mode | Sélection du token | `suppress_tokens` | arrêt EOS |
|---|---|---|---|
| **libre** (`--prompt`, `--repl`, `--max-tokens`) | autonome, réinjectée | **APPLIQUÉ** | **APPLIQUÉ** — any-of `[1,106,50]`, token conservé, strippé à la détok |
| **`--oracle <fixture>`** — ⚠ **génération AUTONOME**, pas du teacher-forcing : `fed = tok` est inconditionnel (`:1438`) ; le mode ne change que la **borne d'arrêt** (`generated.len >= fx.len`, `:1421-1422`) | autonome, réinjectée | **APPLIQUÉ** | **DÉSACTIVÉ** |
| **`--window-vacuity`** | **AUCUNE** — chemin séparé qui feede une séquence imposée, `return` avant `generateOnce` ; « l'argmax est ignoré » | **SANS OBJET** | sans objet |

**Justification de la ligne `--oracle`** (décision Régis n°3) : la fixture borne le nombre de
positions comparées ; l'arrêt n'y a pas de sens, tandis que la suppression doit être exercée des
deux côtés — sinon l'angle mort du finding §5 persiste.

⚠ **Correction de raisonnement (F6)** : la rév. 0 concluait « les gates U8/W4g survivent car le mode
est teacher-forcé ». **C'est faux** : le runner y génère librement, donc appliquer la suppression
**peut** changer sa trajectoire. Ce qui sauve les gates est **F4** : aucun id supprimé n'apparaît
dans ces corpus, donc la trajectoire ne bouge pas. Conclusion identique, **fondement différent** —
et c'est le fondement qui doit être écrit, pas la conclusion.

**Corollaire non négociable** : la suppression en mode `--oracle` n'est correcte **que si l'oracle
Python applique exactement la même** (§4.5). Runner et oracle changent dans le même commit.

### 4.5 Côté oracle — `scripts/69_u8_gen_oracle.py`

**Deux sites** : `step_top5` (`:127-134`) et la boucle chunk (`:300-307`).

**Source de vérité** — le processor de transformers, depuis l'objet modèle :

```python
from transformers.generation.logits_process import SuppressTokensLogitsProcessor
gc = model.generation_config                        # peuplé par from_pretrained
proc = SuppressTokensLogitsProcessor(gc.suppress_tokens) if gc.suppress_tokens else None
```

(`logits_process.py:1903` : `__init__(self, suppress_tokens, device="cpu")` — `device` optionnel.
`configuration_utils.py:427` : défaut `None`. Une liste **vide** est falsy ⇒ `proc = None`.)

**Flag de bascule — `--no-gen-policy`** (manquait en rév. 0, ce qui rendait GC5 **non exécutable
sans éditer le script**, précisément ce que GC5 interdit). Nom **délibérément différent** du
`--no-gen-config` Zig : côté Python aucun fichier n'est lu, la source est `model.generation_config`
peuplé par `from_pretrained` ; un nom miroir laisserait croire à un jumeau `--gen-config <path>` qui
n'existe pas. Défaut = **politique ON**, effet = sauter le processor **aux deux sites**.

**L'ordre imposé** (`69:130` et `:132` sont deux gardes **saines** qui surveillent le softcap ;
`abs(-inf) = inf > 30` les ferait tomber) :

```
1. asserts isfinite / max_abs ≤ SOFTCAP  → sur les logits BRUTS
2. politique                             → sur une COPIE
3. topk                                  → sur la copie
```

⚠ **Publier LES DEUX top-5** : appliquer la politique avant le `topk` ferait archiver un top-5
**post-suppression**, et l'oracle **perdrait la trace du top-5 brut** — or c'est lui qui documente
le finding (`258882` rang 1 @57, marge 0,2499) et lui qui alimente les gates U7/U8. Le manifest doit
porter `top5_raw` **et** `top5_policy`.

**À consigner en sortie**, à côté de `versions` :
`gen_policy: {applied: bool, suppress_tokens: [...]|null, source: "model.generation_config",
script_md5: "<md5 du fichier exécuté>"}`. Le `script_md5` n'est pas un bonus : **GC7 l'exige déjà**
et aucun champ de ce nom n'existe dans le dict `report` (`69:326-346`).

**Sérialisation** : `-inf` n'est pas du JSON valide (`json.dump` écrit `-Infinity`). Avec `|S| = 2`
et `V = 262 144`, aucun `-inf` ne peut entrer dans un top-5 post-suppression — **assertion explicite
avant écriture**, pour que l'invariant soit vérifié plutôt que supposé.

**Version** : venv `g12b` = **transformers 5.14.1** (`gemma4-probe` 5.9.0 ne charge pas
`gemma4_unified`). `SuppressTokensLogitsProcessor` et `_get_logits_processor` sont **identiques**
entre 5.9.0 et 5.14.1 ; seul `EosTokenCriteria` diffère (paramètre `new_token_length=1` par défaut,
sémantiquement équivalent). `requirements.txt` ne pin rien ⇒ **l'oracle asserte sa version**.

⚠ **La VM n'est pas un dépôt git** et sa copie de `69_u8_gen_oracle.py` est **périmée** (322 lignes,
md5 `3f07844e…`, sans `--compute-fp32`). Pire, c'est **déjà arrivé** : `/data/tf_probe/tf200.json`
porte `host: "VM"` et `compute_dtype: float32` — donc un script plus récent y a tourné le 27 juil
**sans laisser de trace au chemin canonique**. R4 n'est pas de la paranoïa, la règle a déjà été
enfreinte une fois. **Contrôle md5 des deux côtés, bloquant, avant tout run.**

### 4.6 Ordre des processors — ce que le chantier suivant devra respecter

Source primaire 5.14.1, `generation/utils.py:1087` (`_get_logits_processor`) :

```
… RepetitionPenalty … SuppressTokens (1234-1240) … | if do_sample: (1260) … TopK … TopP …
```

1. `SuppressTokens` est **hors du bloc `if do_sample`** ⇒ il s'applique **aussi en greedy**.
2. La penalty s'applique **AVANT** la suppression dans HF. Inverser est **inoffensif tant que la
   valeur est `-inf`** (`-inf × p = -inf`), et cesserait de l'être avec `finfo.min`
   (`finfo(f32).min × 1.1` déborde) ⇒ **argument de plus pour rester sur `-inf`**.
3. `temperature: 1.0` ⇒ HF **n'instancie pas** `TemperatureLogitsWarper`. Ne pas porter une division
   par 1.0 « pour la forme ».

*(Le softcap est appliqué dans le modèle, pas dans `generate()` — `modeling_gemma4.py:1867-1871` en
5.9.0, **non re-vérifié en 5.14.1**.)*

### 4.7 Composition avec la repetition penalty — la contrainte à poser MAINTENANT

Le chantier suivant remplace le chemin de sélection : il rapatrie les **262 144 logits** et choisit
le token sur le vecteur complet. **Si rien n'est spécifié, la couche top-5 de §4.2 est purement et
simplement contournée, et `<image|>` revient** dès que `--repetition-penalty ≠ 1.0`. Ce serait un
retour de bug silencieux, dans le chantier même que Régis a ordonné de faire *après* celui-ci
précisément pour l'éviter.

**Contrainte pré-enregistrée, à honorer par le chantier penalty** :

1. La politique de suppression est **une fonction unique** (`gencfg.isSuppressed`), appelée par
   **les deux** chemins de sélection — top-5 (ici) et vocabulaire complet (penalty).
2. Sur le chemin complet, l'ordre est celui de HF : **penalty d'abord, suppression ensuite** (§4.6).
3. Le chantier penalty **rejoue GC3** (mordant) avec `--repetition-penalty 1.2`. Critère :
   `<image|>` reste absent. **Ce gate est pré-enregistré ici**, pas là-bas.
4. Les deux détecteurs de vacuité (`n_suppress_hits` et l'index de divergence) doivent rester
   lisibles quand la penalty est active : leur sémantique change (la trajectoire diverge pour deux
   raisons), donc le doc de résultats penalty devra les **désambiguïser**, pas les réutiliser tels
   quels.

---

## 5. Gates pré-enregistrés

| Gate | Contenu | Critère PASS (pré-enregistré) | Machine |
|---|---|---|---|
| **GC0** | Le graphe n'a pas bougé | md5 de `module_0001.zml.before_optimizations.txt` **identique** au témoin pris AVANT toute édition ; **garde** : les deux dumps existent et sont non vides | 3090 |
| **GC1** | Selftest **sans GPU** : parsing + découverte 1-hop + validations §4.1 + sélection top-5, contre une fixture produite par le **vrai** `SuppressTokensLogitsProcessor` | Token choisi identique sur 100 % des cas ; le selftest **asserte sa propre non-vacuité** (≥ 1 cas où un id supprimé est top-1, ≥ 1 cas `EotNotInEosList`, ≥ 1 cas `SuppressIdOutOfRange`) et FAIL si un compteur vaut 0 | **3090** (⚠ pas M1 : **aucun toolchain Zig** n'y compile ces sources ; la fixture, elle, est produite côté oracle) |
| **GC2** | Non-régression : témoins 48 (`mi_witness_1280`) et 124 (`mi_witness_4k`) | ids **bit-identiques** ; `n_suppress_hits == 0` **et** la ligne `GENCFG:` présente avec `suppress=[258883,258882]` — ⚠ sans ce second contrôle, GC2 ne distingue pas « politique correcte » de « politique pas branchée » | 3090 |
| **GC3** | Mordant : prompt du finding, 200 tokens, AVANT vs APRÈS | `258882` **présent @57** AVANT ; **absent** APRÈS ; `n_suppress_hits ≥ 1` (prefill exclu) ; ids 0..56 **bit-identiques** au témoin AVANT ; **le token @57 après correction est `11814`** (pas « un autre token ») ; index de 1ʳᵉ divergence **publié** | 3090 |
| **GC4** | Non-vacuité, 3 sous-tests | (a) `--no-gen-config` **sur le prompt de GC3** (pas sur le corpus GC2, où il serait vide) reproduit le témoin AVANT **bit-à-bit**, `258882` **de retour @57** ; (b) `--gen-config` pointant `suppress_tokens: []` → **le run va au bout** et `258882` revient @57 (⚠ **pas** une erreur au chargement : un contre-test satisfait par un refus ne prouve rien) ; (c) liste de 5 ids → `error.SuppressListTooLongForTopK` | 3090 |
| **GC5** | L'oracle ne partage plus l'angle mort. Deux runs du **même script**, **un seul flag** de différence, tous deux `--compute-fp32` (rester en fp32 : `tf200.json` l'est) | Lu sur `top5_per_pos[57].ids[0]` : branche **nue** → `258882` (**reproduction** de `tf200.json`, ids attendus `[258882, 11814, 3495, 1548, 13186]`, marge 0,249853 — si elle ne reproduit pas, c'est l'instrument qui est en cause) ; branche **politique** → **`11814`**. Et `n_match` : **199/200 → exactement 198/200** | oracle (`g12b`) |
| **GC6** | Chemin multi-EOS **exercé en run réel** via `generation_config` fabriqué (§4.3, `106` conservé) | Arrêt **exactement** à la 1ʳᵉ occurrence de `<ID_FRÉQUENT>` ; `stop_reason` **nomme** cet id ; sans le fichier fabriqué, le même run va au bout | 3090 |
| **GC7** | Équivalence runner ↔ oracle sur le témoin, **les deux avec politique** | `n_match == n_total` ; le rapport consigne `versions.transformers` **et** `script_md5` | 3090 + oracle |
| **GC8** | **Test décisif de C1** : trajectoire libre du runner corrigé vs décodage HF appliquant la même politique, **arrêt compris** | Séquences identiques jusqu'à l'arrêt **ou** divergences toutes **< 1,873e-3** (publiées avec leur marge). Le `mode_decode` de l'oracle (`69:137-225`) fournit la boucle : y armer politique **et** arrêt EOS | oracle + 3090 |
| **GC9** | **Instruire la divergence @47** (finding §9, marge 0,004587) : logits ZML vs HF **à cette seule position**, oracle fp32, `--dump-top5` | Verdict **publié**, pas PASS/FAIL : artefact de protocole (**C1 tient**) ou écart réel du forward (**C1 réfutée**). Sans GC9, C1 est bornée à « jusqu'à la position 46 » | oracle + 3090 |
| **GC10** | **`--repl` applique bien la politique** (R11 : 3 sites d'appel de `generateOnce`, un oubli la rend silencieusement inopérante) | GC3 rejoué **en mode `--repl`** : mêmes ids, même `n_suppress_hits`, ligne `GENCFG:` présente ; **et** `n_suppress_hits` remis à zéro entre deux prompts | 3090 |
| **GC11** | **La passe de nuance sur la claim est faite** (objectif #7, sinon livrable purement déclaratif) | Le grep de recensement (commande figée au plan, exécutée avant/après) : **0 site** de catégorie (i) sans qualificatif ; **contre-preuve exigée** — réintroduire une formulation nue doit faire **échouer** le grep | M1 |

**Règle d'arrêt** : GC0, GC1, GC2 FAIL ⇒ **STOP**, diagnostiquer. Aucune requalification sans
décision Régis écrite. GC3 sans mordant (`n_suppress_hits == 0`) ⇒ **le prompt change, pas le
critère**. GC9 n'a pas d'issue FAIL : les deux verdicts sont publiables — mais l'un des deux
**réfute C1**.

**Ce qui ne sera PAS prouvé, et doit être écrit comme tel** :
- l'arrêt sur les **vrais** EOS `1` et `50` n'est exercé par aucun corpus réel (GC6 exerce la
  mécanique, pas ces ids-là) ;
- le tie-break `sort`/`argmax` en cas d'égalité exacte reste non vérifié — C2 exige au minimum de
  **compter** les égalités rencontrées ;
- les argmax des steps de **prefill** ne sont écrits nulle part et ne sont exercés par aucun gate :
  on ne sait pas si un id supprimé y est top-1. La garde `in_gen_phase` rend la question sans effet
  sur les mesures publiées — **elle ne l'instruit pas** ;
- la position 706 du témoin 1150 n'est pas re-vérifiée (et n'est pas transposable après divergence) ;
- **le périmètre E2B reste non couvert** (dette §3). La fixture `gen_auto_long` **a été inspectée**
  (F7) : elle n'est **pas** affectée — la dette est le décodage sans politique, pas cette fixture.

### Ce qui réfuterait le chantier ENTIER

1. **GC0 FAIL** ⇒ la politique host a fui dans le graphe : la thèse §4.2 est fausse, il faut un
   design in-graph avec bias rebindable, en perdant le gate de byte-identité.
2. **GC4(a) FAIL** ⇒ le chantier a changé autre chose que la politique : l'A/B à un seul facteur est
   rompu, **plus aucune mesure du chantier n'est interprétable**.
3. **GC9 conclut « écart réel du forward »** ⇒ le finding n'innocente le forward que localement, C1
   tombe, et le vrai chantier n'est pas `generation_config` mais une investigation du forward. La
   correction reste bonne à prendre, mais **la claim doit être réécrite** et la penalty attendra.

---

## 6. Risques

| # | Risque | Réponse |
|---|---|---|
| **R1** | Les 3 EOS cassent U8/W4g (fixture contenant `106` à l'index 1, mode `--oracle` sans arrêt) | Décision Régis n°3 : EOS **off** en mode `--oracle`. Vérifié par GC2 + GC7 |
| **R2** | Le gate E2 attend `[107, 1, 106, 1]` — l'EOS `1` y figure 2× | Hors périmètre (12B seul), **consigné** : si les EOS sont un jour portés sur `engine_e2`, ce gate meurt |
| **R3** | `generation_config.json` introuvable (symlink, `realpath` → `blobs/`) | Découverte 1-hop (§4.1) + erreur dure listant les chemins essayés |
| **R3b** | **La règle « erreur dure » casse le contre-test D11** (F8 : checkpoint corrompu à plat) | D11 passe `--gen-config <dq>` explicitement ; **écrit au plan et au doc de résultats** |
| **R4** | La VM exécute une version périmée de `69` — **déjà arrivé** (`tf200.json` produit par un script absent du chemin canonique) | Contrôle md5 des **deux** côtés, bloquant, avant tout run ; `script_md5` au manifest. ⚠ L'oracle a aussi tourné depuis **M4** (`~/ml-data/weights_12b_dq`) : le contrôle doit nommer l'hôte |
| **R5** | Mauvais venv : `gemma4-probe` (5.9.0) ne charge pas `gemma4_unified` | Venv imposé `g12b` (5.14.1) ; l'oracle **asserte** sa version |
| **R6** | Le patch `@setEvalBranchQuota(100_000)` de `pjrt.zig` saute si le workspace ZML est resynchronisé | Grep du patch **avant** build (vérifié présent le 28 juil : `pjrt/pjrt.zig:26`, fichier **tracké modifié non committé** ⇒ effaçable par un `git checkout`) |
| **R7** | Un `-inf` entre dans la sérialisation | Assertion explicite avant écriture (§4.5) |
| **R8** | Élargissement futur de `suppress_tokens` cassant l'argument de rang | `error.SuppressListTooLongForTopK`, **exercée** par GC4c |
| **R9** | `begin_suppress_tokens` dans un futur snapshot | `error.BeginSuppressUnsupported` |
| **R10** | Ollama recharge un modèle et occupe la VRAM | `nvidia-smi --query-compute-apps` avant chaque run (garde `MIN_FREE_VRAM_GIB = 20`). Mesuré le 28 juil : **24 576 MiB = VRAM TOTALE**, `0 MiB` utilisés, `ollama serve` **actif mais sans modèle chargé** (`/api/ps` vide) ⇒ marge réelle 24 GiB, **latente** |
| **R11** | Politique oubliée à l'un des **3 sites d'appel** de `generateOnce` (`:1244`, `:1254`, `:1283`) ⇒ inopérante en `--repl` | Passage **par pointeur d'une struct unique** + **gate GC10** dédié |
| **R12** | `n_suppress_hits` non remis à zéro entre deux prompts `--repl` ⇒ compteur cumulatif, mordant surestimé | Remise à zéro **dans** `generateOnce` ; vérifié par GC10 |
| **R13** | La penalty contourne la couche et fait revenir le bug | §4.7 : fonction unique + GC3 rejoué à `--repetition-penalty 1.2`, **pré-enregistré ici** |

---

## 7. Ordre d'exécution

```
   [artefacts AVANT — irrattrapable si oublié]
   T0 témoins (ids 200 + 1150, dumps HLO, md5)          ← AUCUNE édition avant
        │
        ├──────────────┬───────────────┐
        ▼              ▼               ▼
   T1 gencfg.zig   T2 oracle 69    T3 fixture GC1
   + selftest      (2 sites +      (produite par le
   → GC1           --no-gen-policy)  vrai processor)
        │              │               │
        └──────┬───────┴───────────────┘
               ▼
        T4 câblage runner (3 sites) → GC0, GC2, GC3, GC4, GC10
               │
               ├──────────────┐
               ▼              ▼
        T5 GC5, GC7      T6 GC6 (config fabriqué)
               │
               ▼
        T7 GC8 (test décisif C1) ─── si divergence > seuil ──▶ GC9 (instruire @47)
               │
               ▼
        T8 docs + passe de nuance (GC11) + PLANNING + PR
```

**Dépendances dures** : T0 avant toute édition (leçon M0 : « un témoin pris après une édition
partielle a coûté une investigation entière ») · GC1 avant GC2/GC3 (inutile de brûler du GPU si le
selftest échoue) · GC0 avant tout gate d'ids (si le graphe a bougé, les comparaisons ne veulent plus
rien dire) · GC5 avant GC7 (l'oracle doit être juste avant de servir de juge) · GC9 déclenché par
GC8, ou décidé d'avance si Régis veut borner C1.

## 8. Gestion d'erreur et reprise

- **Longs runs distants** : `nohup` + log distant + **stdin fermé** + marqueur `DONE rc=$?`, attente
  par `until grep -q "^DONE rc="` — convention du projet, payée deux fois en une session
  (`docs/superpowers/plans/2026-07-27-sampling-repetition-penalty.md`, « Conventions d'exécution »).
- **N'attendre `PASS` que des runs qui en émettent** — un filtre de guetteur trop large a déjà
  matché une bannière (« U7 »).
- **Deploy** : `ZML_REMOTE`/`ZML_DST` en env **obligatoires** (les défauts sont des placeholders et
  le deploy rate **silencieusement** avec la sortie redirigée) ; sortie `rsync` **non vide** exigée.
- **Build GPU** : `--@zml//platforms:cuda=true` sur chaque `bazel.sh run` (sinon repli CPU).
- **Échec à mi-parcours** : chaque gate est indépendant et rejouable ; les artefacts sont nommés par
  gate (`gcN_*`), jamais écrasés. Un gate FAIL n'est **pas** relancé « pour voir » : il est
  diagnostiqué, et le diagnostic est écrit.

## 9. Livrables

- `zml_runner/gencfg.zig` (**nouveau**) — parsing, découverte, validations §4.1, `isSuppressed` /
  `isEos`, selftest.
- `zml_runner/gemma4_g12auto.zig` — chargement, politique, flags `--gen-config` / `--no-gen-config`,
  log `GENCFG:`, compteur, arrêt multi-EOS, log de fin nommant l'EOS.
- **`zml_runner/BUILD.bazel` — `gencfg.zig` à ajouter aux `srcs` de TROIS cibles**
  (`gemma4_g12auto`, `gemma4_g12a4k`, `gemma4_g12a8k`) : les wrappers ont leur propre cible.
- **Garde d'exclusivité `--repl`** (`:868-874`) à étendre au nouveau flag si exclusif.
- `scripts/69_u8_gen_oracle.py` — politique aux **deux** sites, `--no-gen-policy`, `top5_raw` +
  `top5_policy`, `gen_policy` + `script_md5` au manifest, assertion anti-`-inf`.
- **Fixture GC1** + le script qui la produit (côté oracle, via le vrai processor).
- `docs/GENERATION_CONFIG_RESULTS.md` — verdicts, chiffres, **« Périmètre de la claim »** (modèle :
  `MASKS_INGRAPH_RESULTS.md:70-73`), **« Dette connue »** (E2B, §3), **impact sur D11** (F8).
- **Passe de nuance sur la claim** — sites de catégorie (i), recensement par grep figé, gate GC11.
- `PLANNING.md` (4 cases lignes 40-43 + dette E2B), `README.md`,
  `docs/FINDING_GENERATION_CONFIG.md` (statut → corrigé).
- Tags `gate/gc-0-pass` … `gate/gc-11-pass`, PR vers `main`.
- **Anonymisation** : grep §5.4 durci **avant** tout push (chemins `/Users/<user>`, IP LAN, alias de
  machines, noms d'hôtes), **y compris** dans les logs rapatriés et les manifests.

---

## 10. Historique de révision

| # | Correction | Origine |
|---|---|---|
| 0 | Version initiale | Cartographie 5 sondes + décisions Régis 1-3 |
| 1 | **§2bis ajoutée** : claims falsifiables, conviction, prédictions chiffrées, ce qui tue chaque claim, ce qui réfuterait le chantier ; **GC8** (test décisif de C1) et **GC9** (instruire @47) créés | Exigence Régis « scientifiquement falsifiable » (28 juil) |
| 2 | **Seuil de kill corrigé d'un facteur 2** : 9,365e-4 borne l'erreur d'**un logit**, un flip d'argmax exige que **deux** logits se croisent ⇒ **1,873e-3** sur la marge. @47 est 2,4× au-dessus (et non 4,9×) — conclusion inchangée | Revue, lentille ancrage |
| 3 | **6 ancrages `transformers` réancrés sur 5.14.1** (3 pointaient du code sans rapport en 5.14.1 : stop-strings au lieu d'`EosTokenCriteria`) + convention de citation en tête | Revue, 5 lentilles convergentes |
| 4 | **F6 — `--oracle` n'est PAS du teacher-forcing** (`fed = tok` inconditionnel `:1438`) : §4.4 refondue en 3 modes ; la justification « les gates survivent car teacher-forcé » était **fausse**, remplacée par F4 | Revue, lentilles zig/gates/python |
| 5 | **`--window-vacuity` sorti du périmètre** : ce chemin ne sélectionne aucun token et `return` avant `generateOnce` | Revue, lentilles zig/gates |
| 6 | **F7 — `gen_auto_long` inspectée** (999 ids, 0 occurrence) : C5 passe de *probable* à *établie* ; la rév. 0 déclarait à tort un doute non levable | Revue, lentille ancrage |
| 7 | **F8 — la règle « erreur dure » cassait le contre-test D11** (checkpoint corrompu à plat) ⇒ R3b + traitement explicite | Revue, lentille trous |
| 8 | **F9 — API `std.Io.Dir`** (Zig 0.16-dev), pas `std.fs` : signature `readLink` à confirmer à l'implémentation | Revue, lentille zig |
| 9 | **`--no-gen-policy` ajouté à l'oracle** : sans lui GC5 exigeait d'éditer le script entre deux runs, ce que GC5 interdit | Revue, lentilles python/ancrage |
| 10 | **Compteur `n_suppress_hits` gardé par `in_gen_phase`** (le prefill jette son argmax), dénominateur = tokens générés ; la **sélection** reste inconditionnelle | Revue, lentille zig + vérification |
| 11 | **Critères de gate durcis** : GC3/GC5 prédisent l'**id exact** (`11814`) ; GC2 exige la ligne `GENCFG:` ; GC4(a) est posé sur le prompt de GC3 ; GC4(b) doit **aller au bout**, pas échouer au chargement ; GC5 prédit `n_match = 198/200` | Revue, lentille gates |
| 12 | **GC6 réparé** : le `generation_config` fabriqué se faisait refuser par `EotNotInEosList` — `106` doit rester dans la liste ; contenu minimal spécifié | Revue, lentilles gates/python |
| 13 | **GC1 déplacé sur la 3090** (aucun toolchain Zig sur M1) ; ses contre-tests de non-vacuité énumérés | Revue, lentilles gates/zig/trous |
| 14 | **GC10 et GC11 créés** : R11 (`--repl`) et la passe de nuance n'avaient **aucun** gate | Revue, lentilles gates/trous |
| 15 | **§4.7 composition avec la penalty** : sans elle, la couche top-5 est contournée et le bug revient dès `--repetition-penalty ≠ 1.0` | Revue, lentille trous (bloquant) |
| 16 | **§4.1 validations des valeurs lues** (table) et **§8 gestion d'erreur**, **§7 ordre d'exécution** : trois sections absentes | Revue, lentille trous |
| 17 | **§4.5 publier `top5_raw` ET `top5_policy`** : appliquer la politique avant le `topk` faisait perdre le top-5 brut qui documente le finding | Revue, lentille gates |
| 18 | **Livrables complétés** : `BUILD.bazel` × 3 cibles, garde d'exclusivité `--repl`, fixture GC1, impact D11 ; `TOP_K` en constante unique (pas de 4ᵉ copie) | Revue, lentilles zig/trous |
| 19 | **R10 rectifié** : 24 576 MiB est la VRAM **totale**, pas la libre mesurée ; R4 étendu à l'hôte M4 | Revue, lentille ancrage |
