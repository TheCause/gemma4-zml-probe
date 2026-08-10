# Spec — K5 : prefill partiel (reprendre un cache et feeder un prompt NEUF)

> **Date** : 2026-08-10 · **Niveau de travail** : standard ·
> **Statut** : **rév. 3** (2 passes de revue adversariale, cf. encart plus bas) — rédigée
> sur ancrages code au HEAD `f184c46`, AVANT toute mesure et
> avant la première ligne de code. Les claims §2bis sont pré-enregistrées : ce fichier est
> committé avant le premier run. Le `git log` fait foi.
> **Décision Régis (10 août 2026, verbatim)** : « je ne cherche pas à esquiver K3. Je veux
> gagner une capacité. » K5 est le seul des quatre chantiers du cadrage
> (`2026-08-10-cadrage-dettes-restantes.md`, fiche 3) qui rende possible ce qui ne l'est
> pas : reprendre un cache et **feeder un prompt neuf** — conversation, préfixe partagé,
> cache réutilisable. Il débloque K4 (un 2ᵉ prompt après restore EST un prefill partiel).
> **Exécution prévue** : session dédiée (Opus), plan
> `docs/superpowers/plans/2026-08-10-k5-prefill-partiel.md`. En cas d'écart plan/spec,
> **LA SPEC FAIT FOI**.
>
> **⚖ DÉCISIONS OUVERTES (Task 0 du plan — GO Régis requis, défauts proposés §4.7)** :
> D-K5-1 (fed_next toujours inclus), D-K5-2 (rendu du tour 2 : VÉRITÉ = HF mesuré),
> D-K5-3 (garde fenêtre transposée au prompt neuf), D-K5-4 (gates sur 1280 seule),
> D-K5-5 (clôture du tour 1 quand `fed_next` n'est pas un EOS).
>
> **Rév. 2 (10 août, même session)** : 1ʳᵉ passe de revue adversariale — 14 findings
> (2 bloquants : la comparaison ignorait la politique de décodage — cas réel
> `docs/evidence/kvdump/dc4.err.log:16`, argmax brut 258882 supprimé à gen=0 d'une reprise ;
> PF6 comparait un suffixe incluant la réponse assistant — échec par construction), tous
> traités. + 1 finding hors revue : l'assertion `ids_full[step_next] == fed_next` était un
> **contrôle qui ne peut pas échouer** (les deux membres dérivent du même champ du manifest)
> — retirée AVANT d'être codée, leçon `feedback_controle_qui_ne_peut_pas_reussir` ; la
> limite qu'elle prétendait couvrir est documentée §4.6.
> **Rév. 3 (10 août, même session)** : 2ᵉ passe de revue — 8 findings (2 bloquants :
> §4.4.1 portait encore la méthode PF6 d'avant correction, contradiction interne que
> « la spec fait foi » rendait dangereuse ; PF4(d) était à antécédent vide, le rendu du
> tour 2 émettant toujours ses marqueurs), tous traités. Le `ctx_ids` de `--out-ids` est
> restreint à la reprise **avec prompt** (la reprise simple garde son format historique —
> cohérence C-K5-E).

---

## 1. Contexte et faits établis (tous ancrés, HEAD `f184c46`)

### 1.1 Ce que `--load-cache` sait et ne sait pas faire

La reprise (chantier kvdump, PR #20) réimplante l'état E1-E4 et **continue** : `ids` reçoit
`ids_fed` du manifest (`gemma4_g12auto.zig:2226`), puis `step = step_next` et
`fed = fed_next` (`:3153-3156`). Le saut du prefill est **structurel, pas une branche** :
l'invariant `ids_fed.len == step_next` (vérifié `:2923-2928`) rend
`in_gen_phase = step + 1 >= ids.len` (`:3189`) vrai dès la première itération. Il n'existe
**aucun mécanisme** pour concaténer un prompt neuf — et la garde `--load-cache` + `--prompt`
= `error.LoadCacheWithPrompt` (`:1968-1971`) le refuse explicitement, non-objectif §3 de la
spec kvdump (« un prefill partiel qui mérite son propre chantier »). C'est ce chantier.

### 1.2 Le fait porteur : le graphe ne distingue pas prefill et génération

Il n'y a **ni graphe de prefill séparé, ni branche prefill dans le graphe**. Le prompt est
absorbé token par token par le même graphe decode (« boucle prefill-par-decode »,
`:3098`, `:3161`) ; la seule différence est **host** : en prefill l'argmax est jeté et
`fed = ids[step + 1]` (`:3387-3391`). Toute la mécanique de position se réduit à **un
scalaire runtime** `ctrl.step` (`engine.zig:398-403`) :

| Dérivée de la position | Mécanisme | Ancrage |
|---|---|---|
| position absolue | `pickStep(p.positions, step)` sur table **identité** `0..L_MAX-1` | `engine.zig:405-407`, `gemma4_g12auto.zig:571-580` |
| RoPE sliding (θ=1e4) | recalculé **in-graph** depuis `positions[step]` | `engine.zig:175-178`, `:554`, `:573`, `:788` |
| RoPE full (θ=1e6, partial 0.25) | table host `cos_full/sin_full` remplie **jusqu'à L_MAX** à l'init, sélectionnée par `step` | `gemma4_g12auto.zig:483-486`, `:510-530`, `:571-580` |
| masque full | `j <= p`, régénéré in-graph chaque step | `engine.zig:420-425` |
| masque sliding (window=1024) | `p−1023 <= j <= p`, régénéré in-graph chaque step — la borne basse existe déjà | `engine.zig:426-433`, `gemma4_g12auto.zig:72` |
| écriture cache | scatter linéaire à `pos_u = positions[step]`, sans modulo (pas de ring) | `engine.zig:599-612` |

**Analyse (à PROUVER, pas à supposer — c'est l'objet des gates §5)** : si `step` continue
d'incrémenter au-delà de `step_next` avec des tokens feedés en mode prefill, positions,
masques et RoPE sont corrects par construction. Le chantier est donc **entièrement
host-side** ; ce que la fiche 3 demande de « dériver » est dérivé ci-dessus, et ce que la
spec exige est de le **prouver teacher-forcé** — en libre, une erreur de position se
confond avec un quasi ex æquo (`docs/evidence/penalty/FINDING_temoin_ids_non_reproductible.md`,
marge 0,0046).

### 1.3 Le protocole teacher-forcé du repo — et ce qui lui manque

Règle du repo (`FINDING_NONDETERMINISME_TRAJECTOIRE.md:96`) : tout gate de fidélité
position-par-position est teacher-forcé. L'instrument est un **aller-retour en deux
temps** : le runner écrit ses ids (`--out-ids`, une clé `ids` = générés seuls,
`:1814-1830`, `:3613-3618`), l'oracle `scripts/69_u8_gen_oracle.py --teacher-force` rejoue
`full = render(prompt) ++ gen[:-1]` en **UN prefill HF fp32** (`69:~434` et `:~478` —
retrouvables par `grep -n "use_cache=False" scripts/69_u8_gen_oracle.py`) et compare
argmax + marges par position (marges consignées AVANT tout verdict, piège 17).

Trois manques pour K5, tous côté outillage :
1. le 69 reconstruit le contexte depuis `--prompt` **templaté** — sous reprise, le contexte
   est `ids_fed ++ …`, inexprimable par un prompt ;
2. `--dump-top5` est gated `in_gen_phase` (`:3363`) : les argmax des positions de
   **prefill** ne sortent jamais du process — or ce sont précisément les positions
   teacher-forcées par construction (token feedé imposé, aucun effet boule de neige) ;
3. ⚠ `docs/DOCUMENTATION.md:252` annote `--load-cache --oracle` « teacher-forcé » — c'est
   **inexact** (`fed = tok`, `:3444` ; verdict post-boucle `:3621-3647`). À corriger dans
   ce chantier (une annotation fausse sur LE sujet du chantier est une dette de doc active).

### 1.4 Le template chat : single-turn seulement, et la VÉRITÉ est un rendu mesuré

`renderChatTemplate` (`:111-113`) ne porte que le cas single-turn user→assistant ; le
commentaire `:106-107` l'écrit : les branches multi-tours du jinja 12B **ne sont PAS
portées**. Le pattern du repo pour ce genre de question est établi (`:97-105`) : le rendu
de référence est le rendu HF **réel mesuré** (`apply_chat_template`, sha du
`chat_template.jinja` gardé — `ae53464b…4c6d4`, `69:54`), jamais une supposition sur le
jinja. Piège connu des templates à canal de pensée : le re-rendu multi-tour peut
**réécrire l'historique** (retirer le `thought` des tours passés). Si c'est le cas, la
conformité « conversation HF stricte » est **structurellement incompatible** avec un cache
réutilisé (on ne peut pas retirer du cache ce qui y est déjà) — ce serait un fait à
publier, pas un échec (§4.4).

### 1.5 Gardes existantes concernées

| Garde | Ancrage | Sort dans K5 |
|---|---|---|
| `--load-cache` + `--prompt` ⇒ refus | `:1968-1971` | **levée** — devient le chemin nominal |
| `--ignore-prompt` + `--load-cache` ⇒ refus | `:1983-1986` | **conservée** (la frontière du prompt d'origine reste non reconstructible) |
| `ids.len + limit > L_MAX` ⇒ refus | `:3051-3054` | conservée telle quelle (elle raisonne sur `ids.len` total, correct avec ids allongé) |
| `ids.len >= SLIDING_WINDOW` (hors resume) ⇒ refus | `:3058-3061`, `:2300` | **transposée** au prompt neuf : `n_new >= SLIDING_WINDOW` ⇒ refus (D-K5-3). Garde née sans justification écrite (commit `c2211c0`) — la lever exigerait son propre gate, hors périmètre |
| oracle : `positions[0] == ids.len` | `:2249-2252` | conservée — une fixture post-restore+prompt porte `p0 = ids_full.len` |
| `ids_fed.len == step_next` au restore | `:2923-2928` | conservée — l'invariant du manifest ne change pas |

Le seed penalty (`:3076-3089`) est déjà correct par construction : il vit en tête de
`generateOnce` et couvre `ids` quel qu'il soit — sous K5, `prompt_len = ids_full.len`,
cohérent avec le refus `--ignore-prompt` maintenu.

---

## 2. Objectif et critères de succès

1. `--load-cache <f>` + `--prompt "<texte>"` : le runner reprend l'état du dump, **absorbe
   le prompt neuf en prefill-par-decode** aux positions `step_next..`, puis génère —
   libre (`--max-tokens`) ou borné (`--oracle`).
2. La **capacité est chaînable** : le run de reprise peut lui-même `--dump-cache` — c'est
   la brique conversation (restore → tour → dump → restore → …).
3. **Le graphe ne bouge pas** : md5 HLO identique au témoin (PF0).
4. **L'équivalence à HF est prouvée teacher-forcée** aux positions du prompt neuf et à la
   frontière (PF1), **le mordant est prouvé** (PF2 : un état décalé d'une position fait
   échouer le gate), et **la fenêtre sliding est exercée à travers la frontière** (PF3).
5. **Aucun chemin silencieusement faux** : refus bruyants nouveaux/maintenus, chacun VU
   échouer (PF4) ; la reprise simple (sans `--prompt`) est inchangée (PF5).
6. Le **rendu du tour 2** est conforme au rendu HF mesuré, en ids (PF6).

## 2bis. Claims falsifiables — prédictions PRÉ-ENREGISTRÉES

> Toute valeur mesurée qui contredit une prédiction est publiée telle quelle ; une
> requalification exige une décision Régis écrite.

### C-K5-A — « Le graphe est déjà correct au-delà de `step_next` »
- **Conviction : probable** (dérivation §1.2 — mais une dérivation n'est pas une mesure ;
  c'est exactement le genre de claim que le repo a déjà vu tomber).
- **Prédiction** : aux positions du prompt neuf (prefill partiel, teacher-forcé par
  construction) et à la première position de génération, l'argmax ZML == l'argmax HF fp32
  au même préfixe, **n_ctx/n_ctx et 1/1**, marges publiées. Sur les m tokens de génération
  libre qui suivent : première divergence éventuelle à marge ≤ **1,873e-3** (le seuil de
  tie hérité, spec generation-config §2bis), publiée comme tie.
- **Ce qui tue C-K5-A** : un seul mismatch à marge > 1,873e-3 sur une position
  teacher-forcée — signature d'une erreur de position/masque/RoPE, pas d'un tie.
  Porté par **PF1** (et re-porté par **PF3** sous fenêtre mordante).

### C-K5-B — « Le gate mord : un état qui ment d'UNE position échoue » (non-vacuité)
- **Conviction : certain** — sinon tout le dispositif est vide.
- **Prédiction** : un dump forgé **`shift-fwd`** — `step_next+1`, `ids_fed` étendu d'un id
  fantôme, `fed_next` inchangé, caches INTACTS : le manifest déclare un token que le cache
  ne porte pas (la position `step_next` du cache est restée aux zéros du dump) — rejoué
  dans le protocole PF1 produit des mismatches à marges grasses **dès les premières
  positions du tour 2** (l'attention lit un slot vide, mécanisme DC4 localisé à UNE
  position).
- **Pourquoi PAS un `shift-back`** (tronquer d'une position), analysé en revue : l'état
  tronqué est **auto-cohérent** — le run forgé re-feede le token retiré à sa vraie
  position et réécrit le slot à l'identique ; le modèle voit une séquence valide amputée
  d'un token, pas une erreur de position. Un mordant nul y serait plausible ET sain — ce
  serait condamner l'antécédent, pas prouver le mordant (leçon RP4(c)).
- **Ce qui tue C-K5-B** : le mutant `shift-fwd` passe PF1 ⇒ le gate ne prouve rien —
  STOP diagnostic. Les marges de référence du scénario sont publiées (discriminance :
  médiane ≥ 10× le seuil de tie). Porté par **PF2**.

### C-K5-C — « La fenêtre sliding est correcte quand elle ne commence plus à 0 »
- **Conviction : probable** (la borne basse `p−1023` existe in-graph depuis les masques
  in-graph — mais aucun gate ne l'a exercée APRÈS un restore).
- **Prédiction** : sur un scénario dimensionné pour que des positions du tour 2 et de la
  génération excèdent 1024 (fenêtre mordante : `step_next ≈ 1005`, `n_new ≈ 40`,
  `m = 32` ⇒ positions max ≈ 1078), PF1 tient **ET** le témoin fenêtre de l'oracle
  (`69:452-465` : assert `sliding_mask != causal_mask`) atteste que la fenêtre a mordu.
- **Ce qui tue C-K5-C** : équivalence cassée sur CE scénario alors que PF1 court passe
  (⇒ borne basse fausse après restore) ; ou témoin fenêtre inerte (gate vacueux,
  INEXÉCUTABLE, redimensionner). Porté par **PF3**.

### C-K5-D — « Le graphe n'a pas bougé »
- **Conviction : certain** (le chantier est host-only par construction).
- **Prédiction** : md5 HLO `before_optimizations` == `297679847aa04b719942d75d093adf2b`
  (le témoin stable sur six chantiers) ; `git diff engine.zig` vide.
- **Ce qui tue C-K5-D** : un octet. Porté par **PF0**.

### C-K5-E — « La reprise simple est inchangée »
- **Conviction : certain.**
- **Prédiction** : `--selftest-kvdump-eq` re-passe **32/32 bit-identiques** (1280 et 4k) ;
  un `--load-cache` sans `--prompt` suit le chemin actuel à l'identique (mêmes logs
  `KVLOAD:`, entrée directe en génération).
- **Ce qui tue C-K5-E** : 1 bit d'écart au selftest ; un log de reprise qui change de
  forme. Porté par **PF5**.

### C-K5-F — « Le rendu du tour 2 est celui de HF, en ids »
- **Conviction : incertain** (dépend du jinja multi-tour 12B, jamais mesuré ici).
- **Prédiction** : le suffixe rendu par le runner pour le tour 2 == le suffixe
  `apply_chat_template(conversation 2 tours) − apply_chat_template(tour 1 + réponse)`
  mesuré sur HF, **ids identiques** sur le cas canonique.
- **Ce qui tue C-K5-F** : un id d'écart. **Cas de requalification PRÉ-DÉCLARÉ** : si le
  jinja multi-tour **réécrit l'historique** (thought des tours passés retiré), la
  conformité stricte est structurellement impossible avec cache réutilisé — le fait est
  publié avec le diff de rendu, et la sémantique v1 « fidèle au généré » est soumise à
  GO Régis (ce n'est pas une requalification à chaud : le cas est écrit ICI, avant
  mesure). Porté par **PF6**.

### Grandeurs prédites AVANT mesure (arithmétique, pas devinée)

| Grandeur | Valeur prédite | Fondement |
|---|---|---|
| coût du prefill partiel de n_new tokens | ≈ n_new / 9,6 s (1280) | prefill par-decode, 9,6-9,7 tok/s (`MODE_BUILD_AUDIT.md`) |
| scénario PF1 court (step_next≈52, n_new≈20, m=32) | run B ≈ 6 s de steps | même base |
| scénario PF3 (prefill A ≈ 1005 pos) | run A ≈ 105 s | même base |
| gain au scénario PF3 — DEUX métriques, publiées ENSEMBLE | ≈ **×26** en positions évitées ((1005+40)/40) ET ≈ **×14** en temps de steps (1078/~73) | la capacité elle-même — mesure M-K5-1, sans verdict ; publier l'une SANS l'autre serait un chiffre juste et trompeur |
| oracle PF3 (M4, prefill HF fp32 CPU, T≈1078) | ≈ 15× l'oracle du scénario court (T≈90) — **nohup obligatoire**, temps mesuré publié | ratio des longueurs de prefill, pas de chiffre absolu deviné |
| `ALLOC-LOOP` des runs K5 | identique au run nu | tout le travail neuf vit hors boucle |

---

## 3. Non-objectifs (et où vit la dette)

- **K4 (résident à reprise / multi-tour REPL)** : `--load-cache` + `--repl` reste refusé
  (`:1960-1963`). K5 livre la brique one-shot chaînable (restore → tour → dump) ; le
  résident qui enchaîne sans quitter le process est le chantier suivant, débloqué par
  celui-ci.
- **E2B** : même politique de périmètre que kvdump/penalty — 12B seul, l'écart est
  documenté au README (K3 le rendra exact ; il devient « un peu plus faux » à chaque
  capacité 12B, dette assumée).
- **Sampling armé + dump** : politique kvdump inchangée (PRNG non sérialisé, refus).
- **8k** : même code comptime, aucun gate (dette DA-4 inchangée).
- **Multi-tour > 2 tours dans un même run** : la v1 absorbe UN prompt neuf par run ;
  N tours = N runs chaînés par dump/restore. Le coût du chaînage (relecture des ~840 Mio
  par tour à 1280) est le prix v1, mesuré par M-K5-1 — c'est K4 qui l'amortira.
- **Compression du dump, gestion de contexte plein** (résumé/éviction) : hors périmètre.

---

## 4. Design

### 4.1 Sémantique de la reprise avec prompt neuf

La séquence complète devient :

```
ids_full = ids_fed ++ [fed_next] ++ ids_t2
           └─ manifest ─┘  └─ D-K5-1 ─┘  └─ rendu tour 2 (§4.4), SANS BOS ─┘
```

- **`fed_next` est TOUJOURS inclus** (D-K5-1) : c'est le dernier token généré, jamais
  feedé — il fait partie du texte produit (y compris quand c'est l'EOS `<turn|>` id 106 :
  une conversation HF le contient). L'exclure fabriquerait un historique que personne n'a
  vu.
- **Clôture du tour 1** (D-K5-5) : le rendu HF d'une conversation FERME le tour assistant
  avant le tour suivant. Si `policy.isEos(fed_next)` est **faux** (arrêt `max_tokens` —
  le cas nominal d'un run A borné), l'`eot_id` mesuré (`:2064`) est **injecté** entre
  `fed_next` et `ids_t2`, et logué (`K5: tour 1 clos par eot injecté`). Si `fed_next` est
  déjà un EOS, rien n'est injecté. `ids_t2` est le suffixe canonique post-clôture (§4.4).
- Initialisation : `step = step_next`, `fed = fed_next` — inchangée. ⚠ Une assertion
  `ids_full[step_next] == fed_next` a été envisagée puis RETIRÉE en revue : les deux
  membres dérivent du même champ du manifest, elle ne peut pas échouer
  (`feedback_controle_qui_ne_peut_pas_reussir`). La limite réelle est documentée §4.6.
- La boucle existante fait le reste **sans modification** : `in_gen_phase` est faux tant
  que `step + 1 < ids_full.len` (il y a du prompt à absorber), `fed = ids[step + 1]`
  enchaîne les tokens du tour 2, et le dernier step de prefill produit s0. Les positions,
  masques et RoPE suivent `step` (§1.2).
- Le seed penalty reçoit `ids_full` (déjà structurel, `:3076-3089`) ; les compteurs de
  mordant repartent à zéro comme pour tout prompt.
- `--dump-cache` reste disponible sur le run de reprise (chaînage §2.2) — l'invariant du
  dump `ids_fed.len == step_next` reste vrai par construction (`ids_full ++ generated`).

### 4.2 Ce qui change dans le code (host uniquement)

1. **`run()`** : la garde `LoadCacheWithPrompt` (`:1968-1971`) tombe ; sous
   `--load-cache` + `--prompt`, `ids = ids_fed ++ [fed_next] ++ promptToIdsTurn2(...)`.
2. **`renderChatTemplateTurn2` + `promptToIdsTurn2`** : rendu du tour 2 (§4.4), sans BOS
   (l'encoder n'ajoute aucun token spécial, `:94`, `:120-131` — le BOS est un préfixe
   explicite du tour 1 uniquement).
3. **Gardes** (§1.5) : `n_new >= SLIDING_WINDOW` ⇒ `error.PromptTooLong` ;
   place : garde existante `:3051` inchangée (raisonne sur `ids.len` total).
4. **`--dump-top5` en prefill de reprise** : log nouveau
   `top5 @ ctx=<step> : idx=… val=…` émis quand `resume_state != null` et
   `!in_gen_phase` et `dump_top5` — c'est la sortie que PF1 compare côté ZML.
   Format imposé, greppé par le dépouilleur.
5. **`--out-ids` étendu sous reprise AVEC prompt** (et elle seule — la reprise simple
   garde le format historique à une clé, cohérence C-K5-E) : le fichier porte une
   **2ᵉ clé `ctx_ids`** (I32, `ids_full` complet) à côté de `ids` (générés seuls,
   format inchangé pour les consommateurs existants).
6. **`scripts/69_u8_gen_oracle.py`** : mode nouveau `--context-ids <out_ids.safetensors>`
   (exclusif de `--prompt` — la garde `required=True` de `--prompt`, `69:574-575`, apprend
   ce mode) : `full = ctx_ids ++ gen[:-1]`, un prefill, et le rapport JSON publie les
   positions `step_next..T-1` (borne via `--ctx-from <n>` = `step_next`) avec, pour
   CHAQUE position, **les deux canaux** : `argmax_raw`/`margin_raw` (logits bruts — ce que
   le top-5 in-graph du runner voit) ET `argmax_policy`/`margin_policy` (post-politique
   suppress — ce que `generated` reflète). Fait mesuré qui l'impose : à gen=0 d'une
   reprise réelle, l'argmax brut est `258882` (supprimé), le choisi `4509`
   (`docs/evidence/kvdump/dc4.err.log:16`) — un canal unique produirait un FAIL gras
   spuré. Le témoin fenêtre (`69:452-465`) et le self-check tête (`69:483-496`) restent
   actifs.
7. **`scripts/80_pf1_bridge.py`** (nouveau, pattern 75 ; ⚠ 77 est PRIS —
   `77_ngram_repetition.py`) : dépouilleur à verdict machine — lit le JSON de l'oracle +
   le log du runner (lignes `top5 @ ctx=` et `generated`), compare **brut↔brut** en
   régime (i)-ctx et **policy↔policy** en régime gen, applique les critères §5, exit 0
   ssi PASS. Seuils en constantes de tête.
8. **`scripts/74_kvdump_inspect.py`** : sous-commande nouvelle `shift-fwd` (forge PF2 :
   `step_next+1`, `ids_fed` étendu d'un id fantôme, checksum `ids_fed_xxh64` recalculé,
   caches intacts — CLI positionnelle comme les sous-commandes existantes).
9. **`docs/DOCUMENTATION.md:252`** : l'annotation « teacher-forcé » de
   `--load-cache --oracle` est corrigée (§1.3.3).

**Interdits inchangés** : aucun octet dans `engine.zig`, aucune allocation dans la fenêtre
ALLOC-LOOP (le rendu/concat du tour 2 vit AVANT `generateOnce`, comme la tokenisation
actuelle), D10 intact.

### 4.3 Positions, masques, RoPE — la dérivation que les gates transforment en fait

Écrite une fois ici, opposable :

- **Positions** : la position absolue du i-ème token du tour 2 est
  `step_next + 1 + i` (fed_next occupe `step_next`). Aucune table à étendre :
  `positions` est l'identité jusqu'à `L_MAX−1` (`:571-580`) et la garde de place `:3051`
  interdit de la dépasser.
- **Masque sliding** : à la position `p`, la fenêtre est `[p−1023, p]` — le cache
  linéaire porte les clés des positions absolues, la borne basse in-graph
  (`engine.zig:430-433`) coupe les positions antérieures **y compris celles du contexte
  repris**. C'est le comportement attendu du modèle (HF fait pareil sur la séquence
  complète) — PF3 l'exerce à travers la frontière.
- **RoPE** : sliding recalculé in-graph pour tout `p` ; full lu dans une table déjà
  remplie jusqu'à `L_MAX−1`. Rien à recalculer, rien à invalider au restore — le manifest
  n'a pas besoin de porter un état RoPE, `step_next` est l'ancre unique et suffisante
  (position ≡ step).

### 4.4 Rendu du tour 2 — protocole (D-K5-2)

1. **Mesurer d'abord** (Task dédiée du plan, avant tout littéral Zig) : sur M4 (venv
   g12b), `apply_chat_template` sur TROIS cas — (a) `[user1]` + generation prompt (le
   témoin single-turn connu) ; **(c) `[user1, assistant(réponse)]` SANS generation
   prompt** (le tour assistant FERMÉ — c'est lui qui correspond à
   `ids_fed ++ [fed_next] ++ clôture`) ; (b) `[user1, assistant(réponse), user2]` +
   generation prompt. **Le suffixe de référence = `ids_b − préfixe_commun(ids_b,
   ids_c)`** — comparer à (a) inclurait la réponse assistant re-tokenisée (finding
   bloquant de 1ʳᵉ revue : échec par construction, le canal thought du generation prompt
   de (a) casse le préfixe commun). La réécriture d'historique se juge sur
   `préfixe_commun(ids_b, ids_c) < len(ids_c)`. Sha du `chat_template.jinja` asserté
   (garde existante du 69). ⚠ Tout id de clôture excédentaire en queue de `ids_c`
   (au-delà du contenu + EOS) appartient à l'**injection** D-K5-5, jamais retranché du
   littéral du tour 2 — un recoupement scripté de la queue de `ids_c` contre
   `[fed_next] ++ injection` est exigé (les deux côtés de PF1 partagent `ctx_ids`, une
   erreur ici serait invisible de PF1 ET de PF6).
2. **Si le suffixe est une concaténation propre** (l'historique n'est pas réécrit) : le
   littéral Zig `renderChatTemplateTurn2` reproduit ce suffixe, PF6 le prouve en ids.
3. **Si le jinja réécrit l'historique** (cas pré-déclaré C-K5-F) : publier le diff,
   soumettre la sémantique « fidèle au généré » à GO Régis, PF6 porte alors sur le
   suffixe seul à partir de la frontière commune.

⚠ Le rendu attendu contient vraisemblablement le canal de pensée vide
(`<|channel>thought\n<channel|>`) en fin d'amorce comme au tour 1 (`:100-103`) — mais
c'est la mesure qui le dira, pas ce paragraphe.

### 4.5 Protocole PF1 (le cœur) — teacher-forcé à travers la frontière

```
Run A (VM)   : --prompt P1, gen k, --dump-cache dcA         → dcA.kvdump (+ témoin md5 binaire)
Run B (VM)   : --load-cache dcA --prompt P2 --max-tokens m
               --dump-top5 --out-ids outB                    → log (top5 @ ctx=…), outB{ids, ctx_ids}
Oracle (M4)  : 69 --context-ids outB --ctx-from <step_next> --compute-fp32 → pf1.json
Verdict (M1) : 80_pf1_bridge.py pf1.json runB.err.log       → PASS/FAIL machine
```

- Les positions comparées se partagent en deux régimes, jugés séparément :
  **(i) positions teacher-forcées** — celles du tour 2 (`step_next .. ids_full.len−1`,
  le token feedé est imposé) **plus la première position de génération** : exigence
  argmax ZML == argmax HF **sans exception**, marges publiées ;
  **(ii) positions de génération libre** (les m−1 suivantes) : régime DC3 — première
  divergence éventuelle à marge ≤ 1,873e-3, publiée comme tie.
- **Chaque régime compare le BON canal** (finding bloquant de revue) : les lignes
  `top5 @ ctx=` du runner sont des logits **BRUTS** in-graph (la politique s'applique
  APRÈS, host-side, `:3215-3220`) → comparées à `argmax_raw` de l'oracle ; `generated`
  est **post-politique** (`policy.select`) → comparé à `argmax_policy`. Mélanger les
  canaux ferait échouer à tort tout scénario où la suppression mord — cas déjà observé
  à gen=0 d'une reprise (`dc4.err.log:16`).
- ⚡ Règle d'instrumentation du 10 août appliquée : **le md5 du BINAIRE est capturé avec
  tout témoin** (`md5sum` du binaire dans le log d'evidence) — un témoin d'ids sans
  binaire identifié n'est pas une référence.

### 4.6 Validations — refus bruyants, chacun TESTÉ (PF4)

| Cas | Erreur | Vu échouer par |
|---|---|---|
| `n_new >= SLIDING_WINDOW` (tour 2 trop long) | `error.PromptTooLong` | PF4(a) |
| `ids_full.len + limit > L_MAX` | `error.SequenceTooLong` | PF4(b) |
| `--ignore-prompt` + `--load-cache` | `error.IgnorePromptWithLoadCache` (inchangé) | PF4(c) |
| **prompt vide** sous reprise (`--prompt ""`) — gardé sur `prompt_text.len == 0` **AVANT le rendu** (2ᵉ revue : le rendu émet toujours ses marqueurs de tour, `n_new` ne peut jamais valoir 0 — une garde post-rendu serait à antécédent vide, `feedback_test_vacuite_antecedent`) | `error.PromptTooLong` | PF4(d) |
| oracle : `positions[0] != ids_full.len` | `error.OraclePromptMismatch` (inchangé) | PF4(e) — fixture à p0 faux |
| `--load-cache` + `--prompt` | **NE refuse PLUS** | le chemin nominal PF1 en atteste |

**⚠ Limite documentée, PAS un refus** : un `fed_next` **forgé** dans le manifest est
indétectable host-side — les checksums couvrent `ids_fed` et les 4 caches, pas la
cohérence `fed_next`↔cache (le cache est opaque au host). Un tel forge fabrique un
contexte différent, que SEUL l'aller-retour teacher-forcé détecte — c'est exactement ce
que PF2 démontre (le `shift-fwd` est un forge de manifeste). À écrire dans
`K5_RESULTS.md` §limites, pas à « protéger » par une assertion qui ne peut pas échouer.

### 4.7 Décisions ouvertes et défauts proposés (Task 0 du plan)

| # | Question | Défaut proposé |
|---|---|---|
| D-K5-1 | `fed_next` dans `ids_full` ? | **Toujours inclus** (§4.1 — l'exclure fabrique un historique jamais vu ; l'EOS d'un tour fait partie d'une conversation) |
| D-K5-2 | Rendu tour 2 | **VÉRITÉ = rendu HF mesuré** (§4.4) ; si historique réécrit par le jinja : sémantique « fidèle au généré », écart publié, GO Régis |
| D-K5-3 | Garde fenêtre | **Transposée** : `n_new >= 1024` ⇒ refus ; la LEVER exigerait son propre gate (garde historique sans justification écrite, `c2211c0`) — dette |
| D-K5-4 | Variantes gatées | **1280 seule** pour PF1-PF4/PF6 (PF3 y est réalisable : 1005+40+32 < 1280) ; PF5 sur 1280 **et** 4k (selftest bon marché). Gate teacher-forcé 4k = dette écrite |
| D-K5-5 | Tour 1 non clos (arrêt `max_tokens`, `fed_next` ≠ EOS — le cas NOMINAL d'un run A borné) | **Clôture injectée** : `eot_id` appendé entre `fed_next` et le tour 2, logué (§4.1) — le rendu reste conforme au multi-tour HF, qui ferme toujours le tour assistant. L'alternative (exiger un run A arrêté EOS) rendrait les scénarios de gates fragiles |

---

## 5. Gates — chacun avec ce qui le ferait échouer

> Conventions : celles des plans D10/penalty (build `build_3090.sh` 2 flags, bannière
> `BUILD: mode=ReleaseFast` greppée sinon INEXÉCUTABLE, capture `> out.log 2> err.log`,
> FAIL ⇒ STOP sans requalification à chaud, preuves dans `docs/evidence/k5/`,
> md5 du binaire consigné avec chaque témoin). Tags : `gate/pf<N>-pass`.

| Gate | Prouve | Critère PASS | Ce qui le fait FAIL |
|---|---|---|---|
| **PF0** graphe intact | C-K5-D | md5 HLO == `297679847aa04b719942d75d093adf2b` ; `git diff engine.zig` vide | 1 octet |
| **PF1** équivalence à la frontière | C-K5-A | protocole §4.5, scénario court (step_next≈52, n_new≈20, m=32) : régime (i) **n_ctx+1 / n_ctx+1** argmax match (brut↔brut sur ctx, policy↔policy sur gen[0]) ; régime (ii) : zéro divergence ou 1ʳᵉ divergence à marge ≤ 1,873e-3 publiée ; marges consignées AVANT verdict ; verdict par script 80, jamais à l'œil | 1 mismatch en régime (i) ; divergence grasse en (ii) |
| **PF2** le mordant | C-K5-B | dump forgé `shift-fwd` (74) rejoué dans PF1 : **FAIL attendu**, mismatches en régime (i) à marges publiées ; le nominal re-passe sur le dump intact (même binaire, md5 consigné) | le mutant PASSE (STOP diagnostic — antécédent non discriminant ?) |
| **PF3** fenêtre à travers la frontière | C-K5-C | scénario long (step_next≈1005, n_new≈40, m=32) : mêmes critères que PF1 **ET** témoin fenêtre oracle actif (`bites_in_prefill` vrai) | équivalence cassée ; ou témoin inerte ⇒ INEXÉCUTABLE (redimensionner, pas PASS) |
| **PF4** refus bruyants | §4.6 | les 5 cas (a)-(e) chacun VU échouer avec l'erreur nommée ; aucun crash non qualifié | un refus qui ne se déclenche pas |
| **PF5** non-régression | C-K5-E | `--selftest-kvdump-eq` **32/32 bit-identiques** sur 1280 ET 4k ; un `--load-cache` sans `--prompt` : `KVLOAD:` et comportement inchangés | 1 bit ; un log qui change |
| **PF6** rendu tour 2 | C-K5-F | ids du suffixe Zig == ids du suffixe HF mesuré (§4.4), comparaison littérale scriptée | 1 id (hors cas pré-déclaré C-K5-F, qui suit son protocole) |

**Mesure publiée sans verdict — M-K5-1** : coût du run B (prefill partiel n_new + gen m)
vs re-prefill complet équivalent (`ids_full` re-prefillé from scratch), au scénario PF3.
C'est le chiffre de la capacité ; prédiction ×26 en positions évitées (§2bis), le temps
mesuré fait foi. Pas de tag pour une mesure.

**Non-vacuité transversale** : PF1/PF3 ne sont déclarés PASS que si PF2 a produit son
FAIL — un dispositif d'équivalence dont on n'a pas vu le contraire mordre n'a rien prouvé
(`feedback_invariant_tue_le_controle`).

---

## 6. Livrables

1. `zml_runner/gemma4_g12auto.zig` — chemin `--load-cache` + `--prompt` (§4.2.1-4),
   rendu tour 2 + clôture D-K5-5, gardes, log `top5 @ ctx=`, `ctx_ids` dans `--out-ids`,
   usage string.
2. `scripts/69_u8_gen_oracle.py` — mode `--context-ids`/`--ctx-from`, canaux raw+policy (§4.2.6).
3. `scripts/80_pf1_bridge.py` — dépouilleur à verdict machine (§4.2.7).
4. `scripts/74_kvdump_inspect.py` — sous-commande `shift-fwd` (§4.2.8).
5. `docs/K5_RESULTS.md` — verdicts des 7 gates, claims jugées, M-K5-1, dettes.
6. `docs/DOCUMENTATION.md` — correction de l'annotation `:252` + doc de la capacité ;
   README (la capacité conversation chaînée, l'écart E2B rappelé) ; PLANNING ; fiche
   mémoire ; PR.
