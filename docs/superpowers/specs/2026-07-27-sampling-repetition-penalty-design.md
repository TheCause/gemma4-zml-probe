# Spec — Repetition penalty (phase 1, déterministe) puis sampling (phase 2, stochastique)

> **Date** : 2026-07-27 · **Niveau** : standard (spec → plan → gates → PR) ·
> **Base** : PR #17 mergée (`2d4cda6`), HEAD `86d31d2`. Demande Régis : « continuons sur le
> backlog : sampling + repetition penalty ». Symptôme d'origine : en greedy, le 12B **boucle
> sur la récitation** — comportement de modèle, pas un bug du portage.
>
> **Révision 3** (même jour, 2 tours de double revue). Deux corrections de fond :
> (1) la v1 croyait les logits disponibles sur les deux runners — **faux pour l'E2B** ;
> (2) la v2 a déplacé le périmètre sur le 12B **sans déplacer l'oracle**, qui est resté celui
> de l'E2B — et l'oracle de décode du 12B est **bf16 par construction**, c'est-à-dire
> l'instrument que le projet a déclaré corrompu le 25 juillet. D'où le prérequis **RP-1**.
> Détail des 2 tours en §12.

## 0. Décisions de cadrage (arbitrées avec Régis)

| # | Décision | Motif |
|---|---|---|
| **C1** | **Deux phases**, penalty d'abord | La penalty sur greedy reste **déterministe** → HF sait faire `greedy + repetition_penalty` → **l'oracle « ids == HF » reste opérable**. Le stochastique casse l'oracle argmax et exige des gates distributionnels : on ne mélange pas les deux natures de preuve. |
| **C2** | **Approche host-side pure** | Seule voie qui laisse le **graphe intact** : « je n'ai rien cassé » se démontre par un diff, pas par une campagne. |
| **C3** | **`gemma4_g12auto.zig` SEUL est câblé** (rév. 2) | Seul runner dont le graphe sort déjà les logits (F1). Câbler l'E2B exigerait d'ajouter une sortie à son tuple racine, donc de modifier son HLO. La vitesse d'itération vient **de RP1** (`zig test` sans GPU), pas du petit modèle. |
| **C4** | **Directives `:param` à chaud** dans `--repl` | Gratuites en host-side. Recharger le 12B coûte ~1 min 40 : un balayage passe de N compiles à **une**. |
| **C5** | **L'oracle appelle le vrai `RepetitionPenaltyLogitsProcessor`** | Retranscrire la formule ferait comparer deux transcriptions **de la même lecture par le même auteur** : une inversion de branche commise des deux côtés passerait le gate. |
| **C6** | **L'instrument est refondé AVANT d'armer la penalty** (rév. 3) | L'oracle de décode 12B est bf16 (F6) : il fabrique des ties artificiels, et la penalty **comprime les marges de 13 %**, donc en fabriquerait davantage. Armer la penalty sur cet oracle produirait des FAIL dont la cause n'est pas le code livré. → gate **RP-1**. |

**Hors périmètre décidé** : câbler `gemma4_gen_auto.zig` (E2B). Dette **écrite** :
« exposer les logits de `StepTok` → +1 sortie au tuple racine → arbitrage HLO à refaire ».

## 1. Faits établis par lecture du code

**F1 — les logits sortent déjà du graphe du 12B, et PAS de celui de l'E2B.**
`gemma4_g12auto.zig:830` : **7 sorties**, logits en 3ᵉ (ajoutée pour `--window-vacuity`) ;
réception `:1357`, commentaire « NON lue ici (pas de D2H) » `:1356`, `deinit()` `:1401`.
`gemma4_gen_auto.zig:753` : **6 sorties, sans logits** (host : `get` à 6 buffers, `:1041`).
C'est ce qui fonde C3.

**F2 — les logits sont post-softcap, sur les quatre chemins du moteur.**
`softcapPrec` (`engine.zig:193`) appliqué en `:766` (`forward`), **`:807` (`forwardStageGen`)**,
`:850` (`forwardStep`), `:889` (`forwardStageStep`). **Le chemin du 12B est `forwardStageGen`**
(`gemma4_g12auto.zig:822`) → la ligne pertinente est **`engine.zig:807`** ; `:850` appartient à
`forwardStep`, que le commentaire D12 (`:794-801`) rejette explicitement pour le 12B.
L'objet rapatrié est donc exactement celui que HF passe à ses `LogitsProcessor`.

**F3 — formule HF** (`transformers/generation/logits_process.py`) :

```python
penalty_scores = torch.where(last_scores < 0, last_scores * penalty, last_scores / penalty)
scores = torch.where(token_mask, penalty_scores, last_scores)
```

logit **négatif → ×penalty**, **positif ou nul → ÷penalty** · `token_mask` **booléen** ⇒ **au
plus une fois par token distinct** · tokens = **`input_ids` entier, prompt inclus** par défaut,
`prompt_ignore_length` étant un **entier** · `__init__` rejette via `not (penalty > 0)` —
**formulation en acceptation**, qui exclut aussi `NaN` (voir §8).

**F4 — le RNG device est écarté** (phase 2). `ZML_UPSTREAM_AUDIT_2026-07-12.md:36-41` :
`sampleTokens`/`Tensor.Rng` existent déjà dans le ZML vendored (`adee932e`, aucun bump), mais
« le RNG device n'est pas garanti déterministe entre backends/versions ». Piège consigné :
`sampleTokensDynamic` **multiplie** par la température au lieu de diviser. Ce piège dicte SM1.

**F5 — la byte-identité HLO stricte n'existe pas dans ce repo.**
`ENGINE_LOG.md:92` : 1037/1037 fichiers identiques **sauf 2 diffs bénins** — (a)
`debug_options`, qui n'encode que le chemin `--xla_dump_to` et diffère **par construction** ;
(b) un `.ir-with-opt.ll` à **noms SSA LLVM** alpha-équivalents.
⚠ **Contradiction interne au repo** : `ZML_MODULAR_ENGINE_DESIGN.md:132` n'en compte qu'**1**
(`debug_options` seul). RP0 retient la version conservatrice (2) et **tranche la contradiction
en publiant ce qu'il observe**.

**F6 — l'oracle de décode du 12B est bf16 par construction** (rév. 3, fondement de C6/RP-1).
`scripts/69_u8_gen_oracle.py:371-372` refuse `--compute-fp32` hors teacher-force (« le mode
décode U8 reste bf16 tel que consigné ») et `:148` **asserte** `out.logits.dtype == bfloat16`.
`U_12B_RESULTS.md:22-25` en documente l'effet : « quantum bf16 = 0,125 sur des logits ~25 →
l'oracle fabrique des ties artificiels sur ~1 % des steps (48 quasi-ties mesurés sur 1150) »,
d'où « U8 brut **42/48** ». Le **48/48 STRICT** a été obtenu par un **autre instrument**
(`68 --compute-fp32`, teacher-force fp32), pas par la fixture de décode.

Conséquence directe : la penalty **divise par 1,15 les logits positifs**, comprimant les marges
de ~13 % ⇒ elle fabriquerait **davantage** de ties artificiels sur cet oracle. Un RP3 armé sur
la fixture bf16 échouerait pour une raison étrangère au code livré, et déclencherait le STOP de
§7-4 sur un faux signal.

**F7 — les nombres de marge, mesurés, existent déjà** (rév. 3 — ils remplacent une devinette).
`U_12B_RESULTS.md` : U7 fp32 `max_abs` runner↔oracle = **9,365e-4** (`:61`) · U9-iv marge
minimale réelle = **0,0279 @ gen=1043 sur 1150 steps** (`:38-40`) · les 11 « divergences » bf16
avaient des marges fp32 réelles de **0,028 à 0,24** — « toutes des artefacts de quantum ».
`ENGINE_LOG.md:277` documente par ailleurs le précédent exact du risque qu'anticipe RP4 :
« les contre-tests argmax restaient à 0 divergence — l'argmax greedy est trop robuste ».

## 2. Posture en tension avec la spec batching §3.5 — les deux restent valides

`2026-07-12-batching-flash-attn-design.md:212-220` a spécifié un sampling **tronqué au top-K**,
explicitement « approximation de charge, pas une implémentation de référence », « aucun oracle ».

| Voie | Cible | Statut |
|---|---|---|
| §3.5 batching — top-K tronqué | **mode charge, B>1** | **intention de spec, sans code** (`grep temperature gemma4_bbatch.zig` → 0) |
| Cette spec — logits complets, host, B=1 | **mode référence, oraclé** | nouvelle |

La troncature est **structurellement incapable** de porter une claim « == HF » : un token de
l'historique hors du top-K ne peut pas être pénalisé. Limite de la voie de charge, pas défaut.

## 3. Design — module

### 3.1 `zml_runner/sampling.zig` (nouveau)

Responsabilité unique : **transformer un vecteur de logits en un token**. Aucune dépendance ZML
(f32 nus) ⇒ `zig test` sans GPU, sans PJRT, sans poids.

```zig
pub const Params = struct {
    repetition_penalty: f32 = 1.0,   // 1.0 = désactivé (chemin greedy inchangé)
    ignore_prompt: bool = false,     // restreint prompt_ignore_length de HF à {0, len(prompt)}
};

/// Applique la penalty IN-PLACE, au plus une fois par token distinct (F3).
pub fn applyRepetitionPenalty(logits: []f32, hist: []const u32, penalty: f32, seen: *Bitset) void

/// argmax host. Tie-break explicite (premier indice gagnant), gaté en RP1 sur des vecteurs
/// à ties EXACTS, et surveillé en production par la marge du compteur (§3.2).
pub fn argmax(logits: []const f32) u32
```

**Quatre contraintes, chacune adossée à un gate :**

1. **Déduplication obligatoire** (F3) : un `for (hist) |t| …` naïf applique la penalty autant de
   fois que le token apparaît, et diverge exactement dans le cas qui motive le chantier.
   → contre-test RP4-(b).
2. **Le signe commande l'opération** (F3) → contre-test RP4-(a).
3. **Interdiction de l'optimisation `× (1/penalty)`** : mathématiquement équivalente, elle casse
   la bit-exactitude f32 attendue en RP1. La division reste une division.
4. **Bitset ET buffer de logits alloués UNE FOIS**, dimensionnés au vocab **runtime**
   (`gemma4_g12auto.zig:1239`). Le buffer réutilisé, pas un `toSliceAlloc` par step — sinon
   200 tokens × 20 prompts = 4 Gio alloués/libérés et le critère RSS de RP5 saute de bonne foi.

### 3.2 Câblage dans la boucle

**Si `penalty == 1.0`, les logits ne sont pas lus du tout** : le chemin actuel (top-1 du `topK`
in-graph, ~48 octets D2H) reste strictement inchangé. Le chemin host ne s'arme que si la penalty
est active, **et seulement en phase génération** (`in_gen_phase`, `:1363`).

```
DÉBUT d'itération : hist.append(fed)      ← le token qu'on s'apprête à feeder

exe.call → r_t5v, r_t5i, r_logits, caches

  penalty == 1.0 → tok = t5i[0]                     (chemin actuel, r_logits.deinit())
  sinon          → logits ← r_logits (buffer réutilisé, 1 Mio, transfert DÉJÀ synchrone)
                   applyRepetitionPenalty(logits, hist, p, seen)
                   tok = argmax(logits)
                   si tok != t5i[0] : divergences.append(.{ step, marge = t5v[0]-t5v[1] })

FIN d'itération : fed ← ids[step+1] en prefill (:1416), fed ← tok en génération (:1438)
```

⚠ **`hist.append(fed)` en TÊTE d'itération, jamais `hist.append(tok)` en fin** (corrigé rév. 4).
`in_gen_phase = step + 1 >= ids.len` (`:1363`) est **vrai au step `ids.len-1`, où `fed` vaut
encore le DERNIER token du prompt**. Un `append(if (in_gen_phase) tok else fed)` y appendrait
`s0` et **sauterait définitivement `ids[ids.len-1]`**, alors que HF a le prompt complet dans
`input_ids` au même step. La divergence RP3 qui en résulterait aurait une cause quasi
introuvable. Appender `fed` en tête donne **prompt complet ++ générés**, sans cas particulier :
`fed` du step suivant *est* le `tok` du step courant en phase génération (`:1438`).

Le **premier token généré `s0`** est produit au dernier step de prefill (`in_gen_phase` vrai dès
`step + 1 >= ids.len`, `:1363`) : la penalty **s'y applique**. L'oracle doit faire de même (§4).

### 3.2.1 Le compteur de divergences — ce qu'il prouve et ce qu'il ne prouve pas

`divergences` = liste des steps où l'argmax host diffère du top-1 device, **avec la marge**.

| Observation | Signification | Action |
|---|---|---|
| `len == 0` avec penalty ≠ 1.0 | La penalty **n'a rien fait** (paramètre non propagé jusqu'à `generateOnce` — **19** arguments positionnels `:1296`, **3 sites d'appel** `:1244`/`:1254`/`:1283`, `hist` vide, `in_gen_phase` inversé) | **FAIL bruyant** |
| une divergence à **marge exactement 0,0** | Désaccord de **tie-break** host↔device, sans rapport avec la penalty | **FAIL** — et c'est le seul endroit où un vrai tie est observable (voir §3.1-note) |
| `len > 0`, marges non nulles | La penalty **a agi**. **Ce n'est PAS une preuve de correction** : une lecture erronée du D2H (dtype, stride) ferait diverger presque tous les steps | tripwire seulement — la preuve, c'est RP3 |

Seul `len == 0` est informatif comme échec. C'est un détecteur de penalty morte, **pas un
oracle**. Il est exigé en **RP3, RP5, RP6 et RP7** — les quatre gates exposés à tourner
penalty éteinte sans que rien ne le montre.

**Note tie-break** : RP1 compare la politique host à la sémantique **documentée** de `sort`
(ZML n'est pas vendored dans ce dépôt : `tensor.zig` se lit à distance). C'est déclaratif, et le
repo a déjà payé un gate déclaratif. La règle « marge 0,0 ⇒ FAIL » ci-dessus est le complément
**opérable et gratuit**, en production.

### 3.3 Historique

`hist` = ids du prompt ++ générés (défaut HF), ou générés seuls sous `--ignore-prompt`.

**`hist` est le premier état inter-prompts jamais introduit dans ce runner.** Son scope est
`generateOnce`, jamais `run()` : chaque prompt repart d'un historique vide, comme le cache
(`:1295`, « AUCUN état ne survit entre deux appels (R1 le vérifie) »). L'invariant prouvé par
R1/R2 devient **cassable** → gate RP5.

### 3.4 Directives `--repl` et surface CLI

CLI : `--repetition-penalty <f>`, `--ignore-prompt`.
Directives : `:penalty <f>` · `:ignore-prompt on|off` · `:params` · `:help`.

**Une ligne commençant par `:` n'est jamais un prompt.** Limite assumée et documentée (un prompt
ne peut pas commencer par `:`) plutôt qu'un échappement dont personne n'a besoin. Valeur
invalide rejetée **sans tuer la session**.

⚠ **`--repl` est exclusif de `--oracle`, `--window-vacuity`, `--out-ids`, `--ids-only`,
`--selftest-*`** (`gemma4_g12auto.zig:868-874`), garde dont le motif écrit est d'éviter qu'un
repl écrase des artefacts de gate. **Cette garde n'est pas relâchée** : les gates en mode
résident (RP5, RP6) comparent donc le **texte détokenisé de stdout**, comme R1 l'avait fait au
chantier repl. Relâcher une garde de sécurité pour faire passer un test irait dans le mauvais sens.

## 4. Design — oracle (`scripts/69_u8_gen_oracle.py`, PAS le 49)

L'oracle du 12B est **`scripts/69_u8_gen_oracle.py`** (mode décode, fixture `u8_gen48`, clés
`positions` + `fed` — exactement ce que lit le mode `--oracle` de `gemma4_g12auto`).
`scripts/49_gen_custom_oracle.py` est l'oracle **E2B** (`MODEL_ID = "google/gemma-4-E2B-it"`,
`:35`) : hors périmètre depuis C3.

**Favorable à C5** : le 69 lit `out.logits` **déjà softcappé par HF dans le forward** (assert
≤ 30 à chaque step), là où le 49 recalcule la tête à la main. Le point d'insertion du processor
est donc plus propre sur le 69.

Extensions requises :

1. **Lever la restriction `--compute-fp32` au mode décode** (`:371-372`). Les hooks
   `install_fp32_hooks` **existent déjà** (`:69-93`) et l'oracle fp32 est **~5× plus rapide**
   que le bf16 émulé (`U_12B_RESULTS.md:41`) : le coût n'est pas l'obstacle. → RP-1.
2. **Importer et appeler le vrai processor** :
   `from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor`.
   **Interdiction de retranscrire le `torch.where` de F3.**
3. **Appliquer la penalty aussi à `s0`**, produit par le prefill hors boucle (`:143-153`,
   `seq = [int(idxs[0])]`, la boucle démarre `:160`).
4. **Exporter `prompt_ids` comme tenseur** de la fixture : `save_file` (`:191`) ne l'exporte pas,
   il ne vit que dans le manifest JSON (`:204`). Sous penalty les ids du prompt **entrent dans le
   calcul**, or le contrôle côté runner ne porte que sur **la longueur**
   (`gemma4_g12auto.zig:977-981`, déviation assumée : « les prompt_ids complets ne vivent que
   dans le manifest sidecar JSON »). Deux tokenisations de même longueur donneraient
   silencieusement des pénalités différentes.
5. **Journaliser la version de `transformers`** : la formule a évolué entre versions, la claim
   est version-relative.
6. **Producteur de fixtures RP1** : un script séparé émettant des triplets
   `(logits_in, hist, penalty) → logits_out` **au niveau vecteur** (ce n'est pas ce qu'émet un
   oracle de génération). Il ne tourne que là où `transformers` est installé.

Nouveaux flags : `--repetition-penalty`, `--ignore-prompt`.

## 5. Gates — phase 1 (`RP`, déterministe)

### RP-1 — PRÉREQUIS : refonder l'instrument (rév. 3)

**Rien n'est armé tant que RP-1 n'est pas PASS.**

| Étape | Contenu | Critère |
|---|---|---|
| a | Lever `--compute-fp32` au mode décode du 69 (§4-1) | la commande s'exécute, logits **f32** (l'assert bf16 `:148` devient conditionnel) |
| b | Régénérer `u8_gen48` **en fp32** | fixture produite, dtype consigné |
| c | **Re-valider le greedy 48/48 STRICT** via `--oracle` contre la fixture fp32 | **48/48**. C'est **ce run** qui refonde l'instrument. |

Si (c) ne donne pas 48/48, **le chantier penalty s'arrête** : le problème est dans l'instrument,
pas dans un code non encore écrit. Rappel F6 : contre la fixture **bf16**, le runner marque
42/48 — ce chiffre n'est pas un échec du runner, c'est la signature du quantum.

### Gates du chantier

| Gate | Contenu | Critère PASS |
|---|---|---|
| **RP0** | **Graphe intact** — dumps HLO de `g12auto` avant/après, témoins pris **avant toute édition**, worktree homogène | `diff -rq` identique **sauf les tolérances F5** ; **publier** le nombre observé (tranche la contradiction ENGINE_LOG:92 / ZML_MODULAR:132). **+ contre-test** : perturbation délibérée (`RMS_EPS` 1e-6→1e-2, précédent `ZML_MODULAR_ENGINE_DESIGN.md:132`) ⇒ RP0 doit **FAIL**. **+ garde** : dumps non vides et binaire effectivement rebuildé. |
| **RP1** | **Unitaire sans GPU** — `zig test sampling.zig` vs fixtures du **vrai processor** (§4-6), penalty ∈ {0.8, 1.0, 1.15, 1.5} | **bit-exact, 0 ULP**, aucun repli pré-approuvé. **Le test asserte sa propre fixture** (≥1 doublon, ≥1 négatif pénalisé, ≥1 positif) et **échoue si un compteur vaut 0**. **+ tie-break** sur ties f32 exacts (complété en production par la règle marge 0,0, §3.2.1). |
| **RP2** | **Non-régression** — penalty=1.0, 12B | ids **bit-identiques au témoin**, le témoin étant **la sortie du runner AVANT modification**, prise en même temps que les dumps RP0 — **jamais `u8_gen48`** (contre lequel le runner marque 42/48 en bf16, F6). Quasi-tautologique et assumé : couvre le câblage, pas le calcul. |
| **RP3** | **Oracle HF exact, penalty active** — 12B, penalty ∈ {0.8, 1.15}, oracle **69 fp32** étendu | **ids == HF** · **mordant pré-calculé sur l'oracle SEUL, avant tout run GPU** : `hamming(ids_HF_penalty, ids_HF_greedy) ≥ 3` / 48, **sinon la configuration est déclarée inutilisable** et le prompt change · **publier le hamming réel** (attendu ~40 par effet de cascade, pas 3) · **vérifier que penalty 0,8 l'atteint aussi** (elle *récompense* la répétition : c'est le cas le moins évident) · **`divergences.len > 0`** sinon FAIL (§3.2.1) · **publier la marge min top1−top2**, avec et sans penalty. |
| **RP4** | **Non-vacuité** — 3 corruptions : (a) signes inversés, (b) dédup supprimée, (c) prompt inclus/exclu à tort | **chacune doit faire FAIL RP3** et **publier son mordant**, plancher **1**. **De plus** : un mordant `< mordant_naturel(RP3) / 2` est **à instruire, pas à valider** — vu la cascade, une vraie corruption sémantique mord massivement ; un mordant ras-du-plancher signale un test au bord de sa détection. Remédiation **écrite d'avance** : prompt contenant des tokens qui sont aussi des continuations probables. Précédent à garder en tête : `ENGINE_LOG.md:277`, « l'argmax greedy est trop robuste ». |
| **RP5** | **Aucun état ne survit entre prompts** — `--repl`, même prompt joué 2×, penalty active, **`max_tokens = 32`** | **texte détokenisé identique** (l'export d'ids est refusé en repl, §3.4) **+ `divergences.len > 0` sur les DEUX passes** (sinon deux runs greedy identiques passeraient à vide) **+ RSS ≤ +1 Mo sur 20 prompts** (repère R2, atteignable **parce que** le buffer de logits est alloué une fois, §3.1-4). 20 × 32 tokens ≈ 2 min de génération. |
| **RP6** | **Directives repl** — (a) `:penalty` ne produit **aucune** génération (**comptage**) ; (b) la valeur s'applique au prompt suivant, sur un prompt **dont la sensibilité est prouvée par RP3**, avec **`divergences.len > 0`** ; (c) `:params` vérifié **contre le comportement** : `:params` dit 1.15 ⇒ la sortie du repl égale la **détokenisation de la fixture oracle 1.15**, bornée aux **48 premiers tokens** (rév. 4 : la comparaison porte sur le **texte**, pas sur les ids — `--repl` refuse l'export d'ids, §3.4 ; et les conditions d'arrêt diffèrent, EOT/`max_tokens` côté repl vs `fed.len` côté oracle) ; (d) invalides **énumérées** : `0`, `-1`, `nan`, `inf`, `abc`, vide | 4/4 |
| **RP7** | **12B — la récitation est-elle levée** — balayage `:penalty` à chaud, **valeurs {1.05, 1.1, 1.15, 1.3}**, `max_tokens = 200` | **Métrique** : longueur maximale de n-gramme répété sur les 200 derniers tokens. **Ligne de base SAINE mesurée** sur le run U9 « 1150 tok stables, texte cohérent » (`U_12B_RESULTS.md:64`) et **publiée**. **Témoin penalty=1.0 publié d'abord** ; **si le témoin ne récite pas, le gate est vacué** et le prompt change. PASS = métrique ≤ ligne de base saine, pour au moins une valeur du balayage, **avec `divergences.len > 0`**. ⚠ RP7 ne mesure **pas la qualité** : une penalty haute peut casser la boucle en produisant du charabia — la sortie de chaque valeur est **jointe au rapport** pour lecture humaine. |

**M1 — mesure (pas un gate)** : coût du chemin host.
**Instrument** : **chronomètre autour du bloc hôte** (µs/step : lecture D2H + penalty + argmax) —
mesure exacte, sans bruit GPU. **Design apparié** : deux bras dans **une seule session `--repl`**,
entrelacés, une seule compile — ce qui élimine la variance inter-run.
Le tok/s bout-en-bout n'est plus qu'un **recoupement grossier**.
**Attendu pré-enregistré : ~0,5 %** (D2H 1 Mio ≈ 40-100 µs + argmax 262 144 f32 ≈ 100-200 µs,
sur ~110 ms/step). Plafond 10 % = tripwire, **non discriminant et assumé comme tel**.
*Motif du changement d'instrument : à n=3 sur un GPU partagé, exiger « dispersion < écart » pour
un effet de 0,5 % garantit la conclusion « non concluant » — honnête, mais inutile.*

## 6. Gates — phase 2 (`SM`, stochastique) — livrés après la phase 1

| Gate | Contenu | Critère PASS |
|---|---|---|
| **SM0** | **Pont** — `--top-k 1` avec **température ≠ 0** et **10 seeds** | ids identiques aux 10 seeds **et** == greedy. (`--temperature 0` seul serait tautologique : cas spécial branché sur `argmax`, sinon division par zéro.) |
| **SM0-bis** | **« Zéro RNG device » prouvé** — dumps HLO après la phase 2 | HLO **inchangé** (tolérances F5). Un `Tensor.Rng` in-graph le ferait bouger. Le repo a déjà payé un gate déclaratif : celui-ci est opérable et gratuit. |
| **SM1** | **Distributionnel** — 10 000 tirages sur des logits **figés en fixture** | χ² sous **α = 0,01** écrit d'avance. **Distribution théorique issue d'une implémentation INDÉPENDANTE** (torch/scipy) — sinon un `×temp` au lieu de `÷temp` (le piège F4 !) déplace les deux distributions à l'identique et le χ² passe. **Binning** : support restreint ~10 catégories, effectif attendu ~1000 (σ≈31) — 262 144 catégories invalideraient le test. **Non-vacuité** : un biais injecté de **5 %** (≈50 comptes) doit **FAIL**. **Règle de re-run ÉCRITE** : un seul re-run, **seed pré-déclarée**, deux échecs = FAIL. |
| **SM2** | **Reproductibilité** — RNG **host** seedé par (seed, step) | même seed ⇒ sortie identique |
| **SM3** | **Non-vacuité du RNG** — N seeds | **≥ k sorties distinctes parmi N**, k et N **calculés depuis la fixture avant la mesure** : sur une distribution piquée, deux seeds donnent légitimement la même sortie. |

## 7. Vigilances pré-enregistrées (écrites AVANT de mesurer)

1. **La penalty déforme les écarts — asymétriquement.** Branche positive : elle **comprime**
   (÷1,15) et peut créer des quasi-ties. Branche négative : elle **éloigne** de zéro (×1,15).
   La claim est « ids == HF », **jamais** « logits bit-identiques ».
2. **ε de quasi-tie = 2e-3**, **dérivé, pas deviné** : 2 × le bruit fp32 mesuré runner↔oracle
   (9,365e-4, F7) majoré par la branche négative (×1,15) ≈ 2,15e-3, arrondi à **2e-3**.
   **Prédiction falsifiable qui l'accompagne** : la marge minimale réelle observée est **0,0279
   sur 1150 steps** (F7) — donc **aucun step n'a jamais été vu sous 2e-3**. La fenêtre
   d'admissibilité de la vigilance 3 est **attendue VIDE** : l'ouvrir serait un événement à
   instruire, pas une routine.
   ⚠ **1e-3 est explicitement banni** : c'est le seuil du menu de requalification du 25 juillet
   (`U_12B_RESULTS.md:26`, « un menu dont le critère glissait déjà : marge ≤ 1e-3 → ≤ 2 ULP »),
   c'est-à-dire l'instrument mis au rebut. Il est écrit ici pour que personne ne le réintroduise.
3. **Un mismatch RP3 n'est admissible que si les TROIS conditions tiennent** : (i) marge
   `|top1−top2| < 2e-3` ; (ii) le top-2 de HF est **la même paire inversée** ; (iii) **au plus
   1** step de ce type sur 48. Hors bornes : FAIL.
4. **RP1 est à 0 ULP.** Aucun repli pré-approuvé : tout écart est un FAIL publié, puis instruit.
5. **Procédure d'échec** (pattern A2) : au premier mismatch → top-5 du step fautif et marge.
   **Le FAIL brut est publié d'abord** ; toute requalification vient ensuite et est datée.
   **Toute 2ᵉ requalification, de quelque type que ce soit, = STOP** : on diffe l'instrument.
6. **Oracle en fp32** — et cette vigilance n'est honorable **qu'après RP-1** (F6). Avant RP-1,
   elle serait un vœu : l'outillage de décode 12B ne sait pas produire du fp32.
7. **Témoins RP0 pris avant tout deploy**, worktree homogène — le piège qui avait mordu à M0.
8. **Seuils fixés maintenant** : mordant RP3 ≥ 3 (et hamming réel publié) · plancher RP4 = 1,
   avec instruction si `< mordant_naturel/2` · ε = **2e-3** · RP7 = ligne de base saine mesurée ·
   M1 attendu ~0,5 %, plafond 10 % · α SM1 = 0,01, biais de non-vacuité 5 %.

## 8. Gestion d'erreur

- **Garde penalty en ACCEPTATION** : `p > 0 and std.math.isFinite(p)`. La transcription naïve
  `p <= 0 → rejet` **laisse passer `NaN`** (`NaN <= 0` est faux) : `:penalty nan` empoisonnerait
  tous les logits et l'argmax renverrait un token arbitraire **sans erreur** — un repli
  silencieux parfait, dans le dispositif censé les interdire.
- Bitset **et** buffer de logits alloués **une fois**, dimensionnés au vocab runtime.
- Token hors vocab dans `hist` : déjà gardé (`:1343`), assert conservé.
- **Un D2H échoué remonte une erreur** — **pas de repli silencieux** sur le top-5, qui rendrait
  RP3 incapable d'échouer.

## 9. Hors périmètre

Câblage de `gemma4_gen_auto.zig` (E2B — dette écrite, §0) · multi-tour avec contexte accumulé ·
`no_repeat_ngram_size` · sampling par lane en B>1 (voie §3.5) · beam search · toute modification
de `engine.zig` ou du graphe **en dehors de la perturbation jetable du contre-test RP0**
(worktree dédié, jamais committée, §7-7).

## 10. Livrables

- `zml_runner/sampling.zig` + tests `zig test`
- câblage `gemma4_g12auto.zig` (hérité par `g12a4k`/`g12a8k`)
- `scripts/69_u8_gen_oracle.py` étendu : `--compute-fp32` en mode décode (RP-1), vrai processor,
  `s0`, `prompt_ids` en tenseur, version transformers journalisée
- **fixture `u8_gen48` régénérée en fp32** (RP-1)
- **script producteur des fixtures RP1** (triplets au niveau vecteur) + les fixtures committées
- `docs/SAMPLING_RESULTS.md` : résultats, chiffres mesurés, témoins publiés (RP0 nombre de diffs,
  RP3 hamming réel et marge min, RP7 ligne de base et sorties du balayage, M1 µs/step)
- tags `gate/rp-1-pass`, `gate/rp0-pass` … `gate/rp7-pass`, une PR

## 11. Ordre d'exécution

```
RP-1 (refonder l'oracle fp32)  ──┐
                                 ├─→ RP0 (témoins + graphe intact)
RP1 (unitaire, sans GPU)  ───────┘        │
                                          ├─→ RP2 (non-régression)
                                          ├─→ RP3 (oracle HF) → RP4 (non-vacuité)
                                          ├─→ RP5, RP6 (repl)
                                          └─→ RP7 (récitation) + M1 (mesure)
```

RP1 ne dépend d'aucun GPU et peut démarrer immédiatement. **RP3 est bloqué par RP-1.**

## 12. Historique de révision

**Rév. 2** — après le 1ᵉʳ tour de double revue (cohérence/faits et falsifiabilité) : F1 faux pour
l'E2B → périmètre re-arbitré au 12B ; F2 citait `forwardStep` au lieu de `forwardStageGen` ;
F5 ajouté (`diff -rq` vide inatteignable) ; C5 (vrai processor) ; RP3 mordant + compteur ;
RP5 ajouté (état inter-prompts) ; RP0 contre-test ; RP1 à 0 ULP ; ancien RP2 tautologique
supprimé ; SM0-bis ; garde `NaN` ; règle du STOP élargie à **toute** 2ᵉ requalification.
16 corrections.

**Rév. 3** — après le 2ᵉ tour :

| # | Correction |
|---|---|
| 1 | **F6 + C6 + RP-1** : l'oracle de décode 12B est **bf16 par construction** — l'instrument déclaré corrompu le 25 juil. RP3 ne pouvait pas réussir, et §7-6 « oracle fp32 » était inapplicable. Prérequis de refondation ajouté (décision Régis). |
| 2 | **§4 rebranché sur `scripts/69`** : la rév. 2 avait déplacé le périmètre sur le 12B sans déplacer l'oracle, resté celui de l'E2B (`49`, `MODEL_ID = E2B`). RP3 était inexécutable. |
| 3 | **ε : 1e-3 → 2e-3 dérivé** de F7 (9,365e-4 mesuré), **avec la prédiction falsifiable** « fenêtre attendue vide » (marge min réelle 0,0279). 1e-3 était **le seuil du menu de requalification au rebut** : banni explicitement. |
| 4 | **RP5/RP6 comparent le texte**, pas les ids : `--repl` est exclusif de `--oracle`/`--out-ids` (`:868-874`). La garde n'est **pas** relâchée. |
| 5 | **`divergences.len > 0` exigé en RP5, RP6, RP7** — pas seulement RP3 : ces gates pouvaient tourner penalty éteinte et passer à vide. |
| 6 | **§3.2.1** : ce que le compteur prouve et **ne prouve pas** (seul `len == 0` est informatif) ; **marge 0,0 ⇒ FAIL tie-break** — le complément opérable au tie-break déclaratif de RP1. |
| 7 | **Buffer de logits alloué une fois** (comme le bitset) : sans ça, 4 Gio alloués/libérés feraient sauter le critère RSS de RP5 de bonne foi. RP5 chiffré (RSS +1 Mo, `max_tokens=32`). |
| 8 | **M1 : instrument changé** — chronomètre host + bras appariés en session `--repl`. À n=3 sur GPU partagé, « dispersion < écart » pour 0,5 % garantissait « non concluant ». Attendu affiné à ~0,5 %. |
| 9 | **RP7 ancré sur une ligne de base mesurée** (run U9 sain) au lieu d'un ratio ÷2 deviné ; valeurs du balayage et `max_tokens` fixés ; garde-fou « la penalty haute produit du charabia » (RP7 ne mesure pas la qualité). |
| 10 | **RP2 : témoin nommé** = sortie du runner **avant modification**, jamais `u8_gen48` (42/48 en bf16). |
| 11 | **RP4** : plancher 1 complété par « mordant `< naturel/2` ⇒ instruire » ; précédent `ENGINE_LOG.md:277` cité. |
| 12 | **RP3** : mordant **pré-calculable sur l'oracle seul, sans GPU** ; hamming réel à publier ; cas `penalty 0,8` explicitement vérifié. |
| 13 | **SM1 : règle de re-run écrite** (un seul re-run, seed pré-déclarée, deux échecs = FAIL) — elle était exigée par le critère mais absente. |
| 14 | **RP6-(c) borné** aux « 48 premiers ids » (conditions d'arrêt différentes entre repl et oracle). |
| 15 | Citations : §4 → `gemma4_g12auto.zig:977-981` (et non `gen_auto:866-870`) ; renvoi `NaN` §4 → §8. |
| 16 | **F5** : contradiction interne du repo signalée (ENGINE_LOG:92 = 2 diffs, ZML_MODULAR:132 = 1) — RP0 tranche en publiant. |
| 17 | **§9** : le contre-test RP0 perturbe `RMS_EPS`, ce que « aucune modification d'`engine.zig` » interdisait — exception explicite (jetable, worktree dédié, jamais committée). |
| 18 | **§10** : script producteur des fixtures RP1 ajouté aux livrables · **§11** : ordre d'exécution ajouté. |

**Rév. 4** — corrections remontées par la revue du plan d'implémentation (le plan a servi de
test de la spec) :

| # | Correction |
|---|---|
| 1 | **§3.2 — off-by-one sur `hist`, bug réel.** La rév. 3 écrivait `hist.append(tok)` en phase génération et `fed` sinon. Or `in_gen_phase` (`:1363`) est vrai **au step où `fed` vaut encore le dernier token du prompt** : ce token n'entrait **jamais** dans `hist`, alors que HF l'a dans `input_ids`. Règle corrigée : **`append(fed)` en tête d'itération**, sans cas particulier. |
| 2 | **§5 RP6-(c) — comparaison impossible** : elle exigeait des ids en mode `--repl`, qui refuse leur export (§3.4). Portée sur le **texte détokenisé** de la fixture oracle, bornée à 48 tokens. |
| 3 | §3.2.1 : `generateOnce` prend **19** arguments (pas 15) et a **3 sites d'appel** — le mode d'échec « paramètre non propagé » est d'autant plus probable. |
| 4 | **Amendement 1** (du plan) : RP1 se fait par `--selftest-penalty` sur le pattern `--selftest-inputs`, **pas** par `zig test` — `BUILD.bazel` ne charge que `zig_binary`, aucun target de test n'existe, `zig` est absent du PATH de la machine de dev. |
