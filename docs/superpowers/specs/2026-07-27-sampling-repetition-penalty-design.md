# Spec — Repetition penalty (phase 1, déterministe) puis sampling (phase 2, stochastique)

> **Date** : 2026-07-27 · **Niveau** : standard (spec → plan → gates → PR) ·
> **Base** : PR #17 mergée (`2d4cda6`), HEAD `86d31d2`. Demande Régis : « continuons sur le
> backlog : sampling + repetition penalty ». Symptôme d'origine : en greedy, le 12B **boucle
> sur la récitation** — comportement de modèle, pas un bug du portage.
>
> **Révision 2** (même jour, après double revue de spec). La v1 affirmait que les logits
> sortaient déjà du graphe **des deux runners** : c'est **faux pour l'E2B**
> (`gemma4_gen_auto.zig:753` retourne 6 sorties, sans logits). Le périmètre a été
> re-arbitré (§0 C3) et 11 gates ont été durcis ou ajoutés. Détail en §10.

## 0. Décisions de cadrage (arbitrées avec Régis)

| # | Décision | Motif |
|---|---|---|
| **C1** | **Deux phases**, penalty d'abord | La penalty sur greedy reste **déterministe** → HF sait faire `greedy + repetition_penalty` → **l'oracle « ids == HF » reste opérable**. Le stochastique casse l'oracle argmax et exige des gates distributionnels : on ne mélange pas les deux natures de preuve. |
| **C2** | **Approche host-side pure** | Seule voie qui laisse le **graphe intact** : « je n'ai rien cassé » se démontre par un diff, pas par une campagne. |
| **C3** | **`gemma4_g12auto.zig` SEUL est câblé** (rév. 2) | C'est le seul runner dont le graphe sort déjà les logits (F1). Câbler l'E2B exigerait d'ajouter une sortie à son tuple racine, donc de modifier son HLO — C2 et « gates lourds sur E2B » ne peuvent pas tenir ensemble. La vitesse d'itération est préservée **par RP1** (`zig test` sans GPU, où se joue l'essentiel du débogage), pas par le petit modèle. |
| **C4** | **Directives `:param` à chaud** dans `--repl` | Le host-side les rend gratuites (aucun recompile). Recharger le 12B coûte ~1 min 40 : un balayage complet de la penalty passe de N compiles à **une**. |
| **C5** | **L'oracle appelle le vrai `RepetitionPenaltyLogitsProcessor`** de transformers (rév. 2) | Retranscrire la formule dans le script oracle ferait comparer deux transcriptions **de la même lecture par le même auteur** : une inversion de branche commise des deux côtés passerait le gate et la claim serait fausse. Voir §5 RP3. |

**Hors périmètre décidé** : câbler `gemma4_gen_auto.zig` (E2B). Dette **écrite** au backlog :
« exposer les logits de `StepTok` → +1 sortie au tuple racine → arbitrage HLO à refaire ».
Ce n'est pas un oubli, c'est un report explicite.

## 1. Faits établis par lecture du code

**F1 — les logits sortent déjà du graphe du 12B, et PAS de celui de l'E2B.**

- `gemma4_g12auto.zig:830` : `return .{ t5.values, t5.indices, logits, … }` — **7 sorties**,
  logits en 3ᵉ position (ajoutée pour `--window-vacuity`). Réception `:1357`, commentaire
  « NON lue ici (pas de D2H) » `:1356`, `r_logits.deinit()` `:1401`. Le chemin host est donc
  à un `toSliceAlloc` près : **aucune modification du graphe**.
- `gemma4_gen_auto.zig:753` : `return .{ t5.values, t5.indices, slk, slv, flk, flv }` —
  **6 sorties, pas de logits** (côté host, `call_results.get` attend bien 6 buffers `:1041`).

C'est **exactement** ce qui fonde C3. La v1 de cette spec généralisait le premier cas au
second : correction portée en rév. 2.

**F2 — les logits sont post-softcap, sur les quatre chemins du moteur.**
`softcapPrec` (`engine.zig:193`) est appliqué aux logits en `:766` (`forward`), **`:807`
(`forwardStageGen`)**, `:850` (`forwardStep`) et `:889` (`forwardStageStep`).
**Le chemin du 12B est `forwardStageGen`** (`gemma4_g12auto.zig:822`) → la ligne pertinente est
**`engine.zig:807`**. `:850` appartient à `forwardStep` (ouverte `:822`), que le commentaire D12
(`gemma4_g12auto.zig:794-801`) rejette explicitement pour le 12B.

L'objet rapatrié est donc **exactement celui que HF passe à ses `LogitsProcessor`** : aucune
part du modèle n'est à réimplémenter côté host.

**F3 — formule HF** (`transformers/generation/logits_process.py`,
`RepetitionPenaltyLogitsProcessor.__call__`) :

```python
penalty_scores = torch.where(last_scores < 0, last_scores * penalty, last_scores / penalty)
scores = torch.where(token_mask, penalty_scores, last_scores)
```

- logit **négatif → ×penalty** ; **positif ou nul → ÷penalty** ;
- `token_mask` **booléen** ⇒ penalty appliquée **au plus une fois par token distinct** ;
- tokens considérés = **`input_ids` entier, prompt inclus** par défaut ; `prompt_ignore_length`
  (un **entier**, pas un booléen) permet de n'en garder que le généré ;
- `__init__` rejette `penalty ≤ 0` via `not (penalty > 0)` — **formulation en acceptation**,
  qui exclut aussi `NaN`. Voir §4 : la transcription naïve `penalty <= 0` laisse passer `NaN`.

**F4 — le RNG device est écarté** (phase 2). `docs/ZML_UPSTREAM_AUDIT_2026-07-12.md:36-41` :
`sampleTokens`/`sampleTokensDynamic`/`Tensor.Rng` existent déjà dans le ZML vendored
(`adee932e`) — **aucun bump requis** — mais « le RNG device n'est pas garanti déterministe entre
backends/versions et son état global lie le bruit par lane à B ». Piège consigné : dans
`adee932e`, `sampleTokensDynamic` **multiplie** par la température **au lieu de diviser**
(bugfix upstream `73498264` non appliqué). Ce piège dicte la conception de SM1 (§6).

**F5 — la byte-identité HLO stricte n'existe pas dans ce repo.**
`docs/ENGINE_LOG.md:92` (preuve E1/E2, Task 0) : `diff -rq` sur 1037 fichiers → identiques
**sauf 2 diffs bénins** : (a) `debug_options`, qui n'encode que le chemin `--xla_dump_to` et
diffère donc **par construction** entre `before/` et `after/` ; (b) un `.ir-with-opt.ll` où
seuls des **noms SSA LLVM** diffèrent (alpha-équivalent). Le critère RP0 énumère ces
tolérances **avant** la mesure (§5).

## 2. Posture en tension avec la spec batching §3.5 — les deux restent valides

`docs/superpowers/specs/2026-07-12-batching-flash-attn-design.md:212-220` a spécifié un sampling
**tronqué au top-K rapatrié**, explicitement « approximation de charge, pas une implémentation
de sampling de référence », avec « aucun oracle en mode sampling ».

Cette spec **n'annule pas** ce choix, elle en ouvre un second à côté :

| Voie | Cible | Statut |
|---|---|---|
| §3.5 batching — top-K tronqué, D2H ≈ K×8 o/lane/step | **mode charge, B>1** | **intention de spec, sans code à ce jour** (`grep temperature zml_runner/gemma4_bbatch.zig` → 0 occurrence) |
| Cette spec — logits complets, host, B=1 | **mode référence, oraclé** | nouvelle |

La troncature au top-K est **structurellement incapable** de porter une claim « == HF » : un
token de l'historique hors du top-K ne peut pas être pénalisé. C'est une limite de la voie de
charge, pas un défaut à corriger.

## 3. Design — module

### 3.1 `zml_runner/sampling.zig` (nouveau)

Responsabilité unique : **transformer un vecteur de logits en un token**. Aucune dépendance ZML
(f32 nus) ⇒ testable par `zig test` sans GPU, sans PJRT, sans poids.

```zig
pub const Params = struct {
    repetition_penalty: f32 = 1.0,   // 1.0 = désactivé (chemin greedy inchangé)
    ignore_prompt: bool = false,     // restreint prompt_ignore_length de HF à {0, len(prompt)}
};

/// Applique la penalty IN-PLACE, au plus une fois par token distinct (F3).
/// `seen` : bitset dimensionné au vocab RUNTIME, alloué une fois, remis à zéro par appel.
pub fn applyRepetitionPenalty(logits: []f32, hist: []const u32, penalty: f32, seen: *Bitset) void

/// argmax host. Politique de tie-break explicite (premier indice gagnant), gatée en RP1
/// sur des vecteurs à ties EXACTS — le seul endroit où un tie est provoquable.
pub fn argmax(logits: []const f32) u32
```

**Quatre contraintes d'implémentation, chacune adossée à un gate :**

1. **Déduplication obligatoire** (F3). Un `for (hist) |t| logits[t] = …` naïf applique la penalty
   autant de fois que le token apparaît — et diverge exactement dans le cas qui motive le
   chantier. Bitset ⇒ coût `O(|hist|)`, sémantique identique à HF. → contre-test RP4-(b).
2. **Le signe commande l'opération** (F3). → contre-test RP4-(a).
3. **Interdiction de l'optimisation `× (1/penalty)`.** Mathématiquement équivalente, elle **casse
   la bit-exactitude f32** attendue en RP1 : la division doit rester une division.
4. **Le bitset est dimensionné sur le vocab lu au runtime**
   (`model.embed_tokens.dim(.voc)`, cf `gemma4_g12auto.zig:1239`), jamais sur une constante.

### 3.2 Câblage dans la boucle — pourquoi la non-régression est gratuite

**Si `penalty == 1.0`, les logits ne sont pas lus du tout** : le chemin actuel (top-1 du `topK`
in-graph, ~48 octets D2H) reste strictement inchangé. Le chemin host ne s'arme que si la penalty
est active, **et seulement en phase génération** (`in_gen_phase`, `:1363`).

```
exe.call → r_t5v, r_t5i, r_logits, caches

  hist : pendant le PREFILL on mémorise `fed` (le token du prompt),
         PAS `tok` (l'argmax du step, ignoré en prefill — cf :1415).

  penalty == 1.0 → tok = t5i[0]                     (chemin actuel, r_logits.deinit())
  sinon          → logits = r_logits.toSliceAlloc   (1 Mio, transfert DÉJÀ synchrone)
                   applyRepetitionPenalty(logits, hist, p, seen)
                   tok = argmax(logits)
                   compteur_divergence += (tok != t5i[0])   // cf RP3
  hist.append(tok)   // en phase génération
```

Le **premier token généré `s0`** est produit au dernier step de prefill (`in_gen_phase` vrai dès
`step + 1 >= ids.len`, `:1363`) : la penalty **s'y applique**. L'oracle doit faire de même
(§4) — sinon divergence garantie dès le token 0.

Le D2H s'ajoute à un transfert **déjà synchrone** (la boucle lit le top-5 à chaque step) : il ne
sérialise rien de nouveau. Son coût est **mesuré en M1**, il n'est supposé nulle part.

**`compteur_divergence`** (nombre de steps où l'argmax host diffère du top1 device) est exposé en
fin de génération. À penalty ≠ 1.0, un compteur nul signifie que la penalty **n'a rien fait** :
c'est une erreur bruyante, pas un succès silencieux (RP3).

### 3.3 Historique

`hist` = ids du prompt ++ générés (défaut HF), ou générés seuls sous `--ignore-prompt`.

**`hist` est le premier état inter-prompts jamais introduit dans ce runner.** Son scope est
`generateOnce`, jamais `run()` : chaque prompt repart d'un historique vide, comme le cache
(`gemma4_g12auto.zig:1295`). L'invariant « aucun état ne survit entre deux prompts », prouvé
par R1/R2 au chantier repl, devient **cassable** — d'où le gate RP5, qui n'existait pas en v1.

### 3.4 Directives `--repl` et surface CLI

CLI : `--repetition-penalty <f>`, `--ignore-prompt`.
Directives : `:penalty <f>` · `:ignore-prompt on|off` · `:params` · `:help`.

Règle unique et testable : **une ligne commençant par `:` n'est jamais un prompt**. Limite
assumée et documentée (un prompt ne peut pas commencer par `:`) plutôt qu'un mécanisme
d'échappement dont personne n'a besoin (YAGNI). Valeur invalide rejetée **sans tuer la session**.

## 4. Design — oracle

`scripts/49_gen_custom_oracle.py` **n'utilise pas `generate()`** : c'est une boucle greedy
maison (`next_token` `:129-132`, `lg.argmax(dim=-1)`). Son extension obéit à C5 :

1. **Importer et appeler le vrai processor** :
   `from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor`.
   **Interdiction de retranscrire le `torch.where` de F3** — c'est le point où une erreur
   commune aux deux implémentations passerait inaperçue.
2. **Appliquer la penalty aussi à `s0`**, calculé hors boucle (`:137`).
3. **Exporter `prompt_ids` comme tenseur** de la fixture, pas seulement dans le `.manifest.json`
   sidecar (`:196-217`). Sous penalty, les ids du prompt **entrent dans le calcul** : deux
   tokenisations de même longueur donnent silencieusement des pénalités différentes, et le mode
   `--oracle` ne contrôle aujourd'hui que **la longueur** du prompt
   (`gemma4_gen_auto.zig:866-870`, déviation assumée et documentée).
4. **Journaliser la version de `transformers`** dans `SAMPLING_RESULTS.md` : la formule a évolué
   entre versions — la claim est version-relative.

Nouveaux flags : `--repetition-penalty`, `--ignore-prompt`.

## 5. Gates — phase 1 (`RP`, déterministe)

| Gate | Contenu | Critère PASS |
|---|---|---|
| **RP0** | **Graphe intact** — dumps HLO de `g12auto` avant/après, témoins pris **avant toute édition**, worktree homogène | `diff -rq` identique **sauf les 2 tolérances énumérées en F5** (`debug_options`, noms SSA LLVM). **+ contre-test** : une perturbation délibérée du graphe (pattern E1 : `RMS_EPS` 1e-6→1e-2) doit faire **FAIL** RP0. **+ garde** : dumps non vides et binaire effectivement rebuildé (`diff -rq` de deux dossiers vides est vide). |
| **RP1** | **Unitaire sans GPU** — `zig test sampling.zig` vs fixtures produites par le **vrai processor** HF : penalty ∈ {0.8, 1.0, 1.15, 1.5} | **bit-exact, 0 ULP** — aucun repli pré-approuvé. **Le test asserte les propriétés de sa propre fixture** (≥1 token en doublon, ≥1 logit négatif pénalisé, ≥1 positif pénalisé) et **échoue si un compteur vaut 0**. **+ tie-break** : vecteurs à ties f32 **exacts**, politique comparée à la sémantique documentée de `sort`. |
| **RP2** | **Non-régression** — penalty=1.0, 12B | ids **bit-identiques** au témoin. Quasi-tautologique et **assumé comme tel** (même code) : il couvre le câblage, pas le calcul. |
| **RP3** | **Oracle HF exact, penalty active** — 12B, penalty ∈ {0.8, 1.15}, script 49 étendu (C5) | **ids == HF** · **+ mordant pré-enregistré** : `hamming(ids_HF_penalty, ids_HF_greedy) ≥ 3` sur 48 steps, **sinon la configuration est déclarée inutilisable** et le prompt change (le gate ne peut pas passer à vide) · **+ `compteur_divergence` > 0**, sinon **FAIL bruyant** · **+ publier** la marge min top1−top2, avec et sans penalty. |
| **RP4** | **Non-vacuité** — 3 corruptions : (a) branches de signe inversées, (b) dédup supprimée, (c) prompt inclus/exclu à tort | **chacune doit faire FAIL RP3**, et **publier son mordant** (nb de steps dont l'id change), **plancher 1**. Si une corruption ne mord pas, la remédiation est **écrite d'avance** : prompt contenant des tokens qui sont aussi des continuations probables, et re-run — jamais une requalification en « effet trop faible ». |
| **RP5** | **Aucun état ne survit entre prompts** (§3.3) — `--repl`, même prompt joué 2×, penalty active | `ids(p1) == ids(p2)` (rejeu du contrôle p1==p3 de R1) **+ RSS/VRAM plate sur 20 prompts** penalty active (rejeu de R2 — c'est ici que se voit un `hist` ou un bitset qui grossit, pas dans une comparaison d'ids). |
| **RP6** | **Directives repl** — (a) `:penalty` ne produit **aucune** génération (**comptage**) ; (b) la valeur s'applique au prompt suivant, sur un prompt **dont la sensibilité à la penalty est prouvée par RP3** ; (c) `:params` vérifié **contre le comportement** (`:params` dit 1.15 ⇒ ids == référence 1.15), pas contre son propre écho ; (d) valeurs invalides **énumérées** : `0`, `-1`, `nan`, `inf`, `abc`, vide | 4/4 |
| **RP7** | **12B — la récitation est-elle levée** — balayage `:penalty` à chaud (une seule compile) | **Métrique chiffrée** : longueur maximale de n-gramme répété sur les 200 derniers tokens. **Le témoin penalty=1.0 est publié d'abord** ; **si le témoin ne récite pas, le gate est déclaré vacué** et le prompt change. PASS = métrique du témoin divisée par ≥ 2 pour au moins une valeur du balayage. |

**M1 — mesure (pas un gate)** : coût du D2H, tok/s penalty=1.0 vs penalty active, **n ≥ 3 runs
par bras**, médiane et dispersion publiées. **Valeur attendue pré-enregistrée : ~1 %**
(1 Mio greffé sur un transfert déjà synchrone, ~110 ms/step). Plafond 10 %. **La dispersion
inter-run doit être inférieure à l'écart mesuré**, sinon l'instrument ne résout pas ce qu'il
prétend mesurer et la conclusion est « non concluant », pas « négligeable ».

## 6. Gates — phase 2 (`SM`, stochastique) — spécifiés, livrés après la phase 1

| Gate | Contenu | Critère PASS |
|---|---|---|
| **SM0** | **Pont avec la phase 1** — `--top-k 1` avec **température ≠ 0** et **10 seeds** | ids identiques aux 10 seeds **et** == greedy. (`--temperature 0` seul serait tautologique : c'est nécessairement un cas spécial branché sur `argmax`, sinon division par zéro.) |
| **SM0-bis** | **« Zéro RNG device » prouvé, pas déclaré** — dumps HLO après la phase 2 | HLO **inchangé** (mêmes tolérances F5). Un `Tensor.Rng` glissé in-graph le ferait bouger. Le repo a déjà payé un gate déclaratif (Munich) : celui-ci est opérable et gratuit. |
| **SM1** | **Distributionnel** — 10 000 tirages sur des logits **figés en fixture** | χ² sous seuil **α = 0,01 écrit d'avance**. **La distribution théorique vient d'une implémentation INDÉPENDANTE** (torch/scipy dans le script oracle), **jamais du binaire testé** : sinon un `×temp` au lieu de `÷temp` (le piège F4 !) déplace les deux distributions à l'identique et le χ² passe. **Binning** : fixture à support restreint (~10 catégories, effectif attendu ≥ 50) — 262 144 catégories invalideraient le test. **Re-run** : règle écrite d'avance (à α=0,01, un échec légitime sur 100) — un re-run non prévu est une requalification silencieuse. **Non-vacuité** : un biais injecté de 5 % doit faire **FAIL**. |
| **SM2** | **Reproductibilité** — RNG **host** seedé par (seed, step) | même seed ⇒ sortie identique |
| **SM3** | **Non-vacuité du RNG** — N seeds | **≥ k sorties distinctes parmi N**, k et N **calculés depuis la fixture avant la mesure** : sur une distribution piquée, deux seeds donnent légitimement la même sortie et « seed différent ⇒ sortie différente » échouerait sur une implémentation correcte. |

Paramètres phase 2 : `--temperature`, `--top-k`, `--top-p`, `--seed` + directives `:temp`,
`:top-k`, `:top-p`, `:seed`.

## 7. Vigilances pré-enregistrées (écrites AVANT de mesurer)

1. **La penalty déforme les écarts — asymétriquement.** Sur la branche positive elle **comprime**
   (÷1,15 rapproche de zéro) et peut créer des quasi-ties ; sur la branche négative elle
   **éloigne** de zéro (×1,15). La claim du projet est « ids == HF », **jamais** « logits
   bit-identiques » : un mismatch sous penalty n'est pas automatiquement un bug.
2. **Un mismatch RP3 n'est admissible que si les TROIS conditions tiennent**, chacune chiffrée :
   (i) marge `|top1−top2|` au step fautif `< 1e-3` ; (ii) le top-2 de HF est **la même paire
   inversée** ; (iii) **au plus 1** step de ce type sur 48. Hors de ces bornes, c'est un FAIL.
3. **RP1 est à 0 ULP.** Aucun repli n'est pré-approuvé dans son critère : tout écart est un FAIL
   publié, puis instruit.
4. **Procédure d'échec** (pattern A2) : au premier mismatch → top-5 du step fautif et marge
   top1−top2. **Le FAIL brut est publié d'abord** ; toute requalification vient ensuite et est
   datée. **Toute 2ᵉ requalification, de quelque type que ce soit, = STOP** : on diffe
   l'instrument avant d'écrire une ligne de plus (rév. 2 : la v1 disait « du même type », ce qui
   autorisait deux portes différentes sans jamais déclencher la règle).
5. **Oracle en fp32**, jamais bf16.
6. **Témoins RP0 pris avant tout deploy**, sur un worktree homogène — le piège exact qui avait
   mordu au gate M0.
7. **Seuils fixés maintenant** : mordant RP3 ≥ 3 · plancher RP4 = 1 · ε quasi-tie = 1e-3 ·
   récitation RP7 = ÷2 · M1 attendu ~1 %, plafond 10 % · α SM1 = 0,01.

## 8. Gestion d'erreur

- **Garde penalty écrite en ACCEPTATION** : `p > 0 and std.math.isFinite(p)`.
  La transcription naïve `p <= 0 → rejet` **laisse passer `NaN`** (`NaN <= 0` est faux) :
  `:penalty nan` empoisonnerait tous les logits et l'argmax renverrait un token arbitraire
  **sans erreur** — un repli silencieux parfait, dans le dispositif censé les interdire.
- Bitset alloué **une fois** au démarrage, jamais par step ; dimensionné au vocab runtime.
- Token hors vocab dans `hist` : déjà gardé (`:1343`), assert conservé.
- **Un D2H échoué remonte une erreur** — **pas de repli silencieux** sur le top-5 : un tel repli
  rendrait RP3 incapable d'échouer.

## 9. Hors périmètre

Câblage de `gemma4_gen_auto.zig` (E2B — dette écrite, §0) · multi-tour avec contexte accumulé
(`--repl` V1 reste à prompts indépendants) · `no_repeat_ngram_size` · sampling par lane en B>1
(voie §3.5) · beam search · toute modification de `engine.zig` ou du graphe.

## 10. Livrables

- `zml_runner/sampling.zig` + tests `zig test`
- câblage `gemma4_g12auto.zig` (hérité par `g12a4k`/`g12a8k`)
- `scripts/49_gen_custom_oracle.py` étendu (C5 : vrai processor, `s0`, `prompt_ids` en tenseur,
  version transformers journalisée)
- fixtures RP1 (safetensors, committées)
- `docs/SAMPLING_RESULTS.md` (résultats + chiffres mesurés + témoins)
- tags `gate/rp0-pass` … `gate/rp7-pass`, une PR

## 11. Historique de révision

**Rév. 2 (2026-07-27, après double revue de spec — cohérence/faits et falsifiabilité).**

| # | Correction |
|---|---|
| 1 | **F1 faux pour l'E2B** → périmètre re-arbitré (C3), E2B en dette écrite |
| 2 | F2 : citation corrigée `:850` (`forwardStep`, rejetée par D12) → **`:807`** (`forwardStageGen`) |
| 3 | F5 ajouté : `diff -rq` **vide** est inatteignable (ENGINE_LOG:92) → RP0 énumère ses tolérances **avant** la mesure |
| 4 | C5 : l'oracle **appelle** le vrai `RepetitionPenaltyLogitsProcessor` au lieu de retranscrire (risque de faute commune aux deux implémentations) |
| 5 | RP3 : mordant chiffré + `compteur_divergence` → le gate central ne peut plus passer avec une penalty morte |
| 6 | RP5 **ajouté** : `hist` est le premier état inter-prompts du repl, aucun gate ne le couvrait |
| 7 | RP0 : contre-test de non-vacuité + garde « dumps non vides / binaire rebuildé » |
| 8 | RP1 : 0 ULP strict (le repli était pré-approuvé dans le critère) + assertions sur la fixture + tie-break rapatrié depuis l'ancien RP2, qui était tautologique |
| 9 | Ancien RP3 : critère inadéquat à son motif → scindé en RP2 (ids) et RP5 (RSS plate) |
| 10 | RP7 : « récitation levée » chiffrée + témoin publié d'abord + gate vacué si le témoin ne récite pas |
| 11 | M1 : étiqueté **mesure**, n≥3, valeur attendue pré-enregistrée, exigence de résolution |
| 12 | RP6 : (b) prompt à sensibilité prouvée, (c) vérifié contre le comportement, (d) invalides énumérées |
| 13 | SM0 rendu mordant · **SM0-bis ajouté** (zéro RNG device prouvé par HLO) · SM1 : implémentation indépendante + binning + α + re-run + non-vacuité · SM3 : k/N pré-calculés |
| 14 | §8 : garde `NaN` (formulation en acceptation) |
| 15 | §3.1 : interdiction de `× (1/penalty)` · bitset au vocab runtime · §3.2 : `fed` vs `tok` en prefill, et `s0` |
| 16 | §7-4 : la règle du STOP couvre **toute** 2ᵉ requalification, plus seulement « du même type » |
