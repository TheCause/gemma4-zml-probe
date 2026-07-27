# Spec — Repetition penalty (phase 1, déterministe) puis sampling (phase 2, stochastique)

> **Date** : 2026-07-27 · **Niveau** : standard (spec → plan → gates → PR) ·
> **Base** : PR #17 mergée (`2d4cda6`), HEAD `86d31d2`. Demande Régis : « continuons sur le
> backlog : sampling + repetition penalty ». Symptôme d'origine : en greedy, le 12B **boucle
> sur la récitation** — comportement de modèle, pas un bug du portage.

## 0. Décisions de cadrage (arbitrées avec Régis avant rédaction)

| # | Décision | Motif |
|---|---|---|
| **C1** | **Deux phases**, penalty d'abord | La penalty sur greedy reste **déterministe** → HF sait faire `greedy + repetition_penalty` → **l'oracle « == HF exact » reste opérable**. Le stochastique casse l'oracle argmax et exige des gates distributionnels : on ne mélange pas les deux natures de preuve. |
| **C2** | **Approche host-side pure** | Seule voie qui laisse le **graphe intact** (HLO byte-identique) : « je n'ai rien cassé » se démontre par un `diff`, pas par une campagne. |
| **C3** | **Module partagé** `sampling.zig`, prouvé E2B → confirmé 12B | Testable unitairement **sans GPU** ; itération rapide sur E2B ; la cible d'usage (12B `--repl`) reçoit un gate de confirmation. |
| **C4** | **Directives `:param` à chaud** dans `--repl` | Le host-side les rend gratuites (aucun recompile). Recharger le 12B coûte ~1 min 40 : un balayage complet de la penalty passe de N compiles à **une**. |

## 1. Faits établis par lecture du code (pas de mémoire)

**F1 — les logits complets sortent déjà du graphe.** `gemma4_g12auto.zig:830` retourne
`logits` en 3ᵉ position (ajoutée pour `--window-vacuity`). La boucle les reçoit et **ne les lit
pas** (`r_logits.deinit()` direct, `:1360`, commentaire « NON lue ici (pas de D2H) »).
Le chemin host est donc à un `toSliceAlloc` près : **aucune modification du graphe**.

**F2 — les logits sont post-softcap.** `engine.zig:850` :
`const logits = softcapPrec(prec.softcap, prec.compute, raw, cfg.geom.softcap);` puis `return`.
L'objet rapatrié est donc **exactement celui que HF passe à ses `LogitsProcessor`** — aucune
part du modèle n'est à réimplémenter côté host.

**F3 — formule HF exacte** (`transformers/generation/logits_process.py`,
`RepetitionPenaltyLogitsProcessor.__call__`, lue sur le venv de la VM GPU) :

```python
penalty_scores = torch.where(last_scores < 0, last_scores * penalty, last_scores / penalty)
scores = torch.where(token_mask, penalty_scores, last_scores)
```

- logit **négatif → ×penalty** ; **positif ou nul → ÷penalty** ;
- `token_mask` est **booléen** ⇒ la penalty s'applique **au plus une fois par token**, quel que
  soit le nombre d'occurrences ;
- tokens considérés = **`input_ids` entier, prompt inclus**, par défaut (docstring : « for
  decoder-only models like most LLMs, the considered tokens include the prompt by default ») ;
  `prompt_ignore_length` permet de n'en garder que le généré ;
- `__init__` **rejette** `penalty ≤ 0` (« has to be a strictly positive float »).

**F4 — le RNG device est écarté** (phase 2). `docs/ZML_UPSTREAM_AUDIT_2026-07-12.md:37` :
`sampleTokens`/`sampleTokensDynamic`/`Tensor.Rng` existent déjà dans le ZML vendored
(`adee932e`) — **aucun bump requis** — mais « le RNG device n'est pas garanti déterministe entre
backends/versions et son état global lie le bruit par lane à B ». Piège consigné :
dans `adee932e`, `sampleTokensDynamic` **multiplie** les logits par la température **au lieu de
diviser** (bugfix upstream `73498264` non appliqué ; fonction non utilisée ici).

## 2. Posture en tension avec la spec batching §3.5 — les deux restent valides

`docs/superpowers/specs/2026-07-12-batching-flash-attn-design.md` §3.5 a spécifié un sampling
**tronqué au top-K rapatrié**, explicitement qualifié d'« approximation de charge, pas une
implémentation de sampling de référence », avec « aucun oracle en mode sampling ».

Cette spec **n'annule pas** ce choix, elle en ouvre un second à côté :

| Voie | Cible | Statut |
|---|---|---|
| §3.5 batching — top-K tronqué, D2H ≈ K×8 o/lane/step | **mode charge, B>1** (le D2H complet × B lanes changerait le calcul) | reste valide, non touchée |
| Cette spec — logits complets, host, B=1 | **mode référence, oraclé** | nouvelle |

La troncature au top-K est **structurellement incapable** de porter une claim « == HF » : un
token de l'historique situé hors du top-K ne peut pas être pénalisé. C'est une limite de la
voie de charge, pas un défaut à corriger.

## 3. Design

### 3.1 Module `zml_runner/sampling.zig` (nouveau)

Responsabilité unique : **transformer un vecteur de logits en un token**. Aucune dépendance ZML
(f32 nus) ⇒ testable par `zig test` sans GPU, sans PJRT, sans poids.

```zig
pub const Params = struct {
    repetition_penalty: f32 = 1.0,   // 1.0 = désactivé (chemin greedy inchangé)
    ignore_prompt: bool = false,     // == prompt_ignore_length de HF
};

/// Applique la penalty IN-PLACE, au plus une fois par token distinct (F3).
/// `seen` : bitset {vocab} alloué UNE FOIS par l'appelant, remis à zéro par appel.
pub fn applyRepetitionPenalty(logits: []f32, hist: []const u32, penalty: f32, seen: *Bitset) void

/// argmax host — politique de tie-break documentée et gatée (RP2) contre le top1 de `topK`.
pub fn argmax(logits: []const f32) u32
```

**Deux points d'implémentation décident du succès de l'oracle :**

1. **Déduplication obligatoire.** Un `for (hist) |t| logits[t] = …` naïf applique la penalty
   autant de fois que le token apparaît — et diverge exactement dans le cas qui motive le
   chantier, la récitation. Le bitset (262 144 bits = 32 Ko, alloué une fois) rend le coût
   `O(|hist|)` et la sémantique identique à HF.
2. **Le signe commande l'opération** (F3). L'inversion des deux branches est le contre-test de
   non-vacuité RP5-(a).

### 3.2 Câblage dans la boucle — pourquoi la non-régression est gratuite

**Si `penalty == 1.0`, les logits ne sont pas lus du tout** : le chemin actuel (top-1 du `topK`
in-graph, ~48 octets D2H) reste strictement inchangé. Le chemin host ne s'arme que si la
penalty est active, **et seulement en phase génération** (`in_gen_phase`, `:1363` — pendant le
prefill le token suivant est imposé par le prompt).

Conséquence : « penalty=1.0 est bit-identique à l'existant » n'est pas une mesure, c'est **le
même code**. Et l'équivalence `argmax` host ↔ `topK` device devient un gate explicite (RP2)
au lieu d'une hypothèse enfouie — ce qui compte, vu le **piège 15** du repo sur la fragilité
des ties d'argmax.

Flux par step (génération, penalty active) :

```
exe.call → r_t5v, r_t5i, r_logits, caches
  penalty == 1.0 → tok = t5i[0]                     (chemin actuel, r_logits.deinit())
  sinon          → logits = r_logits.toSliceAlloc   (1 Mo, transfert DÉJÀ synchrone)
                   applyRepetitionPenalty(logits, hist, p, seen)
                   tok = argmax(logits)
hist.append(tok)
```

Le D2H s'ajoute à un transfert **déjà synchrone** (la boucle lit le top-5 à chaque step) : il ne
sérialise rien de nouveau. Son coût réel est **mesuré en RP6-(ii)**, il n'est supposé nulle part
dans cette spec.

### 3.3 Historique

`hist` = ids du prompt ++ générés (défaut HF), ou générés seuls sous `--ignore-prompt`.
En `--repl` V1 les prompts sont **indépendants** (cache zéros par prompt) : `hist` repart à zéro
à chaque prompt, sans mécanisme supplémentaire.

### 3.4 Directives `--repl`

`:penalty <f>` · `:ignore-prompt on|off` · `:params` · `:help`

Règle unique et testable : **une ligne commençant par `:` n'est jamais un prompt**. Limite
assumée et documentée (un prompt ne peut pas commencer par `:`) plutôt qu'un mécanisme
d'échappement dont personne n'a besoin (YAGNI). Une valeur invalide est rejetée **sans tuer la
session**.

### 3.5 Runners câblés

- `gemma4_gen_auto.zig` (E2B) — porte les gates lourds (itération rapide) ;
- `gemma4_g12auto.zig` (12B, hérité par `g12a4k`/`g12a8k`) — gate de confirmation + usage réel.

`engine.zig` : **0 octet**. `G12Step` : inchangé.

## 4. Gates — phase 1 (`RP`, déterministe)

| Gate | Contenu | Critère PASS |
|---|---|---|
| **RP0** | **Graphe intact** — dumps HLO des 2 runners, témoins pris **avant toute édition**, worktree homogène | `diff -rq` vide, md5 identiques |
| **RP1** | **Unitaire sans GPU** — `zig test sampling.zig` vs fixtures HF : penalty ∈ {0.8, 1.0, 1.15, 1.5}, historique **avec doublons**, logits **des deux signes** | bit-exact f32 (repli 1 ULP **documenté** si la promotion de type torch surprend) |
| **RP2** | **argmax host == top1 device** — penalty=1.0 + `--force-host-argmax`, E2B | 48/48 ids == témoin |
| **RP3** | **Non-régression** — penalty=1.0 chemin normal, E2B + 12B | ids bit-identiques au témoin |
| **RP4** | **Oracle HF exact, penalty active** — E2B, penalty ∈ {0.8, 1.15}, `scripts/49_gen_custom_oracle.py` étendu `--repetition-penalty` | **ids == HF** |
| **RP5** | **Non-vacuité** — 3 corruptions : (a) signes inversés, (b) dédup supprimée, (c) prompt inclus/exclu à tort | **chacune doit faire FAIL RP4** |
| **RP6** | **12B — effet et coût** — (i) balayage `:penalty` à chaud (une seule compile) ; (ii) **mesure** tok/s penalty=1.0 vs active | récitation levée ; coût **mesuré**, seuil pré-enregistré **10 %** |
| **RP7** | **Directives repl** — (a) `:penalty` ne produit **aucune** génération (**comptage**) ; (b) s'applique au prompt suivant, pas au précédent ; (c) `:params` fidèle ; (d) valeur invalide rejetée sans tuer la session | 4/4 |

**RP3 est quasi-tautologique** (même code) et assumé comme tel : il attrape les erreurs de
**câblage** (bitset alloué par step, `hist` qui grossit indûment), pas de calcul.

**RP7-(a) emploie le comptage** parce que c'est le comptage des générations — et non le code de
sortie — qui avait démasqué la sortie prématurée « exit 0 = faux succès » au gate R1 du mode
résident.

## 5. Gates — phase 2 (`SM`, stochastique) — spécifiés, livrés après la phase 1

| Gate | Contenu | Critère PASS |
|---|---|---|
| **SM0** | `--temperature 0` ou `--top-k 1` ⇒ greedy | ids **bit-identiques** au greedy (pont avec la phase 1) |
| **SM1** | **Distributionnel** — 10 000 tirages sur des logits **figés en fixture**, χ² vs distribution théorique host | χ² sous le seuil pré-enregistré |
| **SM2** | **Reproductibilité** — RNG **host** seedé par (seed, step), **zéro RNG device** (F4) | même seed ⇒ sortie identique |
| **SM3** | **Non-vacuité du RNG** — seed différent | sortie différente (sinon SM2 passait à vide) |

Paramètres phase 2 : `--temperature`, `--top-k`, `--top-p`, `--seed`, plus les directives
`:temp` / `:top-k` / `:top-p` / `:seed`.

## 6. Vigilances pré-enregistrées (écrites AVANT de mesurer)

1. **La penalty comprime les écarts.** Diviser par 1,15 des logits post-softcap bornés à ±30
   les rapproche de zéro et **peut créer des quasi-ties là où il n'y en avait pas**. La claim du
   projet est « argmax == HF », **jamais** « logits bit-identiques » : un mismatch d'ids sous
   penalty n'est pas automatiquement un bug.
2. **Procédure d'échec** (pattern A2) : au premier mismatch → top-5 du step fautif et marge
   top1−top2. **Le FAIL brut est publié d'abord** ; toute requalification vient ensuite et est
   datée. **Une 2ᵉ requalification du même type = STOP** : on diffe l'instrument avant d'écrire
   une ligne de plus (leçon des requalifications en cascade, 25 juil).
3. **Oracle en fp32**, jamais bf16 (même leçon : un instrument dégradé fabrique des
   requalifications).
4. **Témoins RP0 pris avant tout deploy**, sur un worktree homogène — l'état hybride
   « moteur édité + runner pré-bascule » est le piège exact qui avait mordu au gate M0.
5. **Seuil RP6-(ii) pré-enregistré à 10 %** : au-delà, on documente et on ouvre l'approche
   in-graph au backlog. Le seuil est fixé maintenant, pas après lecture du chiffre.

## 7. Gestion d'erreur

- `penalty ≤ 0` → **rejet au parse** (F3 : HF l'exige strictement positif).
- Bitset alloué **une fois** au démarrage, jamais par step.
- Token hors vocab dans `hist` : déjà gardé (`:1343`), assert conservé.
- **Un D2H échoué remonte une erreur** — **pas de repli silencieux** sur le top-5 : un tel repli
  rendrait RP4 incapable d'échouer (leçon « un contrôle qui ne peut pas réussir », et son
  miroir).

## 8. Hors périmètre

Multi-tour avec contexte accumulé (`--repl` V1 reste à prompts indépendants) · `no_repeat_ngram_size`
· sampling par lane en B>1 (couvert par la voie §3.5) · beam search · toute modification de
`engine.zig` ou du graphe.

## 9. Livrables

- `zml_runner/sampling.zig` + ses tests `zig test`
- câblage `gemma4_gen_auto.zig`, `gemma4_g12auto.zig`
- `scripts/49_gen_custom_oracle.py` étendu (`--repetition-penalty`, `--ignore-prompt`)
- fixtures de référence RP1 (safetensors, committées)
- `docs/SAMPLING_RESULTS.md` (résultats + chiffres mesurés)
- tags `gate/rp0-pass` … `gate/rp7-pass`, une PR
