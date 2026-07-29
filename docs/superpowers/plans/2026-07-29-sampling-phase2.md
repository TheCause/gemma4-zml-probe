# Sampling (phase 2) — Plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** faire du 12B un moteur d'inférence conforme — `top_k`, `top_p`, `temperature` et tirage
reproductible à seed fixée — sans toucher au graphe.

**Architecture:** tout est **host-side**. Le graphe sort déjà les logits complets (7 sorties,
logits en 3ᵉ, aujourd'hui non lus) : le chemin B les rapatrie et applique la chaîne de HF
(`penalty → suppress → ÷T → topK → topP → argmax|tirage`). Un module neuf `zml_runner/sampling.zig`
porte des fonctions **pures sur `f32` nus**, exerçables sans GPU. Le chemin greedy actuel (top-5,
`gencfg.select()`) reste **intact** et sert de référence au gate-pont.

**Tech Stack:** Zig `0.16.0-dev.2722` (API `std.Io`) · Bazel sur la VM GPU · transformers
**5.14.1** (venv `g12b`) · `ZML_REMOTE`/`ZML_DST` **obligatoires** en env · `--@zml//platforms:cuda=true`
sur **chaque** run GPU.

**Spec:** `docs/superpowers/specs/2026-07-29-sampling-penalty-design.md` (**rév. 3**).

**⚠ Périmètre — phase 2 SEULEMENT.** La repetition penalty (phase 1) a son propre plan :
`docs/superpowers/plans/2026-07-27-sampling-repetition-penalty.md` (rév. 3), régi par la spec
rév. 4 du 27 juil. **Les gates `SM0…SM3` de cette spec-là sont SUPERSÉDÉS** (spec rév. 3 §0) :
ne pas les exécuter.

**Gates de ce plan :** `S2-U` (fixtures, host-only) · `S2-PONT` (STOP) · `S2-D` · `S2-R` · `S2-G`,
plus la mesure publiée `M-COUT` (sans PASS/FAIL). Les 8 dettes du §7 de la spec sont à reporter
telles quelles au doc de résultats.

---

## Conventions d'exécution (à lire avant la Task 0)

- **Deploy** : `ZML_REMOTE=<user@host> ZML_DST=/data/rqz_workspace/zml/examples/rqz zml_runner/deploy_to_3090.sh`
  — exiger une sortie `rsync` **non vide**. Sans les variables, échec DNS silencieux.
- **⚠ Capture des logs** : **toujours** `> run.out 2> run.err`, **jamais** `> log 2>&1`. La
  redirection fusionnée entrelace stdout et stderr et **tronque** les lignes que les gates
  greppent (piège 24 de `DOCUMENTATION.md`, payé par GC2).
- **Aucun `zig test`** : `zig` est absent du PATH et `BUILD.bazel` ne charge que `zig_binary`
  (vérifié). Les tests unitaires passent par un drapeau `--selftest-*`, patron de
  `--selftest-gencfg` (`gemma4_g12auto.zig`, livré le 29 juil).
- **Un flag se touche à QUATRE endroits** dans `gemma4_g12auto.zig` : le bloc de commentaire CLI
  d'en-tête, la struct `Args`, la constante `usage`, la fonction `parseArgs`.
- **Patch ZML** : `grep -n "local patch rqz" /data/rqz_workspace/zml/pjrt/pjrt.zig` **avant** tout
  build.
- **VRAM** : `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv` avant
  chaque run GPU.
- **Longs runs** : `nohup … > out 2> err < /dev/null &` + fichier `.done`, puis
  `until grep -q "^DONE" .done; do sleep 30; done`.

---

## Task 0 : témoins et prérequis — aucune édition

**Files:** aucun édit.

- [ ] **0.1** Worktree propre, noter le HEAD : `git status --short && git rev-parse --short HEAD`.
- [ ] **0.2** Vérifier que la phase 1 est **faite ou explicitement différée**. Ce plan n'en dépend
      pas techniquement (`penalty = 1.0` est le défaut neutre), mais le doc de résultats doit dire
      lequel des deux chantiers a livré quoi.
- [ ] **0.3** Dump HLO **témoin** du binaire actuel, pour `S2-G` :
      `XLA_FLAGS=--xla_dump_to=/tmp/s2_hlo_before` sur un run 48 tokens.
      **Consigner** : md5 de `module_0001.zml.before_optimizations.txt`, **nombre de fichiers**,
      **volume total**. *(Attendu, d'après GC0 : 510 fichiers, 1 905 860 octets, md5
      `297679847aa04b719942d75d093adf2b`.)*
- [ ] **0.4** Figer `RUN_ARGS` dans un tableau bash **une fois pour toutes** et le réutiliser
      partout. Un argument qui bouge entre AVANT et APRÈS casse l'A/B à un seul facteur.
- [ ] **0.5** Commit : `chore(s2): témoins HLO et RUN_ARGS figés`.

---

## Task 1 : `sampling.zig` — les fonctions pures, sans GPU

**Files:** Create `zml_runner/sampling.zig` · Modify `zml_runner/BUILD.bazel` (srcs de **3** cibles :
`gemma4_g12auto`, `gemma4_g12a4k`, `gemma4_g12a8k`).

- [ ] **Step 1.1 — écrire le selftest D'ABORD** (il doit échouer à la compilation).

Dans `gemma4_g12auto.zig`, ajouter le drapeau `--selftest-sampling <fixture>` aux **quatre**
endroits, puis l'early-return **sur le même patron que `--selftest-gencfg`** (avant tokenizer,
VRAM et Platform) :

```zig
// === S2-U : selftest des warpers, host-only. Même patron d'early-return que --selftest-gencfg :
// c'est ce qui rend le gate exécutable sans GPU, et ce qui interdit d'y mesurer quoi que ce soit
// qui dépende du modèle. ===
if (args.selftest_sampling) |fixture_path| {
    try selftestSampling(allocator, io, fixture_path);
    return;
}
```

- [ ] **Step 1.2 — builder, et vérifier que ça ÉCHOUE**

```bash
ZML_REMOTE=… ZML_DST=… zml_runner/deploy_to_3090.sh
ssh "$ZML_REMOTE" 'cd /data/rqz_workspace/zml && ./bazel.sh build //examples/rqz:gemma4_g12auto 2>&1 | tail -20'
```

Attendu : `error: unable to load 'sampling.zig': FileNotFound`. **C'est le rouge.** S'il compile,
c'est que l'import n'a pas été écrit — corriger avant de continuer.

- [ ] **Step 1.3 — écrire `zml_runner/sampling.zig`**

Contrat exact (spec §5). Chaque fonction est **pure** et opère in-place sur `[]f32`.

```zig
//! Warpers de sampling — HOST-SIDE, hors du graphe. Fonctions pures sur f32 nus : aucune
//! dépendance ZML, donc exerçables par --selftest-sampling sans GPU, sans PJRT, sans poids.
//!
//! ⚠ CHAQUE FONCTION REPRODUIT UN WARPER HF PRÉCIS, LU À LA SOURCE (transformers 5.14.1).
//! Ne pas « simplifier » une formule : les écarts vivent aux BORDS, et deux d'entre eux
//! produisent des ensembles de survivants DISJOINTS de ceux de HF (spec §2, F9/F10).

const std = @import("std");
const gencfg = @import("gencfg.zig");

pub const FILTER: f32 = -std.math.inf(f32); // HF: filter_value = -inf (F11)

/// Température — DIVISION, jamais × (1/T) : l'équivalence mathématique casse la bit-exactitude.
/// ⚠ HF n'instancie PAS ce warper quand T == 1.0 (F8c) : l'appelant doit faire de même.
pub fn applyTemperature(logits: []f32, t: f32) void {
    for (logits) |*x| x.* = x.* / t;
}

/// top-k — F9 : l'élimination est STRICTEMENT sous le k-ième, donc les EX ÆQUO DU k-ième
/// SURVIVENT. `top_k` ne borne donc PAS le nombre de survivants (mesuré : 64 logits égaux,
/// k=8 → 64 survivants). Une implémentation « garder les k premiers du tri » diverge dès la
/// première égalité.
/// `k_eff = min(max(k, min_keep), len)` reproduit __init__ + le "safety check" de __call__.
pub fn applyTopK(logits: []f32, k: u32, min_keep: u32) void {
    if (k == 0) return; // convention HF : désactivé
    const k_eff = @min(@max(k, min_keep), logits.len);
    if (k_eff >= logits.len) return;
    // seuil = valeur du k_eff-ième plus grand ; sélection partielle, JAMAIS un tri complet
    // (spec F16 : un tri complet coûte 177,95 % d'un step).
    const kth = kthLargest(logits, k_eff);
    for (logits) |*x| {
        if (x.* < kth) x.* = FILTER; // STRICT — c'est tout le sujet
    }
}

/// top-p — F10 : HF trie en ASCENDANT, cumsum des softmax, retire avec `cum <= 1 - top_p`,
/// et `min_tokens_to_keep` protège la QUEUE du tri ascendant (= les plus probables).
/// ⚠ Ce n'est PAS la formulation habituelle (tri descendant, garder tant que cum < p) : les
/// deux diffèrent aux bords, et sur 8 logits égaux avec top_p=0,25 elles rendent des ensembles
/// DISJOINTS ({6,7} contre {0,1}).
/// ⚠ Le masque revient dans l'espace vocabulaire par un scatter dont le tenseur de BASE est le
/// masque trié lui-même : un scatter naïf produit un masque juste dans l'espace TRIÉ et faux
/// dans l'espace vocabulaire.
pub fn applyTopP(allocator: std.mem.Allocator, logits: []f32, p: f32, min_keep: u32) !void {
    if (p >= 1.0) return;
    // … tri ascendant des indices, softmax, cumsum, masque `cum <= 1-p`,
    //   puis `mask[len - min_keep ..] = false`, puis application par indice d'origine.
}

/// Tirage multinomial sur les logits FILTRÉS. RNG **host** seedé par (seed, index de token DANS
/// LE PROMPT COURANT) — jamais le RNG device (non garanti déterministe entre backends).
/// ⚠ Invariant à asserter par l'appelant : le token rendu vérifie logits[t] > -inf.
pub fn sample(logits: []const f32, rng: *std.Random.DefaultPrng) u32 { … }

/// argmax host, tie-break explicite : premier indice gagnant.
pub fn argmax(logits: []const f32) u32 { … }
```

- [ ] **Step 1.4 — ajouter `sampling.zig` aux srcs des 3 cibles** dans `zml_runner/BUILD.bazel`
      (`gemma4_g12auto`, `gemma4_g12a4k`, `gemma4_g12a8k`). Tout fichier `@import`-é doit être
      exposé au sandbox Bazel.

- [ ] **Step 1.5 — builder : doit compiler** (le selftest échouera encore faute de fixture).

- [ ] **Step 1.6 — commit** : `feat(s2): sampling.zig — warpers purs, host-side`.

---

## Task 2 : les fixtures — produites par les VRAIS warpers HF

**Files:** Create `scripts/72_sampling_fixture.py` · Create `fixtures/s2_cases.safetensors` + son
`.manifest.json`.

> **Pourquoi un producteur Python et pas des cas écrits à la main** : retranscrire une formule
> ferait comparer **deux transcriptions du même auteur** — une inversion de branche commise des
> deux côtés passerait le gate. Le producteur instancie `TemperatureLogitsWarper`,
> `TopKLogitsWarper` et `TopPLogitsWarper` et publie **leur** sortie.

- [ ] **Step 2.1 — écrire le producteur.** Format safetensors : `logits_in` `{N,V}` f32,
      `mask_expected` `{N,V}` u8 (1 = survivant), plus un sidecar JSON portant, par cas : `name`,
      `warper`, `params`, et les **compteurs d'antécédent**.

- [ ] **Step 2.2 — les cas de bord OBLIGATOIRES** (spec C2). Un gate qui ne les contient pas rend
      100 % sans rien prouver :

| # | Cas | Ce qu'il discrimine |
|---|---|---|
| a | 8 logits **égaux**, `top_p = 0,25` | HF garde **{6,7}**, le naïf descendant **{0,1}** — **ensembles disjoints** |
| b | `[10,0,0,0,0]`, `top_p = 0,1`, `min_keep ∈ {1,2,3}` | `min_keep` **mordant**, et il repêche la **queue du tri ascendant** |
| c | `[3,2,1,1,1]`, `top_k = 3` | **5** survivants, pas 3 (ex æquo du k-ième) |
| d | **`V = 262 144`**, au moins un cas | **`torch.sort` bascule entre n=128 et n=129** : une fixture sous 129 exerce un AUTRE chemin que la production |
| e | `T = 0,7` puis `top_k` | la température **change** le nombre de survivants (F8a/F8b) |

- [ ] **Step 2.3 — le producteur ASSERTE sa propre non-vacuité** et échoue si un compteur vaut 0 :
      au moins un cas où `top_k` laisse **plus** de k survivants ; au moins un cas où HF et la
      formulation naïve **diffèrent** ; au moins un cas à `V = 262 144`.

- [ ] **Step 2.4 — produire la fixture** dans le venv `g12b`, puis **contrôler les md5 des deux
      côtés** (la VM n'est pas un dépôt git ; sa copie d'un script a **déjà** été périmée une fois).

- [ ] **Step 2.5 — commit** : `feat(s2): producteur de fixtures depuis les vrais warpers HF`.

---

## Task 3 : `S2-U` — le gate host-only

**Files:** Modify `zml_runner/gemma4_g12auto.zig` (fonction `selftestSampling`).

- [ ] **Step 3.1 — écrire `selftestSampling`** sur le patron de `selftestGencfg` : lire la fixture,
      lire le sidecar, rejouer chaque cas, comparer.

- [ ] **Step 3.2 — comparer par CLASSE D'ÉQUIVALENCE, pas par ensemble d'indices** (spec F14) :
      `torch.sort` n'étant pas stable au-delà de n=128, l'identité exacte des survivants sur ex æquo
      **n'est pas contractuelle**. Comparer le **multiset trié des logits survivants** et la **masse
      de probabilité**. *(Comparer les indices ferait échouer le gate à tort sur les cas égalitaires
      — et c'est précisément la classe de cas qu'on veut exercer.)*

- [ ] **Step 3.3 — asserter la non-vacuité, patron GC1** : publier les compteurs et **FAIL si l'un
      vaut 0**. Un cas dont l'antécédent est vide est **inexécutable**, pas PASS.

- [ ] **Step 3.4 — `S2-U`** : lancer sur la VM (host-only, **aucune compile GPU**).

```bash
ssh "$ZML_REMOTE" 'cd /data/rqz_workspace/zml && \
  ./bazel-bin/examples/rqz/gemma4_g12auto /dev/null /dev/null \
  --selftest-sampling /data/gemma4-zml-probe/s2/s2_cases.safetensors' > /tmp/s2u.out 2> /tmp/s2u.err
```

PASS = **100 %** des cas **et** tous les compteurs non nuls. **FAIL ⇒ STOP.**

- [ ] **Step 3.5 — contre-preuve du gate** : muter volontairement `applyTopK` (`<` → `<=`) et
      vérifier que `S2-U` **ÉCHOUE**. Puis muter `applyTopP` (tri descendant) et vérifier qu'il
      échoue **aussi**. Restaurer. *Un test vert qui reste vert quand on retire ce qu'il teste ne
      prouve rien.*

- [ ] **Step 3.6 — commit + tag** `gate/s2-u-pass` avec les chiffres.

---

## Task 4 : le chemin B et `S2-PONT`

**Files:** Modify `zml_runner/gemma4_g12auto.zig` (flags, lecture des logits, sélection).

- [ ] **Step 4.1 — les six flags**, chacun aux **quatre** endroits, avec les gardes **en
      ACCEPTATION** (spec §5). ⚠ `p <= 0 → rejet` **laisse passer `NaN`** : toute comparaison avec
      `NaN` est fausse, donc `NaN` échoue toute acceptation et **passe** tout rejet.

```zig
if (!(t >= T_MIN and std.math.isFinite(t))) return error.InvalidTemperature;
```

`--temperature 0` ⇒ **rejet, comme HF**, message renvoyant vers `--top-k 1`.
`T_MIN = 1e-30` ⇒ **divergence délibérée et déclarée** : HF accepte `1e-45` et produit des `NaN`.

- [ ] **Step 4.2 — rapatrier le vecteur complet UNIQUEMENT si une brique est armée.** Si tout est
      neutre, le chemin A (top-5, ~48 octets) reste **strictement inchangé**.

- [ ] **Step 4.3 — le bloc de sélection**, dans l'ordre de F8 :
      `penalty → suppress → ÷T → topK → topP → argmax|sample`. `applySuppression` **délègue à
      `gencfg.isSuppressed`** — pas de seconde implémentation.

- [ ] **Step 4.4 — instrumenter `S2-PONT`** : à chaque step où le chemin B tourne, calculer **aussi**
      le token qu'aurait rendu `gencfg.select()` sur le top-5, et comparer. Les deux sorties du
      graphe sont disponibles **simultanément** — c'est ce qui rend le gate insensible au
      non-déterminisme **par construction**.

Compteurs à publier : `n_steps_compared` · `n_disagree` · `n_exact_top_ties` · `n_suppress_hits`.

- [ ] **Step 4.5 — `S2-PONT`** : `--repetition-penalty 1.0 --top-k 1`, sur un témoin **où la
      suppression MORD** (le témoin 200 teacher-forcé mord @57 — les témoins 48/124 ont
      `n_suppress_hits = 0`, mesuré par GC2 : les y poser validerait le chemin A contre lui-même).

PASS = `n_disagree == 0` **hors égalités exactes** · `n_steps_compared == n_generated` ·
`n_suppress_hits ≥ 1` · `n_exact_top_ties` **publié**. Un désaccord **sur une égalité exacte** est
un **verdict distinct**, pas un FAIL. **FAIL ⇒ STOP.**

- [ ] **Step 4.6 — non-régression** : rejouer les témoins 48 et 124 (celui-ci **borné à ses 110
      premiers ids**, spec F17) en régime neutre ⇒ ids bit-identiques.

- [ ] **Step 4.7 — commit + tag** `gate/s2-pont-pass`.

---

## Task 5 : `S2-D` et `S2-R` — le stochastique

**Files:** Modify `scripts/72_sampling_fixture.py` (bras distributionnel) · Modify le runner
(compteurs).

- [ ] **Step 5.1 — `S2-D`** : 10 000 tirages sur des **logits figés en fixture**, `top_k = 10`
      **ids distincts** (pas des bins agrégés : une permutation d'ids **dans** un bin serait
      invisible). Distribution théorique par implémentation **indépendante** (torch).
      PASS = χ² sous α = 0,01.

- [ ] **Step 5.2 — non-vacuité de `S2-D`, obligatoire** : injecter un biais **half-split**
      (+10 % sur k/2 catégories, −10 % sur l'autre moitié) ⇒ doit **FAIL**.
      *Dérivation : λ = n·b², **indépendant de k** ⇒ puissance ≥ 99,98 %. Une injection
      mono-catégorie à 5 % donnerait χ² = 2,78 contre 21,67 critique — elle ne mordrait pas.*
      **Règle de re-run écrite d'avance** : un seul re-run, **seed pré-déclarée**, deux échecs = FAIL.

- [ ] **Step 5.3 — `S2-R`** : même seed deux fois ⇒ **ligne `generated` identique**.
      ⚠ Comparer **cette ligne seule** : l'écho `seed=<n>` rendrait N sorties trivialement
      distinctes et le gate ne pourrait plus échouer.

- [ ] **Step 5.4 — non-vacuité du RNG** : N seeds ⇒ ≥ k sorties distinctes, **k et N calculés
      depuis la fixture AVANT** la mesure (sur une distribution piquée, deux seeds donnent
      légitimement la même sortie). Plus : `n_draws > 0` et l'assertion dure
      **`logits[token_tiré] > -inf`**, comptée à chaque step.

- [ ] **Step 5.5 — commits + tags** `gate/s2-d-pass`, `gate/s2-r-pass`.

---

## Task 6 : `S2-G` et la mesure `M-COUT`

- [ ] **Step 6.1 — `S2-G`** : re-dumper le HLO et comparer au témoin de la Task 0.
      ⚠ **Fraîcheur obligatoire** : vider le répertoire de dump, vérifier le **mtime postérieur à
      l'édition**, publier **nombre de fichiers ET volume** des deux côtés. Sans cela le gate
      compare le témoin à lui-même et ne peut plus échouer.

- [ ] **Step 6.2 — `M-COUT` — c'est une MESURE, pas un gate.** Chronomètre **in-process** autour du
      bloc {D2H + sample}, bras **alternés A/B dans le même processus**, médiane par step sur 48
      steps. **Publier** en regard de la table F16 (D2H+partial_sort ≈ **1,57 %** d'un step ; softmax
      plein vocab **10,56 %** ; tri complet **177,95 %**).
      **Aucun PASS/FAIL** : le surcoût visé est **sous le plancher de résolution** du protocole de
      débit du projet (bruit inter-compiles 2-16 %).

- [ ] **Step 6.3 — commit + tag** `gate/s2-g-pass`.

---

## Task 7 : documentation et clôture

- [ ] **Step 7.1 — `docs/SAMPLING_RESULTS.md`** : verdicts + chiffres + section **« Périmètre de la
      claim »** + **les 8 dettes du §7 de la spec, recopiées telles quelles**. En particulier **D1** :
      `applyTopP` n'a **aucune couverture GPU** — sa seule couverture est la fixture. Une couverture
      absente doit rester **déclarée** absente.

- [ ] **Step 7.2 — écrire ce que le chantier NE prouve pas** : `applyTemperature` non exercé de bout
      en bout (D2) · interdit d'allocation sans gate porteur (D3) · équivalence de l'arrêt, **aggravée**
      par la penalty (D4) · E2B (D6) · custody de F16 (D7) · tie-break host vs in-graph (D8).

- [ ] **Step 7.3 — GC11 doit rester vert** : `./scripts/gc11_claim_scope.sh` **et** `--self-test`.
      Si `SAMPLING_RESULTS.md` énonce « == HF », il doit porter le marqueur de portée.

- [ ] **Step 7.4 — mettre à jour** `PLANNING.md`, `README.md` (section CLI : les 6 nouveaux flags),
      `docs/DOCUMENTATION.md` (§2.3, §5.1, §5.2, §8 pièges, §9 limites — **le sampling n'est plus
      une limite**).

- [ ] **Step 7.5 — anonymisation** : grep durci **avec contre-preuve** (il doit détecter un canary
      injecté) sur tous les fichiers touchés, logs rapatriés et manifests compris.

- [ ] **Step 7.6 — PR** vers `main` avec les 5 tags de gates.

---

## Ce qui ferait ARRÊTER ce plan

- **`S2-U` FAIL** ⇒ un warper ne reproduit pas HF : tout ce qui est en aval mesure autre chose.
- **`S2-PONT` FAIL** (hors égalité exacte) ⇒ les deux chemins de sélection divergent : la
  non-régression du greedy n'est plus garantie.
- **`S2-D` FAIL** deux fois ⇒ le tirage ne suit pas la distribution.
- **`S2-G` FAIL** ⇒ quelque chose a fui dans le graphe — probablement un RNG device.
