# Plan d'implémentation — Dettes restantes : orientation + relance de la repetition penalty

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal :** Trier les 9 dettes restantes du repo (4 décisions à confirmer, 4 chantiers à
cadrer, 1 chantier mûr) et, sur GO de Régis, exécuter le seul chantier mûr : la **phase 1
repetition penalty**, ré-instruite contre l'état réel de `main` au 10 août (le plan du
27 juil prédate 5 chantiers mergés et n'est plus exécutable tel quel).

**Architecture :** Aucun changement de graphe (gate RP0 = md5 HLO identique). La penalty
s'insère en **tête de la chaîne host-side existante** (`gemma4_g12auto.zig:2893` porte le
commentaire « La penalty appartient à la PHASE 1 : absente ici ») : une fonction pure dans
`sampling.zig` + un historique pré-alloué dans `SamplingCfg`, zéro allocation par step
(interdit D10, compteur toujours actif).

**Tech stack :** Zig 0.16-dev (build via `zml_runner/build_3090.sh` — JAMAIS `-c opt` seul,
cf `docs/MODE_BUILD_AUDIT.md`), Bazel + ZML sur la VM 3090 (`ssh ia@192.168.1.163`,
workspace `/data/rqz_workspace/zml`, repo `/data/gemma4-zml-probe/`), Python venv
`/data/venvs/gemma4-probe` (transformers 5.14.1), oracle fp32 sur M4.

**Spec de référence :** `docs/superpowers/specs/2026-07-27-sampling-repetition-penalty-design.md`
(**rév. 4**, revue 2 tours) — elle reste la source des gates RP0-RP7 et de leurs critères.
Ce plan la RÉFÉRENCE et liste ses amendements de ré-instruction (§ « Écarts ») ; il ne la
réécrit pas (leçon : prolonger une spec revue = la référencer).

---

## Contexte pour une session neuve (lis ceci d'abord)

Ce repo est un portage ZML bit-exact de Gemma 4 (E2B + 12B) prouvé gate par gate (~80 gates
taggés). Règles non négociables :

1. **Le graphe ne bouge pas** : témoin HLO pré-code, md5 `before_optimizations` attendu
   **`297679847aa04b719942d75d093adf2b`** (stable sur 5 chantiers : GC0 27 juil, S2-G,
   DC0 9 août, G-D0 10 août). Tout autre md5 = STOP et instruire.
2. **Un refus/une claim se VOIT échouer** : on exerce le cas, on archive dans `docs/evidence/`.
3. **Mode de build prouvé** : tout build passe par `zml_runner/build_3090.sh` (source unique
   des DEUX flags de mode) et chaque gate greppe la bannière `BUILD: mode=ReleaseFast` au log.
4. **Interdit D10 gardé** : `ALLOC-LOOP: alloc=0` par step — le compteur est toujours actif,
   tout run le re-vérifie gratuitement. Le code de ce plan ne doit RIEN allouer dans la boucle.
5. **VRAM** : `nvidia-smi --query-compute-apps` avant tout run GPU ; un Ollama résident peut
   occuper la carte (`ollama stop <modèle>`, réversible).
6. **Anonymisation** : grep avant push
   (`git grep -nE 'Users/regis|macmini|192\.168' -- ':!docs/superpowers'`).
7. **Branche + PR** : jamais de commit direct sur `main`. Branche **`penalty-phase1`**,
   merge `--no-ff` sur GO Régis.
8. **Runs distants longs** : `nohup … ; echo DONE rc=$?` + attente de `DONE` — un run libre
   n'émet jamais `PASS` (détail : plan du 27 juil, § « Conventions d'exécution »).

Sources de vérité à lire avant d'exécuter : `docs/SAMPLING_RESULTS.md` (chaîne actuelle,
gates S2 et G-D, §7), `zml_runner/sampling.zig` (warpers existants), `docs/D10_RESULTS.md`
(interdit alloc), `docs/KVDUMP_RESULTS.md` §6 (dettes K), la spec rév. 4 du 27 juil.

**Invocations réelles :**

```bash
# Build (depuis M1, à la racine du repo) — source unique des 2 flags de mode :
ZML_REMOTE=ia@192.168.1.163 ZML_WS=/data/rqz_workspace/zml ./zml_runner/build_3090.sh
# (cibles par défaut incluent gemma4_g12auto ; variante via TARGETS=..., pas d'argument positionnel)

# Runner (sur la VM) :
cd /data/rqz_workspace/zml
B1=./bazel-bin/examples/rqz/gemma4_g12auto     # variante 1280
W=/data/gemma4-zml-probe/weights_12b
$B1 $W/model.safetensors $W/tokenizer.json --prompt "..." --max-tokens N <flags...>

# Oracle fp32 (sur M4) :
#   venv ~/ml-venvs/g12b, export dq ~/ml-data/weights_12b_dq/
python3 scripts/69_u8_gen_oracle.py --weights <export dq> --prompt "..." --compute-fp32 --out <fixture>
```

## Inventaire des 9 dettes restantes (état au 10 août 2026, post-PR #21)

### A — Décisions ACTÉES, à CONFIRMER en Task 0 (recommandation : maintenir les 4)

| Dette | Décision en vigueur | Ce que la rouvrir coûterait |
|---|---|---|
| **K1** (PRNG non sérialisé → `--dump-cache` + `--seed` = refus, `gemma4_g12auto.zig:1769`) | GO Régis 9 août | Sérialiser Xoshiro256 ajouterait une claim d'équivalence stochastique qu'aucun gate simple ne prouve |
| **K2 / DA-4** (8k non exercé) | GO Régis 9 août (gates 1280+4k seulement) | Un run 8k complet + gates dédiés (~1 session GPU) |
| **K6** (pas de compression du dump) | Décision de design | « On ne dégrade pas un état exact pour du disque » |
| **DA-6** (`std.Io.Threaded` sur gpa non wrappé) | Structurelle, contrepartie documentée (`D10_RESULTS.md` §6) | Refonte du wrapper d'allocateur, gain nul prouvé (borné par AL-RSS) |

### B — Chantier exécuté par CE plan, sur GO explicite (Tasks 1-7)

| Dette | Quoi | Coût estimé |
|---|---|---|
| **Phase 1 repetition penalty** (SUSPENDUE depuis le 27 juil) | Penalty HF-compatible host-side, `ids == HF` prouvé, graphe intact — spec rév. 4, gates RP0-RP7 | ~1 session : 2 builds, ~6 runs GPU courts, 3 runs oracle M4 (sans GPU) |

### C — Chantiers À CADRER (fiches en Task 8 — AUCUN code ici)

| Dette | Pourquoi pas maintenant |
|---|---|
| **K3 / E2B** (logits hors graphe `gen_auto.zig:753`, pas de `suppress_tokens`) | Exige de modifier le graphe E2B (7ᵉ sortie) → spec dédiée + re-témoins |
| **K4** (`--repl` + dump/load) | Sémantique multi-tour inexistante — chantier « résident à reprise » |
| **K5** (reprise avec prompt neuf) | Prefill partiel : spec propre (positions, masques, RoPE à re-dériver) |
| **Triton paged attention** | Bump ZML + refonte cache YOCO paginé (audit `docs/ZML_UPSTREAM_AUDIT_2026-07-12.md`) — le plus gros chantier restant |
| **Options B/C anonymisation** ; **10 branches distantes mergées** | Décisions/actions manuelles Régis, inchangées depuis le 26 juil |

## Écarts de ré-instruction — pourquoi le plan du 27 juil n'est plus exécutable tel quel

Chaque point ci-dessous est un fait vérifié sur `main` au 10 août ; les Tasks intègrent déjà
ces corrections. **Ne pas exécuter le plan du 27 juil directement.**

1. **RP-1 (oracle décode fp32) est DÉJÀ SOLDÉE.** La garde « `--compute-fp32` n'existe qu'en
   `--teacher-force` » a été **levée** par le chantier `generation_config`
   (`69_u8_gen_oracle.py:547-554`, prérequis GC8), l'assert dtype est conditionnel (`:238-240`).
   La Task 1 du vieux plan devient une simple vérification.
2. **`sampling.zig` EXISTE** (347 lignes, phase 2) : `applyRepetitionPenalty` s'y AJOUTE. Le
   struct `Params` du vieux plan est SUPERSÉDÉ par `SamplingCfg` (`sampling.zig:114`), déjà
   propagé **par pointeur** aux 3 sites d'appel de `generateOnce` — l'échafaudage de
   propagation du vieux plan (Task 4 Step 2) est déjà en place.
3. **Le chemin B existe** (`gemma4_g12auto.zig:2875-2960`) : D2H direct dans `scfg.work`
   pré-alloué (D10/C2, 0 alloc), chaîne `Suppress → Temperature → TopK → TopP → sélection`.
   La penalty = 1 insertion en tête (`:2892-2894`) + un historique. Le gros de la Task 4 du
   vieux plan (FBA, `toSliceAlloc`, garde de forme) est SUPERSÉDÉ.
4. **L'interdit D10 est désormais GARDÉ** (le vieux plan le prédate) : ses
   `hist.append(allocator, …)` et `divergences.append(…)` PAR STEP déclencheraient
   `ALLOC-LOOP: alloc≠0`. → historique et compteurs **pré-alloués une fois** (pattern C5).
5. **Script `71_penalty_vectors.py` COLLISIONNE** avec `71_gc1_fixture.py` (la collision a
   déjà coûté une fois, script 74). Premier numéro libre VÉRIFIÉ : **76**.
6. **Les commandes de build du vieux plan sont INVALIDES** : elles prédatent
   `MODE_BUILD_AUDIT` et omettent `--@rules_zig//zig/settings:mode=release_fast` — M1 (coût)
   y aurait re-mesuré du debug. → `build_3090.sh` partout + bannière greppée.
7. **RP0 se juge au md5 `before_optimizations`** (standard actuel DC0/G-D0), pas au
   `diff -rq` des 510 fichiers du vieux plan (dont 498 de codegen bruités).
8. **Sous penalty armée, les désaccords S2-PONT sont ATTENDUS** (chemin B ≠ argmax nu) — ne
   pas les lire comme FAIL (leçon « ne pas mal lire les compteurs d'un run ARMÉ »,
   `SAMPLING_RESULTS.md` §2). La non-vacuité passe par un compteur dédié (Task 4).
9. **Dump/restore : la penalty est compatible, mais PAS gratuitement.** Le refus K1 porte
   sur `--seed` seul (`:1769`) et `ids_fed` est bien réinjecté dans `ids` par `--load-cache`
   (`:1969`) — MAIS la boucle de reprise entre **directement en phase de génération**
   (`:2802-2809`) : les tokens de `ids_fed` ne sont jamais re-feedés. L'historique doit donc
   être **seedé depuis `ids.items` AVANT la boucle** (Task 4 Step 1), sinon la penalty
   post-restore ignore tout le contexte repris. La Task 4 Step 5 le vérifie à l'exécution.
10. **RP7 (récitation) est vraisemblablement VACUE** : le symptôme n'a jamais été reproduit
    (3 témoins greedy sans boucle, `PLANNING.md` finding 27 juil ; dette D5). Arbitrage en
    Task 0 — reco : requalifier en mesure publiée sans PASS/FAIL.
11. **D4 est AGGRAVÉE par la penalty** (pénaliser un id EOS modifie l'arrêt,
    `SAMPLING_RESULTS.md` §5) : limitation à écrire dans les résultats, pas un gate ici.

---

### Task 0 : État des lieux + décisions Régis

**Files :** aucun (lecture seule).

- [ ] **Step 1 : Vérifier l'état du repo**

Run : `git -C ~/dev/gemma4-zml-probe status --porcelain && git log --oneline -3`
Expected : arbre propre, HEAD = `0461585` ou plus récent sur `main`.
Si l'arbre n'est pas propre : STOP, montrer à Régis (un travail non committé est invisible).

- [ ] **Step 2 : Poser les décisions à Régis, AVANT tout code**

1. **D1 — Confirmer les 4 décisions de la table A** (K1, K2/DA-4, K6, DA-6). Reco :
   **maintenir les 4**. En rouvrir une = chantier séparé, pas celui-ci.
2. **D2 — GO/NO-GO phase 1 penalty** (Tasks 1-7). Coût : ~1 session, 2 builds, ~6 runs GPU
   courts, 3 runs oracle M4.
3. **D3 — Périmètre `--repl`** (directives `:penalty`, gates RP5/RP6). Reco : **GO** (la
   spec §3.4 les exige et RP5 est le seul gate du reset par prompt) ; le NO-GO les reporte
   en dette écrite.
4. **D4 — RP7 (récitation)** : reco — **requalifier** en mesure publiée (métrique n-gramme
   sur témoin long, sans PASS/FAIL), le symptôme n'ayant jamais été reproduit. L'alternative
   (chercher un prompt qui récite) est ouverte mais non bornée.

- [ ] **Step 3 : Créer la branche**

```bash
git -C ~/dev/gemma4-zml-probe switch -c penalty-phase1
```

---

### Task 1 : Vérification RP-1 + témoins AVANT toute édition

**Files :** aucun fichier modifié — artefacts hors arbre + `docs/evidence/penalty/` (créer).

- [ ] **Step 1 : RP-1 — vérifier que l'oracle décode fp32 est opérationnel** (déjà soldé
  par GC8, on le VOIT au lieu de le croire)

Run : `sed -n '545,556p' scripts/69_u8_gen_oracle.py`
Expected : la garde levée + le `print` « --compute-fp32 en mode DÉCODE : instrument fp32 ».
Si la garde est revenue (régression) : STOP, c'est un finding.

- [ ] **Step 2 : Déployer l'état non modifié et capturer le témoin HLO**

```bash
# depuis M1 :
export ZML_REMOTE=ia@192.168.1.163 ZML_DST=/data/rqz_workspace/zml/examples/rqz
./zml_runner/deploy_to_3090.sh          # ⚠ les défauts sont des placeholders qui échouent
ZML_REMOTE=ia@192.168.1.163 ZML_WS=/data/rqz_workspace/zml ./zml_runner/build_3090.sh
# sur la VM :
XLA_FLAGS="--xla_dump_to=/data/gemma4-zml-probe/rp_hlo_witness" \
  $B1 $W/model.safetensors $W/tokenizer.json --prompt "witness" --max-tokens 4
md5sum /data/gemma4-zml-probe/rp_hlo_witness/*before_optimizations.txt
```

Expected : md5 = **`297679847aa04b719942d75d093adf2b`**. L'archiver dans
`docs/evidence/penalty/hlo_witness.md5`. Tout autre md5 : STOP (le graphe a bougé avant nous).

- [ ] **Step 3 : Témoin d'ids LONG, penalty neutre** (le prompt canonique fait EOT au 2ᵉ
  token — deux ids n'exercent rien)

```bash
$B1 $W/model.safetensors $W/tokenizer.json \
  --prompt "Tell me the story of the number zero, from its invention to modern mathematics." \
  --max-tokens 200 --out-ids /data/gemma4-zml-probe/rp_witness_long.safetensors
```

Figer ce prompt et `--max-tokens 200` comme **RUN_ARGS de référence** (à réutiliser à
l'identique en RP2). Vérifier au log : `BUILD: mode=ReleaseFast`, `ALLOC-LOOP: alloc=0`,
et **publier le nombre d'ids générés : `n < 50` → changer de prompt** (en mode libre l'EOT
peut couper tôt — un témoin de 5 ids ferait passer RP2 sur du quasi-vide, leçon vacuité de
l'antécédent). Rapatrier le témoin sur M1 (hors arbre, `logs/`).

- [ ] **Step 4 : Commit des preuves**

```bash
git add docs/evidence/penalty/hlo_witness.md5
git commit -m "penalty(témoins) : RP-1 vérifié soldé (GC8), md5 HLO témoin archivé, témoin ids 200 tokens figé"
```

---

### Task 2 : Fixture penalty — le VRAI processor HF (script 76)

**Files :**
- Create : `scripts/76_penalty_vectors.py`
- Create : `fixtures/penalty_vectors.safetensors` (committée, `git add -f`)

- [ ] **Step 1 : Écrire le producteur** — reprendre le code de la Task 2 du plan du 27 juil
  (`docs/superpowers/plans/2026-07-27-sampling-repetition-penalty.md:283-327`) TEL QUEL, avec
  ces seules corrections : nom de fichier `76_penalty_vectors.py`, et exécution dans le venv
  M4 (`~/ml-venvs/g12b`) où transformers 5.14.1 est installé. Le producteur appelle
  `RepetitionPenaltyLogitsProcessor` — retranscrire la formule est INTERDIT (spec C5).

- [ ] **Step 2 : Exécuter et vérifier la non-vacuité**

Run : `python3 scripts/76_penalty_vectors.py`
Expected : `touched_1.0 == 0` et `touched_{0.8,1.15,1.5}` == **6** (tokens DISTINCTS de
`hist`, pas 9). Si 9 : le processor ne déduplique pas, STOP — toute la spec est à revoir.
Noter la version transformers affichée.

- [ ] **Step 3 : Commit**

```bash
git add scripts/76_penalty_vectors.py
git add -f fixtures/penalty_vectors.safetensors      # .gitignore:15 exclut fixtures/*.safetensors
git status --short                                   # VÉRIFIER que la fixture est stagée
git commit -m "test(rp1): vecteurs penalty produits par le VRAI processor HF (dédup 6/9 vérifiée) + ties f32 exacts — script 76 (71 collisionne)"
```

---

### Task 3 : `applyRepetitionPenalty` + selftest (RP1) — TDD

**Files :**
- Modify : `zml_runner/sampling.zig` (fonction + champs `SamplingCfg`)
- Modify : `zml_runner/gemma4_g12auto.zig` — `Args` (`:149-218`), parse (`~:269-332`), `usage`
  (`~:201-218`), dispatch selftest (près de `:1774`, pattern `--selftest-sampling`)
- `zml_runner/BUILD.bazel` : **AUCUN changement** (`sampling.zig` est déjà dans les `srcs`
  des 3 cibles, `:527/:537/:547`)

- [ ] **Step 1 : Écrire le selftest `--selftest-penalty <fixture>` qui échoue** — pattern
  exact de `--selftest-sampling` (host-only, early-return avant toute init GPU). Critères
  (spec RP1) : comparaison **0 ULP** (`@bitCast` u32, égalité entière) sur les 4 penalties ;
  assertions de non-vacuité de la fixture (≥ 1 doublon dans `hist`, ≥ 1 logit négatif
  pénalisé, ≥ 1 positif) ; tie-break argmax = **premier indice** sur le vecteur à ties.

- [ ] **Step 2 : Builder → échec attendu** (`applyRepetitionPenalty` n'existe pas).
  Build local suffit : `ZML_REMOTE=… ./zml_runner/build_3090.sh` doit échouer à la compile.

- [ ] **Step 3 : Implémenter dans `sampling.zig`**

```zig
/// Repetition penalty — HF `RepetitionPenaltyLogitsProcessor`, lu à la source : logit
/// négatif ×penalty, positif ou nul ÷penalty, AU PLUS UNE FOIS par token distinct.
/// ⚠ La division RESTE une division (jamais ×(1/p)) : bit-exactitude RP1.
/// `seen` : bitset pré-alloué par run (D10 : zéro allocation ici) ; l'appelant fait
/// `@memset(seen, 0)` par step (32 Kio memset, pas une allocation).
/// Retourne `true` si AU MOINS un logit a changé de bits — c'est la sonde de non-vacuité
/// (une comparaison externe sur un seul id aurait l'angle mort `logit == ±0.0`).
pub fn applyRepetitionPenalty(logits: []f32, hist: []const u32, penalty: f32, seen: []u64) bool {
    if (penalty == 1.0) return false;
    var touched = false;
    for (hist) |t| {
        const i: usize = @intCast(t);
        const w = i >> 6;
        const mask = @as(u64, 1) << @truncate(i);
        if ((seen[w] & mask) != 0) continue;
        seen[w] |= mask;
        const v = logits[i];
        const nv = if (v < 0) v * penalty else v / penalty;
        if (@as(u32, @bitCast(nv)) != @as(u32, @bitCast(v))) touched = true;
        logits[i] = nv;
    }
    return touched;
}
```

Champs à ajouter à `SamplingCfg` (pré-alloués au même site que `scratch`/`work`,
`gemma4_g12auto.zig:1930-1937` et `:2054-2074` — JAMAIS dans la boucle) :

```zig
repetition_penalty: f32 = 1.0,
ignore_prompt: bool = false,
hist: []u32 = &.{},        // capacité L_MAX du binaire, longueur courante ci-dessous
hist_len: usize = 0,
prompt_len: usize = 0,     // longueur du prompt courant — posé au même site que hist (Task 4)
seen: []u64 = &.{},        // (VOCAB_CONTRACT+63)/64 = 4096 mots u64 = 32 Kio
n_penalty_touched: usize = 0,  // non-vacuité : steps où la penalty a changé ≥ 1 logit
n_penalty_empty_hist: usize = 0,  // steps à h vide (exemption PenaltyInert, Task 4)
```

et étendre `pathArmed()` : `… or self.repetition_penalty != 1.0`.
⚠ Garde CLI en **ACCEPTATION** (`!(p > 0 and std.math.isFinite(p))` → rejet) : `p <= 0`
laisserait passer `NaN`. `--ignore-prompt` : booléen sans valeur.
⚠ Refus bruyant `--ignore-prompt` + `--load-cache` (près de `:1769`) : sous reprise,
`ids` = `ids_fed` complet (prompt + générés du run dumpé) — « le prompt » n'y est plus une
notion définie. `error.IgnorePromptWithLoadCache`, dette écrite v1.

- [ ] **Step 4 : Builder + lancer le selftest**

```bash
# Les 2 positionnels ckpt/tokenizer sont OBLIGATOIRES (parseArgs exige >= 3 arguments,
# gemma4_g12auto.zig:225-231) — forme canonique des selftests host-only du repo
# (cf plan 2026-07-29-sampling-phase2.md:387-389) :
$B1 /dev/null /dev/null --selftest-penalty /data/gemma4-zml-probe/fixtures/penalty_vectors.safetensors
```

Expected : `RP1 PASS`, 4×512 valeurs bit-identiques, compteurs non nuls, tie-break conforme.

- [ ] **Step 5 : Commit + tag**

```bash
git add zml_runner/sampling.zig zml_runner/gemma4_g12auto.zig
git commit -m "gate(rp1): applyRepetitionPenalty bit-exacte 0 ULP vs processor HF — bitset pré-alloué (D10), selftest host-only"
git tag gate/rp1-pass
```

---

### Task 4 : Câblage chemin B + RP0 + RP2

**Files :**
- Modify : `zml_runner/gemma4_g12auto.zig` — insertion `:2892-2894`, alimentation de `hist`
  dans la boucle, compteur fin de run (près de `:3144`)

- [ ] **Step 1 : Seeder puis alimenter `hist` — le contrat exact**

Contrainte de cohérence qui ANCRE l'implémentation : **au moment de sélectionner le token
de génération k, `hist[0..hist_len]` == prompt ++ tokens générés avant k** — exactement
l'`input_ids` que HF passe à son processor au même point. Câblage en DEUX temps :

1. **Seed EN TÊTE de `generateOnce`**, depuis son paramètre `ids` (`@memcpy` +
   `hist_len = prompt_len = ids.len`). ⚠ PAS au site d'allocation `:2054-2074` : celui-ci
   ne s'exécute qu'une fois par process, alors que `generateOnce` a **6 sites d'appel**
   (one-shot `:2339`, repl `:2349` et `:2378` — un appel PAR prompt de la boucle stdin —,
   kvdump-eq `:2428/:2436/:2465`) ; seul un seed en tête de fonction couvre structurellement
   les 6, la reprise `--load-cache` (`ids` = `ids_fed` complet, `:1969`, boucle entrant
   directement en génération `:2802-2809`) ET le re-seed par prompt du repl. L'allocation
   des buffers, elle, reste à `:2054-2074` (une fois par process).
2. **Append au point où le token généré est acté** — à côté du
   `generated.appendBounded(tok)` existant (`:3025`), PAS « en fin d'itération » : la fin
   littérale (`fed = tok`, `:3059`) vient APRÈS les 3 `break` (borne oracle, EOT,
   max_tokens) et perdrait le dernier token généré — off-by-one de la famille que la spec
   rév. 4-1 pourchasse, qui fausserait l'égalité du Step 5. Écriture directe
   `scfg.hist[scfg.hist_len] = tok; scfg.hist_len += 1;` — zéro allocation ; garde de
   borne `hist_len < hist.len` en erreur propre AVANT l'écriture. Invariant de fin de run :
   `hist_len == prompt_len + generated.items.len`.
   ⚠ Ne PAS appender `fed` en tête d'itération avec ce seed : le prompt y serait compté
   deux fois. (Le piège off-by-one de la spec rév. 4-1 visait le câblage SANS seed — avec
   seed, c'est l'append de `fed` qui devient le bug.)

Pendant le prefill, `hist` reste le seed complet : les sélections intermédiaires sont
jetées (`:2864-2867`), seule celle du dernier step de prefill (s0) compte — et à ce step
HF a vu exactement le prompt complet. Cohérent par construction.

- [ ] **Step 2 : Insérer la penalty en tête de chaîne** (`:2893`, avant `applySuppression`)

```zig
if (scfg.repetition_penalty != 1.0) {
    // Sous --ignore-prompt, h est VIDE pendant tout le prefill (hist_len == prompt_len) :
    // la garde h.len == 0 est OBLIGATOIRE — en ReleaseFast, h[0] sur slice vide est un UB
    // silencieux, pas un panic. (Et la borne basse est @min-ée par défense.)
    const lo = if (scfg.ignore_prompt) @min(scfg.prompt_len, scfg.hist_len) else 0;
    const h = scfg.hist[lo..scfg.hist_len];
    if (h.len > 0) {
        @memset(scfg.seen, 0); // dans le if : 32 Kio de memset inutiles quand h est vide
        if (sampling.applyRepetitionPenalty(scfg.work, h, scfg.repetition_penalty, scfg.seen))
            scfg.n_penalty_touched += 1;
    } else {
        scfg.n_penalty_empty_hist += 1; // cf exemption PenaltyInert ci-dessous
    }
}
```

Le compteur `n_penalty_touched` est la non-vacuité : en fin de run, si
`repetition_penalty != 1.0` et `n_penalty_touched == 0` → `log.err` +
`return error.PenaltyInert` (spec §3.2.1, « paramètre non propagé »). **Exemption** : si
`h` était vide à CHAQUE step (`n_penalty_empty_hist == n_steps` — cas légitime :
`--ignore-prompt --max-tokens 1`, HF ne toucherait rien non plus), `log.warn` au lieu
d'err. Ajouter le champ `n_penalty_empty_hist: usize = 0` à `SamplingCfg` (Task 3).
**Publier en fin de run** (bloc `:3143-3157`, à ajouter) :
`PENALTY: rp=… hist_len=… prompt_len=… n_penalty_touched=… n_penalty_empty_hist=…` —
le Step 5 et RP5/RP6 s'appuient sur cette ligne.
⚠ `n_disagree` (S2-PONT) va monter sous penalty armée : c'est ATTENDU (écart 8) — publier,
ne pas traiter en FAIL. Le `log.err` S2-PONT par désaccord (`:2959`) doit être conditionné :
sous penalty armée, rétrogradé en compteur silencieux (sinon 40+ lignes d'err par run sain).
⚠ Étendre AUSSI `sampling_str` du manifest de dump (`:2691-2695`, ajouter `,rp=<val>`) :
le code exige lui-même qu'un écart de warpers au restore soit VISIBLE (spec kvdump §4.1) —
sans cette extension, la penalty serait le seul réglage invisible du manifest.

- [ ] **Step 3 : RP0 — le graphe n'a pas bougé**

Rebuild (`build_3090.sh`), re-dump HLO (mêmes flags que Task 1 Step 2), md5 identique au
témoin. + `ALLOC-LOOP: alloc=0` au log d'un run avec penalty armée (l'interdit D10 se
re-vérifie gratuitement — c'est LE point qui tuerait une implémentation qui alloue).

- [ ] **Step 4 : RP2 — non-régression penalty neutre**

Re-run des RUN_ARGS de la Task 1 Step 3 (sans `--repetition-penalty`) → ids bit-identiques
au témoin long. Puis un run AVEC `--repetition-penalty 1.0` explicite → également identique
(le chemin à 1.0 est un no-op par construction, on le VOIT).

- [ ] **Step 5 : Round-trip dump/restore sous penalty** (écart 9 — on vérifie la
  compatibilité au lieu de la supposer)

Run `--repetition-penalty 1.15 --dump-cache <f> --max-tokens 24`, puis
`--load-cache <f> --repetition-penalty 1.15 --max-tokens 8`. Expected : la continuation
démarre, et **le seed de reprise est prouvé** par la ligne `PENALTY:` de fin de run —
`hist_len == step_next + generated` (avec `step_next` lu au manifest par
`scripts/74_kvdump_inspect.py`, sous-commande `inspect` ; invariant du dump :
`ids_fed.len == step_next`, `:2607-2611`). Un `hist_len` qui ignorerait les `step_next`
tokens repris (= le bug de l'écart 9) rendrait **8 au lieu de `step_next + 8`** (~40-50
avec le prompt de référence) : la vérification discrimine sans ambiguïté.
Puis **exercer le refus neuf** (règle 2 : un refus se VOIT échouer) :
`--load-cache <f> --ignore-prompt --repetition-penalty 1.15` → exit non-zéro,
`error.IgnorePromptWithLoadCache`, message archivé dans `docs/evidence/penalty/`.

- [ ] **Step 6 : Commit + tags**

```bash
git add zml_runner/gemma4_g12auto.zig zml_runner/sampling.zig docs/evidence/penalty/
git commit -m "gate(rp0,rp2): penalty en tête de chaîne B — HLO intact (md5 témoin), alloc=0, neutre bit-identique, round-trip dump/restore vérifié"
git tag gate/rp0-pass && git tag gate/rp2-pass
```

---

### Task 5 : Oracle 69 pénalisé + RP3 + RP4

**Files :**
- Modify : `scripts/69_u8_gen_oracle.py` (processor, `s0`, export `prompt_ids`)

- [ ] **Step 1 : Brancher le VRAI processor** — `--repetition-penalty <f>` ; appliqué avant
  l'argmax sur `out.logits[0, -1, :]`, historique = prompt ++ généré (défaut HF).
  Retranscrire le `torch.where` est INTERDIT (spec C5). ⚠ `s0` est produit par le prefill
  HORS boucle (`:220` et autour) : la penalty s'y applique AUSSI, sinon divergence garantie
  au token 0. ⚠ Exporter `prompt_ids` en TENSEUR dans la fixture (le manifest JSON seul ne
  permet pas la comparaison littérale côté runner).

- [ ] **Step 2 : Mordant pré-calculé SANS GPU** (M4, 3 runs oracle)

`--repetition-penalty` ∈ {1.0, 1.15, 0.8} sur le prompt de référence, `--compute-fp32`.
Hamming des `fed` entre 1.0 et chacun des deux autres : attendu **≥ 3 sur 48** (réaliste
~40 par cascade). Publier les valeurs réelles. Si < 3 : changer de prompt, le gate ne
passe PAS à vide.

- [ ] **Step 3 : RP3 — le gate** : runner `--oracle <fixture pénalisée>` pour 0.8 et 1.15.
  Expected : **ids == HF**, `n_penalty_touched > 0`, marge min publiée. Mismatch → §7-3 de
  la spec (3 conditions, ε = 2e-3, fenêtre attendue VIDE — marge min historique 0,0279).

- [ ] **Step 4 : RP4 — les trois corruptions**, chacune → RP3 **FAIL vu** :
  (a) branches de signe échangées ; (b) dédup retirée (`seen` ignoré) ; (c) `ignore_prompt`
  forcé à l'inverse. Publier le mordant de chacune (plancher 1 ; un mordant faible face à la
  cascade attendue est à instruire). Corruptions dans un worktree jetable, jamais committées.

- [ ] **Step 5 : Commit + tags** (`gate/rp3-pass`, `gate/rp4-pass`)

---

### Task 6 : Directives `--repl` + RP5 + RP6 (si D3 = GO)

**Files :**
- Modify : `zml_runner/gemma4_g12auto.zig` — boucle stdin `--repl`

- [ ] **Step 1 : Parser les directives** — une ligne commençant par `:` n'est JAMAIS un
  prompt : `:penalty <f>` (même garde en acceptation), `:ignore-prompt on|off`, `:params`,
  `:help`. Valeur invalide → message, la session CONTINUE. ⚠ Pièges std.Io 0.16 du chantier
  repl (`REPL_RESULTS.md`) : `takeDelimiter` (pas `Exclusive`), writer UNIQUE sur stdout.
  ⚠ `hist` est **RE-SEEDÉ par prompt** — le seed en tête de `generateOnce` (Task 4 Step 1)
  le fait déjà à chacun des appels de la boucle stdin (`:2378`) : rien à ajouter, mais le
  VÉRIFIER (un `hist_len` repartant à 0 nu serait la sémantique `ignore_prompt`, pas le
  défaut HF — et RP5 ne le verrait pas, deux passes identiquement fausses restant égales).

- [ ] **Step 2 : RP5** — même prompt 2×, penalty active, 32 tokens : texte détokenisé
  identique ET `n_penalty_touched > 0` aux deux passes. Puis 20 prompts : RSS ≤ +1 Mo
  (le compteur AL-RSS le publie déjà).

- [ ] **Step 3 : RP6** — (a) `:penalty 1.15` ne génère rien (vérifié par COMPTAGE) ;
  (b) s'applique au prompt SUIVANT (sensibilité prouvée par RP3) ; (c) `:params` vérifié
  CONTRE le comportement (référence = `reponse_hf` du manifest oracle 1.15, rapatrié de M4,
  borné aux 48 premiers tokens, comparaison TEXTE) ; (d) valeurs invalides énumérées
  `0, -1, nan, inf, abc, vide` — **`nan` est le cas qui compte**.

- [ ] **Step 4 : Commit + tags** (`gate/rp5-pass`, `gate/rp6-pass`)

---

### Task 7 : RP7 selon l'arbitrage D4 + M1 (coût)

- [ ] **Step 1 : RP7** — si D4 = requalification (reco) : mesurer la métrique n-gramme
  (longueur max de n-gramme répété, 200 derniers tokens) sur le témoin long ET sur un run
  `--repetition-penalty 1.15`, publier les deux SANS verdict, dette D5 close par
  requalification datée. Si D4 = chercher un prompt qui récite : borné à 3 essais, sinon
  vacuité déclarée.

- [ ] **Step 2 : M1** — le chrono `M-COUT` existant (`d2h_ns`/`warp_ns`) englobe déjà la
  chaîne : publier warp µs/step penalty ON vs OFF (2 runs, mêmes RUN_ARGS, build prouvé
  `ReleaseFast`). Mesure publiée, PAS un gate. ⚠ Ne pas armer `--gate-d1d2` pendant la
  mesure (il travaille dans la fenêtre chronométrée et l'invalide, `SAMPLING_RESULTS.md` §7).

- [ ] **Step 3 : Commit** (+ tag `gate/rp7-pass` seulement si D4 a retenu un gate)

---

### Task 8 : Fiches de cadrage (table C) + clôture

**Files :**
- Create : `docs/superpowers/specs/2026-08-XX-cadrage-dettes-restantes.md` (les 4 fiches)
- Modify : `PLANNING.md`, `docs/SAMPLING_RESULTS.md`, `README.md`, `docs/DOCUMENTATION.md`

- [ ] **Step 1 : Écrire les 4 fiches de cadrage** (une ½ page chacune, AUCUN code) :

1. **K3 / conformité E2B** — sortir les logits du graphe E2B (`gen_auto.zig:753`, 6 sorties
   → 7) OU documenter définitivement la claim comme fausse pour E2B. Prérequis : re-témoins
   HLO E2B, décision sur `suppress_tokens` (Google n'en publie pas pour E2B — en coder
   serait faux). Taille : ~1 session.
2. **K4 / résident à reprise** — sémantique multi-tour de `--repl` + `--load-cache`/
   `--dump-cache` par directive (`:dump <f>`, `:load <f>`). Prérequis : K5 (un 2ᵉ prompt
   après restore EST un prefill partiel). Taille : 1-2 sessions, spec obligatoire.
3. **K5 / prefill partiel** — reprendre un cache et feeder un prompt NEUF : positions,
   masques sliding, RoPE au-delà de `step_next` à dériver et prouver vs HF (teacher-forcé).
   Le chemin naturel vers un vrai serving. Taille : 1-2 sessions, spec obligatoire.
4. **Triton paged attention** — bump ZML + cache YOCO → layout paginé. Seul chemin flash
   B>1 crédible (`ZML_UPSTREAM_AUDIT_2026-07-12.md` §2 : B>1 natif, f32, scale custom,
   sliding window ; hd=512 non testé upstream). Le bump invalide les témoins HLO → re-gates
   complets. Taille : 3+ sessions, LE gros chantier. Reco d'ordre : K5 → K4, K3 et Triton
   indépendants.

- [ ] **Step 2 : Clôture documentaire** — checklist §5.4 de `docs/DOCUMENTATION.md`
  déroulée telle quelle. `SAMPLING_RESULTS.md` : section penalty avec LES CHIFFRES (hamming
  RP3 pour 1.15 ET 0.8, mordant des 3 corruptions RP4, marge min, warp µs/step ON/OFF,
  version transformers) ; dettes D4 (aggravée, écrite) et D5 (selon D4). `PLANNING.md` :
  la table des dettes pointe ce plan comme source de vérité (remplace celui du 10 août).

- [ ] **Step 3 : Grep anonymisation** (commande en tête) → 0 occurrence nouvelle.

- [ ] **Step 4 : Push + PR** — titre
  `feat(sampling): repetition penalty host-side — ids == HF, graphe intact (phase 1)`.
  Corps : les gates dans l'ordre avec les chiffres, les écarts de ré-instruction assumés,
  les fiches de cadrage. Merge `--no-ff` sur GO Régis uniquement.

- [ ] **Step 5 : Mémoire** — `project_gemma4_zml_probe.md` + PLANNING : état des dettes
  après solde, avec dates.

---

## Ordre d'exécution

```
Task 0 (décisions) ─→ Task 1 (RP-1 + témoins) ─→ Task 3 (module, RP1) ─→ Task 4 (RP0/RP2)
                       Task 2 (fixture, M4)  ──↗                          ─→ Task 5 (RP3/RP4)
                                                                          ─→ Task 6 (RP5/RP6, si D3)
                                                                          ─→ Task 7 ─→ Task 8
```

Task 2 (fixture, M4) ne demande aucun GPU et peut courir en parallèle de Task 1.
Les fiches de la Task 8 Step 1 peuvent s'écrire à tout moment (doc pure).
