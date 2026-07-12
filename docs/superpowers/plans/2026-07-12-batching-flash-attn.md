# Batching statique + variante sdpa — plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Étendre le moteur decode au batch statique B>1 (banc multi-prompts instrumenté sur la
3090, plafond VRAM = output) puis mesurer A/B une variante d'attention `zml.nn.sdpa`.

**Architecture:** Engine shape-polymorphe (B dérivé des shapes d'entrée, zéro nouveau champ
comptime en Phase 1) ; nouveau runner autonome `gemma4_bbatch` (squelette gen_auto, lanes
indépendantes greedy, oracle par lane vs fixtures 49) ; sweep script à build unique ; Phase 2 :
champ comptime `attn: {manual, sdpa}` livré avec sa branche.

**Tech Stack:** Zig 0.16-dev + ZML (vendored `adee932e`), Bazel sur VM 3090
(`/data/rqz_workspace/zml`, deploy par rsync), oracles PyTorch/HF (`scripts/49`), gates
tagués `gate/batch-*`.

**Spec:** `docs/superpowers/specs/2026-07-12-batching-flash-attn-design.md` (lue AVANT toute
tâche — les contrats de gates y sont pré-enregistrés).

---

## Conventions d'exécution (toutes tâches)

- **Machines** : édition + commit sur M1 (`~/dev/gemma4-zml-probe`) ; build/run sur la VM 3090
  via `zml_runner/deploy_to_3090.sh` (exiger `ZML_REMOTE`/`ZML_DST` non vides — le script rate
  SILENCIEUSEMENT sinon ; placeholders `user@gpu-host` dans tout ce qui est committé,
  JAMAIS d'IP/user réels).
- **Build GPU** : `./bazel.sh build //examples/rqz:<cible> --@zml//platforms:cuda=true`
  (sans le flag = repli CPU silencieux). Avant tout run :
  `nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv` (contention Ollama).
- **Un commit + tag par gate** : `gate/batch-t0-pass`, `gate/batch-b1-pass`, etc.
  FAIL/null = documenté au même titre (la doc porte les résultats, logs/ gitignorés).
- **Noms courts** (quota comptime `@typeName` pjrt structSize) : `gemma4_bbatch`, structs `BB*`.
- Chaque tâche se termine par la mise à jour de la section Résultats du doc de gates
  (`docs/BATCHING_RESULTS.md`, créé en Task 2).

---

# Phase 1 — Batching (shippable seule)

### Task 0: Capture de la révision ZML 3090 + vérification patch pjrt

**Files:**
- Create: `fixtures/batch_manifest.json` (custody du chantier)

- [ ] **Step 1: Relever la rev du workspace 3090**

Run (depuis M1) : `ssh user@gpu-host "cd /data/rqz_workspace/zml && git rev-parse HEAD && git status --short zml/pjrt.zig"`
Expected: un hash (attendu `adee932e...`) + `M zml/pjrt.zig` (patch local structSize).

- [ ] **Step 2: Vérifier le patch pjrt**

Run : `ssh user@gpu-host "grep -n 'setEvalBranchQuota' /data/rqz_workspace/zml/zml/pjrt.zig"`
Expected: une ligne `@setEvalBranchQuota(100_000)` dans `structSize`. Si absente → la
réappliquer AVANT tout build (cf. `PLANNING.md` « patch local rqz »).

- [ ] **Step 3: Consigner dans le manifest**

Créer `fixtures/batch_manifest.json` : `{ "zml_rev_3090": "<hash>", "zml_rev_m1_mirror":
"adee932e", "pjrt_patch": true, "driver": "<nvidia-smi --query-gpu=driver_version>",
"date": "2026-07-12" }`. Si `zml_rev_3090 != adee932e` : **STOP — remonter à Régis**
(toutes les réfs de lignes de la spec supposent adee932e).

- [ ] **Step 4: Commit**

```bash
git add fixtures/batch_manifest.json
git commit -m "chore(batch): T0 infra — rev ZML 3090 capturée + patch pjrt vérifié"
```

### Task 1: Engine shape-polymorphe (5 sites) + gate T0

**Files:**
- Modify: `zml_runner/engine.zig:395,412,417,536,539` (et déclarations 23-24 : commentaire)
- Baseline: dump HLO du HEAD inchangé, déployé + **rebuildé** AVANT modification
  (méthode G2.3.0 gold, cf. Step 1 — pas de worktree)

- [ ] **Step 1: Produire la baseline HLO AVANT modification**

Step 1 précède toute modification → déployer le HEAD inchangé suffit (pas de worktree).
**Le build est OBLIGATOIRE avant le dump** (binaire stale = baseline fausse — vécu 11 juil) :
```bash
ZML_REMOTE=user@gpu-host ZML_DST=/data/rqz_workspace/zml/examples/rqz ./zml_runner/deploy_to_3090.sh
ssh user@gpu-host "cd /data/rqz_workspace/zml && ./bazel.sh build //examples/rqz:gemma4_gen_auto --@zml//platforms:cuda=true"
# puis, sur la 3090 :
XLA_FLAGS="--xla_dump_to=/tmp/hlo_t0_before" ./bazel-bin/examples/rqz/gemma4_gen_auto \
  <model.safetensors> <tokenizer.json> --prompt "test" --max-tokens 2 --force-vram
md5sum /tmp/hlo_t0_before/*before_optimizations.txt   # noter le md5 de module_0001
```
Expected: md5 noté (baseline). NB : SEUL le `before_optimizations` est stable (piège 15) ;
le compile est déclenché par compileFn avant la boucle (`gen_auto.zig:994`), 2 tokens suffisent.

- [ ] **Step 2: Modifier les 5 sites**

Dans `engine.zig`, le SEUL changement autorisé = les deux premiers arguments de chaque
`.reshape(...)` (B→`<scope>.dim(.b)`, S→`<scope>.dim(.s)`). Ne toucher NI aux appels dotPrec
NI aux qualificateurs (`var q` reste `var q` — il est réassigné lignes 396-397) :

```zig
// Site 1 (q, ligne ~395) — code réel, seul le reshape change :
var q = dotPrec(prec.qkv_proj, prec.compute, h0, layer.q_proj, .d)
    .reshape(.{ h0.dim(.b), h0.dim(.s), NH, hd })      // ← avant : .{ B, S, NH, hd }
    .withTags(.{ .b, .s, .nh, .hd });
// Sites 2 et 3 (k ~412, v ~417) : même geste (KVH au lieu de NH), h0 en scope.
// Site 4 (PLE token_identity ~536) — embptl_slice {b,s,lf} en scope :
//   .reshape(.{ embptl_slice.dim(.b), embptl_slice.dim(.s), NUM_LAYERS, PLE_DIM })
// Site 5 (PLE context ~539) — embeds {b,s,d} en scope : idem.
```

Sur les déclarations `pub const B/S` (lignes 23-24) : ajouter le commentaire
`// Plus consommées par le moteur (shape-polymorphe depuis batch T0) ; gardées pour les runners.`
NE TOUCHER À RIEN d'autre (ordre d'émission rmsScale* verrouillé, spec §3.1).

- [ ] **Step 3: Déployer + rebuild gen_auto (témoin)**

```bash
ZML_REMOTE=user@gpu-host ZML_DST=/data/rqz_workspace/zml/examples/rqz ./zml_runner/deploy_to_3090.sh
ssh user@gpu-host "cd /data/rqz_workspace/zml && ./bazel.sh build //examples/rqz:gemma4_gen_auto --@zml//platforms:cuda=true"
```
Expected: build OK (build 3090 tôt — les erreurs comptime sortent ici).

- [ ] **Step 4: Dump HLO APRÈS + comparaison md5**

Même commande de dump que Step 1 vers `/tmp/hlo_t0_after`, puis comparer le md5 du
`module_0001.zml.before_optimizations.txt` des deux dumps.
Expected: **md5 identiques** → T0 PASS. Si diff : appliquer le repli pré-enregistré de la spec
§6 T0 (diff catégorisé publié + G1 == HF + E1 4/4 + replay == HF cumulés), sinon rejeter.

- [ ] **Step 5: Consigner + commit + tag**

Résultat (md5, taille des dumps) dans `docs/BATCHING_RESULTS.md` §T0 (créer si besoin).
Optionnel (confiance two_masks=false, spec §6 T0) : md5 secondaire sur `gemma4_engine_e1`.
```bash
git add zml_runner/engine.zig docs/BATCHING_RESULTS.md
git commit -m "feat(batch): T0 — engine shape-polymorphe (dim(.b)/dim(.s) aux 5 sites), HLO byte-identique"
git tag gate/batch-t0-pass
```

### Task 2: Squelette gemma4_bbatch (host-only : CLI, tokenisation, --ids-only)

**Files:**
- Create: `zml_runner/gemma4_bbatch.zig`
- Modify: `zml_runner/BUILD.bazel` (nouvelle cible, pattern `gemma4_gen_auto` :455-461)
- Create: `docs/BATCHING_RESULTS.md` (squelette des sections T0→B4, S1→S3)

- [ ] **Step 1: Cible Bazel**

Dans `BUILD.bazel`, copier le bloc zig_binary de `gemma4_gen_auto` (lignes ~455-461 — PAS
:373-381 qui est engine_e1, sans mem_probe) : `name = "gemma4_bbatch"`,
`main = "gemma4_bbatch.zig"`, `srcs = ["engine.zig", "mem_probe.zig"]`,
`deps = ["//bazel", "//zml"]`.

- [ ] **Step 2: CLI + tokenisation multi-prompts (host-only, sans GPU)**

Écrire `gemma4_bbatch.zig` en copiant les motifs de gen_auto (mêmes lignes de référence) :
- constantes/Model : reprendre `gen_auto.zig:39-51` à l'identique ;
- CLI (`gen_auto.zig:79-150` comme modèle) : positionnels `<model> <tokenizer>`, flags
  `--prompts <fichier>` (un prompt/ligne — **toujours requis** hors `--selftest-batch`),
  `--oracles <f1,f2,...>` (**apparié par index aux lignes de `--prompts`** : la fixture 49 ne
  contient PAS les ids du prompt, seul son manifest sidecar les a → le runner tokenise les
  prompts et vérifie `positions[0] == ids.len` par lane, exactement comme gen_auto apparie
  `--prompt` + `--oracle` ; nombre de fixtures == nombre de prompts, sinon
  `error.OracleCountMismatch`), `--replicate N`, `--max-tokens N`, `--force-vram`,
  `--allow-cpu`, `--no-prealloc`, `--ids-only`, `--selftest-batch <fixture>` (stub pour Task 3) ;
- tokenisation par prompt : template chat + BOS explicite + EOT mesuré
  (`gen_auto.zig:55-64,783-805`, encoder.reset() entre prompts — automate à état) ;
- **vérif longueurs** : toutes les lanes au même `ids.len`, sinon `error.PromptLengthMismatch`
  avec log des longueurs par lane ; gardes par lane `ids.len < 512`,
  `ids.len + limit <= 1024` (`gen_auto.zig:889-897`) ;
- `--ids-only` : afficher `lane i : len=N ids=[...]` + round-trip détok par lane
  (`gen_auto.zig:807-834`) puis exit — c'est l'outil de constitution du jeu de prompts.

- [ ] **Step 3: Vérifier le mode host-only (binaire 3090, `--ids-only` sans travail GPU)**

Run : compilation locale impossible hors workspace ZML → déployer et builder sur 3090
(mêmes commandes que Task 1 Step 3, cible `gemma4_bbatch`), puis :
`./bazel-bin/examples/rqz/gemma4_bbatch <model> <tok> --prompts /tmp/p3.txt --ids-only`
avec `/tmp/p3.txt` = 3 prompts de longueurs différentes.
Expected: longueurs affichées par lane, round-trip OK, puis
`error.PromptLengthMismatch` sur un run SANS `--ids-only` (non-vacuité de la garde).

- [ ] **Step 4: Commit**

```bash
git add zml_runner/gemma4_bbatch.zig zml_runner/BUILD.bazel docs/BATCHING_RESULTS.md
git commit -m "feat(batch): squelette gemma4_bbatch — CLI, tokenisation N lanes, --ids-only, garde longueurs"
```

### Task 3: Gate B1 — selftest des primitives batchées

**Files:**
- Modify: `zml_runner/gemma4_bbatch.zig` (mode `--selftest-batch`)

- [ ] **Step 1: Écrire le mini-graphe (pattern SgFwd, `gen_auto.zig:495-516`)**

Mode `--selftest-batch <fixture>` (réutilise la fixture SG existante de gen_auto), B=2 en dur.
**Dispatcher ce mode AVANT l'exigence `--prompts`** (contrairement à gen_auto dont le
`--selftest-gather` exige un `--prompt` factice — bbatch n'a pas cette contrainte, le selftest
est autonome) :
1. `gather` : tok `{2,1}` u32 → `embed_tokens.gather(.{.voc=tok}, .{})` — comparer chaque lane
   au gather 1-lane de référence (bit-exact u16, motif SG, AUCUNE tolérance) ;
2. `scatterSlices` batché : cache jouet `{slot=1, b=2, h=1, k=8, hd=4}` zéros, update
   `{b=2,h=1,k=1,hd=4}` à `pos_u` scalaire=3 — relire et vérifier que chaque lane a reçu SA
   ligne d'update à k=3 et zéros ailleurs (valeurs distinctes par lane, ex. lane0=1.0, lane1=2.0) ;
3. `topK` : logits jouets `{b=2, s=1, voc=16}` avec argmax connus distincts par lane →
   `topK(.{.voc=.voc}, 5, .{})`, vérifier layout D2H `{b,5}` (indices ET valeurs par lane,
   assert dtype i32 — motif `gen_auto.zig:1054-1060`) ;
4. `broad` masque rank-égal : masque `{b=1,h=1,q=1,k=8}` additif, scores jouets
   `{b=2,h=1,q=1,k=8}` → `scores.add(mask.broad(scores.shape()))`, vérifier que les DEUX lanes
   reçoivent le masque (spec §2.8 — l'ordre des axes fait que le broadcast positionnel est
   correct ; ce step le PROUVE au lieu de le supposer).

- [ ] **Step 2: Build + run sur 3090**

```bash
ssh user@gpu-host "cd /data/rqz_workspace/zml && ./bazel.sh build //examples/rqz:gemma4_bbatch --@zml//platforms:cuda=true"
./bazel-bin/examples/rqz/gemma4_bbatch <model> <tok> --selftest-batch fixtures/<sg_fixture>.safetensors
```
Expected: 4 sous-tests PASS, valeurs exactes loggées.

- [ ] **Step 3: Consigner + commit + tag**

```bash
git add zml_runner/gemma4_bbatch.zig docs/BATCHING_RESULTS.md
git commit -m "feat(batch): B1 — selftest primitives batchées (gather/scatter/topK/broad) B=2 exact"
git tag gate/batch-b1-pass
```

### Task 4: Boucle de génération batchée complète

**Files:**
- Modify: `zml_runner/gemma4_bbatch.zig`

- [ ] **Step 1: Symboliques et buffers à B lanes**

Reprendre `gen_auto.zig:937-988` en paramétrant b=B (valeur runtime = nombre de lanes) :
`tok_sym {B,1}` ; `packed_sym` INCHANGÉ à b=1 (tables partagées, JAMAIS ×B — spec §2.8) ;
`cache_sym` `{slots, B, 1, L_MAX, HD}` ; caches initiaux zéros dimensionnés ×B
(`HostInputs` : reprendre `gen_auto.zig:204-327`, seules les shapes cache changent).

- [ ] **Step 2: StepTok batché + boucle par lane**

- `BBStep.forward` : copie de `StepTok.forward` (`gen_auto.zig:742-755`) — le code est
  IDENTIQUE (gather/forwardStep/topK sont shape-polymorphes), seuls les shapes d'entrée changent.
- Boucle (`gen_auto.zig:1024-1118` comme modèle), généralisée :
  `fed: [B]u32` (bounds-check PAR lane avant `@bitCast` — gather clampe silencieusement),
  H2D = 1 Buffer `[B]u32` + 1 scalaire step ; D2H = top-K `{b,5}` (extraction par lane,
  stride 5, assert dtype i32) ; `generated`, `gen_top5`, `StopReason` par lane ;
  prefill-par-decode partagé (même `ids.len` ⇒ même frontière) ; lane finie (EOT) →
  feed EOT réinjecté + sortie masquée (flag `active: [B]bool`) ; arrêt quand toutes inactives
  ou `--max-tokens` ; L_MAX garde commune.
- Mesure : reprendre la convention EXACTE de `gen_auto.zig:1020-1023,1091,1119-1150`
  (t_prefill_end au dernier step de prefill, gen_elapsed avant les deinit, s0 compté dans
  generated) ; afficher `BB PERF : prefill X tok/s ; gén agrégé Y tok/s ; par lane [..]` +
  `K=5` consigné.
- Garde de contention : dupliquer `checkVram` (`gen_auto.zig:683-736`) en retirant le seuil
  fixe : refus seulement si d'autres process compute sont listés (échappatoire `--force-vram`).

- [ ] **Step 3: Run de fumée B=2 sur 3090 (mode libre)**

`/tmp/p2.txt` = 2 prompts distincts de même longueur (constitués via `--ids-only`).
Run : `... gemma4_bbatch <model> <tok> --prompts /tmp/p2.txt --max-tokens 32`
Expected: 2 sorties texte distinctes et cohérentes, perf loggée, zéro crash.

- [ ] **Step 4: Commit**

```bash
git add zml_runner/gemma4_bbatch.zig
git commit -m "feat(batch): boucle de génération batchée — lanes indépendantes, EOT par lane, mesure L3-compatible"
```

### Task 5: Oracles par lane + gate B2

**Files:**
- Create: `scripts/60_batch_oracles.sh` (wrapper : N invocations de `scripts/49`)
- Create: `fixtures/bench_prompts_b4.txt` + `fixtures/bench_prompts_b2.txt` (jeux versionnés,
  prompts distincts de même longueur tokenisée)
- Modify: `zml_runner/gemma4_bbatch.zig` (mode `--oracles`, apparié à `--prompts`)

- [ ] **Step 1: Constituer le jeu de prompts versionné**

Via `--ids-only` sur des candidats : trouver 4 prompts distincts de même longueur tokenisée
(ajuster le texte jusqu'à égalité). Committer `fixtures/bench_prompts_b4.txt` et
`fixtures/bench_prompts_b2.txt` (ses 2 premières lignes — pour le gate B2 à B=2).

- [ ] **Step 2: Générer les 4 fixtures oracle sur la 3090**

`scripts/60_batch_oracles.sh` : boucle `python3 scripts/49_gen_custom_oracle.py --prompt "..."
--n-tokens 48 --out /data/fixtures/oracle_lane<i>.safetensors` (une par prompt, GPU requis,
HF offline). **`--n-tokens 48` OBLIGATOIRE** : le défaut du script est 200, mais le critère
pré-enregistré B2 est 48/48 (mitigation ties/GEMM, spec §7) et le bras B=1 du protocole B4
suppose 48 tokens. Vérifier dans chaque manifest sidecar : `seq_len` identique partout.

- [ ] **Step 3: Mode --oracles**

Reprendre le motif `--oracle` de gen_auto (`gen_auto.zig:849-897,1153-1179`) par lane :
`--oracles` est **apparié à `--prompts`** (lane i = ligne i du fichier de prompts + fixture i) ;
lane i comparée token par token à SA fixture (`generated[i][k] == fed_i[k]`), garde
`positions[0]==ids.len_i` par lane, diagnostic top-5 au premier mismatch d'une lane
(step fautif, marge top1−top2 — procédure d'échec spec §4). Vérifier au passage que les
`prompt_ids` du manifest sidecar de chaque fixture == les ids tokenisés de la ligne appariée
(sinon `error.OraclePromptMismatch` — évite le faux PASS sur fixture désappariée).

- [ ] **Step 4: Gate B2 — B=2 puis B=4**

```bash
# B=2 : 2 premières lignes du jeu + les 2 fixtures appariées (fichier prompts à 2 lignes)
... gemma4_bbatch <model> <tok> --prompts fixtures/bench_prompts_b2.txt \
      --oracles /data/fixtures/oracle_lane0.safetensors,/data/fixtures/oracle_lane1.safetensors
# B=4 : le jeu complet + les 4 fixtures
... gemma4_bbatch <model> <tok> --prompts fixtures/bench_prompts_b4.txt --oracles <les 4>
```
Expected: chaque lane 48/48. **Non-vacuité** : re-run avec UNE fixture au `fed` altéré
(script python 3 lignes, ou fixture dédiée) → FAIL obligatoire sur cette lane.
Si mismatch à marge fine : appliquer la procédure §4 (FAIL publié, requalification documentée).

- [ ] **Step 5: Consigner + commit + tag**

```bash
git add scripts/60_batch_oracles.sh fixtures/bench_prompts_b4.txt zml_runner/gemma4_bbatch.zig docs/BATCHING_RESULTS.md
git commit -m "feat(batch): B2 — fidélité par lane B=2/B=4, 48/48 == fixtures HF, non-vacuité vérifiée"
git tag gate/batch-b2-pass
```

### Task 6: Gate B3 — indépendance inter-lanes

**Files:**
- Modify: `zml_runner/gemma4_bbatch.zig` (comparaison inter-lanes en mode `--replicate`)

- [ ] **Step 1: Implémenter la comparaison inter-lanes** : en mode `--replicate`, à chaque
  step, comparer les ids u32 de toutes les lanes actives à la lane 0 ; tracer le step d'EOT
  par lane ; en fin de run, log `B3 : lanes identiques N/N steps` ou FAIL au premier écart
  (step, lanes, ids divergents).

- [ ] **Step 2: Choisir le prompt** (EOT dans la fenêtre des 48 steps si possible — sinon
  documenter la clause EOT vacuous, spec §6 B3) et lancer `--replicate 4` sur 1 prompt,
  `--max-tokens 48`.

- [ ] **Step 3: Vérifier** : ids u32 identiques sur les 4 lanes à chaque step actif +
  même step d'EOT.

- [ ] **Step 4: Consigner + commit + tag** (`gate/batch-b3-pass`).

### Task 7: Protocole pré-enregistré + script sweep

**Files:**
- Create: `docs/BATCH_BENCH_PROTOCOL.md` (AVANT tout run B4 — discipline G2.3 §5)
- Create: `scripts/61_batch_sweep.sh` (pattern `g2_3_sweep.sh` : sha256, manifest, tee)

- [ ] **Step 1: Écrire et committer le protocole** — contenu figé par la spec §3.3/§6 B4 :
  liste B = {1,2,4,8,16,…} ; charge : bras appariés B=1 = fixture 49, B=2/4 = jeu
  `bench_prompts_b4.txt`, B≥8 = `--replicate` + spot-check lane 0 ; 3 runs/bras ; médiane ;
  budget −5 % ; run long 999 pour le pic VRAM (`--no-prealloc` + échantillonnage nvidia-smi,
  vérifier `ids.len + 999 <= 1024`) ; arrêt par projection pic(2B) > 22 GiB ; seuil « marginal »
  = verdict à <5 pts → re-run K apparié qui remplace.

```bash
git add docs/BATCH_BENCH_PROTOCOL.md && git commit -m "docs(batch): protocole du banc pré-enregistré (avant tout run B4)"
```

- [ ] **Step 2: Écrire `scripts/61_batch_sweep.sh`** : build unique + sha256 consigné/vérifié
  avant chaque run, boucle sur B avec **à chaque point : run fidélité/spot-check
  (`--oracles`, ou lane 0 pour B≥8) PUIS run mesure** (la colonne « verdict fidélité » de la
  table B4 en dépend), runs appariés gen_auto/bbatch à B=1, échantillonneur VRAM (boucle
  `nvidia-smi --query-compute-apps` pendant le run long), écriture du manifest
  (`fixtures/batch_manifest.json` : ajouter table des points), tee des logs.

- [ ] **Step 3: Commit.**

### Task 8: Gate B4 — exécution du sweep

- [ ] **Step 1: GPU vierge** (`nvidia-smi` — arrêter Ollama au besoin) puis
  `scripts/61_batch_sweep.sh` complet.
- [ ] **Step 2: Remplir la table** dans `docs/BATCHING_RESULTS.md` : B → {tok/s agrégé,
  par lane, pic VRAM, compile s, verdict fidélité} ; plafond B ; verdict non-régression
  (médiane bbatch B=1 ≥ 0,95 × médiane gen_auto).
- [ ] **Step 3: Consigner + commit + tag** (`gate/batch-b4-pass`).
  **CHECKPOINT : Phase 1 shippable — revue humaine avant Phase 2** (PR possible ici).

# Phase 2 — Variante sdpa (détachable)

### Task 9: EngineCfg.attn + branche + gate S1

**Files:**
- Modify: `zml_runner/engine.zig` (champ `attn` + branche aux lignes ~459-467)

- [ ] **Step 1: Baseline HLO avant** (même méthode que Task 1 Step 1, gen_auto au HEAD courant).
- [ ] **Step 2: Ajouter** `attn: enum { manual, sdpa } = .manual` à `EngineCfg` ET la branche :

```zig
const ctx_out = if (comptime cfg.attn == .sdpa) blk: {
    // q_final.dtype(), PAS qs.dtype() — qs (ligne 459) est déclaré DANS le bloc else,
    // hors scope ici ; erreur qui n'apparaîtrait qu'au premier build sdpa (S2)
    const un = zml.Tensor.scalar(1.0, q_final.dtype());     // scaling Gemma = 1.0, JAMAIS 1/√hd
    break :blk zml.nn.sdpa(q_final, cache_k, cache_v, .{ .attn_mask = mask, .scale = un });
} else blk: {
    // chemin manuel actuel INCHANGÉ (lignes 459-465), déplacé tel quel dans ce bloc
    break :blk ...;
};
```
(Adapter les renames de tags de sortie pour que la suite — merge .m + o_proj — reçoive la même
shape dans les deux branches ; sdpa retourne `transpose(q.shape()).merge(.{.h={.h,.hq}})`.)

- [ ] **Step 3: Build 3090 + dump HLO après + md5** — Expected: **identique** (branche
  comptime-morte). Consigner, commit, tag `gate/batch-s1-pass`.

### Task 10: Gate S2 — fidélité sdpa

**Files:**
- Create: `zml_runner/gemma4_bbs.zig` (second main, `.attn = .sdpa`)
- Modify: `zml_runner/BUILD.bazel` (cible `gemma4_bbs`, pattern e1/e2 :375-390)

- [ ] **Step 1: Second main minimal** (mécanisme TRANCHÉ — pas de `-D`, rules_zig n'a pas de
  defines) : créer `zml_runner/gemma4_bbs.zig` (nom court, quota @typeName) qui instancie
  `Model` avec `.attn = .sdpa` et réutilise tout le reste via `srcs` — pattern déjà validé
  par le couple gemma4_engine_e1/e2 (`BUILD.bazel:375-390`). Pour partager le code : extraire
  le corps de bbatch en fonction `pub fn run(comptime Model: type, ...)` appelée par les deux
  mains, OU dupliquer le main court — choisir ce qui garde les deux `@typeName` courts.
- [ ] **Step 2: Run oracles** B=1 et B=4 (mêmes fixtures que B2). Expected : 48/48 par lane ;
  sinon procédure §4. Consigner, commit, tag `gate/batch-s2-pass`.

### Task 11: Gate S3 — A/B perf

- [ ] **Step 1: Runs appariés** manual vs sdpa aux mêmes B (protocole Task 7 : 3×, médiane).
- [ ] **Step 2: Verdict** : |Δ médiane| < 5 % ⇒ « pas de gain démontrable » (conclusion honnête
  — le cudnn est mort, on mesure la fusion XLA). Consigner tout, commit, tag `gate/batch-s3-pass`.

# Détachables (après B4, ordre libre)

### Task 12: Sampling host-side (mode charge, hors gates)

- [ ] K paramétrable (const K, défaut 64) dans BBStep ; `--temperature/--top-k/--seed` ;
  softmax host sur les K logits par lane, RNG splitmix64 seedé (seed, lane, step) ;
  troncature top-K documentée dans l'aide CLI. Run de fumée B=2. Commit.

### Task 13: Spike F1 — FA2 réel B=1 (optionnel, non bloquant)

- [ ] Essai `zml.attention` backend cuda_fa2 sur un mini-graphe B=1 (hors engine) ; limites
  attendues : fp16/bf16 + head_dim ≤ 256 dans le .so → le lane full hd=512 fp32 échouera
  probablement. Résultat publié quel qu'il soit dans `docs/BATCHING_RESULTS.md` §F1. Commit.

### Task 14: Documentation finale

- [ ] `docs/DOCUMENTATION.md` : usage bbatch + pièges nouveaux (PromptLengthMismatch, garde
  contention, broad rank-égal prouvé, tables Packed jamais ×B).
- [ ] `PLANNING.md` : item [M] batching → état final.
- [ ] Commit. PR `batching` → main (bundle Phase 1+2 ou Phase 1 seule selon checkpoint Task 8).
