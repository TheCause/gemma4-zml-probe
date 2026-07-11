# L3 in-graph `gemma4_gen_auto` — Plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal :** le forward de `gemma4_gen_auto` devient token_in → token_out (gather embeddings +
topK in-graph) — le host ne thread plus qu'un scalaire par step. Spec approuvée :
`docs/L3_INGRAPH_DESIGN.md`.

**Architecture :** wrapper `StepTok` local au runner composant `gather` (embed_tokens DÉJÀ
device via lm_head tied ; embed_tokens_per_layer ajouté via struct `Tabs` chargé par le même
`TensorStore`) → `Model.forwardStep` inchangé → `topK(.voc, 5)` (top1 = next token, top5 =
diagnostic). `engine.zig` intact d'un octet. Suppression de `EmbedGather` et `top5Of`.

**Tech stack :** Zig 0.16.0-dev.2722 / ZML (workspace 3090). Faits d'API vérifiés le 11 juil
2026 dans les sources exactes : `Tensor.gather(.{ .voc = ids })` (validé bit-exact P5.4) ;
`Tensor.topK(named_axis, k, .{})` = `sort` descendant + `slice1d` (tensor.zig:3096, retour
`SortRes = ArgMaxRes` avec `.values`/`.indices`, indices lus en `i32` dans les tests ZML) ;
chargement générique `zml.io.load(Self, self, allocator, io, platform, store, .{...})`
(pattern engine.zig:272-274) ; `Model.embed_tokens` taggué `{.voc,.d}` (engine.zig:487/507).
⚠ `topK` = TRI COMPLET de 262 144 f32 par step — accepté d'office, le bench G3 tranche ;
repli documenté si le tri pèse : `argMax(.voc)` pour le next + top5 conservé seulement en
mode `--oracle` (2e compile). Build : Bazel 3090 uniquement, validation par gates réels.

**Convention accès distant :** commandes avec placeholders (`user@gpu-host`, `/data/...`) ;
vraies valeurs `ZML_REMOTE`/`ZML_DST` passées EN ENV à l'exécution, jamais committées.
⚠ Piège : sans les env vars, `deploy_to_3090.sh` échoue sur les placeholders et on teste
l'ANCIEN binaire — toujours vérifier la ligne `Deployed …`.

> Branche de travail : **`l3-ingraph`** (existe, la spec y est committée). Chemins 3090 :
> checkpoint `/data/gemma4-zml-probe/weights/model.safetensors`, tokenizer
> `/data/gemma4-zml-probe/gemma4-e2b-it-meta/tokenizer.json`, fixtures sous
> `/data/gemma4-zml-probe/fixtures/` (noms exacts : `ls` + `docs/GEN_AUTONOME_PLAN.md`).

---

## Task 1 : B0 — bench AVANT (sur le binaire actuel, AVANT tout édit)

Le workspace 3090 contient le code de `main` post-PR #6 (garde VRAM) déjà buildé. Si un doute :
redéployer/rebuilder AVANT de toucher au code local.

- [ ] **Step 1.1 :** GPU libre (`nvidia-smi` > 10 GiB libres, sinon `ollama stop <modèle>`).
- [ ] **Step 1.2 : run court oracle** — binaire `bazel-bin/examples/rqz/gemma4_gen_auto` avec
  `--oracle <fixture A1 48-steps>` (nom exact dans `/data/gemma4-zml-probe/fixtures/`, cf
  GEN_AUTONOME_PLAN) → noter la ligne `A1 PERF : … tok/s` et le temps de compile.
- [ ] **Step 1.3 : run long libre** — `--prompt "Explique-moi la fenêtre glissante d'attention
  en trois phrases." --max-tokens 200` → noter PERF (attendu ~54 tok/s, réf DOCUMENTATION §2.2).
- [ ] **Step 1.4 :** consigner les 2 mesures dans le rapport de task (pas de commit — elles
  seront écrites dans la doc en Task 9). NB : sur CE binaire, prefill et génération coûtent le
  même prix par step (même call) — le tok/s global EST le per-step rate de référence.

## Task 2 : code — `Tabs` + `StepTok` + boucle réécrite + seuil garde provisoire

**Files:**
- Modify: `zml_runner/gemma4_gen_auto.zig` uniquement. `engine.zig` INTERDIT de modification.

- [ ] **Step 2.1 : supprimer `EmbedGather`** (bloc complet l.456-~605 : bannière « Task 4 —
  gather embeds bruts », `EMB_KEY`/`EPTL_KEY` CONSERVÉS — déplacés juste au-dessus de `Tabs`),
  **supprimer `top5Of`** (l.666-687). GARDER le struct `Top5` (l.664) — il est rempli depuis le
  device désormais. `selftestGather` : le corps sera réécrit en Task 4 ; pour compiler dès cette
  task, le remplacer temporairement par `return error.SelftestGatherRewriteEnCours;` (une ligne,
  documentée `// Task 4 du plan L3`).

- [ ] **Step 2.2 : struct `Tabs`** (au niveau module, près des constantes) :

```zig
// Table L3 (spec docs/L3_INGRAPH_DESIGN.md §2.1) : SEULE table ajoutée au device —
// embed_tokens est déjà résident dans Model (lm_head tied, engine.zig:487), le gather le
// réutilise. Nom court OBLIGATOIRE (piège quota comptime @typeName, cf spec §2).
const Tabs = struct {
    eptl: zml.Tensor, // {voc,lf} bf16 BRUT (scaling ×16 déjà dans forwardStep)

    fn init(base: zml.io.TensorStore.View) Tabs {
        return .{ .eptl = base.createTensor("embed_tokens_per_layer.weight", .{ .voc, .lf }, null) };
    }
    fn load(self: *const Tabs, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, store: *const zml.io.TensorStore, shardings: []const zml.sharding.Sharding) !zml.Bufferized(Tabs) {
        return zml.io.load(Tabs, self, allocator, io, platform, store, .{ .shardings = shardings, .parallelism = 1, .dma_chunks = 1, .dma_chunk_size = 16 * 1024 * 1024 });
    }
};
```

- [ ] **Step 2.3 : struct `StepTok`** (juste au-dessus de `pub fn main`) :

```zig
// L3 (spec §2) : compose gather in-graph + forwardStep (engine INTACT) + topK.
// top1 du topK == argmax (tri descendant, tie-break XLA) ; top5 = diagnostic --oracle.
const StepTok = struct {
    pub fn forward(model: Model, tabs: Tabs, tok: zml.Tensor, p: PackedLong, cache: engine.Cache, ctrl: engine.Ctrl) struct { zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor } {
        const e = model.embed_tokens.gather(.{ .voc = tok }); // {b,s,d} bf16 brut
        const el = tabs.eptl.gather(.{ .voc = tok }); // {b,s,lf} bf16 brut
        const logits, const slk, const slv, const flk, const flv = model.forwardStep(e, el, p, cache, ctrl);
        const t5 = logits.topK(.voc, 5, .{});
        return .{ t5.values, t5.indices, slk, slv, flk, flv };
    }
};
```

Symbolique du token : `const tok_sym = zml.Tensor.init(.{ 1, 1 }, .u32).withTags(.{ .b, .s });`
**Repli si le gather rank-2 ne compile pas** (P5.4 n'a validé que des ids 1-D) : `tok_sym`
en `{ .s }` shape `[1]`, puis dans `forward` : `e = ...gather(.{ .voc = tok }).reshape(.{ 1, 1, D }).withTags(.{ .b, .s, .d })`
(reshape layout-preserving + re-tag, piège ZML #1 connu) — idem `el` avec LF.
**Repli dtype** : si le gather exige des indices i32, passer `tok_sym`/host en `.i32` (le vocab
< 2^31, cast sans perte).

- [ ] **Step 2.4 : chargement.** Après `const eng_buf = try model.load(...)` (l.~950) et AVANT
  `store_ck.deinit()` :

```zig
    const tabs: Tabs = .init(base); // même view withPrefix que Model.init (l.~843)
    const tabs_buf = try tabs.load(arena.allocator(), io, platform, &store_ck, &.{sharding});
```

- [ ] **Step 2.5 : compile.** Remplacer le `compileFn(…, Model.forwardStep, .{ model, embeds_sym,
  embptls_sym, packed_sym, cache_sym, ctrl_sym }, …)` par :

```zig
    var exe = try platform.compileFn(allocator, io, StepTok.forward, .{ model, tabs, tok_sym, packed_sym, cache_sym, ctrl_sym }, .{ .shardings = &.{sharding} });
```

Les symboliques `embeds_sym`/`embptls_sym` deviennent inutiles → supprimer. `packed_sym`
conserve ses champs embeds/embptls factices (type `Packed(true)`, jamais lus — commentaire
existant l.655-657 toujours vrai).

- [ ] **Step 2.6 : boucle réécrite.** Dans la boucle (l.~1006) : supprimer `gather_tbl.*`,
  `embeds_step_buf`, `embptls_step_buf` et leurs deinit. Nouveau corps par step :

```zig
        var tok_host = [1]u32{@intCast(fed)};
        var tok_buf = try zml.Buffer.fromBytes(io, platform, tok_sym.shape(), sharding, std.mem.sliceAsBytes(&tok_host));
        var step_buf = try zml.Buffer.scalar(io, platform, @as(u32, @intCast(step)), .u32, sharding);
        const ctrl_buf = zml.Bufferized(engine.Ctrl){ .step = step_buf };

        var call_args = try exe.args(allocator);
        var call_results = try exe.results(allocator);
        call_args.set(.{ eng_buf, tabs_buf, tok_buf, pk_buf, cache_buf, ctrl_buf });
        exe.call(call_args, &call_results);
        var r_t5v, var r_t5i, const r_slk, const r_slv, const r_flk, const r_flv = call_results.get(struct {
            zml.Buffer, zml.Buffer, zml.Buffer, zml.Buffer, zml.Buffer, zml.Buffer,
        });

        // Top5 depuis le device (~48 octets D2H) — top1 = next token.
        var t5v_s = try r_t5v.toSliceAlloc(allocator, io);
        defer t5v_s.free(allocator);
        var t5i_s = try r_t5i.toSliceAlloc(allocator, io);
        defer t5i_s.free(allocator);
        const t5i = t5i_s.items(i32); // dtype à confirmer au build (test ZML lit i32)
        const t5v = t5v_s.items(f32);
        var top5: Top5 = undefined;
        for (0..5) |j| {
            top5.idx[j] = @intCast(t5i[j]);
            top5.val[j] = t5v[j];
        }
        const tok: i64 = @intCast(top5.idx[0]);
```

Le reste (in_gen_phase, gen_top5.append, cache swap 4 buffers, deinit de `r_t5v`/`r_t5i`/
`tok_buf`/`step_buf`/args/results, progression, phases prefill/gén, early-stop) : INCHANGÉ
structurellement — adapter les noms.

- [ ] **Step 2.7 : perf séparée prefill/génération** (spec [it.6]). Capturer
  `t_prefill_end: std.Io.Timestamp` au moment où `step + 1 == ids.items.len` (fin du dernier
  step de prefill), puis remplacer le log A1 PERF par :

```zig
    const pf_s = <durée t0→t_prefill_end en s>;
    const gen_s = elapsed_s - pf_s;
    const gen_rate = if (gen_s > 0) @as(f64, @floatFromInt(generated.items.len)) / gen_s else 0;
    log.info("L3 PERF : prefill {d} steps en {d:.3}s ({d:.1} tok/s) ; génération {d} tokens en {d:.3}s ({d:.1} tok/s)", .{ ids.items.len, pf_s, ..., generated.items.len, gen_s, gen_rate });
```

- [ ] **Step 2.8 :** `MIN_FREE_VRAM_GIB` : `10` → `16` + commentaire « provisoire L3, seuil
  final = ceil(mesuré/0.90)+1 en G3, spec L3 §3 ». Mettre à jour le commentaire d'en-tête CLI
  (mention L3) et le commentaire de la garde si besoin.

- [ ] **Step 2.9 : relire le diff complet, puis commit** :

```bash
git add zml_runner/gemma4_gen_auto.zig
git commit -m "feat(gen-auto): L3 in-graph — StepTok (gather+forwardStep+topK), Tabs via TensorStore, boucle scalaire, perf prefill/gén séparée (spec docs/L3_INGRAPH_DESIGN.md)"
```

## Task 3 : deploy + build TÔT (fail-fast piège comptime)

- [ ] **Step 3.1 :** `ZML_REMOTE=user@gpu-host ZML_DST=/data/rqz_workspace/zml/examples/rqz ./deploy_to_3090.sh` — vérifier `Deployed`.
- [ ] **Step 3.2 :** `ssh user@gpu-host 'cd /data/rqz_workspace/zml && ./bazel.sh build --@zml//platforms:cuda=true //examples/rqz:gemma4_gen_auto'` (timeout généreux).
  Si erreur « evaluation exceeded 1000 backwards branches » dans `pjrt.zig structSize` :
  c'est le piège comptime documenté — vérifier que le patch local `@setEvalBranchQuota(100_000)`
  est présent dans `pjrt.zig` du workspace (resync upstream l'efface), et raccourcir les noms
  de types si besoin. Si erreur gather/tags : appliquer les replis du Step 2.3. Autre erreur
  de compilation : reporter BLOCKED avec le diagnostic complet, NE PAS improviser.

## Task 4 : SG — `--selftest-gather` converti au gather in-graph

- [ ] **Step 4.1 :** réécrire `selftestGather` : charger `Tabs` + `Model` PAS nécessaire — un
  mini-struct `SgFwd` local suffit :

```zig
// SG (spec [it.4]) : selftest du gather IN-GRAPH — mêmes données (fixture A1), même critère
// bit-exact que l'ancien selftest host. Mode GPU désormais (garde VRAM applicable, assumé).
const SgFwd = struct {
    pub fn forward(emb: zml.Tensor, eptl: zml.Tensor, tok: zml.Tensor) struct { zml.Tensor, zml.Tensor } {
        return .{ emb.gather(.{ .voc = tok }), eptl.gather(.{ .voc = tok }) };
    }
};
```

Corps : ouvrir le checkpoint (TensorStore), créer les 2 tenseurs (`EMB_KEY` split : la view
withPrefix + `embed_tokens.weight` / `embed_tokens_per_layer.weight`), les matérialiser
(`zml.io.load` sur un struct anonyme ou 2 champs), compiler `SgFwd.forward`, puis pour chaque
step de la fixture : feeder `fed[step]`, D2H, comparer les BITS (u16) aux lignes
`embeds`/`embptls` de la fixture via `readFixtureAlloc(u16, .bf16, ...)` (fonction existante).
Log final : `SG PASS — {N} steps × 2 tables bit-exact (gather in-graph)`.

- [ ] **Step 4.2 :** deploy + build (mêmes commandes que Task 3).
- [ ] **Step 4.3 :** run `--selftest-gather <fixture A1>` sur la 3090 → `SG PASS` attendu,
  bit-exact. FAIL = STOP, diagnostiquer avant les gates suivants.
- [ ] **Step 4.4 : commit** `feat(gen-auto): SG — selftest-gather converti au gather in-graph (bit-exact vs fixture A1)`.

## Task 5 : gates G1 (fidélité courte) + G1v (non-vacuité)

- [ ] **Step 5.1 : G1** — run `--oracle <fixture A1 48>` : attendu `A1 PASS — 48/48` + nouveau
  log `L3 PERF` (noter les chiffres). FAIL : lire le diagnostic top5 in-graph au step fautif ;
  si marge `val[0]-val[1]` minuscule → suspecter les ties argmax (spec §4), reporter avec les
  chiffres, décision contrôleur.
- [ ] **Step 5.2 : G1v** — corrompre une copie de la fixture (changer 1 valeur de `fed`,
  script python une ligne sur la 3090, fixture copiée dans /tmp du remote) → run --oracle
  dessus : attendu **A1 FAIL** au step corrompu. Un PASS ici = compare vacueux = STOP.
- [ ] **Step 5.3 :** archiver les sorties → `logs/l3_g1.log` (local, gitignored — référence doc).

## Task 6 : gate G2 — early-stop réel

- [ ] **Step 6.1 :** run `--prompt "What is the capital of France? Answer in one word."` →
  « Paris », early-stop EOT, exit 0.
- [ ] **Step 6.2 :** run prompt libre FR (celui du B0 Step 1.3) → réponse cohérente, early-stop,
  noter `L3 PERF`.

## Task 7 : gate G2b — fidélité longue différentielle

- [ ] **Step 7.1 :** retrouver la fixture longue A2 (~1000 steps) : `ls /data/gemma4-zml-probe/fixtures/`
  + `grep -n "A2" docs/GEN_AUTONOME_PLAN.md` (le nom + le point de bifurcation publié : step
  ~590, marge 0.006).
- [ ] **Step 7.2 :** run `--oracle <fixture A2>` : critère **différentiel** — PASS si la
  bifurcation survient AU MÊME step (~590) ou plus tard que le replay documenté (jamais N/N
  exigé, spec §4). Noter le step exact + la marge top5 (`val[0]-val[1]`) au step de bifurcation.
  Si bifurcation PLUS TÔT que le replay : reporter avec chiffres, décision contrôleur.
- [ ] **Step 7.3 :** archiver → `logs/l3_g2b.log` (local).

## Task 8 : G3 — bench après + mesure VRAM → seuil final + VG

- [ ] **Step 8.1 :** relever `mem_probe` post-load et post-compile dans les logs des runs Task 5/6
  (déjà loggé à chaque run). Mesure de référence = post-compile (pic résident).
- [ ] **Step 8.2 :** seuil final `MIN_FREE_VRAM_GIB = ceil(mesuré_GiB / 0.90) + 1`. Si ≠ 16 :
  éditer la constante + commentaire (avec le chiffre mesuré), redeploy + rebuild.
- [ ] **Step 8.3 : VG** — re-run V1 de la garde (Ollama chargé → `error.GpuBusy` au NOUVEAU
  seuil) et V3 (GPU libre → run normal silencieux). Attendus identiques à
  `VRAM_CHECK_DESIGN.md` §6.
- [ ] **Step 8.4 : comparaison B0 → G3** : tableau (oracle 48 : X → Y tok/s ; libre FR :
  X → Y tok/s génération). Critère : **L3 ≥ B0**. Attente non bloquante ≥ 109 tok/s. Si le
  tri topK pèse visiblement (gén < B0) : appliquer le repli argMax (tête de plan) et re-mesurer.
- [ ] **Step 8.5 : commit** (code seuil éventuel) `feat(gen-auto): G3 — seuil VRAM final mesuré ({X} GiB post-compile)`.

## Task 9 : documentation + solde + PR

- [ ] **Step 9.1 :** `docs/L3_INGRAPH_DESIGN.md` : section « Résultats » (tableau des gates,
  chiffres B0/G3, seuil final, point de bifurcation G2b) — même format que VRAM_CHECK_DESIGN §6.
- [ ] **Step 9.2 :** `docs/DOCUMENTATION.md` §2.2 : usage inchangé, perf mise à jour (chiffres
  G3, prefill/gén séparés), VRAM ~{mesuré} Go, renvoi spec L3. § backlog : L3 soldé.
- [ ] **Step 9.3 :** `docs/VRAM_CHECK_DESIGN.md` : seuil final (errata L3 daté) +
  `--selftest-gather` reclassé GPU (§1/§4).
- [ ] **Step 9.4 :** note « 12 Go » G2.1 : caveat une ligne dans `docs/G2_BF16_FIDELITY.md`
  (§ découverte G2.1) : « post-L3, gen_auto requiert ~{mesuré} Go — le banc 12 Go ne vaut que
  pour les runners sans tables d'embeddings device ».
- [ ] **Step 9.5 :** `PLANNING.md` : item [M] L3 → [x] avec résumé (gates, chiffres, tag).
- [ ] **Step 9.6 : commit + tag + PR** :

```bash
git add docs/ PLANNING.md
git commit -m "docs: L3 in-graph livré — résultats gates, DOCUMENTATION §2.2, seuils, PLANNING"
git tag gate/l3-ingraph-pass
git push -u origin l3-ingraph && git push origin gate/l3-ingraph-pass
gh pr create --title "L3 in-graph : gemma4_gen_auto token→token" --body "<résumé + tableau gates>"
```

Merge = décision Régis.
