# Garde VRAM `gemma4_gen_auto` — Plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal :** `gemma4_gen_auto` refuse de démarrer avec un message explicite (`error.GpuBusy`,
process occupants listés) quand la VRAM libre est sous 10 GiB, au lieu de l'OOM cryptique +
crash error-path upstream ZML. Spec approuvée : `docs/VRAM_CHECK_DESIGN.md`.

**Architecture :** deux fonctions ajoutées à `gemma4_gen_auto.zig` — `parseFreeMiB` (parsing pur,
testable) et `checkVram` (spawn `nvidia-smi` via `std.process.run`, best-effort : tout échec de
l'outil → warn + continue) — appelées dans `main` après les early-returns host-only, avant tout
travail GPU. Nouveau flag `--force-vram`. Aucun autre runner touché, `BUILD.bazel` inchangé.

**Tech stack :** Zig 0.16.0-dev.2722 (toolchain hermétique rules_zig du workspace ZML de la 3090 —
**API vérifiée le 11 juil 2026** : `std.process.run(gpa, io, .{ .argv = … }) → RunResult{ term,
stdout, stderr }`, `Term = union(enum){ exited: u8, signal, stopped, unknown }`, résolution
d'`argv[0]` via le PATH parent). Build/run : Bazel sur la 3090 uniquement (pas de toolchain local
M1) — d'où validation par gates réels V1-V3, pas de zig_test (aucune infra test dans
`zml_runner/BUILD.bazel`).

**Convention accès distant :** les commandes ci-dessous utilisent les placeholders du repo
(`user@gpu-host`, `/data/...`). Les vraies valeurs `ZML_REMOTE`/`ZML_DST` sont dans la mémoire
infra locale de Régis, à passer EN ENV au moment de l'exécution — ne jamais les committer
(convention anonymisation du repo). ⚠ Piège vécu 11 juil : sans `ZML_REMOTE`/`ZML_DST` exportés,
`deploy_to_3090.sh` échoue sur les placeholders et, sortie redirigée, on teste l'ANCIEN binaire.

---

## Task 1 : flag `--force-vram` + fonctions de garde + insertion dans `main`

> Branche de travail : **`vram-check`** (existe déjà — la spec y est committée). Vérifier
> `git branch --show-current` avant le premier commit ; si sur `main`, `git checkout vram-check`.

**Files:**
- Modify: `zml_runner/gemma4_gen_auto.zig` (Args ~l.56-71, parseArgs ~l.114, nouvelles fns avant
  `main` ~l.684, insertion dans `main` après le bloc `--ids-only` ~l.769)

- [ ] **Step 1.1 : ajouter le flag aux Args + usage**

Dans le struct `Args` (après `allow_cpu: bool = false,`) :

```zig
    force_vram: bool = false,
```

Dans la constante `usage`, après `[--allow-cpu (débogage uniquement)] ` :

```zig
    "[--force-vram] " ++
```

Dans `parseArgs`, après la branche `--allow-cpu` :

```zig
        } else if (std.mem.eql(u8, a, "--force-vram")) {
            args.force_vram = true;
```

- [ ] **Step 1.2 : constante seuil + `parseFreeMiB` + `checkVram`**

Insérer AVANT `pub fn main` (vers l.684), au niveau module :

```zig
// ============================================================================================
// Garde VRAM au lancement (docs/VRAM_CHECK_DESIGN.md) — incident du 11 juil 2026 : Ollama à
// ~22/24 Go → OOM dès la matérialisation + crash `io.zig deinit` (double-free post-OOM, bug
// d'error-path UPSTREAM ZML, cosmétique — l'OOM est la vraie erreur). Best-effort : la garde ne
// bloque JAMAIS à tort — nvidia-smi absent/cassé/illisible → warn + continue (l'OOM reste le
// filet) ; seul « VRAM libre < seuil » mesuré avec succès fait échouer le lancement.
// ============================================================================================

// Seuil requis : mesure G2.1 = 8,5 GiB réels pour ce modèle (poids déjà bf16 on-device,
// cf docs/G2_BF16_FIDELITY.md). Marge honnête : le binaire initialise CUDA avec BFC
// `memory_fraction 0.90` → à 10 GiB libres, réserve utilisable ≈ 9 GiB, soit ~0,5 GiB
// au-dessus des 8,5 mesurés (pas 1,5). Pas de flag de réglage (YAGNI).
const MIN_FREE_VRAM_GIB: u64 = 10;

// Parse `nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits` : première ligne =
// GPU 0 (VM mono-GPU), entier en MiB. null = sortie illisible (l'appelant warn + continue).
fn parseFreeMiB(stdout: []const u8) ?u64 {
    var lines = std.mem.splitScalar(u8, stdout, '\n');
    const first = lines.next() orelse return null;
    const trimmed = std.mem.trim(u8, first, " \t\r");
    if (trimmed.len == 0) return null;
    return std.fmt.parseInt(u64, trimmed, 10) catch null;
}

fn checkVram(gpa: std.mem.Allocator, io: std.Io) !void {
    const res = std.process.run(gpa, io, .{
        .argv = &.{ "nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits" },
    }) catch |err| {
        log.warn("garde VRAM sautée : nvidia-smi indisponible ({s}) — machine sans GPU ?", .{@errorName(err)});
        return;
    };
    defer gpa.free(res.stdout);
    defer gpa.free(res.stderr);
    switch (res.term) {
        .exited => |code| if (code != 0) {
            log.warn("garde VRAM sautée : nvidia-smi exit={d}", .{code});
            return;
        },
        else => {
            log.warn("garde VRAM sautée : nvidia-smi terminé anormalement", .{});
            return;
        },
    }
    const free_mib = parseFreeMiB(res.stdout) orelse {
        log.warn("garde VRAM sautée : sortie nvidia-smi illisible", .{});
        return;
    };
    if (free_mib >= MIN_FREE_VRAM_GIB * 1024) return;

    // Une décimale en arithmétique ENTIÈRE (pas de format float : API std.fmt 0.16-dev mouvante).
    const gib10 = free_mib * 10 / 1024;
    log.err("GPU occupé — VRAM libre {d}.{d} GiB < {d} GiB requis", .{ gib10 / 10, gib10 % 10, MIN_FREE_VRAM_GIB });
    // Déviation assumée vs spec §2 : pas de `parseComputeApps` structuré — les lignes CSV brutes
    // trimées suffisent au message (PID, nom, MiB lisibles) et restent best-effort.
    if (std.process.run(gpa, io, .{
        .argv = &.{ "nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader" },
    })) |apps| {
        defer gpa.free(apps.stdout);
        defer gpa.free(apps.stderr);
        var it = std.mem.splitScalar(u8, apps.stdout, '\n');
        while (it.next()) |line| {
            const l = std.mem.trim(u8, line, " \t\r");
            if (l.len != 0) log.err("  {s}", .{l});
        }
    } else |err| {
        log.warn("liste des process compute indisponible ({s})", .{@errorName(err)});
    }
    log.err("Libérer d'abord : `ollama ps` puis `ollama stop <modèle>` (réversible), ou --force-vram pour tenter quand même", .{});
    return error.GpuBusy;
}
```

- [ ] **Step 1.3 : insertion dans `main`**

Juste APRÈS la fermeture du bloc `if (args.ids_only) { … return; }` (l.769) et AVANT le
commentaire `// === --oracle : …` :

```zig
    // === Garde VRAM (docs/VRAM_CHECK_DESIGN.md) — avant tout travail GPU. Les modes host-only
    // (--selftest-inputs/--selftest-gather/--ids-only) ont déjà early-return au-dessus. Tourne
    // AUSSI en --allow-cpu : ce flag ne force pas le CPU (l'init .cuda est tentée d'abord,
    // --allow-cpu ne tolère que le repli) — sur machine sans GPU, nvidia-smi absent → warn +
    // continue, donc pas de blocage à tort. Seul --force-vram saute la garde. ===
    if (args.force_vram) {
        log.warn("--force-vram : garde VRAM sautée (OOM possible en aval, assumé)", .{});
    } else {
        try checkVram(allocator, io);
    }
```

- [ ] **Step 1.4 : mettre à jour le commentaire CLI d'en-tête de fichier** (l.5-6) : ajouter
  `[--force-vram]` à la ligne d'usage.

- [ ] **Step 1.5 : commit**

```bash
cd ~/dev/gemma4-zml-probe
git add zml_runner/gemma4_gen_auto.zig
git commit -m "feat(gen-auto): garde VRAM au lancement — error.GpuBusy explicite si libre < 10 GiB, --force-vram (spec docs/VRAM_CHECK_DESIGN.md)"
```

## Task 2 : deploy + build 3090

- [ ] **Step 2.1 : deploy** (vraies valeurs en env, cf convention en tête) :

```bash
cd ~/dev/gemma4-zml-probe/zml_runner
ZML_REMOTE=user@gpu-host ZML_DST=/data/rqz_workspace/zml/examples/rqz ./deploy_to_3090.sh
```

Attendu : liste rsync incluant `gemma4_gen_auto.zig`, dernière ligne `Deployed … -> …`.
**Vérifier cette ligne** (piège du deploy silencieusement raté).

- [ ] **Step 2.2 : build**

```bash
ssh user@gpu-host 'cd /data/rqz_workspace/zml && ./bazel.sh build --@zml//platforms:cuda=true //examples/rqz:gemma4_gen_auto'
```

Attendu : `Build completed successfully`. En cas d'erreur de compilation sur l'API
`std.process.run` : relire `lib/std/process.zig` du toolchain hermétique (chemin dans la note
tech-stack) — l'API a été vérifiée mais c'est une dev-version.

## Task 3 : gate V1 — la garde mord (GPU occupé)

- [ ] **Step 3.1 : occuper le GPU.** `ssh user@gpu-host 'ollama ps'` — si vide, charger le modèle
  résident habituel (`ollama run <modèle> "ping"` puis vérifier). Confirmer
  `nvidia-smi --query-gpu=memory.free --format=csv` < 10 GiB libres.

- [ ] **Step 3.2 : run réel** (binaire depuis `bazel-bin`, PAS `bazel.sh run` — sortie plus propre) :

```bash
ssh user@gpu-host 'cd /data/rqz_workspace/zml && ./bazel-bin/examples/rqz/gemma4_gen_auto \
  /data/gemma4-zml-probe/weights/model.safetensors \
  /data/gemma4-zml-probe/weights/tokenizer.json \
  --prompt "ping" 2>&1' | tee /tmp/vram_v1.log
```

Attendu : `GPU occupé — VRAM libre X.Y GiB < 10 GiB requis`, la/les ligne(s) process (PID, nom,
MiB), la suggestion `ollama stop`, exit `error.GpuBusy`. **Interdit dans la sortie** : OOM
(`CreateBuffersForAsyncHostToDevice`), crash `General protection exception`.
NB : le chemin réel de `tokenizer.json` sur la 3090 fait foi (cf run du 11 juil) — l'ajuster si besoin.

- [ ] **Step 3.3 : archiver** `cp /tmp/vram_v1.log ~/dev/gemma4-zml-probe/logs/vram_check_v1.log`
  (convention logs/ du repo).

## Task 4 : gate V2 — échappatoire `--force-vram`

- [ ] **Step 4.1 : même état (GPU occupé) :**

```bash
ssh user@gpu-host 'cd /data/rqz_workspace/zml && ./bazel-bin/examples/rqz/gemma4_gen_auto \
  /data/gemma4-zml-probe/weights/model.safetensors \
  /data/gemma4-zml-probe/weights/tokenizer.json \
  --prompt "ping" --force-vram 2>&1' | tee /tmp/vram_v2.log
```

Attendu : warn
  `--force-vram : garde VRAM sautée …`, PUIS l'OOM assumé peut suivre (c'est le contrat — le crash
  error-path upstream connu peut suivre l'OOM, il est cosmétique, ne pas le « corriger »).
  Archiver → `logs/vram_check_v2.log`.

## Task 5 : gate V3 — non-régression (GPU libre)

- [ ] **Step 5.1 : libérer** : `ssh user@gpu-host 'ollama stop <modèle>'` (celui vu par `ollama ps`),
  vérifier > 10 GiB libres.

- [ ] **Step 5.2 : run normal** — même commande que V1 avec
  `--prompt "What is the capital of France? Answer in one word."` (sans `--force-vram`).
  Attendu : la garde passe SILENCIEUSEMENT (aucune ligne VRAM), génération normale, early-stop
  EOT, « Paris » sur stdout (référence : DOCUMENTATION.md §2.2). Archiver → `logs/vram_check_v3.log`.

- [ ] **Step 5.3 : consigner + tag** — amendement en cours d'exécution : `logs/` est ENTIÈREMENT
  gitignored dans ce repo (aucun log de gate n'a jamais été tracké — les logs restent locaux, la
  doc porte les résultats). Donc : PAS de `git add logs/…` ; ajouter une section « Résultats
  (11 juil 2026) » à `docs/VRAM_CHECK_DESIGN.md` §6 (une ligne par gate : état initial mesuré,
  verdict, référence du log local), committée avec la doc en Task 6. Le tag reste :

```bash
cd ~/dev/gemma4-zml-probe
git tag gate/vram-check-pass   # après le commit doc de la Task 6
```

## Task 6 : documentation + solde PLANNING

**Files:**
- Modify: `docs/DOCUMENTATION.md` (§2.2, le ⚠ VRAM l.92-95)
- Modify: `PLANNING.md` (item [B] l.39-41 → [x] ; garde-fou « Contention VRAM 3090 » l.61-66)

- [ ] **Step 6.1 : DOCUMENTATION.md §2.2** — remplacer le paragraphe
  `⚠ **Vérifier la VRAM avant de lancer** : …` par :

```markdown
⚠ **VRAM** : le GPU peut être occupé par un autre service local (ex. Ollama, ~22 Go).
`gemma4_gen_auto` refuse alors de démarrer : garde intégrée au lancement (`error.GpuBusy`
si VRAM libre < 10 GiB, process occupants listés, cf `docs/VRAM_CHECK_DESIGN.md`),
échappatoire `--force-vram`. Libérer : `ollama ps` puis `ollama stop <modèle>` (réversible —
rechargé à la demande). Garde best-effort (nvidia-smi absent/cassé → warn + continue) et
propre à `gemma4_gen_auto` — pour les AUTRES runners GPU, vérifier à la main.
```

- [ ] **Step 6.2 : PLANNING.md** — passer l'item `[B] Check VRAM…` en `[x]` avec une ligne de
  résumé (gates V1-V3 PASS, tag `gate/vram-check-pass`) ; dans le garde-fou « Contention VRAM
  3090 », noter que `gemma4_gen_auto` a désormais la garde intégrée (le réflexe `nvidia-smi`
  manuel reste valable pour les autres runners).

- [ ] **Step 6.3 : commit**

```bash
git add docs/DOCUMENTATION.md PLANNING.md
git commit -m "docs: garde VRAM gen_auto livrée — DOCUMENTATION §2.2 + PLANNING item [B] soldé"
```

## Task 7 : PR vers main

- [ ] **Step 7.1 :** `git push -u origin vram-check && git push --tags`, puis
  `gh pr create --title "Garde VRAM au lancement de gemma4_gen_auto (item [B])" --body …`
  (résumé : spec, gates V1-V3, périmètre gen_auto seul). Merge = décision Régis
  (convention repo : PR reviewée avant main).
