# Spec — D10 : zéro allocation par step dans le décodage 12B (rév. 2)

> **Statut : SPEC — committée AVANT toute mesure** (exigence de falsifiabilité, le git log fait foi).
> Branche : `s2-d10-alloc-par-step`. Chantier ouvert sur décision Régis du 30 juil 2026 (niveau de
> travail : meute, via ultracode).
>
> Dette d'origine : `docs/SAMPLING_RESULTS.md` §5, **D10** — « `toSliceAlloc` alloue 1 Mo par
> step », violant l'interdit §5 de la spec sampling (« aucune allocation par step »), lui-même
> déclaré **sans gate porteur** (dette D3). Ce chantier corrige la violation ET pose le gate,
> conformément à l'arbitrage **B9** (« compteur d'allocations à 0, pas le débit ») et **B10**
> (critère RSS chiffré) acceptés le 29 juil et jamais câblés.
>
> **Historique** — rév. 1 : `7a1437b`. **Rév. 2 après revue adversariale (4 relecteurs, 31
> findings, 3 BLOQUANTS)**. Ce que la revue a attrapé : EQ-PONT greppait une ligne que personne
> n'émet (contrôle qui ne peut pas réussir) · la commande de build §8 produisait un frontend Zig
> en Debug, rendant TOUS les gates inexécutables par la règle du §8 lui-même · AL-RSS n'avait
> aucun véhicule (le QUOI sans le COMMENT) · le « repli automatique » de C7 n'existait pas dans
> l'API et son porteur §6 était un `catch unreachable` = UB en ReleaseFast, pas un panic · la
> borne C5 était fausse en `--oracle` et `appendAssumeCapacity` y est une UB silencieuse · le
> témoin 124 de P4 n'avait pas de gate porteur · V-EQ pouvait FAIL à tort sur le jitter de
> compile (`max_abs` flottant dans la ligne diffée).

## 0. Décisions de cadrage (Régis, 30 juil 2026)

| # | Question | Décision |
|---|---|---|
| K1 | Périmètre | **Boucle de génération entière + gate compteur à 0** — pas seulement le chemin B. Implique de hisser des allocations **antérieures** au chantier sampling (`exe.args`/`exe.results`, top-5, ArrayLists) |
| K2 | Mémoire pinned | **Incluse, en expérience mesurée** (M-PIN) — claim falsifiable avec repli propre, pas un gate |
| K3 | Boucle `--window-vacuity` | **Corrigée aussi**, avec re-validation de l'instrument contre témoin frais (V-EQ) |

Invariants imposés : **falsifiabilité** (prédiction chiffrée + ce qui la tue, écrites ici, avant
mesure) et **modularité** (chaque composant isolé et testable seul).

## 1. État mesuré — les faits qui fondent le chantier

Réconciliation d'unités (leçon « raffiner un chiffre faux ») : **1 step ≈ 106 000 µs ⇔ 9,4 tok/s**
(run M-COUT, n = 88). Toute fraction ci-dessous se rapporte à cette base.

1. **F1 — D10 sous-compte d'un facteur 2, et il y a 3 parcours d'1 MiB au lieu de 0.**
   `toSliceAlloc` (zml `buffer.zig:282-301` : allocs `:283` et `:289`) fait DEUX allocations par
   appel (le résultat + un staging par shard) plus un memcpy interne (`:297`), avant le
   `@memcpy(scfg.work, lg)` du runner (`gemma4_g12auto.zig:2145`). Par step armé : **2 MiB
   alloués/libérés + 3 parcours d'1 MiB**.
2. **F2 — Inventaire exhaustif de la boucle de génération** (`gemma4_g12auto.zig:2077-2271`),
   par step, chemin B armé : **12 appels d'allocation host, ≈ 2 397 096 octets** (le chiffre
   exact d'octets est une arithmétique de lecture — `@sizeOf(zml.Shape)` = 288 calculé, jamais
   imprimé — le baseline AL-BASE le confirmera ou le corrigera) :
   - `exe.args(allocator)` ligne 2090 : 3 allocs, ≈ 297 784 o (dont `dupe` de 1006 `Shape`) ;
   - `exe.results(allocator)` ligne 2091 : 3 allocs, ≈ 2 080 o ;
   - top-5 `toSliceAlloc` ×2 lignes 2104/2106 : 4 allocs, 80 o ;
   - chemin B `toSliceAlloc` ligne 2138 : 2 allocs, 2 097 152 o (**87,5 % des octets, 16,7 % des
     appels**) ;
   - `gen_top5.append` (2193) et `generated.append` (2244) : croissances amorties.
   Tout sauf la ligne 2138 est **antérieur au chantier sampling** (vérifié par `git log -S`).
3. **F3 — La voie sans allocation existe en API publique ZML** : `Buffer.toSlice(io, slice)`
   (`buffer.zig:259`) écrit directement dans une destination fournie, sans allocateur ;
   `Slice.init(shape, bytes)` (`slice.zig:29`) emprunte un `[]u8` existant sans copie. Les 3
   sessions LLM amont de ZML utilisent ce motif dans leur boucle de décode
   (`examples/llm/models/llama/session.zig:49-50` — Slice allouée une fois à l'init — et `:210`,
   `toSlice` par step ; idem qwen3_5 et lfm2).
4. **F4 — Les gardes de cette voie sont actives dans TOUS les modes de build** :
   `stdx.debug.assert` est inconditionnel (`stdx/debug.zig:11-15`). `Shape.eql` ne compare que
   dims + dtype, PAS les tags (`shape.zig:501-503`) → une destination construite avec
   `r_logits.shape()` passe par construction. Un viol panique au premier run, jamais silencieux.
   ⚠ Cette garantie vaut pour `stdx.debug.assert` (ZML) — **pas** pour `std.debug.assert` de la
   std Zig, qui est de l'UB en ReleaseFast : aucun composant de ce chantier ne doit s'appuyer
   sur `std.debug.assert` comme porteur (cf. C5).
5. **F5 — `exe.args`/`results` sont hissables** : seuls leurs `init` allouent ; `set()`
   (`exe.zig:129-146`), `call()` et `get()` n'allouent rien (lu dans `exe.zig:66-121`, `:218-236`,
   `:259-299`). `Buffer.fromBytes` et `Buffer.scalar` (H2D du token, chaque step) n'allouent
   **rien côté Zig** (`buffer.zig:92-174` : `Slice.init` + structures par valeur + PJRT côté C).
   **Jamais compilé ni exécuté : c'est la prédiction P2, pas un acquis.**
6. **F6 — Coût actuel du bloc chemin B** (build opt, D9) : total 3 796 µs = 3,6 % d'un step,
   décomposé **D2H+alloc+copie 1 514 µs** / **warpers 2 268 µs** / **~14 µs de sélection et
   comptage non décomposés** (1 514 + 2 268 = 3 782 ≠ 3 796 : le résidu est nommé pour que
   M-D10 ne reproduise pas un total qui ne somme pas sans explication). Le chrono `t_d2h`
   agrège transfert, allocations, memcpy et page faults — la part de chacun est inconnue
   (c'est M-D10 qui la départage).
7. **F7 — Deuxième site** : la boucle `--window-vacuity` (`gemma4_g12auto.zig:1889-1936`) contient
   les mêmes motifs (`exe.args`/`results` + `r_logits.toSliceAlloc` ligne 1901), antérieurs au
   chantier (commit `c2211c0`). C'est un **instrument** (replay U9-ii), pas le chemin chaud.
8. **F8 — L'identité de l'allocateur `init.gpa` dépend du mode de build** (std `start.zig:668-692`) :
   Debug ⇒ `DebugAllocator` ; ReleaseFast ⇒ `c_allocator` si libc, sinon `smp_allocator`. La
   commande exacte du build « opt » de D9 est **perdue**, et le mode Zig de rules_zig est un
   flag Bazel **indépendant** de `-c` dont le défaut est `debug` (`.bazelrc:53` et `:106-108` :
   `--config=debug` met le backend en opt et le frontend Zig en debug). D'où le §8 : la commande
   de build porte les DEUX flags, et le binaire publie `builtin.mode`.

## 2. Périmètre

**Dedans** : la boucle de génération de `generateOnce` (`:2077-2271`) et la boucle vacuity
(`:1889-1936`) du seul runner `gemma4_g12auto.zig` (et ses variantes 4k/8k qui compilent le même
fichier — la 4k est exercée par EQ-124, la 8k reste dette DA-4) ; le module compteur ;
l'expérience pinned sur `scfg.work` ; la sonde RSS (C8) ; les commentaires périmés attenants
(en-tête CLI lignes 25-26 « 2 des 8 clés », ligne 2094 « NON lue ici ») ; le libellé M-COUT
« D2H+copie » → « D2H seul » (dans le même commit que C2, sinon la ligne publiée devient
mensongère) ; un errata d'une ligne dans `docs/superpowers/plans/2026-07-29-sampling-phase2.md`
(base de coût réfutée 1,57 %/177,95 % → renvoi à la spec rév. 4).

**Hors périmètre, écrit** : E2B (logits hors graphe) · batching · phase 1 penalty (suspendue) ·
dette D1 (`applyTopP` sans couverture GPU) · streaming · les allocations **C/PJRT internes**
(invisibles à l'allocateur Zig — dette DA-1, partiellement couverte par AL-RSS) · les
allocations internes de `std.Io.Threaded` (DA-6) · les autres runners du dépôt.

## 3. Design — composants (modularité : chacun isolé, testable seul)

### C1 — `alloc_count.zig` : compteur d'allocations, module std-only

Nouveau fichier `zml_runner/alloc_count.zig`, **sans aucune dépendance ZML** (même règle que
`sampling.zig`), ajouté aux `srcs` des TROIS cibles qui compilent le runner (`gemma4_g12auto`,
`gemma4_g12a4k`, `gemma4_g12a8k` — `BUILD.bazel`).

```zig
//! Compteur d'allocations — wrapper transparent de std.mem.Allocator.
//! Les 4 fonctions vtable délèguent au parent et comptent. Les corps suivent
//! std.testing.FailingAllocator de CE toolchain (0.16.0-dev.2722), signatures copiées.
//! ⚠ Compteurs u64 NON atomiques : corrects parce que seul le thread principal passe par ce
//! wrapper (std.Io.Threaded est construit sur le gpa non wrappé AVANT run() — dette DA-6).
const std = @import("std");

pub const CountingAllocator = struct {
    parent: std.mem.Allocator,
    n_alloc: u64 = 0,
    n_resize: u64 = 0,
    n_remap: u64 = 0,
    n_free: u64 = 0,
    bytes_alloc: u64 = 0,

    pub fn init(parent: std.mem.Allocator) CountingAllocator {
        return .{ .parent = parent };
    }

    pub fn allocator(self: *CountingAllocator) std.mem.Allocator {
        return .{ .ptr = self, .vtable = &.{ .alloc = alloc, .resize = resize, .remap = remap, .free = free } };
    }

    /// Somme des compteurs d'ACQUISITION (alloc + resize + remap) : c'est le delta
    /// qu'observent les gates. `free` est publié mais hors somme (libérer n'acquiert pas).
    pub fn calls(self: *const CountingAllocator) u64 {
        return self.n_alloc + self.n_resize + self.n_remap;
    }

    fn alloc(ctx: *anyopaque, len: usize, alignment: std.mem.Alignment, ret_addr: usize) ?[*]u8 {
        const self: *CountingAllocator = @ptrCast(@alignCast(ctx));
        self.n_alloc += 1;
        self.bytes_alloc += len;
        return self.parent.rawAlloc(len, alignment, ret_addr);
    }
    fn resize(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) bool {
        const self: *CountingAllocator = @ptrCast(@alignCast(ctx));
        self.n_resize += 1;
        return self.parent.rawResize(memory, alignment, new_len, ret_addr);
    }
    fn remap(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) ?[*]u8 {
        const self: *CountingAllocator = @ptrCast(@alignCast(ctx));
        self.n_remap += 1;
        return self.parent.rawRemap(memory, alignment, new_len, ret_addr);
    }
    fn free(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, ret_addr: usize) void {
        const self: *CountingAllocator = @ptrCast(@alignCast(ctx));
        self.n_free += 1;
        return self.parent.rawFree(memory, alignment, ret_addr);
    }
};
```

⚠ Les signatures vtable ci-dessus sont celles LUES dans la std du toolchain (4 fonctions,
`std.mem.Alignment`) — la revue technique les a re-confirmées contre `mem/Allocator.zig`. Le
compteur est **toujours actif** (un incrément par appel — et zéro incrément par step une fois le
chantier fini, par définition).

Installation : dans `run()` (`gemma4_g12auto.zig:1509`), `init.gpa` est remplacé par le wrapper —
**point de substitution unique**, aucun autre changement de signature aval que le paramètre
`*CountingAllocator` passé à `generateOnce` (même traitement par pointeur que `scfg`, mêmes
3 sites d'appel).

Câblage du selftest : flag booléen `--selftest-alloc-count` ajouté à `parseArgs` et à l'usage,
inscrit dans la **garde d'exclusivité `--repl`** (`:1518-1525`, liste des selftests), early-return
**host-only** avant tokenizer, garde VRAM et Platform — même patron que `--selftest-sampling`
(`:1543-1546`).

### C2 — Chemin B : `toSlice` directement dans `scfg.work` (le cœur de D10)

Remplace `toSliceAlloc` + check longueur + `@memcpy` (lignes 2138-2145) par :

```zig
// Garde en ACCEPTATION (avant l'assert de Slice.init, pour une erreur propre) :
const need = r_logits.shape().byteSize();
if (need != scfg.work.len * @sizeOf(f32)) {
    log.err("chemin B : logits {d} octets != work {d} octets", .{ need, scfg.work.len * @sizeOf(f32) });
    return error.UnexpectedShape;
}
try r_logits.toSlice(io, zml.Slice.init(r_logits.shape(), std.mem.sliceAsBytes(scfg.work)));
```

Le D2H écrit **directement** dans `work` : 0 allocation, 0 copie intermédiaire. `Slice.init` est
une construction par valeur (aucun coût par step). Le chrono `t_d2h` mesure désormais le transfert
seul — c'est ce qui rend M-D10 interprétable — et son libellé publié passe de « D2H+copie » à
« D2H seul » dans le même commit.

### C3 — Top-5 : `getValue` vers la pile

Remplace les deux `toSliceAlloc` (2104/2106) par `getValue` (`buffer.zig:249`, délègue à `toSlice`,
zéro allocation) :

```zig
if (r_t5v.shape().dtype() != .f32) return error.UnexpectedDtype; // gardes conservées,
if (r_t5i.shape().dtype() != .i32) return error.UnexpectedDtype; // déplacées AVANT le D2H
const t5v: [gencfg.TOP_K]f32 = try r_t5v.getValue([gencfg.TOP_K]f32, io);
const t5i: [gencfg.TOP_K]i32 = try r_t5i.getValue([gencfg.TOP_K]i32, io);
```

`getValue` assert `byteSize == @sizeOf(T)` (20 octets) — actif dans tous les modes (F4,
`stdx.debug.assert`). Le check dtype est **renforcé** (le code actuel ne vérifiait que i32 ; f32
était supposé). L'ordre mémoire `[5]f32` ≡ `items(f32)` actuel (vérifié en revue).

### C4 — `exe.args`/`exe.results` hissés hors des deux boucles

`init` une fois AVANT la boucle, `set()`/`call()`/`get()` par step (aucun n'alloue — F5),
`deinit` après. S'applique à la boucle de génération ET à la boucle vacuity.

### C5 — ArrayLists : capacité réservée avant la boucle, garde ACTIVE

`generated` et `gen_top5` : `ensureTotalCapacity(allocator, borne)` avant la boucle, avec
**`borne = ids.len + limit`** où `limit` est la borne réelle des DEUX modes (variable existante,
`gemma4_g12auto.zig:2027` — en `--oracle` la boucle s'arrête sur la longueur de la fixture, PAS
sur `max_tokens` : une borne `ids.len + max_tokens` paniquerait ou corromprait sur les fixtures
historiques de 1 150 ids). Dans la boucle : **`appendBounded`** (`array_list.zig:912`, rend
`error.OutOfMemory` si plein) — garde **active dans tous les modes de build** et zéro allocation,
là où `appendAssumeCapacity` repose sur `std.debug.assert` = UB silencieuse en ReleaseFast (F4).
Un run `--oracle` court figure dans les runs post-correction (le compteur toujours actif
re-vérifie ce mode gratuitement — DA-2).

### C6 — Boucle vacuity (K3) : même motif, buffer local

Un buffer d'1 MiB alloué UNE FOIS avant les passes vacuity (local au mode), `toSlice` dedans,
`exe.args`/`results` hissés (C4), libération après. La **fenêtre de delta d'ALLOC-VAC couvre les
boucles de steps seulement** — l'alloc/free du buffer C6 et les `init`/`deinit` hissés sont HORS
fenêtre (mêmes bornes de principe qu'ALLOC-LOOP), sinon le gate FAIL par construction.
L'instrument est re-validé par V-EQ contre un témoin produit AVANT la modification (leçon
« témoins avant tout deploy »).

### C7 — Expérience pinned (K2, M-PIN) : le repli est ÉCRIT, la propriété est STRUCTURELLE

⚠ La revue a établi que ZML ne replie JAMAIS de lui-même : `DmaMapAllocator.alloc` rend `null`
si `dmaMap` échoue (`mem.zig:145-153`), ce qui remonte en `error.OutOfMemory` — le repli
appartient au runner, et le voici :

```zig
// Un booléen UNIQUE pilote la bannière ET la libération — la propriété « libéré par
// l'allocateur qui a alloué » devient STRUCTURELLE (un seul chemin possible), au lieu
// d'être « punie » par un panic non garanti (`catch unreachable` = UB en ReleaseFast).
var dma = zml.mem.DmaAllocator.init(allocator, &platform.devices[0]);
var pin_on = true;
scfg.work = dma.allocator().alloc(f32, VOCAB_CONTRACT) catch blk: {
    pin_on = false;
    break :blk try allocator.alloc(f32, VOCAB_CONTRACT);
};
// … à la libération (defer existant, adapté) :
if (pin_on) dma.allocator().free(scfg.work) else allocator.free(scfg.work);
```

Bannière `PIN: ON` / `PIN: OFF (DmaMap indisponible)` dérivée de `pin_on`. Les trois issues
(indisponible / sans gain / gain) sont publiables, aucune n'est un échec du chantier. Le flag
`--no-pin` force `pin_on = false` (véhicule de l'A/B de M-PIN). Détail d'API à fixer au plan :
`DmaAllocator.init(parent, device)` (`mem.zig:22-27`) — le runner nomme le device via la
Platform qu'il possède déjà.

### C8 — Sonde RSS (véhicule d'AL-RSS, arbitrage B10)

`mem_probe.rssKb()` (lit **VmRSS** — c'est VmRSS que B10 vise, pas un high-water mark ; le pic
process est posé au load/compile ~19 GiB et ne peut structurellement pas bouger dans la boucle)
échantillonné dans `generateOnce` aux **tokens GÉNÉRÉS 20 et 200** (prefill exclu), zéro
allocation (buffers de pile de `mem_probe`, vérifié). En fin de run :
`RSS-DELTA: <n> KiB (t20=<a> t200=<b>)` ; si le run n'atteint pas le token généré 200 :
`RSS-DELTA: INEXECUTABLE (run trop court)` — jamais l'absence silencieuse de ligne. Le run
porteur est **teacher-forcé** (`--oracle` sur la fixture témoin 200 de `rp0_witness/`, déjà sur
la VM) : 200 steps **garantis**, insensible à l'arrêt EOS et à la bistabilité.

### Publication (véhicule des gates)

- Bannière au démarrage : `BUILD: mode=<builtin.mode> pin=<ON|OFF>` ;
- Fin de `generateOnce` (donc par prompt en `--repl`) :
  `ALLOC-LOOP: alloc=<n> resize=<n> remap=<n> free=<n> bytes=<n> steps=<n>`
  (deltas des compteurs pris juste avant/après la boucle) ;
- Fin des passes vacuity : `ALLOC-VAC: alloc=<n> resize=<n> remap=<n> free=<n> bytes=<n> steps=<n>`
  (fenêtre : boucles de steps seulement, cf. C6) ;
- Fin de run : `ALLOC-TOTAL: alloc=<n> free=<n> bytes=<n> shards=<n>` — les compteurs PROCESS
  (non-vacuité d'AL-0 : le compteur a compté hors boucle, sinon il est mort) et le nombre de
  shards de `r_logits` (solde DA-3) ;
- `RSS-DELTA:` (cf. C8) ;
- M-COUT existant : mêmes lignes, libellé « D2H+copie » → « D2H seul » (C2).

## 4. Claims falsifiables — écrites AVANT toute mesure

| # | Claim | Prédiction chiffrée | Ce qui la TUE |
|---|---|---|---|
| **P1** | L'inventaire F2 est juste | Baseline AL-BASE (compteur seul, code non corrigé, run armé `--top-k 1`) : `alloc+resize+remap` ∈ **[10 ; 14] × steps**, `bytes` ∈ **[2,2 ; 2,6] MiB × steps** | Un compte hors fourchette ⇒ l'inventaire de lecture a raté des sites ou `@sizeOf(Shape)` ≠ 288 ⇒ retour à l'inventaire AVANT de corriger |
| **P2** | Le hissage (C2-C5) tient à l'exécution | Après corrections, même run : `alloc+resize+remap = 0` et `free = 0` sur la boucle | Un seul appel résiduel ⇒ un site a été raté OU une API réputée sans allocation en fait une ⇒ le site est nommé (bissection par retrait de composant) |
| **P3** | Les 2 allocs + 3 parcours coûtaient une part mesurable des 1 514 µs | `d2h_us` post-C2 ∈ **[800 ; 1 450] µs** (transfert pur : les memcpy ~2×40 µs + allocs/page faults ~100-300 µs disparaissent) | `> 1 450 µs` ⇒ allocs et copies ne coûtaient rien de mesurable — le gain du chantier est le zéro-alloc (gate), pas la latence. **`< 800 µs`** ⇒ elles coûtaient PLUS que le modèle (économie > 714 µs) : la claim qualitative tient, le modèle chiffré était faux — corrigé et publié. Les deux issues sont publiées telles quelles |
| **P4** | La sémantique est inchangée | S2-PONT re-run : **0 désaccord** ; témoin 48 **bit-identique** (EQ-48) ; témoin 124 identique sur ses **110 premiers ids** (EQ-124, variante 4k) ; vacuity == témoin frais (V-EQ, champs structurels) | Un seul désaccord/écart ⇒ STOP, le remplacement `toSlice`/`getValue` a changé une valeur — enquête avant tout autre pas (pour EQ-124 : regarder la POSITION de divergence avant d'accuser le code — au-delà de 110, c'est la bistabilité connue, pas le chantier) |
| **P5** | Rien ne touche le graphe | md5 HLO du dump `before_optimizations` **identique** au témoin figé avant la 1ʳᵉ ligne de code | Un md5 différent ⇒ une modification host a fui dans le graphe ⇒ STOP |
| **P6** | (M-PIN) Si `PIN: ON`, l'épinglage lève le goulot D2H | `d2h_us` < **500 µs** (1 MiB à ≥ 2 Go/s effectif, latence PJRT comprise) | `PIN: ON` et `d2h_us` ≥ **1 000 µs** ⇒ l'hypothèse « pinned » de D10 est RÉFUTÉE (goulot ailleurs — staging interne du plugin). **Zone [500 ; 1 000) : verdict pré-enregistré MIXTE** — gain partiel, ni confirmé ni réfuté, publié avec le chiffre |
| **P7** | (B10) Le VmRSS ne dérive plus par step | `VmRSS(token généré 200) − VmRSS(token généré 20)` < **5 MiB**, run teacher-forcé 200 steps (C8) | Une dérive ≥ 5 MiB ⇒ une source d'allocation par step échappe au compteur Zig (C/PJRT) ⇒ dette DA-1 requalifiée d'urgente |

⚠ P3 et P6 sont des MESURES à fourchette, pas des gates : leur réfutation est un résultat
publiable, pas un échec. P1, P2, P4, P5, P7 sont portés par des gates à règle d'arrêt.

## 5. Gates et mesures

Méthode commune : run sur la 3090, capture **séparée** `> out.log 2> err.log` (leçon flux
entrelacés), critère = ligne littérale greppée **émise par un émetteur nommé au §3**, FAIL ⇒ STOP,
tag annoté `gate/d10-<slug>-pass` portant les chiffres (préfixe `d10-` : aucune collision dans les
77 tags existants). Non-vacuité patron GC1 : tout compteur d'antécédent à 0 ⇒ INEXÉCUTABLE, pas
PASS.

**Le build mutant M1** (contre-preuve commune, local, jamais committé) : une allocation d'1 MiB
**non libérée** par step, dans la boucle de génération ET dans la boucle vacuity. Il doit faire
échouer AL-0 (`alloc=steps` > 0), AL-VAC (idem) et AL-RSS (dérive ~180 MiB ≥ 5 MiB) — trois FAIL
**vus**, pas déclarés.

| Gate | Prouve | Véhicule | Critère PASS (ligne greppée) | Contre-preuve |
|---|---|---|---|---|
| **S-AC** | Le compteur compte | `--selftest-alloc-count` (host, sans GPU, câblage C1) : 3 alloc + 1 resize + 1 remap + 2 free via le wrapper, compteurs comparés aux attendus | `SELFTEST-AC: PASS 6/6` | Mutation locale : `n_alloc += 1` retiré ⇒ FAIL vu |
| **AL-BASE** | Vitalité du gate en réel + P1 | Run GPU armé `--top-k 1 --max-tokens 60`, binaire AVEC compteur SANS corrections (état constructible : C1 est committé seul, avant C2-C6) | `ALLOC-LOOP:` avec `alloc+resize+remap` ∈ [10;14]×steps — gate de VITALITÉ : il doit voir le code actuel allouer | Trivialement non-vide (le code actuel alloue) |
| **AL-0** | P2 — l'interdit est GARDÉ | Même run, binaire corrigé | `ALLOC-LOOP: alloc=0 resize=0 remap=0 free=0` avec `steps` > 0, ET `ALLOC-TOTAL: alloc=<n>` > 0 (le compteur vit) | Mutant M1 ⇒ `alloc=steps` (> 0) ⇒ FAIL vu |
| **AL-VAC** | P2 sur l'instrument | Run `--window-vacuity` court, binaire corrigé | `ALLOC-VAC: alloc=0 resize=0 remap=0 free=0` avec `steps` > 0 | Mutant M1 (branche vacuity) ⇒ `ALLOC-VAC: alloc=<steps>` ⇒ FAIL vu |
| **EQ-48** | P4 | Re-run 48 ids free-run + `cmp` binaire vs témoin (véhicule du chantier donation) ; émetteur : `cmp out_ids.bin temoin.bin && echo "EQ-48: identique"` | `EQ-48: identique` | Le témoin existe AVANT le chantier ; tout écart = FAIL par construction |
| **EQ-124** | P4 (et exerce la variante 4k) | Run `gemma4_g12a4k`, cmp des **110 premiers ids** vs témoin `mi_witness` ; émetteur : `cmp -n 440 … && echo "EQ-124: identique (110 ids)"` | `EQ-124: identique (110 ids)` | Borné à 110 par la bistabilité aval documentée (FINDING §7bis) — au-delà, un écart n'accuserait pas le chantier |
| **EQ-PONT** | P4 | Protocole S2-PONT repris (3 runs `--top-k 1`) | **Lignes réellement émises par le runner** (`:2289-2291`) : `S2-PONT: steps_comparés=<n> désaccords=0` avec n cumulé ≥ 100, ET antécédent `GENCFG: suppress a mordu <k> fois` avec k ≥ 1 (au 2ᵉ essai si variante bistable, écrit — précédent S2-PONT) | Héritée de S2-PONT (mutations du chantier sampling) |
| **V-EQ** | P4 sur vacuity | Témoin : run `--window-vacuity` AVANT modification (binaire baseline), sortie archivée ; re-run après | `diff` vide sur les **champs STRUCTURELS** des lignes vacuity (`n_ident`, `q` — invariants sémantiques) ; `max_abs` EXCLU du critère (flottant soumis au jitter de compile — deux binaires = deux compiles ; FINDING §7bis), publié à titre informatif | Le témoin est produit d'abord — un instrument modifié sans témoin préalable serait invérifiable |
| **G-0** | P5 | Dump HLO `before_optimizations`, md5, AVANT la 1ʳᵉ ligne de code puis après | md5 identique (méthode du gate S2-G, fraîcheur par mtime) | **Produite pour de vrai une fois** : build scratch avec un op ajouté au graphe ⇒ md5 différent constaté ⇒ jeté (aucun gate du projet n'a encore VU un md5 différer ; l'instrument doit avoir échoué une fois pour compter) |
| **AL-RSS** | P7 | Run teacher-forcé 200 steps (C8), sonde VmRSS aux tokens générés 20/200 | `RSS-DELTA: <n> KiB` avec n < 5 120 (ligne `INEXECUTABLE` = gate non rendu, jamais PASS) | Mutant M1 ⇒ dérive ~180 MiB ⇒ FAIL vu |
| **M-D10** | P3 (mesure) | Protocole M-COUT identique (n ≥ 60, in-process, build §8) | Publie `d2h_us`/`warp_us` avant/après — pas de PASS/FAIL | — |
| **M-PIN** | P6 (mesure) | Même run, `PIN: ON` vs `--no-pin` (A/B) | Publie les deux `d2h_us` + la bannière PIN | — |

Ordre d'exécution imposé : G-0 (témoin) et V-EQ (témoin) et AL-BASE **avant** toute correction ;
puis corrections ; puis le reste. Chaque gate FAIL ⇒ STOP et le résultat est écrit, jamais
requalifié à chaud (leçon instrument dégradé : 2ᵉ requalification du même type ⇒ STOP, diff
l'instrument).

## 6. Interdits et gardes (chacun avec son porteur, aucun « tenu par la revue »)

| Interdit | Porteur |
|---|---|
| Aucune allocation Zig par step dans les deux boucles | **AL-0 / AL-VAC** (compteur, gate) |
| Pas de dérive VmRSS par step | **AL-RSS** (B10, véhicule C8) |
| Pas de changement de graphe | **G-0** |
| Pas de changement de sémantique | **EQ-48 / EQ-124 / EQ-PONT / V-EQ** |
| Gardes en acceptation, jamais en rejet (NaN-safe) | Les checks C2/C3 sont des égalités strictes sur entiers (pas de NaN possible) |
| Le buffer `work` est libéré par l'allocateur qui l'a alloué | **Structurel** (C7) : le booléen `pin_on` unique pilote l'alloc, la bannière ET le free — il n'existe pas de chemin de code où les deux divergent. (La rév. 1 s'appuyait sur le panic de `dmaUnmap` : `catch unreachable` est de l'UB en ReleaseFast, et le cas croisé gpa.free-d'un-buffer-dma était silencieux — porteur retiré par la revue) |
| Aucun composant ne s'appuie sur `std.debug.assert` comme garde (UB en ReleaseFast) | `appendBounded` (C5, erreur active tous modes) + `stdx.debug.assert` ZML (F4) pour les gardes de shape |

## 7. Dettes déclarées de CE chantier

| # | Dette | Motif |
|---|---|---|
| **DA-1** | Les allocations **C/PJRT** par step (`Buffer.fromBytes` ×2, `exe.call`, `toHostBuffer`) sont invisibles au compteur Zig | L'implémentation est dans le .so du plugin. AL-RSS borne la dérive nette à < 5 MiB/180 steps mais ne compte pas les appels. Voie future : `LD_PRELOAD` d'un malloc compteur — non faite ici |
| **DA-2** | Le gate AL-0 n'est exercé que sur ses runs (60 tokens `--top-k 1`, plus un run `--oracle` court — C5) | Un chemin conditionnel non emprunté pourrait allouer sans être vu. Le compteur étant TOUJOURS actif et publié à chaque run, toute exécution ultérieure re-vérifie gratuitement |
| **DA-3** | `n_shards` de `r_logits` est supposé 1 (mono-GPU, sharding répliqué), jamais imprimé | Soldé au premier run : `ALLOC-TOTAL: … shards=<n>` (§3 Publication) |
| **DA-4** | La variante 8k (`gemma4_g12a8k`) compile le même fichier mais aucun gate ne la re-exécute | La 4k est exercée par EQ-124 ; pour la 8k : graphe intouché (G-0) et code host identique — dette écrite, pas re-prouvée |
| **DA-5** | Les logs bruts des gates S2 du chantier précédent ne sont pas dans `logs/` | Constat de la recon. Vérification de récupérabilité sur la VM au début de l'exécution ; rapatriés si présents, constat d'irretraçabilité écrit sinon |
| **DA-6** | Les allocations internes de `std.Io.Threaded` passent par le gpa NON wrappé (construit dans `start.zig` avant `run()`) — invisibles au compteur | Sœur de DA-1. Contrepartie : seul le thread principal traverse le wrapper ⇒ les compteurs u64 non atomiques de C1 sont corrects par construction |

## 8. Build et exécution — le mode est PROUVÉ, pas supposé

- Build : `cd /data/rqz_workspace/zml && ./bazel.sh build -c opt --@rules_zig//zig/settings:mode=release_fast --@zml//platforms:cuda=true //examples/rqz:gemma4_g12auto`
  ⚠ Les DEUX flags sont nécessaires : le mode Zig de rules_zig est **indépendant** de `-c` et son
  défaut est `debug` (F8) — la rév. 1 n'avait que `-c opt` et aurait produit un frontend Zig en
  Debug, rendant tous les gates inexécutables par la règle ci-dessous. La commande est écrite ICI
  et recopiée dans chaque log de gate ;
- La bannière `BUILD: mode=ReleaseFast` est greppée dans CHAQUE log de gate — critère **fixe** :
  un log qui porte un autre mode est INEXÉCUTABLE, pas PASS ;
- Déploiement : `zml_runner/deploy_to_3090.sh` (rsync, inchangé) ;
- Runs : `ssh ia@192.168.1.163`, nohup + log distant + stdin fermé pour les runs longs (leçon
  26 juil), capture `> out.log 2> err.log` systématique.

## 9. Ce que ce chantier NE proclame PAS

- Il ne change pas la claim « == HF » (portée inchangée : même argmax sur les logits bruts).
- Il ne promet pas de gain de débit global : 3,6 % d'un step au mieux, et P3/P6 peuvent être
  réfutées — le livrable garanti est l'interdit §5 GARDÉ par un gate, plus la vérité mesurée sur
  les leviers latence.
- Il ne couvre pas les allocations C/PJRT (DA-1) ni celles de `std.Io.Threaded` (DA-6) : « zéro
  allocation » signifie « zéro appel à l'allocateur Zig du runner dans les boucles de step »,
  périmètre écrit et instrumenté.
