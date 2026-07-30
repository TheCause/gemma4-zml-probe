# Spec — D10 : zéro allocation par step dans le décodage 12B (rév. 1)

> **Statut : SPEC — committée AVANT toute mesure** (exigence de falsifiabilité, le git log fait foi).
> Branche : `s2-d10-alloc-par-step`. Chantier ouvert sur décision Régis du 30 juil 2026 (niveau de
> travail : meute, via ultracode).
>
> Dette d'origine : `docs/SAMPLING_RESULTS.md` §5, **D10** — « `toSliceAlloc` alloue 1 Mo par
> step », violant l'interdit §5 de la spec sampling (« aucune allocation par step »), lui-même
> déclaré **sans gate porteur** (dette D3). Ce chantier corrige la violation ET pose le gate,
> conformément à l'arbitrage **B9** (« compteur d'allocations à 0, pas le débit ») et **B10**
> (critère RSS chiffré) acceptés le 29 juil et jamais câblés.

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
   `toSliceAlloc` (zml `buffer.zig:283` et `:289`) fait DEUX allocations par appel (le résultat +
   un staging par shard) plus un memcpy interne (`:297`), avant le `@memcpy(scfg.work, lg)` du
   runner (`gemma4_g12auto.zig:2145`). Par step armé : **2 MiB alloués/libérés + 3 parcours
   d'1 MiB**.
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
   sessions LLM amont de ZML utilisent ce motif dans leur boucle de décode (`llama/session.zig:50`).
4. **F4 — Les gardes de cette voie sont actives dans TOUS les modes de build** :
   `stdx.debug.assert` est inconditionnel (`stdx/debug.zig:11-15`). `Shape.eql` ne compare que
   dims + dtype, PAS les tags (`shape.zig:501-503`) → une destination construite avec
   `r_logits.shape()` passe par construction. Un viol panique au premier run, jamais silencieux.
5. **F5 — `exe.args`/`results` sont hissables** : seuls leurs `init` allouent ; `set()`, `call()`
   et `get()` n'allouent rien (lu dans `exe.zig:66-121`, `:218-236`, `:259-299`). `Buffer.fromBytes`
   et `Buffer.scalar` (H2D du token, chaque step) n'allouent **rien côté Zig** (`buffer.zig:92-174` :
   `Slice.init` + structures par valeur + PJRT côté C). **Jamais compilé ni exécuté : c'est la
   prédiction P2, pas un acquis.**
6. **F6 — Coût actuel du bloc chemin B** (build opt, D9) : total 3 796 µs = 3,6 % d'un step,
   décomposé **D2H+alloc+copie 1 514 µs** / **warpers 2 268 µs**. Le chrono `t_d2h` agrège
   transfert, allocations, memcpy et page faults — la part de chacun est inconnue (c'est M-D10
   qui la départage).
7. **F7 — Deuxième site** : la boucle `--window-vacuity` (`gemma4_g12auto.zig:1889-1936`) contient
   les mêmes motifs (`exe.args`/`results` + `r_logits.toSliceAlloc` ligne 1901), antérieurs au
   chantier (commit `c2211c0`). C'est un **instrument** (replay U9-ii), pas le chemin chaud.
8. **F8 — L'identité de l'allocateur `init.gpa` dépend du mode de build** (std `start.zig:668-692`) :
   Debug ⇒ `DebugAllocator` ; ReleaseFast ⇒ `c_allocator` si libc, sinon `smp_allocator`. De plus
   la commande exacte du build « opt » de D9 est **perdue**, et `--config=debug` de ZML compile le
   backend en opt mais le frontend Zig en debug (`.bazelrc:106-108`). D'où l'exigence §8 : le
   binaire publie `builtin.mode`, la commande de build est écrite.

## 2. Périmètre

**Dedans** : la boucle de génération de `generateOnce` (`:2077-2271`) et la boucle vacuity
(`:1889-1936`) du seul runner `gemma4_g12auto.zig` (et ses variantes 4k/8k qui compilent le même
fichier) ; le module compteur ; l'expérience pinned sur `scfg.work` ; les commentaires périmés
attenants (en-tête CLI lignes 25-26 « 2 des 8 clés », ligne 2094 « NON lue ici ») ; un errata
d'une ligne dans `docs/superpowers/plans/2026-07-29-sampling-phase2.md` (base de coût réfutée
1,57 %/177,95 % → renvoi à la spec rév. 4).

**Hors périmètre, écrit** : E2B (logits hors graphe) · batching · phase 1 penalty (suspendue) ·
dette D1 (`applyTopP` sans couverture GPU) · streaming · les allocations **C/PJRT internes**
(invisibles à l'allocateur Zig — dette DA-1, partiellement couverte par AL-RSS) · les autres
runners du dépôt.

## 3. Design — composants (modularité : chacun isolé, testable seul)

### C1 — `alloc_count.zig` : compteur d'allocations, module std-only

Nouveau fichier `zml_runner/alloc_count.zig`, **sans aucune dépendance ZML** (même règle que
`sampling.zig`), ajouté aux `srcs` des TROIS cibles qui compilent le runner (`gemma4_g12auto`,
`gemma4_g12a4k`, `gemma4_g12a8k` — `BUILD.bazel`).

```zig
//! Compteur d'allocations — wrapper transparent de std.mem.Allocator.
//! Les 4 fonctions vtable délèguent au parent et comptent. Les corps suivent
//! std.testing.FailingAllocator de CE toolchain (0.16.0-dev.2722), signatures copiées.
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
`std.mem.Alignment`) ; si la compilation les contredit, c'est la std qui fait foi, et l'écart est
consigné. Le compteur est **toujours actif** (un incrément par appel — et zéro incrément par step
une fois le chantier fini, par définition).

Installation : dans `run()` (`gemma4_g12auto.zig:1509`), `init.gpa` est remplacé par le wrapper —
**point de substitution unique**, aucun autre changement de signature aval que le paramètre
`*CountingAllocator` passé à `generateOnce` (même traitement par pointeur que `scfg`, mêmes
3 sites d'appel).

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
seul — c'est ce qui rend M-D10 interprétable.

### C3 — Top-5 : `getValue` vers la pile

Remplace les deux `toSliceAlloc` (2104/2106) par `getValue` (`buffer.zig:249`, délègue à `toSlice`,
zéro allocation) :

```zig
if (r_t5v.shape().dtype() != .f32) return error.UnexpectedDtype; // gardes conservées,
if (r_t5i.shape().dtype() != .i32) return error.UnexpectedDtype; // déplacées AVANT le D2H
const t5v: [gencfg.TOP_K]f32 = try r_t5v.getValue([gencfg.TOP_K]f32, io);
const t5i: [gencfg.TOP_K]i32 = try r_t5i.getValue([gencfg.TOP_K]i32, io);
```

`getValue` assert `byteSize == @sizeOf(T)` (20 octets) — actif dans tous les modes (F4). Le check
dtype est **renforcé** (le code actuel ne vérifiait que i32 ; f32 était supposé).

### C4 — `exe.args`/`exe.results` hissés hors des deux boucles

`init` une fois AVANT la boucle, `set()`/`call()`/`get()` par step (aucun n'alloue — F5),
`deinit` après. S'applique à la boucle de génération ET à la boucle vacuity.

### C5 — ArrayLists : capacité réservée avant la boucle

`generated` et `gen_top5` : `ensureTotalCapacity(allocator, borne)` avant la boucle, où
`borne = ids.len + max_tokens` (connue avant d'entrer) ; `appendAssumeCapacity` dans la boucle
(échec impossible par construction ; si la borne était fausse, panic explicite — jamais une
réallocation silencieuse qui fausserait le gate).

### C6 — Boucle vacuity (K3) : même motif, buffer local

Un buffer d'1 MiB alloué UNE FOIS avant les passes vacuity (local au mode), `toSlice` dedans,
`exe.args`/`results` hissés (C4), libération après. L'instrument est re-validé par V-EQ contre un
témoin produit AVANT la modification (leçon « témoins avant tout deploy »).

### C7 — Expérience pinned (K2, M-PIN)

`scfg.work` alloué via `zml.mem.DmaAllocator` (→ `PJRT_Client_DmaMap`) au lieu de gpa, UNE FOIS.
Échec de `dmaMap` ⇒ l'alloc rend `null` ⇒ **repli automatique sur gpa**, et la bannière publie
`PIN: ON` ou `PIN: OFF (DmaMap indisponible)` — les trois issues (indisponible / sans gain / gain)
sont publiables, aucune n'est un échec du chantier. ⚠ Le buffer pinned est libéré par le MÊME
allocateur qui l'a créé (dmaUnmap dans le free du DmaMapAllocator) — le defer existant est adapté.

### Publication (véhicule des gates)

- Bannière au démarrage : `BUILD: mode=<builtin.mode> pin=<ON|OFF>` ;
- Fin de `generateOnce` (donc par prompt en `--repl`) :
  `ALLOC-LOOP: alloc=<n> resize=<n> remap=<n> free=<n> bytes=<n> steps=<n>`
  (deltas des compteurs pris juste avant/après la boucle) ;
- Fin des passes vacuity : `ALLOC-VAC: alloc=<n> resize=<n> remap=<n> free=<n> bytes=<n> steps=<n>` ;
- M-COUT existant inchangé (mêmes lignes, même décomposition d2h/warp).

## 4. Claims falsifiables — écrites AVANT toute mesure

| # | Claim | Prédiction chiffrée | Ce qui la TUE |
|---|---|---|---|
| **P1** | L'inventaire F2 est juste | Baseline AL-BASE (compteur seul, code non corrigé, run armé `--top-k 1`) : `alloc+resize+remap` ∈ **[10 ; 14] × steps**, `bytes` ∈ **[2,2 ; 2,6] MiB × steps** | Un compte hors fourchette ⇒ l'inventaire de lecture a raté des sites ou `@sizeOf(Shape)` ≠ 288 ⇒ retour à l'inventaire AVANT de corriger |
| **P2** | Le hissage (C2-C5) tient à l'exécution | Après corrections, même run : `alloc+resize+remap = 0` et `free = 0` sur la boucle | Un seul appel résiduel ⇒ un site a été raté OU une API réputée sans allocation en fait une ⇒ le site est nommé (le compteur donne les deltas par run, la bissection par retrait de composant le localise) |
| **P3** | Les 2 allocs + 3 parcours coûtaient une part mesurable des 1 514 µs | `d2h_us` post-C2 ∈ **[800 ; 1 450] µs** (transfert pur : les memcpy ~2×40 µs + allocs/page faults ~100-300 µs disparaissent) | `d2h_us` > 1 450 µs ⇒ allocs et copies ne coûtaient rien de mesurable — le gain du chantier est le zéro-alloc (gate), pas la latence, et c'est écrit tel quel |
| **P4** | La sémantique est inchangée | S2-PONT re-run : **0 désaccord** ; témoin 48 **bit-identique** ; témoin 124 identique sur ses 110 premiers ids ; vacuity == témoin frais | Un seul désaccord/écart ⇒ STOP, le remplacement `toSlice`/`getValue` a changé une valeur — enquête avant tout autre pas |
| **P5** | Rien ne touche le graphe | md5 HLO du dump `before_optimizations` **identique** au témoin figé avant la 1ʳᵉ ligne de code | Un md5 différent ⇒ une modification host a fui dans le graphe ⇒ STOP |
| **P6** | (M-PIN) Si `PIN: ON`, l'épinglage lève le goulot D2H | `d2h_us` < **500 µs** (1 MiB à ≥ 2 Go/s effectif, latence PJRT comprise) | `PIN: ON` et `d2h_us` ≥ 1 000 µs ⇒ l'hypothèse « pinned » de D10 est RÉFUTÉE (le goulot est ailleurs — staging interne du plugin) ; publié comme tel |
| **P7** | (B10) Le RSS ne dérive plus par step | `max_rss(token 200) − max_rss(token 20)` < **5 MiB** (véhicule `mem_probe.zig`) | Une dérive ≥ 5 MiB ⇒ une source d'allocation par step échappe au compteur Zig (C/PJRT) ⇒ dette DA-1 requalifiée d'urgente |

⚠ P3 et P6 sont des MESURES à fourchette, pas des gates : leur réfutation est un résultat
publiable, pas un échec. P1, P2, P4, P5, P7 sont portés par des gates à règle d'arrêt.

## 5. Gates et mesures

Méthode commune : run sur la 3090, capture **séparée** `> out.log 2> err.log` (leçon flux
entrelacés), critère = ligne littérale greppée, FAIL ⇒ STOP, tag annoté `gate/d10-<slug>-pass`
portant les chiffres. Non-vacuité patron GC1 : tout compteur d'antécédent à 0 ⇒ INEXÉCUTABLE,
pas PASS.

| Gate | Prouve | Véhicule | Critère PASS (ligne greppée) | Contre-preuve |
|---|---|---|---|---|
| **S-AC** | Le compteur compte | `--selftest-alloc-count` (host, sans GPU) : 3 alloc + 1 resize + 1 remap + 2 free via le wrapper, compteurs comparés aux attendus | `SELFTEST-AC: PASS 6/6` | Mutation locale : `n_alloc += 1` retiré ⇒ FAIL vu |
| **AL-BASE** | Vitalité du gate en réel + P1 | Run GPU armé `--top-k 1 --max-tokens 60`, binaire AVEC compteur SANS corrections | `ALLOC-LOOP:` avec `alloc+resize+remap` ∈ [10;14]×steps — c'est un gate de VITALITÉ : il doit voir le code actuel allouer | Trivialement non-vide (le code actuel alloue) |
| **AL-0** | P2 — l'interdit est GARDÉ | Même run, binaire corrigé | `ALLOC-LOOP: alloc=0 resize=0 remap=0 free=0` | Build mutant local (une alloc réintroduite dans la boucle) ⇒ `alloc=60` ⇒ FAIL vu. Non-vacuité : `steps` > 0 publié ET compteur process total > 0 (le compteur vit) |
| **AL-VAC** | P2 sur l'instrument | Run `--window-vacuity` court, binaire corrigé | `ALLOC-VAC: alloc=0 resize=0 remap=0 free=0` | Couverte par la mutation d'AL-0 (même compteur) |
| **EQ-48** | P4 | `cmp` binaire des 48 ids free-run vs témoin (véhicule du chantier donation, repris tel quel) | `EQ-48: identique` (cmp exit 0) | Le témoin existe AVANT le chantier ; tout écart = FAIL par construction |
| **EQ-PONT** | P4 | Protocole S2-PONT repris : 3 runs `--top-k 1`, compteurs publiés | `S2-PONT: <n> steps compared, 0 désaccord` avec n ≥ 100 et `n_suppress_hits` ≥ 1 (antécédent non vide — au 2ᵉ essai si bistabilité, écrit) | Héritée de S2-PONT (mutations du chantier sampling) |
| **V-EQ** | P4 sur vacuity | Témoin : run `--window-vacuity` AVANT modification (binaire baseline), sortie archivée ; re-run après | `diff` vide sur les lignes vacuity | Le témoin est produit d'abord — un instrument modifié sans témoin préalable serait invérifiable |
| **G-0** | P5 | Dump HLO `before_optimizations`, md5, AVANT la 1ʳᵉ ligne de code puis après | md5 identique (méthode S2-G, fraîcheur par mtime) | Un dump d'un graphe réellement modifié donne un md5 différent (prouvé par S2-G au chantier masques) |
| **AL-RSS** | P7 | Run 200 tokens, `mem_probe` à token 20 et token 200 | `RSS-DELTA: <n> KiB` avec n < 5 120 | Le baseline actuel (2,4 MiB/step alloués/libérés) rend le delta OBSERVABLE si une fuite par step existait |
| **M-D10** | P3 (mesure) | Protocole M-COUT identique (n ≥ 60, in-process, build écrit §8) | Publie `d2h_us`/`warp_us` avant/après — pas de PASS/FAIL | — |
| **M-PIN** | P6 (mesure) | Même run, `PIN: ON` vs `PIN: OFF` forcé (flag `--no-pin` pour l'A/B) | Publie les deux `d2h_us` + la bannière PIN | — |

Ordre d'exécution imposé : G-0 (témoin) et V-EQ (témoin) et AL-BASE **avant** toute correction ;
puis corrections ; puis le reste. Chaque gate FAIL ⇒ STOP et le résultat est écrit, jamais
requalifié à chaud (leçon instrument dégradé : 2ᵉ requalification du même type ⇒ STOP, diff
l'instrument).

## 6. Interdits et gardes (chacun avec son porteur, aucun « tenu par la revue »)

| Interdit | Porteur |
|---|---|
| Aucune allocation Zig par step dans les deux boucles | **AL-0 / AL-VAC** (compteur, gate) |
| Pas de dérive RSS par step | **AL-RSS** (B10) |
| Pas de changement de graphe | **G-0** |
| Pas de changement de sémantique | **EQ-48 / EQ-PONT / V-EQ** |
| Gardes en acceptation, jamais en rejet (NaN-safe) | Revue + les checks C2/C3 sont des comparaisons d'égalité stricte sur entiers (pas de NaN possible) |
| Le buffer pinned est libéré par l'allocateur qui l'a créé | Panic `dmaUnmap` au premier run si violé (free du DmaMapAllocator, `mem.zig:176-179`) |

## 7. Dettes déclarées de CE chantier

| # | Dette | Motif |
|---|---|---|
| **DA-1** | Les allocations **C/PJRT** par step (`Buffer.fromBytes` ×2, `exe.call`, `toHostBuffer`) sont invisibles au compteur Zig | L'implémentation est dans le .so du plugin. AL-RSS borne la dérive nette à < 5 MiB/180 steps mais ne compte pas les appels. Voie future : `LD_PRELOAD` d'un malloc compteur — non faite ici |
| **DA-2** | Le gate AL-0 n'est exercé que sur le run du gate (60 tokens, `--top-k 1`) | Un chemin conditionnel non emprunté (EOS précoce, désaccord S2-PONT loggé) pourrait allouer sans être vu. Le compteur étant TOUJOURS actif et publié à chaque run, toute exécution ultérieure re-vérifie gratuitement |
| **DA-3** | `n_shards` de `r_logits` est supposé 1 (mono-GPU, sharding répliqué), jamais imprimé | `toSlice` à n_shards > 1 écrirait n fois la même zone (répliqué) — le run AL-0 publiera `shards=<n>` une fois pour solder |
| **DA-4** | La variante 8k (`gemma4_g12a8k`) compile le même fichier mais aucun gate ne la re-exécute | Le graphe est intouché (G-0) et le code host est identique ; dette écrite, pas re-prouvée |
| **DA-5** | Les logs bruts des gates S2 du chantier précédent ne sont pas dans `logs/` | Constat de la recon. Vérification de récupérabilité sur la VM au début de l'exécution ; rapatriés si présents, constat d'irretraçabilité écrit sinon |

## 8. Build et exécution — le mode est PROUVÉ, pas supposé

- Build : `cd /data/rqz_workspace/zml && ./bazel.sh build -c opt --@zml//platforms:cuda=true //examples/rqz:gemma4_g12auto`
  (la commande est écrite ICI et recopiée dans chaque log de gate ; `-c opt` règle aussi le mode
  Zig de rules_zig — vérifié par la bannière, pas supposé : F8) ;
- La bannière `BUILD: mode=<builtin.mode>` est greppée dans CHAQUE log de gate — un gate dont le
  log ne porte pas `mode=ReleaseFast` (valeur attendue de `-c opt`, à confirmer au premier build
  et à figer alors) est INEXÉCUTABLE, pas PASS ;
- Déploiement : `zml_runner/deploy_to_3090.sh` (rsync, inchangé) ;
- Runs : `ssh ia@192.168.1.163`, nohup + log distant + stdin fermé pour les runs longs (leçon
  26 juil), capture `> out.log 2> err.log` systématique.

## 9. Ce que ce chantier NE proclame PAS

- Il ne change pas la claim « == HF » (portée inchangée : même argmax sur les logits bruts).
- Il ne promet pas de gain de débit global : 3,6 % d'un step au mieux, et P3/P6 peuvent être
  réfutées — le livrable garanti est l'interdit §5 GARDÉ par un gate, plus la vérité mesurée sur
  les leviers latence.
- Il ne couvre pas les allocations C/PJRT (DA-1) : « zéro allocation » signifie « zéro appel à
  l'allocateur Zig du runner dans les boucles de step », périmètre écrit et instrumenté.
