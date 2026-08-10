# Dump/restore du KV-cache 12B — résultats

> **Date d'exécution** : 9-10 août 2026 · **Branche** : `kv-dump-restore` ·
> **Spec** (fait foi) : `docs/superpowers/specs/2026-08-09-kv-cache-dump-restore-design.md` rév. 3 ·
> **Plan** : `docs/superpowers/plans/2026-08-09-kv-cache-dump-restore.md` rév. 3 ·
> **Claims et prédictions PRÉ-ENREGISTRÉES** avant la première ligne de code (commit `2d19d94`,
> décisions actées `afbadf6`) — le `git log` fait foi.
> **Build** : `-c opt --@rules_zig//zig/settings:mode=release_fast --@zml//platforms:cuda=true`
> (`zml_runner/build_3090.sh`). Chaque log de gate porte `BUILD: mode=ReleaseFast` — un log qui
> ne l'affiche pas est INEXÉCUTABLE, pas PASS.
> **Deux décisions Régis appliquées telles quelles** (GO 9 août) : (1) `--dump-cache` + sampling
> armé = refus bruyant, PRNG non sérialisé (dette) ; (2) gates sur 1280 + 4k, 8k non exercé (dette).

## 1. Ce que le chantier livre

`--dump-cache <fichier>` sérialise l'état complet d'une génération 12B — 4 caches KV f32 + tous
les tokens feedés + un manifest auto-décrivant — dans **un** safetensors relisible par le runner
**et** par `safetensors` Python. `--load-cache <fichier>` réimplante cet état et **reprend la
génération sans re-calculer le préfixe**.

```
gemma4_g12auto <ckpt> <tok> --prompt "..." --max-tokens 16 --dump-cache etat.kvdump
gemma4_g12auto <ckpt> <tok> --load-cache etat.kvdump --max-tokens 32      # process NEUF
```

**Aucun octet dans `engine.zig`, aucun changement de graphe** : le dump est un D2H après la
boucle, le restore remplit les mêmes slices host que le `@memset(0)` d'origine.

## 2. Verdicts des 8 gates

| Gate | Prouve | Verdict | Chiffres |
|---|---|---|---|
| **DC0** graphe intact | §2.3 | **PASS** | md5 `before_optimizations` **1280 = `297679847aa04b719942d75d093adf2b`**, **4k = `704de4bc1999f5956724b184bd097ce6`** — **identiques** aux témoins capturés AVANT la 1ʳᵉ ligne de code (commit `eae5f42`), `cmp` byte-à-byte OK des deux côtés ; `git diff main -- engine.zig` : **0 ligne**. Le md5 1280 est de surcroît celui de **GC0 (27 juil) et de S2-G (29 juil)** — continuité sur quatre chantiers |
| **DC1** round-trip host + Python | C-A | **PASS** | Volet host (`--selftest-kvdump-io`) : round-trip bit-identique + **mutant VU** (`KvDumpChecksumMismatch`). Volet Python sur dump **RÉEL** (840 Mo) : les **5 xxh64 recalculés concordent** avec ceux écrits par Zig ; mutant réel (`sl_v[12345]` `0xf9`→`0x06`, manifest intact) **refusé** |
| **DC2** équivalence intra-process | C-B | **PASS** | **32/32 bit-identiques** — ids **et** indices top-5 **et** *bits* des valeurs (`@bitCast u32`). Pré-conditions franchies : les deux appels atteignent `max_tokens`, et les 16 premiers ids concordent |
| **DC3** équivalence inter-process | C-C | **PASS, au-delà du critère** | Process neuf : **32/32**, **aucune divergence**. Le critère admettait une 1ʳᵉ divergence à marge ≤ 1,873e-3 ; il n'y en a pas |
| **DC4** le mordant | C-E | **PASS** | Cache zéroté au manifest **valide** : divergence dès **`@gen=0`**, `n_match` **0/32**, alors que la marge de référence à ce step vaut **12,62** — divergence *grasse*, pas un tie |
| **DC5** refus bruyants | §4.5 | **PASS** | **11 refus**, chacun VU avec SON erreur : 7 fichier/état + 4 flags (§4 ci-dessous) |
| **DC6** interdit D10 intact | §2.4 | **PASS** | Trois runs ≥ 260 tokens (nu / `--dump-cache` / `--load-cache`) : `ALLOC-LOOP: alloc=0 resize=0 remap=0 free=0 bytes=0` **identique aux trois**. `RSS-DELTA` **140 / 148 / 184 KiB**, chiffrés (jamais `INEXECUTABLE`), tous sous le plafond B10 de **5 120 KiB** |
| **DC7** le gain, mesuré | C-D | **PASS** | À 4k, atteindre l'état @**3 927** positions **par le calcul** : **449,485 s** (`PERF : prefill 28 steps en 3,220 s ; génération 3900 tokens en 446,265 s`, compile exclue). Le **restaurer** : **`KVLOAD-PERF: 0,898 s`** (lecture des **2,62 GiB incluse**, compile exclue des deux côtés). **Gain = ×500,5** (prédit ≥ ×30, kill < ×5) |

**Non-vacuité transversale** : DC2 et DC3 ne sont déclarés PASS que parce que **DC4 a produit sa
divergence**. Un dispositif d'équivalence dont on n'a jamais vu le contraire mordre n'a rien
prouvé (`feedback_invariant_tue_le_controle`).

## 3. Les 5 claims pré-enregistrées, jugées

| Claim | Prédiction | Verdict | Mesure |
|---|---|---|---|
| **C-A** round-trip sans perte | xxh64 égaux, manifest survit | **CONFIRMÉE** | DC1, deux volets (Zig et Python) |
| **C-B** intra-process = n'avoir jamais quitté | top-5 (ids **et** vals) bit-identiques sur tous les steps comparés | **CONFIRMÉE** | 32/32, zéro bit d'écart |
| **C-C** inter-process fidèle, borné par la bistabilité | 1ᵉʳ step identique ; 1ʳᵉ divergence éventuelle à marge ≤ 1,873e-3 | **CONFIRMÉE au-delà** | **aucune divergence** sur 32 tokens |
| **C-D** gain ≥ ×30 à 4k | ~451 s de calcul contre < 15 s de restore | **CONFIRMÉE au-delà** | **449,485 s** contre **0,898 s** ⇒ **×500,5** |
| **C-E** le cache porte la mémoire | cache zéroté ⇒ divergence dans les 4 premiers tokens | **CONFIRMÉE** | divergence au **1ᵉʳ** token |

### Grandeurs prédites AVANT mesure

| Grandeur | Prédiction §2bis | Mesure | Écart |
|---|---|---|---|
| octets tensoriels du dump 1280 | **880 803 840** | **880 803 840** (`octets=880804012` − 172 d'ids) | **0 — exact à l'octet** |
| octets tensoriels du dump 4k | **2 818 572 288** | **2 818 572 288** (`octets=2818587996` − 15 708 d'ids) | **0 — exact à l'octet** |
| temps de calcul pour atteindre @3 927 pos., 4k | **~451 s** (8,7 tok/s) | **449,485 s** | **−0,3 %** |
| `ALLOC-LOOP` avec dump ou load | identique au run nu | identique aux trois runs | **0** |

**⚠ Nuance honnête sur le 0,898 s de DC7(ii)** : le dump venait d'être écrit ~1 min plus tôt, donc
la lecture des 2,62 GiB a très probablement bénéficié du **cache de pages** du système. Un restore
« à froid » serait borné par le NVMe. Même dans une hypothèse pessimiste à 1 GiB/s (≈ 2,6 s de
lecture, soit ~3,5 s au total), le gain resterait **≥ ×130** — la claim C-D (≥ ×30) tient très
largement dans les deux régimes. La mesure à froid n'a pas été faite (elle exige `drop_caches`,
donc root sur la VM) : c'est une **dette de mesure**, pas un résultat.

## 4. Les 11 refus bruyants, tous VUS échouer

| Cas | Erreur | Preuve |
|---|---|---|
| dump 1280 chargé par le binaire 4k | `KvDumpVariantMismatch` | `dc5_all.log` (a) |
| `ckpt_hdr_xxh64` forgé | `KvDumpCheckpointMismatch` | (b) |
| fichier tronqué à 1000 octets | `KvDumpTruncated` (qualifiée, aucun crash) | (c) |
| `format` altéré | `KvDumpBadFormat` | (d) |
| shape `sl_k` `[40]`→`[41]` | `KvDumpShapeMismatch` | (e) |
| `step_next` forgé (7 ≠ 43) | `KvDumpInconsistentState` | (f) |
| `--max-tokens 999999` | `SequenceTooLong` | (g) |
| `--load-cache` + `--repl` | `LoadCacheReplUnsupported` | `dc5_flags.log` |
| `--dump-cache` + `--repl` | `DumpCacheReplUnsupported` | idem |
| `--load-cache` + `--prompt` | `LoadCacheWithPrompt` | idem |
| `--dump-cache` + `--seed` | `DumpWithSamplingArmed` | idem |
| checksum d'un tenseur ≠ manifest | `KvDumpChecksumMismatch` | `dc1m.err.log` (dump réel) |

Un cas n'est **pas** un refus, par décision de design : `gencfg_path` du manifest ≠ chemin courant
⇒ **WARN logué**. La politique est re-dérivée des fichiers courants — c'est le comportement voulu,
mais l'écart doit être visible.

## 5. Périmètre de la claim (ce que ce chantier ne dit PAS)

- **12B seul** (`gemma4_g12auto` et sa variante 4k). L'E2B ne sort pas ses logits du graphe :
  même politique de périmètre que `generation_config` et le sampling.
- **Argmax / politique host-side**. Un restore avec d'autres warpers diverge légitimement — la
  clé `sampling` du manifest le rend visible, elle ne l'empêche pas.
- **Un fichier = un état**. Pas de multi-tour, pas de prompt neuf sur un cache restauré.
- L'équivalence **bit-exacte** n'est affirmée qu'**intra-process** (DC2). Inter-process, le
  critère est borné par la bistabilité connue (DC3) — même si, sur ce témoin, aucune divergence
  n'est apparue.

## 6. Dettes (écrites, pas tues)

| # | Dette | Pourquoi |
|---|---|---|
| K1 | **Sampling armé + dump = refus** — l'état du PRNG Xoshiro256 (32 octets) n'est pas sérialisé | Décision Régis actée : le sérialiser ajouterait une claim d'équivalence stochastique qu'aucun gate simple ne prouve |
| K2 | **8k non exercé** — le code est le même `comptime` pour `g12a8k`, aucun gate ne le couvre | Décision Régis actée (comme DA-4) |
| K3 | **E2B non couvert** | Logits hors du graphe (`gen_auto.zig`) |
| K4 | **`--repl` + dump/load refusés** | La sémantique multi-tour n'existe pas ; c'est le chemin naturel d'un futur « résident à reprise » |
| K5 | **Reprise avec prompt neuf** hors périmètre v1 | C'est un prefill partiel : son propre chantier |
| K6 | **Pas de compression** (f32 ; ~×2 possible en bf16) | On ne dégrade pas un état exact pour du disque |
| K7 | Le refus « fichier tronqué » n'a **pas de message `log.err` propre** avant `KvDumpTruncated` | L'erreur reste nommée et sans crash (critère spec rempli) ; cosmétique |
| K8 | **Le `0,898 s` de DC7(ii) est une lecture À CHAUD** (cache de pages) | La mesure à froid exige `drop_caches` (root sur la VM). Le gain resterait ≥ ×130 dans l'hypothèse pessimiste — la claim ne dépend pas de cette dette |

## 7. Écarts au plan, assumés et déclarés

1. **`--selftest-kvdump-eq` prend le chemin du dump en ARGUMENT** (le plan le codait en dur à
   `$K/dc2.kvdump` — un chemin en dur dans le binaire). La référence est écrite en
   `<dump>.ref.json`.
2. **`gate/dc1-pass` posé à la Task 7**, pas à la Task 2 : le tag couvre les deux volets (host et
   Python). Un tag PASS posé avant le volet Python aurait affirmé plus que ce qui était prouvé.
   Le volet host porte son propre tag `gate/dc1-host-pass`.
3. **La garde §4.3(7)** (`step_next + limit > L_MAX`) est réalisée par le pré-check **existant**
   de `run()` alimenté par `ids = ids_fed` : même condition, même erreur, même moment. Pas de
   duplication.
4. **`SL_SHAPE`/`FL_SHAPE` déclarées une fois** (le plan les inlinait dans `dumpCacheFile`) : le
   restore exige exactement ce que le dump écrit ; deux listes séparées auraient pu diverger en
   silence et `KvDumpShapeMismatch` n'aurait plus rien discriminé.
5. **Contrôle AJOUTÉ** au selftest DC1 : `xxh64("x")` comparé à la valeur mesurée de `xxhash`
   Python. Si les deux implémentations divergeaient, tous les checksums seraient un décor — et on
   ne l'apprendrait qu'après un run GPU.
6. **`Resume` ne porte pas `ids_fed`** (le plan le prévoyait) : `run()` les tient déjà dans `ids`,
   nécessaires aux pré-checks. Éviter la double propriété.

## 8. Ce que l'exécution a appris

**Un segfault au premier restore réel — causé par une déviation « prudente » du plan.** Le plan
prescrivait `std.json.parseFromSliceLeaky` ; j'avais préféré `parseFromSlice` + conservation du
`Parsed`, par mimétisme avec `gencfg.zig:239`. Or `Parsed.deinit()` lit
`self.arena.child_allocator`, et ce `child_allocator` est l'`arena.allocator()` **local à
`readHeader`** : il capture l'adresse d'une variable de pile. Le `Header` étant retourné **par
valeur**, le pointeur devient pendouillant et le `deinit` segfaulte — après un run par ailleurs
parfaitement correct (les 8 tokens étaient générés, le texte écrit). Le patron copié était juste
*dans son contexte* (une fonction qui consomme et libère sur place) et faux **transporté** dans
un struct retourné. Leçon : un patron du repo est un patron **avec son contexte de vie**.

**Un ollama de 21,7 GiB s'est chargé sur la 3090 entre deux runs** (`GpuBusy`, VRAM libre
2,5 GiB). Déchargé par `ollama stop` — réversible, et c'est l'action que le message d'erreur du
runner recommande lui-même. À re-vérifier avant tout run long : la garde VRAM du runner l'a
attrapé avant de perdre 8 minutes de calcul.

## 9. Reproduire

```bash
# host-only, une seconde, sans GPU :
gemma4_g12auto <ckpt> <tok> --selftest-kvdump-io /tmp/kvio

# DC2 (GPU) :
gemma4_g12auto <ckpt> <tok> --selftest-kvdump-eq <dir>/dc2.kvdump --prompt "..."

# inspection / mutants (Python) :
scripts/71_kvdump_inspect.py inspect       <dump>
scripts/71_kvdump_inspect.py make-zeroed   <dump> <out>
scripts/71_kvdump_inspect.py verdict --log <run.err.log> --ref <dump>.ref.json --mode dc3|dc4
```

Preuves versionnées : `docs/evidence/kvdump/` (`logs/` est gitignoré — précédent D10).
Tags : `gate/dc0-pass` … `gate/dc7-pass`.
