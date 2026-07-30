# D10 — zéro allocation par step : résultats

> **Statut : LES 9 GATES SONT VERTS** — `S-AC`, `AL-BASE`, `AL-0`, `AL-VAC`, `V-EQ`, `EQ-48`,
> `EQ-124`, `EQ-PONT`, `G-0`, `AL-RSS`, plus les mesures publiées `M-D10` et `M-PIN`. Les trois
> contre-preuves (mutant M1) ont produit des FAIL **vus et archivés** (`docs/evidence/d10/`).
> Ce qui n'y porte pas de chiffre n'a pas été mesuré.
>
> Spec : `docs/superpowers/specs/2026-07-30-d10-zero-alloc-design.md` (rév. 4, committée AVANT
> toute mesure — le git log fait foi : `7a1437b` → `7df0cd9`) ·
> Plan : `docs/superpowers/plans/2026-07-30-d10-zero-alloc.md` (rév. 3) ·
> Build : `./bazel.sh build -c opt --@rules_zig//zig/settings:mode=release_fast
> --@zml//platforms:cuda=true //examples/rqz:gemma4_g12auto //examples/rqz:gemma4_g12a4k` —
> le mode est PROUVÉ par la bannière `BUILD: mode=ReleaseFast` greppée dans chaque log.

## 1. Ce que le chantier livre

**La boucle de décodage du 12B (et la boucle vacuity) ne fait plus AUCUN appel à l'allocateur
Zig par step** — et cet interdit, écrit au §5 de la spec sampling puis violé par son propre
chantier (dette D10) et laissé sans gate (dette D3), est désormais **GARDÉ** par un
compteur-gate toujours actif (arbitrage B9) et un critère VmRSS (B10).

Par step de génération, AVANT → APRÈS :

| Poste | Avant | Après |
|---|---|---|
| Chemin B (logits) | 2 allocs × 1 MiB + 3 parcours d'1 MiB (`toSliceAlloc` + memcpy interne + `@memcpy`) | `toSlice` **direct dans `work`** : 0 alloc, 0 copie (C2) |
| Top-5 | 4 allocs (40 o) | `getValue` → pile : 0 alloc, dtype f32 vérifié en plus (C3) |
| `exe.args`/`results` | 6 allocs, ~293 KiB (dupe de 1006 `Shape`) | hissés hors boucle : 0 (C4) |
| ArrayLists | croissances amorties | pré-réservées + `appendBounded` — garde active en ReleaseFast, là où `appendAssumeCapacity` est une UB silencieuse (C5) |
| Vacuity (instrument) | 2 allocs × 1 MiB/step | buffer par-mode + `toSlice` (C6) |
| **Total** | **12 appels, ≈ 2,29 MiB/step** | **0 appel, 0 octet** |

`work` est alloué une fois en mémoire **DMA-mappée** (`PIN: ON`, repli écrit, `--no-pin` pour
l'A/B) — et l'A/B a montré que ça ne change rien (§4, P6 réfutée).

## 2. Verdicts des 7 claims pré-enregistrées

| # | Claim | Verdict | Chiffres |
|---|---|---|---|
| **P1** | L'inventaire de lecture est juste | **CONFIRMÉE** | AL-BASE : **12,10 appels/step** ∈ [10;14] et **2 397 100 o/step** ∈ [2,2;2,6] MiB — l'inventaire prédisait 2 397 096 : **écart 4 o/step** (remap amortis). Régime neutre : 299 955 mesuré vs 299 944 calculé (0,004 %) |
| **P2** | Le hissage tient à l'exécution | **CONFIRMÉE** | `ALLOC-LOOP: alloc=0 resize=0 remap=0 free=0 bytes=0` sur 87, 828 et 228 steps (armé, long, oracle) ; `ALLOC-VAC:` quatre zéros sur 2 356 steps. Compteur vivant : `ALLOC-TOTAL: alloc=757` |
| **P3** | Les allocs/copies coûtaient une part mesurable des 1 514 µs | **CONFIRMÉE, modèle chiffré dépassé** (issue « < 800 µs » pré-enregistrée) | `d2h` : **1 514 → 447,1 µs** (sans pinned, config nominale). L'économie (1 067 µs) dépasse le modèle (~380 µs) — voir §5 : une part revient probablement au MODE DE BUILD de la mesure D9, pas aux allocs |
| **P4** | La sémantique est inchangée | **CONFIRMÉE** | EQ-48 **bit-identique** · EQ-124 **110/110** (run 4k : 124 tokens, même arrêt EOS 106 que le témoin) · EQ-PONT **1 088 steps comparés, 0 désaccord, 0 égalité exacte** (antécédent suppress k=1 au 2ᵉ essai — run 800 tokens, écrit) · V-EQ n_ident=1024, q=1024 == témoin |
| **P5** | Rien ne touche le graphe | **CONFIRMÉE** | md5 HLO **identique** b0/b1/b2 (`297679847aa04b719942d75d093adf2b` — le même que S2-G). Contre-preuve VUE : un op ajouté → `8151d080…`, jeté |
| **P6** | Le pinned lève le goulot D2H | **RÉFUTÉE par l'A/B** | `d2h` PIN ON **441,7 µs** vs OFF **447,1 µs** : **5,4 µs d'écart, dans le bruit**. Le « 0,65 Go/s faute de pinned » de D10 mesurait en réalité les allocs+copies que C2 a supprimées ; après C2 le transfert fait **2,37 Go/s sans ni avec pinned**. (Le seuil « < 500 µs » de P6 est techniquement atteint mais VIDE — le OFF y est aussi : c'est l'A/B qui juge) |
| **P7** | Le VmRSS ne dérive plus | **CONFIRMÉE** | `RSS-DELTA: 148 KiB` < 5 120 (baseline b1 : 164 KiB — le seuil était fondé par la mesure AVANT jugement) |

## 3. Coût du chemin B — M-D10 (config nominale `top_k 64 top_p 0.95 seed 42`, n=128)

| | Bloc complet | D2H seul | Warpers |
|---|---|---|---|
| Avant (D9, 29 juil) | 3 796 µs = 3,6 % d'un step | 1 514 µs (« D2H+copie ») | 2 268 µs |
| Après (PIN OFF) | **908,7 µs = 0,86 %** | **447,1 µs** | **447,1 µs** |
| Après (PIN ON) | 915,5 µs | 441,7 µs | 459,9 µs |

Le bloc chemin B coûte désormais **4,2× moins** qu'avant le chantier.

## 4. Les contre-preuves — trois FAIL vus, pas déclarés (mutant M1, `docs/evidence/d10/`)

Mutant : `alloc(u8, 1<<20)` + `@memset` (un mmap non écrit n'est PAS résident — sans le memset,
VmRSS ne dérive pas et la « contre-preuve » aurait prouvé l'inverse), jamais libéré, deux sites
HORS du `if (pathArmed)` (le run AL-RSS neutre doit les traverser), jamais committé :

| Gate | FAIL vu | Prédit |
|---|---|---|
| AL-0 | `ALLOC-LOOP: alloc=87` sur 87 steps | `alloc=steps` ✓ |
| AL-VAC | `ALLOC-VAC: alloc=152` sur 152 steps | `alloc=steps` ✓ |
| AL-RSS | `RSS-DELTA: 184 364 KiB` | « ~184 320 KiB » (180 tokens × 1 MiB) — **à 44 KiB près** |

Retour sain prouvé après chaque mutation (S-AC re-PASS, AL-0 re-PASS). Les instruments G-0 et
S-AC ont eux aussi été **vus échouer** (op ajouté → md5 différent ; `n_alloc += 1` retiré →
FAIL 4/6) : aucun gate de ce chantier ne repose sur un contrôle jamais vu mordre.

## 5. ⚠ FINDING — le « coût structurel » de D9 était en partie un artefact de MODE DE BUILD

Les warpers passent de **2 268 → 261,1 µs** (`--top-k 1`) sans qu'une ligne de leur code ait
changé. L'explication la plus probable : la commande du build « opt » de D9 est **perdue**, et
`-c opt` seul laisse le frontend Zig en **debug** (le mode rules_zig est un flag indépendant,
défaut `debug` — établi par la revue de spec, bloquant n°2). La mesure D9 « en opt le bloc coûte
autant » comparait donc vraisemblablement deux builds au frontend Zig debug. b2 est le premier
build au mode **prouvé** (`BUILD: mode=ReleaseFast` dans chaque log) — l'ambiguïté meurt pour
tous les chantiers suivants. La conclusion de D9 (« coût structurel — parcourir 262 144 logits
trois fois ») est **rectifiée** : à 0,86 % d'un step, le chemin complet est ~4× moins cher que
publié, et le « 0,96 ns par élément-opération » était du code debug.

## 6. Dettes — état après chantier

| # | Dette | État |
|---|---|---|
| DA-1 | Allocations C/PJRT invisibles au compteur Zig | **BORNÉE (30 juil, `LD_PRELOAD` mallocount.so)** : deux runs différentiels (88 vs 828 steps) — le run 10× plus long fait *moins* de mallocs (215,28 M vs 215,62 M, jitter inter-compiles ±335 K) ⇒ **< ~450 mallocs C/step**, indiscernable de zéro par méthode différentielle. Les ~215 M d'appels (30,5 Go cumulés) sont le load+compile. Faire mieux = marqueurs in-process dans le .so, disproportionné au vu de la borne ; AL-RSS borne déjà la dérive nette |
| DA-2 | AL-0 exercé sur ses runs seulement | **Amortie en continu** : le compteur toujours actif a déjà re-vérifié quatre zéros sur les runs 800 tokens, oracle 200 et vacuity 2 356 steps |
| DA-3 | `n_shards` supposé | **SOLDÉE** : `shards=1` publié à chaque run |
| DA-4 | Variantes 4k/8k non re-exécutées | **Réduite** : la 4k est exercée par EQ-124 ; la 8k reste (graphe intouché G-0, code host identique) |
| DA-5 | Logs bruts S2 hors de `logs/` | **SOLDÉE** : les 30 logs `/tmp/s2*` rapatriés sur M1 avant qu'un reboot ne les efface |
| DA-6 | `std.Io.Threaded` sur le gpa non wrappé | **Ouverte** (structurelle) — contrepartie : compteurs mono-thread corrects |

## 7. Écarts au plan, consignés

1. Contre-preuve G-0 exécutée sur l'état C1+mutation (pas b0+mutation) : C1 est host-only, la
   preuve « l'instrument détecte un op ajouté » est identique — et le retour sain a prouvé du
   même coup que C1 ne touche pas le graphe.
2. Builds groupés C2-C7 (un build au lieu de six) : la VM était occupée par les runs baselines ;
   bissection possible par les 6 commits séparés. Le build a réussi du premier coup.
3. Mutant AL-VAC sur la fixture 48 (2×76 steps) au lieu de u9 (2×1178) : même preuve
   (`alloc=steps`), 10 min de GPU économisées.
4. `logs/` est gitignoré (convention du 9 juil) : les preuves vont dans **`docs/evidence/d10/`**
   (décision Régis, 30 juil) — la convention logs/ reste intacte.
5. Le RSS baseline b1 a été pris sur un run sorti en `A1Mismatch` (bistabilité @47, variante B
   tirée — la marge mesurée ce jour : 0,0019). La mesure est valide : `RSS-DELTA` est émis AVANT
   le verdict oracle, exactement pour ça. La bistabilité prompt-spécifique du FINDING du 29 juil
   s'est donc re-manifestée en conditions réelles, à la position cartographiée.

## 8. Périmètre de la claim

« Zéro allocation par step » signifie : **zéro appel à l'allocateur Zig du runner dans les
boucles de step** (génération et vacuity), compté par un wrapper au point de substitution unique
`init.gpa`, publié à chaque run. Les allocations du C de PJRT (DA-1) et de `std.Io.Threaded`
(DA-6) sont hors de cet instrument — bornées indirectement par AL-RSS (< 5 120 KiB sur 180
tokens). La claim « == HF » du projet est inchangée (EQ-48/EQ-124/EQ-PONT le prouvent).
