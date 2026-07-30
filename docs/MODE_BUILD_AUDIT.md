# Audit — les mesures prises en mode de build ambigu (30 juil 2026)

> **Origine** : le chantier D10 (`docs/D10_RESULTS.md` §5) a établi que la commande du build
> « opt » historique est **perdue**, et que `-c opt` seul laisse le **frontend Zig en debug** —
> le mode rules_zig est un flag Bazel indépendant (`--@rules_zig//zig/settings:mode`, défaut
> `debug`, valeurs `debug`/`release_safe`/`release_small`/`release_fast`).
> Effet mesuré : **warpers 2 268 → 261 µs** (×8,7) au mode prouvé, sans une ligne de code changée.
>
> Cet audit répond à la question : **quelles autres claims du projet portent cet artefact ?**
> Méthode : inventaire des chiffres de performance de tout le repo (docs, PLANNING, README,
> specs, plans, 90+ tags annotés, messages de merge) — **174 claims relevées, 80 à mode non
> prouvé, 76 après déduplication** — puis triage par SENSIBILITÉ au mode, pas par ancienneté.

## 1. La règle de sensibilité (ce qui décide du triage)

| Nature de la mesure | Sensible au mode Zig ? | Motif |
|---|---|---|
| **Chrono de code host Zig** (warpers, boucles host, copies host, parsing) | **OUI, massivement** (×8,7 mesuré) | C'est du code Zig exécuté sur CPU : `debug` n'optimise rien |
| **Débit tok/s GPU-bound** | **Faiblement** (**+7,8 % mesuré**, cf §3) | Le GPU domine ; la part host par step est marginale |
| **Différentiels appariés** (ratio A/B, même binaire, même session) | **NON** | L'artefact est présent des deux côtés et s'annule dans le ratio |
| **Équivalences** (ids identiques, bit-exact, md5 HLO) | **NON** | Le mode ne change pas les valeurs calculées, seulement leur vitesse |
| **VRAM device**, **temps de compile XLA** | **NON** | Hors du code Zig frontend |

## 2. Panier A — chronos host purs, à re-mesurer

**Tous déjà re-mesurés et publiés par le chantier D10** (aucun GPU supplémentaire nécessaire) :

| Claim d'origine | Source | Valeur au mode PROUVÉ |
|---|---|---|
| warpers **2 268 µs/step** (« ~0,96 ns/élément, ordre de grandeur normal ») | `SAMPLING_RESULTS:49`, spec sampling `:271`, merge `c8f78c7` | **261 µs** (`--top-k 1`) / 447-460 µs (nominal) — **×8,7 / ×5** |
| D2H+copie **1 514 µs** (« 0,65 Go/s, 48× sous PCIe 4.0, faute de pinned ») | `SAMPLING_RESULTS:54`, `edf3831` | **441-447 µs** (**2,37 Go/s**) — et l'hypothèse *pinned* **RÉFUTÉE par A/B** (ON 441,7 vs OFF 447,1) |
| bloc chemin B **3 796 µs = 3,6 % d'un step** (« coût STRUCTUREL ») | `SAMPLING_RESULTS:41,49`, `PLANNING:80`, `edf3831`, `c8f78c7` | **908,7 µs = 0,86 %** — conclusion « structurel » **rectifiée** |
| D9 : « en opt 3 828,8 µs vs dbg 3 730,1 ⇒ le mode n'explique rien » | `edf3831`, spec sampling `:271` | **Le cas d'école** : ce « opt » était `-c opt` seul (frontend Zig resté debug) — les deux bras étaient du debug |

**Reste non re-mesuré, et pourquoi ce n'est pas nécessaire** : « tri complet 15,2 % d'un step »
(spec sampling `:112`) est un **microbench C++ -O3 déclaré**, pas le binaire Zig ; sa base (le
step) bouge de −3 % (§3), donc 15,2 % → ~15,7 %. Décision inchangée (« cher mais pas
rédhibitoire »).

## 3. Le chiffre-titre du projet, re-mesuré au mode prouvé

**« Le 12B tourne sur la 3090 à 9,0 tok/s »** (`gate/j2-u10-pass`, `U_12B_RESULTS:83`,
`PLANNING:196`, `README:46`) — mesuré en mode ambigu.

Mesures du 30 juil, **`BUILD: mode=ReleaseFast` dans chaque log** :

| Run | Génération |
|---|---|
| oracle 200 steps (`logs/d10_rss_b1.err.log`) | **9,7 tok/s** |
| oracle 200 steps, b2 (`logs/d10_rss_b2.err.log`) | **9,6 tok/s** |
| libre 60 tokens (`logs/d10_albase.err.log`) | **9,7 tok/s** |

**Verdict : 9,0 → 9,6-9,7 tok/s, soit +7,8 %.** La claim d'affiche tient (elle était
conservatrice) ; le débit est bien **GPU-bound**, comme l'inventaire le prédisait. Conséquence
arithmétique : la base des pourcentages passe de ~106 000 à **~103 000 µs/step** (−3 %) — tous
les % de coût du projet bougent de 3 %, aucun verdict ne change.

## 4. Panier C — insensibles par construction (aucune action)

- **Tous les différentiels appariés** : ratio de non-régression batching **0,999** (B4), A/B sdpa
  **Δ +1,0 % / +0,0 %** (S3, décision « pas de gain » → sdpa écarté), coût du dequant W4 **×2,2**
  (40,9 vs 91,7 tok/s, même session). L'artefact est des deux côtés : il s'annule.
- **Toutes les équivalences** : 48/48, 1150/1150, 1020/1020, 4041 positions, md5 HLO, cmp d'ids.
  Ce sont les claims qui portent la valeur scientifique du projet — **aucune n'est affectée**.
- **VRAM device** (22 234 MiB, 16 680 MiB, −37 % W4) et **temps de compile XLA** (37,9 s, 38,9 s).

## 5. Panier B — chiffres d'archive, marqués sans re-mesure

Les débits GPU-bound historiques (E2B 109-113 tok/s, W4 40,9 tok/s, batching 113→2 106 tok/s,
4k 8,2 tok/s, RSS host d'époque CPU) : aucune décision vivante n'en dépend au-delà de ce que les
différentiels prouvent déjà, et l'exposition est celle mesurée au §3 (+7,8 %). Ils sont
**marqués** ici plutôt que re-mesurés.

⚠ **Une exception notée** : `PLANNING:250` « B0 pré-L3 91,4/57,1 tok/s » est le bras le plus
exposé du repo (avant L3, le host rapatriait les logits complets par step). Le gate G3 en dépend
— mais c'est un **différentiel** avec le bras L3 mesuré dans les mêmes conditions, et le code
pré-L3 n'existe plus : re-mesurer serait de l'archéologie sans décision à la clé. Écrit, pas fait.

## 6. La prévention — pour que l'ambiguïté ne revienne pas

1. **`zml_runner/build_3090.sh`** : la commande canonique, avec les **deux** flags, en un seul
   endroit. Toute mesure future part de là.
2. **Bannière `BUILD: mode=<builtin.mode>`** émise par le runner (livrée par D10, C1) : chaque
   log de run porte son mode. Un gate dont le log n'affiche pas le mode attendu est
   **INEXÉCUTABLE**, pas PASS.
3. **Règle** : toute claim de performance publiée doit citer le mode ET la commande. « opt » sans
   commande n'est pas une preuve — c'est ce qui a coûté le verdict D9.

## 7. Leçon

> Un chiffre peut être **juste, reproductible et faux** : les deux bras de D9 étaient
> reproductibles et concordants (3 730 vs 3 828 µs) — et tous deux mesuraient du code debug. La
> concordance de deux mesures ne prouve pas leur validité si elles partagent le même angle mort.
> C'est la variante « instrument » de la leçon `feedback_instrument_degrade_requalifications_cascade`,
> et la raison pour laquelle le mode doit être **publié par le binaire lui-même**, jamais déduit
> de la commande qu'on croit avoir lancée.
