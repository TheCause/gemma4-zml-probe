# Spec — Masques d'attention in-graph (lever le mur quadratique {L_MAX,L_MAX})

> **Date** : 2026-07-26 · **Niveau de travail** : léger (décision Régis) · **Statut** : design validé
> (approche A, GO Régis), spec à relire avant plan.
> Base : PR #13 mergée (`e738df9`, variante contexte long 4k `gemma4_g12a4k`), PR #14 mergée
> (`ed23179`, U7-fp32). Mur documenté au commit probe ctx-long (`e251e52`) : « masques
> {L_MAX,L_MAX} quadratiques → 8k infaisable avec ce design (piste : masques in-graph) ».

## 1. Contexte et problème

Le runner 12B (`zml_runner/gemma4_g12auto.zig`, corps générique `G12Auto(comptime L_MAX)`)
précalcule côté host deux tables de masques additifs f32 :

- `masks_sliding` : `{step=L_MAX, b=1, h=1, q=1, k=L_MAX}` — fenêtre 1024 (`maskRows`,
  valeurs ∈ {0, `MASK_MIN` = −floatMax(f32)}) ;
- `masks_full` : idem, causal plein.

Elles transitent par `engine.Packed(two_masks=true)` et chaque step de décode extrait sa ligne
par `pickStep` (= `dynamicSlice(.step)`). C'est le **seul terme quadratique** du design :

| L_MAX | 2 tables f32 (device + host) | Cache KV K+V (linéaire) |
|---:|---:|---:|
| 1280 | 12,5 MiB | 0,88 GiB |
| 4096 | 128 MiB | 2,81 GiB |
| 8192 | 512 MiB | 5,62 GiB |
| 16384 | 2,0 GiB | 11,2 GiB |

Pic VRAM mesuré (probe 26 juil, `--no-prealloc`) : 16 680 MiB à 1280, **22 234 MiB à 4096**.
La croissance vient des postes linéaires (caches, RoPE, embeds factices) + du scratch
XLA (~7-8 GiB, dominé par la compile — finding batching), les masques quadratiques
s'y ajoutent et sont le seul poste qui explose asymptotiquement.

## 2. Objectif et critères de succès (décisions Régis, 26 juil)

**Objectif** : générer les lignes de masque **dans le graphe** à partir du scalaire de
position déjà threadé → tout le design devient **linéaire en L_MAX**, et sonder 8k.

**Critères de succès** (« brique + mesure 8k ») :

1. **Équivalence stricte** du chemin in-graph au chemin table, aux bornes déjà prouvées
   (1280 et 4096) — mêmes critères que PR #13 (cmp d'ids free-run, préfixe long).
2. **Neutralité** : option off → HLO **byte-identique** (discipline `two_masks`/T0).
3. **Sonde 8k technique** : nouvelle cible `gemma4_g12a8k` (= `G12Auto(8192)`) ; on
   publie le **verdict mesuré** — PASS (pas d'OOM, pic VRAM, tok/s, fenêtre qui mord)
   **ou** le mur suivant chiffré (OOM → décomposition et pointeur backlog arène XLA).
   La sonde 8k n'est **pas une promesse** : M0-M2 suffisent à clore la brique.

**Preuve 8k** : sonde technique seule (décision Régis). **Aucun oracle HF 8k** : la claim
« == HF-fp32 » reste celle de PR #13 (4041 positions), *héritée* par le chemin in-graph via
l'équivalence stricte M1/M2. Toute claim de fidélité au-delà de 4041 positions est interdite
dans les docs de ce chantier.

## 3. Non-objectifs

- == HF-fp32 teacher-forcé à 8k (exigerait un oracle M4 dédié — hors scope, décision §2).
- Réduction de l'arène/scratch XLA (item backlog distinct ; la sonde M3 peut le chiffrer,
  pas le résoudre).
- Triton paged attention, batching B>1 du 12B, masques du chemin `two_masks=false`
  (E2B mono-masque) et du prefill chunké E2B (`gchunk*`) — inchangés.
- Redimensionnement dynamique du cache (reste alloué à `.k=L_MAX`, linéaire — inévitable
  sans paged cache).

## 4. Design (approche A — option comptime dans l'engine)

### 4.1 Config

`EngineCfg` (engine.zig) gagne un champ à défaut neutre :

```zig
ingraph_masks: bool = false, // two_masks=true requis ; génère les masques depuis positions[step]
```

La fenêtre glissante n'est **PAS** comptime : elle est un **scalaire runtime** `window`
(§4.2) — c'est ce qui préserve le contre-test de vacuité « même executable, une compile,
rebind en données » (mécanisme U9-ii) : en ingraph il n'y a plus de buffer masque à
rebinder, on rebinde le scalaire `window` à la place.

Garde comptime : `ingraph_masks and !two_masks` → `@compileError`. Garde runtime côté
runner : `window > 0` sinon erreur au lancement. (Le chemin mono-masque E2B n'est pas
couvert : YAGNI.) `forwardStep`/`forwardStageStep` (les variantes non-gen) ne sont **pas
câblées** : `cfg.ingraph_masks` y lève un `@compileError` explicite — les runners 12B
n'utilisent que `forwardStageGen` (g12auto:793), l'analyse comptime paresseuse fait le
reste, la garde évite une instanciation accidentelle future.

### 4.2 Packed — 3ᵉ variant

`zml.io.load` réfléchit récursivement sur les champs (pas de champ void conditionnel —
commentaire existant engine.zig:312). `Packed` devient paramétré par un mode :

```zig
pub const MaskMode = enum { single, tables, ingraph };
pub fn Packed(comptime mode: MaskMode) type { ... }
```

- `.single` = ancien `Packed(false)` (E2B, byte-identique) ;
- `.tables` = ancien `Packed(true)` (chemin table conservé — il reste le témoin des
  gates M1/M2) ;
- `.ingraph` = `.tables` **sans** `masks_sliding`/`masks_full` (les 5 autres champs
  inchangés : embeds, embptls, cos_full, sin_full, positions), **plus** un scalaire
  `window: zml.Tensor` (`{}` i32, 4 octets) — la fenêtre glissante en DONNÉE, fournie
  par le runner (12B : 1024), rebindable pour le contre-test de vacuité (§4.5).

Compat : les call-sites existants passent de `Packed(bool)` à `Packed(mode)` — mapping
mécanique (`false`→`.single`, `true`→`.tables`), vérifié par le gate M0 (HLO byte-identique,
qui prouve que la migration de signature n'a rien changé aux graphes existants).

### 4.3 Génération in-graph (décode q=1)

Dans `forward`/`forwardStageGen`, à l'endroit exact où `pickStep(p.masks_*)` extrayait la
ligne, quand `cfg.ingraph_masks` (branche comptime) :

```
pos      = pickStep(p.positions, step)            // scalaire i32 (existe déjà)
iota_k   = Tensor.iota({k = KMAX}, .k)            // i32 (zml tensor.zig:2014, vérifié workspace 3090 le 26 juil)
full     = select(iota_k <= pos, 0.0, MASK_MIN)   // f32
sliding  = select((iota_k <= pos) and (iota_k >= pos − (p.window − 1)), 0.0, MASK_MIN)
```

`p.window` est le scalaire runtime de §4.2 (12B : 1024) — pas une constante comptime.

puis broadcast aux tags `{b,h,q,k}` attendus par `runLayerGen` (le masque est consommé par
`scores.add(mask.broad(...))` — chemin `.manual` — ou `sdpa attn_mask`).

**Valeurs strictement identiques** à `maskRows` host (g12auto:286-292) : `MASK_MIN =
−std.math.floatMax(f32)`, bornes `j > p` (full) et `j > p or j < p−SW+1` (sliding).
Mêmes constantes f32 exactes → mêmes adds → mêmes scores. L'égalité est vérifiée par
sortie (M1/M2), pas promise au bit sur les logits (non-déterminisme inter-compiles
XLA-GPU documenté, G2b).

### 4.4 Côté runner (g12auto)

- `HostTables` n'alloue/ne remplit plus `masks_sliding`/`masks_full` en mode ingraph
  (gain host : 128 MiB à 4k, 512 MiB à 8k ; gain device idem ; le `dynamicSlice` masque
  par step disparaît aussi) ; il fournit à la place le scalaire `window` (= 1024,
  vérifié > 0 au lancement).
- `maskRows` **reste** : il sert le selftest (§4.5) et le chemin `.tables` témoin.
- Le mode est **comptime par cible** (pas un flag runtime) : le graphe change. Les cibles
  12B (`gemma4_g12auto` 1280, `gemma4_g12a4k` 4096) basculent en ingraph une fois M1/M2
  PASS ; nouvelle cible `gemma4_g12a8k` (8192, `G12Auto(8192)`, ~10 lignes comme
  `gemma4_g12a4k.zig` — piège 18 : pas de `@import("root")`).

### 4.5 Selftest et non-vacuité

- Le selftest inputs actuel compare les tables host aux fixtures — la partie masques n'a
  plus d'objet en ingraph. Remplacement : on garde `maskRows` host comme référence témoin
  et on prouve l'équivalence par les sorties (M1/M2) + le contre-test de fenêtre
  `--window-vacuity` **adapté** (anti « masque devenu vide » — leçon vacuité de
  l'antécédent).
- **Adaptation du contre-test (attrapée en revue de spec)** : le mécanisme actuel
  (g12auto:1065-1110) rebinde le **buffer** `masks_sliding` ← contenu de `masks_full`
  (passe 2, même executable). En ingraph ce buffer n'existe plus — le rebind porte
  désormais sur le scalaire **`window` ← L_MAX** (fenêtre non mordante). Sémantique
  identique : passe 2 divergente de la passe 1 au-delà de q ≥ window ⇒ la fenêtre
  mordait. Toujours **une seule compile, même executable, rebind en données**.
- M3 exécute ce contre-test à 8k : la fenêtre doit mordre à q ≥ 1024 exactement
  comme à 4k.

## 5. Gates pré-enregistrés

| Gate | Contenu | Critère PASS | Machine |
|---|---|---|---|
| **M0** neutralité | cibles existantes compilées option off (`.single` E2B via e1/e2, `.tables` g12auto défaut) avant/après édits engine | HLO **byte-identique** (md5, mécanisme T0/U1) | 3090 |
| **M1** équivalence 1280 | `gemma4_g12auto` ingraph vs **témoin `.tables` = binaire buildé au commit parent (pré-bascule), out-ids archivé dans logs/** | **48/48 ids free-run identiques** (critère PR #13 défaut) | 3090 |
| **M2** équivalence 4096 | `gemma4_g12a4k` ingraph vs témoin `.tables` (même véhicule : build pré-bascule, out-ids archivé) | **300/300 préfixe identique** (critère PR #13 variante) | 3090 |
| **M3** sonde 8k | `gemma4_g12a8k` : compile, run libre long, `--no-prealloc`, `--window-vacuity` | **verdict mesuré publié** : soit {pas d'OOM, pic VRAM, tok/s, fenêtre mord}, soit mur chiffré (OOM/compile) + pointeur backlog | 3090 |

Règle d'arrêt : M0-M2 FAIL → STOP, diagnostiquer avant tout (pas de requalification sans
décision Régis — leçon vocabulaire de rigueur). M3 n'a pas d'issue FAIL : les deux issues
sont des résultats publiables.

## 6. Risques

| # | Risque | Réponse |
|---|---|---|
| R1 | Scratch XLA à 8k → OOM malgré la linéarisation (statique estimé ~17,4 GiB + scratch ~7 GiB ≈ limite 24 Go) | Prévu par design : M3 publie le mur chiffré ; la brique reste PASS sur M0-M2 |
| R2 | `select`/`iota` change la numérique | Valeurs f32 exactes {0, −floatMax} identiques au host ; vérifié par M1/M2 (sorties), pas supposé |
| R3 | Migration `Packed(bool)`→`Packed(mode)` casse un runner existant | M0 (HLO byte-identique) + build de tous les runners `two_masks` avant merge |
| R4 | Compile 8k très longue / quota comptime (`@typeName`, piège workspace pjrt) | Parade connue : patch local `@setEvalBranchQuota` pjrt.zig (à re-vérifier sur le workspace 3090) ; temps de compile = donnée publiée de M3 |
| R5 | RAM host VM (~23 Go) à 8k : caches zéros ~5,9 GiB à allouer | Marge OK ; mesurer au passage (le gain host des masques joue pour nous) |
| R6 | Longs runs distants coupés | Leçon session U7 : **nohup + log distant + stdin fermé** systématiques ; guetteurs sur états TERMINAUX seulement |

## 7. Livrables

- `engine.zig` : `MaskMode`, `Packed(mode)`, champs `EngineCfg`, génération in-graph
  (branches comptime, défauts neutres).
- `gemma4_g12auto.zig` : HostTables sans tables masques en ingraph, selftest ajusté.
- `gemma4_g12a8k.zig` + cible Bazel.
- `docs/` : résultats M0-M3 (fichier de résultats du chantier), PLANNING mis à jour,
  pièges nouveaux le cas échéant.
- Anonymisation : grep §5.4 durci AVANT tout push (aucun chemin utilisateur, alias,
  IP, ni nom de machine perso dans docs/commits).
