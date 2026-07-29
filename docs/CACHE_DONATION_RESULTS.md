# Donation des caches KV — Résultats (chantier du 26 juillet 2026, soir)

> **⚠ Portée de la claim « == HF »** (passe de nuance, chantier `generation_config` du 29 juil 2026).
> Partout dans ce document, « == HF » signifie **même argmax sur les logits bruts** — un critère
> plus strict que comparer deux `generate()`, mais **pas** le même énoncé. Jusqu'au 29 juil le
> portage n'appliquait **pas** `generation_config.json` (`suppress_tokens`, EOS multiples) : la
> lecture « reproduit ce que `generate()` produirait » était **fausse**. Elle est devenue vraie
> **pour le 12B en mode libre** et reste **fausse pour les runners E2B**.
> Détail et chiffres : `docs/GENERATION_CONFIG_RESULTS.md` · `docs/FINDING_GENERATION_CONFIG.md`.

> Spec (spec+plan combinés, niveau léger) : `docs/superpowers/specs/2026-07-26-cache-donation-design.md`
> Base : PR #15 (masques in-graph) — mur M3 : OOM 8k par double-buffering des caches KV.
> Branche `cache-donation`.

## Verdict en une ligne

**Le 12B QAT décode 8028 positions sur la 3090.** La donation des 4 buffers de cache
(`reuseBuffer` au return de `G12Step.forward` — `engine.zig` à 0 octet) supprime le
double-buffering qui bloquait 8k ; l'aliasing est vérifié dans le module compilé et le mur
M3 est levé.

## Le changement

4 appels chaînés dans le return du runner (gemma4_g12auto.zig, hérité par les 3 cibles
1280/4k/8k) :

```zig
return .{ t5.values, t5.indices, logits,
    slk.reuseBuffer(cache.sl_k), slv.reuseBuffer(cache.sl_v),
    flk.reuseBuffer(cache.fl_k), flv.reuseBuffer(cache.fl_v) };
```

`reuseBuffer` (zml tensor.zig:254) → attribut `tf.aliasing_output` sur l'argument
(module.zig:437) → input_output_alias PJRT. Précédent upstream : lfm2. La boucle host
respectait déjà le contrat (rebind systématique, zéro relecture des buffers donnés).

## Gates

| Gate | Critère | Mesure | Verdict |
|---|---|---|---|
| **D0** moteur intact | `git diff` engine.zig vide | **0 ligne** ; builds verts | **PASS** |
| **D1** équivalence 1280 | 48/48 ids == témoin + vacuity q=1024 | `cmp` vide ; vacuity : bit-identiques ≤1023, divergence **exactement à q=1024** ; pas de crash sur les `deinit` post-donation (R1 n'a pas mordu) ; pic VRAM 16 680 MiB | **PASS** |
| **D2** équivalence 4096 | 124/124 ids == témoin | `cmp` vide + early-stop même step ; 8,7 tok/s ; pic 22 234 MiB | **PASS** |
| **D3** re-sonde 8k | verdict mesuré | **PASS EXÉCUTION** : 2 passes complètes de **8028 steps** teacher-forcés (exit 0 — M3 mourait au step 0), vacuity 8k : bit-identiques sur 1024 positions puis divergence **exactement à q=1024** (la fenêtre mord à 8k), compile 38,9 s, **pic VRAM 22 234 MiB / 24 576** | **PASS** |

Aliasing vérifié dans le HLO du module 8k (`grep aliasing` = présent, dump `d3_hlo`).
Logs : `logs/d1_d2.log`, `logs/d3_probe.log`.

## Le fait notable — le pic VRAM ne bouge pas, et c'est normal

Les pics mesurés (16 680 MiB à 1280, 22 234 MiB à 4k **et à 8k**) sont identiques
avant/après donation. Le pic est fait par le **transitoire de compile/autotune** (finding
batching : « le plafond n'est pas la VRAM mais le compute » ; l'autotune alloue ses buffers
d'essai sans connaître la donation). Ce que la donation change n'est pas le pic mais
l'**alloc au premier step** : le buffer cache de sortie (2,50 GiB à 8k) n'existe plus —
c'est exactement là que M3 mourait. Conséquence : 4k et 8k plafonnent au même pic, et
la marche suivante (12k-16k ?) se jouera sur les postes linéaires restants (caches
~8,4 GiB à 12k) + ce même transitoire.

## Périmètre de la claim (inchangé)

« == HF-fp32 STRICT » reste la claim de PR #13 (**4041 positions**), héritée par les gates
d'équivalence D1/D2. La sonde 8k est **technique** : stabilité, VRAM, fenêtre — aucune claim
de fidélité HF au-delà de 4041 positions (décision pré-enregistrée, pas d'oracle 8k).

## Notes

- Le HLO de g12auto n'est PAS byte-identique et ne doit pas l'être (les annotations
  d'aliasing font partie du module) — gates d'équivalence de sortie, comme M1/M2.
- R1 (deinit d'un handle donné) : pattern PJRT normal confirmé par les runs — aucun
  double-free, aucun crash, repli non exercé.
