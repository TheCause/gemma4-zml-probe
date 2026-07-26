# Spec — Donation des buffers de cache KV (lever le double-buffering, ouvrir 8k)

> **Date** : 2026-07-26 · **Niveau** : léger (continuité du chantier masques in-graph,
> enchaînement décidé par Régis) · **Base** : PR #15 mergée (`5f3e236`) — mur M3 chiffré :
> OOM 8k par double-buffering des caches KV (alloc 2,50 GiB = un cache sliding),
> cf `docs/MASKS_INGRAPH_RESULTS.md` §M3.

## 1. Problème

Le step compilé (`G12Step.forward`) prend les 4 caches KV en entrée et les retourne en sortie.
Sans donation, XLA/PJRT alloue les sorties SÉPARÉMENT → les caches existent **en double**
en VRAM (2 × 5,6 GiB à 8k, 2 × 2,8 GiB à 4k). C'est le mur 8k résiduel (le statique sans
double serait ~17,3 GiB / 24).

## 2. Design — donation côté RUNNER, moteur intact

ZML expose la donation par tenseur : `Tensor.reuseBuffer(origin)` (tensor.zig:254 — marque
l'output comme aliasable sur l'argument `origin` ; collecté par module.zig:313 à la compile).
Précédent upstream : `examples/llm/models/lfm2/model.zig:243` (`cache.reuseBuffer(cache_)`).

Application : dans `gemma4_g12auto.zig`, `G12Step.forward` (:780) retourne les 4 caches issus
de `forwardStageGen` — on les marque au return :

```zig
return .{ t5v, t5i, logits,
    slk.reuseBuffer(cache.sl_k), slv.reuseBuffer(cache.sl_v),
    flk.reuseBuffer(cache.fl_k), flv.reuseBuffer(cache.fl_v) };
```

- **`engine.zig` : 0 octet modifié** (la donation est une propriété du module compilé par le
  runner ; `cache` est un argument de `G12Step.forward` → `id_to_argument` le résout).
- Les 3 cibles 12B héritent (g12auto/g12a4k/g12a8k = même corps `G12Auto`).
- Aucun autre runner touché ; aucune neutralité HLO à prouver ailleurs (diff engine vide).
- **Compatibilité de la boucle host** : la boucle rebinde déjà les caches de sortie comme
  entrée du step suivant (`:1279`, vacuity `:1173`) et ne relit jamais les vieux buffers ;
  le vacuity recrée des caches zéros par passe. Contrat `reuseBuffer` (« caller not allowed
  to reuse the donated input buffer after the call ») : déjà respecté.

## 3. Non-objectifs

- Ring buffer sliding 1024 (`ring=true`) — chantier distinct.
- Donation des autres inputs (poids, Packed, tok/step) — YAGNI, seuls les caches doublent.
- Claim de fidélité HF au-delà de 4041 positions (périmètre inchangé, cf PR #15).

## 4. Gates pré-enregistrés

| Gate | Contenu | Critère PASS |
|---|---|---|
| **D0** moteur intact | `git diff` de la branche sur `engine.zig` | **vide** ; builds 12B verts |
| **D1** équivalence 1280 | run libre 48 + vacuity replay 1178 (`--no-prealloc`) | **48/48 ids == `logs/mi_witness_1280_ids.safetensors`** (cmp vide) ; vacuity : bit-identiques ≤1023, divergence **exactement à q=1024** ; pic VRAM publié |
| **D2** équivalence 4096 | `gemma4_g12a4k`, même protocole que M2 | **124/124 ids == témoin** + early-stop même step ; pic VRAM publié (réf ère tables : 22 234 MiB) |
| **D3** re-sonde 8k | `gemma4_g12a8k` `--no-prealloc --window-vacuity` replay 8000 (fixture M3 existante) | **verdict mesuré publié** : soit {2 passes complètes 8028 steps, divergence à q=1024, pic VRAM, tok/s}, soit mur suivant chiffré |

NB : le HLO de g12auto N'EST PAS attendu byte-identique (les annotations d'aliasing font
partie du module) — les gates D sont des gates d'**équivalence de sortie**, comme M1/M2.
Règle d'arrêt : D0-D2 FAIL → STOP, diagnostiquer (pas de requalification sans décision Régis).

## 5. Risques

| # | Risque | Réponse |
|---|---|---|
| R1 | `old_cache.deinit()` post-donation (g12auto:1174-1177, :1279+) = destroy d'un handle consommé → double-free/crash | Pattern PJRT normal (destroy de handle ≠ libération mémoire donnée) ; si crash au premier run D1 : retirer les deinit des buffers donnés et re-mesurer (fuite de handles bornée par la boucle) |
| R2 | XLA refuse l'aliasing en silence (pas d'erreur, pas de gain) | Le verdict est le **pic VRAM mesuré** (D1-D3), jamais la théorie |
| R3 | 8k OOM malgré la donation (activations/scratch) | Prévu : D3 publie le mur suivant chiffré, D0-D2 suffisent à la brique |

## 6. Livrables

`gemma4_g12auto.zig` (4 reuseBuffer + commentaires), `docs/CACHE_DONATION_RESULTS.md`
(gates D0-D3, pics VRAM avant/après), PLANNING (item donation clos, verdict 8k), grep
anonymisation §5.4 avant push, PR.
