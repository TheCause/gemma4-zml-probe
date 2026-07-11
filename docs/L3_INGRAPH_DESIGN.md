# Design — L3 in-graph : `gemma4_gen_auto` token→token

> Item [M] du PLANNING (« boucle de décode dans le graphe »), défini historiquement dans
> `GENERATION_LONGUE_DESIGN.md` §gates (L3 : « le forward devient token_in → token_out, le host
> ne thread qu'un scalaire ») et `GPU_PORT_PLAN.md` §9.6 (tables en VRAM bf16, gather in-graph).
>
> Décisions de cadrage (Régis, 11 juil 2026) : L3 **remplace** la boucle actuelle de
> `gemma4_gen_auto` (pas de coexistence, rollback = git) ; critère perf = **mesurer sans objectif
> dur** (fidélité obligatoire, non-régression perf) ; architecture = **wrapper au niveau runner**
> (`engine.zig` intact d'un octet). Design durci par un cycle /iterate 7 passes (amendements
> intégrés ci-dessous, marqués [it.N]).

## 1. But

Éliminer les allers-retours host par step de la boucle de décode. Aujourd'hui, chaque step coûte :
lecture disque de 2 lignes d'embeddings (`EmbedGather`), 2 `Buffer.fromBytes` + 1 `Buffer.scalar`
H2D (embeds, embptls, step), l'appel graphe, D2H des logits complets (~1 Mo fp32) et un scan
top-5 host. Après L3 :
le host envoie **un scalaire u32** (le token à feeder) et reçoit `{next_tok, top5}` (~48 octets).
La structure de la boucle host (prefill-par-decode, early-stop EOT, gardes L_MAX) est conservée.

## 2. Architecture — wrapper `StepTok`, `engine.zig` intact

Un struct local à `gemma4_gen_auto.zig` compose, dans UN graphe compilé par `compileFn` :

```
StepTok.forward(model, tabs, tok u32, packed, cache, ctrl) :
    e    = model.embed_tokens.gather(.{ .voc = tok })  // [1,1,1536] bf16 — table DÉJÀ device
                                                       // (lm_head tied) : zéro Go ajouté
    el   = tabs.eptl.gather(.{ .voc = tok })     // [1,1,8960]  bf16, ligne brute
    logits, cache' = Model.forwardStep(e, el, packed, cache, ctrl)   // INCHANGÉ
    next = logits.argMax(.voc)                   // in-graph
    t5   = logits.topK(.voc, 5, .{})             // diagnostic --oracle, ~40 octets
    return { next, t5, cache' }
```

- Le scaling √H/√D des embeddings est DÉJÀ dans `forwardStep` (contrat actuel : `EmbedGather`
  livre des lignes brutes, « AUCUN scaling host ») — le gather in-graph livre la même chose.
- `argMax`/`topK` : primitives ZML existantes (`tensor.zig:2913/3096`) ; `gather` validé
  depuis P5.4 (bit-exact).
- **[it.5] Nommage court obligatoire** (`StepTok`, `Tabs`) : le quota comptime 1000 de
  `pjrt.zig structSize` (indexOf sur `@typeName`) a déjà tué un runner au nom long
  (`gemma4_attention_softmax` → `gemma4_softmax`). Garde-fou associé : le patch local
  `@setEvalBranchQuota(100_000)` dans `pjrt.zig` (à réappliquer si le workspace ZML de la 3090
  est resynchronisé). Le plan placera un **build 3090 tôt** pour fail-faster sur ce piège.

### 2.1 Chargement des tables [it.7, amendé revue]

- `embed_tokens` : **aucun chargement nouveau** — le poids est déjà device-résident dans `Model`
  (lm_head tied, cf `engine.zig` `dotPrec(…, self.embed_tokens, .d)`) ; le gather le réutilise
  (sous réserve du tag `.voc` sur l'axe vocab — sinon repli : le dupliquer dans `Tabs`, +0,8 Go).
- `embed_tokens_per_layer` : SEULE table ajoutée — struct `Tabs` avec le nom complet du checkpoint
  (`model.language_model.embed_tokens_per_layer.weight`, clé exacte = `EPTL_KEY` actuel,
  [262144,8960] bf16 ≈ **4,7 Go**), matérialisé par le MÊME `TensorStore`/`zml.io` que les poids
  du modèle — un seul chemin de lecture, offsets gérés par la lib.
- **`EmbedGather` est supprimé intégralement** (avec ses offsets absolus recalculés à la main et
  son commentaire périmé « ~1,6 Go » de l'ère fp32). Pic mémoire au chargement surveillé par
  `mem_probe`. VRAM ajoutée par L3 ≈ **4,7 Go** (pas 5,5).

### 2.2 Boucle host après L3

Par step : 1 `Buffer.scalar` (tok) + 1 `Buffer.scalar` (step ctrl) H2D → call → D2H de
`next_tok` + `top5`. Le host décide quoi feeder : id du prompt en phase prefill, `next_tok`
retourné en phase génération. `top5Of` (scan host) est supprimé ; `gen_top5` est alimenté par le
`topK` in-graph. Le motif cache-swap actuel (4 buffers échangés/deinit par step) est conservé —
cf non-objectif donation §6.

## 3. VRAM — le seuil de la garde devient un livrable de mesure [it.1]

Usage réel attendu ~8,5 + 4,7 ≈ **13 Go** — mais ce 13 est un a priori (8,5 mesuré G2.1 sur une
autre config + 4,7 théorique). Doctrine §9.7 (« mesurer, pas présupposer ») :

- Sémantique BFC **vérifiée** (11 juil, `platform.zig:544` + PJRT) : `memory_fraction 0.90`
  préalloue 0.90 × la mémoire **libre à l'init**. Contrainte : `0.90 × libre ≥ besoin réel`.
- G3 loggue `mem_probe` post-load / post-compile ; **seuil final
  `MIN_FREE_VRAM_GIB = ceil(mesuré / 0.90) + 1`**, consigné avec la mesure dans la doc.
- Valeur provisoire dans le code pendant le chantier : 16. Les gates de la garde VRAM (V1/V3 de
  `VRAM_CHECK_DESIGN.md`) sont re-runs avec le seuil final.
- Doc à corriger en conséquence : la note G2.1 « le banc tiendrait sur une carte 12 Go » ne vaut
  plus pour `gemma4_gen_auto` post-L3.

## 4. Vigilance numérique nommée : les ties d'argmax

L'argmax passe d'un scan host (`top5Of`, premier max rencontré) à l'op XLA device. Sur égalité
stricte de logits fp32 — ou sur les marges fines déjà observées (bifurcation 0.006 au step ~590,
cf A2) — les deux peuvent départager différemment sans qu'il y ait infidélité. Conséquences au
design :

- Le critère long est **différentiel** (« L3 ≥ replay »), jamais N/N (cf G2b).
- Le `top5` in-graph (indices + valeurs fp32) est le diagnostic embarqué : en cas de divergence
  au step k, la marge `t5.vals[0] - t5.vals[1]` qualifie « tie/marge fine » sans re-run
  instrumenté.

## 5. Validation (fidélité obligatoire ; perf mesurée, sans objectif dur)

| Gate | Contenu | Critère |
|---|---|---|
| **B0 — bench avant** | HEAD actuel (post-PR #6), prompt court `--oracle` 48 + run long ; **temps de prefill et tok/s de génération mesurés SÉPARÉMENT** [it.6] | chiffres de référence consignés |
| **SG — selftest gather in-graph** [it.4] | `--selftest-gather` **converti** (pas supprimé) : mini-graphe `{tables, ids fixture} → gather`, comparé aux embeds/embptls de la fixture A1 existante. ⚠ Changement de catégorie : ce mode devient un mode **GPU** (graphe compilé) — la garde VRAM s'y applique désormais telle quelle (comportement assumé, plus de « host-only » pour ce flag) | bit-exact (critère inchangé) |
| **G1 — fidélité courte** | `--oracle` fixture 48 steps (protocole A1) | 48/48 == HF |
| **G1v — non-vacuité** | perturbation : corrompre `fed` de la fixture (convention repo) | le compare DOIT FAIL |
| **G2 — early-stop** | « Paris » + prompt libre FR (protocole A3) | EOT naturel, réponse correcte |
| **G2b — fidélité longue différentielle** [it.3] | protocole A2 rejoué sur la fixture longue existante | **L3 ≥ replay** (bifurcation au même point ou plus tard ; marge qualifiée via top5) |
| **G3 — bench après + mesure VRAM** | mêmes mesures que B0 + `mem_probe` post-load/post-compile → seuil final de la garde [it.1] | fidélité PASS + **L3 ≥ boucle actuelle** en tok/s de génération ; attente (non bloquante) ≥ 109 tok/s (replay) |
| **VG — garde VRAM re-run** | V1/V3 de `VRAM_CHECK_DESIGN.md` avec le seuil final | mêmes critères |

## 6. Non-objectifs (explicites)

- **Donation des buffers de cache** [it.2] : hors scope — le churn du swap (~38 Mo/step
  device-device) est petit devant les transferts supprimés, et l'API de donation ZML n'a aucun
  usage exemple consulté. **Spike optionnel** en fin de plan uniquement si G3 montre que le swap
  pèse (comparaison step-time L3 vs replay).
- **Vrai prefill S>1 (« option B »)** [it.6] : hors scope — le prefill reste P appels graphe.
  B0/G3 en mesurent le coût séparément : c'est la baseline du chantier batching, pas une dette
  cachée.
- **Boucle `while` complète in-graph** (multi-steps par call) : c'est un éventuel L4 — early-stop
  dynamique complexe, diagnostic par step perdu. Le L3 du repo = token_in → token_out par call.
- Sampling (greedy only), batching, multimodal : inchangés hors scope.

## 7. Périmètre exact

- `zml_runner/gemma4_gen_auto.zig` : struct `Tabs` + `StepTok`, chargement tables via TensorStore,
  boucle réécrite (suppression `EmbedGather` + `top5Of`), `--selftest-gather` converti [it.4],
  seuil garde VRAM ajusté [it.1].
- `docs/DOCUMENTATION.md` (§2.2 usage + perf + note VRAM), `PLANNING.md` (item [M] L3),
  `docs/G2_BF16_FIDELITY.md` ou note G2.1 : caveat « 12 Go » (une ligne),
  `docs/VRAM_CHECK_DESIGN.md` : seuil final [it.1] + `--selftest-gather` reclassé GPU (§1/§4).
- **Intacts** : `engine.zig` (invariant d'un octet), `BUILD.bazel`, tous les autres runners
  (`gemma4_gen_long_gpu` reste le replay de référence pour G2b/B0).
- Chantier suivant (sous-projet séparé, propre cycle spec→plan) : batching / flash-attention.
