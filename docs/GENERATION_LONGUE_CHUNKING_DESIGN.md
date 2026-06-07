# Génération longue — Chunking du decode (perf) — Design

> Statut : **design validé par délégation** (Régis « tu t'en occupes », 7 juin 2026).
> Branche : `generation-longue`. Spec sœur : `GENERATION_LONGUE_DESIGN.md`, `ENGINE_LOG.md`.

## 1. Problème

Le compile/exécution XLA-CPU du forward decode **35 couches en un seul graphe** atteint ~33 Go (RAM 23 Go de la VM + 11 Go swap) dès que `.k ≳ 512` — le pic est dominé par les buffers d'attention des 35 couches matérialisés simultanément. Conséquence : L1a a PASS mais **lentement** (Eigen mono-thread `--xla_cpu_multi_thread_eigen=false` + swap thrash, ~heures pour 1020 steps). L1b/L2 seraient aussi lents.

**But** : borner le pic <23 Go (pas de swap) pour rebrancher le multi-thread Eigen → runs en minutes. Cible `L_MAX` **paramétrable, défaut 1024** ; découpage adaptatif (remontée à 2048+ ensuite).

## 2. Idée : chunker le forward decode (réutilise le mode « chain » du prefill)

`gemma4_prefill.zig` a déjà résolu le même mur pour le **prefill** (mode `chain`, ENGINE_LOG 2 juin) : forward 35 couches découpé en stages compilés séparément, `hidden`+`KV` threadés **device→device** entre stages, **sync forcée (`toSliceAlloc`) + `exe.deinit()`** après chaque stage → working set libéré → **pic ~18,5 Go**. On adapte ce pattern au **decode**.

Deux différences vs le prefill chain :
- **Cache KV threadé** : en decode le cache est entrée+sortie de chaque stage (les producer-stages 0-14 le scatter, les reader-stages 15-34 le lisent), et il grandit entre steps.
- **Boucle de steps** : les `N` stages sont compilés **une fois** et **réutilisés** à chaque step (pas de `deinit` entre steps, sinon recompilation = catastrophe).

## 3. Approche retenue (Approche 1)

Stages **dans `engine.zig`**, `runLayerGen` **réutilisé** (donc tout le calcul est partagé, une seule source de vérité). E1/E2 gardent `forward` mono-graphe → **preuve HLO intacte**. Le runner `gemma4_gen_long` orchestre `N stages × steps`.

*Rejetées* : moteur chunké séparé (duplique `runLayerGen`, 2 sources à synchroniser) ; tout-en-chunké (casse la preuve HLO E1/E2).

## 4. Composants

### 4.1 `engine.zig` — méthodes de stage
`EngineModel` ajoute une méthode générique de stage :
```
forwardStageGen(comptime start: usize, comptime end: usize, comptime first: bool, comptime last: bool)
    (self, p: Packed(cfg.two_masks), cache_in: Cache, hidden_in: zml.Tensor, ctrl: Ctrl)
    -> { hidden_out, sl_k, sl_v, fl_k, fl_v [, logits si last] }
```
- `first` : `hidden = embeds` (embed+PLE depuis `p`/`ctrl`) au lieu de `hidden_in`. Le PLE per-layer est recalculé dans chaque stage (peu coûteux ; évite de threader le tenseur `ple` 35×256).
- couches `[start,end)` via `runLayerGen` (inchangé — gère producer/reader/scatter/masque/ring par `comptime i`).
- `last` : `final norm + lm_head + softcap → logits`.
- Le **cache** est threadé : `cache_in` → scatter (producer-stages) → `cache_out`. Les reader-stages lisent les slots writers (déjà dans `runLayerGen`).

`forward` mono **reste** (E1/E2).

### 4.2 Découpage
`comptime CHUNK` (nombre de couches/stage), constante ajustée pour borner le pic <23 Go à `L_MAX` donné. Les stages couvrent `0..NUM_LAYERS` par tranches de `CHUNK`. Le prefill chain utilise 5 couches/stage (MLP double-wide 12288 des readers) ; on part de là et on ajuste empiriquement (mesure du pic). La liste des `(start,end,first,last)` est dérivée de `CHUNK` à la compilation.

### 4.3 Runner `gemma4_gen_long`
- Compile les `N` stages **une fois** (boucle comptime sur les tranches).
- Pour chaque step : exécute les stages en séquence. Entre stages, réinjecte `hidden_out`→`hidden_in` et le cache (`sl_k/sl_v/fl_k/fl_v`) **device→device** (override des buffers, pas de copie host). **Sync forcée** après chaque stage (matérialiser `hidden_out`) pour libérer le working set.
- Le dernier stage produit `logits` → argmax → compare `expected[step]`.
- Entre steps : le cache final (sortie du dernier producer-stage) est threadé vers le step suivant.

## 5. Mémoire (point délicat)

Pic visé ~1 stage (`CHUNK` couches). Les `N` exe compilés coexistent (code + poids partagés via `eng_buf`), mais le **working set d'exécution** doit être libéré après chaque stage-call via sync.

**⚠️ Inconnue n°1** : le prefill chain `deinit` chaque exe (one-shot) ; le decode **réutilise** les exe (boucle steps) → on dépend de XLA pour libérer le working set après call+sync **sans accumuler** entre stages réutilisés. À **mesurer tôt** (un step, monitorer le pic). Replis si le pic monte : (a) `CHUNK` plus petit ; (b) `deinit`+recompile par stage seulement au 1er step puis cache des exe ; (c) limiter le nombre d'exe vivants.

## 6. Validation

- **Non-régression** : E1/E2 PASS (forward mono inchangé) + preuve HLO `diff -rq` toujours valide.
- **Équivalence (oracle)** : re-run **L1a avec le moteur chunké → 1020/1020** tokens (mêmes que la version mono). Le chunking ne change pas le calcul, seulement l'exécution.
- **Mémoire/perf** : pic <23 Go mesuré (`free -h` pendant le run), **swap ≈ 0**, **multi-thread Eigen rebranché** → run L1a en minutes. C'est le critère de succès.

## 7. Hors-scope
- Pas de chunking pour E1/E2 (mono suffit, `.k=8`).
- Tuning fin `CHUNK`/`L_MAX` + remontée à 2048 : après validation à 1024.
- L1b/L2 : bénéficient du moteur chunké mais sont des gates ultérieurs (plan principal).
