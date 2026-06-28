# Génération longue — Chunking du decode (perf) — Design

> Statut : **design validé** (7 juin 2026).
> Branche : `generation-longue`. Spec sœur : `GENERATION_LONGUE_DESIGN.md`, `ENGINE_LOG.md`.

## 1. Problème

Le compile/exécution XLA-CPU du forward decode **35 couches en un seul graphe** atteint ~33 Go (RAM 23 Go de la VM + 11 Go swap) dès que `.k ≳ 512` → swap thrash → **lent** (Eigen mono-thread `--xla_cpu_multi_thread_eigen=false`, ~heures pour 1020 steps). L1b/L2 seraient aussi lents.

**Cause racine (corrigée en revue)** : à `S=1` et `.k≤1024`, les tenseurs d'attention sont **minuscules** et le cache linéaire ne pèse que des dizaines de Mo — ce n'est PAS le coupable. Le pic vient des **conversions f32 des poids** (`c(layer.*)` = `convert(.f32)` sur `gate/up/down` ≈ 12288×1536 chacun, plus q/k/v/o, ×35 couches) matérialisées dans un seul graphe. **Conséquence directe : chunker par couches est le bon levier** (il borne combien de couches ont leurs poids f32 coexistant en mémoire). La métrique de succès (§6) cible donc ce pic, pas l'attention.

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
`EngineModel` expose une méthode générique :
```
forwardStageGen(comptime start, comptime end, comptime first, comptime last)
    (self, p: Packed(cfg.two_masks), cache_in: Cache, hidden_in: zml.Tensor, ctrl: Ctrl)
    -> { hidden_out, sl_k, sl_v, fl_k, fl_v [, logits si last] }
```
- `first` : `hidden = embeds` (embed+PLE depuis `p`/`ctrl`), `hidden_in` ignoré (mais le runner doit quand même lui binder un buffer `{b,s,d}` — un dummy/zéros). Le PLE per-layer est **recalculé** dans chaque stage (pur fonction de `embeds`+`embptl_slice` → bit-exact, évite de threader le tenseur `ple` 35×256).
- couches `[start,end)` via `runLayerGen` avec l'**index ABSOLU** de couche : `inline for (start..end) |i|` (comme `readerRange`). `i` pilote `isFull(i)`/`slidingSlot(i)`/`fullSlot(i)`/`ple.choose1d(.layer,i)` — un offset 0-based corromprait silencieusement slots et masques. Threader `self.brick` (neutre pour `struct{}` mais requis pour type-check).
- `last` : `final norm + lm_head + softcap → logits`.
- **Cache threadé** : `cache_in` → scatter (producers, chacun dans son slot, lu dans la même couche) → `cache_out`. Les reader-stages **ne modifient pas** le cache ; ils lisent les slots writers 13/14 (sliding slot 11 / full slot 2). Le KV writers **survit** dans le cache threadé de stage en stage (revue : composition correcte même si une frontière `CHUNK` coupe les producers 0-14). → **on threade le cache complet et on carry la sortie du DERNIER stage** vers le step suivant (équivalent au « dernier producer », plus simple).

`forward` mono **reste** (E1/E2).

### 4.2 Découpage + mécanisme de compile
`comptime CHUNK` = couches/stage. **⚠️ `platform.compile` exige une méthode NOMMÉE** (`DeclEnum`), donc on ne peut pas lui passer `forwardStageGen` à args comptime liés. Deux voies (à trancher empiriquement, **gate 0 ci-dessous**) :
- **(b) `compileFn`/`compileModel`** (`func: anytype`, `platform.zig:429/441`) piloté par une fn-factory comptime dans un `inline for` sur les tranches → **CHUNK reste un vrai knob comptime** (découpage adaptatif voulu par Régis). NON utilisé ailleurs dans ce repo → à dé-risquer.
- **(a) wrappers nommés** par stage (`forwardS0`, `forwardS1`… comme `forwardP`/`forwardR1..R4` du prefill) → **prouvé**, mais CHUNK figé par le jeu de wrappers (changer CHUNK = éditer les wrappers).

Préférence : tenter **(b)** d'abord (knob tunable) ; repli **(a)** si `compileFn` ne se comporte pas. Granularité de départ : 5 couches/stage (cf prefill, MLP double-wide 12288). NB : le cache est alloué à `.k=L_MAX` **dès le départ** (gen_long.zig) — seul son *contenu* se remplit quand `pos` avance ; **pas de redimensionnement dynamique**.

### 4.3 Runner `gemma4_gen_long`
- Compile les `N` stages **une fois** (hors boucle steps), garde les exe en vie.
- Pour chaque step : exécute les stages en séquence. Entre stages, réinjecte `hidden_out`→`hidden_in` et le cache (`sl_k/sl_v/fl_k/fl_v`) **device→device** (override des buffers, pas de copie host). **Sync forcée** après chaque stage (matérialiser `hidden_out`) pour libérer le working set d'exécution.
- Le dernier stage produit `logits` → argmax → compare `expected[step]`.
- Entre steps : le cache (sortie du dernier stage) est threadé vers le step suivant.

## 5. Mémoire — le vrai risque (recadré en revue)

Deux pics distincts, et le **dominant n'est pas celui d'un step** :
- **Pic d'exécution par stage** : working set de `CHUNK` couches (poids f32 convertis + activations). Borné si XLA libère après chaque call+sync.
- **Pic résident / compilation (DOMINANT)** : le decode **réutilise** les `N` exe → tous sont **compilés up-front et restent résidents simultanément**. Le prefill chain a justement mesuré une accumulation résidente 14,6→17,6→22,6 Go (`gemma4_prefill.zig:584`) qu'il combat par `exe.deinit()` entre stages (one-shot) — *exactement la configuration que le decode ne peut pas éviter*. Le prefill n'a donc **jamais démontré que sync SANS deinit borne la mémoire** : c'est l'inconnue centrale.

**Conséquences :**
1. La **première mesure** doit être le **pic post-compile** (les `N` exe résidents, avant le 1er step), pas un step isolé.
2. `CHUNK` est **non-monotone** : plus petit ⇒ pic/stage moindre MAIS **plus d'exe résidents** ⇒ peut *empirer* le pic dominant. Le bon `CHUNK` est un optimum, pas « le plus petit possible ».
3. C'est le **go/no-go de l'architecture** : si les `N` exe résidents dépassent 23 Go, ce n'est pas un `CHUNK` à régler mais la réutilisation des exe qui est invalidée → repli : `deinit`+recompile par stage à **chaque** step (mémoire bornée garantie comme le prefill, mais recompile/step = lent — à arbitrer vs le swap actuel) ; ou une stratégie hybride (sous-ensemble d'exe vivants).

## 6. Validation — gate 0 d'abord (de-risk)

- **Gate 0 — DE-RISK MÉMOIRE (première étape, avant de tout construire)** : mécanisme de compile `compileFn` (§4.2(b)) + compiler les `N` stages et **mesurer le pic post-compile** (`free -h` : RAM résidente tous exe compilés) + exécuter **1 step**. Critère : pic <23 Go, swap ≈ 0. **Si KO → l'architecture de réutilisation est invalidée** : basculer sur le repli (deinit+recompile/step, ou hybride) AVANT d'écrire le reste. Cette mesure tranche le go/no-go.
- **Non-régression** : E1/E2 PASS (forward mono inchangé) + preuve HLO `diff -rq` toujours valide.
- **Équivalence (oracle)** : re-run **L1a chunké → 1020/1020** tokens (mêmes que la version mono). Le chunking ne change pas le calcul, seulement l'exécution.
- **Perf (critère de succès)** : pic <23 Go, **swap ≈ 0**, **multi-thread Eigen rebranché** → L1a en minutes au lieu d'heures.

## 7. Hors-scope
- Pas de chunking pour E1/E2 (mono suffit, `.k=8`).
- Tuning fin `CHUNK`/`L_MAX` + remontée à 2048 : après validation à 1024.
- L1b/L2 : bénéficient du moteur chunké mais sont des gates ultérieurs (plan principal).
