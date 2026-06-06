# Génération longue — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Faire du decode ZML un moteur de génération longue (fenêtre glissante 512, cache borné, jusqu'à `L_max=2048` tokens), d'abord validé contre oracle HF (L1), puis en inférence autonome host-orchestrée (L2).

**Architecture:** On fait évoluer `engine.zig` (le socle modulaire publié) sous gardes **comptime** dont les valeurs par défaut n'émettent aucune op nouvelle → E1/E2 restent byte-identiques (preuve HLO `diff -rq`). Gates séquentiels : Task 0 (paramétrisation comptime neutre) → L0 (oracle) → L1a (cache linéaire borné + masque bande) → L1b (vrai ring 512 + masque circulaire) → L2 (autonome host). L3 (in-graph) est hors de ce plan.

**Tech Stack:** Zig + ZML (XLA/PJRT-CPU), Python + HuggingFace transformers (oracles), Bazel, exécution sur RTX 3090 (Proxmox).

**Spec de référence :** `docs/GENERATION_LONGUE_DESIGN.md`.

---

## Conventions communes (lire avant toute tâche)

**Contexte 3090 (toute compilation/exécution s'y fait — JAMAIS en local) :**
- Hôte : `ssh ia@192.168.1.163`. Workspace ZML : `/data/rqz_workspace/zml`. Dir runner : `/data/rqz_workspace/zml/examples/rqz/`.
- Checkpoint : `/data/gemma4-zml-probe/weights/model.safetensors`.
- Déploiement des sources locales → 3090 : depuis `~/dev/gemma4-zml-probe/zml_runner/` :
  ```bash
  ZML_REMOTE=ia@192.168.1.163 ZML_DST=/data/rqz_workspace/zml/examples/rqz ./deploy_to_3090.sh
  ```
- Build + run d'une cible : sur la 3090, **via le wrapper `./bazel.sh`** (bazelisk ; `bazel` n'est PAS dans le PATH) :
  ```bash
  cd /data/rqz_workspace/zml && ./bazel.sh run //examples/rqz:<TARGET> -- <args...>
  ```
  Le PATH n'est pas chargé en SSH non-interactif → préfixer par `bash -lc "..."`.
- Serveur Bazel actif = `pgrep -x java`. Fixtures volumineuses (>100 MB) restent sur la 3090 (gitignorées).
- **⚠️ PRÉREQUIS MÉMOIRE (OOM)** : le compile XLA-CPU d'un graphe 35 couches monte à **~22,7 Go RSS**, au ras des 23 Go de la VM → OOM-killer (`tf_XLAEigen invoked oom-killer`, mort silencieuse exit 255 pile à `Compiling gen step`). Débloqué le 5 juin par un **swapfile temporaire 16 Go** (`/swapfile_xla`, swap total → 22 Go). Vérifier `swapon --show` avant tout build, sinon TOUS les runners OOM. Solution durable à arbitrer (RAM VM ou swap permanent).
- **Régis ne colle aucune commande** : l'agent exécute via SSH lui-même (compute local non payant).

**Oracle = source de vérité.** Avant d'écrire un runner/oracle, lire la vérité terrain dans `modeling_gemma4.py` (présent sur la 3090) plutôt que de présumer.

**Validation d'un gate (analogue TDD) :**
1. Écrire l'oracle Python (le « test ») → produit une fixture + une séquence `expected`.
2. Écrire/modifier le runner Zig (l'« implémentation »).
3. Build + run sur 3090 → comparer argmax (par step) ou scan `max_abs` contre l'oracle.
4. **Non-vacuité obligatoire** : corrompre l'oracle (ex. zéro/perturbation) → le gate doit **FAIL** (réfute l'aliasing). Documenter ce contre-test.
5. Commit.

**Non-régression à CHAQUE commit touchant `engine.zig` :**
```bash
# sur 3090
cd /data/rqz_workspace/zml
bazel run //examples/rqz:gemma4_engine_e1 -- /data/gemma4-zml-probe/weights/model.safetensors <p5_7_8_gen.safetensors>   # doit PASS (4 tokens)
bazel run //examples/rqz:gemma4_engine_e2 -- /data/gemma4-zml-probe/weights/model.safetensors <decode_vq_gen.safetensors> # doit PASS (4 tokens)
```
Plus, pour Task 0 (et tout commit prétendant « HLO inchangé ») : preuve HLO `diff -rq` (cf. Task 0, Step 7).

---

## File Structure

| Fichier | Rôle | Action |
|---------|------|--------|
| `zml_runner/engine.zig` | Socle decode. Gagne : `Cache(KMAX_SLIDING, KMAX_FULL)` et `Packed(mode)` paramétrés comptime, flag `ring`, sélection de masque par `comptime isFull(i)`. | Modifier |
| `zml_runner/gemma4_engine_e1.zig`, `gemma4_engine_e2.zig` | Runners de non-régression. **Doivent rester PASS.** Adaptés seulement si la signature des types change (instanciation en config défaut). | Modifier a minima |
| `scripts/46_gen_long_oracle.py` | L0 : oracle HF 2048 tokens + fixture `gen_long.safetensors` (2 masques, caches `L_max`). | Créer |
| `zml_runner/gemma4_gen_long.zig` | Runner L1a : replay long linéaire borné, compare argmax vs `expected`. | Créer |
| `zml_runner/gemma4_gen_long_ring.zig` | Runner L1b : replay long ring 512 (instanciation `ring=true`). | Créer |
| `zml_runner/gemma4_gen_auto.zig` | Runner L2 : decode autonome host-orchestré (argmax + gather + cos/sin/masque host). | Créer |
| `zml_runner/BUILD.bazel` | Cibles `gemma4_gen_long`, `gemma4_gen_auto` (avec `srcs=["engine.zig"]`). | Modifier |
| `docs/ENGINE_LOG.md` | Journal des gates (pattern projet). | Modifier (append) |

---

## Task 0 : Paramétrisation comptime neutre de `engine.zig`

**But :** introduire les paramètres comptime (`KMAX_SLIDING`, `KMAX_FULL`, `ring`, mode de masque) **sans changer le comportement par défaut**. À la fin, E1/E2 PASS et le HLO est byte-identique à avant Task 0. Aucune logique de génération longue encore : c'est la fondation rétro-compatible.

**Files:**
- Modify: `zml_runner/engine.zig`
- Modify (a minima) : `zml_runner/gemma4_engine_e1.zig`, `zml_runner/gemma4_engine_e2.zig`
- Verify: cibles `gemma4_engine_e1`, `gemma4_engine_e2`

- [ ] **Step 1 : Capturer le HLO de référence (avant toute modif)**

Sur 3090, dumper le HLO d'E1 *tel quel* (baseline) :
```bash
cd /data/rqz_workspace/zml
XLA_FLAGS="--xla_dump_to=/tmp/hlo_e1_before" bazel run //examples/rqz:gemma4_engine_e1 -- \
  /data/gemma4-zml-probe/weights/model.safetensors /data/.../p5_7_8_gen.safetensors
```
Garder `/tmp/hlo_e1_before` pour le diff final (Step 7).

- [ ] **Step 2 : Exposer `KMAX_SLIDING`/`KMAX_FULL` comme scalaires comptime (la dim cache est déjà inférée de la fixture)**

⚠️ Important : `Cache.init` (engine.zig:156-163) crée `cache_sl_k`/`fl_k` avec `createTensor(..., null)` → la dim `.k` est **inférée du safetensors chargé**, pas codée en Zig. Donc on n'a **pas** besoin de reparamétrer `Cache` pour changer la taille `.k` (elle suit la fixture : 8 aujourd'hui, `L_max`/512 demain). `Cache` reste tel quel.

Le besoin réel : avoir `KMAX_SLIDING` comme **scalaire comptime** côté Zig pour (a) le modulo du ring (Step 4) et (b) le check d'égalité de forme du dédoublement de masque. On les place donc dans la config `EngineModel` (Step 4), pas dans `Cache`.

> Vérifier que `scores`/`mask` broadcast (engine.zig:266-268) tient toujours en défaut (un seul masque, `.k` sliding==full venant de la même fixture KMAX=8).

- [ ] **Step 3 : Paramétrer `Packed` par un mode comptime (masque simple vs double)**

`fn Packed(comptime two_masks: bool) type` :
- `two_masks=false` (défaut) : champ unique `masks` (`{step,b,h,q,k}`), comme aujourd'hui.
- `two_masks=true` : champs `masks_sliding` (`.k=KMAX_SLIDING`) + `masks_full` (`.k=KMAX_FULL`), **pas** de champ `masks`.

Alias défaut `pub const PackedDefault = Packed(false);`. (Cf. spec §5.3 : `zml.io.load` réfléchit sur les champs → la paramétrisation doit être au niveau **type**, pas un `if` runtime.)

- [ ] **Step 4 : Étendre la signature `EngineModel` + ajouter `ring`/sélection de masque dans `runLayerGen` (gardés comptime, défaut neutre)**

`EngineModel` est aujourd'hui `EngineModel(comptime Brick: type)` (engine.zig:292). **Nouvelle signature** : ajouter un 2e paramètre comptime de config avec des champs **defaulted** :
```zig
pub const EngineCfg = struct {
    ring: bool = false,
    two_masks: bool = false,
    kmax_sliding: i64 = 8,
    kmax_full: i64 = 8,
};
pub fn EngineModel(comptime Brick: type, comptime cfg: EngineCfg) type { ... }
```
- E1 : `engine.EngineModel(struct{}, .{})` (tout par défaut → neutre).
- E2 : `engine.EngineModel(TurboQuantVBrick, .{})`.
- L1a : `EngineModel(struct{}, .{ .two_masks = true, .kmax_sliding = L_MAX, .kmax_full = L_MAX })`.
- L1b : `…{ .ring = true, .two_masks = true, .kmax_sliding = 512, .kmax_full = L_MAX }`.

`cfg` est lisible dans `forward`/`runLayerGen` (capturé par la closure de type). Dans `runLayerGen` :
```zig
// scatter sliding — modulo COMPTIME-élidé en défaut
const write_k = if (cfg.ring) pos_u.rem(Tensor.scalar(@intCast(cfg.kmax_sliding), .u32)) else pos_u; // ring=false -> aucune op rem
// ...
// sélection de masque par type de couche
const mask = if (cfg.two_masks) (if (isFull(i)) mask_full else mask_sliding) else mask_single;
```
⚠️ `cfg.ring` et `cfg.two_masks` sont **comptime** → en défaut, la branche morte n'émet rien (ni `rem`, ni second masque). C'est la condition de l'égalité HLO. `runLayerGen` reçoit `mask_single` (mode défaut) **ou** `mask_sliding`+`mask_full` (mode `two_masks`) ; threader selon `cfg.two_masks`.

- [ ] **Step 5 : Faire compiler E1/E2 en config défaut**

`gemma4_engine_e1.zig` : `const Model = engine.EngineModel(struct{}, .{});` (defaults → ring=false, KMAX 8/8, masque simple). Idem E2 : `EngineModel(TurboQuantVBrick, .{})`. `Packed` reste `Packed(false)` (champ `masks` unique). Aucune autre logique modifiée.

- [ ] **Step 6 : Re-run E1 + E2 → PASS**

```bash
bazel run //examples/rqz:gemma4_engine_e1 -- <ckpt> <p5_7_8_gen.safetensors>   # PASS 4 tokens
bazel run //examples/rqz:gemma4_engine_e2 -- <ckpt> <decode_vq_gen.safetensors> # PASS 4 tokens
```
Expected : `E1 PASS` / `E2 PASS`, séquences `[1018,6398,25967,53121]` resp. `[107,1,106,1]`.

- [ ] **Step 7 : Preuve HLO — `diff -rq` byte-identique**

```bash
XLA_FLAGS="--xla_dump_to=/tmp/hlo_e1_after" bazel run //examples/rqz:gemma4_engine_e1 -- <ckpt> <p5_7_8_gen.safetensors>
diff -rq /tmp/hlo_e1_before /tmp/hlo_e1_after
```
Expected : **aucune différence** (hors `debug_options` = chemin de dump, comme établi pour E1). Si une op `rem`/un second masque apparaît → la garde comptime fuit, corriger avant commit.

- [ ] **Step 8 : Commit**

```bash
git add zml_runner/engine.zig zml_runner/gemma4_engine_e1.zig zml_runner/gemma4_engine_e2.zig
git commit -m "refactor(engine): paramétrisation comptime neutre (Cache/Packed/ring/masque) — E1/E2 PASS, HLO byte-identique"
```

---

## Task L0 : Oracle de génération longue (`46_gen_long_oracle.py`)

**But :** produire `gen_long.safetensors` + séquence `expected` de N≈2048 tokens greedy HF, avec sliding window 512 réellement actif, deux masques, caches `L_max`. Réécriture ciblée de `scripts/45_gen_vq_oracle.py` (PAS de hooks V-quant).

**Files:**
- Create: `scripts/46_gen_long_oracle.py`
- Reference: `scripts/45_gen_vq_oracle.py` (structure prefill/cache/embptls), `modeling_gemma4.py` (vérité sliding window)

- [ ] **Step 1 : Paramètres + génération HF**

Constantes : `L_MAX=2048`, `SLIDING_WINDOW=512`, `KMAX_SLIDING=L_MAX` (L1a ; passera à 512 en L1b côté fixture), `KMAX_FULL=L_MAX`. Générer la séquence greedy HF avec `min_new_tokens`/EOS ignoré pour garantir N tokens (cf. spec §7). Sauver `expected` (i32, `{step}`).

- [ ] **Step 2 : Embeds + embptls par step (gather HF)**

Pour chaque token généré : `embeds[k] = embed_tokens(tid)` (bf16) ; `embptls[k] = embed_tokens_per_layer.weight[tid].view(1,1,8960)` (comme script 45 ligne 211). `cos_full/sin_full` pour la position (RoPE full, theta=1e6, partial 0.25).

- [ ] **Step 3 : Deux masques (bande sliding + causal full)**

- `masks_sliding[k]` : `.k=KMAX_SLIDING`, vaut 0 pour `pos-511 <= j <= pos`, sinon `finfo.min`.
- `masks_full[k]` : `.k=KMAX_FULL`, causal plein (0 pour `j <= pos`, sinon `finfo.min`).
- (L1b ajoutera la variante circulaire de `masks_sliding` — voir Task L1b Step 1.)

- [ ] **Step 4 : Caches prefill du prompt dimensionnés `L_max`**

Comme script 45 (extraction prefill couches 0-14, padding) mais `.k=L_MAX` au lieu de KMAX=8. Slots sliding/full séparés. Pas de quantification V.

- [ ] **Step 5 : Écrire la fixture + run de sanity**

Sauver `gen_long.safetensors` (tous tenseurs + `expected`). Logger `prompt`, `N_DECODE`, `kmax`, premiers/derniers tokens. Vérifier la cohérence de shape.

- [ ] **Step 6 : Commit**

```bash
git add scripts/46_gen_long_oracle.py
git commit -m "feat(gen-long): L0 oracle HF 2048 tokens + fixture (2 masques, caches L_max, sans V-quant)"
```

---

## Task L1a : Cache linéaire borné + masque bande (replay)

**But :** premier gate de génération longue. Cache sliding **linéaire** `.k=L_max` (pas encore de ring), masque bande. Le runner rejoue les N tokens de la fixture et compare argmax vs `expected`.

**Files:**
- Create: `zml_runner/gemma4_gen_long.zig`
- Modify: `zml_runner/BUILD.bazel` (cible `gemma4_gen_long`, `srcs=["engine.zig"]`)
- Uses: `engine.EngineModel` avec `ring=false`, `two_masks=true`, `Cache(L_MAX, L_MAX)`, `Packed(true)`

- [ ] **Step 1 : Cible Bazel**

Ajouter dans `BUILD.bazel` :
```python
zig_binary(
    name = "gemma4_gen_long",
    main = "gemma4_gen_long.zig",
    srcs = ["engine.zig"],
    visibility = ["//visibility:public"],
    deps = ["//bazel", "//zml"],
)
```

- [ ] **Step 2 : Runner replay long (calqué sur `gemma4_engine_e1.zig`)**

Instancier `engine.EngineModel(struct{}, .{ .two_masks = true, .kmax_sliding = L_MAX, .kmax_full = L_MAX })` (ring=false → linéaire borné). `Packed(true)` (deux masques). `NUM_STEPS` lu depuis la fixture (taille de `expected`). Boucle : pour chaque step, `forward` → argmax → comparer `expected[step]`. Threader le cache grandi (comme E1 lignes 115-120).

> **Garde (issue de la revue Task 0)** : les scalaires comptime `kmax_sliding`/`kmax_full` sont **découplés** des dims `.k` du cache (inférées de la fixture). Ajouter au runner un **assert host** que `kmax_sliding`/`kmax_full` == les dims `.k` des caches chargés (`cache_sl_k`/`cache_fl_k`) **et** des masques (`masks_sliding`/`masks_full`). Un mismatch = bug silencieux d'aliasing ring/masque (la classe d'erreur que le contre-test L1b traque).

- [ ] **Step 3 : Build + run → PASS sur N tokens**

```bash
ZML_REMOTE=ia@192.168.1.163 ZML_DST=/data/rqz_workspace/zml/examples/rqz ./deploy_to_3090.sh
# sur 3090 :
bazel run //examples/rqz:gemma4_gen_long -- /data/gemma4-zml-probe/weights/model.safetensors /data/.../gen_long.safetensors
```
Expected : `L1a PASS` — argmax ZML == HF sur les N (~2048) steps.

- [ ] **Step 4 : Contre-test de non-vacuité**

Rejouer en corrompant le masque bande de l'oracle (ex. tout `finfo.min`, ou décaler la fenêtre) → le gate doit **FAIL** (prouve que l'attention dépend réellement du masque bande, pas d'un alias). Documenter dans `ENGINE_LOG.md`.

- [ ] **Step 5 : Non-régression E1/E2** (cf. Conventions) — doivent rester PASS.

- [ ] **Step 6 : Commit**

```bash
git add zml_runner/gemma4_gen_long.zig zml_runner/BUILD.bazel docs/ENGINE_LOG.md
git commit -m "feat(gen-long): L1a cache linéaire borné + masque bande — PASS argmax==HF sur 2048 tokens"
```

---

## Task L1b : Vrai ring-buffer 512 + masque circulaire (replay)

**But :** convertir le cache sliding en ring 512 (`ring=true`, `KMAX_SLIDING=512`) avec masque circulaire host. Isole la difficulté du wrap. Full reste linéaire `L_max`.

**Files:**
- Modify: `scripts/46_gen_long_oracle.py` (variante masque circulaire + caches sliding `.k=512`)
- Modify: `zml_runner/gemma4_gen_long.zig` (instanciation `ring=true`, `KMAX_SLIDING=512`)

- [ ] **Step 1 : Oracle — masque sliding circulaire + cache sliding 512**

Ajouter au script 46 un mode `--ring` : `masks_sliding[k]` a `.k=512`, valide aux indices `(pos-511..pos) % 512` (wrap), sinon `finfo.min`. Caches **sliding** prefill ré-indexés modulo 512 (les caches full restent `.k=L_max`). Regénérer `gen_long_ring.safetensors`.

> Vérité à respecter : le scatter ZML écrira K/V du token courant à `pos % 512`. Le masque doit pointer les mêmes slots physiques. Construire le masque depuis la même formule modulo que le scatter.

- [ ] **Step 2 : Runner — nouvelle cible `gemma4_gen_long_ring`**

Créer un runner court `gemma4_gen_long_ring.zig` (ou factoriser le corps de `gemma4_gen_long.zig` et n'y changer que l'instanciation) qui instancie `engine.EngineModel(struct{}, .{ .ring = true, .two_masks = true, .kmax_sliding = 512, .kmax_full = L_MAX })`. Ajouter la cible Bazel `gemma4_gen_long_ring` (`srcs=["engine.zig"]`). On garde **deux cibles distinctes** (linéaire vs ring) pour rejouer L1a en non-régression à tout moment.

- [ ] **Step 3 : Build + run → PASS**

```bash
bazel run //examples/rqz:gemma4_gen_long_ring -- <ckpt> /data/.../gen_long_ring.safetensors
```
Expected : `L1b PASS` — argmax == HF sur N tokens, y compris **après** le premier wrap (pos ≥ 512).

- [ ] **Step 4 : Contre-test wrap**

Vérifier que le PASS dépend du wrap : corrompre l'indexation circulaire de l'oracle (masque non-wrappé) → doit FAIL aux positions ≥ 512. Documenter.

- [ ] **Step 5 : Non-régression E1/E2 + HLO** — E1/E2 PASS (ils tournent en `ring=false`, masque simple → HLO inchangé). Le `rem` n'apparaît **que** dans la cible ring.

- [ ] **Step 6 : Commit**

```bash
git add scripts/46_gen_long_oracle.py zml_runner/gemma4_gen_long.zig zml_runner/BUILD.bazel docs/ENGINE_LOG.md
git commit -m "feat(gen-long): L1b vrai ring-buffer 512 + masque circulaire — PASS argmax==HF après wrap"
```

---

## Task L2 : Inférence autonome host-orchestrée

**But :** plus de fixture par-token. Le host fait argmax, gather embeds/embptls, calcule cos/sin + masque pour pos+1, réinjecte. Prefill du prompt **conservé via `cache0`** (fixture). Greedy → séquence == HF.

**Files:**
- Create: `zml_runner/gemma4_gen_auto.zig`
- Modify: `zml_runner/BUILD.bazel` (cible `gemma4_gen_auto`)
- Reuse: `gen_long.safetensors` pour `cache0` (prefill) + `expected` (oracle) ; tables `embed_tokens` / `embed_tokens_per_layer` lues du checkpoint.

- [ ] **Step 1 : Cible Bazel `gemma4_gen_auto`** (même forme que L1, `srcs=["engine.zig"]`).

- [ ] **Step 2 : Gather host des embeddings**

Le runner charge `embed_tokens` (`{voc,d}`) et `embed_tokens_per_layer` (`{voc,8960}`) du checkpoint. À chaque step, à partir du `tok` argmax : extraire `embeds = embed_tokens[tok]` (host → buffer ZML, `*√1536` appliqué côté graphe comme aujourd'hui) et `embptls = embed_tokens_per_layer[tok]`.

> Décision : gather **host** (lire la ligne du tenseur) le plus simple ; alternative = mini-forward d'embedding ZML. Préférer host (cohérent D2 « host d'abord »).

- [ ] **Step 3 : cos/sin + masque host pour pos+1**

Calculer côté host (formules RoPE full theta=1e6 partial 0.25, identiques à l'oracle 46) les `cos_full/sin_full` de la position courante, et les deux masques (bande/circulaire selon ring, causal full) pour `pos`. Les passer en entrée d'un `forward` 1-step (graphe inchangé vs L1).

- [ ] **Step 4 : Boucle decode autonome**

Prefill via `cache0` (fixture). Puis : `forward` → argmax `tok` → (Step 2/3) → `forward` suivant. Comparer chaque `tok` à `expected[step]`.

- [ ] **Step 5 : Build + run → PASS séquence == HF**

```bash
bazel run //examples/rqz:gemma4_gen_auto -- <ckpt> /data/.../gen_long.safetensors
```
Expected : `L2 PASS` — séquence **générée** (embeds gather host, pas depuis la fixture) == HF greedy sur N tokens.

- [ ] **Step 6 : Contre-test**

Corrompre le gather (ex. embptls à zéro) → divergence rapide vs `expected` (prouve que la génération utilise réellement les embeddings gather, pas la fixture). Documenter.

- [ ] **Step 7 : Non-régression E1/E2** — PASS.

- [ ] **Step 8 : Commit**

```bash
git add zml_runner/gemma4_gen_auto.zig zml_runner/BUILD.bazel docs/ENGINE_LOG.md
git commit -m "feat(gen-long): L2 inférence autonome host-orchestrée — séquence générée == HF greedy"
```

---

## Hors-plan (différé)

- **L3** — internalisation in-graph (gather + RoPE + masque + argmax dans le forward). Gate optionnel, planifié séparément si besoin (D2).
- **Briques de compression sur génération longue** (TurboQuant & co. en régime long) — chantier suivant, une fois le socle long validé.

## Checklist de complétude (fin de plan)

- [ ] Task 0 : E1/E2 PASS + HLO byte-identique.
- [ ] L0 : fixture 2048 tokens + 2 masques + caches L_max, sans V-quant.
- [ ] L1a : PASS argmax==HF (linéaire borné) + contre-test.
- [ ] L1b : PASS argmax==HF après wrap (ring 512) + contre-test wrap.
- [ ] L2 : PASS séquence générée == HF (autonome host) + contre-test.
- [ ] E1/E2 verts à chaque commit ; `ENGINE_LOG.md` à jour ; mémoire `zml_modular_engine.md` mise à jour en clôture.
