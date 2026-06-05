# Socle ZML modulaire — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extraire le moteur decode en `engine.zig` paramétré comptime par une brique (`EngineModel(comptime Brick)`), avec un point d'extension `post_v_norm`, de sorte qu'une hypothèse se branche sans copier le moteur.

**Architecture:** `EngineModel(Brick)` = `decode4.zig` factorisé + champ `brick: Brick` + au point post-v_norm `if (@hasDecl(Brick,"post_v_norm")) v = self.brick.post_v_norm(v, ctx)`. Brique vide `struct{}` → identité → reproduit decode4 bit-exact (gate E1). `TurboQuantVBrick` → reproduit gen_vq (gate E2). Faisabilité comptime déjà confirmée dans le source ZML (review : `platform.compile`/`meta.visit`/`Bufferized` gèrent le model générique et la sous-struct `brick`).

**Tech Stack:** Zig 0.16-dev + ZML (Bazel) sur RTX 3090. Spec : `docs/ZML_MODULAR_ENGINE_DESIGN.md`.

---

## Environnement (rappel)
- Compute `ssh ia@192.168.1.163`. Workspace ZML `/data/rqz_workspace/zml/examples/rqz/`. Build : `ssh ia@192.168.1.163 'cd /data/rqz_workspace/zml && ./bazel.sh build //examples/rqz:<cible> 2>&1 | tail -3'`. Run : idem + `./bazel-bin/examples/rqz/<cible> <fixture>`. Déploiement `scp <f> ia@192.168.1.163:/data/rqz_workspace/zml/examples/rqz/`. **Écrire que sur /data.**
- **Cibles BUILD** : ajout idempotent côté 3090 (`grep -q <cible> $B || cat bloc >> $B`), garder le BUILD local M1 (`zml_runner/BUILD.bazel`) synchronisé. **⚠️ `srcs` OBLIGATOIRE** : un runner qui `@import("engine.zig")` exige `srcs = ["engine.zig", ...]` dans sa règle `zig_binary` (sinon le sandbox Bazel n'expose pas le fichier → build "file not found"). Voir les blocs exacts aux Steps build.
- **Fixtures (existent déjà, vérifiées)** : E1 → `/data/gemma4-zml-probe/fixtures/p5_7_8_gen.safetensors` (caches, embeds, pos, mask, `expected` = tokens réf I32[4]) ; E2 → `/data/gemma4-zml-probe/decode_vq_gen.safetensors` (idem + `codebook_256/512`, `hadamard_256/512`). **Aucun oracle Python à écrire.** Les runners prennent **`<model.safetensors> <fixture>`** en argv — **reprendre la commande de run exacte de `gemma4_decode4`/`gemma4_gen_vq`** (mêmes args checkpoint + fixture) ; identifier le chemin du checkpoint via la commande de run existante de decode4 (argv[1]).
- **Pièges ZML capitalisés** : `gather` exige `squeeze` de l'axe argMax ; codebook tag `.c` ; format log `{e:.1}` ; reshape perd les tags → `.withTags` ; pas de broadcast implicite → `.broad` ; nom runner ≤ 20 c.
- **Branche** : `turboquant-zml-vonly`. **NE PAS modifier** `gemma4_decode3.zig`, `gemma4_decode4.zig`, `gemma4_gen_vq.zig`, `gemma4_decode_vq.zig` (gates immuables = oracles).

## File Structure

| Fichier | Responsabilité | Action |
|---|---|---|
| `zml_runner/engine.zig` | `EngineModel(comptime Brick)` + `LayerCtx` + `quantizeV` (déplacée/partagée) — extrait de decode4, point `post_v_norm` | Create |
| `zml_runner/brick_turboquant.zig` | `TurboQuantVBrick` (`init(View)` + `post_v_norm` via `quantizeV_4d`) | Create |
| `zml_runner/gemma4_engine_e1.zig` | Runner gate E1 : `EngineModel(struct{})`, compare à l'oracle de la fixture decode4 (bit-exact) | Create |
| `zml_runner/gemma4_engine_e2.zig` | Runner gate E2 : `EngineModel(TurboQuantVBrick)`, compare à la fixture/tokens de gen_vq | Create |
| `zml_runner/BUILD.bazel` | cibles `gemma4_engine_e1`, `gemma4_engine_e2` | Modify |

---

## Task 1 (Gate E1) : Extraire `engine.zig` + non-régression `struct{}` == decode4

**Files:** Create `zml_runner/engine.zig`, `zml_runner/gemma4_engine_e1.zig` ; Modify `BUILD.bazel`.
**Lire d'abord** : `zml_runner/gemma4_decode4.zig` en entier (le moteur à factoriser : struct `Engine`/model, `forward(self, Packed, Cache, Ctrl) -> 5-tuple`, l'`inline for` des couches, le point v_norm). `zml_runner/gemma4_gen_vq.zig:262-273` (montre le point d'insertion V exact, à transformer en hook).

- [x] **Step 1 — Écrire `engine.zig`.** Copier la logique de `gemma4_decode4.zig` dans une fonction générique :
```zig
pub const LayerCtx = struct { layer_idx: usize, is_full: bool, pos: zml.Tensor }; // idx/is_full comptime
pub fn EngineModel(comptime Brick: type) type {
    return struct {
        // ...TOUS les champs de Engine (decode4) : layers: []LayerW, embed, norms, etc...
        brick: Brick,
        const Self = @This();
        pub fn forward(self: Self, p: Packed, cache_in: Cache, ctrl: Ctrl)
            struct { zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor } {
            // ...corps IDENTIQUE à decode4.forward, SAUF au point post-v_norm dans l'inline for couche i :
            //   v = zml.nn.rmsNorm(v, .hd, RMS_EPS);   // v_norm existant
            //   if (@hasDecl(Brick, "post_v_norm")) {
            //       const ctx = LayerCtx{ .layer_idx = i, .is_full = isFull(i), .pos = pos_tensor };
            //       v = self.brick.post_v_norm(v, ctx);
            //   }
            //   ...transpose/scatter inchangés...
        }
    };
}
```
Déplacer aussi `quantizeV` (de `gemma4_vquant.zig`) dans `engine.zig` (`pub fn`) pour qu'elle soit partagée par les briques. Garder `decode4.zig` intact (sa copie de `quantizeV` reste — il est gelé, dé-duplication seulement côté engine).

**⚠️ Point d'insertion réel (review)** : l'insert V n'est pas dans la boucle `inline for` elle-même mais dans **`runLayerGen`** (la fonction par-couche que la boucle appelle). Donc **threader `self.brick`** dans `runLayerGen` (comme `gemma4_gen_vq.zig:367` thread `p.codebook_256, ...`), et y faire le `if (@hasDecl(@TypeOf(brick), "post_v_norm"))` juste après `v = rmsNorm(v, .hd)` et avant le transpose. `is_full` et `layer_idx` y arrivent en comptime (la boucle est `inline for`). Pour `struct{}`, `@hasDecl`=false → branche comptime-morte → `runLayerGen` identique à decode4.

- [x] **Step 2 — Écrire `gemma4_engine_e1.zig`.** Copier le `main` de `gemma4_decode4.zig` (load fixture, compile, run, compare à l'oracle), mais instancier `EngineModel(struct{})` au lieu de la struct decode4. **Réutiliser la MÊME fixture que decode4** (déjà sur la 3090 ; identifier son chemin via `ls /data/rqz_workspace/zml/examples/rqz/fixtures/ | grep -i gen` ou la fixture de gen_oracle). Comparer aux mêmes références (tokens + last_hidden), mêmes seuils que decode4.

- [x] **Step 3 — Build + run.** Ajouter la cible (bloc EXACT, `srcs` obligatoire) :
```
zig_binary(
    name = "gemma4_engine_e1",
    main = "gemma4_engine_e1.zig",
    srcs = ["engine.zig"],
    visibility = ["//visibility:public"],
    deps = ["//bazel", "//zml"],
)
```
(idempotent 3090 + local). Build, run avec **la commande de run de decode4** (`<model.safetensors> /data/gemma4-zml-probe/fixtures/p5_7_8_gen.safetensors`).
  Expected : **mêmes 4 tokens argmax** que decode4 vs `expected` (decode4 compare l'argmax, PAS `last_hidden` — la fixture n'a pas de réf last_hidden). La branche brick étant comptime-morte, le graphe est identique → l'égalité des tokens doit tenir. Si divergence → erreur d'extraction (transcription du corps decode4) ; corriger.

- [x] **Step 4 — Commit + tag.** Fait : commit `e0d53ba` + tag `engine-e1-noregression-pass` (message « 4 tokens argmax », pas « bit-exact » — la fixture n'a pas de réf last_hidden).

> **E2 — point d'attention identifié à l'exécution d'E1 (multi-store)** : decode4 charge les **poids** depuis le checkpoint (`store_ck`) et les inputs/caches depuis la **fixture** (`store_fx`) — deux stores distincts. Le §3.4 du design suppose implicitement poids+constantes brick dans le même store. Les constantes `TurboQuantVBrick` (`codebook_*`/`hadamard_*`) vivent dans la fixture (`store_fx`), pas le checkpoint. Or `zml.io.load(Self, …, store)` résout les `Tensor` d'une struct contre **un seul** store. Donc `EngineModel(TurboQuantVBrick).load(store_ck)` ne résoudra pas les `id` de brick (bindés à `store_fx`) → crash `getReaderById null`. **À trancher en Task 2 Step 1** : (a) `init(allocator, base_ck, brick_view_fx)` + chargement séparé poids/brique puis assemblage manuel du `Bufferized(EngineModel)` (pattern decode4 qui assemble `cache_buf` à la main) ; ou (b) store combiné (agréger les 2 registries). E1 n'est pas concerné (brique vide = 0 tenseur).

---

## Task 2 (Gate E2) : `TurboQuantVBrick` + `EngineModel(brick)` == gen_vq

**Files:** Create `zml_runner/brick_turboquant.zig`, `zml_runner/gemma4_engine_e2.zig` ; Modify `BUILD.bazel`.
**Lire d'abord** : `zml_runner/gemma4_gen_vq.zig` (l'oracle E2 : insertion V-quant + sa fixture + tokens de référence ; lignes ~182-198 le `Packed.init`/`createTensor` des constantes, ~262-273 l'insert V).

- [x] **Step 1 — Écrire `brick_turboquant.zig`.** Fait. **Amendement vs esquisse ci-dessous** : `post_v_norm(self, v, comptime is_full: bool, ctx: LayerCtx)` — `is_full` est un **paramètre comptime** (pas `ctx.is_full`), car codebook/Hadamard 256 vs 512 ont des shapes différentes → la sélection doit être comptime (un select runtime exige des shapes égales). `LayerCtx` réduit à `{layer_idx}`. Définir la brique avec le **contrat de chargement** (sinon le load crashe — cf spec §3.4) :
```zig
const TurboQuantVBrick = struct {
    codebook_256: zml.Tensor, hadamard_256: zml.Tensor,
    codebook_512: zml.Tensor, hadamard_512: zml.Tensor,
    pub fn init(view: zml.io.TensorStore.View) TurboQuantVBrick {
        return .{
            .codebook_256 = view.createTensor("codebook_256", .{.c}, null),
            .hadamard_256 = view.createTensor("hadamard_256", .{ .e, .hd }, null),
            .codebook_512 = view.createTensor("codebook_512", .{.c}, null),
            .hadamard_512 = view.createTensor("hadamard_512", .{ .e, .hd }, null),
        };
    }
    pub fn post_v_norm(self: @This(), v: zml.Tensor, ctx: LayerCtx) zml.Tensor {
        const cb = if (ctx.is_full) self.codebook_512 else self.codebook_256;
        const Pi = if (ctx.is_full) self.hadamard_512 else self.hadamard_256;
        // quantizeV_4d : reprendre VERBATIM le reshape du POC (gen_vq.zig ~268-270),
        // car reshape perd les tags -> withTags explicite au retour (pas v.shape()) :
        const hd = v.dim(.hd);
        const v2 = v.reshape(.{ 1, hd }).withTags(.{ .k, .hd });   // B=S=KVH=1 en decode
        const o = engine.quantizeV(v2, cb, Pi);                    // [.k,.hd]
        return o.reshape(.{ 1, 1, 1, hd }).withTags(.{ .b, .s, .nh, .hd });
    }
};
```

- [x] **Step 2 — Écrire `gemma4_engine_e2.zig`.** Fait. **Multi-store résolu** (cf DESIGN §3.4 correctif) : poids chargés via `EngineModel(struct{}).load(store_ck)`, brique via `zml.io.load(TurboQuantVBrick, …, store_fx)`, puis `Bufferized(EngineModel(TurboQuantVBrick))` **assemblé à la main** (mapping positionnel). Model symbolique via `initBrick(base_ck, fixture_view)`. Réutilise la fixture de gen_vq + compare aux tokens `expected`.

- [x] **Step 3 — Build + run.** Fait : build OK, run `4/4 tokens [107,1,106,1] == HF-V-quant`. Ajouter la cible (bloc EXACT, `srcs` = engine + brick) :
```
zig_binary(
    name = "gemma4_engine_e2",
    main = "gemma4_engine_e2.zig",
    srcs = ["engine.zig", "brick_turboquant.zig"],
    visibility = ["//visibility:public"],
    deps = ["//bazel", "//zml"],
)
```
Build, run avec la commande de gen_vq (`<model.safetensors> /data/gemma4-zml-probe/decode_vq_gen.safetensors`).
  Expected : **mêmes tokens générés que `gemma4_gen_vq`** (le POC). Le socle+brique reproduit la copie, sans copie.

- [x] **Step 4 — Commit + tag.** Fait : commit `d6146ba` + tag `engine-e2-brick-pass` (inclut les modifs socle `engine.zig` : hook is_full comptime + `initBrick`). **Gates E1+E2 verts — socle modulaire fonctionnel de bout en bout.**

---

## Notes de séquencement
- E1 → E2 strict (E1 ferme la factorisation avant de brancher la brique).
- **Risque principal** (cf review) : l'extraction fidèle de decode4 dans `engine.zig` — E1 bit-exact l'attrape. Si E1 ne passe pas bit-exact, c'est une divergence de transcription, pas un problème de design.
- Mini dé-risquage comptime optionnel avant Task 1 : un `EngineModel(struct{})` trivial qui compile (`bazel build`) confirme que le pattern générique passe `platform.compile` (déjà confirmé par lecture du source ZML, mais un build vaut une preuve).
- **Livrable final** : `docs/` guide « écrire une brique » (peut être une section ajoutée à `ZML_MODULAR_ENGINE_DESIGN.md` §6, déjà présente) + mémoire `zml_modular_engine.md` mise à jour. Ne pas pousser le repo (décision Régis).
