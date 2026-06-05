# Socle ZML modulaire — Design : moteur decode + briques greffables (comptime policy)

**Date** : 2026-06-04
**Statut** : design validé (brainstorming), prêt pour plan d'implémentation
**Repo** : gemma4-zml-probe
**Mémoire liée** : `~/dev/Ma_MEMOIRE/memory/zml_modular_engine.md`, `project_gemma4_zml_probe.md`, `turboquant.md`

---

## 1. Objectif

Transformer `gemma4-zml-probe` en **banc d'essai ZML modulaire** : un moteur decode invariant
(« le socle ») sur lequel des **briques** expérimentales (transformations greffées à un point du
graphe) se branchent **sans copier le moteur**. But : tester des hypothèses (KV-quant, K-quant,
compression de contexte, sketches…) chacune comme une brique, pas comme un fork du moteur.

## 2. Motivation

Le POC TurboQuant (gates Q1-Q5) a prouvé le **pattern** — une transformation (`quantizeV`) greffée
au point `post_v_norm` — mais l'a implémenté en **copiant** le moteur : `decode3.zig → decode_vq.zig`,
`decode4.zig → gen_vq.zig`. Chaque hypothèse = une copie (3 copies de `quantizeV`, signalé en review).
**Ça ne scale pas.** La modularité = **injecter** la brique au lieu de copier le moteur.

## 3. Design (comptime policy, minimal-extensible)

### 3.1 Le socle — `engine.zig`

Model générique paramétré **comptime** par le type de brique, extrait de `decode4.zig` (boucle de
génération, cache threadé, 35 couches) :

```zig
fn EngineModel(comptime Brick: type) type {
    return struct {
        // ...poids/couches/cache repris de decode4 (inchangés : layers: []LayerW, etc.)...
        brick: Brick,                 // struct{} si aucune brique
        // signature RÉELLE héritée de decode4 (pas `forward(self) Tensor`) :
        pub fn forward(self: @This(), p: Packed, cache_in: Cache, ctrl: Ctrl)
            struct { Tensor, Tensor, Tensor, Tensor, Tensor } { /* boucle + cache */ }
    };
}
```

À chaque **point d'extension**, le moteur applique la brique si elle implémente ce point, sinon passe.
**Le point est appelé depuis l'`inline for` des couches** (decode4 boucle les 35 couches en `inline for`),
donc `ctx.is_full` est **comptime-connu** (il sélectionne le codebook) :

```zig
// dans la couche i (inline for), au point post-v_norm (v tagué [.b,.s,.nh,.hd]) :
if (@hasDecl(Brick, "post_v_norm"))
    v = self.brick.post_v_norm(v, ctx);   // ctx.is_full comptime
```

**Validé contre le source ZML** (review 4 juin) : `platform.compile` résout `forward` via `DeclEnum`/`@field`
sur le type instancié (la généricité est effacée) ; la reflection `meta.visit` **descend dans le champ
`brick`** et expose ses `Tensor` comme arguments MLIR ; `brick: struct{}` (sans `Tensor`) est **skippé**
(`Contains(struct{},Tensor)=false`) → aucun buffer fantôme ; `@hasDecl` comptime-false → branche **non
compilée** → `v` inchangé → E1 bit-exact.

**Un seul point d'extension au départ : `post_v_norm`** (le seul prouvé). Les autres
(`post_k_rope`, `scores`…) s'ajoutent trivialement — une ligne `if (@hasDecl(...))` + la méthode —
quand une brique réelle les demande (YAGNI).

### 3.2 Le contexte — `LayerCtx`

Passé à chaque point : `LayerCtx = struct { layer_idx, is_full (comptime), pos: Tensor (runtime) }`.
`layer_idx`/`is_full` sont **comptime** (issus de l'`inline for`, `is_full = isFull(i)` comme le POC),
`pos` est un `Tensor` runtime. Route la brique : `cb_512/Pi_512` si `is_full` sinon `cb_256/Pi_256`
(sélection **comptime**, cf `gen_vq.zig:266`).

### 3.3 Une brique = un type Zig

- **Identité** : `struct {}` (vide) → aucun `@hasDecl` ne matche → le moteur ne fait rien →
  `EngineModel(struct{})` reproduit `decode4` (**mêmes tokens générés** ; la branche brick est
  comptime-morte donc le graphe MLIR est identique). C'est l'**oracle de non-régression gratuit**.
- **TurboQuant V-quant** : une struct avec la méthode + ses constantes en champs `Tensor` :

```zig
const TurboQuantVBrick = struct {
    codebook_256: Tensor, hadamard_256: Tensor,
    codebook_512: Tensor, hadamard_512: Tensor,
    // contrat de chargement OBLIGATOIRE (cf §3.4) :
    pub fn init(v: View) TurboQuantVBrick { /* createTensor("codebook_256",...) etc. */ }
    pub fn post_v_norm(self: @This(), v: Tensor, ctx: LayerCtx) Tensor {
        const cb = if (ctx.is_full) self.codebook_512 else self.codebook_256;
        const Pi = if (ctx.is_full) self.hadamard_512 else self.hadamard_256;
        return quantizeV_4d(v, cb, Pi);
    }
};
```

**`quantizeV_4d`** (à spécifier) = wrapper d'axes autour de la fonction `quantizeV` prouvée en Q3 (inchangée).
Corps exact, repris du POC (`gen_vq.zig:268-270`) : `const v2 = v.reshape(.{B, hd}).withTags(.{.k,.hd}); const o = quantizeV(v2, cb, Pi); return o.reshape(v.shape()...)` (en decode, `.b=.s=.nh=1` → `B=1`). **Note structurelle** : dans le POC, codebook/Pi vivent dans `Packed` (le bundle d'**inputs** par-step) et l'insert est inline ; ici ils deviennent **champs du model** (`self.brick`) et l'insert devient une **méthode**. Le graphe compilé est **équivalent en valeurs** (un `Tensor` est un argument MLIR qu'il vienne d'un champ-model ou d'un champ-input), mais E2 doit valider l'**égalité des tokens générés**, pas une identité structurelle d'arguments.

### 3.4 Chargement des constantes (le point composé)

Les tenseurs de la brique sont des **champs de `self.brick`**, donc **partie du model** `EngineModel(Brick)`.
La reflection ZML (`meta.visit`) descend dans le champ `brick` et bufferise ses `Tensor` comme les poids —
composition native, pas de second chemin. La fixture contient : poids engine **+** tenseurs nommés de la brique
(`codebook_256`…). Brique vide → aucun tenseur en plus.

**⚠️ Contrat de chargement (load-bearing, validé review)** : le loader résout chaque `Tensor` par son **`id`**,
pas par nom de champ. Un `id` n'a de clé safetensors que si le `Tensor` a été créé via
`View.createTensor("codebook_256", …)` (qui appelle `store.bindIdToKey`, `io.zig:162`). **Donc une brique
DOIT exposer un `init(View) Brick`** qui crée ses tenseurs via la `store View` (exactement le pattern
`LayerW`/`Packed.init` du POC, `gen_vq.zig:195-198`) — sinon `getReaderById` renvoie null et le load **crashe**.
Le `load`/`Bufferized` de la sous-struct est géré nativement (`mem.zig`, `io.zig`) une fois les `id` bindés.

## 4. Validation — deux gates

| Gate | Contenu | Oracle | Critère |
|---|---|---|---|
| **E1 — non-régression** | `EngineModel(struct{})` génère N tokens | `gemma4_decode4.zig` (intact) | **tokens identiques** (4 argmax == HF greedy ; la fixture n'a pas de réf `last_hidden`) — ✅ PASS (tag `engine-e1-noregression-pass`) |
| **E2 — brique == POC** | `EngineModel(TurboQuantVBrick)` | `gemma4_gen_vq.zig` (le POC) | mêmes tokens générés (le socle remplace la copie) |

E1 prouve que la factorisation ne casse rien ; E2 prouve que la brique branchée == la copie, **sans copie**.
Discipline gate/oracle du projet (oracle → runner ZML → compare → commit+tag).

## 5. Migration

- `decode3.zig`, `decode4.zig`, `decode_vq.zig`, `gen_vq.zig` **restent intacts** (gates immuables =
  oracles E1/E2). On ne les supprime pas.
- `engine.zig` est **neuf** : extrait+généralisé de `decode4` (mêmes ops, mêmes seuils numériques).
- Une fois E1+E2 verts, le POC TurboQuant **vit comme brique** (`TurboQuantVBrick`), et toute future
  hypothèse = une nouvelle brique, jamais un fork du moteur.

## 6. Comment écrire une brique (guide, livrable du spec)

1. Définir un `struct` avec : les champs `Tensor` de ses constantes + une méthode par point voulu
   (`post_v_norm(self, v, ctx) Tensor`, etc.).
2. Exporter ses constantes dans la fixture (`.safetensors`).
3. Instancier `EngineModel(MaBrique)`, charger la fixture, compiler, run.
4. Valider contre un oracle PyTorch (le pattern du projet).
Aucune ligne de `engine.zig` à toucher tant que les points existent ; sinon ajouter le point (1 ligne).

## 7. Périmètre — dans / hors

**Dans** : `engine.zig` (`EngineModel(Brick)`, point `post_v_norm`, `LayerCtx`), brique vide (gate E1),
`TurboQuantVBrick` (gate E2), guide brique.
**Hors (au besoin)** : points supplémentaires (`post_k_rope`, `scores`…), prefill modulaire, registre de
briques / config-driven, composition de plusieurs briques, efficience (le socle reste correctness-first).

## 8. Risques

- **Comptime ZML** : `@hasDecl` + struct générique paramétrée — à valider qu'ils compilent dans le flux
  `platform.compile` (le forward d'un model générique). Risque faible (Zig comptime standard), à dé-risquer
  par un mini-build si doute.
- **Extraction fidèle de decode4** : `engine.zig` doit reproduire decode4 op-pour-op (gate E1 bit-exact le
  garantit). Risque = divergence subtile dans la boucle/cache → E1 l'attrape.
- **Adaptation des axes** : `quantizeV` opère sur `[.k,.hd]` ; le V engine est 4D → `quantizeV_4d` wrapper
  (reshape, comme en Q4/Q5).
