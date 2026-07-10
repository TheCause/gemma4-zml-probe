# G2.3 — Cartographie de sensibilité bf16 par-op : plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Un binaire ZML unique dont la précision bf16 se pilote par famille d'ops au runtime (arg positionnel `<familles>` = `fam1,fam2` ou `none`), un sweep one-hot de 12 familles + une config combinée, verdicts vs l'enveloppe G2 — le tout pré-enregistré et auditable.

**Architecture:** Refactor `PrecCfg` comptime → `PrecRt` runtime dans `engine.zig` (approche B de la spec) ; runner `gemma4_g23_sweep` piloté par `--bf16 fam1,fam2` ; orchestration bash sur la 3090 (build unique, analyse au fil de l'eau, purge) ; analyse Python (métriques vs D0/A, ratios vs enveloppe B, sanité, custody).

**Tech Stack:** Zig 0.16-dev + ZML (workspace 3090 `/data/rqz_workspace/zml`), Bazel, Python (numpy, safetensors), bash. GPU RTX 3090 (`--@zml//platforms:cuda=true`).

**Spec:** `docs/superpowers/specs/2026-07-10-g2-3-op-sensitivity-design.md` — la spec PRIME en cas de doute.

---

## Contexte opératoire (à lire avant toute tâche)

- **Machines** : le code vit sur M1 (`~/dev/gemma4-zml-probe`) ; il se compile et tourne sur la 3090 (`ssh ia@gpu-host` via jumphost, cf `zml_runner/deploy_to_3090.sh`, variables `ZML_REMOTE/ZML_JUMP/ZML_DST`). Workspace ZML : `/data/rqz_workspace/zml`, build via `./bazel.sh`.
- **Prérequis 3090** : swap actif (sinon OOM-kill exit 255 au compile XLA — vérifié par `scripts/smoke.sh`) ; patch local `@setEvalBranchQuota(100_000)` dans `pjrt.zig` (fonction `structSize`) présent.
- **Vérification** : ce projet ne teste pas par pytest mais par **gates** (compile → run → comparaison à un oracle → tag). Chaque tâche code se termine par un build (smoke) ; les gates G2.3.0/1/2 sont les vraies validations.
- **Zig 0.16-dev** : `pub fn main(init: std.process.Init)`, `init.minimal.args.toSlice(...)`, `std.Io.Dir.cwd()` + API Io threadée (cf `docs/DOCUMENTATION.md` pièges).
- **Noms de runners ≤ ~20 caractères** (quota comptime `pjrt.zig`, cf piège capitalisé).
- **Discipline commits** : un commit par étape logique, messages `feat(g2.3):` / `docs(g2.3):` / `fix(g2.3):`, jamais de `git push --force`.

---

### Task 1 : `PrecRt` runtime + refactor des sites GEMM

**Files:**
- Modify: `zml_runner/engine.zig` (lignes ~245-274 : `EngineCfg`, `PrecCfg`, `dotPrec` ; ~280-361 : `runLayerGen` ; ~366+ : `EngineModel` et ses forwards)

- [ ] **Step 1.1 : Écrire le nouveau struct `PrecRt` (remplace `PrecCfg`)**

Dans `engine.zig`, remplacer le bloc `PrecCfg` (lignes 253-264) par :

```zig
/// Config de précision RUNTIME (G2.3, approche B de la spec). Un champ par FAMILLE d'ops ;
/// `null` = f32 (baseline). Portée par le modèle comme CHAMP RUNTIME (self.prec) — plus dans
/// EngineCfg comptime : le traçage émet les converts d'après la valeur au moment du compile,
/// le binaire est unique, le graphe diffère par run. Sémantique contractuelle (spec §4) :
/// « bf16 » = arrondi des opérandes aux bornes de l'op, calcul interne au gré d'XLA.
/// NEUTRALITÉ : tout-null doit émettre un graphe identique à la baseline (convert same-dtype
/// = `return self`, cf tensor.zig).
pub const PrecRt = struct {
    compute: zml.DataType = .f32,
    // familles GEMM (spec §4, 1-7)
    qkv_proj: ?zml.DataType = null,
    qk_scores: ?zml.DataType = null,
    pv_ctx: ?zml.DataType = null,
    o_proj: ?zml.DataType = null,
    mlp: ?zml.DataType = null,
    ple: ?zml.DataType = null,
    head: ?zml.DataType = null,
    // familles non-GEMM (spec §4, 8-12)
    norms: ?zml.DataType = null,
    softmax: ?zml.DataType = null,
    rope: ?zml.DataType = null,
    softcap: ?zml.DataType = null,
    kv_store: ?zml.DataType = null,

    /// Parse "fam1,fam2" (noms des champs) → PrecRt avec ces familles à .bf16. Erreur si nom inconnu.
    pub fn fromSpecList(list: []const u8) !PrecRt {
        var p: PrecRt = .{};
        if (list.len == 0) return p;
        var it = std.mem.splitScalar(u8, list, ',');
        while (it.next()) |name| {
            var matched = false;
            inline for (@typeInfo(PrecRt).@"struct".fields) |f| {
                if (comptime std.mem.eql(u8, f.name, "compute")) continue;
                if (std.mem.eql(u8, name, f.name)) {
                    @field(p, f.name) = .bf16;
                    matched = true;
                }
            }
            if (!matched) {
                log.err("famille inconnue: '{s}'", .{name});
                return error.UnknownFamily;
            }
        }
        return p;
    }
};
```

NB : vérifier l'introspection Zig 0.16 (`@typeInfo(...).@"struct".fields` vs ancien `.Struct`) au premier build — ajuster si l'API diffère.

- [ ] **Step 1.2 : Réécrire `dotPrec` (par-famille, runtime)**

Remplacer `dotPrec` (lignes 266-274) par :

```zig
/// GEMM prec-aware par famille (G2.3). `fam` = le champ PrecRt de la famille du site d'appel.
/// fam=null : émission STRICTEMENT identique à `a.dot(convert(b))` d'aujourd'hui (neutralité).
fn dotPrec(fam: ?zml.DataType, compute: zml.DataType, a: zml.Tensor, b: zml.Tensor, comptime axis: @TypeOf(.enum_literal)) zml.Tensor {
    if (fam) |g| return a.convert(g).dot(b.convert(g), axis).convert(compute);
    return a.dot(b.convert(compute), axis);
}
```

- [ ] **Step 1.3 : Retirer `prec` de `EngineCfg` et threader `PrecRt` en runtime**

- Supprimer le champ `prec: PrecCfg = .{}` de `EngineCfg` (ligne 250).
- Dans `EngineModel(Brick, cfg)` : ajouter un champ **runtime** `prec: PrecRt = .{}` au struct du modèle (à côté de `brick`), et supprimer `const prec: PrecCfg = cfg.prec;` (ligne ~376) au profit de `self.prec` lu dans les forwards.
- `runLayerGen` : signature `comptime prec: PrecCfg` → **runtime** `prec: PrecRt` (le paramètre cesse d'être comptime ; `i` et `cfg` restent comptime).
- ⚠ À vérifier au smoke (comme le caveat `@typeInfo` du Step 1.1) : le champ runtime `prec: PrecRt` du struct modèle traverse `zml.Bufferized(Self)` / `zml.io.load` — le précédent `brick` (champ non-Tensor) suggère que oui ; si non, porter `prec` hors du struct (param de `forward` via une closure au compile).
- Adapter chaque site d'appel de `dotPrec` à sa famille :

| Site (ligne actuelle) | Appel refactoré |
|---|---|
| q_proj (289) | `dotPrec(prec.qkv_proj, prec.compute, h0, layer.q_proj, .d)` |
| k_proj (306) / v_proj (311) | `dotPrec(prec.qkv_proj, ...)` |
| scores (341) | `dotPrec(prec.qk_scores, ...)` |
| ctx_attn (346) | `dotPrec(prec.pv_ctx, ...)` |
| attn_out (348) | `dotPrec(prec.o_proj, ...)` |
| mlp gate/up/down (352) | `dotPrec(prec.mlp, ...)` ×3 |
| ple gate (355) / proj (357) | `dotPrec(prec.ple, ...)` ×2 |
| projection frontend PLE (~415) | `dotPrec(prec.ple, ...)` |
| head/lm_head (~454, 494, 536, 574) | `dotPrec(prec.head, ...)` |

- Les closures `c()` (`t.convert(prec.compute)`) passent de `prec.compute` comptime à runtime — même code, la valeur vient de `self.prec`.
- Chercher **tous** les usages restants de `PrecCfg` : `grep -n "PrecCfg" zml_runner/*.zig` — les runners qui l'utilisent sont traités en Task 4/5.

- [ ] **Step 1.4 : Vérifier la compilation à blanc (M1, zig fmt + lecture)**

Pas de toolchain ZML sur M1 : vérifier `zig fmt --check zml_runner/engine.zig` si zig dispo, sinon relecture attentive. La vraie vérification = smoke 3090 (Task 6).

- [ ] **Step 1.5 : Commit**

```bash
git add zml_runner/engine.zig
git commit -m "feat(g2.3): PrecRt runtime par famille — refactor des 11 sites dotPrec GEMM (spec §3.1)"
```

---

### Task 2 : Wrappers non-GEMM (`norms`, `softmax`, `rope`, `softcap`)

**Files:**
- Modify: `zml_runner/engine.zig` (`runLayerGen`, `rmsScaleD`, `manualRope`/`slidingRope`, forward head)

- [ ] **Step 2.1 : Helper générique d'encadrement**

```zig
/// Encadre une valeur aux bornes d'une op non-GEMM : arrondi d'entrée si la famille est active.
/// La sortie de l'op est re-upcastée par l'appelant (via .convert(compute) ou c()).
fn inPrec(fam: ?zml.DataType, x: zml.Tensor) zml.Tensor {
    if (fam) |g| return x.convert(g);
    return x;
}
```

- [ ] **Step 2.2 : Famille `norms`**

Contrat (spec §4, question ouverte tranchée ici et pré-enregistrée en Task 7) : **entrée ET poids arrondis**, sortie re-upcastée. Passer `prec` à `rmsScaleD` (ou wrapper à ses sites) :

```zig
// avant : rmsScaleD(hidden, c(layer.input_layernorm))
// après (famille norms active) :
fn rmsScaleDPrec(fam: ?zml.DataType, compute: zml.DataType, x: zml.Tensor, w: zml.Tensor) zml.Tensor {
    const xi = inPrec(fam, x);
    const wi = inPrec(fam, w.convert(compute)); // le c() d'origine reste la base
    return rmsScaleD(xi, wi).convert(compute);
}
```

Sites : input_layernorm (287), post_attention (350), pre/post_feedforward (351/353), post_per_layer_input_norm (358), q_norm/k_norm/v_norm (290/307/312 — le `zml.nn.rmsNorm` + `mul`), final_norm (forward head). Chaque site garde exactement sa sémantique actuelle quand `fam=null` (neutralité).

- [ ] **Step 2.3 : Famille `softmax`**

Ligne 343 : `const probs = scores.softmax(.k);` →

```zig
const probs = inPrec(prec.softmax, scores).softmax(.k).convert(prec.compute);
```

- [ ] **Step 2.4 : Famille `rope`**

Dans `manualRope` et `slidingRope` (et leurs appels lignes 291/308) : arrondir q/k **et** cos/sin en entrée (contrat pré-enregistré : les deux), re-upcast en sortie. Passer `prec.rope` + `prec.compute` en paramètres runtime de ces helpers.

- [ ] **Step 2.5 : Famille `softcap`**

Au forward head (site du `scale(1/30).tanh().scale(30)`) : encadrer l'entrée du softcap par `inPrec(prec.softcap, ...)` + `.convert(prec.compute)` en sortie.

- [ ] **Step 2.6 : Commit**

```bash
git add zml_runner/engine.zig
git commit -m "feat(g2.3): familles non-GEMM norms/softmax/rope/softcap — arrondi aux bornes (spec §4)"
```

---

### Task 3 : Famille `kv_store` (stockage du cache en bf16)

**Files:**
- Modify: `zml_runner/engine.zig` (`Cache`, scatter/reads dans `runLayerGen`)

**⚠ Investigation d'abord** : le dtype des tenseurs de cache vient du header de la fixture via `createTensor` (`io.zig maybeCreateTensor`). Deux mécanismes possibles, choisir au vu du code ZML local (`/data/rqz_workspace/zml` ou miroir `~/dev/zml`) :

- [ ] **Step 3.1 : Trancher le mécanisme de dtype du cache**

Option (a) — préférée si l'API le permet : déclarer les tenseurs symboliques du cache au dtype `prec.kv_store` (override à la déclaration) et convertir les buffers initiaux après load (`Buffer` f32 → bf16 via un mini-graphe de conversion, ou `convert` au premier step).
Option (b) — repli sûr : variante de fixture cache bf16 (le cache initial est structurel/zéros : ajouter un flag `--kv-dtype bf16` à `scripts/46_gen_long_oracle.py` qui écrit les tenseurs cache de la fixture en bf16 ; ~10 lignes).
Documenter le choix dans le commit et dans le doc protocole (Task 7).

- [ ] **Step 3.2 : Écritures/lectures encadrées**

Quand `prec.kv_store` est actif : `k_new`/`v_new` convertis en bf16 **avant** `scatterSlices` (lignes 324-325/333-334) ; les lectures `cache_k`/`cache_v` (choose1d, lignes 299-336) re-upcastées `.convert(prec.compute)` avant les dots QK/PV. Quand null : zéro op émise (neutralité).

Attention à l'interaction avec `qk_scores`/`pv_ctx` : si les deux familles sont actives (combiné), l'ordre des converts doit rester bf16→f32→bf16 = émis tel quel (pas d'optimisation manuelle — XLA simplifiera ou non, c'est le contrat « aux bornes »).

- [ ] **Step 3.3 : Commit**

```bash
git add zml_runner/engine.zig scripts/46_gen_long_oracle.py
git commit -m "feat(g2.3): famille kv_store — stockage cache bf16 (mécanisme: <a|b>, spec §4 contrat kv_store)"
```

---

### Task 4 : Runner `gemma4_g23_sweep` + BUILD

**Files:**
- Create: `zml_runner/gemma4_g23_sweep.zig` (copie adaptée de `gemma4_gen_long_gpu_bf16.zig`)
- Modify: `zml_runner/BUILD.bazel`

- [ ] **Step 4.1 : Créer le runner**

Base = `gemma4_gen_long_gpu_bf16.zig` (172 lignes, structure identique), différences :

```zig
// CLI : gemma4_g23_sweep <model.safetensors> <fixture.safetensors> <logits_out.bin> <familles> [max_steps]
//   <familles> : "fam1,fam2,..." (champs de PrecRt), ou "none" = baseline D0 (tout f32).
// En-tête de log OBLIGATOIRE (consommé par le manifest) : config, familles actives.
const Model = engine.EngineModel(struct {}, .{
    .two_masks = true,
    .kmax_sliding = L_MAX,
    .kmax_full = L_MAX,
    // plus de .prec ici — runtime
});
...
const fam_arg = process_args[4];
const prec: engine.PrecRt = if (std.mem.eql(u8, fam_arg, "none")) .{} else try engine.PrecRt.fromSpecList(fam_arg);
var model: Model = try .init(arena.allocator(), base);
model.prec = prec;
log.info("G2.3 config: familles=[{s}] compute=f32", .{fam_arg});
```

Le reste (load, compile, boucle steps, dump logits f32 par step, argmax indicatif, perf, mem_probe) : identique au runner G2.2. Ajouter le flag `--no-prealloc` (copie du mécanisme de `gemma4_gen_long_gpu.zig`) pour la mesure VRAM kv_store.

- [ ] **Step 4.2 : Cible BUILD.bazel**

Dupliquer l'entrée `gemma4_gen_long_gpu_bf16` de `zml_runner/BUILD.bazel` → `gemma4_g23_sweep` (mêmes deps : `//zml`, `engine.zig`, `mem_probe.zig`).

- [ ] **Step 4.3 : Commit**

```bash
git add zml_runner/gemma4_g23_sweep.zig zml_runner/BUILD.bazel
git commit -m "feat(g2.3): runner sweep unique — PrecRt par argv, dump logits, --no-prealloc"
```

---

### Task 5 : Adapter les runners existants (compat `PrecCfg` retiré)

**Files:**
- Modify: tout runner référençant `cfg.prec`/`PrecCfg` — `grep -ln "PrecCfg\|\.prec" zml_runner/*.zig`
- Delete (différé) : `zml_runner/gemma4_gen_long_gpu_bf16.zig` — **après** la non-régression G2.2 (Task 11)

- [ ] **Step 5.1 : Recenser et adapter**

`gemma4_gen_long_gpu_bf16.zig` : retirer `.prec = .{ .gemm = .bf16 }` de son EngineCfg et poser `model.prec` = config G2.2 exprimée en familles (`qkv_proj,qk_scores,pv_ctx,o_proj,mlp,ple,head` à .bf16) — il devient un alias de vérification, supprimé après Task 11. Tous les autres runners (E1/E2, gchunk*, gen_long_gpu, bench…) n'utilisent pas `.prec` (défaut) : vérifier qu'ils compilent sans changement (le champ EngineCfg.prec supprimé ne doit être référencé nulle part ailleurs).

- [ ] **Step 5.2 : Commit**

```bash
git add zml_runner/
git commit -m "feat(g2.3): runners adaptés au retrait de EngineCfg.prec (bf16 G2.2 réexprimé en PrecRt)"
```

---

### Task 6 : Smoke build 3090

- [ ] **Step 6.1 : Déployer**

```bash
ZML_REMOTE=<user@gpu-host> ZML_JUMP=<jump> ZML_DST=/data/rqz_workspace/zml/examples/rqz \
  zml_runner/deploy_to_3090.sh
```

- [ ] **Step 6.2 : Smoke (build only, tous les runners clés + le nouveau)**

Sur la 3090 : `SMOKE_TARGETS="gemma4_engine_e1 gemma4_engine_e2 gemma4_gen_long_gpu gemma4_g23_sweep gemma4_gen_long_gpu_bf16" bash /data/gemma4-zml-probe/scripts/smoke.sh`
Attendu : `OK` sur chaque cible. Toute erreur de compile se corrige ICI (leçon : le compilateur est irremplaçable), commit des fixes `fix(g2.3): ...`.

---

### Task 7 : Pré-enregistrement — doc protocole + table des converts attendus

**Files:**
- Create: `docs/G2_3_OP_SENSITIVITY.md`
- Create: `fixtures/g2_3_expected_converts.json`

- [ ] **Step 7.1 : Dériver la table des converts attendus par famille**

Depuis l'architecture (35 layers, producers 0-14, readers 15-34, YOCO writers 13/14) et le code refactoré. Format :

**⚠ Le compte se fait sur les converts ÉMIS, pas sur les opérandes** : les poids sont **déjà bf16
sur device** (dtype header safetensors, cf G2.1) et `convert` vers le même dtype **n'émet rien**
(`return self`, tensor.zig). Conséquences :
- GEMM à poids (qkv_proj, o_proj, mlp, ple, head), famille **active** : 2 converts émis par site
  (activation-in f32→bf16 + résultat-out bf16→f32) — le convert du poids est un no-op.
- MAIS la **baseline D0** émet, elle, 1 convert par site à poids (`b.convert(f32)` sur le poids
  bf16) qui **disparaît** quand la famille s'active (le poids reste bf16). Le compte pertinent est
  donc un **DELTA vs le recensement de D0**, pas un absolu.
- GEMM à 2 opérandes f32 (qk_scores, pv_ctx) : +3 converts émis par site (2 in + 1 out).

```json
{
  "qkv_proj":   {"sites": "35 q + 15 k + 15 v = 65 (YOCO: readers sans k/v)",
                 "delta_converts_vs_d0": "+2 émis −1 baseline = +1 net/site → +65"},
  "qk_scores":  {"sites": 35, "delta_converts_vs_d0": "+3/site → +105"},
  "...": "chaque famille : sites, delta attendu, dérivé en comptant les ÉMISSIONS du code refactoré"
}
```

Le script 53 mesure ce **delta** (recensement converts du run − recensement D0). Les comptes exacts
se dérivent en comptant les émissions du code (une passe de lecture de `runLayerGen` + forwards par
famille). ⚠ minutie YOCO : readers sans k/v_proj ni scatter.

- [ ] **Step 7.2 : Écrire `docs/G2_3_OP_SENSITIVITY.md` (protocole AVANT runs)**

Contenu = reprise opérationnelle de la spec : bras A/B/D0/Dᵢ/D\* ; custody (md5 attendus, remplis à la génération) ; métriques et buckets (§5.2 spec, verbatim) ; départage ; non-vacuité + comptage converts ; sanité (seuils calibrés AVANT le sweep via `52 --calibrate-sanity` sur les **memmaps A/B de la 3090** — le `g2_envelope_metrics.npz` ne contient pas d'entropie) ; procédure combinée + 3 essais ; stabilité S49 ; protocole VRAM ; contrats norms/rope tranchés (entrée+poids arrondis / cos-sin arrondis) ; note de préséance sur le croquis G2.3 du doc G2.

- [ ] **Step 7.3 : Commit (le pré-enregistrement est un commit AVANT tout run de sweep)**

```bash
git add docs/G2_3_OP_SENSITIVITY.md fixtures/g2_3_expected_converts.json
git commit -m "docs(g2.3): protocole PRÉ-ENREGISTRÉ (métriques, seuils, sanité, custody) + table converts attendus"
```

---

### Task 8 : `scripts/52_g2_3_analyze.py`

**Files:**
- Create: `scripts/52_g2_3_analyze.py` (généralise `51_g2_2_analyze.py` — le lire d'abord)

- [ ] **Step 8.1 : Écrire le script**

Interface :

```
python scripts/52_g2_3_analyze.py \
  --run-logits g2_3_logits_<fam>.bin --run-name <fam> \
  --ref-a /data/gemma4-zml-probe/g2_logits_a_f32.npy \
  --ref-d0 /data/g2_3_logits_none.bin \
  --envelope fixtures/g2_envelope_manifest.json \
  --expected-converts fixtures/g2_3_expected_converts.json \
  --hlo-report <sortie de 53, json> \
  --manifest fixtures/g2_3_manifest.json \
  --bin-sha <sha> --zml-rev <rev> --repo-rev <rev>   # provenance complète (spec §7.1)
# Flags de mode : --register-reference (run D0 : custody+sanité seulement, pas d'auto-analyse) ;
#   --ref-a none (mode vs-D0 seul, runs S49) ; --calibrate-sanity ; --selfcheck.
# Ré-analyse d'un run existant : UPSERT de l'entrée manifest (clé = run-name), avec compteur
#   re_run et md5 des deux passages — jamais d'entrée dupliquée silencieuse.
```

**⚠ Formats hétérogènes des références (vérifié dans script 50/51)** : le bras A est un **`.npy`
memmap** (`g2_logits_a_f32.npy`, écrit par `np.lib.format.open_memmap` dans le script 50, lu par
`np.load(..., mmap_mode="r")` dans le 51) ; les dumps ZML (D0, Dᵢ) sont du **f32 brut** (`.bin`,
`np.memmap(dtype=np.float32).reshape(steps, VOC)`). Le loader de 52 choisit par extension — traiter
le `.npy` comme du brut décalerait toutes les valeurs du header numpy (métriques poubelle).

Pipeline par run :
1. **Custody** : md5 de A et D0 vs manifest (première exécution : les enregistre ; ensuite : REFUSE si mismatch, exit 2).
2. **Sanité** (spec §5.4) : NaN/Inf scan ; entropie moyenne vs A ; répétition argmax. Échec → verdict `FAIL-SANITY`, entrée manifeste écrite, exit 0 (le verdict EST le résultat).
3. **Non-vacuité logits** : `max_abs(run vs D0) == 0` → verdict `VACUOUS`.
4. **Métriques** : max_abs p50/p95/max, KL p50/p95/max, argmax mismatches, 1re bifurcation — vs D0 ET vs A (mêmes formules que 51 ; lecture memmap step par step, jamais tout en RAM).
5. **Verdict** : buckets sur `KL p50 vs A / enveloppe B`, départage `max_abs p50` (le pire l'emporte).
6. **Écriture manifest + npz** : entrée manifest complète (config, provenance `bin-sha`/`zml-rev`/`repo-rev`, toutes métriques `*_vs_D0`/`*_vs_A`, ratios, verdict, converts attendus/observés, **md5 du dump logits**) **et** un npz de courbes par-step par run (max_abs/KL/match, comme le 51), nommé par 52 **depuis le run-name seul** (`<run-name>_metrics.npz` — le run-name arrive déjà préfixé par l'orchestrateur, pas de double préfixe) — c'est ce npz qui survit à la purge du dump (spec §7.2).

Cas spécial `--run-name none` : D0 est **référence seule** — pas d'auto-analyse vs soi-même (pas de verdict `VACUOUS` absurde) ; 52 enregistre seulement custody (md5) + sanité de D0 dans le manifest.

Mode calibration sanité : `--calibrate-sanity` calcule les seuils (entropie moyenne, répétition argmax) **depuis les memmaps A et B sur la 3090** (le `g2_envelope_metrics.npz` ne contient PAS d'entropie — max_abs/kl/match seulement) et les écrit dans le manifest ; à lancer AVANT le sweep, valeurs recopiées dans le doc protocole (Task 7). **⚠ Format du bras B** : `/data/gemma4-zml-probe/g2_logits_b_bf16u16.npy` = **bit-patterns bf16 stockés en uint16** (script 50 : `.view(torch.uint16)`) — réinterpréter via `torch.from_numpy(u16).view(torch.bfloat16).float()`, jamais lire comme entiers (même classe de piège que le header .npy du bras A).

Self-check intégré (pattern repo) : `--selfcheck` rejoue les métriques G2.2 depuis `g2_2_metrics.npz` et vérifie la reproduction des ratios publiés (garde anti-régression de formule).

- [ ] **Step 8.2 : Vérifier le selfcheck en local (M1)**

```bash
python3 scripts/52_g2_3_analyze.py --selfcheck
# Attendu : "SELFCHECK PASS — ratios G2.2 reproduits (max_abs p50 0.44x, KL p50 0.28x)"
```

- [ ] **Step 8.3 : Commit**

```bash
git add scripts/52_g2_3_analyze.py
git commit -m "feat(g2.3): analyse par run — custody, sanité, non-vacuité, métriques vs D0/A, verdict, manifest"
```

---

### Task 9 : `scripts/53_g2_3_hlo_check.py`

**Files:**
- Create: `scripts/53_g2_3_hlo_check.py`

- [ ] **Step 9.1 : Écrire le check HLO**

Entrées : dossier de dump XLA du run (`XLA_FLAGS=--xla_dump_to=<dir>` posé par l'orchestrateur), dossier de dump de D0 (baseline), famille, table `g2_3_expected_converts.json`.
Sorties (json) : `{"differs_from_d0": bool, "convert_ops_observed": N, "convert_ops_expected": N, "verdict": "OK|INVALID|IDENTICAL"}`.
Méthode : compter les ops `convert` dans le HLO texte du module principal — **sur le dump
PRÉ-optimisation** (`*before_optimizations*`), pas le post-opt (XLA peut simplifier les chaînes de
converts → mismatchs spurieux vs l'oracle des émissions) ; grep structuré `= bf16[...] convert(` /
`= f32[...] convert(`, diff du nombre de fichiers/hash vs D0 (réutiliser l'approche E1 `diff -rq`).

**Runs multi-familles** (D\*, S49 combiné) : l'attendu = **somme des deltas** des familles actives
(valide car Task 3.2 impose d'émettre les chaînes telles quelles) ; si la somme s'avère non
vérifiable en pratique, 53 dégrade en `differs_from_d0` seul pour les multi-familles et le consigne.

- [ ] **Step 9.2 : Commit**

```bash
git add scripts/53_g2_3_hlo_check.py
git commit -m "feat(g2.3): check HLO par run — diff vs D0 + comptage converts attendus/observés (anti-câblage-croisé)"
```

---

### Task 10 : `scripts/g2_3_sweep.sh` (orchestration 3090)

**Files:**
- Create: `scripts/g2_3_sweep.sh`

- [ ] **Step 10.1 : Écrire l'orchestrateur**

```bash
#!/usr/bin/env bash
# g2_3_sweep.sh — sweep G2.3 sur la 3090. Un BINAIRE UNIQUE pour tout le sweep (spec §7.1).
# Usage : bash g2_3_sweep.sh [liste de configs, défaut = none + les 12 one-hot]
#   KEEP=1 bash g2_3_sweep.sh "fam1,fam2"   # conserver le dump (run combiné D*, spec §7.2)
#   FIXTURE=... (défaut S46) — surchargée pour la stabilité S49 et la variante kv_store option (b)
set -euo pipefail
ZML_WS=${ZML_WS:-/data/rqz_workspace/zml}
DATA=/data/gemma4-zml-probe
FIXTURE=${FIXTURE:-$DATA/gen_long.safetensors}   # ⚠ RACINE du repo côté 3090 (cf script 46), PAS fixtures/
RUN_PREFIX=${RUN_PREFIX:-g2_3}                   # namespace des sorties — S49 utilise RUN_PREFIX=g2_3_s49
REF_A=${REF_A:-$DATA/g2_logits_a_f32.npy}        # .npy memmap (script 50) — custody vérifiée par 52 ; REF_A=none → mode vs-D0 seul (S49)
mkdir -p "$DATA/logs"
BIN=$ZML_WS/bazel-bin/examples/rqz/gemma4_g23_sweep
MIN_FREE_GB=6   # 1 dump (~1.1 Go) + dumps HLO + marge

# 0. build unique + hash + provenance (bin, workspace ZML, repo probe)
(cd "$ZML_WS" && ./bazel.sh build //examples/rqz:gemma4_g23_sweep --@zml//platforms:cuda=true)
BIN_SHA=$(sha256sum "$BIN" | cut -d' ' -f1)
ZML_REV=$(git -C "$ZML_WS" rev-parse HEAD 2>/dev/null || echo "n/a")
REPO_REV=$(git -C "$DATA" rev-parse HEAD 2>/dev/null || echo "n/a")

FAMILIES=${1:-"none qkv_proj qk_scores pv_ctx o_proj mlp ple head norms softmax rope softcap kv_store"}
for fam in $FAMILIES; do
  # check binaire inchangé + espace disque
  [ "$(sha256sum "$BIN" | cut -d' ' -f1)" = "$BIN_SHA" ] || { echo "FATAL: binaire modifié en cours de sweep"; exit 1; }
  free_gb=$(df --output=avail -BG /data | tail -1 | tr -dc '0-9')
  [ "$free_gb" -ge "$MIN_FREE_GB" ] || { echo "FATAL: espace insuffisant (${free_gb}G)"; exit 1; }

  safe=$(echo "$fam" | tr ',' '+')             # nom de fichier pour configs combinées
  hlo_dir=/data/${RUN_PREFIX}_hlo_$safe ; logits=/data/${RUN_PREFIX}_logits_$safe.bin
  XLA_FLAGS="--xla_dump_to=$hlo_dir" "$BIN" \
    "$DATA/weights/model.safetensors" "$FIXTURE" "$logits" "$fam" \
    2>&1 | tee "$DATA/logs/${RUN_PREFIX}_run_$safe.log"

  md5sum "$logits" | tee -a "$DATA/logs/${RUN_PREFIX}_md5.log"   # AVANT purge (déterminisme Task 12.1)

  if [ "$fam" = "none" ]; then
    # D0 = référence seule : custody + sanité, pas d'auto-analyse ni de diff HLO vs soi-même
    python3 "$DATA/scripts/52_g2_3_analyze.py" --run-logits "$logits" --run-name "${RUN_PREFIX}_none" \
      --ref-a "$REF_A" --ref-d0 "$logits" --register-reference \
      --manifest "$DATA/fixtures/g2_3_manifest.json" \
      --bin-sha "$BIN_SHA" --zml-rev "$ZML_REV" --repo-rev "$REPO_REV"
    continue   # jamais purgé
  fi

  python3 "$DATA/scripts/53_g2_3_hlo_check.py" --run-dir "$hlo_dir" --d0-dir /data/${RUN_PREFIX}_hlo_none \
    --family "$fam" --expected "$DATA/fixtures/g2_3_expected_converts.json" --out /tmp/hlo_${RUN_PREFIX}_$safe.json
  python3 "$DATA/scripts/52_g2_3_analyze.py" --run-logits "$logits" --run-name "${RUN_PREFIX}_$safe" \
    --ref-a "$REF_A" --ref-d0 /data/${RUN_PREFIX}_logits_none.bin \
    --envelope "$DATA/fixtures/g2_envelope_manifest.json" \
    --expected-converts "$DATA/fixtures/g2_3_expected_converts.json" \
    --hlo-report /tmp/hlo_${RUN_PREFIX}_$safe.json --manifest "$DATA/fixtures/g2_3_manifest.json" \
    --bin-sha "$BIN_SHA" --zml-rev "$ZML_REV" --repo-rev "$REPO_REV"

  # purge (spec §7.2) — on garde : D0 (jamais purgé), les npz de métriques, et le dump si KEEP=1 (run D*)
  if [ "${KEEP:-0}" != "1" ]; then rm -f "$logits" && rm -rf "$hlo_dir"; fi
done
echo "SWEEP DONE — manifest: $DATA/fixtures/g2_3_manifest.json"
```

NB : `none` tourne en PREMIER (D0 = référence des suivants, jamais purgé). Le run combiné (Task 13)
se lance avec `KEEP=1` pour conserver D\* (spec §7.2).

- [ ] **Step 10.2 : Commit**

```bash
git add scripts/g2_3_sweep.sh
git commit -m "feat(g2.3): orchestrateur sweep — binaire unique, D0 d'abord, analyse au fil de l'eau, purge, checks espace"
```

---

### Task 11 : Gate G2.3.0 — neutralité (3090, BLOQUANTE)

**⚠ Piège de circularité (à ne pas réintroduire)** : après les Tasks 1-5, TOUS les runners embarquent
l'`engine.zig` refactoré (via `srcs` de BUILD.bazel) — comparer `gemma4_g23_sweep none` à un G1
**rebuildé** ne prouverait rien (un bug de refactor qui altère les deux graphes à l'identique
passerait « gold » en silence). La baseline HLO doit venir de la révision **PRÉ-refactor**.

- [ ] **Step 11.0 : Baseline HLO pré-refactor (À FAIRE AVANT de déployer le refactor, ou depuis un worktree)**

Sur M1 : `git worktree add /tmp/g23-baseline <commit parent de Task 1>` (le dernier commit de la
branche AVANT `feat(g2.3): PrecRt...`), puis déployer CE worktree :
```bash
ZML_REMOTE=... ZML_DST=... /tmp/g23-baseline/zml_runner/deploy_to_3090.sh
# 3090 : build G1 pré-refactor + dump HLO baseline (conservé, jamais purgé)
./bazel.sh build //examples/rqz:gemma4_gen_long_gpu --@zml//platforms:cuda=true
XLA_FLAGS="--xla_dump_to=/data/g2_3_hlo_baseline_prerefactor" ./bazel-bin/examples/rqz/gemma4_gen_long_gpu \
  /data/gemma4-zml-probe/weights/model.safetensors /data/gemma4-zml-probe/gen_long.safetensors 4
```
Consigner le commit baseline + md5 du dump dans le manifest. Puis `git worktree remove /tmp/g23-baseline`
et re-déployer les sources refactorées (Task 6.1).

- [ ] **Step 11.1 : Sync du repo côté 3090 + run D0**

⚠ `deploy_to_3090.sh` ne synchronise QUE `zml_runner/` vers le workspace ZML — les scripts 52/53,
`g2_3_expected_converts.json` et le doc protocole doivent arriver par le repo :
`ssh <3090> "cd /data/gemma4-zml-probe && git fetch && git checkout g2.3-op-sensitivity && git pull"`.
Puis `bash scripts/g2_3_sweep.sh "none"` — vérifie que le binaire tourne et produit D0.

- [ ] **Step 11.2 : Preuve gold — HLO tout-null vs baseline PRÉ-refactor**

Diff façon E1 (`diff -rq`, hors chemins) : `/data/g2_3_hlo_none` vs `/data/g2_3_hlo_baseline_prerefactor`.
**Gold = identique.** Sinon : hiérarchie de repli pré-enregistrée (spec §6.1) — publier le diff
catégorisé ET exiger cumulativement G1 64/64 + E1 4/4 + replay 1020/1020 (tous rebuildés post-refactor,
comparés à leurs verdicts HISTORIQUES, pas entre eux). En dessous → STOP, retour au refactor.

- [ ] **Step 11.3 : Non-régression G2.2**

`gemma4_g23_sweep ... "qkv_proj,qk_scores,pv_ctx,o_proj,mlp,ple,head"` → 52 doit reproduire les ratios G2.2 (0.44×/0.28×, argmax 1016/1020, bifurcation 96). PASS → supprimer `gemma4_gen_long_gpu_bf16.zig` + cible BUILD (commit dédié).

- [ ] **Step 11.4 : Custody des références**

Vérifier/enregistrer md5 de A ; si A absent/mismatch → régénérer (script 50, bras A). D0 md5 enregistré.

- [ ] **Step 11.5 : Commit + tag**

```bash
git add fixtures/g2_3_manifest.json logs/ docs/G2_3_OP_SENSITIVITY.md
git commit -m "feat(g2.3): gate G2.3.0 PASS — neutralité PrecRt prouvée (HLO <gold|repli>), non-régression G2.2, custody A/D0"
git tag gate/G2.3.0-neutrality-pass
```

---

### Task 12 : Gate G2.3.1 — sweep one-hot

- [ ] **Step 12.1 : Lancer le sweep complet (nuit 3090)**

`bash scripts/g2_3_sweep.sh` (none + les 12). Déterminisme : relancer UNE config (ex `mlp`) et comparer son md5 à celui du premier passage dans `logs/g2_3_md5.log` (l'orchestrateur md5-somme chaque dump AVANT purge). Attendu : identique.

- [ ] **Step 12.2 : Rapatrier + classement**

Rapatrier manifest + logs + les `g2_3_metrics_*.npz` par run (spec §7.2 les conserve — ce sont eux qui survivent à la purge des dumps) sur M1. Vérifier : 12 verdicts non-`INVALID`, comptages converts OK. Écrire le classement (tableau par `KL p50 vs D0`) dans `docs/G2_3_OP_SENSITIVITY.md` §résultats.

NB `kv_store` : si l'option (b) de Task 3 a été retenue (fixture cache bf16), la lancer **hors du
sweep par défaut** avec sa fixture variante (`FIXTURE=<gen_long_kvbf16> bash scripts/g2_3_sweep.sh "kv_store"`)
— la lancer avec la fixture standard f32 donnerait un `VACUOUS` garanti (gaspillage détecté mais évitable).

- [ ] **Step 12.3 : Commit + tag**

```bash
git add fixtures/g2_3_manifest.json logs/ docs/G2_3_OP_SENSITIVITY.md
git commit -m "feat(g2.3): gate G2.3.1 PASS — sweep 12 familles, classement de sensibilité publié"
git tag gate/G2.3.1-sweep-pass
```

---

### Task 13 : Gate G2.3.2 — config combinée + stabilité + VRAM

- [ ] **Step 13.1 : Run combiné**

Config = familles `SAFE` du classement. `KEEP=1 bash scripts/g2_3_sweep.sh "<fam1,fam2,...>"` (D\* conservé, spec §7.2). Si FAIL au critère ≤2× : procédure pré-enregistrée (retrait glouton pire `KL p50 vs D0`, max 3 essais, CHAQUE essai committé au manifest). 3 échecs → null result publié.

- [ ] **Step 13.2 : Interaction + stabilité S49**

Métriques d'interaction (52 les calcule depuis le manifest, spec §5.5) : `KL p50(D*) / max_i KL p50(Dᵢ actives)` **et** le ratio à la **somme** des KL p50 actives.

**Stabilité S49 — sémantique PRÉ-ENREGISTRÉE (pas d'enveloppe A/B pour S49)** : les runs S49 sont
**vs D0-S49 uniquement** (diagnostic), jamais vs A/B (qui sont des trajectoires S46 — les mélanger
donnerait des métriques poubelle). Le critère de concordance = **l'ordre relatif** : les familles
testées gardent-elles leur rang de `KL p50 vs D0` relatif l'une à l'autre et au combiné ?
Concrètement :

```bash
python3 scripts/49_gen_custom_oracle.py --prompt "..." --n-tokens 48   # fixture S49 (racine)
# runs namespacés — nouveau D0-S49 d'abord, puis combiné + 2 familles frontières, refs S46 intouchées
RUN_PREFIX=g2_3_s49 FIXTURE=/data/gemma4-zml-probe/gen_custom.safetensors REF_A=none \
  bash scripts/g2_3_sweep.sh "none <combiné> <fam_frontière_1> <fam_frontière_2>"
```

(`REF_A=none` → 52 saute custody/métriques vs A et n'écrit que les `*_vs_D0` ; le namespace
`RUN_PREFIX` protège D0-S46 et son md5 de tout écrasement.) Les seuils de sanité étant calibrés
sur S46, la sanité des runs S49 est **informative-only** (pas de FAIL-SANITY bloquant sur le
diagnostic de stabilité). Concordance ou discordance : publiée telle quelle au manifest et au doc.

- [ ] **Step 13.3 : VRAM `kv_store` (protocole G2.1)**

GPU vierge, `--no-prealloc`, pic PID, 2 mesures : run `kv_store` vs référence 8 494 MiB. Consigner au manifest.

- [ ] **Step 13.4 : Doc résultats + commit + tag + PR**

```bash
git add fixtures/g2_3_manifest.json logs/ docs/G2_3_OP_SENSITIVITY.md
git commit -m "feat(g2.3): gate G2.3.2 — verdict combiné <PASS|null result>, interaction, stabilité S49, VRAM kv_store"
git tag gate/G2.3.2-combined-<pass|null>
git push -u origin g2.3-op-sensitivity && gh pr create --base main ...
```

MAJ `PLANNING.md` (G2.3 soldé) + fiche mémoire projet.

---

## Ordre et dépendances

Tasks 1→5 (code, M1) sont séquentielles. Task 6 (smoke) valide 1-5. Tasks 7-10 (protocole + scripts) peuvent se faire en parallèle de 6. Task 11 bloque 12, 12 bloque 13. **Aucun run de sweep avant le commit du protocole (Task 7) et le PASS de G2.3.0 (Task 11).**

## Ce que le plan ne couvre pas (voir spec §10)

fp8/int8, run exploratoire TOLERABLE (décision post-résultats), transfert TurboQuant/alambic.
