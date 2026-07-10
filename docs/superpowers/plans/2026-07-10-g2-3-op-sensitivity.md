# G2.3 — Cartographie de sensibilité bf16 par-op : plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Un binaire ZML unique dont la précision bf16 se pilote par famille d'ops au runtime, un sweep one-hot de 12 familles + une config combinée, verdicts vs l'enveloppe G2 — le tout pré-enregistré et auditable.

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

```json
{
  "qkv_proj":   {"convert_pairs": "35 q_proj + 15 k_proj + 15 v_proj = 65 sites", "hlo_convert_ops_expected": 195},
  "qk_scores":  {"sites": 35, "hlo_convert_ops_expected": 105},
  "...": "chaque famille : sites, converts émis attendus (2 opérandes + 1 re-upcast par site GEMM ; in+out par op non-GEMM)"
}
```

Les comptes exacts se dérivent en comptant les émissions du code (une passe de lecture de `runLayerGen` + forwards par famille). ⚠ minutie YOCO : readers sans k/v_proj ni scatter.

- [ ] **Step 7.2 : Écrire `docs/G2_3_OP_SENSITIVITY.md` (protocole AVANT runs)**

Contenu = reprise opérationnelle de la spec : bras A/B/D0/Dᵢ/D\* ; custody (md5 attendus, remplis à la génération) ; métriques et buckets (§5.2 spec, verbatim) ; départage ; non-vacuité + comptage converts ; sanité (seuils calibrés sur A/B : à remplir depuis `g2_envelope_metrics.npz` AVANT le sweep) ; procédure combinée + 3 essais ; stabilité S49 ; protocole VRAM ; contrats norms/rope tranchés (entrée+poids arrondis / cos-sin arrondis) ; note de préséance sur le croquis G2.3 du doc G2.

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
  --run-logits g2_3_run_<fam>.bin --run-name <fam> \
  --ref-a <path A.bin> --ref-d0 <path D0.bin> \
  --envelope fixtures/g2_envelope_manifest.json \
  --expected-converts fixtures/g2_3_expected_converts.json \
  --hlo-report <sortie de 53, json> \
  --manifest fixtures/g2_3_manifest.json   # append/update l'entrée du run
```

Pipeline par run :
1. **Custody** : md5 de A et D0 vs manifest (première exécution : les enregistre ; ensuite : REFUSE si mismatch, exit 2).
2. **Sanité** (spec §5.4) : NaN/Inf scan ; entropie moyenne vs A ; répétition argmax. Échec → verdict `FAIL-SANITY`, entrée manifeste écrite, exit 0 (le verdict EST le résultat).
3. **Non-vacuité logits** : `max_abs(run vs D0) == 0` → verdict `VACUOUS`.
4. **Métriques** : max_abs p50/p95/max, KL p50/p95/max, argmax mismatches, 1re bifurcation — vs D0 ET vs A (mêmes formules que 51 ; lecture memmap step par step, jamais tout en RAM).
5. **Verdict** : buckets sur `KL p50 vs A / enveloppe B`, départage `max_abs p50` (le pire l'emporte).
6. **Écriture manifest** : entrée complète (config, provenance binaire/revs passées par args, toutes métriques `*_vs_D0`/`*_vs_A`, ratios, verdict, converts attendus/observés).

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
Méthode : compter les ops `convert` dans le HLO texte du module principal (grep structuré `= bf16[...] convert(` / `= f32[...] convert(`), diff du nombre de fichiers/hash vs D0 (réutiliser l'approche E1 `diff -rq`).

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
# Usage : bash g2_3_sweep.sh [liste de configs, défaut = les 12 one-hot + none]
set -euo pipefail
ZML_WS=${ZML_WS:-/data/rqz_workspace/zml}
DATA=/data/gemma4-zml-probe
BIN=$ZML_WS/bazel-bin/examples/rqz/gemma4_g23_sweep
MIN_FREE_GB=6   # 1 dump (~1.1 Go) + dumps HLO + marge

# 0. build unique + hash + provenance
(cd "$ZML_WS" && ./bazel.sh build //examples/rqz:gemma4_g23_sweep --@zml//platforms:cuda=true)
BIN_SHA=$(sha256sum "$BIN" | cut -d' ' -f1)
ZML_REV=$(git -C "$ZML_WS" rev-parse HEAD 2>/dev/null || echo "n/a")

FAMILIES=${1:-"none qkv_proj qk_scores pv_ctx o_proj mlp ple head norms softmax rope softcap kv_store"}
for fam in $FAMILIES; do
  # check binaire inchangé + espace disque
  [ "$(sha256sum "$BIN" | cut -d' ' -f1)" = "$BIN_SHA" ] || { echo "FATAL: binaire modifié en cours de sweep"; exit 1; }
  free_gb=$(df --output=avail -BG /data | tail -1 | tr -dc '0-9')
  [ "$free_gb" -ge "$MIN_FREE_GB" ] || { echo "FATAL: espace insuffisant (${free_gb}G)"; exit 1; }

  hlo_dir=/data/g2_3_hlo_$fam ; logits=/data/g2_3_logits_$fam.bin
  XLA_FLAGS="--xla_dump_to=$hlo_dir" "$BIN" \
    "$DATA/weights/model.safetensors" "$DATA/fixtures/gen_long.safetensors" "$logits" "$fam" \
    2>&1 | tee "$DATA/logs/g2_3_run_$fam.log"

  python3 "$DATA/scripts/53_g2_3_hlo_check.py" --run-dir "$hlo_dir" --d0-dir /data/g2_3_hlo_none \
    --family "$fam" --expected "$DATA/fixtures/g2_3_expected_converts.json" --out /tmp/hlo_$fam.json
  python3 "$DATA/scripts/52_g2_3_analyze.py" --run-logits "$logits" --run-name "$fam" \
    --ref-a /data/g2_logits_a.bin --ref-d0 /data/g2_3_logits_none.bin \
    --envelope "$DATA/fixtures/g2_envelope_manifest.json" \
    --expected-converts "$DATA/fixtures/g2_3_expected_converts.json" \
    --hlo-report /tmp/hlo_$fam.json --manifest "$DATA/fixtures/g2_3_manifest.json" \
    --bin-sha "$BIN_SHA" --zml-rev "$ZML_REV"

  # purge (on garde D0 = "none" et les manifests ; spec §7.2)
  [ "$fam" != "none" ] && rm -f "$logits" && rm -rf "$hlo_dir"
done
echo "SWEEP DONE — manifest: $DATA/fixtures/g2_3_manifest.json"
```

NB : `none` tourne en PREMIER (D0 = référence des suivants, jamais purgé). Ajuster les chemins réels de A (`g2_logits_a.bin`, généré en G2.0 — custody vérifiée par 52).

- [ ] **Step 10.2 : Commit**

```bash
git add scripts/g2_3_sweep.sh
git commit -m "feat(g2.3): orchestrateur sweep — binaire unique, D0 d'abord, analyse au fil de l'eau, purge, checks espace"
```

---

### Task 11 : Gate G2.3.0 — neutralité (3090, BLOQUANTE)

- [ ] **Step 11.1 : Rerun de déploiement + D0**

Deploy (Task 6.1), puis run baseline : `bash scripts/g2_3_sweep.sh "none"` — vérifie que le binaire tourne et produit D0.

- [ ] **Step 11.2 : Preuve gold — HLO tout-null vs baseline G1**

Dump HLO de `gemma4_gen_long_gpu` (G1, inchangé) et de `gemma4_g23_sweep none` → diff façon E1 (`diff -rq` des dossiers, hors chemins). **Gold = identique.** Sinon : appliquer la hiérarchie de repli pré-enregistrée (spec §6.1) — publier le diff catégorisé ET exiger G1 64/64 + E1 4/4 + replay 1020/1020. En dessous → STOP, retour au refactor.

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

`bash scripts/g2_3_sweep.sh` (les 12 + none). Déterminisme : relancer UNE config (ex `mlp`) et vérifier logits bit-identiques (md5 des dumps avant purge — l'orchestrateur log le md5 de chaque dump).

- [ ] **Step 12.2 : Rapatrier + classement**

Rapatrier manifest + logs sur M1. Vérifier : 12 verdicts non-`INVALID`, comptages converts OK. Écrire le classement (tableau par `KL p50 vs D0`) dans `docs/G2_3_OP_SENSITIVITY.md` §résultats.

- [ ] **Step 12.3 : Commit + tag**

```bash
git add fixtures/g2_3_manifest.json logs/ docs/G2_3_OP_SENSITIVITY.md
git commit -m "feat(g2.3): gate G2.3.1 PASS — sweep 12 familles, classement de sensibilité publié"
git tag gate/G2.3.1-sweep-pass
```

---

### Task 13 : Gate G2.3.2 — config combinée + stabilité + VRAM

- [ ] **Step 13.1 : Run combiné**

Config = familles `SAFE` du classement. `bash scripts/g2_3_sweep.sh "<fam1,fam2,...>"`. Si FAIL au critère ≤2× : procédure pré-enregistrée (retrait glouton pire `KL p50 vs D0`, max 3 essais, CHAQUE essai committé au manifest). 3 échecs → null result publié.

- [ ] **Step 13.2 : Interaction + stabilité S49**

Métrique d'interaction (52 la calcule depuis le manifest : `KL(D*) / max KL(Dᵢ)`). Stabilité : regénérer la fixture S49 (`49_gen_custom_oracle.py`), runs combiné + 2 familles frontières, verdicts concordants ou discordance publiée.

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
