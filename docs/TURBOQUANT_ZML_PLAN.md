# TurboQuant × ZML V-only — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Porter le quantizer KV TurboQuant **MSE V-only** dans le moteur decode ZML de Gemma-4-E2B-it et mesurer le coût/gain sur la génération, garde-fou = fidélité vs HuggingFace.

**Architecture:** Constantes MSE (codebook + Hadamard, par head_dim 256/512) exportées offline en `.safetensors`. Quantizer ZML `v → v_hat = dequant(quant(v))` (fake-quant) inséré au point cache du producer V de `gemma4_decode3.zig`. Validation gate-par-gate (oracle PyTorch → runner ZML → compare → commit+tag), pattern P5.* du projet. Q1 (Hadamard) + Q2 (nearest-centroid) déjà prouvés bit-exact par le spike `gemma4_hadq.zig`.

**Tech Stack:** Zig 0.16-dev + ZML (Bazel) sur RTX 3090 ; PyTorch 2.12 + transformers 5.9 (oracle, `TurboQuantMSE`) ; safetensors. Spec : `docs/TURBOQUANT_ZML_DESIGN.md`.

---

## Environnement (rappel)

- **Compute** : `ssh user@gpu-host` (accès direct, pas de jumphost). Venv `/data/venvs/gemma4-probe` (torch 2.12, transformers 5.9). Modèle `google/gemma-4-E2B-it` en cache `/data/hf_cache` (utiliser `HF_HOME=/data/hf_cache HF_HUB_OFFLINE=1`). **Écrire uniquement sur `/data`** (`/` à 90 %).
- **Workspace ZML** : `/data/rqz_workspace/zml/examples/rqz/`. Build : `cd /data/rqz_workspace/zml && ./bazel.sh build //examples/rqz:<cible>`. Run : `./bazel-bin/examples/rqz/<cible> <fixture.safetensors>`.
- **turboquant.py** : déployé `/data/gemma4-zml-probe/turboquant.py` (version M1 `~/dev/turboquant/turboquant.py`). `sys.path.insert(0, "/data/gemma4-zml-probe")`.
- **Déploiement fichiers** : `scp <fichier> user@gpu-host:<dest>` (direct, validé). Sources Zig → workspace ; oracles Python + fixtures → `/data/gemma4-zml-probe/`.
- **Pièges ZML capitalisés** (spike) : `gather(.{.c=idx})` exige `idx.squeeze(.c)` (argMax garde l'axe réduit) ; **le codebook DOIT être tagué `.c`** (pas `.k`, déjà pris par l'axe-position du cache) ; format log Zig `{e:.1}` (pas `{e:.0e}`) ; reshape perd les tags → `.withTags` ; pas de broadcast implicite → `.broad` ; noms runner ≤ 20 c.
- **Artefacts spike déjà dans le repo** (rapatriés 2026-06-04) : `zml_runner/gemma4_hadq.zig` (+ cible BUILD locale), `scripts/spike_hadq_oracle.py`, `scripts/test_kv_quant_generation.py`, `scripts/measure_k_distribution_gemma4.py`. Le worker les lit/copie en local (pas besoin d'aller sur le 3090 pour le squelette).
- **Source de vérité du build = workspace 3090.** Pour chaque nouvelle cible, **append idempotent** au BUILD 3090 (`grep -q <cible> $B || cat bloc >> $B`), ne PAS écraser le BUILD 3090 par le BUILD local (ils peuvent diverger). Garder le BUILD local M1 synchronisé en parallèle (pour le commit).

## File Structure

| Fichier | Responsabilité | Action |
|---|---|---|
| `scripts/30_export_turboquant_constants.py` | Exporte `codebook_256/512` + `hadamard_256/512` en `.safetensors` | Create |
| `scripts/31_vquant_oracle.py` | Q3 oracle : `v` jouet + `v_hat` attendu via `TurboQuantMSE` | Create |
| `zml_runner/gemma4_vquant.zig` | Q3 : quantizer MSE V-only standalone (chaîne complète), runner compare | Create |
| `scripts/32_decode_vq_oracle.py` | Q4 oracle : decode HF 1-step avec **V quantifié** (fake-quant au point v_norm) | Create |
| `zml_runner/gemma4_decode_vq.zig` | Q4 : copie de `gemma4_decode3.zig` + insertion quant V au write cache | Create |
| `scripts/33_gen_vq_measure.py` | Q5 : génération HF V-quant vs baseline + sortie séquence référence | Create |
| `zml_runner/gemma4_gen_vq.zig` | Q5 : copie de `gemma4_decode4.zig` (boucle génération, cache threadé) + insertion `quantizeV` | Create |
| `zml_runner/BUILD.bazel` | cibles `gemma4_vquant`, `gemma4_decode_vq`, `gemma4_gen_vq` | Modify |

> `gemma4_decode3.zig` n'est **PAS modifié** (gate `p5.7.7` immuable) : on crée une variante `gemma4_decode_vq.zig`.

---

## Task 0 : Export des constantes MSE (codebook + Hadamard)

**Files:**
- Create: `scripts/30_export_turboquant_constants.py`
- Output: `/data/gemma4-zml-probe/turboquant_constants.safetensors`

- [ ] **Step 1 — Écrire l'export.** `scripts/30_export_turboquant_constants.py` :

```python
import os, sys
os.environ.setdefault("HF_HOME", "/data/hf_cache")
import torch
from safetensors.torch import save_file
sys.path.insert(0, "/data/gemma4-zml-probe")
from turboquant import TurboQuantMSE  # _make_hadamard accessible via mse.Pi

B_BITS = 4  # K = 16 niveaux (V-only 4 bits, cf test A)
OUT = "/data/gemma4-zml-probe/turboquant_constants.safetensors"
state = {}
for d in (256, 512):
    mse = TurboQuantMSE(d, B_BITS, device="cpu", rotation="hadamard")
    state[f"codebook_{d}"] = mse.codebook.to(torch.float32).contiguous()        # [K]
    state[f"hadamard_{d}"] = mse.Pi.to(torch.float32).contiguous()              # [d,d], Pi[e,d]
save_file(state, OUT)
print("Wrote", OUT, {k: tuple(v.shape) for k, v in state.items()})
```

- [ ] **Step 2 — Déployer + exécuter.**
```bash
scp scripts/30_export_turboquant_constants.py user@gpu-host:/data/gemma4-zml-probe/
ssh user@gpu-host 'source /data/venvs/gemma4-probe/bin/activate && HF_HOME=/data/hf_cache python /data/gemma4-zml-probe/30_export_turboquant_constants.py'
```
Expected : `Wrote ... {'codebook_256': (16,), 'hadamard_256': (256,256), 'codebook_512': (16,), 'hadamard_512': (512,512)}`

- [ ] **Step 3 — Commit.** `git add scripts/30_export_turboquant_constants.py && git commit -m "feat(tq-zml): export MSE codebook+Hadamard constants (256/512)"`

---

## Task 1 (Gate Q3) : Quantizer MSE V-only complet en ZML

Étend le spike (`gemma4_hadq.zig`, Hadamard + nearest-centroid bit-exact) avec norm L2, inverse-Hadamard, ×norm → chaîne `v → v_hat` complète, validée contre `MSE.dequant(MSE.quant(v))`.

**Files:**
- Create: `scripts/31_vquant_oracle.py`, `zml_runner/gemma4_vquant.zig`
- Modify: `zml_runner/BUILD.bazel`

- [ ] **Step 1 — Oracle Q3.** `scripts/31_vquant_oracle.py` : tenseur V jouet `[K_pos=8, d]` pour d∈{256,512}, calcule `v_hat = MSE.dequant(MSE.quant(v))` (chaîne de référence), exporte fixture `spike_vquant_<d>.safetensors` avec `v` (tag `.k,.hd`), `codebook` (`.c`), `hadamard` (`.e,.hd`), `v_hat_oracle` (`.k,.hd`). Réutiliser `TurboQuantMSE(d,4,rotation='hadamard')`. **Note** : `MSE.quant` normalise par `norm L2` puis `u@Pi.T` → reproduire EXACTEMENT (norm = `linalg.norm(v,dim=-1,keepdim=True)`).

- [ ] **Step 2 — Runner Q3.** `zml_runner/gemma4_vquant.zig` : copier le squelette de `gemma4_hadq.zig`. Fixture `{v:[.k,.hd], hadamard:[.e,.hd], codebook:[.c], v_hat_oracle:[.k,.hd]}`. **Écrire la chaîne comme une fonction libre `quantizeV` réutilisable telle quelle en Q4** — donc **AUCUNE constante globale** (`gemma4_decode3.zig` a déjà `D=1536`, collision) : dériver le shape de broadcast dynamiquement via `.dim()`.

```zig
/// v:[.k,.hd], cb:[.c], Pi:[.e,.hd]  ->  v_hat:[.k,.hd]. Reutilisable en Q4.
fn quantizeV(v: zml.Tensor, cb: zml.Tensor, Pi: zml.Tensor) zml.Tensor {
    // norm L2 par vecteur — MSE.quant ARRONDIT la norm en fp16 (turboquant.py:267) : reproduire.
    const norm = v.mul(v).sum(.hd).sqrt().convert(.f16).convert(.f32);  // [.k,.hd=1]
    const u = v.div(norm);                                  // broadcast (.hd=1)
    const y = u.dot(Pi, .hd);                               // [.k,.e]  (= u @ Pi.T)
    // shape cible DERIVEE DYNAMIQUEMENT (pas de D/K/KPOS globaux)
    const target = zml.Shape.init(.{ y.dim(.k), y.dim(.e), cb.dim(.c) }, .f32)
        .withTags(.{ .k, .e, .c });
    const yr3 = y.appendAxes(.{.c}).broad(target);          // [.k,.e,.c]
    const cb3 = cb.insertAxes(0, .{ .k, .e }).broad(target);
    const diff = yr3.sub(cb3);
    const idx = diff.mul(diff).scale(-1.0).argMax(.c).indices.squeeze(.c);  // [.k,.e]
    const y_hat = cb.gather(.{ .c = idx }, .{});            // [.k,.e]
    const u_hat = y_hat.dot(Pi, .e);                        // [.k,.hd] (= y_hat @ Pi)
    return u_hat.mul(norm);
}
// forward standalone Q3 :
pub fn forward(self: VquantFixture) zml.Tensor {
    return quantizeV(self.v, self.codebook, self.hadamard);
}
```
Ajouter la cible `gemma4_vquant` à `BUILD.bazel` (copier le bloc `gemma4_q_proj`).
**Note norm fp16** : si l'oracle Q3 reste sur fp32 (chaîne idéalisée), retirer le `.convert(.f16).convert(.f32)` ; sinon (oracle = `MSE.dequant(MSE.quant(v))`, recommandé) le garder pour matcher l'arrondi fp16 de `turboquant.py:267`. **Choisir l'option B (oracle MSE round-trip + fp16 ZML).**

- [ ] **Step 3 — Build.** Déployer le runner + ajouter la cible au BUILD 3090 (idempotent, sans écraser) :
```bash
scp zml_runner/gemma4_vquant.zig user@gpu-host:/data/rqz_workspace/zml/examples/rqz/
# bloc cible dans /tmp/vquant_block.txt (copie de gemma4_q_proj, name+main=gemma4_vquant)
scp /tmp/vquant_block.txt user@gpu-host:/tmp/
ssh user@gpu-host 'B=/data/rqz_workspace/zml/examples/rqz/BUILD.bazel; grep -q gemma4_vquant $B || cat /tmp/vquant_block.txt >> $B'
ssh user@gpu-host 'cd /data/rqz_workspace/zml && ./bazel.sh build //examples/rqz:gemma4_vquant 2>&1 | tail -3'
```
Expected : `Build completed successfully`. (Mettre aussi à jour `zml_runner/BUILD.bazel` local pour le commit.)

- [ ] **Step 4 — Run + compare (d=256).** `scp scripts/31_vquant_oracle.py` ; générer fixture ; `./bazel-bin/examples/rqz/gemma4_vquant /data/gemma4-zml-probe/spike_vquant_256.safetensors`. Expected : `max_abs ≤ 1e-4` (cf spec Q3 ; quelques flips frontière possibles, tolérer comme dans le spike). Répéter d=512.

- [ ] **Step 5 — Commit + tag.** `git add scripts/31_vquant_oracle.py zml_runner/gemma4_vquant.zig zml_runner/BUILD.bazel && git commit -m "feat(tq-zml): Q3 — MSE V-only quantizer chain (norm+Hadamard+nearest-centroid+inv) ZML vs oracle" && git tag tq-zml-q3-vquant-pass`

---

## Task 2 (Gate Q4) : Insertion dans le decode (V quantifié, 1 step)

**Files:**
- Create: `scripts/32_decode_vq_oracle.py`, `zml_runner/gemma4_decode_vq.zig`
- Modify: `zml_runner/BUILD.bazel`

- [ ] **Step 1 — Oracle Q4.** `scripts/32_decode_vq_oracle.py` : copier l'oracle decode existant (`scripts/*decode3_oracle.py`), mais appliquer le **fake-quant V** : hook `register_forward_hook` sur les `v_norm` des producers (cf `test_kv_quant_generation.py`) qui remplace V par `fake_quant_v(V)` (= `MSE.dequant(MSE.quant(V))`, quantizer par head_dim, constantes du Task 0). Exporter la fixture decode (caches V-quant, etc.) + `last_hidden`/`logits`/`argmax` de référence **HF-V-quant**.

- [ ] **Step 2 — Runner Q4.** `zml_runner/gemma4_decode_vq.zig` : copier `gemma4_decode3.zig`. Charger en plus `codebook_256/512` + `hadamard_256/512`. Insérer le quantizer V **après `v_norm`, avant `scatterSlices`** (producer branch, ~L221-230) : remplacer `v_new` par sa version quantifiée via la chaîne du Task 1 (factoriser en `fn quantizeV(v, cb, Pi) Tensor`). Sélectionner `{cb,Pi}` par le flag `full` (dispatch `isFull(i)` existant L174). **K reste fp32** (V-only). Ajouter cible `gemma4_decode_vq`.

- [ ] **Step 3 — Build + run.** Build `//examples/rqz:gemma4_decode_vq` ; run sur la fixture du Step 1. Expected (cf spec §5, ZML-quant vs HF-quant) : `last_hidden` **max_abs ≤ 1e-2 ET mean_abs ≤ 1e-4** ; argmax ZML == argmax HF-V-quant. (Le portage ne doit rien ajouter à la quant : bit-near attendu.)

- [ ] **Step 4 — Commit + tag.** `git commit -m "feat(tq-zml): Q4 — V-quant insere au point cache, decode 1-step ZML == HF-V-quant" && git tag tq-zml-q4-decode-vq-pass`

---

## Task 3 (Gate Q5) : Génération comparée — coût & portage

**Files:**
- Create: `scripts/33_gen_vq_measure.py`
- (Réutilise `gemma4_decode_vq.zig` en boucle, ou étend le moteur génération `gemma4_decode4.zig` avec quant V.)

- [ ] **Step 1 — Mesure HF (coût).** `scripts/33_gen_vq_measure.py` : étendre `test_kv_quant_generation.py` au seul mode `vonly_mse4`, **≥ 8 prompts**, N ≥ 48 tokens, greedy. Sortir : métriques de coût (divergence-point, KL, top5, argmax-match vs baseline) **et** la séquence générée HF-V-quant par prompt (référence pour le portage). **PAS de ROUGE-1.**

- [ ] **Step 2 — Génération ZML-quant.** **Copier `gemma4_decode4.zig` → `gemma4_gen_vq.zig`** (la boucle génération avec cache threadé est DÉJÀ prouvée là : `NUM_STEPS`, boucle `while`, scatter `(.slot,.k=pos)`, cache réinjecté) et **insérer `quantizeV` après `v_norm`** (même point qu'en Q4, ~L230 ; réutiliser la fonction factorisée). Ne PAS réécrire la plomberie cache. Ajouter la cible `gemma4_gen_vq`. (Si le budget se tend : `NUM_STEPS` modeste — 8-16 tokens — suffit pour mesurer le portage ; documenter la limite. C'est un copier-coller de decode4 + 1 insert, pas une boucle à inventer.)

- [ ] **Step 3 — Compare (portage).** Séquence ZML-V-quant vs HF-V-quant (Step 1) : argmax-match ≥ 95 %. Documenter le gain compression **théorique** (4 bits + 1 fp16 norm / vecteur de d vs 16 bits).

- [ ] **Step 4 — Rapport + commit.** Écrire `docs/TURBOQUANT_ZML_RESULTS.md` (coût V-only mesuré, fidélité portage, limites). `git commit -m "feat(tq-zml): Q5 — generation V-quant, cost+portage measured" && git tag tq-zml-q5-generation-pass`. Mettre à jour la mémoire `turboquant.md`.

---

## Notes de séquencement

- Q3 → Q4 → Q5 strictement séquentiels (chaque gate ferme avant le suivant — discipline `feedback_one_gate_at_a_time`).
- Q5 Step 2 (boucle génération ZML quantifiée) est le morceau le plus lourd ; si le budget se tend, livrer Q3+Q4 (le quantizer ZML prouvé inséré + validé 1-step) est déjà un résultat complet et publiable comme jalon.
- Ne pas pousser le repo public sans accord de Régis (local-only par défaut).
