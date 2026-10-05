# SD — Couche de décision typée (Choice mono-token) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Mesurer, sur le moteur ZML de Gemma 4 E2B, ce que rapporte une décision lue directement dans les logits (bras C) face à une étiquette générée (B) et à un appel JSON généré (A), sans toucher au moteur ni au runner existant.

**Architecture:** Un runner neuf `gemma4_decide.zig`, clone ciblé de `gemma4_gen_auto.zig` (duplication assumée, précédent `gemma4_w4auto`), compile une entrée `StepDec` = gather + `forwardStep` (inchangé) + top-5 + gather des 4 candidats + logsumexp. Un évaluateur pur (`sd_policy.zig`, sans ZML) transforme les 4 logits en `Decision`. Les ids viennent d'un oracle HF (script 84) ; un dépouilleur Python (86) calcule les contrôles et les prédictions, une contre-épreuve (87) prouve qu'il condamne.

**Tech Stack:** Zig (toolchain du workspace ZML, via Bazel sur la 3090), ZML/PJRT CUDA, Python 3 (venv `gemma4-probe` sur la 3090 : transformers 5.9.0, torch ; M1 : stdlib seule pour 86/87).

**Source de vérité :** `docs/superpowers/specs/2026-09-24-decision-layer-design.md` (rév. 4, commit `7226fc7`). **En cas d'écart plan/spec, LA SPEC FAIT FOI** ; tout écart découvert est écrit dans la spec (note de révision datée) AVANT d'être codé.

---

## Carte des fichiers

| Fichier | Responsabilité | Tâche |
|---|---|---|
| `scripts/88_sd_hlo_fingerprint.py` | empreinte normalisée du module HLO pré-opt principal (réutilise `53_g2_3_hlo_check.py`) | 1 |
| `fixtures/sd_cases.json` | 24 requêtes + classe attendue, prompts gabarits | 2 |
| `scripts/84_sd0_oracle.py` | ids HF, `label_ids`, C-SD-A, logits HF des candidats, lse, top-5 | 3 |
| `fixtures/sd_manifest.json`, `fixtures/sd_oracle.json` | produits par 84 (versionnés) | 3 |
| `zml_runner/sd_policy.zig` | évaluateur pur (softmax, marge, masse, erreurs) + `test` blocks | 4 |
| `zml_runner/gemma4_decide.zig` | runner : chargement, `StepDec`, boucle par cas, chronométrage, JSONL | 5-6 |
| `zml_runner/BUILD.bazel` | cibles `gemma4_decide` et `sd_policy_test` | 4-5 |
| `scripts/86_sd_report.py` | contrôles C-SD-B/B′/D/E/F, P1/P2, qualité → `docs/SD_RESULTS.md` | 8 |
| `scripts/87_sd_selfproof.py` | mutants (a)-(e), vérifie que 86 condamne | 9 |
| `docs/SD_RESULTS.md`, `docs/evidence/sd/*` | résultats et logs rapatriés | 7, 10 |
| `PLANNING.md` | entrée « chantier SD » | 10 |
| `zml_runner/engine.zig`, `zml_runner/gemma4_gen_auto.zig` | **INTOUCHÉS** (C-SD-C) | — |

**Conventions de la machine de calcul** (dépôt PUBLIC : jamais d'IP ni d'utilisateur réel dans un fichier versionné — les commandes ci-dessous utilisent `$GPU` = `user@gpu-host`, à exporter dans le shell) :
- workspace ZML distant : `$ZML_WS` (ex. `/data/rqz_workspace/zml`), runners dans `$ZML_WS/examples/rqz/` ;
- copie de travail des scripts/fixtures : `/data/gemma4-zml-probe/` (**pas un dépôt git** : on y synchronise `scripts/` et `fixtures/` par `rsync`) ;
- poids : `/data/gemma4-zml-probe/weights/model.safetensors` ; venv : `/data/venvs/gemma4-probe/bin/python3` ;
- build : TOUJOURS `ZML_REMOTE=$GPU ZML_WS=$ZML_WS TARGETS="…" zml_runner/build_3090.sh` (les deux flags de mode ; un log chronométré sans `BUILD: mode=ReleaseFast` est INEXÉCUTABLE) ;
- binaire construit : `$ZML_WS/bazel-bin/examples/rqz/<cible>`.

---

### Task 0: Décision VRAM (GO Régis) et synchronisation

Constat du 24 sept 2026 : la 3090 a **19 152 MiB** occupés par `llama-server` (Ollama, `qwen3.8:27b-64k`, 17,4 Go). Le runner E2B a un pic mesuré de **16 658 MiB** (`gemma4_gen_auto.zig:676`) : il ne tient pas à côté. `checkVram` refusera le lancement (c'est son rôle).

- [ ] **Step 1: Mesurer l'état au moment de lancer**

```bash
ssh $GPU 'nvidia-smi --query-gpu=memory.used,memory.total --format=csv; curl -s localhost:11434/api/ps'
```
Attendu : soit la VRAM est libre (modèle expiré), soit un modèle Ollama est chargé.

- [ ] **Step 2: Si un modèle Ollama est chargé → DEMANDER à Régis** (ne jamais décharger sans GO : d'autres projets consomment ce service). Formulation : « Qwen 27B occupe 17,4 Go de la 3090 ; je le décharge (`keep_alive: 0`) pour ~30 min de runs SD, il se rechargera au prochain appel. GO ? ». Sur GO :

```bash
ssh $GPU 'curl -s localhost:11434/api/generate -d "{\"model\":\"qwen3.8:27b-64k\",\"keep_alive\":0}"; sleep 3; nvidia-smi --query-gpu=memory.used --format=csv'
```
Attendu : `memory.used` < 1000 MiB. Consigner l'heure et le GO dans `docs/evidence/sd/RUN_LOG.md`.

- [ ] **Step 3: Vérifier la présence des entrées existantes**

```bash
ssh $GPU 'ls -la /data/gemma4-zml-probe/weights/model.safetensors /data/gemma4-zml-probe/gen_custom.safetensors; ls $HOME/.cache 2>/dev/null; ls /data/hf_cache/hub | grep -i gemma-4-E2B'
```
Attendu : les trois présents. Si `gen_custom.safetensors` manque : le régénérer par `scripts/49_gen_custom_oracle.py --prompt "What is the capital of France? Answer in one word." --n-tokens 48` (commande A1 de `docs/GEN_AUTONOME_PLAN.md:430`).

---

### Task 1: Témoin HLO et A1 sur `main` — AVANT toute ligne du runner (C-SD-C)

**Files:**
- Create: `scripts/88_sd_hlo_fingerprint.py`
- Create: `docs/evidence/sd/hlo_witness_main.txt`

Note de spec : la spec écrit « md5 » ; l'empreinte réellement calculée est le **sha256 normalisé** de `53_g2_3_hlo_check.py::normalized_hash` (même rôle, neutralisations éprouvées). Consigner cet écart de nom dans la spec (rév. 4a) au Step 5.

- [ ] **Step 1: Écrire le script d'empreinte**

```python
#!/usr/bin/env python3
"""88 — Empreinte du module HLO pré-optimisation principal d'un dump XLA (C-SD-C).

Réutilise les neutralisations éprouvées de 53_g2_3_hlo_check.py (chemins de dump, --xla_dump_to,
id numérique éphémère du nom de module). Module principal = le plus gros *before_optimizations*.txt.
Refuse bruyamment : dump absent, aucun pré-opt, décodage corrompu, module principal ambigu.

Usage : python3 scripts/88_sd_hlo_fingerprint.py <dump_dir>  → imprime `sha256 <hex> <fichier>`
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("hlo53", HERE / "53_g2_3_hlo_check.py")
hlo53 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hlo53)


def main() -> int:
    if len(sys.argv) != 2:
        sys.exit("usage: 88_sd_hlo_fingerprint.py <dump_dir>")
    d = sys.argv[1]
    files = hlo53.collect_before_opt(d, "dump")          # exit 1 explicite si rien
    txt = [p for p in files if p.suffix == ".txt"]
    if not txt:
        sys.exit(f"[erreur] aucun pré-opt .txt dans {d}")
    main_mod, ambiguous = hlo53.pick_main(txt)
    if ambiguous:
        sys.exit(f"[erreur] module principal ambigu dans {d} (2e fichier ≥ 90 % du 1er) — trancher à la main")
    h, corrupted = hlo53.normalized_hash(main_mod, [str(Path(d).resolve()), d])
    if corrupted:
        sys.exit(f"[erreur] décodage UTF-8 corrompu : {main_mod}")
    print(f"sha256 {h} {main_mod.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Vérifier qu'il refuse un dossier vide (contre-épreuve)**

Run: `mkdir -p /tmp/sd_empty && python3 scripts/88_sd_hlo_fingerprint.py /tmp/sd_empty; echo "exit=$?"`
Expected: message `[erreur] dump: ... aucun fichier *before_optimizations*` et `exit=1`.

- [ ] **Step 3: Déployer `main` et construire `gemma4_gen_auto`** (le dépôt local est sur `sd-decision-layer` = `main` + spec : aucun fichier Zig ne diffère)

```bash
git diff --stat main -- zml_runner/   # attendu : vide
ZML_REMOTE=$GPU ZML_DST=$ZML_WS/examples/rqz zml_runner/deploy_to_3090.sh
ZML_REMOTE=$GPU ZML_WS=$ZML_WS TARGETS="//examples/rqz:gemma4_gen_auto" zml_runner/build_3090.sh 2>&1 | tee docs/evidence/sd/build_witness.log
ssh $GPU "sha256sum $ZML_WS/bazel-bin/examples/rqz/gemma4_gen_auto" | tee -a docs/evidence/sd/build_witness.log
rsync -a scripts/53_g2_3_hlo_check.py scripts/88_sd_hlo_fingerprint.py $GPU:/data/gemma4-zml-probe/scripts/
```

- [ ] **Step 4: Run A1 avec dump HLO, puis empreinte**

```bash
ssh $GPU 'cd /data/gemma4-zml-probe && rm -rf /tmp/sd_hlo_main && TOK=$(find /data/hf_cache -path "*gemma-4-E2B-it*" -name tokenizer.json | head -1) && XLA_FLAGS=--xla_dump_to=/tmp/sd_hlo_main '"$ZML_WS"'/bazel-bin/examples/rqz/gemma4_gen_auto weights/model.safetensors "$TOK" --prompt "What is the capital of France? Answer in one word." --oracle gen_custom.safetensors 2>&1 | tail -5; /data/venvs/gemma4-probe/bin/python3 scripts/88_sd_hlo_fingerprint.py /tmp/sd_hlo_main 2>&1' | tee docs/evidence/sd/hlo_witness_main.txt
```
Expected : une ligne `A1 PASS — 48/48 argmax-match` et une ligne `sha256 <hex> <module>`. Si A1 échoue sur `main` : **STOP**, remonter à Régis (le témoin n'a pas de sens sur un main rouge).

- [ ] **Step 5: Consigner l'écart md5→sha256 dans la spec et committer**

Dans la spec, en tête sous l'encart rév. 4, ajouter :
```
> **Rév. 4a (24 sept)** — C-SD-C : l'empreinte est le **sha256 normalisé** de
> `53_g2_3_hlo_check.py::normalized_hash` (script `88_sd_hlo_fingerprint.py`), pas un md5 brut ;
> même rôle, neutralisations éprouvées. Témoin `main` : `docs/evidence/sd/hlo_witness_main.txt`.
```
```bash
git add scripts/88_sd_hlo_fingerprint.py docs/evidence/sd/ docs/superpowers/specs/2026-09-24-decision-layer-design.md
git commit -m "sd(task1): témoin HLO + A1 sur main avant tout code du runner (C-SD-C)"
```

---

### Task 2: Cas de test figés

**Files:**
- Create: `fixtures/sd_cases.json`

- [ ] **Step 1: Écrire le fichier de cas** (contenu exact ; écrit AVANT toute mesure, jamais retouché après un run — biais d'auteur déclaré en spec §4.1)

```json
{
  "version": 1,
  "classes": ["direct", "search", "calculate", "insufficient"],
  "letter_template": "Decide how to handle the request below.\n{options}\n\nRequest: {q}\n\nAnswer with a single letter: A, B, C or D.",
  "json_template": "Decide how to handle the request below.\n{options}\n\nRequest: {q}\n\nAnswer with JSON only: {\"choice\": \"<direct|search|calculate|insufficient>\"}",
  "option_text": {
    "direct": "answer directly without any tool",
    "search": "search the documentation",
    "calculate": "perform a calculation",
    "insufficient": "not enough information to choose"
  },
  "perms": {
    "orig": ["direct", "search", "calculate", "insufficient"],
    "rev": ["insufficient", "calculate", "search", "direct"]
  },
  "json_option_order": ["direct", "search", "calculate", "insufficient"],
  "warmup": "Decide how to handle the request below.\nA: answer directly without any tool\nB: search the documentation\n\nRequest: Say hello.\n\nAnswer with a single letter: A or B.",
  "cases": [
    {"id": "direct-01", "gold": "direct", "q": "What is the capital of Japan?"},
    {"id": "direct-02", "gold": "direct", "q": "Translate 'good morning' into Spanish."},
    {"id": "direct-03", "gold": "direct", "q": "Give me a synonym for 'happy'."},
    {"id": "direct-04", "gold": "direct", "q": "Who wrote the novel 'Pride and Prejudice'?"},
    {"id": "direct-05", "gold": "direct", "q": "Write a one-line birthday wish for a friend."},
    {"id": "direct-06", "gold": "direct", "q": "What colour do you get by mixing blue and yellow?"},
    {"id": "search-01", "gold": "search", "q": "Which configuration key sets the request timeout in our internal API gateway?"},
    {"id": "search-02", "gold": "search", "q": "What does error code E1047 mean in the billing service's manual?"},
    {"id": "search-03", "gold": "search", "q": "How do I rotate the signing keys according to our deployment runbook?"},
    {"id": "search-04", "gold": "search", "q": "What is the default retention period for audit logs in the product documentation?"},
    {"id": "search-05", "gold": "search", "q": "Which environment variable enables verbose tracing in the SDK, per its reference docs?"},
    {"id": "search-06", "gold": "search", "q": "What are the rate limits of the v2 export endpoint as documented?"},
    {"id": "calculate-01", "gold": "calculate", "q": "What is 17.5% of 2,348.60?"},
    {"id": "calculate-02", "gold": "calculate", "q": "Compute the monthly payment on a 250,000 loan at 3.9% annual interest over 20 years."},
    {"id": "calculate-03", "gold": "calculate", "q": "How many seconds are there in 7 weeks and 3 days?"},
    {"id": "calculate-04", "gold": "calculate", "q": "What is the square root of 98,596?"},
    {"id": "calculate-05", "gold": "calculate", "q": "Convert 68 degrees Fahrenheit to Celsius, to two decimals."},
    {"id": "calculate-06", "gold": "calculate", "q": "If 3 machines make 450 parts in 6 hours, how many parts do 5 machines make in 10 hours?"},
    {"id": "insufficient-01", "gold": "insufficient", "q": "Can you fix it?"},
    {"id": "insufficient-02", "gold": "insufficient", "q": "How much will it cost?"},
    {"id": "insufficient-03", "gold": "insufficient", "q": "Is the second one better?"},
    {"id": "insufficient-04", "gold": "insufficient", "q": "Send it to them before the deadline."},
    {"id": "insufficient-05", "gold": "insufficient", "q": "What should I change in that file?"},
    {"id": "insufficient-06", "gold": "insufficient", "q": "Why did it fail this time?"}
  ]
}
```

Rendu des options (normatif, utilisé par 84) : pour une permutation `P` (liste de 4 classes), `{options}` = les 4 lignes `"{L}: {option_text[P[i]]}"` pour `L` dans `A,B,C,D`, jointes par `\n`. Pour le prompt JSON, `{options}` = les 4 lignes `"- {c}: {option_text[c]}"` pour `c` dans `json_option_order`.

- [ ] **Step 2: Valider la forme**

Run: `python3 -c "import json,collections;d=json.load(open('fixtures/sd_cases.json'));c=collections.Counter(x['gold'] for x in d['cases']);print(len(d['cases']),dict(c));assert len(d['cases'])==24 and set(c.values())=={6};assert len({x['id'] for x in d['cases']})==24"`
Expected: `24 {'direct': 6, 'search': 6, 'calculate': 6, 'insufficient': 6}`

- [ ] **Step 3: Commit**

```bash
git add fixtures/sd_cases.json
git commit -m "sd(task2): 24 cas figés avant toute mesure (6 par classe)"
```

---

### Task 3: Oracle HF et contrôle C-SD-A (SD0)

**Files:**
- Create: `scripts/84_sd0_oracle.py`
- Create (produits): `fixtures/sd_manifest.json`, `fixtures/sd_oracle.json`, `docs/evidence/sd/84_oracle.log`

- [ ] **Step 1: Écrire l'oracle**

```python
#!/usr/bin/env python3
"""84 — SD0 : ids HF, label_ids indépendants, contrôle C-SD-A, logits HF des candidats (spec SD §4.2).

Modèle : MÊME construction que l'oracle A1 (`build_hybrid_model` de 49_gen_custom_oracle.py : poids
fp32 sauf embed_tokens_per_layer bf16), logits = softcap·tanh((last_hidden @ embed_tokensᵀ)/softcap)
en fp32 — la définition déjà utilisée par tout le dépôt.
Ids : apply_chat_template(tokenize=True) UNIQUEMENT (jamais texte → tok(text), qui doublerait le BOS).
Sortie : fixtures/sd_manifest.json (consommé par le runner) et fixtures/sd_oracle.json (consommé par 86).
Tout échec de C-SD-A ARRÊTE le script (exit 2) sans écrire le manifest.

Usage (3090, venv gemma4-probe) : python3 scripts/84_sd0_oracle.py [--cases fixtures/sd_cases.json]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/data/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
import transformers
from transformers import AutoConfig, AutoTokenizer

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("o49", HERE / "49_gen_custom_oracle.py")
o49 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(o49)

LETTERS = ["A", "B", "C", "D"]
BOS = 2


def render_letter(cases, q, perm):
    opts = "\n".join(f"{L}: {cases['option_text'][c]}" for L, c in zip(LETTERS, perm))
    return cases["letter_template"].replace("{options}", opts).replace("{q}", q)


def render_json(cases, q):
    opts = "\n".join(f"- {c}: {cases['option_text'][c]}" for c in cases["json_option_order"])
    return cases["json_template"].replace("{options}", opts).replace("{q}", q)


def chat_ids(tok, msgs, gen_prompt, thinking_kw):
    kw = dict(tokenize=True, add_generation_prompt=gen_prompt)
    kw.update(thinking_kw)
    out = tok.apply_chat_template(msgs, **kw)
    if isinstance(out, dict) or hasattr(out, "input_ids"):
        out = out["input_ids"]
    if out and isinstance(out[0], list):
        out = out[0]
    return [int(x) for x in out]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="fixtures/sd_cases.json")
    ap.add_argument("--manifest", default="fixtures/sd_manifest.json")
    ap.add_argument("--oracle", default="fixtures/sd_oracle.json")
    args = ap.parse_args()
    cases = json.load(open(args.cases))

    tok = AutoTokenizer.from_pretrained(o49.MODEL_ID)
    # enable_thinking : passé seulement si le template le connaît (vérifié, pas supposé — spec §4.1)
    tmpl = tok.chat_template or ""
    thinking_kw = {"enable_thinking": False} if "enable_thinking" in tmpl else {}
    print(f"template: enable_thinking {'PRÉSENT → False' if thinking_kw else 'ABSENT → rendu sans le paramètre'}")

    # label_ids INDÉPENDANTS du rendu (C-SD-A (ii))
    label_ids = []
    for L in LETTERS:
        e = tok.encode(L, add_special_tokens=False)
        if len(e) != 1:
            print(f"C-SD-A FAIL : la lettre {L!r} encode en {len(e)} ids {e}")
            return 2
        label_ids.append(int(e[0]))
    eot = tok.encode("<turn|>", add_special_tokens=False)
    if len(eot) != 1:
        print(f"C-SD-A FAIL : '<turn|>' encode en {eot}")
        return 2
    eot_id = int(eot[0])

    failures = []

    def check_bos(ids, name):
        if ids[0] != BOS or ids[1] == BOS:
            failures.append(f"{name}: BOS absent ou doublé, ids[:3]={ids[:3]}")

    letter_entries, json_entries = [], []
    for c in cases["cases"]:
        for perm_name, perm in cases["perms"].items():
            text = render_letter(cases, c["q"], perm)
            msgs = [{"role": "user", "content": text}]
            ids = chat_ids(tok, msgs, True, thinking_kw)
            name = f"{c['id']}/{perm_name}"
            check_bos(ids, name)
            for L, lid in zip(LETTERS, label_ids):       # C-SD-A (i) + (ii)
                full = chat_ids(tok, msgs + [{"role": "assistant", "content": L}], False, thinking_kw)
                if full[:len(ids)] != ids:
                    failures.append(f"{name}/{L}: (i) le rendu d'historique ne prolonge pas le prompt")
                elif len(full) <= len(ids) or full[len(ids)] != lid:
                    got = full[len(ids)] if len(full) > len(ids) else None
                    failures.append(f"{name}/{L}: (ii) id lu {got} ≠ label_id {lid}")
            letter_entries.append({"case_id": c["id"], "perm": perm_name, "gold": c["gold"],
                                   "ids": ids, "label_ids": label_ids, "classes": list(perm)})
        text = render_json(cases, c["q"])
        ids = chat_ids(tok, [{"role": "user", "content": text}], True, thinking_kw)
        check_bos(ids, f"{c['id']}/json")
        json_entries.append({"case_id": c["id"], "gold": c["gold"], "ids": ids})
    warm = chat_ids(tok, [{"role": "user", "content": cases["warmup"]}], True, thinking_kw)
    check_bos(warm, "warmup")

    if failures:
        print(f"C-SD-A FAIL — {len(failures)} échec(s) ; SD0 ARRÊTÉ, remonter à Régis :")
        for f in failures[:40]:
            print("  " + f)
        return 2
    print(f"C-SD-A PASS — {len(letter_entries)} prompts lettre × 4 lettres, {len(json_entries)} prompts JSON, BOS unique partout")

    # logits HF (même définition que l'oracle A1)
    cfg = AutoConfig.from_pretrained(o49.MODEL_ID)
    tc = getattr(cfg, "text_config", cfg)
    softcap = float(getattr(tc, "final_logit_softcapping", 30.0))
    model = o49.build_hybrid_model(tc).to(o49.DEVICE)
    lm_w = model.embed_tokens.weight.to(torch.float32)

    oracle = {}
    for e in letter_entries:
        with torch.no_grad():
            out = model(input_ids=torch.tensor([e["ids"]], device=o49.DEVICE), use_cache=False)
        lh = out.last_hidden_state.to(torch.float32)[0, -1, :]
        lg = softcap * torch.tanh((lh @ lm_w.t()) / softcap)
        t5 = torch.topk(lg, 5)
        oracle[f"{e['case_id']}/{e['perm']}"] = {
            "zc": {str(i): float(lg[i].item()) for i in e["label_ids"]},
            "lse": float(torch.logsumexp(lg, 0).item()),
            "top5_idx": [int(x) for x in t5.indices.tolist()],
            "top5_val": [float(x) for x in t5.values.tolist()],
        }

    from huggingface_hub import hf_hub_download, scan_cache_dir
    tok_json = Path(hf_hub_download(o49.MODEL_ID, "tokenizer.json"))  # hors ligne : lit le cache
    revs = [r.commit_hash for repo in scan_cache_dir().repos if repo.repo_id == o49.MODEL_ID for r in repo.revisions]
    meta = {
        "tokenizer_json_md5": hashlib.md5(tok_json.read_bytes()).hexdigest(),
        "model_revisions_in_cache": revs,
        "transformers": transformers.__version__, "torch": torch.__version__,
        "model_id": o49.MODEL_ID, "enable_thinking_passed": bool(thinking_kw),
        "chat_template_sha256": hashlib.sha256(tmpl.encode()).hexdigest(),
        "cases_sha256": hashlib.sha256(Path(args.cases).read_bytes()).hexdigest(),
        "softcap": softcap, "device": o49.DEVICE,
    }
    json.dump({"meta": meta, "eot_id": eot_id, "warmup": {"ids": warm},
               "letter": letter_entries, "json": json_entries},
              open(args.manifest, "w"), indent=1)
    json.dump({"meta": meta, "cases": oracle}, open(args.oracle, "w"), indent=1)
    lens_l = [len(e["ids"]) for e in letter_entries]
    lens_j = [len(e["ids"]) for e in json_entries]
    print(f"longueurs prompts lettre min/med/max = {min(lens_l)}/{sorted(lens_l)[len(lens_l)//2]}/{max(lens_l)}")
    print(f"longueurs prompts JSON   min/med/max = {min(lens_j)}/{sorted(lens_j)[len(lens_j)//2]}/{max(lens_j)}")
    print(f"wrote {args.manifest} + {args.oracle}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Lancer sur la 3090**

```bash
rsync -a scripts/49_gen_custom_oracle.py scripts/84_sd0_oracle.py $GPU:/data/gemma4-zml-probe/scripts/
rsync -a fixtures/sd_cases.json $GPU:/data/gemma4-zml-probe/fixtures/
ssh $GPU 'cd /data/gemma4-zml-probe && /data/venvs/gemma4-probe/bin/python3 scripts/84_sd0_oracle.py' 2>&1 | tee docs/evidence/sd/84_oracle.log
```
Expected : `C-SD-A PASS — 48 prompts lettre × 4 lettres, 24 prompts JSON, BOS unique partout`, puis les longueurs. Si `C-SD-A FAIL` : **STOP**, montrer le log à Régis (spec §7 : pas de requalification automatique).

- [ ] **Step 3: Rapatrier, reporter les longueurs dans la spec, committer**

```bash
rsync -a $GPU:/data/gemma4-zml-probe/fixtures/sd_manifest.json $GPU:/data/gemma4-zml-probe/fixtures/sd_oracle.json fixtures/
```
Dans la spec §8 P2, ajouter une ligne datée : `Longueurs mesurées en SD0 (84_oracle.log) : lettre médiane N_l, JSON médiane N_j` (sans toucher aux seuils — c'est ce que la spec prévoit).
```bash
git add scripts/84_sd0_oracle.py fixtures/sd_manifest.json fixtures/sd_oracle.json docs/evidence/sd/84_oracle.log docs/superpowers/specs/2026-09-24-decision-layer-design.md
git commit -m "sd(task3): SD0 — C-SD-A PASS, manifest + oracle HF des candidats"
```

---

### Task 4: Évaluateur pur `sd_policy.zig` (TDD, C-SD-F)

**Files:**
- Create: `zml_runner/sd_policy.zig`
- Modify: `zml_runner/BUILD.bazel` (ajout d'une cible de test)

- [ ] **Step 1: Écrire les tests d'abord (le module ne contient que des déclarations qui échouent)**

```zig
// sd_policy.zig — Évaluateur de décision SD (spec docs/superpowers/specs/2026-09-24-decision-layer-design.md
// §4.4-4.5). PUR : aucune dépendance ZML — testable seul (`bazel test //examples/rqz:sd_policy_test`)
// et appelé tel quel par gemma4_decide (`--policy-selftest` rejoue les mêmes cas sur le binaire livré).
const std = @import("std");

pub const N = 4;
pub const Class = enum { direct, search, calculate, insufficient };
pub const ErrKind = enum { empty_set, duplicate_label, label_out_of_vocab, non_finite };

pub const Decision = union(enum) {
    decision: struct { option: Class, p: [N]f32, p_max: f32, margin: f32, mass_in: f32 },
    abstain: struct { option: Class, p: [N]f32, margin: f32, mass_in: f32, reason: enum { insufficient, low_margin } },
    err: ErrKind,
};

pub const Input = struct {
    zc: [N]f32, // logits après softcap, dans l'ordre des lettres A..D
    lse: f32, // logsumexp du vocabulaire complet
    label_ids: [N]u32,
    classes: [N]Class, // affectation lettre→classe du cas (propre au perm)
    vocab: u32,
    n_valid: usize = N, // < N simule un ensemble tronqué (empty_set si 0)
    theta: f32 = 0, // V1 : 0 (aucune abstention par marge)
};

pub fn evaluate(in: Input) Decision {
    _ = in;
    @panic("not implemented");
}

pub fn classFromStr(s: []const u8) ?Class {
    return std.meta.stringToEnum(Class, s);
}

// ---------------------------------------------------------------- tests (C-SD-F)
const ids_ok = [N]u32{ 236776, 236799, 236780, 236796 }; // valeurs quelconques < vocab
const orig = [N]Class{ .direct, .search, .calculate, .insufficient };
const rev = [N]Class{ .insufficient, .calculate, .search, .direct };

fn base(zc: [N]f32, classes: [N]Class) Input {
    return .{ .zc = zc, .lse = 20.0, .label_ids = ids_ok, .classes = classes, .vocab = 262144 };
}

pub const selftest_cases = [_]struct { name: []const u8, in: Input, want: []const u8 }{
    .{ .name = "empty_set", .in = blk: {
        var i = base(.{ 1, 2, 3, 4 }, orig);
        i.n_valid = 0;
        break :blk i;
    }, .want = "err:empty_set" },
    .{ .name = "duplicate_label", .in = blk: {
        var i = base(.{ 1, 2, 3, 4 }, orig);
        i.label_ids[3] = i.label_ids[0];
        break :blk i;
    }, .want = "err:duplicate_label" },
    .{ .name = "label_out_of_vocab", .in = blk: {
        var i = base(.{ 1, 2, 3, 4 }, orig);
        i.label_ids[2] = 262144;
        break :blk i;
    }, .want = "err:label_out_of_vocab" },
    .{ .name = "non_finite", .in = base(.{ 1, std.math.nan(f32), 3, 4 }, orig), .want = "err:non_finite" },
    .{ .name = "argmax_pos0", .in = base(.{ 9, 2, 3, 4 }, orig), .want = "decision:direct" },
    .{ .name = "argmax_pos3_insufficient", .in = base(.{ 1, 2, 3, 9 }, orig), .want = "abstain:insufficient" },
    .{ .name = "argmax_pos1_rev", .in = base(.{ 1, 9, 3, 4 }, rev), .want = "decision:calculate" },
    .{ .name = "argmax_pos2", .in = base(.{ 1, 2, 9, 4 }, orig), .want = "decision:calculate" },
    .{ .name = "argmax_pos3_rev", .in = base(.{ 1, 2, 3, 9 }, rev), .want = "decision:direct" }, // spec C-SD-F : argmax en position 3 → .decision
};

/// Σp = 1 ± 1e-6 pour toute variante qui porte p (vérifié AUSSI par --policy-selftest, binaire livré).
pub fn sumOk(d: Decision) bool {
    const p = switch (d) {
        .decision => |x| x.p,
        .abstain => |x| x.p,
        .err => return true,
    };
    var s: f64 = 0;
    for (p) |v| s += v;
    return @abs(s - 1.0) <= 1e-6;
}

pub fn describe(d: Decision, buf: []u8) []const u8 {
    return switch (d) {
        .decision => |x| std.fmt.bufPrint(buf, "decision:{s}", .{@tagName(x.option)}) catch "?",
        .abstain => |x| std.fmt.bufPrint(buf, "abstain:{s}", .{@tagName(x.reason)}) catch "?",
        .err => |e| std.fmt.bufPrint(buf, "err:{s}", .{@tagName(e)}) catch "?",
    };
}

test "C-SD-F : chaque cas fabriqué rend la variante attendue" {
    var buf: [64]u8 = undefined;
    for (selftest_cases) |c| {
        const got = describe(evaluate(c.in), &buf);
        std.testing.expectEqualStrings(c.want, got) catch |e| {
            std.debug.print("cas {s} : got {s} want {s}\n", .{ c.name, got, c.want });
            return e;
        };
    }
}

test "p somme à 1 et mass_in dans ]0,1]" {
    const d = evaluate(base(.{ 9, 2, 3, 4 }, orig));
    const x = d.decision;
    var s: f32 = 0;
    for (x.p) |v| s += v;
    try std.testing.expectApproxEqAbs(@as(f32, 1), s, 1e-6);
    try std.testing.expect(x.mass_in > 0 and x.mass_in <= 1);
    try std.testing.expect(x.margin > 0 and x.p_max == x.p[0]);
}
```

Ajouter à `zml_runner/BUILD.bazel` : remplacer la 1ʳᵉ ligne par `load("@rules_zig//zig:defs.bzl", "zig_binary", "zig_test")` et ajouter en fin de fichier :

```python
# SD (spec docs/superpowers/specs/2026-09-24-decision-layer-design.md §4.4) — évaluateur pur, sans ZML.
zig_test(
    name = "sd_policy_test",
    main = "sd_policy.zig",
)
```

- [ ] **Step 2: Lancer les tests, vérifier l'échec**

```bash
ZML_REMOTE=$GPU ZML_DST=$ZML_WS/examples/rqz zml_runner/deploy_to_3090.sh
ssh $GPU "cd $ZML_WS && bazel test -c opt --@rules_zig//zig/settings:mode=release_fast //examples/rqz:sd_policy_test --test_output=errors" 2>&1 | tail -20
```
Expected: FAIL avec `panic: not implemented`. (Si `zig_test` n'existe pas dans la version de `rules_zig` du workspace : à la place, déplacer les tests dans le mode `--policy-selftest` de la Task 5 et l'écrire dans la spec, rév. 4b.)

- [ ] **Step 3: Implémenter `evaluate`**

Remplacer le corps de `evaluate` par :

```zig
pub fn evaluate(in: Input) Decision {
    // Validations de schéma AVANT toute lecture de p (spec §4.4) : une distribution concentrée
    // ne court-circuite jamais une erreur.
    if (in.n_valid == 0) return .{ .err = .empty_set };
    for (0..N) |i| {
        if (in.label_ids[i] >= in.vocab) return .{ .err = .label_out_of_vocab };
        for (0..i) |j| if (in.label_ids[i] == in.label_ids[j]) return .{ .err = .duplicate_label };
    }
    for (in.zc) |z| if (!std.math.isFinite(z)) return .{ .err = .non_finite };
    if (!std.math.isFinite(in.lse)) return .{ .err = .non_finite };

    // Softmax stable T=1 (§4.5), calcul en f64, rendu en f32.
    var m: f64 = in.zc[0];
    for (in.zc) |z| m = @max(m, @as(f64, z));
    var e: [N]f64 = undefined;
    var s: f64 = 0;
    for (in.zc, 0..) |z, i| {
        e[i] = @exp(@as(f64, z) - m);
        s += e[i];
    }
    var p: [N]f32 = undefined;
    var best: usize = 0;
    for (0..N) |i| {
        p[i] = @floatCast(e[i] / s);
        if (e[i] > e[best]) best = i; // égalité : la plus petite lettre (déterministe)
    }
    var second: f32 = 0;
    for (0..N) |i| if (i != best) {
        second = @max(second, p[i]);
    };
    const margin = p[best] - second;
    const mass_in: f32 = @floatCast(@exp(m + @log(s) - @as(f64, in.lse)));
    const option = in.classes[best];

    if (option == .insufficient)
        return .{ .abstain = .{ .option = option, .p = p, .margin = margin, .mass_in = mass_in, .reason = .insufficient } };
    if (margin < in.theta)
        return .{ .abstain = .{ .option = option, .p = p, .margin = margin, .mass_in = mass_in, .reason = .low_margin } };
    return .{ .decision = .{ .option = option, .p = p, .p_max = p[best], .margin = margin, .mass_in = mass_in } };
}
```

- [ ] **Step 4: Relancer les tests**

Même commande qu'au Step 2. Expected: `PASSED` (2 tests).

- [ ] **Step 5: Mutation manuelle (le test mord)** — remplacer temporairement `if (e[i] > e[best])` par `if (e[i] < e[best])` (argmin), relancer : Expected FAIL sur `argmax_pos0`. Restaurer, relancer : PASS. Noter les deux sorties dans `docs/evidence/sd/policy_test.log`.

- [ ] **Step 6: Commit**

```bash
git add zml_runner/sd_policy.zig zml_runner/BUILD.bazel docs/evidence/sd/policy_test.log
git commit -m "sd(task4): évaluateur pur sd_policy.zig + tests C-SD-F (mutant argmin vu échouer)"
```

---

### Task 5: Runner `gemma4_decide.zig` — squelette, manifest, selftest, compile

**Files:**
- Create: `zml_runner/gemma4_decide.zig`
- Modify: `zml_runner/BUILD.bazel`

- [ ] **Step 1: Créer le fichier par copie ciblée de `gemma4_gen_auto.zig`** (duplication assumée ; `gen_auto` n'est PAS modifié)

Copier VERBATIM depuis `zml_runner/gemma4_gen_auto.zig` @ `main` :
- l.31-52 (`const std`, imports, `std_options`, constantes `L_MAX`…`NUM_FULL_SLOTS`, `Model`, `PackedLong`) — **sans** `BOS_ID` (l.53-55 : le runner ne préfixe rien, spec §3.3) ; vérifier à la copie que la l.52 clôt bien `PackedLong` ;
- l.156-234 : `MASK_MIN` (l.158), constantes `ROPE_FULL_*` (l.177-180), `ropeFull` (l.204), `maskRows` (se termine l.234) ;
- `HostInputs` (l.240-333) ;
- `Tabs` et `EPTL_KEY` (l.466-491 : `EMB_KEY` inutile, `SgTabs` inutile) ;
- `parseFreeMiB` et `checkVram` (l.664-736, avec leur commentaire de seuil).

Ajouter en tête un commentaire d'en-tête :

```zig
// SD — runner de DÉCISION typée par lecture des logits (spec
// docs/superpowers/specs/2026-09-24-decision-layer-design.md, plan docs/superpowers/plans/2026-09-24-decision-layer.md).
// Clone ciblé de gemma4_gen_auto.zig (duplication assumée, précédent gemma4_w4auto) : engine.zig et
// gemma4_gen_auto.zig INTOUCHÉS (C-SD-C). Entrée compilée : `StepDec.forward` = gather + forwardStep
// (inchangé) + topK(5) + gather des 4 candidats + logSumExp. Ids lus dans le manifest (BOS inclus :
// RIEN n'est préfixé ici). Évaluateur : sd_policy.zig (pur).
//
// CLI : gemma4_decide <model.safetensors> --manifest f --arm {letter,json} --reps R --out f.jsonl
//       [--order {fwd,both}] [--policy-selftest] [--allow-cpu] [--force-vram]
const builtin = @import("builtin");
const policy = @import("sd_policy.zig");
```

- [ ] **Step 2: Arguments et lecture du manifest**

```zig
const Arm = enum { letter, json };
const Order = enum { fwd, both };

const Args = struct {
    ckpt: []const u8,
    manifest: ?[]const u8 = null,
    arm: Arm = .letter,
    reps: usize = 5,
    out: ?[]const u8 = null,
    order: Order = .fwd,
    policy_selftest: bool = false,
    allow_cpu: bool = false,
    force_vram: bool = false,
};

const usage = "Usage: gemma4_decide <model.safetensors> --manifest f --arm {letter,json} --reps R --out f.jsonl [--order {fwd,both}] [--policy-selftest] [--allow-cpu] [--force-vram]";

fn nextVal(pa: []const [:0]const u8, i: *usize, flag: []const u8) ![]const u8 {
    i.* += 1;
    if (i.* >= pa.len) {
        log.err("{s} attend une valeur", .{flag});
        return error.MissingArgument;
    }
    return pa[i.*];
}

fn parseArgs(pa: []const [:0]const u8) !Args {
    if (pa.len < 2) {
        log.err("{s}", .{usage});
        return error.MissingArgument;
    }
    var a: Args = .{ .ckpt = pa[1] };
    var i: usize = 2;
    while (i < pa.len) : (i += 1) {
        const s = pa[i];
        if (std.mem.eql(u8, s, "--manifest")) {
            a.manifest = try nextVal(pa, &i, s);
        } else if (std.mem.eql(u8, s, "--arm")) {
            a.arm = std.meta.stringToEnum(Arm, try nextVal(pa, &i, s)) orelse return error.InvalidArgument;
        } else if (std.mem.eql(u8, s, "--reps")) {
            a.reps = try std.fmt.parseInt(usize, try nextVal(pa, &i, s), 10);
        } else if (std.mem.eql(u8, s, "--out")) {
            a.out = try nextVal(pa, &i, s);
        } else if (std.mem.eql(u8, s, "--order")) {
            a.order = std.meta.stringToEnum(Order, try nextVal(pa, &i, s)) orelse return error.InvalidArgument;
        } else if (std.mem.eql(u8, s, "--policy-selftest")) {
            a.policy_selftest = true;
        } else if (std.mem.eql(u8, s, "--allow-cpu")) {
            a.allow_cpu = true;
        } else if (std.mem.eql(u8, s, "--force-vram")) {
            a.force_vram = true;
        } else {
            log.err("argument inconnu: {s}\n{s}", .{ s, usage });
            return error.InvalidArgument;
        }
    }
    if (a.reps < 2 and a.order == .both) {
        log.err("--order both exige --reps >= 2 (témoin rep1/rep2 de C-SD-D)", .{});
        return error.InvalidArgument;
    }
    return a;
}

const Case = struct {
    case_id: []const u8,
    perm: []const u8, // "orig" | "rev" | "-" (json/warmup)
    ids: []u32,
    label_ids: [policy.N]u32,
    classes: [policy.N]policy.Class,
};

const Manifest = struct { eot_id: u32, warmup: Case, cases: []Case };

fn readManifest(allocator: std.mem.Allocator, arena: std.mem.Allocator, io: std.Io, path: []const u8, arm: Arm) !Manifest {
    var mf = std.Io.Dir.cwd().openFile(io, path, .{ .mode = .read_only }) catch |err| {
        log.err("manifest illisible ({s}) : {s}", .{ path, @errorName(err) });
        return error.BadManifest;
    };
    defer mf.close(io);
    const mlen: usize = @intCast(try mf.length(io));
    const mtext = try allocator.alloc(u8, mlen);
    defer allocator.free(mtext);
    if (try mf.readPositionalAll(io, mtext, 0) != mlen) return error.ShortRead;
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, mtext, .{ .allocate = .alloc_always });
    defer parsed.deinit();
    const root = parsed.value.object;

    const eot = (root.get("eot_id") orelse return badField("eot_id")).integer;
    const warm_ids = (((root.get("warmup") orelse return badField("warmup")).object.get("ids")) orelse return badField("warmup.ids")).array;
    const warmup = Case{ .case_id = "warmup", .perm = "-", .ids = try idsFrom(arena, warm_ids), .label_ids = .{ 0, 1, 2, 3 }, .classes = .{ .direct, .search, .calculate, .insufficient } };

    const key = if (arm == .letter) "letter" else "json";
    const arr = (root.get(key) orelse return badField(key)).array;
    const cases = try arena.alloc(Case, arr.items.len);
    for (arr.items, 0..) |v, k| {
        const o = v.object;
        var c = Case{
            .case_id = try arena.dupe(u8, (o.get("case_id") orelse return badField("case_id")).string),
            .perm = if (o.get("perm")) |p| try arena.dupe(u8, p.string) else "-",
            .ids = try idsFrom(arena, (o.get("ids") orelse return badField("ids")).array),
            .label_ids = warmup.label_ids,
            .classes = warmup.classes,
        };
        if (arm == .letter) {
            const li = (o.get("label_ids") orelse return badField("label_ids")).array;
            const cl = (o.get("classes") orelse return badField("classes")).array;
            if (li.items.len != policy.N or cl.items.len != policy.N) return badField("label_ids/classes (taille ≠ 4)");
            for (0..policy.N) |j| {
                c.label_ids[j] = @intCast(li.items[j].integer);
                c.classes[j] = policy.classFromStr(cl.items[j].string) orelse return badField("classes (valeur inconnue)");
            }
        } else if (cases.len > 0) {
            // bras json : cand = label_ids du 1er cas lettre n'existe pas ici → ids factices distincts
            // (cand est ignoré par ce bras ; il doit seulement être dans le vocabulaire).
            c.label_ids = .{ 0, 1, 2, 3 };
        }
        if (c.ids.len == 0) return badField("ids (vide)");
        cases[k] = c;
    }
    // spec §9 : un prompt trop long est REFUSÉ NOMMÉMENT, les autres continuent (compté par 86 :
    // cas absent du JSONL ⇒ INEXÉCUTABLE de complétude — le refus est donc visible, pas silencieux).
    var kept: usize = 0;
    for (cases) |c| {
        if (c.ids.len >= @as(usize, @intCast(SLIDING_WINDOW))) {
            log.err("cas {s}/{s} REFUSÉ : {d} ids ≥ SLIDING_WINDOW ({d})", .{ c.case_id, c.perm, c.ids.len, SLIDING_WINDOW });
            continue;
        }
        cases[kept] = c;
        kept += 1;
    }
    return .{ .eot_id = @intCast(eot), .warmup = warmup, .cases = cases[0..kept] };
}

fn badField(name: []const u8) error{BadManifest} {
    log.err("manifest : champ '{s}' absent ou invalide", .{name});
    return error.BadManifest;
}

fn idsFrom(arena: std.mem.Allocator, arr: std.json.Array) ![]u32 {
    const out = try arena.alloc(u32, arr.items.len);
    for (arr.items, 0..) |v, k| out[k] = @intCast(v.integer);
    return out;
}
```

- [ ] **Step 3: `StepDec`**

```zig
// Nom court OBLIGATOIRE (piège quota comptime @typeName, cf gemma4_gen_auto.zig:739-741).
const StepDec = struct {
    pub fn forward(model: Model, tabs: Tabs, tok: zml.Tensor, cand: zml.Tensor, p: PackedLong, cache: engine.Cache, ctrl: engine.Ctrl) struct { zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor } {
        const e = model.embed_tokens.gather(.{ .voc = tok }, .{});
        const el = tabs.eptl.gather(.{ .voc = tok }, .{});
        const logits, const slk, const slv, const flk, const flv = model.forwardStep(e, el, p, cache, ctrl);
        const t5 = logits.topK(.{ .voc = .voc }, 5, .{});
        const zc = logits.gather(.{ .voc = cand }, .{}); // {b,s,4} (axe d'indices de `cand`)
        const lse = logits.logSumExp(.voc); // {b,s,voc=1}
        return .{ t5.values, t5.indices, zc, lse, slk, slv, flk, flv };
    }
};
```
(Si `logSumExp` manque dans la version de ZML du workspace distant : `const mx = logits.max(.voc); const lse = logits.sub(mx.broad(logits.shape())).exp().sum(.voc).log().add(mx);` et commentaire « composition, spec §4.3 ».)

- [ ] **Step 4: `main` — bannière, selftest, plateforme, chargement, compile**

```zig
pub fn main(init: std.process.Init) !void {
    @setEvalBranchQuota(200000);
    const arena = init.arena;
    const allocator = init.gpa;
    const io = init.io;
    log.info("BUILD: mode={s}", .{@tagName(builtin.mode)}); // C-SD-E (motif gemma4_g12auto.zig:2027)

    const args = try parseArgs(try init.minimal.args.toSlice(arena.allocator()));

    if (args.policy_selftest) { // C-SD-F sur le binaire LIVRÉ (mêmes cas que les tests)
        var buf: [64]u8 = undefined;
        var n_fail: usize = 0;
        for (policy.selftest_cases) |c| {
            const d = policy.evaluate(c.in);
            const got = policy.describe(d, &buf);
            const ok = std.mem.eql(u8, got, c.want) and policy.sumOk(d);
            if (!ok) n_fail += 1;
            log.info("policy-selftest {s} : got={s} want={s} {s}", .{ c.name, got, c.want, if (ok) "OK" else "FAIL" });
        }
        if (n_fail != 0) return error.PolicySelftestFailed;
        log.info("C-SD-F PASS — {d}/{d}", .{ policy.selftest_cases.len, policy.selftest_cases.len });
        return;
    }

    const mpath = args.manifest orelse return badField("--manifest");
    const out_path = args.out orelse return badField("--out");
    const mani = try readManifest(allocator, arena.allocator(), io, mpath, args.arm);
    log.info("manifest : {d} cas ({s}), eot_id={d}", .{ mani.cases.len, @tagName(args.arm), mani.eot_id });

    if (args.force_vram) log.warn("--force-vram : garde VRAM sautée", .{}) else try checkVram(allocator, io);
    // … puis COPIER VERBATIM gemma4_gen_auto.zig:899-989 SAUF l.919-928 (bloc --selftest-gather) :
    //   Platform CUDA + garde CUDA dure + sharding ; chargement Model/Tabs ; packed_sym/cache_sym/
    //   ctrl_sym ; eng_buf/tabs_buf ; HostInputs ; pk_buf ; cache_buf ; store_ck.deinit ; mem_probe.
    //   (Les l.875-898 appartiennent au bloc --oracle de gen_auto : NE PAS les copier.)
    const cand_sym = zml.Tensor.init(.{policy.N}, .u32).withTags(.{.c});
    log.info("Compiling StepDec.forward ...", .{});
    const t_compile: std.Io.Timestamp = .now(io, .awake);
    var exe = try platform.compileFn(allocator, io, StepDec.forward, .{ model, tabs, tok_sym, cand_sym, packed_sym, cache_sym, ctrl_sym }, .{ .shardings = &.{sharding} });
    defer exe.deinit();
    log.info("  compile: {f}", .{t_compile.untilNow(io, .awake)});
    // (Task 6 : boucle des cas)
}
```

Ajouter à `zml_runner/BUILD.bazel` (après la cible `gemma4_gen_auto`) :

```python
# SD (spec docs/superpowers/specs/2026-09-24-decision-layer-design.md) — runner de décision typée.
# Clone ciblé de gemma4_gen_auto (duplication assumée) ; engine.zig et gemma4_gen_auto.zig INTACTS.
zig_binary(
    name = "gemma4_decide",
    main = "gemma4_decide.zig",
    srcs = ["engine.zig", "mem_probe.zig", "sd_policy.zig"],
    visibility = ["//visibility:public"],
    deps = ["//bazel", "//zml"],
)
```

- [ ] **Step 5: Build + selftest sur le binaire**

```bash
ZML_REMOTE=$GPU ZML_DST=$ZML_WS/examples/rqz zml_runner/deploy_to_3090.sh
ZML_REMOTE=$GPU ZML_WS=$ZML_WS TARGETS="//examples/rqz:gemma4_decide" zml_runner/build_3090.sh 2>&1 | tail -5
ssh $GPU "$ZML_WS/bazel-bin/examples/rqz/gemma4_decide x --policy-selftest" 2>&1 | tee docs/evidence/sd/policy_selftest.log
```
Expected: `BUILD: mode=ReleaseFast` puis `C-SD-F PASS — 9/9`.

- [ ] **Step 5b: Test de démarrage — manifest invalide (spec §9)**

```bash
ssh $GPU 'cd /data/gemma4-zml-probe && python3 -c "import json;m=json.load(open(\"fixtures/sd_manifest.json\"));del m[\"letter\"][0][\"label_ids\"];json.dump(m,open(\"/tmp/sd_bad.json\",\"w\"))"'
ssh $GPU "$ZML_WS/bazel-bin/examples/rqz/gemma4_decide x --manifest /tmp/sd_bad.json --arm letter --reps 2 --out /tmp/x.jsonl --allow-cpu" 2>&1 | tee docs/evidence/sd/bad_manifest.log
```
Expected : `manifest : champ 'label_ids' absent ou invalide` et `error: BadManifest`, **avant** tout chargement de poids (le manifest est lu avant la plateforme).

- [ ] **Step 6: Vérifier la non-modification et committer**

Run: `git diff --stat main -- zml_runner/engine.zig zml_runner/gemma4_gen_auto.zig` → Expected: vide.
```bash
git add zml_runner/gemma4_decide.zig zml_runner/BUILD.bazel docs/evidence/sd/policy_selftest.log docs/evidence/sd/bad_manifest.log
git commit -m "sd(task5): runner gemma4_decide — manifest, StepDec, selftest C-SD-F sur le binaire"
```

---

### Task 6: Boucle des cas, chronométrage, JSONL

**Files:**
- Modify: `zml_runner/gemma4_decide.zig` (remplacer `// (Task 6 : boucle des cas)`)

- [ ] **Step 1: Fonctions utilitaires (reset du cache, pas unique)**

```zig
fn uploadZeroCache(io: std.Io, platform: *zml.Platform, sharding: zml.sharding.Sharding, cache_sym: engine.Cache, host: *const HostInputs) !zml.Bufferized(engine.Cache) {
    return .{
        .sl_k = try zml.Buffer.fromBytes(io, platform, cache_sym.sl_k.shape(), sharding, host.cache_sl_k),
        .sl_v = try zml.Buffer.fromBytes(io, platform, cache_sym.sl_v.shape(), sharding, host.cache_sl_v),
        .fl_k = try zml.Buffer.fromBytes(io, platform, cache_sym.fl_k.shape(), sharding, host.cache_fl_k),
        .fl_v = try zml.Buffer.fromBytes(io, platform, cache_sym.fl_v.shape(), sharding, host.cache_fl_v),
    };
}

fn deinitCache(c: *zml.Bufferized(engine.Cache)) void {
    c.sl_k.deinit();
    c.sl_v.deinit();
    c.fl_k.deinit();
    c.fl_v.deinit();
}

const StepOut = struct { t5v: zml.Buffer, t5i: zml.Buffer, zc: zml.Buffer, lse: zml.Buffer };
```

- [ ] **Step 2: `runCase` — un cas complet, frontières de la spec §4.3**

```zig
const Timings = struct { reset_ns: u64 = 0, absorb_ns: u64 = 0, read_b_ns: u64 = 0, read_c_ns: u64 = 0, policy_ns: u64 = 0, gen_ns: u64 = 0 };

const CaseResult = struct {
    step0: u32,
    t: Timings,
    top5_idx: [5]i32 = .{0} ** 5,
    top5_val: [5]f32 = .{0} ** 5,
    zc: [policy.N]f32 = .{0} ** policy.N,
    lse: f32 = 0,
    read_order: []const u8 = "BC",
    decision: ?policy.Decision = null,
    gen: std.ArrayList(u32) = .empty,
    stop: []const u8 = "-",
};

/// Exécute un cas. `b_first` fixe l'ordre des deux lectures (alterné par la parité de la rép.).
/// Bras json : génération gloutonne jusqu'à EOT ou 96 tokens (spec §4.3).
fn runCase(allocator: std.mem.Allocator, io: std.Io, platform: *zml.Platform, sharding: zml.sharding.Sharding, exe: anytype, bufs: anytype, cache_sym: engine.Cache, host: *const HostInputs, c: Case, arm: Arm, eot_id: u32, vocab: i64, b_first: bool) !CaseResult {
    var r = CaseResult{ .step0 = 0, .t = .{}, .read_order = if (b_first) "BC" else "CB" };

    const t_r: std.Io.Timestamp = .now(io, .awake);
    var cache = try uploadZeroCache(io, platform, sharding, cache_sym, host);
    r.t.reset_ns = @intCast(t_r.untilNow(io, .awake).toNanoseconds());

    var cand_host = c.label_ids;
    var cand_buf = try zml.Buffer.fromBytes(io, platform, zml.Shape.init(.{policy.N}, .u32).withTags(.{.c}), sharding, std.mem.sliceAsBytes(&cand_host));
    defer cand_buf.deinit();

    const n = c.ids.len;
    var step: u32 = 0; // repart de 0 à CHAQUE cas (C-SD-D) — journalisé en step0
    r.step0 = step;
    var fed: u32 = c.ids[0];
    const t0: std.Io.Timestamp = .now(io, .awake);
    var t_gen0: std.Io.Timestamp = t0;
    const max_gen: usize = 96;

    while (true) : (step += 1) {
        if (fed >= vocab) return error.TokenOutOfRange;
        if (step + 1 >= @as(u32, @intCast(L_MAX))) {
            r.stop = "l_max";
            break;
        }
        var tok_host = [1]u32{fed};
        var tok_buf = try zml.Buffer.fromBytes(io, platform, zml.Shape.init(.{ 1, 1 }, .u32).withTags(.{ .b, .s }), sharding, std.mem.sliceAsBytes(&tok_host));
        var step_buf = try zml.Buffer.scalar(io, platform, step, .u32, sharding);
        const ctrl_buf = zml.Bufferized(engine.Ctrl){ .step = step_buf };
        var call_args = try exe.args(allocator);
        var call_results = try exe.results(allocator);
        call_args.set(.{ bufs.eng, bufs.tabs, tok_buf, cand_buf, bufs.pk, cache, ctrl_buf });

        const last_absorb = (step + 1 == n);
        if (last_absorb) exe.callOpts(io, call_args, &call_results, .{ .wait = true }) else exe.call(call_args, &call_results);
        var o_t5v, var o_t5i, var o_zc, var o_lse, const slk, const slv, const flk, const flv = call_results.get(struct { zml.Buffer, zml.Buffer, zml.Buffer, zml.Buffer, zml.Buffer, zml.Buffer, zml.Buffer, zml.Buffer });
        if (last_absorb) r.t.absorb_ns = @intCast(t0.untilNow(io, .awake).toNanoseconds());

        var old = cache;
        cache = .{ .sl_k = slk, .sl_v = slv, .fl_k = flk, .fl_v = flv };
        deinitCache(&old);
        tok_buf.deinit();
        step_buf.deinit();
        call_args.deinit(allocator);
        call_results.deinit(allocator);
        defer {
            o_t5v.deinit();
            o_t5i.deinit();
            o_zc.deinit();
            o_lse.deinit();
        }

        if (step + 1 < n) { // absorption : lecture bloquante du top-5 (comme gen_auto), ignoré
            var s = try o_t5i.toSliceAlloc(allocator, io);
            s.free(allocator);
            fed = c.ids[step + 1];
            continue;
        }

        if (last_absorb and arm == .letter) {
            try readB(allocator, io, &r, &o_t5v, &o_t5i, b_first, &o_zc, &o_lse);
            const t_p: std.Io.Timestamp = .now(io, .awake);
            r.decision = policy.evaluate(.{ .zc = r.zc, .lse = r.lse, .label_ids = c.label_ids, .classes = c.classes, .vocab = @intCast(vocab) });
            r.t.policy_ns = @intCast(t_p.untilNow(io, .awake).toNanoseconds());
            break;
        }

        // bras json : top1 = token suivant ; t_gen de la fin de l'absorption à la lecture de l'EOT
        if (last_absorb) t_gen0 = .now(io, .awake);
        var sv = try o_t5i.toSliceAlloc(allocator, io);
        defer sv.free(allocator);
        const nxt: u32 = @intCast(sv.items(i32)[0]);
        try r.gen.append(allocator, nxt);
        if (nxt == eot_id) {
            r.stop = "eot";
            r.t.gen_ns = @intCast(t_gen0.untilNow(io, .awake).toNanoseconds());
            break;
        }
        if (r.gen.items.len >= max_gen) {
            r.stop = "max_tokens";
            r.t.gen_ns = @intCast(t_gen0.untilNow(io, .awake).toNanoseconds());
            break;
        }
        fed = nxt;
    }
    deinitCache(&cache);
    return r;
}

/// Les deux lectures du dernier pas, chronométrées séparément, dans l'ordre demandé.
fn readB(allocator: std.mem.Allocator, io: std.Io, r: *CaseResult, t5v: *zml.Buffer, t5i: *zml.Buffer, b_first: bool, zc: *zml.Buffer, lse: *zml.Buffer) !void {
    for (0..2) |k| {
        const do_b = (k == 0) == b_first;
        const t: std.Io.Timestamp = .now(io, .awake);
        if (do_b) {
            var v = try t5v.toSliceAlloc(allocator, io);
            defer v.free(allocator);
            var ix = try t5i.toSliceAlloc(allocator, io);
            defer ix.free(allocator);
            for (0..5) |j| {
                r.top5_val[j] = v.items(f32)[j];
                r.top5_idx[j] = ix.items(i32)[j];
            }
            r.t.read_b_ns = @intCast(t.untilNow(io, .awake).toNanoseconds());
        } else {
            var z = try zc.toSliceAlloc(allocator, io);
            defer z.free(allocator);
            var l = try lse.toSliceAlloc(allocator, io);
            defer l.free(allocator);
            for (0..policy.N) |j| r.zc[j] = z.items(f32)[j];
            r.lse = l.items(f32)[0];
            r.t.read_c_ns = @intCast(t.untilNow(io, .awake).toNanoseconds());
        }
    }
}
```

Note : le bras json lit aussi `zc/lse` inutilement ? Non — il ne les lit pas (seul `t5i`). Les buffers sont libérés par le `defer`.

- [ ] **Step 3: Sérialisation d'une ligne JSONL**

```zig
fn appendLine(al: std.mem.Allocator, buf: *std.ArrayList(u8), arm: Arm, rep: usize, order: []const u8, c: Case, r: *const CaseResult) !void {
    const w = struct {
        fn f(a: std.mem.Allocator, b: *std.ArrayList(u8), comptime fmt: []const u8, x: anytype) !void {
            const s = try std.fmt.allocPrint(a, fmt, x);
            defer a.free(s);
            try b.appendSlice(a, s);
        }
    }.f;
    try w(al, buf, "{{\"arm\":\"{s}\",\"rep\":{d},\"order\":\"{s}\",\"case_id\":\"{s}\",\"perm\":\"{s}\",\"step0\":{d},\"n_prompt\":{d}", .{ @tagName(arm), rep, order, c.case_id, c.perm, r.step0, c.ids.len });
    try w(al, buf, ",\"t_reset_ns\":{d},\"t_absorb_ns\":{d},\"t_read_b_ns\":{d},\"t_read_c_ns\":{d},\"t_policy_ns\":{d},\"t_gen_ns\":{d},\"read_order\":\"{s}\"", .{ r.t.reset_ns, r.t.absorb_ns, r.t.read_b_ns, r.t.read_c_ns, r.t.policy_ns, r.t.gen_ns, r.read_order });
    // Tableaux écrits À LA MAIN : `{any}` imprime `{ 2, 105 }`, qui n'est pas du JSON.
    try w(al, buf, ",\"top5_idx\":[{d},{d},{d},{d},{d}],\"top5_val\":[{e},{e},{e},{e},{e}]", .{ r.top5_idx[0], r.top5_idx[1], r.top5_idx[2], r.top5_idx[3], r.top5_idx[4], r.top5_val[0], r.top5_val[1], r.top5_val[2], r.top5_val[3], r.top5_val[4] });
    try buf.appendSlice(al, ",\"zc\":{");
    for (0..policy.N) |j| try w(al, buf, "{s}\"{d}\":{e}", .{ if (j == 0) "" else ",", c.label_ids[j], r.zc[j] });
    try buf.appendSlice(al, "},\"zc_bits\":{");
    for (0..policy.N) |j| try w(al, buf, "{s}\"{d}\":\"{x:0>8}\"", .{ if (j == 0) "" else ",", c.label_ids[j], @as(u32, @bitCast(r.zc[j])) });
    try w(al, buf, "}},\"lse\":{e},\"lse_bits\":\"{x:0>8}\"", .{ r.lse, @as(u32, @bitCast(r.lse)) });
    if (r.decision) |d| {
        switch (d) {
            .decision => |x| try w(al, buf, ",\"decision\":{{\"kind\":\"decision\",\"option\":\"{s}\",\"p\":[{e},{e},{e},{e}],\"margin\":{e},\"mass_in\":{e}}}", .{ @tagName(x.option), x.p[0], x.p[1], x.p[2], x.p[3], x.margin, x.mass_in }),
            .abstain => |x| try w(al, buf, ",\"decision\":{{\"kind\":\"abstain\",\"option\":\"{s}\",\"reason\":\"{s}\",\"p\":[{e},{e},{e},{e}],\"margin\":{e},\"mass_in\":{e}}}", .{ @tagName(x.option), @tagName(x.reason), x.p[0], x.p[1], x.p[2], x.p[3], x.margin, x.mass_in }),
            .err => |e| try w(al, buf, ",\"decision\":{{\"kind\":\"err\",\"err\":\"{s}\"}}", .{@tagName(e)}),
        }
    }
    try w(al, buf, ",\"n_gen\":{d},\"stop\":\"{s}\",\"gen_ids\":[", .{ r.gen.items.len, r.stop });
    for (r.gen.items, 0..) |t, j| try w(al, buf, "{s}{d}", .{ if (j == 0) "" else ",", t });
    try buf.appendSlice(al, "]}\n");
}
```

- [ ] **Step 4: Orchestration dans `main`** (remplace `// (Task 6 : boucle des cas)`)

```zig
    const bufs = .{ .eng = eng_buf, .tabs = tabs_buf, .pk = pk_buf };
    const vocab = model.embed_tokens.dim(.voc);
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(allocator);

    // Échauffement : prompt hors liste, froid, journalisé, exclu des statistiques (spec §6).
    {
        var r = try runCase(allocator, io, platform, sharding, &exe, bufs, cache_sym, &host, mani.warmup, .letter, mani.eot_id, vocab, true);
        defer r.gen.deinit(allocator);
        try appendLine(allocator, &out, .letter, 0, "warmup", mani.warmup, &r);
    }
    for (1..args.reps + 1) |rep| { // la répétition parcourt TOUTE la liste (C-SD-D)
        for (mani.cases) |c| {
            var r = try runCase(allocator, io, platform, sharding, &exe, bufs, cache_sym, &host, c, args.arm, mani.eot_id, vocab, rep % 2 == 1);
            defer r.gen.deinit(allocator);
            try appendLine(allocator, &out, args.arm, rep, "fwd", c, &r);
        }
        log.info("rep {d}/{d} terminée ({d} cas)", .{ rep, args.reps, mani.cases.len });
    }
    if (args.order == .both) { // une passe INVERSE, même processus, même compile (C-SD-D)
        var k = mani.cases.len;
        while (k > 0) {
            k -= 1;
            const c = mani.cases[k];
            var r = try runCase(allocator, io, platform, sharding, &exe, bufs, cache_sym, &host, c, args.arm, mani.eot_id, vocab, true);
            defer r.gen.deinit(allocator);
            try appendLine(allocator, &out, args.arm, 1, "rev", c, &r);
        }
    }
    const f = try std.Io.Dir.createFile(.cwd(), io, out_path, .{});
    defer f.close(io);
    try f.writePositionalAll(io, out.items, 0);
    log.info("écrit {s} ({d} octets)", .{ out_path, out.items.len });
```

`HostInputs` est lu par pointeur (`&host`) ; la copie verbatim de la Task 5 déclare `var host` — adapter en conséquence. `cache_buf` de la copie verbatim devient inutile (chaque cas remonte son cache) : le libérer juste après le chargement (`deinitCache(&cache_buf)`).

- [ ] **Step 5: Build, puis run fumée (3 cas, 2 reps)**

Préparer un mini-manifest de fumée sur la 3090 :
```bash
ssh $GPU 'cd /data/gemma4-zml-probe && /data/venvs/gemma4-probe/bin/python3 -c "import json;m=json.load(open(\"fixtures/sd_manifest.json\"));m[\"letter\"]=m[\"letter\"][:3];m[\"json\"]=m[\"json\"][:2];json.dump(m,open(\"/tmp/sd_smoke.json\",\"w\"))"'
ZML_REMOTE=$GPU ZML_DST=$ZML_WS/examples/rqz zml_runner/deploy_to_3090.sh
ZML_REMOTE=$GPU ZML_WS=$ZML_WS TARGETS="//examples/rqz:gemma4_decide" zml_runner/build_3090.sh 2>&1 | tail -3
ssh $GPU "cd /data/gemma4-zml-probe && $ZML_WS/bazel-bin/examples/rqz/gemma4_decide weights/model.safetensors --manifest /tmp/sd_smoke.json --arm letter --reps 2 --order both --out /tmp/sd_smoke_l.jsonl 2>&1 | tail -8; head -c 900 /tmp/sd_smoke_l.jsonl"
ssh $GPU "cd /data/gemma4-zml-probe && $ZML_WS/bazel-bin/examples/rqz/gemma4_decide weights/model.safetensors --manifest /tmp/sd_smoke.json --arm json --reps 2 --out /tmp/sd_smoke_j.jsonl 2>&1 | tail -4; python3 -c \"import json;[print(json.loads(l)['stop'],json.loads(l)['n_gen']) for l in open('/tmp/sd_smoke_j.jsonl')]\""
```
Vérifier d'abord que chaque ligne est du JSON valide : `ssh $GPU "python3 -c \"import json;[json.loads(l) for l in open('/tmp/sd_smoke_l.jsonl')];print('JSONL OK')\""`.
Expected : `JSONL OK`, `BUILD: mode=ReleaseFast`, compile ~17 s, 1 + 3×2 + 3 = 10 lignes lettre avec `step0: 0`, des `zc` finis, une `decision` ; lignes json avec `stop: eot` et `n_gen` ~8-15. Toute erreur de compilation : corriger au plus près (APIs ZML/Zig), jamais en touchant `engine.zig`.

- [ ] **Step 6: Commit**

```bash
git add zml_runner/gemma4_decide.zig
git commit -m "sd(task6): boucle par cas, frontières de chronométrage §4.3, JSONL ; fumée 3 cas OK"
```

---

### Task 7: Runs mesurés + non-régression (C-SD-C, C-SD-E)

**Files:**
- Create: `docs/evidence/sd/r_letter.jsonl`, `r_json.jsonl`, `run_letter.log`, `run_json.log`, `hlo_witness_branch.txt`, `a1_branch.log`

- [ ] **Step 1: Re-vérifier la VRAM (Task 0 Step 1-2)** — nouveau GO si un modèle Ollama a été rechargé entre-temps.

- [ ] **Step 2: Rebuild des deux cibles depuis la branche**

```bash
ZML_REMOTE=$GPU ZML_DST=$ZML_WS/examples/rqz zml_runner/deploy_to_3090.sh
ZML_REMOTE=$GPU ZML_WS=$ZML_WS TARGETS="//examples/rqz:gemma4_decide //examples/rqz:gemma4_gen_auto" zml_runner/build_3090.sh 2>&1 | tee docs/evidence/sd/build_branch.log | tail -3
ssh $GPU "sha256sum $ZML_WS/bazel-bin/examples/rqz/gemma4_gen_auto $ZML_WS/bazel-bin/examples/rqz/gemma4_decide" | tee -a docs/evidence/sd/build_branch.log
```

- [ ] **Step 3: C-SD-C — A1 + empreinte HLO sur le binaire de la branche**

Même commande qu'en Task 1 Step 4 avec `/tmp/sd_hlo_branch`, sortie dans `docs/evidence/sd/hlo_witness_branch.txt`. Puis :
Run (les deux empreintes doivent EXISTER, sinon le contrôle est INEXÉCUTABLE — un 88 en échec écrit sur stderr et laisserait sinon comparer deux lignes `A1 PASS`) :
```bash
hm=$(grep '^sha256 ' docs/evidence/sd/hlo_witness_main.txt | awk '{print $2}'); hb=$(grep '^sha256 ' docs/evidence/sd/hlo_witness_branch.txt | awk '{print $2}')
if [ -z "$hm" ] || [ -z "$hb" ]; then echo "C-SD-C INEXÉCUTABLE (empreinte absente)"; elif [ "$hm" = "$hb" ]; then echo "C-SD-C IDENTIQUE $hm"; else echo "C-SD-C DIFFÉRENT $hm $hb"; fi
```
Expected: `C-SD-C IDENTIQUE <hex>`, et `A1 PASS — 48/48` dans le fichier branche. Rediriger aussi stderr (`2>&1`) de 88 dans la commande de ce step, comme en Task 1 Step 4.

- [ ] **Step 4: Run lettre (5 reps + passe inverse) et run JSON (5 reps)**

```bash
ssh $GPU "cd /data/gemma4-zml-probe && $ZML_WS/bazel-bin/examples/rqz/gemma4_decide weights/model.safetensors --manifest fixtures/sd_manifest.json --arm letter --reps 5 --order both --out /tmp/r_letter.jsonl" 2>&1 | tee docs/evidence/sd/run_letter.log | tail -4
ssh $GPU "cd /data/gemma4-zml-probe && $ZML_WS/bazel-bin/examples/rqz/gemma4_decide weights/model.safetensors --manifest fixtures/sd_manifest.json --arm json --reps 5 --out /tmp/r_json.jsonl" 2>&1 | tee docs/evidence/sd/run_json.log | tail -4
rsync -a $GPU:/tmp/r_letter.jsonl $GPU:/tmp/r_json.jsonl docs/evidence/sd/
wc -l docs/evidence/sd/r_letter.jsonl docs/evidence/sd/r_json.jsonl
```
Expected : `r_letter.jsonl` = 1 + 48×5 + 48 = **289** lignes ; `r_json.jsonl` = 1 + 24×5 = **121** lignes ; les deux logs portent `BUILD: mode=ReleaseFast`.

- [ ] **Step 5: Restaurer la 3090** — rien à faire côté Ollama (le modèle se recharge au prochain appel) ; consigner l'heure de fin dans `docs/evidence/sd/RUN_LOG.md`.

- [ ] **Step 6: Commit**

```bash
git add docs/evidence/sd/
git commit -m "sd(task7): runs mesurés lettre (289 l.) + json (121 l.), C-SD-C (A1 + HLO) sur la branche"
```

---

### Task 8: Dépouilleur `86_sd_report.py`

**Files:**
- Create: `scripts/86_sd_report.py`
- Create (produit): `docs/SD_RESULTS.md`

- [ ] **Step 1: Écrire le dépouilleur** (stdlib seule ; tokenizer HF seulement pour décoder le JSON du bras A — importé paresseusement, et si indisponible sur M1, décoder via `--gen-text` fourni par un appel sur la 3090, cf. Step 2)

```python
#!/usr/bin/env python3
"""86 — Dépouilleur SD (spec §4.6, §6, §7, §8). Stdlib seule.

Entrées : --manifest, --oracle, --letter r_letter.jsonl, --json r_json.jsonl, --gen-text gen_text.json
(ids→texte du bras A, produit sur la 3090 par `--decode-only`), --logs run_letter.log run_json.log.
Sortie : docs/SD_RESULTS.md + code retour : 0 = tous contrôles PASS ; 1 = au moins un FAIL ;
3 = INEXÉCUTABLE (preuve manquante, témoin rouge…). Jamais PASS par défaut.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

TOL = 0.05


def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


class Verdicts:
    def __init__(self):
        self.rows = []

    def add(self, cid, status, detail):
        assert status in ("PASS", "FAIL", "INEXÉCUTABLE", "DESCRIPTIF")
        self.rows.append((cid, status, detail))

    def code(self):
        s = {r[1] for r in self.rows}
        return 3 if "INEXÉCUTABLE" in s else (1 if "FAIL" in s else 0)


def softmax_option(zc_by_id, label_ids, classes):
    z = [zc_by_id[str(i)] for i in label_ids]
    best = max(range(4), key=lambda k: (z[k], -k))  # égalité → plus petite lettre (comme sd_policy)
    return classes[best]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="fixtures/sd_manifest.json")
    ap.add_argument("--oracle", default="fixtures/sd_oracle.json")
    ap.add_argument("--letter", default="docs/evidence/sd/r_letter.jsonl")
    ap.add_argument("--json", default="docs/evidence/sd/r_json.jsonl")
    ap.add_argument("--gen-text", default="docs/evidence/sd/gen_text.json")
    ap.add_argument("--logs", nargs="+", default=["docs/evidence/sd/run_letter.log", "docs/evidence/sd/run_json.log"])
    ap.add_argument("--out", default="docs/SD_RESULTS.md")
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args(argv)

    v = Verdicts()
    for p in [a.manifest, a.oracle, a.letter, a.json, a.gen_text, *a.logs]:
        if not Path(p).is_file():
            print(f"INEXÉCUTABLE : preuve absente {p}")
            return 3
    man = json.load(open(a.manifest))
    ora = json.load(open(a.oracle))["cases"]
    L = load_jsonl(a.letter)
    J = load_jsonl(a.json)
    gen_text = json.load(open(a.gen_text))
    mcase = {f"{e['case_id']}/{e['perm']}": e for e in man["letter"]}

    # --- C-SD-E : bannière dans chaque log chronométré
    bad = [p for p in a.logs if "BUILD: mode=ReleaseFast" not in open(p, errors="replace").read()]
    v.add("C-SD-E", "INEXÉCUTABLE" if bad else "PASS", f"logs sans bannière : {bad}" if bad else "bannière présente dans tous les logs chronométrés")

    # --- complétude
    fwd = [r for r in L if r["order"] == "fwd" and r["rep"] >= 1]
    rev = [r for r in L if r["order"] == "rev"]
    keys = set(mcase)
    for rep in range(1, a.reps + 1):
        got = {f"{r['case_id']}/{r['perm']}" for r in fwd if r["rep"] == rep}
        if got != keys:
            print(f"INEXÉCUTABLE : rep {rep} incomplète ({len(got)}/{len(keys)})")
            return 3
    if {f"{r['case_id']}/{r['perm']}" for r in rev} != keys:
        print("INEXÉCUTABLE : passe inverse incomplète")
        return 3
    jkeys = {e["case_id"] for e in man["json"]}
    for rep in range(1, a.reps + 1):
        if {r["case_id"] for r in J if r["rep"] == rep} != jkeys:
            print(f"INEXÉCUTABLE : bras JSON rep {rep} incomplète")
            return 3

    # --- C-SD-B : écarts ZML − HF (rep 1)
    max_dz, max_dl, tight = 0.0, 0.0, []
    for r in fwd:
        if r["rep"] != 1:
            continue
        k = f"{r['case_id']}/{r['perm']}"
        o = ora[k]
        for i, zh in o["zc"].items():
            max_dz = max(max_dz, abs(r["zc"][i] - zh))
        max_dl = max(max_dl, abs(r["lse"] - o["lse"]))
        zs = sorted(o["zc"].values(), reverse=True)
        if zs[0] - zs[1] < 0.1:
            tight.append(k)
    ok_b = max_dz <= TOL and max_dl <= TOL
    v.add("C-SD-B", "PASS" if ok_b else "FAIL", f"max|Δzc|={max_dz:.3e}, max|Δlse|={max_dl:.3e} (seuil {TOL}) ; marge HF < 0,1 : {len(tight)} cas {tight[:6]}")

    # --- C-SD-D : step0, témoin rep1/rep2, contrôle rep1/rev (bits)
    if any(r["step0"] != 0 for r in L + J):
        v.add("C-SD-D", "FAIL", "step0 ≠ 0 sur au moins une ligne (remise à 0 de step absente)")
    else:
        by = defaultdict(dict)
        for r in L:
            if r["order"] == "fwd" and r["rep"] in (1, 2):
                by[r["rep"]][f"{r['case_id']}/{r['perm']}"] = (r["zc_bits"], r["lse_bits"])
            elif r["order"] == "rev":
                by["rev"][f"{r['case_id']}/{r['perm']}"] = (r["zc_bits"], r["lse_bits"])
        wit = [k for k in keys if by[1][k] != by[2][k]]
        ctl = [k for k in keys if by[1][k] != by["rev"][k]]
        if wit:
            v.add("C-SD-D", "INEXÉCUTABLE", f"témoin rouge : {len(wit)} cas diffèrent entre rep1 et rep2 (non-déterminisme GPU) {wit[:4]}")
        else:
            v.add("C-SD-D", "FAIL" if ctl else "PASS", f"rep1 vs passe inverse : {len(ctl)} cas diffèrent {ctl[:4]}" if ctl else "bits identiques rep1/rep2 et rep1/inverse sur les 48 prompts")

    # --- C-SD-F (partie dépouillement) : option runner == option recalculée
    mism = []
    for r in fwd + rev:
        m = mcase[f"{r['case_id']}/{r['perm']}"]
        want = softmax_option(r["zc"], m["label_ids"], m["classes"])
        d = r.get("decision") or {}
        if d.get("kind") == "err" or d.get("option") != want:
            mism.append(f"{r['case_id']}/{r['perm']}/rep{r['rep']}:{d.get('option')}≠{want}")
    v.add("C-SD-F(86)", "FAIL" if mism else "PASS", f"{len(mism)} désaccords {mism[:4]}" if mism else "option runner == option recalculée sur toutes les lignes")

    # --- P1
    med_c = st.median([(r["t_read_c_ns"] + r["t_policy_ns"]) for r in fwd]) / 1e6
    med_b = st.median([r["t_read_b_ns"] for r in fwd]) / 1e6
    p1 = abs(med_c - med_b) < 1.0
    v.add("P1", "PASS" if p1 else "FAIL", f"médiane(t_read_C+t_policy)={med_c:.3f} ms, médiane(t_read_B)={med_b:.3f} ms, écart {abs(med_c-med_b):.3f} ms (seuil 1 ms)")

    # --- P2 (population : cas dont le bras A finit par EOT)
    C_ABS, C_GEN = 14.0, 9.0
    tC = defaultdict(list)
    lenL = defaultdict(list)
    for r in fwd:
        tC[r["case_id"]].append((r["t_absorb_ns"] + r["t_read_c_ns"] + r["t_policy_ns"]) / 1e6)
        lenL[r["case_id"]].append(r["n_prompt"])
    tA, nA, lenJ, stops = defaultdict(list), {}, {}, defaultdict(set)
    for r in J:
        if r["rep"] < 1:
            continue
        tA[r["case_id"]].append((r["t_absorb_ns"] + r["t_gen_ns"]) / 1e6)
        nA[r["case_id"]] = r["n_gen"]
        lenJ[r["case_id"]] = r["n_prompt"]
        stops[r["case_id"]].add(r["stop"])
    ngen_all = defaultdict(set)
    for r in J:
        if r["rep"] >= 1:
            ngen_all[r["case_id"]].add(r["n_gen"])
    unstable = [c for c, st_ in ngen_all.items() if len(st_) != 1]
    if unstable:
        print(f"INEXÉCUTABLE : n_gen du bras A varie entre répétitions pour {unstable[:4]}")
        return 3
    pop = sorted(c for c in jkeys if stops[c] == {"eot"})
    G, F, part = {}, {}, {}
    for c in pop:
        G[c] = st.median(tA[c]) - st.median(tC[c])
        F[c] = (lenJ[c] - st.mean(lenL[c])) * C_ABS + (nA[c] - 1) * C_GEN
        part[c] = G[c] / st.median(tA[c])
    if not pop:
        v.add("P2", "INEXÉCUTABLE", "aucun cas du bras A terminé par EOT")
    else:
        a_ok = all(G[c] > 0 for c in pop)
        dev = st.median([abs(G[c] - F[c]) for c in pop])
        b_ok = dev <= max(0.3 * st.median(F.values()), 20.0)
        c_ok = st.median(part.values()) < 0.25
        v.add("P2", "PASS" if (a_ok and b_ok and c_ok) else "FAIL",
              f"population {len(pop)}/{len(jkeys)} ; (a) G>0 partout : {a_ok} ; (b) médiane|G−F|={dev:.1f} ms vs tol {max(0.3*st.median(F.values()),20.0):.1f} : {b_ok} ; (c) médiane part={100*st.median(part.values()):.1f} % : {c_ok}")

    # --- Qualité (descriptif)
    acc = defaultdict(lambda: [0, 0])
    for r in fwd:
        m = mcase[f"{r['case_id']}/{r['perm']}"]
        opt = (r.get("decision") or {}).get("option")
        acc[("C", m["gold"])][0] += opt == m["gold"]
        acc[("C", m["gold"])][1] += 1
        top1 = r["top5_idx"][0]
        lab = m["classes"][m["label_ids"].index(top1)] if top1 in m["label_ids"] else "off_label"
        acc[("B", m["gold"])][0] += lab == m["gold"]
        acc[("B", m["gold"])][1] += 1
    fmt_err = 0
    for r in J:
        if r["rep"] != 1:
            continue
        gold = next(e["gold"] for e in man["json"] if e["case_id"] == r["case_id"])
        txt = gen_text.get(r["case_id"], "")
        try:
            choice = json.loads(txt.strip().strip("`").removeprefix("json").strip())["choice"]
        except Exception:
            choice, fmt_err = None, fmt_err + 1
        acc[("A", gold)][0] += choice == gold
        acc[("A", gold)][1] += 1
    flips = 0
    by_case = defaultdict(dict)
    for r in fwd:
        if r["rep"] == 1:
            by_case[r["case_id"]][r["perm"]] = (r.get("decision") or {}).get("option")
    flips = sum(1 for c in by_case.values() if c.get("orig") != c.get("rev"))
    mass = sorted((r.get("decision") or {}).get("mass_in", float("nan")) for r in fwd if r["rep"] == 1)
    v.add("Qualité", "DESCRIPTIF", f"bascules orig↔rev (bras C) : {flips}/24 ; format_error bras A : {fmt_err}/24 ; mass_in min/méd/max = {mass[0]:.3g}/{st.median(mass):.3g}/{mass[-1]:.3g}")

    # --- Descriptifs exigés par la spec (§4.4, §6, §8, §9) — aucun n'influe sur un verdict
    def p95(xs):
        xs = sorted(xs)
        return xs[min(len(xs) - 1, math.ceil(0.95 * len(xs)) - 1)]
    tB = [(r["t_absorb_ns"] + r["t_read_b_ns"]) / 1e6 for r in fwd]
    tCl = [(r["t_absorb_ns"] + r["t_read_c_ns"] + r["t_policy_ns"]) / 1e6 for r in fwd]
    tAl = [(r["t_absorb_ns"] + r["t_gen_ns"]) / 1e6 for r in J if r["rep"] >= 1]
    desc = [f"bras A : médiane {st.median(tAl):.1f} ms, p95 {p95(tAl):.1f} ms",
            f"bras B : médiane {st.median(tB):.1f} ms, p95 {p95(tB):.1f} ms",
            f"bras C : médiane {st.median(tCl):.1f} ms, p95 {p95(tCl):.1f} ms"]
    c_abs_m = st.median([r["t_absorb_ns"] / 1e6 / r["n_prompt"] for r in fwd])
    c_gen_m = st.median([r["t_gen_ns"] / 1e6 / (r["n_gen"] - 1) for r in J if r["rep"] >= 1 and r["n_gen"] > 1])
    desc.append(f"coûts MESURÉS : c_abs = {c_abs_m:.2f} ms/pas (publié 14), c_gen = {c_gen_m:.2f} ms/pas (publié 9)")
    desc.append(f"t_reset médian {st.median([r['t_reset_ns'] for r in fwd])/1e6:.2f} ms (hors temps de décision)")
    warm = [r for r in L if r["order"] == "warmup"]
    if warm:
        desc.append(f"échauffement (froid) : absorption {warm[0]['t_absorb_ns']/1e6:.1f} ms sur {warm[0]['n_prompt']} pas")
    for lg in a.logs:
        for line in open(lg, errors="replace"):
            if "compile:" in line:
                desc.append(f"compile ({Path(lg).name}) : {line.split('compile:')[1].strip()}")
    off = sum(1 for r in fwd if r["top5_idx"][0] not in mcase[f"{r['case_id']}/{r['perm']}"]["label_ids"])
    desc.append(f"bras B hors étiquettes : {off}/{len(fwd)}")
    theta_rows = []
    for th in (0.0, 0.1, 0.2, 0.4, 0.6, 0.8):
        kept = [r for r in fwd if (r.get("decision") or {}).get("margin", 0) >= th]
        err = sum(1 for r in kept if (r["decision"]["option"] != mcase[f"{r['case_id']}/{r['perm']}"]["gold"]))
        theta_rows.append(f"| {th} | {len(kept)}/{len(fwd)} | {err}/{len(kept) if kept else 0} |")
    thinking = man["meta"].get("enable_thinking_passed")
    desc.append(f"rendu du template : enable_thinking {'passé à False' if thinking else 'ABSENT du template (rendu sans le paramètre)'}")

    # --- rapport
    lines = ["# SD — Résultats (généré par scripts/86_sd_report.py, ne pas éditer à la main)", "",
             "| Contrôle | Verdict | Détail |", "|---|---|---|"]
    lines += [f"| {c} | **{s}** | {d} |" for c, s, d in v.rows]
    lines += ["", "## Exactitude par bras et par classe", "", "| Bras | Classe | Justes / total |", "|---|---|---|"]
    for (arm, cl), (ok, n) in sorted(acc.items()):
        lines.append(f"| {arm} | {cl} | {ok}/{n} |")
    lines += ["", "## Descriptifs", ""] + [f"- {d}" for d in desc]
    lines += ["", "## Couverture / erreur du bras C en fonction de θ (descriptif, spec §4.4)", "",
              "| θ | décisions gardées | erreurs parmi gardées |", "|---|---|---|"] + theta_rows
    lines += ["", "## Gain par cas (bras A − bras C, ms)", "", "| Cas | len JSON | len lettre (moy.) | n_gen A | G mesuré | F prédit | part de t_A |", "|---|---|---|---|---|---|---|"]
    for c in pop:
        lines.append(f"| {c} | {lenJ[c]} | {st.mean(lenL[c]):.1f} | {nA[c]} | {G[c]:.1f} | {F[c]:.1f} | {100*part[c]:.1f} % |")
    Path(a.out).write_text("\n".join(lines) + "\n")
    for c, s, d in v.rows:
        print(f"{c:12s} {s:13s} {d}")
    return v.code()


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Produire `gen_text.json` sur la 3090** (décodage HF des `gen_ids` du bras A, rep 1, EOT exclu)

```bash
ssh $GPU 'cd /data/gemma4-zml-probe && /data/venvs/gemma4-probe/bin/python3 - <<EOF
import json, os
os.environ.setdefault("HF_HOME","/data/hf_cache"); os.environ.setdefault("HF_HUB_OFFLINE","1")
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("google/gemma-4-E2B-it")
eot = json.load(open("fixtures/sd_manifest.json"))["eot_id"]
out = {}
for l in open("/tmp/r_json.jsonl"):
    r = json.loads(l)
    if r["rep"] == 1:
        out[r["case_id"]] = tok.decode([i for i in r["gen_ids"] if i != eot], skip_special_tokens=True)
json.dump(out, open("/tmp/gen_text.json","w"), indent=1)
print(len(out), "textes")
EOF'
rsync -a $GPU:/tmp/gen_text.json docs/evidence/sd/
```
Expected: `24 textes`.

- [ ] **Step 3: Lancer le dépouilleur**

Run: `python3 scripts/86_sd_report.py; echo "exit=$?"`
Expected : une ligne par contrôle ; `docs/SD_RESULTS.md` écrit. Le code de sortie EST le verdict (0/1/3) : **ne rien « corriger » pour faire passer** — un FAIL de P1/P2 est une réfutation à publier, pas un bug (sauf erreur démontrée du dépouilleur, auquel cas corriger le dépouilleur et le dire).

- [ ] **Step 4: Commit**

```bash
git add scripts/86_sd_report.py docs/evidence/sd/gen_text.json docs/SD_RESULTS.md
git commit -m "sd(task8): dépouilleur 86 + SD_RESULTS.md (verdicts tels que mesurés)"
```

---

### Task 9: Contre-épreuve `87_sd_selfproof.py`

**Files:**
- Create: `scripts/87_sd_selfproof.py`
- Create (produit): `docs/evidence/sd/87_selfproof.log`

- [ ] **Step 1: Écrire la contre-épreuve**

```python
#!/usr/bin/env python3
"""87 — Contre-épreuve du dépouilleur 86 (spec §4.6). Pour chaque mutant, copie les preuves dans un
dossier temporaire, applique la mutation, VÉRIFIE que le fichier muté diffère du nominal (sinon
INEXÉCUTABLE — leçon 81_selfproof_80.py), relance 86, et exige le verdict attendu du contrôle visé.
"""
from __future__ import annotations

import importlib.util
import io
import json
import shutil
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("r86", HERE / "86_sd_report.py")
r86 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r86)
EV = Path("docs/evidence/sd")


def rows(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def write(path, rs):
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rs))


def run86(d):
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = r86.main(["--letter", str(d / "r_letter.jsonl"), "--json", str(d / "r_json.jsonl"),
                         "--gen-text", str(EV / "gen_text.json"), "--out", str(d / "out.md"),
                         "--logs", str(EV / "run_letter.log"), str(EV / "run_json.log")])
    return code, buf.getvalue()


def verdict(out, cid):
    for l in out.splitlines():
        if l.startswith(cid + " ") or l.startswith(cid.ljust(12)):
            return l.split()[1]
    return None


def mutant_a(rs, ora):   # zc permutés d'un cran sur un cas où le décalage mord (> 2×TOL)
    for r in rs:
        if r["order"] == "fwd" and r["rep"] == 1:
            ids = list(r["zc"].keys())
            zh = [ora[f"{r['case_id']}/{r['perm']}"]["zc"][i] for i in ids]
            if max(abs(zh[(k + 1) % 4] - zh[k]) for k in range(4)) > 0.1:
                vals = [r["zc"][i] for i in ids]
                r["zc"] = {ids[k]: vals[(k + 1) % 4] for k in range(4)}
                return True
    return False


def mutant_b(rs, _):     # passe inverse altérée d'1 ulp sur un cas
    for r in rs:
        if r["order"] == "rev":
            k = next(iter(r["zc_bits"]))
            r["zc_bits"][k] = f"{(int(r['zc_bits'][k], 16) ^ 1):08x}"
            return True
    return False


def mutant_c(rs, _):     # option orig appliquée à un cas rev
    man = json.load(open("fixtures/sd_manifest.json"))
    orig = {e["case_id"]: e for e in man["letter"] if e["perm"] == "orig"}
    for r in rs:
        if r["perm"] == "rev" and r.get("decision", {}).get("kind") == "decision":
            o = orig[r["case_id"]]
            want_orig = r86.softmax_option(r["zc"], o["label_ids"], o["classes"])
            if want_orig != r["decision"]["option"]:
                r["decision"]["option"] = want_orig
                return True
    return False


def mutant_d(rs, _):     # un cas manquant
    for k, r in enumerate(rs):
        if r["order"] == "fwd" and r["rep"] == 3:
            del rs[k]
            return True
    return False


def mutant_e(rs, _):     # step0 ≠ 0
    rs[5]["step0"] = 17
    return True


MUTANTS = [("a", mutant_a, "C-SD-B", {"FAIL"}), ("b", mutant_b, "C-SD-D", {"FAIL"}),
           ("c", mutant_c, "C-SD-F(86)", {"FAIL"}), ("d", mutant_d, None, {3}),
           ("e", mutant_e, "C-SD-D", {"FAIL"})]


def main() -> int:
    ora = json.load(open("fixtures/sd_oracle.json"))["cases"]
    nominal = "".join(json.dumps(r) + "\n" for r in rows(EV / "r_letter.jsonl"))  # même sérialisation que write()
    bad = 0
    for name, fn, cid, want in MUTANTS:
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            shutil.copy(EV / "r_json.jsonl", d / "r_json.jsonl")
            rs = rows(EV / "r_letter.jsonl")
            applied = fn(rs, ora)
            write(d / "r_letter.jsonl", rs)
            if not applied or (d / "r_letter.jsonl").read_text() == nominal:
                print(f"mutant {name} : INEXÉCUTABLE (mutation vide)")
                bad += 1
                continue
            code, out = run86(d)
            got = code if cid is None else verdict(out, cid)
            ok = got in want
            print(f"mutant {name} : {cid or 'exit'} = {got} (attendu {want}) → {'CONDAMNÉ' if ok else 'NON CONDAMNÉ'}")
            bad += not ok
    print("87 PASS — le dépouilleur condamne les 5 mutants" if bad == 0 else f"87 FAIL — {bad} mutant(s) non condamné(s) ou inexécutable(s)")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Lancer**

Run: `python3 scripts/87_sd_selfproof.py | tee docs/evidence/sd/87_selfproof.log; echo "exit=${PIPESTATUS[0]}"`
Expected: 5 lignes `CONDAMNÉ` et `87 PASS`. Un mutant non condamné = défaut du dépouilleur : corriger 86 (Task 8), relancer 86 puis 87.

- [ ] **Step 3: Commit**

```bash
git add scripts/87_sd_selfproof.py docs/evidence/sd/87_selfproof.log
git commit -m "sd(task9): contre-épreuve 87 — les 5 mutants condamnés"
```

---

### Task 10: Synthèse et planning

**Files:**
- Modify: `docs/SD_RESULTS.md` (en-tête de lecture ajouté AU-DESSUS du bloc généré), `PLANNING.md`

- [ ] **Step 1: Écrire la lecture** en tête de `docs/SD_RESULTS.md` (5-10 lignes, claim-safe : « on observe ») : verdict de chaque prédiction P1/P2 tel que mesuré (une réfutation s'écrit « réfutée » avec le chiffre), exactitude par bras, `mass_in`, bascules de permutation, et la phrase de la spec §8 « lecture attendue » confirmée ou non. Pas de requalification de seuil.

- [ ] **Step 2: Entrée PLANNING** — sous les chantiers, un bloc « 🔬 Chantier SD — couche de décision typée (24 sept 2026, branche `sd-decision-layer`) » : spec, plan, verdicts en une ligne, et les suites hors périmètre de la spec §10 (Score/Noul, lots, projection réduite, prefill S>1, E2B décideur devant Qwen 27B) en « non planifié ».

- [ ] **Step 3: Anonymisation puis commit**

Run : `n=$(git diff main...HEAD | grep -cEf ~/.config/anonymisation/motifs.txt); echo $n` → Expected: `0`. Les motifs (IP du réseau local, utilisateur SSH, chemins personnels) vivent HORS du dépôt dans `~/.config/anonymisation/motifs.txt` ; s'il n'existe pas, le créer avant ce step — jamais les motifs dans un fichier versionné.
```bash
git add docs/SD_RESULTS.md PLANNING.md
git commit -m "sd(task10): synthèse SD_RESULTS + entrée PLANNING"
```
Ne PAS pousser : le push et la PR attendent le GO de Régis.

---

## Auto-revue du plan (faite à l'écriture)

- Couverture spec : §4.1 → T2 ; §4.2/C-SD-A → T3 ; §4.3 → T5-T6 ; §4.4-4.5/C-SD-F → T4 (+ selftest T5, recalcul T8) ; §4.6 → T8-T9 ; §5 → T0-T7 ; §6 → T6-T8 ; C-SD-B/B′ → T8/T9(a) ; C-SD-C → T1+T7 ; C-SD-D → T6 (`step0`, ordre both) + T8 + T9(b,e) ; C-SD-E → T5 (bannière) + T8 ; P1/P2/P3 → T8 ; §11 fichiers → carte ; §12 → cohérent.
- Écarts assumés et à inscrire dans la spec avant codage : empreinte sha256 normalisée (T1 Step 5) ; repli si `zig_test` indisponible (T4 Step 2).
- Types cohérents : `policy.Class`/`Decision`/`Input` (T4) utilisés tels quels en T5-T6 ; champs JSONL écrits en T6 = champs lus en T8/T9 (`step0`, `zc`, `zc_bits`, `lse_bits`, `t_*_ns`, `decision.option`, `n_gen`, `stop`, `gen_ids`, `n_prompt`, `read_order`).
