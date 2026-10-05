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
    scan = scan_cache_dir()
    revs = [r.commit_hash for repo in scan.repos if repo.repo_id == o49.MODEL_ID for r in repo.revisions]
    rev_source = "scan_cache_dir"
    if not revs:
        # scan_cache_dir écarte en silence un dépôt qu'il juge corrompu (ex. blob de poids absent) :
        # on lit alors les snapshots du dossier cache, et on garde l'avertissement du scan.
        from huggingface_hub import constants as hf_constants
        repo_dir = Path(hf_constants.HF_HUB_CACHE) / ("models--" + o49.MODEL_ID.replace("/", "--"))
        snaps = repo_dir / "snapshots"
        revs = sorted(p.name for p in snaps.iterdir()) if snaps.is_dir() else []
        rev_source = "snapshots_dir (scan_cache_dir: " + "; ".join(
            str(w) for w in scan.warnings if o49.MODEL_ID.replace("/", "--") in str(w))[:300] + ")"
    print(f"révisions du modèle en cache : {revs} (source : {rev_source})")
    meta = {
        "tokenizer_json_md5": hashlib.md5(tok_json.read_bytes()).hexdigest(),
        "model_revisions_in_cache": revs,
        "model_revisions_source": rev_source,
        "tokenizer_json_path_in_cache": str(tok_json.relative_to(tok_json.parents[3])),
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
