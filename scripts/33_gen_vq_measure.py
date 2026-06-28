#!/usr/bin/env python3
"""Q5 Volet A — COÛT de la quantification V-only (MSE 4 bits) sur la GÉNÉRATION.

Adapté de scripts/test_kv_quant_generation.py, réduit au seul mode `vonly_mse4`
(V quantifié 4 bits via TurboQuantMSE Hadamard, K fp32) vs `baseline` (KV non quantifié).

Fake-quant V IN-PLACE au point canonique du KV-cache (post-v_norm, AVANT transpose/cache) :
  forward hook sur les `v_norm` des producers -> remplace V par V_hat = MSE.dequant(MSE.quant(V)).
Reproduit exactement le point d'insertion du runner ZML (gemma4_decode_vq.zig / gemma4_gen_vq.zig).
K reste fp32. Constantes MSE déterministes == turboquant_constants.safetensors (Task 0).

Métriques de COÛT (vs baseline, greedy, contexte identique avant divergence) :
  - divergence_point : 1er token généré où l'argmax diffère de baseline (N_NEW = jamais)
  - KL(base||vq) / top5-overlap / argmax-match-rate sur les positions AVANT divergence
  PAS de ROUGE-1 (piège verbatim). PAS de MSE-attention.

Et SORTIE de PORTAGE (référence pour le volet B ZML) :
  - la séquence générée HF-V-quant complète par prompt (vq_gen) : les tokens que le moteur
    ZML-V-quant doit reproduire en greedy.

≥ 8 prompts variés, N_NEW = 48 tokens, greedy.
Output : print + JSON /data/gemma4-zml-probe/gen_vq_measure.json. Rien sur /.
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("HF_HOME", "/data/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn.functional as F

sys.path.insert(0, "/data/gemma4-zml-probe")
from turboquant import TurboQuantMSE  # noqa: E402

REPO = "google/gemma-4-E2B-it"
OUT_DIR = Path("/data/gemma4-zml-probe")
OUT_JSON = OUT_DIR / "gen_vq_measure.json"

N_NEW = 48          # tokens generes par prompt
HEAD_DIMS = [256, 512]

# >= 8 prompts varies (factuel, narratif, code, listes, raisonnement, multilingue, math, instruction)
PROMPTS = [
    "Explain in one paragraph why the sky appears blue during the day.",
    "It is a truth universally acknowledged, that a single man in possession of a good fortune,",
    "Write a short Python function that returns the n-th Fibonacci number.",
    "What are three practical tips for improving focus while working?",
    "Summarize the causes of the French Revolution in a few sentences.",
    "Translate the following sentence into French: The weather is nice today and I want to walk.",
    "If a train travels 60 km in 45 minutes, what is its average speed in km/h? Show your reasoning.",
    "List five common species of birds found in European gardens, with one fact each.",
    "Describe how photosynthesis converts sunlight into chemical energy.",
    "Give step-by-step instructions to make a simple cup of black tea.",
]


def build_quantizers(device):
    """MSE n'a pas besoin de calibration donnees. Un quantizer 4 bits par head_dim."""
    q = {}
    for d in HEAD_DIMS:
        q[("mse4", d)] = TurboQuantMSE(d, 4, device=device, rotation="hadamard")
    return q


def fake_quant(x, quantizer):
    """quantize -> dequantize en fp32, meme shape. x: [..., d]."""
    dt = x.dtype
    xf = x.to(torch.float32)
    comp = quantizer.quant(xf)
    xhat = quantizer.dequant(comp)
    return xhat.to(dt)


class QState:
    def __init__(self, quantizers, n_kv, n_q):
        self.q = quantizers
        self.n_kv = n_kv
        self.n_q = n_q
        self.quant_v = False  # baseline -> False ; vonly_mse4 -> True

    def set_mode(self, mode):
        self.quant_v = (mode == "vonly_mse4")

    def quantizer_for(self, d):
        return self.q[("mse4", d)]


STATE: QState | None = None


def install_v_hooks(model):
    """Forward hooks sur v_norm des producers : remplace l'output par V_hat (V-only fake-quant)."""
    import re
    handles = []
    for name, mod in model.named_modules():
        if not name.endswith("self_attn"):
            continue
        vnorm = getattr(mod, "v_norm", None)
        if vnorm is None:
            continue
        if not re.search(r"layers\.(\d+)\.self_attn$", name):
            continue

        def hook(module, inp, out):
            if STATE is not None and STATE.quant_v:
                t = out[0] if isinstance(out, tuple) else out
                if torch.is_tensor(t) and t.dim() == 4 and t.shape[-2] == STATE.n_kv:
                    d = t.shape[-1]
                    return fake_quant(t, STATE.quantizer_for(d))
            return None  # ne change rien

        handles.append(vnorm.register_forward_hook(hook))
    return handles


@torch.no_grad()
def generate_with_scores(model, input_ids, pad_id):
    out = model.generate(
        input_ids,
        max_new_tokens=N_NEW,
        do_sample=False,
        num_beams=1,
        output_scores=True,
        return_dict_in_generate=True,
        use_cache=True,
        pad_token_id=pad_id,
    )
    seq = out.sequences[0]
    gen = seq[input_ids.shape[1]:]            # tokens generes
    scores = [s[0].float().cpu() for s in out.scores]  # liste [vocab] par step
    return gen.cpu().tolist(), scores


def compare_to_baseline(base_gen, base_scores, mode_gen, mode_scores):
    """Compare jusqu'a la 1ere divergence (contexte identique avant)."""
    n = min(len(base_gen), len(mode_gen))
    div_point = n  # pas de divergence par defaut
    for k in range(n):
        if base_gen[k] != mode_gen[k]:
            div_point = k
            break

    compare_upto = min(div_point + 1, n)
    kls, top5s, argmatch = [], [], []
    for k in range(compare_upto):
        pb = F.log_softmax(base_scores[k], dim=-1)
        pm = F.log_softmax(mode_scores[k], dim=-1)
        kl = float((pb.exp() * (pb - pm)).sum())
        kls.append(kl)
        tb = set(base_scores[k].topk(5).indices.tolist())
        tm = set(mode_scores[k].topk(5).indices.tolist())
        top5s.append(len(tb & tm) / 5.0)
        argmatch.append(int(base_gen[k] == mode_gen[k]))

    return {
        "divergence_point": div_point,         # N_NEW = jamais diverge
        "diverged": div_point < n,
        "n_compared": compare_upto,
        "kl_mean": float(sum(kls) / len(kls)) if kls else None,
        "kl_max": float(max(kls)) if kls else None,
        "kl_at_divergence": float(kls[-1]) if (div_point < n and kls) else None,
        "top5_overlap_mean": float(sum(top5s) / len(top5s)) if top5s else None,
        "argmax_match_rate": float(sum(argmatch) / len(argmatch)) if argmatch else None,
    }


def load_model(device, dtype):
    from transformers import AutoTokenizer, AutoConfig
    cfg = AutoConfig.from_pretrained(REPO)
    tc = cfg.text_config
    tok = AutoTokenizer.from_pretrained(REPO)

    def _load(cls):
        try:
            return cls.from_pretrained(REPO, dtype=dtype, attn_implementation="eager")
        except TypeError:
            return cls.from_pretrained(REPO, torch_dtype=dtype, attn_implementation="eager")

    model = None
    for name in ("AutoModelForCausalLM", "AutoModelForImageTextToText",
                 "Gemma4ForConditionalGeneration"):
        try:
            import transformers as T
            cls = getattr(T, name, None)
            if cls is None:
                continue
            model = _load(cls)
            print(f"# modele charge via {name}")
            break
        except Exception as e:
            print(f"#   {name} echoue: {type(e).__name__}: {e}")
    if model is None:
        raise RuntimeError("chargement modele impossible")
    if device.startswith("cuda"):
        model = model.to(device)
    model.train(False)
    return tok, model, tc


def build_ids(tok, text, device):
    try:
        enc = tok.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True, return_tensors="pt", return_dict=True,
        )
        ids = enc["input_ids"]
    except Exception:
        ids = tok(text, return_tensors="pt").input_ids
    return ids.to(device)


def main():
    global STATE
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print("=" * 78)
    print("Gemma-4-E2B-it — Q5 Volet A : COÛT de la quant V-only (MSE 4 bits) sur GÉNÉRATION")
    print("=" * 78)
    print(f"device={device} dtype={dtype} N_new={N_NEW} prompts={len(PROMPTS)}")

    tok, model, tc = load_model(device, dtype)
    n_kv, n_q = tc.num_key_value_heads, tc.num_attention_heads
    print(f"n_kv={n_kv} n_q={n_q} head_dim={tc.head_dim}/{tc.global_head_dim}")

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    quantizers = build_quantizers(device)
    STATE = QState(quantizers, n_kv, n_q)
    install_v_hooks(model)

    MODE = "vonly_mse4"
    per_prompt = []

    for pi, prompt in enumerate(PROMPTS):
        ids = build_ids(tok, prompt, device)
        # baseline d'abord
        STATE.set_mode("baseline")
        base_gen, base_scores = generate_with_scores(model, ids, pad_id)
        # V-quant
        STATE.set_mode(MODE)
        vq_gen, vq_scores = generate_with_scores(model, ids, pad_id)
        cmp = compare_to_baseline(base_gen, base_scores, vq_gen, vq_scores)
        cmp["prompt_idx"] = pi
        cmp["prompt"] = prompt
        cmp["baseline_gen"] = base_gen           # sequence baseline complete
        cmp["vq_gen"] = vq_gen                    # sequence HF-V-quant (reference portage volet B)
        per_prompt.append(cmp)

        print(f"\n### prompt {pi}: {prompt[:58]!r}")
        print(f"  baseline gen[:12] = {base_gen[:12]}")
        print(f"  vq_gen   gen[:12] = {vq_gen[:12]}")
        dp = cmp["divergence_point"]
        print(f"  vonly_mse4  div@{dp:>2}/{N_NEW}  "
              f"top5={cmp['top5_overlap_mean']:.3f}  "
              f"KLmean={cmp['kl_mean']:.4f}  "
              f"argmatch={cmp['argmax_match_rate']:.3f}")

    # ----------------------------------------------------------------------
    # Agregation
    # ----------------------------------------------------------------------
    dps = [r["divergence_point"] for r in per_prompt]
    t5 = [r["top5_overlap_mean"] for r in per_prompt if r["top5_overlap_mean"] is not None]
    kl = [r["kl_mean"] for r in per_prompt if r["kl_mean"] is not None]
    am = [r["argmax_match_rate"] for r in per_prompt if r["argmax_match_rate"] is not None]
    n_div = sum(1 for r in per_prompt if r["diverged"])

    def avg(L):
        return (sum(L) / len(L)) if L else float("nan")

    summary = {
        "mode": MODE,
        "eff_bits": 4.0,
        "n_prompts": len(PROMPTS),
        "n_new": N_NEW,
        "div_point_mean": avg(dps),
        "div_point_min": min(dps),
        "div_point_max": max(dps),
        "n_diverged": n_div,
        "top5_overlap_mean": avg(t5),
        "kl_mean": avg(kl),
        "argmax_match_rate": avg(am),
    }

    print("\n" + "=" * 78)
    print("SYNTHESE V-only MSE 4 bits (moyenne sur prompts)")
    print("=" * 78)
    print(f"  eff_bits           = {summary['eff_bits']}")
    print(f"  div_point_mean     = {summary['div_point_mean']:.2f}/{N_NEW}  "
          f"(min {summary['div_point_min']}, max {summary['div_point_max']})")
    print(f"  prompts diverged   = {n_div}/{len(PROMPTS)}")
    print(f"  top5_overlap_mean  = {summary['top5_overlap_mean']:.4f}")
    print(f"  kl_mean            = {summary['kl_mean']:.5f}")
    print(f"  argmax_match_rate  = {summary['argmax_match_rate']:.4f}")
    print("\nLecture : div_pt eleve + KL faible + top5~1 + argmatch~1 => V-only MSE 4 bits")
    print("preserve la generation. Compression theorique : 4 bits + 1 fp16 norm / vecteur d.")

    out = {
        "model": REPO, "device": device, "dtype": str(dtype).replace("torch.", ""),
        "n_new": N_NEW, "n_prompts": len(PROMPTS),
        "mode": MODE,
        "config": {"n_kv": n_kv, "n_q": n_q,
                   "head_dim_sliding": tc.head_dim, "head_dim_full": tc.global_head_dim},
        "metric_notes": "V-only MSE 4 bits (K fp32) vs baseline. div_point=1er token diff; "
                        "KL/top5 sur contexte identique avant divergence; greedy; PAS de ROUGE-1.",
        "summary": summary,
        "per_prompt": per_prompt,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
