#!/usr/bin/env python3
"""
test_kv_quant_generation.py — Creusage A : quel niveau de complexite TurboQuant
faut-il REELLEMENT sur Gemma-4-E2B-it ?

Compare la GENERATION greedy de Gemma-4-E2B-it en :
  - baseline (KV non quantifie)
  - kv_mse_b4 / kv_mse_b3 (K+V, TurboQuantMSE Hadamard, le mode SIMPLE)
  - kv_prod_b4 (K+V, TurboQuantPROD = MSE + residu QJL 1-bit)
  - vonly_mse_b4 (V seul quantifie)

Fake-quant IN-PLACE au point canonique du KV-cache :
  - K : monkeypatch apply_rotary_pos_emb (post-k_norm, post-RoPE), quantifie les
        tenseurs K (identifies par n_kv=1, distincts de Q n_q=8).
  - V : forward hook sur v_norm (post-v_norm, pas de RoPE), remplace l'output.
Reproduit exactement un KV-cache quantifie (cache stocke K_hat/V_hat).

Metriques (cf critere de succes) :
  - divergence_point : 1er token genere ou l'argmax differe de baseline
  - top5_overlap & KL : sur les positions AVANT divergence (contexte identique
    -> distributions strictement comparables)
  PAS de ROUGE-1 (piege verbatim), PAS de MSE-attention.

head_dim 256 (sliding) et 512 (full) sont pow2 -> Hadamard direct, pas de split.
n_kv=1 -> per-head trivial. MSE/PROD ne necessitent AUCUNE calibration donnees
(codebook sur N(0,1/d), normalisation par vecteur intrinseque).

Output : print + JSON /data/gemma4-zml-probe/kv_quant_generation.json. Rien sur /.
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

# turboquant.py copie a cote de ce script (version M1 a jour) -> /data/gemma4-zml-probe
sys.path.insert(0, "/data/gemma4-zml-probe")
from turboquant import TurboQuantMSE, TurboQuantPROD  # noqa: E402

REPO = "google/gemma-4-E2B-it"
OUT_DIR = Path("/data/gemma4-zml-probe")
OUT_JSON = OUT_DIR / "kv_quant_generation.json"

N_NEW = 48          # tokens generes par prompt
HEAD_DIMS = [256, 512]

PROMPTS = [
    "Explain in one paragraph why the sky appears blue during the day.",
    "It is a truth universally acknowledged, that a single man in possession of a good fortune,",
    "Write a short Python function that returns the n-th Fibonacci number.",
    "What are three practical tips for improving focus while working?",
]


# ==========================================================================
# Quantizers (un par head_dim ; partages entre layers de meme d)
# ==========================================================================
def build_quantizers(device):
    """MSE/PROD n'ont pas besoin de calibration donnees. Un quantizer par d."""
    q = {}
    for d in HEAD_DIMS:
        q[("mse4", d)] = TurboQuantMSE(d, 4, device=device, rotation="hadamard")
        q[("mse3", d)] = TurboQuantMSE(d, 3, device=device, rotation="hadamard")
        q[("prod4", d)] = TurboQuantPROD(d, 4, device=device, rotation="hadamard")
    return q


def fake_quant(x, quantizer):
    """quantize -> dequantize en fp32, meme shape. x: [..., d]."""
    dt = x.dtype
    xf = x.to(torch.float32)
    comp = quantizer.quant(xf)
    xhat = quantizer.dequant(comp)
    return xhat.to(dt)


# ==========================================================================
# Mode global (controle ce que les patchs quantifient)
# ==========================================================================
class QState:
    def __init__(self, quantizers, n_kv, n_q):
        self.q = quantizers
        self.n_kv = n_kv
        self.n_q = n_q
        self.mode = "baseline"   # baseline | mse4 | mse3 | prod4 | vonly_mse4
        self.quant_k = False
        self.quant_v = False
        self.kind = None         # 'mse4'|'mse3'|'prod4' -> quel quantizer

    def set_mode(self, mode):
        self.mode = mode
        if mode == "baseline":
            self.quant_k = self.quant_v = False
            self.kind = None
        elif mode == "vonly_mse4":
            self.quant_k = False
            self.quant_v = True
            self.kind = "mse4"
        else:  # kv_mse4 / kv_mse3 / kv_prod4
            self.quant_k = self.quant_v = True
            self.kind = mode  # 'mse4'|'mse3'|'prod4'

    def quantizer_for(self, d):
        return self.q[(self.kind, d)]


STATE: QState | None = None


def install_k_patch(gm):
    """Monkeypatch apply_rotary_pos_emb : quantifie la sortie K (n_kv au -2)."""
    orig = gm.apply_rotary_pos_emb

    def patched(*args, **kwargs):
        out = orig(*args, **kwargs)
        if STATE is not None and STATE.quant_k and torch.is_tensor(out) \
                and out.dim() == 4 and out.shape[-2] == STATE.n_kv \
                and STATE.n_kv != STATE.n_q:
            d = out.shape[-1]
            out = fake_quant(out, STATE.quantizer_for(d))
        return out

    gm.apply_rotary_pos_emb = patched
    return orig


def install_v_hooks(model):
    """Forward hooks sur v_norm des producers : remplace l'output par V_hat."""
    handles = []
    import re
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


# ==========================================================================
# Generation + metriques
# ==========================================================================
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

    # positions comparables = 0..div_point inclus (contexte identique a ces steps)
    compare_upto = min(div_point + 1, n)
    kls, top5s, argmatch = [], [], []
    for k in range(compare_upto):
        pb = F.log_softmax(base_scores[k], dim=-1)
        pm = F.log_softmax(mode_scores[k], dim=-1)
        # KL(base || mode) = sum exp(pb)*(pb-pm)
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
        "mode_gen_head": mode_gen[:12],
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
        if ids.dim() == 2:
            ids = ids
    except Exception:
        ids = tok(text, return_tensors="pt").input_ids
    return ids.to(device)


def main():
    global STATE
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print("=" * 78)
    print("Gemma-4-E2B-it — cout de quantification KV sur la GENERATION")
    print("=" * 78)
    print(f"device={device} dtype={dtype} N_new={N_NEW} prompts={len(PROMPTS)}")

    tok, model, tc = load_model(device, dtype)
    n_kv, n_q = tc.num_key_value_heads, tc.num_attention_heads
    print(f"n_kv={n_kv} n_q={n_q} head_dim={tc.head_dim}/{tc.global_head_dim}")

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    quantizers = build_quantizers(device)
    STATE = QState(quantizers, n_kv, n_q)

    import transformers.models.gemma4.modeling_gemma4 as gm
    install_k_patch(gm)
    install_v_hooks(model)

    MODES = ["baseline", "kv_mse4", "kv_mse3", "kv_prod4", "vonly_mse4"]
    # map mode -> kind pour QState.set_mode
    def set_mode(m):
        if m == "baseline":
            STATE.set_mode("baseline")
        elif m == "vonly_mse4":
            STATE.set_mode("vonly_mse4")
        elif m == "kv_mse4":
            STATE.set_mode("mse4")
        elif m == "kv_mse3":
            STATE.set_mode("mse3")
        elif m == "kv_prod4":
            STATE.set_mode("prod4")

    results = {m: [] for m in MODES}

    for pi, prompt in enumerate(PROMPTS):
        ids = build_ids(tok, prompt, device)
        # baseline d'abord
        set_mode("baseline")
        base_gen, base_scores = generate_with_scores(model, ids, pad_id)
        results["baseline"].append({"prompt_idx": pi, "gen_head": base_gen[:12]})
        print(f"\n### prompt {pi}: {prompt[:60]!r}")
        print(f"  baseline gen[:12] = {base_gen[:12]}")

        for m in MODES:
            if m == "baseline":
                continue
            set_mode(m)
            mgen, mscores = generate_with_scores(model, ids, pad_id)
            cmp = compare_to_baseline(base_gen, base_scores, mgen, mscores)
            cmp["prompt_idx"] = pi
            results[m].append(cmp)
            dp = cmp["divergence_point"]
            print(f"  {m:<11} div@{dp:>2}/{N_NEW}  "
                  f"top5={cmp['top5_overlap_mean']:.3f}  "
                  f"KLmean={cmp['kl_mean']:.4f}  "
                  f"argmatch={cmp['argmax_match_rate']:.3f}")

    # ----------------------------------------------------------------------
    # Agregation
    # ----------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("SYNTHESE (moyenne sur prompts)")
    print("=" * 78)
    print(f"{'mode':<12}{'eff.bits':>9}{'div_pt':>9}{'top5':>8}{'KLmean':>10}{'argmatch':>10}")
    summary = {}
    for m in MODES:
        if m == "baseline":
            continue
        rs = results[m]
        dps = [r["divergence_point"] for r in rs]
        t5 = [r["top5_overlap_mean"] for r in rs if r["top5_overlap_mean"] is not None]
        kl = [r["kl_mean"] for r in rs if r["kl_mean"] is not None]
        am = [r["argmax_match_rate"] for r in rs if r["argmax_match_rate"] is not None]
        # effective bits
        d = 256
        if m in ("kv_mse4", "vonly_mse4"):
            eb = 4.0
        elif m == "kv_mse3":
            eb = 3.0
        elif m == "kv_prod4":
            eb = 4.0
        else:
            eb = float("nan")
        avg = lambda L: (sum(L) / len(L)) if L else float("nan")
        summary[m] = {
            "div_point_mean": avg(dps), "div_point_min": min(dps),
            "top5_overlap_mean": avg(t5), "kl_mean": avg(kl),
            "argmax_match_rate": avg(am), "eff_bits": eb,
        }
        print(f"{m:<12}{eb:>9.1f}{avg(dps):>9.1f}{avg(t5):>8.3f}"
              f"{avg(kl):>10.4f}{avg(am):>10.3f}")

    print("\nLecture : div_pt eleve + KL faible + top5~1 + argmatch~1 => le mode")
    print("preserve la generation. Si kv_mse* (le SIMPLE) ~ kv_prod4, le residu QJL")
    print("est superflu sur Gemma-4-E2B -> portage ZML minimal (MSE seul) suffit.")

    out = {
        "model": REPO, "device": device, "dtype": str(dtype).replace("torch.", ""),
        "n_new": N_NEW, "n_prompts": len(PROMPTS),
        "config": {"n_kv": n_kv, "n_q": n_q,
                   "head_dim_sliding": tc.head_dim, "head_dim_full": tc.global_head_dim},
        "metric_notes": "div_point=1er token diff vs baseline; KL/top5 sur contexte "
                        "identique avant divergence; greedy; PAS de ROUGE-1/MSE-attn.",
        "summary": summary,
        "per_prompt": {m: results[m] for m in MODES},
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
