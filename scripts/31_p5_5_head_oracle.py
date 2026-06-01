"""P5.5 — PyTorch oracle + fixture : tête de sortie (final norm + lm_head + softcap).

Fin du forward (modeling_gemma4 Gemma4TextModel.forward L1708 + Gemma4ForCausalLM) :
    hidden = norm(last_hidden_state)              # model.norm = Gemma4RMSNorm(1536, with_scale)
    logits = lm_head(hidden)                       # lm_head.weight = embed_tokens.weight (TIED) [262144,1536]
    logits = final_logit_softcapping * tanh(logits / final_logit_softcapping)   # softcap=30.0

lm_head tied + vocab 262144 -> table 1.6 GB. On teste sur slice vocab 4096 (lm_head pleine table
= mécaniquement identique). Op neuve = softcap (30*tanh(x/30)) ; tanh = zml Tensor.tanh.

Pipeline oracle :
    hidden_final = synthétique seed 1337 [1,4,1536]
    normed       = Gemma4RMSNorm(hidden_final)         # module réel
    logits       = normed @ lm_head_slice.T            # [1,4,4096]
    logits       = 30 * tanh(logits / 30)

Fixture (4 tenseurs) : hidden_final, norm_weight, lm_head_slice (inputs), logits_out (oracle).
Interdits : couches décodeur, embedding, PLE.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights" / "model.safetensors"
OUT_FIXTURE = ROOT / "fixtures" / "p5_5_head.safetensors"
OUT_MANIFEST = ROOT / "fixtures" / "p5_5_head_manifest.json"

HIDDEN = 1536
VOCAB_SLICE = 4096
SEQ_LEN = 4
RMS_EPS = 1e-6
SOFTCAP = 30.0
NORM_KEY = "model.language_model.norm.weight"
EMBED_KEY = "model.language_model.embed_tokens.weight"   # lm_head tied


def main() -> None:
    assert WEIGHTS.exists(), f"missing weights {WEIGHTS}"
    torch.manual_seed(1337)
    hidden_final = torch.randn(1, SEQ_LEN, HIDDEN, dtype=torch.float32)

    with safe_open(str(WEIGHTS), framework="pt") as s:
        assert NORM_KEY in s.keys(), f"missing {NORM_KEY}"
        norm_weight = s.get_tensor(NORM_KEY).to(torch.float32).contiguous()
        lm_head_slice = s.get_tensor(EMBED_KEY)[:VOCAB_SLICE].to(torch.float32).contiguous()  # tied
    assert tuple(norm_weight.shape) == (HIDDEN,)
    assert tuple(lm_head_slice.shape) == (VOCAB_SLICE, HIDDEN)

    print("=" * 70)
    print(f"P5.5 — PyTorch oracle head : final norm + lm_head(tied) + softcap (vocab slice {VOCAB_SLICE})")
    print("=" * 70)
    print(f"softcap = {SOFTCAP} ; lm_head TIED = embed_tokens.weight")

    norm = Gemma4RMSNorm(HIDDEN, eps=RMS_EPS)
    with torch.no_grad():
        norm.weight.copy_(norm_weight)

    normed = norm(hidden_final)                                # [1,4,1536]
    logits_raw = torch.nn.functional.linear(normed, lm_head_slice)   # [1,4,4096]
    logits_out = (SOFTCAP * torch.tanh(logits_raw / SOFTCAP)).contiguous()
    assert tuple(logits_out.shape) == (1, SEQ_LEN, VOCAB_SLICE)
    assert not torch.isnan(logits_out).any()

    # Sanity softcap : borne |logits| <= softcap.
    print(f"softcap effect : |logits_raw| max = {logits_raw.abs().max():.3f} -> |logits_out| max = {logits_out.abs().max():.3f} (<= {SOFTCAP})")
    assert logits_out.abs().max() <= SOFTCAP + 1e-4
    # Sanity : la non-linéarité mord vraiment (logits_raw a des valeurs > softcap).
    print(f"raw logits > softcap : {(logits_raw.abs() > SOFTCAP).sum().item()} valeurs (softcap actif)")

    print("\nFixed points (logits_out):")
    for q in [0, 3]:
        vals = logits_out[0, q, :8].tolist()
        print(f"  logits_out[0,{q},:8] = [{', '.join(f'{v:.10f}' for v in vals)}]")
    print(f"\nStats logits_out: mean={logits_out.mean():.4e} std={logits_out.std():.4e} "
          f"min={logits_out.min():.4e} max={logits_out.max():.4e}")

    tensors = {
        "hidden_final": hidden_final.contiguous(),
        "norm_weight": norm_weight,
        "lm_head_slice": lm_head_slice,
        "logits_out": logits_out,
    }
    OUT_FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(OUT_FIXTURE))
    print(f"\nwrote {OUT_FIXTURE}  ({sum(t.numel()*t.element_size() for t in tensors.values())} bytes)")

    manifest = {
        "source": "P5.5 PyTorch oracle head (final norm + lm_head tied + softcap), vocab slice 4096",
        "spec_refs": ["Gemma4TextModel.forward norm finale ; lm_head tied = embed_tokens.weight",
                      "final_logit_softcapping=30 : logits = 30*tanh(logits/30)"],
        "config": {"hidden": HIDDEN, "vocab_slice": VOCAB_SLICE, "seq_len": SEQ_LEN,
                   "rms_eps": RMS_EPS, "softcap": SOFTCAP, "lm_head_tied": True},
        "tensors": {n: {"shape": list(t.shape), "dtype": "float32"} for n, t in tensors.items()},
        "zml_pipeline_hint": [
            "hidden_final {.b,.s,.d=1536} ; norm_weight {.d} ; lm_head_slice {.voc=4096,.d=1536}",
            "normed = rmsNorm(hidden_final,.d,1e-6).mul(norm_weight.broad)",
            "logits = normed.dot(lm_head_slice,.d)   {.b,.s,.voc}",
            "logits_out = logits.scale(1/30).tanh().scale(30)   # softcap",
            "compare vs logits_out oracle [1,4,4096], tol 1e-4",
        ],
        "expected_zml_max_abs_le": 1.0e-4,
        "interdits_p5_5": ["couches décodeur", "embedding", "PLE"],
        "note": "lm_head pleine table 262144 = 1.6GB impraticable ; slice 4096 (mécaniquement identique).",
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {OUT_MANIFEST}")
    print("\nP5.5 oracle + fixture export PASS.")


if __name__ == "__main__":
    main()
