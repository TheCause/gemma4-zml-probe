"""P5.2.G — PyTorch oracle + fixture : post_attention_layernorm + résiduel.

Ferme la SOUS-COUCHE ATTENTION. Reproduit la fin du bloc attention de
`Gemma4TextDecoderLayer.forward` (modeling_gemma4.py 5.9.0, L1395-1406) :

    residual = hidden_states                       # PRE input_layernorm
    hidden_states = input_layernorm(hidden_states)
    hidden_states, _ = self.self_attn(...)         # -> o_proj output (P5.2.F)
    hidden_states = post_attention_layernorm(hidden_states)
    hidden_states = residual + hidden_states        # <-- ce que valide P5.2.G

Donc la gate valide l'OPÉRATION : `out = residual + post_attention_layernorm(attn_output)`.

Inputs :
- `attn_output` = `o_proj_out` de P5.2.F (sortie de la projection o_proj), [1,4,1536].
- `residual`    = stand-in : `hidden_input` de C.0 (l'input d'attention de layer 15, [1,4,1536]).
  ⚠️ Subtilité architecturale : le VRAI résiduel est le hidden state PRÉ-`input_layernorm`.
  Notre pilote synthétique (C.0) part du POST-norm (input de q_proj) et ne modélise pas
  `input_layernorm`. On utilise donc `hidden_input` comme stand-in réaliste : l'oracle ET le
  ZML consomment le MÊME tenseur, ce qui valide proprement l'op `norm + add`. Le câblage
  end-to-end du résiduel (avec input_layernorm) sera fait à l'assemblage de la couche.
- `post_attention_layernorm.weight` [1536] : Gemma4RMSNorm(with_scale=True), pattern Llama
  `_norm(x) * weight` (init weight=1, PAS `(1+weight)`).

Oracle = source de vérité : on instancie le MODULE réel `Gemma4RMSNorm` de transformers (pas
une ré-dérivation manuelle de la formule RMSNorm), avec le poids du checkpoint.

Fixture (4 tenseurs) : `attn_output`, `residual`, `pa_norm_weight` (inputs), `attn_sublayer_out`
(oracle). Le ZML fera `residual + rmsNorm(attn_output,.d).mul(weight)` et comparera (tol 1e-4).

Interdits P5.2.G : MLP, pre/post_feedforward_layernorm, input_layernorm (non modélisé), layer 14.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm


ROOT = Path(__file__).resolve().parents[1]
F_FIXTURE = ROOT / "fixtures" / "p5_2_f_oproj_layer15.safetensors"          # P5.2.F (o_proj_out)
C0_FIXTURE = ROOT / "fixtures" / "q_only_reader_layer15.safetensors"        # C.0 (hidden_input)
WEIGHTS = ROOT / "weights" / "model.safetensors"
OUT_FIXTURE = ROOT / "fixtures" / "p5_2_g_attn_residual_layer15.safetensors"
OUT_MANIFEST = ROOT / "fixtures" / "p5_2_g_attn_residual_layer15_manifest.json"

LAYER_IDX = 15
HIDDEN = 1536
SEQ_LEN = 4
RMS_EPS = 1e-6
PA_LN_KEY = f"model.language_model.layers.{LAYER_IDX}.post_attention_layernorm.weight"

EXPECTED_SHAPE = (1, SEQ_LEN, HIDDEN)   # [1,4,1536]


def main() -> None:
    assert F_FIXTURE.exists(), f"missing P5.2.F fixture {F_FIXTURE}"
    assert C0_FIXTURE.exists(), f"missing C.0 fixture {C0_FIXTURE}"
    assert WEIGHTS.exists(), f"missing weights {WEIGHTS}"

    attn_output = load_file(str(F_FIXTURE))["o_proj_out"].to(torch.float32).contiguous()
    residual = load_file(str(C0_FIXTURE))["hidden_input"].to(torch.float32).contiguous()
    for name, t in [("attn_output", attn_output), ("residual", residual)]:
        assert tuple(t.shape) == EXPECTED_SHAPE, f"{name} {tuple(t.shape)} != {EXPECTED_SHAPE}"

    with safe_open(str(WEIGHTS), framework="pt") as s:
        assert PA_LN_KEY in s.keys(), f"missing {PA_LN_KEY}"
        pa_norm_weight = s.get_tensor(PA_LN_KEY).to(torch.float32).contiguous()
    assert tuple(pa_norm_weight.shape) == (HIDDEN,), f"pa_norm_weight {tuple(pa_norm_weight.shape)} != ({HIDDEN},)"

    print("=" * 70)
    print(f"P5.2.G — PyTorch oracle post_attention_layernorm + résiduel (layer {LAYER_IDX})")
    print("=" * 70)
    print(f"out = residual + Gemma4RMSNorm(attn_output)   [eps={RMS_EPS}, with_scale, pattern Llama *weight]")
    print(f"attn_output = o_proj_out (P5.2.F) ; residual = hidden_input (C.0, stand-in)")
    print()

    # === Oracle = MODULE réel Gemma4RMSNorm (pas de ré-dérivation manuelle) ===
    ln = Gemma4RMSNorm(HIDDEN, eps=RMS_EPS)          # with_scale=True par défaut
    assert ln.with_scale, "post_attention_layernorm doit avoir with_scale=True"
    with torch.no_grad():
        ln.weight.copy_(pa_norm_weight)
    normed = ln(attn_output)                          # post_attention_layernorm(attn_output)
    attn_sublayer_out = residual + normed             # résiduel
    assert tuple(attn_sublayer_out.shape) == EXPECTED_SHAPE
    assert not torch.isnan(attn_sublayer_out).any() and not torch.isinf(attn_sublayer_out).any()

    # Sanity : RMSNorm recomputé à la main (formule Gemma4 _norm) doit matcher le module.
    ms = attn_output.float().pow(2).mean(-1, keepdim=True) + RMS_EPS
    normed_manual = (attn_output.float() * torch.pow(ms, -0.5)) * pa_norm_weight.float()
    norm_diff = (normed_manual.type_as(attn_output) - normed).abs().max().item()
    print(f"Sanity RMSNorm module vs formule manuelle |diff|_max = {norm_diff:.3e} (attendu ~0)")
    assert norm_diff < 1e-5, f"formule RMSNorm divergente (got {norm_diff})"

    # Le résiduel doit déplacer la sortie (sinon le test du add serait vide).
    add_shift = (attn_sublayer_out - normed).abs().max().item()
    print(f"Effet résiduel : max|out - normed| = {add_shift:.4f} (doit être > 0, sinon add vide)")
    assert add_shift > 1e-3, "residual sans effet — add non testé"
    print()

    print("Fixed points (attn_sublayer_out):")
    for q in [0, 3]:
        vals = attn_sublayer_out[0, q, :8].tolist()
        print(f"  attn_sublayer_out[0,{q},:8] = [{', '.join(f'{v:.10f}' for v in vals)}]")
    print()

    print("Stats:")
    for name, t in [("attn_output", attn_output), ("residual", residual),
                    ("pa_norm_weight", pa_norm_weight), ("normed", normed),
                    ("attn_sublayer_out", attn_sublayer_out)]:
        print(
            f"  {name:<18} shape={tuple(t.shape)!s:<14} "
            f"mean={t.mean().item(): .6e} std={t.std().item(): .6e} "
            f"min={t.min().item(): .6e} max={t.max().item(): .6e}"
        )

    tensors = {
        "attn_output": attn_output.contiguous(),
        "residual": residual.contiguous(),
        "pa_norm_weight": pa_norm_weight.contiguous(),
        "attn_sublayer_out": attn_sublayer_out.contiguous(),
    }
    OUT_FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(OUT_FIXTURE))
    total_bytes = sum(t.numel() * t.element_size() for t in tensors.values())
    print()
    print(f"wrote {OUT_FIXTURE}  ({total_bytes} bytes of tensor payload)")

    manifest = {
        "source": "P5.2.G PyTorch oracle post_attention_layernorm + résiduel (layer 15)",
        "spec_refs": [
            "modeling_gemma4.py (5.9.0) Gemma4TextDecoderLayer.forward L1395-1406 (post_attn_ln + residual add)",
            "Gemma4RMSNorm L191-211 (with_scale=True : _norm(x) * weight, init weight=1, pattern Llama)",
        ],
        "layer_idx": LAYER_IDX,
        "config": {"hidden": HIDDEN, "seq_len": SEQ_LEN, "rms_eps": RMS_EPS, "with_scale": True},
        "residual_note": (
            "residual = hidden_input de C.0 (stand-in). Le VRAI résiduel est le hidden state "
            "pré-input_layernorm ; pilote synthétique ne modélise pas input_layernorm. La gate "
            "valide l'op (post_attn_norm + add), pas la sémantique end-to-end du résiduel."
        ),
        "pipeline": [
            "normed = Gemma4RMSNorm(attn_output)   [_norm(x) * weight, eps=1e-6]",
            "out    = residual + normed             [1,4,1536]",
        ],
        "tensors": {name: {"shape": list(t.shape), "dtype": "float32"} for name, t in tensors.items()},
        "zml_pipeline_hint": [
            "attn_output, residual = load -> tags {.b,.q,.d}   [1,4,1536]",
            "pa_norm_weight = load -> tag {.d}   [1536]",
            "normed = zml.nn.rmsNorm(attn_output, .d, 1e-6)",
            "scaled = normed.mul(pa_norm_weight.broad(normed.shape()))   [pattern Llama]",
            "out = residual.add(scaled)   [1,4,1536]",
            "compare out vs oracle attn_sublayer_out, tolérance 1e-4",
        ],
        "expected_zml_max_abs_le": 1.0e-4,
        "expected_zml_residual_hint": "rmsNorm + mul + add, fp32 ; résidu attendu très faible (~1e-6, cf q_norm C.2 6.7e-6)",
        "checks": {"rmsnorm_formula_diff": norm_diff, "residual_add_shift": add_shift},
        "interdits_p5_2_g": ["MLP", "pre/post_feedforward_layernorm", "input_layernorm (non modélisé)", "layer 14"],
        "closes": "sous-couche ATTENTION (qkv->attn->o_proj->post_attn_norm->residual)",
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {OUT_MANIFEST}")
    print()
    print("P5.2.G oracle + fixture export PASS.")


if __name__ == "__main__":
    main()
