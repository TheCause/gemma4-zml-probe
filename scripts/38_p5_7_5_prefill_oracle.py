"""P5.7.5 — Oracle prefill 35 couches : Gemma4TextModel réel → last_hidden_state.

Le moteur ZML (embedding + PLE frontend + 35 couches dispatchées sliding/full + KV sharing YOCO
+ final norm) sera comparé à ce last_hidden_state. On construit juste le modèle TEXTE (pas les
tours vision/audio), on charge les 600 poids model.language_model.*, on forward sur des token ids.

Exporte : input_ids, cos/sin sliding (256) + full (512) pour les S positions, attn_mask causal,
last_hidden_state (oracle). Les POIDS viennent du checkpoint réel côté ZML (pas de la fixture).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel, Gemma4TextRotaryEmbedding
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights" / "model.safetensors"
OUT_FIXTURE = ROOT / "fixtures" / "p5_7_5_prefill.safetensors"
OUT_MANIFEST = ROOT / "fixtures" / "p5_7_5_prefill_manifest.json"

SEQ_LEN = 4
INPUT_IDS = [2, 105, 2048, 4095]
PREFIX = "model.language_model."


def main() -> None:
    assert WEIGHTS.exists()
    torch.manual_seed(1337)
    cfg = AutoConfig.from_pretrained("google/gemma-4-E2B-it")
    tc = getattr(cfg, "text_config", cfg)

    # Le modèle complet en fp32 (~17 Go, embed_tokens_per_layer fp32 = 9.4 Go) ne tient pas dans
    # les 23 Go de la VM -> on construit/charge en bf16 (dtype natif du checkpoint, ~9 Go). HF fait
    # déjà les RMSNorm en fp32 en interne. La référence last_hidden est exportée en fp32 (convertie).
    print("Construction Gemma4TextModel (texte seul, bf16) + chargement des poids...")
    torch.set_default_dtype(torch.bfloat16)
    model = Gemma4TextModel(tc)
    model.train(False)

    # Charger les poids model.language_model.* (strip prefix), bf16.
    state = {}
    with safe_open(str(WEIGHTS), framework="pt") as s:
        for k in s.keys():
            if k.startswith(PREFIX):
                state[k[len(PREFIX):]] = s.get_tensor(k)
    missing, unexpected = model.load_state_dict(state, strict=False)
    real_missing = [m for m in missing if "weight" in m or m.endswith("layer_scalar")]
    print(f"load_state_dict: {len(state)} poids ; missing(non-buffer)={len(real_missing)} unexpected={len(unexpected)}")
    assert not real_missing, real_missing[:10]

    input_ids = torch.tensor(INPUT_IDS, dtype=torch.long).view(1, SEQ_LEN)
    pos = torch.arange(SEQ_LEN).unsqueeze(0)

    print("Forward prefill (35 couches, bf16)...")
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=False)
    last_hidden = out.last_hidden_state.to(torch.float32).contiguous()   # [1,4,1536]
    assert tuple(last_hidden.shape) == (1, SEQ_LEN, 1536)
    assert not torch.isnan(last_hidden).any()

    # cos/sin sliding + full pour la RoPE manuelle ZML (fp32).
    torch.set_default_dtype(torch.float32)
    rot = Gemma4TextRotaryEmbedding(tc)
    cos_s, sin_s = rot(last_hidden, pos, layer_type="sliding_attention")   # [1,4,256]
    cos_f, sin_f = rot(last_hidden, pos, layer_type="full_attention")      # [1,4,512]
    min_val = torch.finfo(torch.float32).min
    idx = torch.arange(SEQ_LEN)
    causal = (idx.view(SEQ_LEN, 1) >= idx.view(1, SEQ_LEN))
    attn_mask = torch.where(causal, torch.zeros(()), torch.full((), min_val)).view(1, 1, SEQ_LEN, SEQ_LEN)

    print("=" * 70)
    print("P5.7.5 — oracle prefill 35 couches")
    print("=" * 70)
    print(f"input_ids = {INPUT_IDS}")
    print("Fixed points (last_hidden_state):")
    for q in [0, 3]:
        vals = last_hidden[0, q, :8].tolist()
        print(f"  last_hidden[0,{q},:8] = [{', '.join(f'{v:.10f}' for v in vals)}]")
    print(f"Stats: mean={last_hidden.mean():.4e} std={last_hidden.std():.4e} "
          f"min={last_hidden.min():.4e} max={last_hidden.max():.4e}")

    tensors = {
        "input_ids": input_ids.to(torch.int32),
        "cos_sliding": cos_s.contiguous(), "sin_sliding": sin_s.contiguous(),
        "cos_full": cos_f.contiguous(), "sin_full": sin_f.contiguous(),
        "attn_mask": attn_mask.contiguous(),
        "last_hidden": last_hidden,
    }
    OUT_FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(OUT_FIXTURE))
    print(f"\nwrote {OUT_FIXTURE}")

    manifest = {
        "source": "P5.7.5 oracle prefill 35 couches (Gemma4TextModel réel -> last_hidden_state)",
        "input_ids": INPUT_IDS, "seq_len": SEQ_LEN,
        "tensors": {n: {"shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", "")} for n, t in tensors.items()},
        "expected_zml_max_abs_le": 2.0e-3,
        "note": "poids depuis checkpoint réel côté ZML ; embedding+PLE+35 couches+final norm ; KV sharing YOCO.",
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {OUT_MANIFEST}\nP5.7.5 oracle PASS.")


if __name__ == "__main__":
    main()
