"""P5.7.7 decode-3 — Oracle e2e du MOTEUR decode (35 couches, 1 token) -> logits == HF.

Le test décisif du decode : un pas de génération complet à travers les 35 couches en mode incrémental,
avec les 2 caches multi-slots (12 producers sliding hd=256 + 3 producers full hd=512) et le YOCO
(writers 13/14 -> readers 15-34), → final norm → last_hidden → logits → argmax.

Méthode (oracle = modèle réel) :
  1. build hybride streaming (cf 39).
  2. prefill use_cache=True -> past_key_values (DynamicCache). pkv.layers[i].keys/.values = cache
     producer i [1,1,4,hd]. Readers (YOCO) absents de pkv.layers (15 entrées = producers 0..14).
  3. nouveau token = argmax prefill. Pré-gather embed/embptl du token (entrée du moteur ZML).
  4. cos/sin full à pos p=4 (Gemma4TextRotaryEmbedding, full_attention) — sliding rope calculée en ZML.
  5. decode 1 step -> last_hidden_decode + logits + argmax (référence e2e).

Caches empaquetés : cache_sl_{k,v} [12,1,1,5,256] (slots = ordre sliding_producers), cache_fl_{k,v}
[3,1,1,5,512] (slots = ordre full_producers), col 4 = 0 (à scatter côté ZML).
Fixture : fixtures/p5_7_7_decode3.safetensors
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel, Gemma4TextRotaryEmbedding

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights" / "model.safetensors"
OUT_FIXTURE = ROOT / "fixtures" / "p5_7_7_decode3.safetensors"
OUT_MANIFEST = ROOT / "fixtures" / "p5_7_7_decode3_manifest.json"
SEQ_LEN = 4
INPUT_IDS = [2, 105, 2048, 4095]
PREFIX = "model.language_model."
EMBPTL = "embed_tokens_per_layer.weight"

SLIDING_PRODUCERS = [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13]  # 12, hd=256
FULL_PRODUCERS = [4, 9, 14]                                   # 3,  hd=512
HD_S, HD_F = 256, 512


def set_submodule_attr(root, dotted, value):
    parts = dotted.split(".")
    obj = root
    for p in parts[:-1]:
        obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
    setattr(obj, parts[-1], value)


def build_hybrid_model(tc):
    torch.set_default_dtype(torch.bfloat16)
    model = Gemma4TextModel(tc)
    model.train(False)
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    loaded = set()
    with safe_open(str(WEIGHTS), framework="pt") as s:
        for k in s.keys():
            if not k.startswith(PREFIX):
                continue
            name = k[len(PREFIX):]
            t = s.get_tensor(k)
            with torch.no_grad():
                if name in params:
                    if name == EMBPTL:
                        params[name].copy_(t)
                    else:
                        set_submodule_attr(model, name, torch.nn.Parameter(t.to(torch.float32), requires_grad=False))
                elif name in buffers:
                    set_submodule_attr(model, name, t.to(torch.float32))
                else:
                    del t
                    continue
            loaded.add(name)
            del t
    ls_missing = [n for n in buffers if n.endswith("layer_scalar") and n not in loaded]
    assert not ls_missing, f"layer_scalar buffers manquants: {ls_missing[:5]}"
    for name, buf in list(model.named_buffers()):
        if buf.dtype == torch.bfloat16:
            set_submodule_attr(model, name, buf.float())
    model.embed_tokens.embed_scale = torch.tensor(1536.0 ** 0.5, dtype=torch.float32)
    torch.set_default_dtype(torch.float32)
    return model


def pad_cache(t):
    """[1,1,4,hd] -> [1,1,5,hd] col 4 = 0 (fp32)."""
    hd = t.shape[-1]
    out = torch.zeros(1, 1, SEQ_LEN + 1, hd, dtype=torch.float32)
    out[:, :, :SEQ_LEN, :] = t.float()
    return out


def main() -> None:
    assert WEIGHTS.exists(), WEIGHTS
    torch.manual_seed(1337)
    cfg = AutoConfig.from_pretrained("google/gemma-4-E2B-it")
    tc = getattr(cfg, "text_config", cfg)
    softcap = float(getattr(tc, "final_logit_softcapping", 30.0))

    print("Build hybride (streaming) ...")
    model = build_hybrid_model(tc)

    input_ids = torch.tensor(INPUT_IDS, dtype=torch.long).view(1, SEQ_LEN)
    print("Prefill use_cache=True ...")
    with torch.no_grad():
        out_pref = model(input_ids=input_ids, use_cache=True)
    pkv = out_pref.past_key_values
    assert pkv is not None
    last_hidden_pref = out_pref.last_hidden_state.to(torch.float32)
    lm_w = model.embed_tokens.weight.to(torch.float32)
    logits_last = last_hidden_pref[:, -1, :] @ lm_w.t()
    logits_last = softcap * torch.tanh(logits_last / softcap)
    new_token = int(logits_last.argmax(dim=-1).item())
    print(f"  nouveau token (decode p={SEQ_LEN}) = {new_token}")

    # --- extraire les 15 caches producers depuis pkv.layers[i] ---
    n_layers_cached = len(pkv.layers)
    print(f"  pkv.layers = {n_layers_cached} entrées (attendu >=15 ; producers 0..14)")
    cache_sl_k = torch.zeros(len(SLIDING_PRODUCERS), 1, 1, SEQ_LEN + 1, HD_S, dtype=torch.float32)
    cache_sl_v = torch.zeros_like(cache_sl_k)
    cache_fl_k = torch.zeros(len(FULL_PRODUCERS), 1, 1, SEQ_LEN + 1, HD_F, dtype=torch.float32)
    cache_fl_v = torch.zeros_like(cache_fl_k)
    for slot, i in enumerate(SLIDING_PRODUCERS):
        ki, vi = pkv.layers[i].keys, pkv.layers[i].values
        assert tuple(ki.shape) == (1, 1, SEQ_LEN, HD_S), f"L{i} k {ki.shape}"
        cache_sl_k[slot] = pad_cache(ki)
        cache_sl_v[slot] = pad_cache(vi)
    for slot, i in enumerate(FULL_PRODUCERS):
        ki, vi = pkv.layers[i].keys, pkv.layers[i].values
        assert tuple(ki.shape) == (1, 1, SEQ_LEN, HD_F), f"L{i} k {ki.shape}"
        cache_fl_k[slot] = pad_cache(ki)
        cache_fl_v[slot] = pad_cache(vi)

    # --- pré-gather embed + embptl du nouveau token (entrée moteur ZML, cf 39) ---
    ids = torch.tensor([new_token], dtype=torch.long)
    embed_slice = model.embed_tokens.weight[ids].to(torch.bfloat16).contiguous().view(1, 1, 1536)
    embptl_slice = model.embed_tokens_per_layer.weight[ids].contiguous().view(1, 1, -1)  # bf16 [1,1,8960]

    # --- cos/sin full à pos p=4 (sliding rope calculée en ZML) ---
    rot = Gemma4TextRotaryEmbedding(tc)
    pos = torch.tensor([[SEQ_LEN]])  # [1,1] = position 4
    cos_f, sin_f = rot(last_hidden_pref, pos, layer_type="full_attention")  # [1,1,512]
    assert tuple(cos_f.shape) == (1, 1, HD_F), cos_f.shape

    mask_decode = torch.zeros(1, 1, 1, SEQ_LEN + 1, dtype=torch.float32)

    # --- decode 1 step (référence e2e) ---
    print("Decode 1 step (past_key_values=pkv) ...")
    dec_ids = torch.tensor([[new_token]], dtype=torch.long)
    with torch.no_grad():
        out_dec = model(input_ids=dec_ids, past_key_values=pkv, use_cache=True)
    last_hidden = out_dec.last_hidden_state.to(torch.float32).contiguous().clone()  # [1,1,1536]
    logits = last_hidden[:, -1, :] @ lm_w.t()
    logits = (softcap * torch.tanh(logits / softcap)).contiguous()
    argmax_dec = int(logits.argmax(dim=-1).item())
    assert not torch.isnan(last_hidden).any()
    print(f"  last_hidden decode shape {tuple(last_hidden.shape)} ; argmax (token suivant) = {argmax_dec}")

    tensors = {
        "embed_slice": embed_slice, "embptl_slice": embptl_slice,
        "cos_full": cos_f.contiguous(), "sin_full": sin_f.contiguous(),
        "mask_decode": mask_decode.contiguous(),
        "cache_sl_k": cache_sl_k.contiguous(), "cache_sl_v": cache_sl_v.contiguous(),
        "cache_fl_k": cache_fl_k.contiguous(), "cache_fl_v": cache_fl_v.contiguous(),
        "pos_idx": torch.tensor([SEQ_LEN], dtype=torch.int32),
        "last_hidden": last_hidden,                                 # référence e2e
        "logits": logits.contiguous(),                              # [1,262144] softcappés — magnitudes (cf P5.7.6)
        "argmax": torch.tensor([argmax_dec], dtype=torch.int32),    # token suivant prédit par HF
        "decode_token": torch.tensor([new_token], dtype=torch.int32),
    }
    for k, t in tensors.items():
        assert not torch.isnan(t.float()).any(), f"NaN dans {k}"

    print("=" * 72)
    print("P5.7.7 decode-3 — oracle e2e moteur decode 35 couches")
    print(f"  token decode = {new_token} ; pos = {SEQ_LEN} ; token suivant (argmax) = {argmax_dec}")
    print(f"  last_hidden[0,0,:6] = {[round(v,7) for v in last_hidden[0,0,:6].tolist()]}")
    print(f"  stats last_hidden: mean={last_hidden.mean():.4e} std={last_hidden.std():.4e}")

    OUT_FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(OUT_FIXTURE))
    print("wrote", OUT_FIXTURE)

    manifest = {
        "source": "P5.7.7 decode-3 oracle e2e (moteur decode 35 couches, 1 token)",
        "design": "docs/P5_7_7_decode.md",
        "input_ids": INPUT_IDS, "decode_token": new_token, "next_token_argmax": argmax_dec, "pos_idx": SEQ_LEN,
        "caches": "cache_sl_{k,v} [12,1,1,5,256] slots=SLIDING_PRODUCERS ; cache_fl_{k,v} [3,1,1,5,512] slots=FULL_PRODUCERS ; col4=0",
        "sliding_producers": SLIDING_PRODUCERS, "full_producers": FULL_PRODUCERS,
        "tensors": {n: {"shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", "")} for n, t in tensors.items()},
        "pass_threshold": {"last_hidden_max_abs_le": 1e-2, "mean_abs_le": 1e-4, "argmax": "== HF"},
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print("wrote", OUT_MANIFEST, "\nP5.7.7 decode-3 oracle OK.")


if __name__ == "__main__":
    main()
