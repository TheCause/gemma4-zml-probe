"""P5.7.7 decode-2 — Oracle pilote decode FULL (writer 14 full × reader 19 full).

Symétrique à decode-1 (script 40) mais sur les couches FULL : head_dim 512, RoPE manuelle partielle
(partial_rotary 0.25, theta 1e6, proportional) — cos/sin fournis par HF (512-wide), appliqués en ZML
via manualRope (mécanisme prouvé prefill P5.7.4). attention_k_eq_v=False -> V séparé (comme decode-1).

shared_kv_states["full_attention"] = cache writer 14 complet (prefill [0..3], decode [0..4]).
Fixture : fixtures/p5_7_7_decode2.safetensors
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights" / "model.safetensors"
OUT_FIXTURE = ROOT / "fixtures" / "p5_7_7_decode2.safetensors"
OUT_MANIFEST = ROOT / "fixtures" / "p5_7_7_decode2_manifest.json"
SEQ_LEN = 4
INPUT_IDS = [2, 105, 2048, 4095]
PREFIX = "model.language_model."
EMBPTL = "embed_tokens_per_layer.weight"
FULL_WRITER = 14
FULL_READER = 19
HD_FULL = 512


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


def main() -> None:
    assert WEIGHTS.exists(), WEIGHTS
    torch.manual_seed(1337)
    cfg = AutoConfig.from_pretrained("google/gemma-4-E2B-it")
    tc = getattr(cfg, "text_config", cfg)
    assert tc.layer_types[FULL_WRITER] == "full_attention", tc.layer_types[FULL_WRITER]
    assert tc.layer_types[FULL_READER] == "full_attention", tc.layer_types[FULL_READER]
    assert getattr(tc, "attention_k_eq_v", False) is False, "attention_k_eq_v True -> V=K, à gérer"
    softcap = float(getattr(tc, "final_logit_softcapping", 30.0))

    print("Build hybride (streaming) ...")
    model = build_hybrid_model(tc)

    input_ids = torch.tensor(INPUT_IDS, dtype=torch.long).view(1, SEQ_LEN)
    print("Prefill use_cache=True + return_shared_kv_states ...")
    with torch.no_grad():
        out_pref = model(input_ids=input_ids, use_cache=True, return_shared_kv_states=True)
    pkv = out_pref.past_key_values
    assert pkv is not None
    k_pref, v_pref = out_pref.shared_kv_states["full_attention"]    # [1,1,4,512]
    assert tuple(k_pref.shape) == (1, 1, SEQ_LEN, HD_FULL), k_pref.shape

    last_hidden = out_pref.last_hidden_state.to(torch.float32)
    lm_w = model.embed_tokens.weight.to(torch.float32)
    logits_last = last_hidden[:, -1, :] @ lm_w.t()
    logits_last = softcap * torch.tanh(logits_last / softcap)
    new_token = int(logits_last.argmax(dim=-1).item())
    print(f"  nouveau token (decode p={SEQ_LEN}) = {new_token}")

    cap: dict[str, torch.Tensor] = {}

    def pre_hook(tag):
        def _h(module, args, kwargs):
            hs = kwargs.get("hidden_states", args[0] if args else None)
            cap[f"attn_in_{tag}"] = hs.detach().to(torch.float32).cpu().clone()
            pe = kwargs.get("position_embeddings")
            if pe is not None:
                cap[f"cos_{tag}"] = pe[0].detach().to(torch.float32).cpu().clone()
                cap[f"sin_{tag}"] = pe[1].detach().to(torch.float32).cpu().clone()
            am = kwargs.get("attention_mask")
            cap[f"mask_{tag}"] = None if am is None else am.detach().to(torch.float32).cpu().clone()
        return _h

    def out_hook(tag):
        def _h(module, args, kwargs, output):
            o = output[0] if isinstance(output, tuple) else output
            cap[f"attn_out_{tag}"] = o.detach().to(torch.float32).cpu().clone()
        return _h

    handles = []
    for idx, tag in [(FULL_WRITER, "14"), (FULL_READER, "19")]:
        sa = model.layers[idx].self_attn
        handles.append(sa.register_forward_pre_hook(pre_hook(tag), with_kwargs=True))
        handles.append(sa.register_forward_hook(out_hook(tag), with_kwargs=True))

    print("Decode 1 step (past_key_values=pkv) ...")
    dec_ids = torch.tensor([[new_token]], dtype=torch.long)
    with torch.no_grad():
        out_dec = model(input_ids=dec_ids, past_key_values=pkv, use_cache=True, return_shared_kv_states=True)
    for h in handles:
        h.remove()
    k_after, v_after = out_dec.shared_kv_states["full_attention"]   # [1,1,5,512]
    assert tuple(k_after.shape) == (1, 1, SEQ_LEN + 1, HD_FULL), k_after.shape

    drift03 = (k_after[:, :, :SEQ_LEN, :].float() - k_pref.float()).abs().max().item()
    print(f"  sanity cache[0..3] decode vs prefill : max|dK| = {drift03:.3e}")
    k14_new = k_after[:, :, SEQ_LEN:SEQ_LEN + 1, :].contiguous()
    v14_new = v_after[:, :, SEQ_LEN:SEQ_LEN + 1, :].contiguous()

    mask_hf = cap.get("mask_14")
    mask_decode = torch.zeros(1, 1, 1, SEQ_LEN + 1, dtype=torch.float32)
    if mask_hf is not None:
        finite = mask_hf[torch.isfinite(mask_hf)]
        if finite.numel():
            assert finite.abs().max().item() == 0.0, f"mask HF non nul: {finite.abs().max()}"
        print(f"  mask HF shape {tuple(mask_hf.shape)} — finies == 0 (full causal q=1)")
    else:
        print("  mask HF = None (sdpa is_causal) -> mask ZML zeros [1,1,1,5]")

    k_pref_pad = torch.zeros(1, 1, SEQ_LEN + 1, HD_FULL, dtype=torch.float32)
    v_pref_pad = torch.zeros(1, 1, SEQ_LEN + 1, HD_FULL, dtype=torch.float32)
    k_pref_pad[:, :, :SEQ_LEN, :] = k_pref.float()
    v_pref_pad[:, :, :SEQ_LEN, :] = v_pref.float()

    def f32c(t):
        return t.detach().to(torch.float32).contiguous().clone()

    tensors = {
        "token_id": torch.tensor([new_token], dtype=torch.int32),
        "pos_idx": torch.tensor([SEQ_LEN], dtype=torch.int32),
        "attn_in_14": f32c(cap["attn_in_14"]).view(1, 1, 1536),
        "attn_in_19": f32c(cap["attn_in_19"]).view(1, 1, 1536),
        "cos_full": f32c(cap["cos_14"]),                              # [1,1,512] pos 4 (full rope, appliqué par manualRope)
        "sin_full": f32c(cap["sin_14"]),
        "mask_decode": mask_decode.contiguous(),
        "cache14_k_prefill": k_pref_pad.contiguous(),                 # [1,1,5,512] col4=0
        "cache14_v_prefill": v_pref_pad.contiguous(),
        "cache14_k_after": f32c(k_after).contiguous(),
        "cache14_v_after": f32c(v_after).contiguous(),
        "k14_new": f32c(k14_new),                                     # [1,1,1,512]
        "v14_new": f32c(v14_new),
        "attn_out_14": f32c(cap["attn_out_14"]).view(1, 1, 1536),
        "attn_out_19": f32c(cap["attn_out_19"]).view(1, 1, 1536),
    }
    for k, t in tensors.items():
        assert not torch.isnan(t.float()).any(), f"NaN dans {k}"

    print("=" * 72)
    print("P5.7.7 decode-2 — oracle pilote FULL (writer 14 × reader 19)")
    print(f"  token decode = {new_token} ; pos = {SEQ_LEN} ; head_dim full = {HD_FULL}")
    print(f"  attn_out_14[0,0,:6] = {[round(v,7) for v in tensors['attn_out_14'][0,0,:6].tolist()]}")
    print(f"  attn_out_19[0,0,:6] = {[round(v,7) for v in tensors['attn_out_19'][0,0,:6].tolist()]}")

    OUT_FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(OUT_FIXTURE))
    print("wrote", OUT_FIXTURE)

    manifest = {
        "source": "P5.7.7 decode-2 oracle pilote FULL (writer 14 × reader 19)",
        "design": "docs/P5_7_7_decode.md",
        "input_ids": INPUT_IDS, "decode_token": new_token, "pos_idx": SEQ_LEN, "head_dim_full": HD_FULL,
        "rope_full": "partial_rotary 0.25, theta 1e6, proportional ; cos/sin HF 512-wide appliqués par manualRope (cf P5.7.4)",
        "attention_k_eq_v": False,
        "tensors": {n: {"shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", "")} for n, t in tensors.items()},
        "pass_threshold": {"max_abs_le": 1e-2, "mean_abs_le": 1e-4},
        "checks": ["cache14_*_after == HF", "k14_new/v14_new == HF", "attn_out_14 (writer full)", "attn_out_19 (reader full YOCO)"],
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print("wrote", OUT_MANIFEST, "\nP5.7.7 decode-2 oracle OK.")


if __name__ == "__main__":
    main()
