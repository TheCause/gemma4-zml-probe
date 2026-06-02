"""P5.7.7 decode-1 — Oracle pilote decode incrémental (writer 13 sliding × reader 15 sliding).

Isole la mécanique decode neuve : cache KV qui grandit d'une colonne, append du token p, pos_idx, mask
S=1, reuse reader YOCO du cache grandi. On compare la sortie du module `self_attn` (post o_proj), pas la
couche entière (input_ln / MLP / PLE déjà validés P5.3).

Méthode (discipline projet : ne rien supposer, oracle = modèle réel + hooks) :
  1. build hybride streaming (réutilise la logique de 39 : corps fp32, embed_tokens_per_layer bf16).
  2. prefill `use_cache=True, return_shared_kv_states=True` -> past_key_values + shared (writer 13 = [0..3]).
  3. nouveau token = argmax du logit prefill de la dernière position (softcap 30, lm_head tied).
  4. decode 1 step (past_key_values=pkv) avec hooks sur layers[13|15].self_attn -> entrées + sorties.
     shared_kv_states["sliding_attention"] du decode = cache writer 13 grandi [0..4] (pas de DynamicCache).

Fixture fixtures/p5_7_7_decode1.safetensors : voir docs/P5_7_7_decode.md.
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
OUT_FIXTURE = ROOT / "fixtures" / "p5_7_7_decode1.safetensors"
OUT_MANIFEST = ROOT / "fixtures" / "p5_7_7_decode1_manifest.json"
SEQ_LEN = 4
INPUT_IDS = [2, 105, 2048, 4095]   # identique aux oracles 39 (prompt de référence)
PREFIX = "model.language_model."
EMBPTL = "embed_tokens_per_layer.weight"
SLIDING_WRITER = 13
SLIDING_READER = 15
HD_SLIDING = 256


def set_submodule_attr(root, dotted, value):
    parts = dotted.split(".")
    obj = root
    for p in parts[:-1]:
        obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
    setattr(obj, parts[-1], value)


def build_hybrid_model(tc):
    """Build + streaming-load hybride (copie fidèle de 39 : corps fp32, embed_tokens_per_layer bf16,
    layer_scalar buffers fp32, rotary fp32, embed_scale forcé fp32)."""
    torch.set_default_dtype(torch.bfloat16)
    model = Gemma4TextModel(tc)
    model.train(False)
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())   # layer_scalar = registered_buffer, PAS un param
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
    assert tc.layer_types[SLIDING_WRITER] == "sliding_attention", tc.layer_types[SLIDING_WRITER]
    assert tc.layer_types[SLIDING_READER] == "sliding_attention", tc.layer_types[SLIDING_READER]
    softcap = float(getattr(tc, "final_logit_softcapping", 30.0))

    print("Build hybride (streaming) ...")
    model = build_hybrid_model(tc)

    # ---- 1) PREFILL : use_cache + shared KV (writer 13 sliding = [0..3]) ----
    input_ids = torch.tensor(INPUT_IDS, dtype=torch.long).view(1, SEQ_LEN)
    print("Prefill use_cache=True + return_shared_kv_states ...")
    with torch.no_grad():
        out_pref = model(input_ids=input_ids, use_cache=True, return_shared_kv_states=True)
    pkv = out_pref.past_key_values
    assert pkv is not None, "past_key_values None — use_cache non honoré"
    shared_pref = out_pref.shared_kv_states
    k_pref, v_pref = shared_pref["sliding_attention"]    # [1,1,4,256] post-rope K / post-norm V
    assert tuple(k_pref.shape) == (1, 1, SEQ_LEN, HD_SLIDING), k_pref.shape

    # nouveau token = argmax du logit prefill (dernière position), lm_head tied + softcap 30
    last_hidden = out_pref.last_hidden_state.to(torch.float32)         # [1,4,1536]
    lm_w = model.embed_tokens.weight.to(torch.float32)
    logits_last = last_hidden[:, -1, :] @ lm_w.t()                    # [1,vocab]
    logits_last = softcap * torch.tanh(logits_last / softcap)
    new_token = int(logits_last.argmax(dim=-1).item())
    print(f"  nouveau token (decode p={SEQ_LEN}) = {new_token}")

    # ---- 2) Hooks sur self_attn de 13 (writer) et 15 (reader) ----
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
    for idx, tag in [(SLIDING_WRITER, "13"), (SLIDING_READER, "15")]:
        sa = model.layers[idx].self_attn
        handles.append(sa.register_forward_pre_hook(pre_hook(tag), with_kwargs=True))
        handles.append(sa.register_forward_hook(out_hook(tag), with_kwargs=True))

    # ---- 3) DECODE 1 step ----
    print("Decode 1 step (past_key_values=pkv) ...")
    dec_ids = torch.tensor([[new_token]], dtype=torch.long)
    with torch.no_grad():
        out_dec = model(input_ids=dec_ids, past_key_values=pkv, use_cache=True, return_shared_kv_states=True)
    for h in handles:
        h.remove()
    shared_dec = out_dec.shared_kv_states
    k_after, v_after = shared_dec["sliding_attention"]    # [1,1,5,256] cache writer 13 GRANDI [0..4]
    assert tuple(k_after.shape) == (1, 1, SEQ_LEN + 1, HD_SLIDING), k_after.shape

    # Cache grandi : col 4 = token p ; cols 0..3 == prefill (sanity : le decode n'a pas touché 0..3).
    drift03 = (k_after[:, :, :SEQ_LEN, :].float() - k_pref.float()).abs().max().item()
    print(f"  sanity cache[0..3] decode vs prefill : max|dK| = {drift03:.3e} (attendu ~0)")
    k13_new = k_after[:, :, SEQ_LEN:SEQ_LEN + 1, :].contiguous()      # [1,1,1,256]
    v13_new = v_after[:, :, SEQ_LEN:SEQ_LEN + 1, :].contiguous()

    # mask decode capturé (sliding, q=1) : à p<512 tout visible -> 0. On exporte un mask propre [1,1,1,5]=0
    # (ce que le runner ZML utilise) et on vérifie que le mask HF, là où il est fini, vaut 0.
    mask_hf = cap.get("mask_13")
    mask_decode = torch.zeros(1, 1, 1, SEQ_LEN + 1, dtype=torch.float32)
    if mask_hf is not None:
        finite = mask_hf[torch.isfinite(mask_hf)]
        if finite.numel():
            assert finite.abs().max().item() == 0.0, f"mask HF non nul aux positions finies: {finite.abs().max()}"
        print(f"  mask HF shape {tuple(mask_hf.shape)} — positions finies == 0 (sliding p<512 = causal tout-visible)")
    else:
        print("  mask HF = None (sdpa is_causal) -> mask ZML = zeros [1,1,1,5]")

    # cache prefill padded à [1,1,5,256] (cols 0..3 réelles, col 4 = 0) = buffer cache initial côté ZML
    k_pref_pad = torch.zeros(1, 1, SEQ_LEN + 1, HD_SLIDING, dtype=torch.float32)
    v_pref_pad = torch.zeros(1, 1, SEQ_LEN + 1, HD_SLIDING, dtype=torch.float32)
    k_pref_pad[:, :, :SEQ_LEN, :] = k_pref.float()
    v_pref_pad[:, :, :SEQ_LEN, :] = v_pref.float()

    def f32c(t):
        return t.detach().to(torch.float32).contiguous().clone()

    tensors = {
        "token_id": torch.tensor([new_token], dtype=torch.int32),
        "pos_idx": torch.tensor([SEQ_LEN], dtype=torch.int32),         # p = 4
        # entrées (self_attn = hidden POST input_layernorm)
        "attn_in_13": f32c(cap["attn_in_13"]).view(1, 1, 1536),
        "attn_in_15": f32c(cap["attn_in_15"]).view(1, 1, 1536),
        "cos_sliding": f32c(cap["cos_13"]),                            # [1,1,256] pos 4 (référence)
        "sin_sliding": f32c(cap["sin_13"]),
        "mask_decode": mask_decode.contiguous(),                       # [1,1,1,5]
        # cache initial (prefill [0..3] padded col4=0) -> buffer à scatter côté ZML
        "cache13_k_prefill": k_pref_pad.contiguous(),
        "cache13_v_prefill": v_pref_pad.contiguous(),
        # références
        "cache13_k_after": f32c(k_after).contiguous(),                 # [1,1,5,256] post-append
        "cache13_v_after": f32c(v_after).contiguous(),
        "k13_new": f32c(k13_new),                                      # [1,1,1,256] token p
        "v13_new": f32c(v13_new),
        "attn_out_13": f32c(cap["attn_out_13"]).view(1, 1, 1536),      # post o_proj
        "attn_out_15": f32c(cap["attn_out_15"]).view(1, 1, 1536),
    }
    for k, t in tensors.items():
        assert not torch.isnan(t.float()).any(), f"NaN dans {k}"

    print("=" * 72)
    print("P5.7.7 decode-1 — oracle pilote (writer 13 × reader 15, sliding)")
    print(f"  input_ids prefill = {INPUT_IDS} ; token decode = {new_token} ; pos = {SEQ_LEN}")
    print(f"  attn_out_13[0,0,:6] = {[round(v,7) for v in tensors['attn_out_13'][0,0,:6].tolist()]}")
    print(f"  attn_out_15[0,0,:6] = {[round(v,7) for v in tensors['attn_out_15'][0,0,:6].tolist()]}")
    print(f"  k13_new[0,0,0,:6]   = {[round(v,7) for v in tensors['k13_new'][0,0,0,:6].tolist()]}")

    OUT_FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(OUT_FIXTURE))
    print("wrote", OUT_FIXTURE)

    manifest = {
        "source": "P5.7.7 decode-1 oracle pilote (writer 13 sliding × reader 15 sliding)",
        "design": "docs/P5_7_7_decode.md",
        "build": "hybride streaming (corps fp32, embed_tokens_per_layer bf16) — cf scripts/39",
        "input_ids": INPUT_IDS, "decode_token": new_token, "pos_idx": SEQ_LEN,
        "isolation": "self_attn (post o_proj) — input_ln/MLP/PLE/layer_scalar hors scope (validés P5.3)",
        "tensors": {n: {"shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", "")} for n, t in tensors.items()},
        "pass_threshold": {"max_abs_le": 1e-2, "mean_abs_le": 1e-4},
        "checks": [
            "cache13_*_after == HF (scatterSlices append col 4)",
            "k13_new/v13_new == HF (K/V du token p)",
            "attn_out_13 == HF (writer : append + attention sur cache grandi)",
            "attn_out_15 == HF (reader : reuse YOCO du cache grandi)",
        ],
        "note": "shared_kv_states['sliding_attention'] = cache writer 13 complet (prefill [0..3], decode [0..4]). cache13_*_prefill padded col4=0.",
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print("wrote", OUT_MANIFEST, "\nP5.7.7 decode-1 oracle OK.")


if __name__ == "__main__":
    main()
