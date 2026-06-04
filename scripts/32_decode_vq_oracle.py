"""Q4 — Oracle decode e2e avec FAKE-QUANT V (MSE V-only) -> last_hidden/argmax == HF-V-quant.

Adapté de 43_p5_7_7_decode3_oracle.py : MÊME moteur decode (35 couches, 1 token, 2 caches multi-slots,
YOCO), mais on remplace chaque V au point v_norm (post-v_norm, AVANT transpose/cache) par son fake-quant
MSE V-only :  V_hat = MSE.dequant(MSE.quant(V))  avec  TurboQuantMSE(head_dim, 4, rotation="hadamard").

Le hook forward sur les `v_norm` des producers (cf scripts/test_kv_quant_generation.py, classe de hook V)
quantifie TOUT le V cache : (1) le V du prefill (cache producers 0..14) ET (2) le V du token decode.
Cohérence avec le runner ZML gemma4_decode_vq.zig qui insère quantizeV au même point (v_norm) côté producers.
K reste fp32 (V-only). Les constantes MSE sont déterministes -> == turboquant_constants.safetensors (Task 0).

La fixture exportée (/data/gemma4-zml-probe/decode_vq.safetensors) reprend TOUS les tenseurs de la fixture
decode existante (caches V quantifiés, embed, pos, mask, cos/sin...) + codebook_256/512 + hadamard_256/512
(copie de turboquant_constants.safetensors) + références last_hidden/logits/argmax calculées avec V-quant.

CLI : python3 scripts/32_decode_vq_oracle.py   (lance sur la 3090, venv gemma4-probe).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/data/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel, Gemma4TextRotaryEmbedding

# turboquant.py est à la racine du repo sur la 3090
sys.path.insert(0, "/data/gemma4-zml-probe")
from turboquant import TurboQuantMSE  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights" / "model.safetensors"
CONSTANTS = ROOT / "turboquant_constants.safetensors"
OUT_FIXTURE = ROOT / "decode_vq.safetensors"
OUT_MANIFEST = ROOT / "decode_vq_manifest.json"
SEQ_LEN = 4
INPUT_IDS = [2, 105, 2048, 4095]
PREFIX = "model.language_model."
EMBPTL = "embed_tokens_per_layer.weight"

SLIDING_PRODUCERS = [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13]  # 12, hd=256
FULL_PRODUCERS = [4, 9, 14]                                   # 3,  hd=512
HD_S, HD_F = 256, 512
N_KV = 1  # gemma-4-E2B : num_key_value_heads = 1


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


# --------------------------------------------------------------------------
# Fake-quant V : un quantizer MSE par head_dim (déterministe == constantes Task 0)
# --------------------------------------------------------------------------
_QUANTIZERS = {d: TurboQuantMSE(d, 4, device="cpu", rotation="hadamard") for d in (HD_S, HD_F)}


def fake_quant_v(x):
    """x:[...,d] (sortie v_norm) -> dequant(quant(x)) en fp32, même shape."""
    dt = x.dtype
    xf = x.to(torch.float32)
    q = _QUANTIZERS[x.shape[-1]]
    xhat = q.dequant(q.quant(xf))
    return xhat.to(dt)


def install_v_hooks(model):
    """Forward hooks sur v_norm des producers : remplace l'output par V_hat (V-only fake-quant).

    v_norm output = [B, seq, n_kv, head_dim] (avant transpose). n_kv=1 -> dim -2.
    Reproduit le point d'insertion ZML (entre v_norm et transpose, branche producer).
    """
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
            t = out[0] if isinstance(out, tuple) else out
            if torch.is_tensor(t) and t.dim() == 4 and t.shape[-2] == N_KV:
                return fake_quant_v(t)
            return None  # ne change rien

        handles.append(vnorm.register_forward_hook(hook))
    return handles


def pad_cache(t):
    """[1,1,4,hd] -> [1,1,5,hd] col 4 = 0 (fp32)."""
    hd = t.shape[-1]
    out = torch.zeros(1, 1, SEQ_LEN + 1, hd, dtype=torch.float32)
    out[:, :, :SEQ_LEN, :] = t.float()
    return out


def load_constants():
    """codebook_256/512 + hadamard_256/512 depuis turboquant_constants.safetensors (Task 0)."""
    assert CONSTANTS.exists(), CONSTANTS
    consts = {}
    with safe_open(str(CONSTANTS), framework="pt") as s:
        for k in s.keys():
            consts[k] = s.get_tensor(k).to(torch.float32).contiguous()
    return consts


def main() -> None:
    assert WEIGHTS.exists(), WEIGHTS
    torch.manual_seed(1337)
    cfg = AutoConfig.from_pretrained("google/gemma-4-E2B-it")
    tc = getattr(cfg, "text_config", cfg)
    softcap = float(getattr(tc, "final_logit_softcapping", 30.0))

    print("Build hybride (streaming) ...")
    model = build_hybrid_model(tc)

    print("Install V hooks (fake-quant MSE V-only au point v_norm) ...")
    install_v_hooks(model)

    input_ids = torch.tensor(INPUT_IDS, dtype=torch.long).view(1, SEQ_LEN)
    print("Prefill use_cache=True (V quantifié dans le cache) ...")
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

    # --- extraire les 15 caches producers depuis pkv.layers[i] (V déjà quantifié via hook) ---
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

    # --- pré-gather embed + embptl du nouveau token (entrée moteur ZML) ---
    ids = torch.tensor([new_token], dtype=torch.long)
    embed_slice = model.embed_tokens.weight[ids].to(torch.bfloat16).contiguous().view(1, 1, 1536)
    embptl_slice = model.embed_tokens_per_layer.weight[ids].contiguous().view(1, 1, -1)  # bf16 [1,1,8960]

    # --- cos/sin full à pos p=4 (sliding rope calculée en ZML) ---
    rot = Gemma4TextRotaryEmbedding(tc)
    pos = torch.tensor([[SEQ_LEN]])  # [1,1] = position 4
    cos_f, sin_f = rot(last_hidden_pref, pos, layer_type="full_attention")  # [1,1,512]
    assert tuple(cos_f.shape) == (1, 1, HD_F), cos_f.shape

    mask_decode = torch.zeros(1, 1, 1, SEQ_LEN + 1, dtype=torch.float32)

    # --- decode 1 step (référence e2e, V quantifié) ---
    print("Decode 1 step (past_key_values=pkv, V du token quantifié via hook) ...")
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
        "last_hidden": last_hidden,                                 # référence e2e V-quant
        "logits": logits.contiguous(),                             # [1,262144] softcappés
        "argmax": torch.tensor([argmax_dec], dtype=torch.int32),    # token suivant HF-V-quant
        "decode_token": torch.tensor([new_token], dtype=torch.int32),
    }

    # --- constantes MSE (== Task 0) embarquées pour le runner ZML ---
    consts = load_constants()
    for k, t in consts.items():
        tensors[k] = t  # codebook_256/512, hadamard_256/512
    print(f"  constantes embarquées : {sorted(consts.keys())}")

    for k, t in tensors.items():
        assert not torch.isnan(t.float()).any(), f"NaN dans {k}"

    print("=" * 72)
    print("Q4 — oracle decode e2e V-QUANT (moteur decode 35 couches)")
    print(f"  token decode = {new_token} ; pos = {SEQ_LEN} ; token suivant (argmax) = {argmax_dec}")
    print(f"  last_hidden[0,0,:6] = {[round(v,7) for v in last_hidden[0,0,:6].tolist()]}")
    print(f"  stats last_hidden: mean={last_hidden.mean():.4e} std={last_hidden.std():.4e}")

    OUT_FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(OUT_FIXTURE))
    print("wrote", OUT_FIXTURE)

    manifest = {
        "source": "Q4 oracle decode e2e V-QUANT (moteur decode 35 couches, 1 token, fake-quant MSE V-only)",
        "vquant": "TurboQuantMSE(head_dim, 4, rotation=hadamard) hook sur v_norm producers ; K fp32 ; "
                  "constantes == turboquant_constants.safetensors (Task 0)",
        "input_ids": INPUT_IDS, "decode_token": new_token, "next_token_argmax": argmax_dec, "pos_idx": SEQ_LEN,
        "caches": "cache_sl_{k,v} [12,1,1,5,256] (V quantifié) ; cache_fl_{k,v} [3,1,1,5,512] (V quantifié) ; col4=0",
        "sliding_producers": SLIDING_PRODUCERS, "full_producers": FULL_PRODUCERS,
        "constants": sorted(consts.keys()),
        "tensors": {n: {"shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", "")} for n, t in tensors.items()},
        "pass_threshold": {"last_hidden_max_abs_le": 1e-2, "mean_abs_le": 1e-4, "argmax": "== HF-V-quant"},
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print("wrote", OUT_MANIFEST, "\nQ4 oracle V-quant OK.")


if __name__ == "__main__":
    main()
