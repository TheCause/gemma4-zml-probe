"""Q5 Volet B — Oracle de GÉNÉRATION V-QUANT (boucle decode N tokens) -> séquence == HF-V-quant.

Jumeau de 44_p5_7_8_gen_oracle.py (boucle decode N tokens, cache threadé), mais V est quantifié
(fake-quant MSE V-only) au point v_norm, EXACTEMENT comme le fait gemma4_decode_vq.zig (Q4) et comme
le fera gemma4_gen_vq.zig (Q5 volet B). K reste fp32.

Le hook forward sur les `v_norm` des producers (cf 32_decode_vq_oracle.py) quantifie :
  (1) le V du prefill (caches producers 0..14) ET (2) le V de chaque token décodé.
La séquence greedy HF est donc celle SOUS V-quant (HF-V-quant). C'est cette séquence que le moteur
ZML-V-quant doit reproduire (argmax ZML[k] == expected[k]).

Fixture : decode_vq_gen.safetensors (mêmes tenseurs que p5_7_8_gen + codebook_256/512 + hadamard_256/512).
Le runner gemma4_gen_vq.zig requantifie le V des tokens fed (cohérence avec l'oracle).

CLI : python3 scripts/45_gen_vq_oracle.py   (3090, venv gemma4-probe).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/data/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel, Gemma4TextRotaryEmbedding

sys.path.insert(0, "/data/gemma4-zml-probe")
from turboquant import TurboQuantMSE  # noqa: E402

ROOT = Path("/data/gemma4-zml-probe")
WEIGHTS = ROOT / "weights" / "model.safetensors"
CONSTANTS = ROOT / "turboquant_constants.safetensors"
OUT_FIXTURE = ROOT / "decode_vq_gen.safetensors"
OUT_MANIFEST = ROOT / "decode_vq_gen_manifest.json"
SEQ_LEN = 4
N_DECODE = 4                 # nb de pas de génération (= decode4 NUM_STEPS=4)
KMAX = SEQ_LEN + N_DECODE    # 8
INPUT_IDS = [2, 105, 2048, 4095]
PREFIX = "model.language_model."
EMBPTL = "embed_tokens_per_layer.weight"

SLIDING_PRODUCERS = [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13]
FULL_PRODUCERS = [4, 9, 14]
HD_S, HD_F = 256, 512
N_KV = 1


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


# --- fake-quant V (MSE V-only) au point v_norm, == constantes Task 0 ---
_QUANTIZERS = {d: TurboQuantMSE(d, 4, device="cpu", rotation="hadamard") for d in (HD_S, HD_F)}


def fake_quant_v(x):
    dt = x.dtype
    xf = x.to(torch.float32)
    q = _QUANTIZERS[x.shape[-1]]
    return q.dequant(q.quant(xf)).to(dt)


def install_v_hooks(model):
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
            return None

        handles.append(vnorm.register_forward_hook(hook))
    return handles


def pad_cache(t):
    hd = t.shape[-1]
    out = torch.zeros(1, 1, KMAX, hd, dtype=torch.float32)
    out[:, :, :SEQ_LEN, :] = t.float()
    return out


def load_constants():
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
    lm_w = model.embed_tokens.weight.to(torch.float32)

    def next_token(out):
        lh = out.last_hidden_state.to(torch.float32)[:, -1, :]
        lg = softcap * torch.tanh((lh @ lm_w.t()) / softcap)
        return int(lg.argmax(dim=-1).item())

    input_ids = torch.tensor(INPUT_IDS, dtype=torch.long).view(1, SEQ_LEN)
    print("Prefill use_cache=True (V quantifié dans le cache) ...")
    with torch.no_grad():
        out_pref = model(input_ids=input_ids, use_cache=True)
    pkv = out_pref.past_key_values
    s0 = next_token(out_pref)
    print(f"  s0 (prefill argmax, à feed pos 4) = {s0}")

    # extraire les caches prefill [0..14] (AVANT de muter pkv par la boucle) -> paddés kmax. V quantifié.
    cache_sl_k = torch.zeros(len(SLIDING_PRODUCERS), 1, 1, KMAX, HD_S, dtype=torch.float32)
    cache_sl_v = torch.zeros_like(cache_sl_k)
    cache_fl_k = torch.zeros(len(FULL_PRODUCERS), 1, 1, KMAX, HD_F, dtype=torch.float32)
    cache_fl_v = torch.zeros_like(cache_fl_k)
    for slot, i in enumerate(SLIDING_PRODUCERS):
        cache_sl_k[slot] = pad_cache(pkv.layers[i].keys)
        cache_sl_v[slot] = pad_cache(pkv.layers[i].values)
    for slot, i in enumerate(FULL_PRODUCERS):
        cache_fl_k[slot] = pad_cache(pkv.layers[i].keys)
        cache_fl_v[slot] = pad_cache(pkv.layers[i].values)

    # greedy manuel V-quant : feed s_k (pos 4+k) -> s_{k+1}. Séquence s0..sN.
    seq = [s0]
    cur = s0
    for k in range(N_DECODE):
        with torch.no_grad():
            out = model(input_ids=torch.tensor([[cur]], dtype=torch.long), past_key_values=pkv, use_cache=True)
        nxt = next_token(out)
        seq.append(nxt)
        cur = nxt
    print(f"  séquence HF-V-quant greedy (s0..s{N_DECODE}) = {seq}")
    fed = seq[:N_DECODE]
    expected = seq[1:N_DECODE + 1]

    rot = Gemma4TextRotaryEmbedding(tc)
    embeds = torch.zeros(N_DECODE, 1, 1, 1536, dtype=torch.bfloat16)
    embptls = torch.zeros(N_DECODE, 1, 1, 8960, dtype=torch.bfloat16)
    cos_full = torch.zeros(N_DECODE, 1, 1, HD_F, dtype=torch.float32)
    sin_full = torch.zeros(N_DECODE, 1, 1, HD_F, dtype=torch.float32)
    positions = torch.zeros(N_DECODE, dtype=torch.int32)
    masks = torch.zeros(N_DECODE, 1, 1, 1, KMAX, dtype=torch.float32)
    MIN = torch.finfo(torch.float32).min
    dummy = torch.zeros(1, 1, 1536, dtype=torch.float32)
    for k in range(N_DECODE):
        tid = torch.tensor([fed[k]], dtype=torch.long)
        embeds[k] = model.embed_tokens.weight[tid].to(torch.bfloat16).view(1, 1, 1536)
        embptls[k] = model.embed_tokens_per_layer.weight[tid].view(1, 1, 8960)
        p = SEQ_LEN + k
        positions[k] = p
        cf, sf = rot(dummy, torch.tensor([[p]]), layer_type="full_attention")
        cos_full[k] = cf
        sin_full[k] = sf
        for j in range(KMAX):
            if j > p:
                masks[k, 0, 0, 0, j] = MIN

    tensors = {
        "embeds": embeds.contiguous(), "embptls": embptls.contiguous(),
        "cos_full": cos_full.contiguous(), "sin_full": sin_full.contiguous(),
        "positions": positions.contiguous(), "masks": masks.contiguous(),
        "cache_sl_k": cache_sl_k.contiguous(), "cache_sl_v": cache_sl_v.contiguous(),
        "cache_fl_k": cache_fl_k.contiguous(), "cache_fl_v": cache_fl_v.contiguous(),
        "expected": torch.tensor(expected, dtype=torch.int32),
        "fed": torch.tensor(fed, dtype=torch.int32),
    }
    # constantes MSE (== Task 0) pour le runner gemma4_gen_vq.zig
    consts = load_constants()
    for k, t in consts.items():
        tensors[k] = t
    print(f"  constantes embarquées : {sorted(consts.keys())}")

    for k, t in tensors.items():
        assert not torch.isnan(t.float()).any(), f"NaN dans {k}"

    print("=" * 72)
    print("Q5 Volet B — oracle génération V-QUANT")
    print(f"  prompt={INPUT_IDS} ; N_DECODE={N_DECODE} ; kmax={KMAX}")
    print(f"  fed (s0..s{N_DECODE-1})      = {fed}")
    print(f"  expected (s1..s{N_DECODE}) = {expected}")
    print(f"  positions = {positions.tolist()}")

    OUT_FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(OUT_FIXTURE))
    print("wrote", OUT_FIXTURE)

    manifest = {
        "source": "Q5 Volet B oracle génération V-QUANT (boucle decode N tokens, greedy == HF-V-quant)",
        "vquant": "TurboQuantMSE(head_dim, 4, rotation=hadamard) hook sur v_norm producers ; K fp32 ; "
                  "constantes == turboquant_constants.safetensors (Task 0)",
        "prompt": INPUT_IDS, "n_decode": N_DECODE, "kmax": KMAX,
        "sequence_hf_vquant": seq, "fed": fed, "expected": expected,
        "sliding_producers": SLIDING_PRODUCERS, "full_producers": FULL_PRODUCERS,
        "constants": sorted(consts.keys()),
        "tensors": {n: {"shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", "")} for n, t in tensors.items()},
        "pass": "argmax ZML[k] == expected[k] pour tout k (séquence == HF-V-quant greedy)",
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print("wrote", OUT_MANIFEST, "\nQ5 Volet B oracle V-quant OK.")


if __name__ == "__main__":
    main()
