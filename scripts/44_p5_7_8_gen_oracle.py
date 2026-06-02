"""P5.7.8 — Oracle de GÉNÉRATION (boucle decode N tokens) -> séquence == HF greedy.

Le tour d'honneur du decode : le moteur ZML génère N tokens en boucle (en threadant le cache grandi de
step en step) et doit produire la MÊME séquence que HF en greedy.

Approche teacher-forcing vérifié (la séquence greedy HF est déterministe et connue à l'avance) :
  - prefill use_cache -> caches producers [0..3] (paddés à kmax=4+N).
  - greedy manuel : feed s_k -> argmax s_{k+1}, k=0..N. Séquence s0..sN.
  - on pré-gather embed/embptl des tokens FED (s0..s_{N-1}) -> évite de charger embed_tokens_per_layer
    (4.7 Go) côté ZML.
  - cos/sin full par position 4..4+N-1.
Le ZML boucle : à chaque step feed s_k, et on vérifie que son argmax == s_{k+1} (séquence == HF).

Fixture : fixtures/p5_7_8_gen.safetensors
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel, Gemma4TextRotaryEmbedding

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights" / "model.safetensors"
OUT_FIXTURE = ROOT / "fixtures" / "p5_7_8_gen.safetensors"
OUT_MANIFEST = ROOT / "fixtures" / "p5_7_8_gen_manifest.json"
SEQ_LEN = 4
N_DECODE = 4                 # nb de pas de génération
KMAX = SEQ_LEN + N_DECODE    # 8
INPUT_IDS = [2, 105, 2048, 4095]
PREFIX = "model.language_model."
EMBPTL = "embed_tokens_per_layer.weight"

SLIDING_PRODUCERS = [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13]
FULL_PRODUCERS = [4, 9, 14]
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
    hd = t.shape[-1]
    out = torch.zeros(1, 1, KMAX, hd, dtype=torch.float32)
    out[:, :, :SEQ_LEN, :] = t.float()
    return out


def main() -> None:
    assert WEIGHTS.exists(), WEIGHTS
    torch.manual_seed(1337)
    cfg = AutoConfig.from_pretrained("google/gemma-4-E2B-it")
    tc = getattr(cfg, "text_config", cfg)
    softcap = float(getattr(tc, "final_logit_softcapping", 30.0))
    lm = None

    print("Build hybride (streaming) ...")
    model = build_hybrid_model(tc)
    lm_w = model.embed_tokens.weight.to(torch.float32)

    def next_token(out):
        lh = out.last_hidden_state.to(torch.float32)[:, -1, :]
        lg = softcap * torch.tanh((lh @ lm_w.t()) / softcap)
        return int(lg.argmax(dim=-1).item())

    input_ids = torch.tensor(INPUT_IDS, dtype=torch.long).view(1, SEQ_LEN)
    print("Prefill use_cache=True ...")
    with torch.no_grad():
        out_pref = model(input_ids=input_ids, use_cache=True)
    pkv = out_pref.past_key_values
    s0 = next_token(out_pref)
    print(f"  s0 (prefill argmax, à feed pos 4) = {s0}")

    # extraire les caches prefill [0..3] (AVANT de muter pkv par la boucle) -> paddés kmax
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

    # greedy manuel : feed s_k (pos 4+k) -> s_{k+1}. Séquence s0..sN.
    seq = [s0]
    cur = s0
    for k in range(N_DECODE):
        with torch.no_grad():
            out = model(input_ids=torch.tensor([[cur]], dtype=torch.long), past_key_values=pkv, use_cache=True)
        nxt = next_token(out)
        seq.append(nxt)
        cur = nxt
    print(f"  séquence HF greedy (s0..s{N_DECODE}) = {seq}")
    fed = seq[:N_DECODE]          # tokens feedés s0..s_{N-1}
    expected = seq[1:N_DECODE + 1]  # argmax attendus s1..sN

    # pré-gather embed/embptl des tokens fed + cos/sin full par pos
    rot = Gemma4TextRotaryEmbedding(tc)
    embeds = torch.zeros(N_DECODE, 1, 1, 1536, dtype=torch.bfloat16)
    embptls = torch.zeros(N_DECODE, 1, 1, 8960, dtype=torch.bfloat16)
    cos_full = torch.zeros(N_DECODE, 1, 1, HD_F, dtype=torch.float32)
    sin_full = torch.zeros(N_DECODE, 1, 1, HD_F, dtype=torch.float32)
    positions = torch.zeros(N_DECODE, dtype=torch.int32)
    # mask par step : à pos p, le nouveau token (q=1) voit k=0..p (positions remplies), masque k>p (slots futurs).
    masks = torch.zeros(N_DECODE, 1, 1, 1, KMAX, dtype=torch.float32)
    MIN = torch.finfo(torch.float32).min
    dummy = torch.zeros(1, 1, 1536, dtype=torch.float32)
    for k in range(N_DECODE):
        tid = torch.tensor([fed[k]], dtype=torch.long)
        embeds[k] = model.embed_tokens.weight[tid].to(torch.bfloat16).view(1, 1, 1536)
        embptls[k] = model.embed_tokens_per_layer.weight[tid].view(1, 1, 8960)
        p = SEQ_LEN + k
        positions[k] = p
        cf, sf = rot(dummy, torch.tensor([[p]]), layer_type="full_attention")  # [1,1,512]
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
    for k, t in tensors.items():
        assert not torch.isnan(t.float()).any(), f"NaN dans {k}"

    print("=" * 72)
    print("P5.7.8 — oracle génération")
    print(f"  prompt={INPUT_IDS} ; N_DECODE={N_DECODE} ; kmax={KMAX}")
    print(f"  fed (s0..s{N_DECODE-1})      = {fed}")
    print(f"  expected (s1..s{N_DECODE}) = {expected}")
    print(f"  positions = {positions.tolist()}")

    OUT_FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(OUT_FIXTURE))
    print("wrote", OUT_FIXTURE)

    _ = lm
    manifest = {
        "source": "P5.7.8 oracle génération (boucle decode N tokens, greedy == HF)",
        "design": "docs/P5_7_7_decode.md",
        "prompt": INPUT_IDS, "n_decode": N_DECODE, "kmax": KMAX,
        "sequence_hf": seq, "fed": fed, "expected": expected,
        "sliding_producers": SLIDING_PRODUCERS, "full_producers": FULL_PRODUCERS,
        "tensors": {n: {"shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", "")} for n, t in tensors.items()},
        "pass": "argmax ZML[k] == expected[k] pour tout k (séquence == HF greedy)",
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print("wrote", OUT_MANIFEST, "\nP5.7.8 oracle OK.")


if __name__ == "__main__":
    main()
