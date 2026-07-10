"""49 — Oracle de génération pour un PROMPT CUSTOM (texte libre) → fixture reproductible par le moteur ZML.

Dérivé de 46_gen_long_oracle.py. Différences : le prompt vient de --prompt (langage naturel, encodé via
le CHAT TEMPLATE Gemma au lieu d'INPUT_IDS hardcodés), SEQ_LEN = longueur du prompt tokenisé (variable),
N_DECODE = --n-tokens (défaut 200, pas 1020). Produit le MÊME format de fixture que 46 (embeds, embptls,
cos/sin full, positions, masks_sliding/full .k=L_MAX, caches prefill paddés, expected/fed) → reproductible
tel quel par `gemma4_gen_long_gpu` (L_MAX=1024). Détokenise ensuite avec scripts/48_detokenize.py.

Le moteur ZML reste un BANC op-par-op : HF (ici) est l'oracle de référence ; ZML doit reproduire `expected`
au token près. La chaîne complète testée : prompt texte → (HF tokenise+génère) → fixture → (ZML reproduit)
→ (48 détokenise + valide round-trip) → texte.

CLI : python3 scripts/49_gen_custom_oracle.py --prompt "..." [--n-tokens 200] [--out gen_custom.safetensors]
Prérequis : GPU (cuda) recommandé (sinon lent), venv gemma4-probe, weights + tokenizer en cache.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", "/data/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig, AutoTokenizer
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel, Gemma4TextRotaryEmbedding

ROOT = Path("/data/gemma4-zml-probe")
WEIGHTS = ROOT / "weights" / "model.safetensors"
MODEL_ID = "google/gemma-4-E2B-it"

L_MAX = 1024
SLIDING_WINDOW = 512
PREFIX = "model.language_model."
EMBPTL = "embed_tokens_per_layer.weight"
SLIDING_PRODUCERS = [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13]
FULL_PRODUCERS = [4, 9, 14]
HD_S, HD_F = 256, 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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


def pad_cache(t, seq_len):
    hd = t.shape[-1]
    out = torch.zeros(1, 1, L_MAX, hd, dtype=torch.float32)
    out[:, :, :seq_len, :] = t.float().cpu()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True, help="prompt en langage naturel (rôle user)")
    ap.add_argument("--n-tokens", type=int, default=200, help="nombre de tokens à générer (défaut 200)")
    ap.add_argument("--out", default=str(ROOT / "gen_custom.safetensors"))
    ap.add_argument("--kv-dtype", choices=["f32", "bf16"], default="f32",
                    help="dtype d'écriture du cache K/V de la fixture (bf16 = variante kv_store G2.3)")
    args = ap.parse_args()

    assert WEIGHTS.exists(), WEIGHTS
    torch.manual_seed(1337)
    cfg = AutoConfig.from_pretrained(MODEL_ID)
    tc = getattr(cfg, "text_config", cfg)
    softcap = float(getattr(tc, "final_logit_softcapping", 30.0))

    # --- Tokenisation via chat template Gemma (vraie conversation user -> model) ---
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    enc = tok.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        add_generation_prompt=True, return_tensors="pt",
    )
    input_ids = (enc if torch.is_tensor(enc) else enc["input_ids"]).to(DEVICE)
    seq_len = int(input_ids.shape[1])
    n_decode = int(args.n_tokens)
    assert seq_len < SLIDING_WINDOW, f"prompt {seq_len} tok >= fenêtre {SLIDING_WINDOW} (cache prefill tronqué non géré)"
    assert seq_len + n_decode <= L_MAX, f"seq_len({seq_len})+n_decode({n_decode}) > L_MAX({L_MAX})"
    print(f"prompt='{args.prompt}'  → {seq_len} tokens ; génère {n_decode} ; device={DEVICE}")

    model = build_hybrid_model(tc).to(DEVICE)
    lm_w = model.embed_tokens.weight.to(torch.float32)

    def next_token(out):
        lh = out.last_hidden_state.to(torch.float32)[:, -1, :]
        lg = softcap * torch.tanh((lh @ lm_w.t()) / softcap)
        return int(lg.argmax(dim=-1).item())

    with torch.no_grad():
        out_pref = model(input_ids=input_ids, use_cache=True)
    pkv = out_pref.past_key_values
    s0 = next_token(out_pref)

    cache_sl_k = torch.zeros(len(SLIDING_PRODUCERS), 1, 1, L_MAX, HD_S, dtype=torch.float32)
    cache_sl_v = torch.zeros_like(cache_sl_k)
    cache_fl_k = torch.zeros(len(FULL_PRODUCERS), 1, 1, L_MAX, HD_F, dtype=torch.float32)
    cache_fl_v = torch.zeros_like(cache_fl_k)
    for slot, i in enumerate(SLIDING_PRODUCERS):
        cache_sl_k[slot] = pad_cache(pkv.layers[i].keys, seq_len)
        cache_sl_v[slot] = pad_cache(pkv.layers[i].values, seq_len)
    for slot, i in enumerate(FULL_PRODUCERS):
        cache_fl_k[slot] = pad_cache(pkv.layers[i].keys, seq_len)
        cache_fl_v[slot] = pad_cache(pkv.layers[i].values, seq_len)

    print(f"Génération greedy {n_decode} tokens (sliding window {SLIDING_WINDOW}) ...")
    seq = [s0]
    cur = s0
    for k in range(n_decode):
        with torch.no_grad():
            out = model(input_ids=torch.tensor([[cur]], dtype=torch.long, device=DEVICE),
                        past_key_values=pkv, use_cache=True)
        cur = next_token(out)
        seq.append(cur)
    fed = seq[:n_decode]
    expected = seq[1:n_decode + 1]

    rot = Gemma4TextRotaryEmbedding(tc).to(DEVICE)
    embeds = torch.zeros(n_decode, 1, 1, 1536, dtype=torch.bfloat16)
    embptls = torch.zeros(n_decode, 1, 1, 8960, dtype=torch.bfloat16)
    cos_full = torch.zeros(n_decode, 1, 1, HD_F, dtype=torch.float32)
    sin_full = torch.zeros(n_decode, 1, 1, HD_F, dtype=torch.float32)
    positions = torch.zeros(n_decode, dtype=torch.int32)
    masks_sliding = torch.zeros(n_decode, 1, 1, 1, L_MAX, dtype=torch.float32)
    masks_full = torch.zeros(n_decode, 1, 1, 1, L_MAX, dtype=torch.float32)
    MIN = torch.finfo(torch.float32).min
    dummy = torch.zeros(1, 1, 1536, dtype=torch.float32, device=DEVICE)
    emb_w = model.embed_tokens.weight
    eptl_w = model.embed_tokens_per_layer.weight
    for k in range(n_decode):
        tid = fed[k]
        embeds[k] = emb_w[tid].to(torch.bfloat16).view(1, 1, 1536).cpu()
        embptls[k] = eptl_w[tid].view(1, 1, 8960).to(torch.bfloat16).cpu()
        p = seq_len + k
        positions[k] = p
        cf, sf = rot(dummy, torch.tensor([[p]], device=DEVICE), layer_type="full_attention")
        cos_full[k] = cf.float().cpu()
        sin_full[k] = sf.float().cpu()
        lo = max(0, p - (SLIDING_WINDOW - 1))
        for j in range(L_MAX):
            if j > p or j < lo:
                masks_sliding[k, 0, 0, 0, j] = MIN
            if j > p:
                masks_full[k, 0, 0, 0, j] = MIN

    # --kv-dtype bf16 (G2.3, miroir du script 46) : cache K/V écrit en bf16 — le dtype de STOCKAGE
    # du cache ZML suit le header de la fixture (mécanisme (b), famille kv_store). L'état prefill
    # est arrondi aussi (contrat kv_store appliqué au prefill). En f32 (défaut), .to() est un no-op
    # → fixture standard inchangée.
    kv_dt = torch.bfloat16 if args.kv_dtype == "bf16" else torch.float32
    tensors = {
        "embeds": embeds.contiguous(), "embptls": embptls.contiguous(),
        "cos_full": cos_full.contiguous(), "sin_full": sin_full.contiguous(),
        "positions": positions.contiguous(),
        "masks_sliding": masks_sliding.contiguous(), "masks_full": masks_full.contiguous(),
        "cache_sl_k": cache_sl_k.to(kv_dt).contiguous(), "cache_sl_v": cache_sl_v.to(kv_dt).contiguous(),
        "cache_fl_k": cache_fl_k.to(kv_dt).contiguous(), "cache_fl_v": cache_fl_v.to(kv_dt).contiguous(),
        "expected": torch.tensor(expected, dtype=torch.int32),
        "fed": torch.tensor(fed, dtype=torch.int32),
    }
    for k, t in tensors.items():
        assert not torch.isnan(t.float()).any(), f"NaN dans {k}"

    save_file(tensors, args.out)
    print("=" * 60)
    print(f"prompt tokens (seq_len={seq_len}) : {input_ids[0].tolist()}")
    print(f"aperçu réponse HF : {tok.decode(fed, skip_special_tokens=True)[:300]!r}")
    print(f"wrote {args.out}  (seq_len={seq_len}, n_decode={n_decode}, L_MAX={L_MAX})")
    print("→ reproduire : gemma4_gen_long_gpu <weights> " + args.out + f" {n_decode}")
    print("→ détokeniser+valider : python3 scripts/48_detokenize.py " + args.out)

    Path(args.out + ".manifest.json").write_text(json.dumps({
        "prompt": args.prompt, "prompt_ids": input_ids[0].tolist(), "seq_len": seq_len,
        "n_decode": n_decode, "l_max": L_MAX, "fed_head": fed[:8], "expected_head": expected[:8],
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
