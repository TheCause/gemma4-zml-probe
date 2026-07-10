"""L0 — Oracle de GÉNÉRATION LONGUE (boucle decode N tokens, fenêtre glissante 512 active).

Réécriture ciblée de 45_gen_vq_oracle.py pour le chantier "génération longue" (cf
docs/GENERATION_LONGUE_DESIGN.md / _PLAN.md). Différences vs 45 :
  - N_DECODE ≈ 2044 tokens (au lieu de 4), positions 4..2047 → franchit la fenêtre 512 ~4 fois.
  - DEUX masques par step : `masks_sliding` (bande causale `[p-511, p]`, .k=L_MAX) et `masks_full`
    (causal plein `[0, p]`, .k=L_MAX), sélectionnés côté moteur ZML par `comptime isFull(i)`.
  - Caches prefill (producers 0..14) paddés à .k=L_MAX au lieu de KMAX=8.
  - PAS de V-quant (ni hook, ni constantes) — compression hors-scope de la génération longue.
  - Génération sur GPU (cuda) : 2044 forwards fp32 sur CPU prendraient des heures.

Définition sliding window (vérifiée dans modeling_gemma4.sliding_window_mask_function) :
  dist = q_idx - kv_idx ; visible ssi (dist >= 0) & (dist < sliding_window=512)
  → key j visible pour la query à la position p ssi  max(0, p-511) <= j <= p.

Le moteur ZML (L1a) part du cache prefill (cache0), SCATTER le KV de chaque token décodé à .k=position,
et lit le cache via le masque par type de couche. Donc on ne fournit QUE le cache prefill ; le KV des
2044 tokens décodés est construit par le moteur (comme E1/decode4).

Fixture : gen_long.safetensors. Manifest : gen_long_manifest.json.
CLI : python3 scripts/46_gen_long_oracle.py   (3090, venv gemma4-probe).

Variante G2.3 famille kv_store (mécanisme b, cf plan Task 3) : `--kv-dtype bf16` écrit les 4
tenseurs cache_* en bf16 (le dtype de STOCKAGE du cache ZML vient du header de la fixture ;
l'arrondi bf16 du KV prefill = le contrat kv_store appliqué à l'état initial). Sorties renommées
gen_long_kvbf16.safetensors / gen_long_kvbf16_manifest.json — la fixture standard f32 est intacte.
Tout le reste (oracle HF, masques, embeds, expected) est identique bit-à-bit au run f32 à seed égale.
NB provenance : la clé `kv_dtype` est désormais écrite dans le manifest dans TOUS les cas (y compris
le run par défaut f32) ; le .safetensors par défaut, lui, reste byte-identique.
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
from transformers import AutoConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel, Gemma4TextRotaryEmbedding

ROOT = Path("/data/gemma4-zml-probe")
WEIGHTS = ROOT / "weights" / "model.safetensors"
OUT_FIXTURE = ROOT / "gen_long.safetensors"
OUT_MANIFEST = ROOT / "gen_long_manifest.json"

SEQ_LEN = 4
# L_MAX = taille comptime du cache. Réduit à 1024 (de 2048) : le pic compile XLA-CPU à .k=2048
# (~34 Go) dépasse les 32 Go de l'hôte Proxmox → swap thrashing. 1024 franchit la fenêtre 512
# (valide masque bande + ring L1b) avec un pic plus modéré. Remonter quand la RAM VM augmentera.
L_MAX = 1024
N_DECODE = L_MAX - SEQ_LEN    # 1020 ; positions décodées 4..1023
SLIDING_WINDOW = 512
INPUT_IDS = [2, 105, 2048, 4095]
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
    """Identique à 45 : poids fp32, layer_scalar buffers, embed_scale fp32 (oracle = source de vérité)."""
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
    """Cache prefill [1,1,SEQ_LEN,hd] -> [1,1,L_MAX,hd] (positions 0..SEQ_LEN-1 remplies, reste 0)."""
    hd = t.shape[-1]
    out = torch.zeros(1, 1, L_MAX, hd, dtype=torch.float32)
    out[:, :, :SEQ_LEN, :] = t.float().cpu()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kv-dtype", choices=["f32", "bf16"], default="f32",
                    help="dtype de STOCKAGE des tenseurs cache_* de la fixture (bf16 = variante "
                         "G2.3 famille kv_store, mécanisme b ; sorties *_kvbf16)")
    args = ap.parse_args()
    kv_dt = torch.bfloat16 if args.kv_dtype == "bf16" else torch.float32
    out_fixture = ROOT / "gen_long_kvbf16.safetensors" if args.kv_dtype == "bf16" else OUT_FIXTURE
    out_manifest = ROOT / "gen_long_kvbf16_manifest.json" if args.kv_dtype == "bf16" else OUT_MANIFEST

    assert WEIGHTS.exists(), WEIGHTS
    torch.manual_seed(1337)
    cfg = AutoConfig.from_pretrained("google/gemma-4-E2B-it")
    tc = getattr(cfg, "text_config", cfg)
    softcap = float(getattr(tc, "final_logit_softcapping", 30.0))
    sw = int(getattr(tc, "sliding_window", SLIDING_WINDOW))
    assert sw == SLIDING_WINDOW, f"sliding_window config={sw} != {SLIDING_WINDOW}"

    print(f"Build hybride (streaming) ... device={DEVICE}")
    model = build_hybrid_model(tc).to(DEVICE)
    lm_w = model.embed_tokens.weight.to(torch.float32)

    def next_token(out):
        lh = out.last_hidden_state.to(torch.float32)[:, -1, :]
        lg = softcap * torch.tanh((lh @ lm_w.t()) / softcap)
        return int(lg.argmax(dim=-1).item())

    input_ids = torch.tensor(INPUT_IDS, dtype=torch.long, device=DEVICE).view(1, SEQ_LEN)
    print("Prefill use_cache=True ...")
    with torch.no_grad():
        out_pref = model(input_ids=input_ids, use_cache=True)
    pkv = out_pref.past_key_values
    s0 = next_token(out_pref)
    print(f"  s0 (prefill argmax, à feed pos {SEQ_LEN}) = {s0}")

    # caches prefill producers 0..14 (prompt < 512 → sliding non tronqué), paddés à L_MAX.
    cache_sl_k = torch.zeros(len(SLIDING_PRODUCERS), 1, 1, L_MAX, HD_S, dtype=torch.float32)
    cache_sl_v = torch.zeros_like(cache_sl_k)
    cache_fl_k = torch.zeros(len(FULL_PRODUCERS), 1, 1, L_MAX, HD_F, dtype=torch.float32)
    cache_fl_v = torch.zeros_like(cache_fl_k)
    for slot, i in enumerate(SLIDING_PRODUCERS):
        cache_sl_k[slot] = pad_cache(pkv.layers[i].keys)
        cache_sl_v[slot] = pad_cache(pkv.layers[i].values)
    for slot, i in enumerate(FULL_PRODUCERS):
        cache_fl_k[slot] = pad_cache(pkv.layers[i].keys)
        cache_fl_v[slot] = pad_cache(pkv.layers[i].values)

    # greedy manuel (sliding window natif HF) : feed s_k (pos 4+k) -> s_{k+1}. EOS ignoré (on continue).
    print(f"Génération greedy {N_DECODE} tokens (sliding window {SLIDING_WINDOW} natif) ...")
    seq = [s0]
    cur = s0
    for k in range(N_DECODE):
        with torch.no_grad():
            out = model(input_ids=torch.tensor([[cur]], dtype=torch.long, device=DEVICE),
                        past_key_values=pkv, use_cache=True)
        cur = next_token(out)
        seq.append(cur)
        if (k + 1) % 256 == 0:
            print(f"    {k + 1}/{N_DECODE} tokens")
    fed = seq[:N_DECODE]
    expected = seq[1:N_DECODE + 1]

    # par-step : embeds, embptls, cos/sin (RoPE full), positions, 2 masques.
    rot = Gemma4TextRotaryEmbedding(tc).to(DEVICE)
    embeds = torch.zeros(N_DECODE, 1, 1, 1536, dtype=torch.bfloat16)
    embptls = torch.zeros(N_DECODE, 1, 1, 8960, dtype=torch.bfloat16)
    cos_full = torch.zeros(N_DECODE, 1, 1, HD_F, dtype=torch.float32)
    sin_full = torch.zeros(N_DECODE, 1, 1, HD_F, dtype=torch.float32)
    positions = torch.zeros(N_DECODE, dtype=torch.int32)
    masks_sliding = torch.zeros(N_DECODE, 1, 1, 1, L_MAX, dtype=torch.float32)
    masks_full = torch.zeros(N_DECODE, 1, 1, 1, L_MAX, dtype=torch.float32)
    MIN = torch.finfo(torch.float32).min
    dummy = torch.zeros(1, 1, 1536, dtype=torch.float32, device=DEVICE)
    emb_w = model.embed_tokens.weight        # [voc,1536]
    eptl_w = model.embed_tokens_per_layer.weight  # [voc,8960]
    for k in range(N_DECODE):
        tid = fed[k]
        embeds[k] = emb_w[tid].to(torch.bfloat16).view(1, 1, 1536).cpu()
        embptls[k] = eptl_w[tid].view(1, 1, 8960).to(torch.bfloat16).cpu()
        p = SEQ_LEN + k
        positions[k] = p
        cf, sf = rot(dummy, torch.tensor([[p]], device=DEVICE), layer_type="full_attention")
        cos_full[k] = cf.float().cpu()
        sin_full[k] = sf.float().cpu()
        lo = max(0, p - (SLIDING_WINDOW - 1))   # [p-511, p]
        for j in range(L_MAX):
            if j > p or j < lo:
                masks_sliding[k, 0, 0, 0, j] = MIN
            if j > p:
                masks_full[k, 0, 0, 0, j] = MIN

    tensors = {
        "embeds": embeds.contiguous(), "embptls": embptls.contiguous(),
        "cos_full": cos_full.contiguous(), "sin_full": sin_full.contiguous(),
        "positions": positions.contiguous(),
        "masks_sliding": masks_sliding.contiguous(), "masks_full": masks_full.contiguous(),
        # dtype de stockage du cache : kv_dt (bf16 = variante kv_store option b ; f32 = standard)
        "cache_sl_k": cache_sl_k.to(kv_dt).contiguous(), "cache_sl_v": cache_sl_v.to(kv_dt).contiguous(),
        "cache_fl_k": cache_fl_k.to(kv_dt).contiguous(), "cache_fl_v": cache_fl_v.to(kv_dt).contiguous(),
        "expected": torch.tensor(expected, dtype=torch.int32),
        "fed": torch.tensor(fed, dtype=torch.int32),
    }
    for k, t in tensors.items():
        assert not torch.isnan(t.float()).any(), f"NaN dans {k}"

    print("=" * 72)
    print("L0 — oracle génération longue (sliding window 512 natif)")
    print(f"  prompt={INPUT_IDS} ; SEQ_LEN={SEQ_LEN} ; N_DECODE={N_DECODE} ; L_MAX={L_MAX}")
    print(f"  positions = {positions[0].item()}..{positions[-1].item()}")
    print(f"  fed[:6]      = {fed[:6]}")
    print(f"  expected[:6] = {expected[:6]}")
    print(f"  expected[-4:] = {expected[-4:]}")

    out_fixture.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_fixture))
    print("wrote", out_fixture)

    manifest = {
        "source": "L0 oracle génération longue (boucle decode N tokens, greedy == HF sliding window 512)",
        "prompt": INPUT_IDS, "seq_len": SEQ_LEN, "n_decode": N_DECODE, "l_max": L_MAX,
        "sliding_window": SLIDING_WINDOW, "kv_dtype": args.kv_dtype,
        "sliding_producers": SLIDING_PRODUCERS, "full_producers": FULL_PRODUCERS,
        "fed_head": fed[:8], "expected_head": expected[:8], "expected_tail": expected[-8:],
        "tensors": {n: {"shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", "")} for n, t in tensors.items()},
        # Critère honnête par variante : un run bf16 (kv_store) N'A PAS d'exigence argmax == expected
        # (bifurcations attendues en bf16, cf G2.0) — son verdict est RELATIF vs D0/enveloppe.
        "pass_criterion_L1a": (
            "run kv_store G2.3 : verdict RELATIF vs D0/enveloppe (cf docs/G2_3_OP_SENSITIVITY.md) — "
            "pas d'exigence argmax ZML[k] == expected[k]"
            if args.kv_dtype == "bf16"
            else "argmax ZML[k] == expected[k] pour tout k (cache linéaire L_MAX + masque bande)"
        ),
    }
    out_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print("wrote", out_manifest, "\nL0 oracle génération longue OK.")


if __name__ == "__main__":
    main()
