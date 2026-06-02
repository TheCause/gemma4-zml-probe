"""P5.7.5 — Oracle prefill 35 couches HYBRIDE (fp32 sauf embed_tokens_per_layer bf16).

Conforme au contrat docs/P5_7_5_precision_contract.md (option 1, gate P5.7.5-prep, commit f78f5db).
Remplace le script 38 (full bf16, incompatible). Chargement STREAMING (jamais le state_dict complet
co-résident) pour tenir sous la VM 23 Go : build bf16 (~9,3 Go) -> upcast du corps en fp32 param par
param (embed_tokens_per_layer reste bf16). Corps + rotary fp32 ; seul embed_tokens_per_layer bf16
(bit-identique au full fp32 : x256=16 exact, fusion frontend PLE en fp32).

Exporte : input_ids, cos/sin sliding (256) + full (512), attn_mask causal, last_hidden (post final norm),
ET les 36 hidden states (embeddings + sortie de chaque couche) pour la localisation drift-vs-cablage (contrat §6).
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
OUT_FIXTURE = ROOT / "fixtures" / "p5_7_5_hybrid.safetensors"
OUT_MANIFEST = ROOT / "fixtures" / "p5_7_5_hybrid_manifest.json"
SEQ_LEN = 4
INPUT_IDS = [2, 105, 2048, 4095]
PREFIX = "model.language_model."
EMBPTL = "embed_tokens_per_layer.weight"  # seul tensor gardé en bf16 (exception mémoire)


def set_submodule_attr(root, dotted, value):
    """Remplace root.<dotted> (gère les ModuleList via index entier)."""
    parts = dotted.split(".")
    obj = root
    for p in parts[:-1]:
        obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
    setattr(obj, parts[-1], value)


def main() -> None:
    assert WEIGHTS.exists(), WEIGHTS
    torch.manual_seed(1337)
    cfg = AutoConfig.from_pretrained("google/gemma-4-E2B-it")
    tc = getattr(cfg, "text_config", cfg)

    print("Construction Gemma4TextModel (bf16) ...")
    torch.set_default_dtype(torch.bfloat16)
    model = Gemma4TextModel(tc)
    model.train(False)

    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())  # CRUCIAL : layer_scalar est un registered_buffer, PAS un param.
    loaded = set()
    print("Streaming load (params corps -> fp32, embed_tokens_per_layer -> bf16, buffers layer_scalar) ...")
    with safe_open(str(WEIGHTS), framework="pt") as s:
        for k in s.keys():
            if not k.startswith(PREFIX):
                continue
            name = k[len(PREFIX):]
            t = s.get_tensor(k)  # bf16 (disque)
            with torch.no_grad():
                if name in params:
                    if name == EMBPTL:
                        params[name].copy_(t)  # bf16 -> bf16
                    else:
                        set_submodule_attr(model, name, torch.nn.Parameter(t.to(torch.float32), requires_grad=False))
                elif name in buffers:
                    # buffer (layer_scalar, ...) -> tensor fp32 (PAS un Parameter)
                    set_submodule_attr(model, name, t.to(torch.float32))
                else:
                    del t
                    continue  # k/v des readers non instanciés (YOCO) -> ignorés
            loaded.add(name)
            del t

    missing = [n for n in params if n not in loaded and ("weight" in n or n.endswith("layer_scalar"))]
    assert not missing, f"params manquants: {missing[:10]}"
    # Vérifie que les 35 layer_scalar (buffers) sont bien chargés (sinon oracle faux, cf bug du 2 juin).
    ls_missing = [n for n in buffers if n.endswith("layer_scalar") and n not in loaded]
    assert not ls_missing, f"layer_scalar buffers manquants: {ls_missing[:5]}"
    n_ls = sum(1 for n in loaded if n.endswith("layer_scalar"))
    print(f"  layer_scalar buffers chargés: {n_ls}/35")

    # Buffers bf16 (rotary inv_freq, etc.) -> fp32 pour un forward fp32 fidèle.
    for name, buf in list(model.named_buffers()):
        if buf.dtype == torch.bfloat16:
            set_submodule_attr(model, name, buf.float())

    # embed_scale (√1536) a été arrondi bf16 (39.25) par le build bf16-first ; un oracle fp32 fidèle
    # utilise fp32(39.1918) — ce que le moteur ZML applique aussi. (embed_tokens_per_layer.embed_scale
    # reste ×16, exact. per_layer_*_scale sont déjà des floats fp64.) Sinon drift STAGE0 ~0.032.
    model.embed_tokens.embed_scale = torch.tensor(1536.0 ** 0.5, dtype=torch.float32)

    input_ids = torch.tensor(INPUT_IDS, dtype=torch.long).view(1, SEQ_LEN)
    pos = torch.arange(SEQ_LEN).unsqueeze(0)

    print("Forward prefill hybride (35 couches, corps fp32) + hidden states + shared KV ...")
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=False, output_hidden_states=True,
                    return_shared_kv_states=True)
    # KV partagé YOCO (writers 13 sliding / 14 full) pour valider le mode reader du moteur ZML.
    # shared_kv_states[type] = (K, V), shape [1,1,4,256] sliding / [1,1,4,512] full.
    skv = out.shared_kv_states
    kv_k_sliding, kv_v_sliding = skv["sliding_attention"]
    kv_k_full, kv_v_full = skv["full_attention"]
    last_hidden = out.last_hidden_state.to(torch.float32).contiguous().clone()  # [1,4,1536] post final norm
    assert tuple(last_hidden.shape) == (1, SEQ_LEN, 1536)
    assert not torch.isnan(last_hidden).any()

    # Logits (P5.7.6) : lm_head tied = embed_tokens.weight (poids BRUT, sans embed_scale en sortie),
    # puis softcap final_logit_softcapping=30. C'est ce que calcule Gemma4ForCausalLM.
    softcap = float(getattr(tc, "final_logit_softcapping", 30.0))
    lm_w = model.embed_tokens.weight.to(torch.float32)  # [vocab, 1536]
    logits = last_hidden @ lm_w.t()                      # [1,4,vocab]
    logits = (softcap * torch.tanh(logits / softcap)).contiguous()
    argmax = logits.argmax(dim=-1)                       # [1,4] token prédit par position
    print(f"  logits shape {tuple(logits.shape)} ; argmax par position = {argmax.flatten().tolist()}")
    # out.hidden_states a 36 entrées ; la dernière (index 35) est la sortie POST-final-norm = alias de
    # last_hidden_state. On garde les 35 premières (00=embeddings, kk=sortie couche kk-1 PRÉ-norm) +
    # last_hidden séparé. clone() casse tout partage mémoire (sinon safetensors refuse).
    assert len(out.hidden_states) == tc.num_hidden_layers + 1, len(out.hidden_states)
    hs = [h.to(torch.float32).contiguous().clone() for h in out.hidden_states[:-1]]  # 35 taps

    torch.set_default_dtype(torch.float32)
    rot = Gemma4TextRotaryEmbedding(tc)
    cos_s, sin_s = rot(last_hidden, pos, layer_type="sliding_attention")  # [1,4,256]
    cos_f, sin_f = rot(last_hidden, pos, layer_type="full_attention")     # [1,4,512]
    min_val = torch.finfo(torch.float32).min
    idx = torch.arange(SEQ_LEN)
    causal = (idx.view(SEQ_LEN, 1) >= idx.view(1, SEQ_LEN))
    attn_mask = torch.where(causal, torch.zeros(()), torch.full((), min_val)).view(1, 1, SEQ_LEN, SEQ_LEN)

    print("=" * 70)
    print("P5.7.5 — oracle prefill HYBRIDE")
    print("input_ids =", INPUT_IDS)
    for q in [0, 3]:
        print(f"  last_hidden[0,{q},:6] =", [round(v, 8) for v in last_hidden[0, q, :6].tolist()])
    print(f"  stats last_hidden: mean={last_hidden.mean():.4e} std={last_hidden.std():.4e} "
          f"min={last_hidden.min():.4e} max={last_hidden.max():.4e}")
    print(f"  embed_tokens_per_layer dtype = {model.embed_tokens_per_layer.weight.dtype} (attendu bfloat16)")
    print(f"  norm.weight dtype = {model.norm.weight.dtype} (attendu float32)")

    # Pré-gather des lignes d'embedding pour les input_ids -> le moteur ZML lit ces slices au lieu
    # de résider les tables complètes (embed_tokens 0,8 Go + embed_tokens_per_layer 4,7 Go) qui le
    # faisaient OOM. bf16 = valeurs disque exactes (embed_tokens fp32 upcast -> redescendu bf16 = exact).
    ids_flat = input_ids.view(-1)
    embed_slice = model.embed_tokens.weight[ids_flat].to(torch.bfloat16).contiguous().view(1, SEQ_LEN, 1536)
    embptl_slice = model.embed_tokens_per_layer.weight[ids_flat].contiguous().view(1, SEQ_LEN, -1)  # bf16 [1,4,8960]

    tensors = {
        "input_ids": input_ids.to(torch.int32),
        "embed_slice": embed_slice, "embptl_slice": embptl_slice,
        "cos_sliding": cos_s.contiguous(), "sin_sliding": sin_s.contiguous(),
        "cos_full": cos_f.contiguous(), "sin_full": sin_f.contiguous(),
        "attn_mask": attn_mask.contiguous(),
        "last_hidden": last_hidden,
        "logits": logits,  # [1,4,vocab] softcapped — référence P5.7.6
        # KV partagé YOCO (mode reader du moteur). hidden_15 (résiduel entrant couche 15) = hidden_15 ci-dessous.
        "kv_k_sliding": kv_k_sliding.to(torch.float32).contiguous(),
        "kv_v_sliding": kv_v_sliding.to(torch.float32).contiguous(),
        "kv_k_full": kv_k_full.to(torch.float32).contiguous(),
        "kv_v_full": kv_v_full.to(torch.float32).contiguous(),
    }
    for i, h in enumerate(hs):
        tensors[f"hidden_{i:02d}"] = h  # 00=embeddings ; kk=sortie couche kk-1 (pré-norm), kk=01..34
    OUT_FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(OUT_FIXTURE))
    print("wrote", OUT_FIXTURE)

    manifest = {
        "source": "P5.7.5 oracle prefill HYBRIDE (fp32 corps, embed_tokens_per_layer bf16)",
        "contract": "docs/P5_7_5_precision_contract.md",
        "supersedes": "scripts/38_p5_7_5_prefill_oracle.py (full bf16)",
        "input_ids": INPUT_IDS, "seq_len": SEQ_LEN,
        "dtype_policy": {"body": "float32", "embed_tokens_per_layer": "bfloat16", "rotary": "float32"},
        "tensors": {n: {"shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", "")} for n, t in tensors.items()},
        "pass_threshold": {"max_abs_le": 1e-2, "mean_abs_le": 1e-4},
        "warn_band": "1e-2 < max_abs <= 1e-1 OR (max_abs<=1e-2 AND mean_abs>1e-4)",
        "note": "hidden_00=embeddings ; hidden_kk=sortie couche kk-1 PRÉ-norm (kk=01..34) ; last_hidden=POST final norm (=sortie couche 34 normée). Localisation §6 : comparer moteur couche-k vs hidden_(k+1), final vs last_hidden.",
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print("wrote", OUT_MANIFEST, "\nP5.7.5 oracle hybride OK.")


if __name__ == "__main__":
    main()
