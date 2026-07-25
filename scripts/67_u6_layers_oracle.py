#!/usr/bin/env python3
"""U6 — oracle couche décodeur 12B COMPLÈTE + chaîne L0→L5 : modules RÉELS
`Gemma4TextDecoderLayer` (piège 12 : jamais de recomposition).

Six modules réels `Gemma4TextDecoderLayer(text_config, layer_idx=i)` pour i in 0..5
(transformers 5.14.1, venv g12b) : 5 couches sliding (GQA 8KV×256) + 1 couche full
(L5 : MQA 1KV×512, K=V, p-RoPE proportional 512/0.25/1e6). Chaque couche = sandwich
norms (input/post_attention/pre_feedforward/post_feedforward) + attention + MLP
gelu-tanh + **layer_scalar** (buffer [1] du checkpoint, appliqué en sortie —
`hidden_states *= self.layer_scalar`, modeling_gemma4.py). Poids chargés depuis
l'EXPORT dq (D9 : oracles Python = export, gates Zig = PACKÉ), module passé en **f32**
(poids bf16 → f32 exact) : le flux comparé est celui du moteur ZML (`dotPrec fam=null`)
— même choix déclaré que U3/U5 (scripts 65/66).

L'expected principal = la SORTIE de chaque couche, ENCHAÎNÉE (hidden → hidden : la
sortie de la couche i est l'entrée de la couche i+1) — « couche 0 seule » == le premier
maillon de la chaîne (même entrée : le hidden de la fixture) ; la sortie de chaîne ==
out_l5. Pas de hooks nécessaires : les 6 sorties de couche sont les expected (les
intermédiaires fins sont déjà gatés par U3/U4/U5).

Distribution des 48 layer_scalar : safe_open (via load_dq_weights) sur les 48 clés
`model.language_model.layers.{i}.layer_scalar` → valeurs au manifest (tous 1.0 ?).

S=8 (périmètre U6, Amendement 2 : « S=8 explicite (couche 0 seule ET chaîne L0→L5) ») ;
seuil gate max_abs ≤ 1e-3 — resserrable au vu de l'oracle, JAMAIS élargissable.
Témoin masques S=8 : le masque sliding HF == masque causal HF (fenêtre 1024 non
mordante, asserté bit) — le gate moteur (two_masks=false) utilise UNE table causale.

Sorties : fixtures/u_layers.safetensors (hidden bf16 + out_l0..out_l5 f32)
        + fixtures/u_layers_manifest.json.

Venv : /data/venvs/g12b. Lancement (VM) :
  /data/venvs/g12b/bin/python3 scripts/67_u6_layers_oracle.py
"""
import json
import os

os.environ.setdefault("HF_HOME", "/data/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from safetensors.torch import save_file

from _oracle_common import load_dq_weights  # module partagé (extrait au 3e usage, Task 6)

ROOT = "/data/gemma4-zml-probe"
DQ = os.path.join(ROOT, "weights_12b_dq")  # export dq (D8/D9) — PAS le packé
FX = os.path.join(ROOT, "fixtures")
SEED = 1337  # même graine que 65/66 — consignée au manifest
S = 8  # périmètre U6 (Amendement 2) : S=8 explicite
N_CHAIN = 6  # L0→L5 : 5 sliding + 1 full (L5)
FULL_LAYER = 5
N_TOTAL_LAYERS = 48  # distribution layer_scalar sur les 48 couches
MAX_ABS_THR = 1.0e-3  # §3 U6 — resserrable au vu de l'oracle, JAMAIS élargissable

LN_NAMES = ("input_layernorm", "post_attention_layernorm",
            "pre_feedforward_layernorm", "post_feedforward_layernorm")


def layer_keys(i, sliding):
    """Clés checkpoint d'une couche décodeur complète (v_proj seulement si sliding)."""
    base = f"model.language_model.layers.{i}."
    attn = ("q_proj", "k_proj", "o_proj", "q_norm", "k_norm") + (("v_proj",) if sliding else ())
    ks = [base + f"{n}.weight" for n in LN_NAMES]
    ks += [base + f"self_attn.{n}.weight" for n in attn]
    ks += [base + f"mlp.{n}.weight" for n in ("gate_proj", "up_proj", "down_proj")]
    ks += [base + "layer_scalar"]
    return base, ks


def main() -> None:
    import transformers
    import transformers.models.gemma4.modeling_gemma4 as mg
    from transformers import AutoConfig
    from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

    cfg = AutoConfig.from_pretrained(DQ).text_config
    cfg._attn_implementation = "eager"
    # Contrat U0 §1 (D1) : géométrie + absence PLE/MoE (bloc PLE du layer sous
    # `if self.hidden_size_per_layer_input:` — 0 => comptime-mort côté moteur aussi).
    assert cfg.hidden_size == 3840 and cfg.num_hidden_layers == 48
    assert not cfg.enable_moe_block, "MoE inattendu (contrat U0 : ni PLE ni YOCO ni MoE)"
    assert not cfg.hidden_size_per_layer_input, \
        f"PLE inattendu : hidden_size_per_layer_input={cfg.hidden_size_per_layer_input}"
    assert cfg.layer_types[:N_CHAIN] == ["sliding_attention"] * 5 + ["full_attention"], \
        f"motif L0-L5 inattendu : {cfg.layer_types[:N_CHAIN]}"

    # --- Distribution des 48 layer_scalar (manifest : valent-ils tous 1.0 ?) ---
    ls_keys = [f"model.language_model.layers.{i}.layer_scalar" for i in range(N_TOTAL_LAYERS)]
    ls_raw = load_dq_weights(ls_keys)
    ls_vals = []
    for k in ls_keys:
        t = ls_raw[k]
        assert t.dtype == torch.bfloat16 and list(t.shape) == [1], f"{k}: {t.dtype} {t.shape}"
        ls_vals.append(float(t.float().item()))
    all_ones = all(v == 1.0 for v in ls_vals)
    print(f"layer_scalar x{N_TOTAL_LAYERS} : min={min(ls_vals)} max={max(ls_vals)} "
          f"all_ones={all_ones}", flush=True)

    rotary = mg.Gemma4TextRotaryEmbedding(cfg)

    # --- Les 6 modules réels, poids export dq, f32 ---
    layers = []
    layer_scalar_used = []
    for i in range(N_CHAIN):
        sliding = i != FULL_LAYER
        lay = mg.Gemma4TextDecoderLayer(cfg, layer_idx=i)
        assert lay.self_attn.is_sliding == sliding and lay.self_attn.scaling == 1.0
        if sliding:
            assert lay.self_attn.v_proj is not None and lay.self_attn.num_key_value_groups == 2
            assert lay.self_attn.head_dim == 256
        else:
            assert lay.self_attn.v_proj is None and lay.self_attn.num_key_value_groups == 16
            assert lay.self_attn.head_dim == 512 and lay.self_attn.use_alternative_attention
        assert not any(True for _ in lay.self_attn.v_norm.parameters()), \
            "v_norm devrait être SANS poids (with_scale=False)"
        assert not lay.enable_moe_block and not lay.hidden_size_per_layer_input

        base, ks = layer_keys(i, sliding)
        dq = load_dq_weights(ks)
        sd = {k[len(base):]: v for k, v in dq.items()}
        for k, v in sd.items():
            assert v.dtype == torch.bfloat16, f"{base}{k}: {v.dtype} != bf16 (export dq)"
        hd = 256 if sliding else 512
        kvh = 8 if sliding else 1
        assert list(sd["self_attn.q_proj.weight"].shape) == [16 * hd, 3840]
        assert list(sd["self_attn.k_proj.weight"].shape) == [kvh * hd, 3840]
        assert list(sd["self_attn.o_proj.weight"].shape) == [3840, 16 * hd]
        assert list(sd["mlp.gate_proj.weight"].shape) == [15360, 3840]
        assert list(sd["mlp.down_proj.weight"].shape) == [3840, 15360]
        assert list(sd["layer_scalar"].shape) == [1]
        lay.load_state_dict(sd, strict=True)  # strict : couvre TOUTES les clés du module (dont layer_scalar)
        lay = lay.float()  # bf16 -> f32 EXACT (flux moteur dotPrec fam=null, même choix que U3/U5)
        lay.train(False)
        assert float(lay.layer_scalar.item()) == ls_vals[i], f"L{i}: layer_scalar module != checkpoint"
        layer_scalar_used.append(float(lay.layer_scalar.item()))
        layers.append(lay)
        print(f"L{i} ({'sliding' if sliding else 'full K=V'}) : module réel chargé "
              f"(strict), layer_scalar={layer_scalar_used[-1]}", flush=True)

    # --- Entrée + masques + cos/sin (mêmes mécaniques HF que 65/66) ---
    torch.manual_seed(SEED)
    hidden_bf16 = torch.randn(1, S, cfg.hidden_size, dtype=torch.float32).to(torch.bfloat16)
    h = hidden_bf16.to(torch.float32)
    pos = torch.arange(S, dtype=torch.long).unsqueeze(0)

    mask_kw = dict(config=cfg, inputs_embeds=h, attention_mask=None,
                   past_key_values=None, position_ids=pos)
    mask_sl = create_sliding_window_causal_mask(**mask_kw)
    mask_fl = create_causal_mask(**mask_kw)
    assert isinstance(mask_sl, torch.Tensor) and list(mask_sl.shape) == [1, 1, S, S]
    assert isinstance(mask_fl, torch.Tensor) and list(mask_fl.shape) == [1, 1, S, S]
    mask_sl = mask_sl.to(torch.float32)
    mask_fl = mask_fl.to(torch.float32)
    causal = S * (S - 1) // 2
    n_sl = int((mask_sl < -1.0e30).sum().item())
    n_fl = int((mask_fl < -1.0e30).sum().item())
    assert n_sl == causal and n_fl == causal, f"masques S={S} : {n_sl}/{n_fl} != causal {causal}"
    # Témoin S=8 : fenêtre 1024 non mordante => sliding == full (bit). Le gate moteur
    # (two_masks=false) utilise UNE table causale pour les 6 couches — licite ssi cet assert tient.
    assert torch.equal(mask_sl, mask_fl), "S=8 : masque sliding != causal (fenêtre mordante ??)"

    cos_sl, sin_sl = rotary(h, pos, layer_type="sliding_attention")
    cos_fl, sin_fl = rotary(h, pos, layer_type="full_attention")
    assert list(cos_sl.shape) == [1, S, 256] and list(cos_fl.shape) == [1, S, 512]
    assert cos_sl.dtype == torch.float32 and cos_fl.dtype == torch.float32

    # --- Chaîne L0→L5 (hidden → hidden), expected = sortie de CHAQUE couche ---
    outs = []
    with torch.no_grad():
        for i, lay in enumerate(layers):
            sliding = i != FULL_LAYER
            h = lay(
                h,
                per_layer_input=None,
                shared_kv_states=None,
                position_embeddings=(cos_sl, sin_sl) if sliding else (cos_fl, sin_fl),
                attention_mask=mask_sl if sliding else mask_fl,
                position_ids=pos,
            )
            assert isinstance(h, torch.Tensor) and list(h.shape) == [1, S, 3840]
            assert h.dtype == torch.float32 and torch.isfinite(h).all(), f"L{i}: sortie non f32/finie"
            outs.append(h.detach().clone())
            print(f"L{i} : |out| max={float(h.abs().max()):.4f} "
                  f"out[0,0,:3]={[float(x) for x in h[0, 0, :3]]}", flush=True)

    tensors = {"hidden": hidden_bf16.contiguous()}
    for i, o in enumerate(outs):
        tensors[f"out_l{i}"] = o.contiguous()
    os.makedirs(FX, exist_ok=True)
    save_file(tensors, os.path.join(FX, "u_layers.safetensors"))

    manifest = {
        "source": "67_u6_layers_oracle.py (Task 6, plan J2 — Amendement 2 + arbitrage 23618bb)",
        "oracle": "6x Gemma4TextDecoderLayer(text_config, layer_idx=0..5) RÉELS, poids export dq "
                  "(D9), modules f32 (flux moteur bf16->f32, précédent déclaré U3/U5) ; expected = "
                  "SORTIE de chaque couche, ENCHAÎNÉE (out_l{i} = entrée de la couche i+1) ; "
                  "couche 0 seule == premier maillon (même entrée : hidden) ; chaîne == out_l5 ; "
                  "layer_scalar chargé du checkpoint par load_state_dict strict et appliqué par le "
                  "module réel (hidden *= layer_scalar, fin de couche)",
        "mask": "create_sliding_window_causal_mask / create_causal_mask (transformers 5.14, "
                "mask_kwargs du forward Gemma4TextModel) — S=8 : sliding == causal BIT-ÉGAL "
                "(asserté ; fenêtre 1024 non mordante) => le gate moteur two_masks=false "
                "(une table causale) est licite",
        "layouts": {
            "hidden": "[1,8,3840] bf16 (graine consignée, modules nourris en f32)",
            "out_l0..out_l5": "[1,8,3840] f32 — sortie de couche i de la CHAÎNE (layer_scalar inclus)",
        },
        "geometry": {
            "chain": "L0-L4 sliding (GQA 8KVx256 groupe 2, rope default 1e4) + L5 full "
                     "(MQA 1KVx512 K=V, p-RoPE proportional 1e6 partial 0.25)",
            "layer_scalar": "buffer [1] bf16 par couche, applique en sortie de couche",
            "ple": 0, "moe": False, "d": 3840, "inter": 15360,
        },
        "seed": SEED, "seq_len": S,
        "thresholds": {"u6_max_abs": MAX_ABS_THR,
                       "note": "§3 U6 : max_abs <= 1e-3 par couche ET chaîne — resserrable au vu "
                               "de l'oracle, JAMAIS élargissable ; un dépassement = FAIL"},
        "layer_scalar_distribution": {
            "n": N_TOTAL_LAYERS, "values": ls_vals,
            "min": min(ls_vals), "max": max(ls_vals), "all_ones": all_ones,
            "n_not_one": sum(1 for v in ls_vals if v != 1.0),
            "used_by_chain_l0_l5": layer_scalar_used,
        },
        "points_fixes": {
            f"out_l{i}[0,{r},:3]": [float(x) for x in outs[i][0, r, :3]]
            for i in (0, 5) for r in (0, S - 1)
        },
        "pass": "gate u6 (MOTEUR — arbitrage 23618bb) : chaîne via forwardStageGen/Geom tronqué "
                "6 couches (runLayerGen partagé), branche k_eq_v_full DANS le graphe sur L5, "
                "placeholder v_proj [1] non consommé (preuve = compile) ; comparaison par couche "
                "(profondeur de chaîne) + chaîne au seuil 1e-3",
        "versions": {"transformers": transformers.__version__, "torch": torch.__version__},
    }
    with open(os.path.join(FX, "u_layers_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    sz = os.path.getsize(os.path.join(FX, "u_layers.safetensors")) / 1.0e6
    print(f"PASS fixture U6 écrite : S={S}, 6 couches -> fixtures/u_layers.safetensors ({sz:.1f} Mo) ; "
          f"layer_scalar all_ones={all_ones}", flush=True)


if __name__ == "__main__":
    main()
