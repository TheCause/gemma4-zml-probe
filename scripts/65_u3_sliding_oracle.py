#!/usr/bin/env python3
"""U3/U4 — oracle attention sliding 12B couche 0 : module RÉEL Gemma4TextAttention (piège 12).

Module réel `Gemma4TextAttention(text_config, layer_idx=0)` (transformers 5.14.1, venv g12b),
poids chargés depuis l'EXPORT dq (D9 : oracles Python = export, gates Zig = PACKÉ), via
`safe_open` sur le shard désigné par `model.safetensors.index.json` (2 shards). Le module est
passé en **f32** (poids bf16 → f32 exact) : le flux comparé est celui du moteur ZML
(`dotPrec fam=null` : opérandes bf16 convertis f32, GEMM f32) — précédent W3 (script 58) et
P5.2 E2B ; les seuils §3 (max_abs ≤ 1e-4, mean_abs ≤ 1e-6) sont des seuils f32, un oracle
laissé en bf16 porterait ~4e-3 de bruit de quantization par étage et les rendrait inatteignables.

Intermédiaires capturés par HOOKS sur le module réel — JAMAIS recomposés (piège 12) :
  (a) q_states post q_proj+q_norm+rope  — wrapper sur `apply_rotary_pos_emb` (1er appel du
      forward = Q), layout [B,S,16,256] (pré-transpose, unsqueeze_dim=2) ;
  (b) k_states post k_proj+k_norm+rope  — même wrapper (2e appel = K), [B,S,8,256] ;
  (c) v_states post v_norm              — forward hook sur `module.v_norm`, [B,S,8,256]
      (v_norm SANS poids, with_scale=False — l'op RMSNorm existe quand même, piège 9) ;
  (d) sortie o_proj                     — forward hook sur `module.o_proj`, [B,S,3840].
Témoins du chemin eager : wrapper sur `eager_attention_forward` (doit être appelé) +
`attn_weights is not None` (sdpa renverrait None).

Masque : la VRAIE fonction de transformers 5.14 `create_sliding_window_causal_mask`, appelée
avec exactement les `mask_kwargs` du forward de `Gemma4TextModel` (config/inputs_embeds/
attention_mask=None/past_key_values=None/position_ids) — additive f32 [1,1,S,S]. Témoin de
morsure (S=1040 > fenêtre 1024) : comptage masqué ASSERTÉ == causal pur + Σ_{q≥1024}(q-1023)
= 540280 + 136 (S=8 : == causal pur 28, la fenêtre ne mord pas — consigné, pas un témoin).

Sorties : fixtures/u_sliding.safetensors (S=8 ET S=1040 : hidden bf16, étages a-d f32,
masques f32) + fixtures/u_sliding_manifest.json (seuils, seed, comptages, points fixes, versions).

Périmètre du gating U3 (Amendement 2 du plan) : le plan était MUET sur le S du gating —
l'interprétation retenue (gating S=8 aux seuils §3, tripwire 1e-3 à S=1040 sans mean_abs)
est une décision d'interprétation DÉCLARÉE, ratification Régis en attente ; le chemin rope
à S=1040 est gaté au seuil plein par U4 (cas mordant).

Venv : /data/venvs/g12b. Lancement (VM) :
  /data/venvs/g12b/bin/python3 scripts/65_u3_sliding_oracle.py
"""
import json
import os

os.environ.setdefault("HF_HOME", "/data/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = "/data/gemma4-zml-probe"
DQ = os.path.join(ROOT, "weights_12b_dq")  # export dq (D8/D9) — PAS le packé
FX = os.path.join(ROOT, "fixtures")
SEED = 1337  # même graine que le précédent W3 (script 58)
SEQ_LENS = (8, 1040)  # 1040 > fenêtre 1024 : cas mordant (témoin U4)
SLIDING_WINDOW = 1024
LAYER = 0
PREFIX = f"model.language_model.layers.{LAYER}.self_attn."
MAX_ABS_THR, MEAN_ABS_THR = 1.0e-4, 1.0e-6  # §3 U3 (U4 : max_abs seul)


def load_dq_weights(keys):
    """Lit les clés depuis l'export dq 2-shards via l'index JSON (safe_open par shard)."""
    with open(os.path.join(DQ, "model.safetensors.index.json")) as fh:
        wmap = json.load(fh)["weight_map"]
    out, by_shard = {}, {}
    for k in keys:
        by_shard.setdefault(wmap[k], []).append(k)
    for shard, ks in by_shard.items():
        with safe_open(os.path.join(DQ, shard), framework="pt") as f:
            for k in ks:
                out[k] = f.get_tensor(k)
    return out


def expected_mask_counts(s):
    causal = s * (s - 1) // 2
    bite = sum(q - (SLIDING_WINDOW - 1) for q in range(SLIDING_WINDOW, s))
    return causal, bite


def main() -> None:
    import transformers
    import transformers.models.gemma4.modeling_gemma4 as mg
    from transformers import AutoConfig
    from transformers.masking_utils import create_sliding_window_causal_mask

    cfg = AutoConfig.from_pretrained(DQ).text_config
    cfg._attn_implementation = "eager"
    assert (cfg.head_dim, cfg.num_attention_heads, cfg.num_key_value_heads) == (256, 16, 8)
    assert cfg.sliding_window == SLIDING_WINDOW and cfg.layer_types[LAYER] == "sliding_attention"

    # Module RÉEL (piège 12), couche 0 sliding, poids de l'EXPORT dq, module passé f32.
    mod = mg.Gemma4TextAttention(cfg, layer_idx=LAYER)
    assert mod.is_sliding and mod.scaling == 1.0 and mod.num_key_value_groups == 2, \
        f"contrat U0 violé : is_sliding={mod.is_sliding} scaling={mod.scaling} groups={mod.num_key_value_groups}"
    assert mod.v_proj is not None and mod.sliding_window == SLIDING_WINDOW
    assert not any(True for _ in mod.v_norm.parameters()), "v_norm devrait être SANS poids (with_scale=False)"

    sd_keys = {f"{p}.weight": PREFIX + f"{p}.weight"
               for p in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm")}
    dq = load_dq_weights(list(sd_keys.values()))
    for k in sd_keys.values():
        assert dq[k].dtype == torch.bfloat16, f"{k}: {dq[k].dtype} != bf16 (export dq)"
    assert list(dq[PREFIX + "q_proj.weight"].shape) == [4096, 3840]
    assert list(dq[PREFIX + "k_proj.weight"].shape) == [2048, 3840]
    assert list(dq[PREFIX + "v_proj.weight"].shape) == [2048, 3840]
    assert list(dq[PREFIX + "o_proj.weight"].shape) == [3840, 4096]
    assert list(dq[PREFIX + "q_norm.weight"].shape) == [256] and list(dq[PREFIX + "k_norm.weight"].shape) == [256]
    mod.load_state_dict({k: dq[v] for k, v in sd_keys.items()}, strict=True)
    mod = mod.float()  # bf16 -> f32 EXACT (flux moteur dotPrec fam=null)
    mod.train(False)

    rotary = mg.Gemma4TextRotaryEmbedding(cfg)  # buffers inv_freq f32, theta sliding 1e4 (default)

    # --- HOOKS sur le module réel (aucun expected recomposé) ---
    cap = {}
    orig_rope = mg.apply_rotary_pos_emb

    def rope_hook(x, cos, sin, unsqueeze_dim=1):
        out = orig_rope(x, cos, sin, unsqueeze_dim=unsqueeze_dim)
        cap.setdefault("rope", []).append((unsqueeze_dim, out.detach().clone()))
        return out

    orig_eager = mg.eager_attention_forward

    def eager_hook(*a, **kw):
        cap["eager_called"] = cap.get("eager_called", 0) + 1
        return orig_eager(*a, **kw)

    mg.apply_rotary_pos_emb = rope_hook
    mg.eager_attention_forward = eager_hook
    h_v = mod.v_norm.register_forward_hook(lambda m, i, o: cap.__setitem__("v", o.detach().clone()))
    h_o = mod.o_proj.register_forward_hook(lambda m, i, o: cap.__setitem__("out", o.detach().clone()))

    tensors, manifest_cases = {}, {}
    torch.manual_seed(SEED)
    try:
        for s in SEQ_LENS:
            cap.clear()
            hidden_bf16 = torch.randn(1, s, cfg.hidden_size, dtype=torch.float32).to(torch.bfloat16)
            hidden = hidden_bf16.to(torch.float32)
            pos = torch.arange(s, dtype=torch.long).unsqueeze(0)

            # Masque : la VRAIE mécanique HF (mêmes kwargs que Gemma4TextModel.forward l.1697).
            mask = create_sliding_window_causal_mask(
                config=cfg, inputs_embeds=hidden, attention_mask=None,
                past_key_values=None, position_ids=pos,
            )
            assert isinstance(mask, torch.Tensor) and list(mask.shape) == [1, 1, s, s], \
                f"masque S={s} : {type(mask)} {getattr(mask, 'shape', None)}"
            mask = mask.to(torch.float32)
            n_masked = int((mask < -1.0e30).sum().item())
            causal, bite = expected_mask_counts(s)
            assert n_masked == causal + bite, \
                f"S={s} : {n_masked} masqués != causal {causal} + morsure {bite}"
            if s > SLIDING_WINDOW:
                assert bite > 0 and n_masked > causal, f"S={s} : la fenêtre ne mord pas (vacuité)"

            cos, sin = rotary(hidden, pos, layer_type="sliding_attention")
            assert cos.dtype == torch.float32, f"cos {cos.dtype}"
            with torch.no_grad():
                attn_out, attn_w = mod(hidden, (cos, sin), mask, {}, None)

            # Témoins du chemin eager + câblage des hooks.
            assert cap.get("eager_called") == 1, f"eager_attention_forward non appelé (S={s})"
            assert attn_w is not None, "attn_weights None : chemin non-eager"
            assert len(cap["rope"]) == 2, f"{len(cap['rope'])} appels rope != 2 (Q puis K)"
            (ud_q, q_states), (ud_k, k_states) = cap["rope"]
            assert ud_q == 2 and ud_k == 2, f"unsqueeze_dim {ud_q}/{ud_k} != 2 (layout [B,S,H,D])"
            assert list(q_states.shape) == [1, s, 16, 256] and list(k_states.shape) == [1, s, 8, 256]
            v_states = cap["v"]
            assert list(v_states.shape) == [1, s, 8, 256]
            out = cap["out"]
            assert torch.equal(out, attn_out), "hook o_proj != sortie module"
            for name, t in (("q", q_states), ("k", k_states), ("v", v_states), ("out", out)):
                assert t.dtype == torch.float32 and torch.isfinite(t).all(), f"{name} S={s} non f32/fini"

            sfx = f"_s{s}"
            tensors["hidden" + sfx] = hidden_bf16.contiguous()
            tensors["mask" + sfx] = mask.contiguous()
            tensors["q" + sfx] = q_states.contiguous()
            tensors["k" + sfx] = k_states.contiguous()
            tensors["v" + sfx] = v_states.contiguous()
            tensors["out" + sfx] = out.contiguous()
            manifest_cases[f"S={s}"] = {
                "mask_counts": {"masked": n_masked, "causal_pure": causal, "sliding_bite": bite,
                                "morsure": "TEMOIN U4 (fenetre franchie)" if bite else "fenetre non mordante (== causal pur)"},
                "points_fixes": {
                    "out[0,0,:3]": [float(x) for x in out[0, 0, :3]],
                    "out[0,-1,3837:]": [float(x) for x in out[0, -1, 3837:]],
                    "q[0,-1,0,:3]": [float(x) for x in q_states[0, -1, 0, :3]],
                    "k[0,-1,7,:3]": [float(x) for x in k_states[0, -1, 7, :3]],
                    "v[0,0,0,:3]": [float(x) for x in v_states[0, 0, 0, :3]],
                    "mask[0,0,-1,:3]": [float(x) for x in mask[0, 0, -1, :3]],
                },
            }
            print(f"S={s} : masqués {n_masked} (causal {causal} + morsure {bite}), "
                  f"étages a-d capturés par hooks, eager témoin OK", flush=True)
    finally:
        mg.apply_rotary_pos_emb = orig_rope
        mg.eager_attention_forward = orig_eager
        h_v.remove()
        h_o.remove()

    os.makedirs(FX, exist_ok=True)
    save_file(tensors, os.path.join(FX, "u_sliding.safetensors"))

    manifest = {
        "source": "65_u3_sliding_oracle.py (Task 4, plan J2 amendé 2026-07-24)",
        "oracle": "Gemma4TextAttention(text_config, layer_idx=0) RÉEL, poids export dq (D9), "
                  "module f32 (flux moteur bf16->f32, précédent W3/P5.2) ; intermédiaires par "
                  "HOOKS (apply_rotary_pos_emb x2, v_norm, o_proj), témoin eager asserté",
        "mask": "create_sliding_window_causal_mask (transformers 5.14, mask_kwargs du forward "
                "Gemma4TextModel), additive f32 [1,1,S,S]",
        "layouts": {
            "hidden": "[1,S,3840] bf16 (graine consignée, module nourri en f32)",
            "q": "[1,S,16,256] f32 post q_proj+q_norm+rope, PRE-transpose (unsqueeze_dim=2)",
            "k": "[1,S,8,256] f32 post k_proj+k_norm+rope, PRE-transpose",
            "v": "[1,S,8,256] f32 post v_norm (RMSNorm SANS poids), PRE-transpose",
            "out": "[1,S,3840] f32 sortie o_proj (== sortie module, asserté)",
        },
        "geometry": {"gqa_groups": 2, "nh": 16, "kvh": 8, "hd": 256, "sliding_window": SLIDING_WINDOW,
                     "scaling": 1.0, "rope": "default theta 1e4, 256 dims pleines", "layer": LAYER},
        "seed": SEED, "seq_lens": list(SEQ_LENS),
        "thresholds": {"u3_max_abs": MAX_ABS_THR, "u3_mean_abs": MEAN_ABS_THR, "u4_max_abs": MAX_ABS_THR,
                       "note": "seuils f32 (§3) — comparaisons f32, PAS bit-exact (matmuls)"},
        "cases": manifest_cases,
        "pass": "gate u3 : étages a-c (q/k/v) aux seuils §3 sur S=8 — périmètre : le plan "
                "était MUET sur le S du gating, interprétation (gating S=8, tripwire 1e-3 à "
                "S=1040 sans mean_abs) déclarée en Amendement 2, ratification Régis en "
                "attente ; le chemin rope à S=1040 est gaté au seuil plein par U4 ; gate u4 : "
                "attention complète (étage d) max_abs <= 1e-4 sur S=8 ET S=1040 (cas "
                "mordant), comptage masque recompté in-gate",
        "finding_ulp_rope_2026_07_24": {
            "fait": "zml.nn.rope (moteur, chemin sliding) génère inv_freq par exp(-log(theta)*n/N) "
                    "f32 ; HF par theta**(-n/N) f32 — écart 1 ULP (max 5.96e-8, relatif 5.7e-7), "
                    "amplifié linéairement par la position : delta-angle = pos*delta-inv "
                    "(6.1e-5 rad a pos 708) -> etage a S=1040 max_abs mesure 4.9e-4 > seuil §3",
            "diagnostic": "pas un bug de cablage (S=8 : marges x26-x134) ; verifie au chiffre : "
                          "d_inv[3]=5.96e-8, d_cos@708=5.76e-5, pire ecart a hd=131 (freq imag 3)",
            "consequence": "l'effet s'annule en rotation RELATIVE dans les scores QK : U4 S=1040 "
                           "mordant tient max_abs 2.78e-5 << 1e-4 ; sans impact decode (bf16 ~4e-3, "
                           "U8/U9 = argmax)",
        },
        "versions": {"transformers": transformers.__version__, "torch": torch.__version__},
    }
    with open(os.path.join(FX, "u_sliding_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    sz = os.path.getsize(os.path.join(FX, "u_sliding.safetensors")) / 1.0e6
    print(f"PASS fixture U3/U4 écrite : S={SEQ_LENS} -> fixtures/u_sliding.safetensors ({sz:.1f} Mo)", flush=True)


if __name__ == "__main__":
    main()
