#!/usr/bin/env python3
"""U5 — oracle attention FULL 12B couche 5 : module RÉEL Gemma4TextAttention (MQA 16Q×512 /
1KV×512, K=V, p-RoPE proportional 512/0.25/1e6). Piège 12 : jamais de recomposition.

Module réel `Gemma4TextAttention(text_config, layer_idx=5)` (transformers 5.14.1, venv g12b),
poids chargés depuis l'EXPORT dq (D9 : oracles Python = export, gates Zig = PACKÉ), via
`safe_open` sur les shards désignés par `model.safetensors.index.json`. Le module est passé
en **f32** (poids bf16 → f32 exact) : le flux comparé est celui du moteur ZML (`dotPrec
fam=null` : opérandes bf16 convertis f32, GEMM f32) — même choix déclaré que U3 (script 65).

Intermédiaires capturés par HOOKS sur le module réel — JAMAIS recomposés (piège 12) :
  (a) q_states post q_proj+q_norm+p-rope — wrapper sur `apply_rotary_pos_emb` (1er appel du
      forward = Q), layout [B,S,16,512] (pré-transpose, unsqueeze_dim=2) ;
  (b) k_states FINAL post k_proj+k_norm+p-rope — même wrapper (2e appel = K), [B,S,1,512] ;
  (v) value_states == v_norm(k_proj BRUT) réel — forward hook sur `module.v_norm`, capturé
      AVANT transpose, [B,S,1,512]. TÉMOIN de source : l'ENTRÉE du hook v_norm est assertée
      bit-égale au k_proj brut capturé (hook sur `module.k_proj`) — prouve que V part de kp
      AVANT k_norm et AVANT rope (modeling_gemma4.py 5.14.1 : `value_states = ... if
      self.v_proj is not None else key_states`, exécuté avant `key_states = self.k_norm(...)`) ;
  (c) sortie attention complète — forward hook sur `module.o_proj`, [B,S,3840].
Témoins du chemin eager : wrapper sur `eager_attention_forward` + `attn_weights is not None`.

TEST DE DISCRIMINABILITÉ CÂBLÉ (non-vacuité R3 — option B ratifiée par Régis, plan commit
04eaa8d) : la fixture K=V doit POUVOIR échouer. Le critère se mesure sur la SORTIE d'attention
(l'étage qui gate) : le module réel est re-forwardé avec l'hypothèse FAUSSE historique
(V = v_norm(k_norm(kp)), bug R3) injectée dans v_norm, et
`max_abs(out_vrai - out_faux) >= 10x le seuil` (10 × 1e-4 = 1e-3), sinon sys.exit(1) fixture
NON-DISCRIMINANTE. Historique de la décision (NEEDS_DECISION du 25 juil) : le critère initial
au niveau v_norm supposait des poids k_norm non uniformes — FAIT de checkpoint découvert : les
q/k_norm des 8 couches full sont UNIFORMES (artefact QAT, n_unique=1, vérifié sur le packé
officiel), le discriminant v_norm résiduel (rupture de scale-invariance par l'eps du RMSNorm)
mesure x7,4 seulement à S=8. La mesure v_norm per-S ET le témoin structurel bit-exact restent
consignés au manifest — ils DOCUMENTENT ; seule la sortie gate le x10.

p-RoPE : cos/sin du module réel `Gemma4TextRotaryEmbedding` (rope proportional theta 1e6,
partial 0.25 sur head_dim 512 → 64 fréquences actives, 192 zéros dans inv_freq — vérifié sur
les buffers). Sanity position 0 : identité STRICTE (cos==1, sin==0 partout, comparaison
exacte). Les cos/sin S=1040 sont exportés dans la fixture : le gate Zig les compare à SES
cos/sin host (formule du RUNNER, w4auto ropeFull) — tripwire ULP positions {708, 1030} à 1e-3.

Sorties : fixtures/u_full.safetensors (S=8 ET S=1040 : hidden bf16, étages a/b/v/c f32,
masque causal f32, cos/sin f32) + fixtures/u_full_manifest.json.

Venv : /data/venvs/g12b. Lancement (VM) :
  /data/venvs/g12b/bin/python3 scripts/66_u5_full_oracle.py
"""
import json
import os
import sys

os.environ.setdefault("HF_HOME", "/data/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = "/data/gemma4-zml-probe"
DQ = os.path.join(ROOT, "weights_12b_dq")  # export dq (D8/D9) — PAS le packé
FX = os.path.join(ROOT, "fixtures")
SEED = 1337  # même graine que 65 (U3/U4) — consignée au manifest
SEQ_LENS = (8, 1040)  # gating S=8 ; S=1040 : étage (c) au seuil plein + cos/sin tripwire
TRIPWIRE_POS = (708, 1030)  # positions ULP (708 = pire point mesuré U3 ; 1030 < 1040)
LAYER = 5
PREFIX = f"model.language_model.layers.{LAYER}.self_attn."
MAX_ABS_THR, MEAN_ABS_THR = 1.0e-4, 1.0e-6  # §3 U5 (étage c : max_abs seul)
DISCRIM_THR = 10.0 * MAX_ABS_THR  # >= 10x le seuil, sinon fixture non-discriminante


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


def main() -> None:
    import transformers
    import transformers.models.gemma4.modeling_gemma4 as mg
    from transformers import AutoConfig
    from transformers.masking_utils import create_causal_mask

    cfg = AutoConfig.from_pretrained(DQ).text_config
    cfg._attn_implementation = "eager"
    # Contrat U0 §1 (D1) : géométrie full + rope proportional — vérifié sur le config réel.
    assert (cfg.global_head_dim, cfg.num_attention_heads, cfg.num_global_key_value_heads) == (512, 16, 1)
    assert cfg.attention_k_eq_v is True and cfg.layer_types[LAYER] == "full_attention"
    rp = cfg.rope_parameters["full_attention"]
    assert rp == {"partial_rotary_factor": 0.25, "rope_theta": 1.0e6, "rope_type": "proportional"}, \
        f"rope_parameters full inattendus : {rp}"

    # Module RÉEL (piège 12), couche 5 full, poids de l'EXPORT dq, module passé f32.
    mod = mg.Gemma4TextAttention(cfg, layer_idx=LAYER)
    assert not mod.is_sliding and mod.use_alternative_attention and mod.head_dim == 512
    assert mod.scaling == 1.0 and mod.num_key_value_groups == 16, \
        f"contrat U0 violé : scaling={mod.scaling} groups={mod.num_key_value_groups}"
    assert mod.v_proj is None, "couche 5 full : v_proj devrait être ABSENT (attention_k_eq_v)"
    assert not any(True for _ in mod.v_norm.parameters()), "v_norm devrait être SANS poids (with_scale=False)"

    sd_keys = {f"{p}.weight": PREFIX + f"{p}.weight"
               for p in ("q_proj", "k_proj", "o_proj", "q_norm", "k_norm")}
    dq = load_dq_weights(list(sd_keys.values()))
    for k in sd_keys.values():
        assert dq[k].dtype == torch.bfloat16, f"{k}: {dq[k].dtype} != bf16 (export dq)"
    assert list(dq[PREFIX + "q_proj.weight"].shape) == [8192, 3840]
    assert list(dq[PREFIX + "k_proj.weight"].shape) == [512, 3840]
    assert list(dq[PREFIX + "o_proj.weight"].shape) == [3840, 8192]
    assert list(dq[PREFIX + "q_norm.weight"].shape) == [512] and list(dq[PREFIX + "k_norm.weight"].shape) == [512]
    mod.load_state_dict({k: dq[v] for k, v in sd_keys.items()}, strict=True)
    mod = mod.float()  # bf16 -> f32 EXACT (flux moteur dotPrec fam=null, même choix que U3)
    mod.train(False)

    # FAIT de checkpoint (découvert au 1er run, consigné au plan avec l'option B) : les poids
    # q_norm/k_norm des 8 couches full sont UNIFORMES (n_unique=1, artefact QAT, vérifié sur le
    # packé officiel) — le mécanisme « non-uniformité de k_norm » du critère initial est mort ;
    # d'où le critère sur la SORTIE. Valeurs exactes consignées au manifest.
    knw = mod.k_norm.weight.detach()
    qnw = mod.q_norm.weight.detach()
    k_norm_spread = float((knw.max() - knw.min()).item())
    uniform_norms_fact = {
        "k_norm_L5_n_unique": int(torch.unique(knw).numel()),
        "k_norm_L5_value": float(knw[0].item()),
        "q_norm_L5_n_unique": int(torch.unique(qnw).numel()),
        "q_norm_L5_value": float(qnw[0].item()),
        "k_norm_spread": k_norm_spread,
        "note": "les 8 couches full (5,11,...,47) ont q/k_norm uniformes (mesuré 25 juil : "
                "k_norm ∈ {0.060546875..0.0654296875}, q_norm ∈ {0.953125..1.03125} selon la "
                "couche, n_unique=1 partout, contre-vérifié sur le packé officiel) — artefact "
                "QAT ; le discriminant v_norm résiduel = rupture de scale-invariance par l'eps "
                "du RMSNorm, x7,4 seuil à S=8 -> critère déplacé sur la sortie (option B)",
    }

    rotary = mg.Gemma4TextRotaryEmbedding(cfg)
    assert rotary.rope_type["full_attention"] == "proportional"
    assert float(getattr(rotary, "full_attention_attention_scaling")) == 1.0
    inv_freq = getattr(rotary, "full_attention_inv_freq")
    assert list(inv_freq.shape) == [256] and inv_freq.dtype == torch.float32
    assert int((inv_freq != 0).sum().item()) == 64 and bool((inv_freq[64:] == 0).all().item()), \
        "p-RoPE partial 0.25 : attendu 64 fréquences actives + 192 zéros (nope)"

    # --- HOOKS sur le module réel (aucun expected recomposé) — pattern 65 : try/finally ---
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

    def v_norm_hook(m, inputs, output):
        cap["v_in"] = inputs[0].detach().clone()  # TÉMOIN : doit == k_proj BRUT (avant k_norm/rope)
        cap["v"] = output.detach().clone()  # value_states réel = v_norm(kp), AVANT transpose

    mg.apply_rotary_pos_emb = rope_hook
    mg.eager_attention_forward = eager_hook
    h_kp = mod.k_proj.register_forward_hook(lambda m, i, o: cap.__setitem__("kp", o.detach().clone()))
    h_v = mod.v_norm.register_forward_hook(v_norm_hook)
    h_o = mod.o_proj.register_forward_hook(lambda m, i, o: cap.__setitem__("out", o.detach().clone()))

    tensors, manifest_cases = {}, {}
    torch.manual_seed(SEED)
    try:
        for s in SEQ_LENS:
            cap.clear()
            hidden_bf16 = torch.randn(1, s, cfg.hidden_size, dtype=torch.float32).to(torch.bfloat16)
            hidden = hidden_bf16.to(torch.float32)
            pos = torch.arange(s, dtype=torch.long).unsqueeze(0)

            # Masque couche full : la VRAIE mécanique HF (create_causal_mask, mêmes mask_kwargs
            # que Gemma4TextModel.forward l.1697-1707) — causal pur, additive f32 [1,1,S,S].
            mask = create_causal_mask(
                config=cfg, inputs_embeds=hidden, attention_mask=None,
                past_key_values=None, position_ids=pos,
            )
            assert isinstance(mask, torch.Tensor) and list(mask.shape) == [1, 1, s, s]
            mask = mask.to(torch.float32)
            n_masked = int((mask < -1.0e30).sum().item())
            causal = s * (s - 1) // 2
            assert n_masked == causal, f"S={s} : {n_masked} masqués != causal pur {causal} (couche full)"

            # p-RoPE du module réel + sanity pos 0 = identité STRICTE (comparaison exacte).
            cos, sin = rotary(hidden, pos, layer_type="full_attention")
            assert cos.dtype == torch.float32 and list(cos.shape) == [1, s, 512]
            assert bool((cos[0, 0] == 1.0).all().item()) and bool((sin[0, 0] == 0.0).all().item()), \
                f"S={s} : p-RoPE position 0 != identité stricte"
            # 384 colonnes identité à TOUTE position (nope dims 64:256 dupliquées 320:512).
            nope_ok = bool((cos[0, :, 64:256] == 1.0).all().item()) and bool((sin[0, :, 64:256] == 0.0).all().item()) \
                and bool((cos[0, :, 320:512] == 1.0).all().item()) and bool((sin[0, :, 320:512] == 0.0).all().item())
            assert nope_ok, f"S={s} : colonnes nope non identité (partial 0.25 mal appliqué ?)"

            with torch.no_grad():
                attn_out, attn_w = mod(hidden, (cos, sin), mask, {}, None)

            # Témoins du chemin eager + câblage des hooks.
            assert cap.get("eager_called") == 1, f"eager_attention_forward non appelé (S={s})"
            assert attn_w is not None, "attn_weights None : chemin non-eager"
            assert len(cap["rope"]) == 2, f"{len(cap['rope'])} appels rope != 2 (Q puis K)"
            (ud_q, q_states), (ud_k, k_states) = cap["rope"]
            assert ud_q == 2 and ud_k == 2, f"unsqueeze_dim {ud_q}/{ud_k} != 2 (layout [B,S,H,D])"
            assert list(q_states.shape) == [1, s, 16, 512] and list(k_states.shape) == [1, s, 1, 512]
            kp = cap["kp"].view(1, s, 1, 512)  # k_proj BRUT (sortie Linear, avant k_norm/rope)
            v_states = cap["v"]
            assert list(v_states.shape) == [1, s, 1, 512]
            # TÉMOIN K=V (contrat §2) : l'entrée réelle de v_norm == k_proj BRUT, bit-égal —
            # V part de kp AVANT k_norm et AVANT rope (pas du key_states normé/roté).
            assert torch.equal(cap["v_in"], kp), f"S={s} : entrée v_norm != k_proj brut (câblage K=V ?)"
            out = cap["out"]
            assert torch.equal(out, attn_out), "hook o_proj != sortie module"

            # === TEST DE DISCRIMINABILITÉ CÂBLÉ (option B ratifiée, plan 04eaa8d) ===
            # Le critère x10 se mesure sur la SORTIE d'attention (l'étage qui gate) : re-forward
            # du module RÉEL avec l'hypothèse FAUSSE historique (bug R3 : V = v_norm(k_norm(kp)))
            # injectée dans v_norm. La mesure au niveau v_norm est CONSERVÉE, documentaire.
            with torch.no_grad():
                v_wrong = mod.v_norm(mod.k_norm(kp))
            discrim_vnorm = float((v_states - v_wrong).abs().max().item())  # documentaire (pas le gate)
            orig_vn_fwd = mod.v_norm.forward

            def wrong_vn(x):
                return orig_vn_fwd(mod.k_norm(x))

            mod.v_norm.forward = wrong_vn
            try:
                with torch.no_grad():
                    out_wrong, _ = mod(hidden, (cos, sin), mask, {}, None)
            finally:
                del mod.v_norm.forward  # retire l'attribut d'instance, restaure le forward de classe
            discrim_out = float((out - out_wrong).abs().max().item())
            if discrim_out < DISCRIM_THR:
                print(f"FAIL U5 : fixture NON-DISCRIMINANTE à S={s} — hypothèse fausse propagée à la "
                      f"SORTIE : max_abs = {discrim_out:.3e} < {DISCRIM_THR:.1e} "
                      f"(niveau v_norm documentaire : {discrim_vnorm:.3e} ; spread poids k_norm = {k_norm_spread:.3e})",
                      flush=True)
                sys.exit(1)
            print(f"S={s} : discriminabilité K=V (SORTIE) = {discrim_out:.3e} >= {DISCRIM_THR:.1e} "
                  f"(x{discrim_out / MAX_ABS_THR:.0f} le seuil ; niveau v_norm documentaire = "
                  f"{discrim_vnorm:.3e}, x{discrim_vnorm / MAX_ABS_THR:.1f})", flush=True)

            for name, t in (("q", q_states), ("k", k_states), ("v", v_states), ("out", out)):
                assert t.dtype == torch.float32 and torch.isfinite(t).all(), f"{name} S={s} non f32/fini"

            sfx = f"_s{s}"
            tensors["hidden" + sfx] = hidden_bf16.contiguous()
            tensors["mask" + sfx] = mask.contiguous()
            tensors["q" + sfx] = q_states.contiguous()
            tensors["k" + sfx] = k_states.contiguous()
            tensors["v" + sfx] = v_states.contiguous()
            tensors["out" + sfx] = out.contiguous()
            tensors["cos" + sfx] = cos[0].contiguous()  # [S,512] f32 — comparé aux cos/sin HOST du gate
            tensors["sin" + sfx] = sin[0].contiguous()
            manifest_cases[f"S={s}"] = {
                "mask_counts": {"masked": n_masked, "causal_pure": causal,
                                "note": "couche full : causal pur, pas de fenetre"},
                "discriminability": {
                    "criterion": "SORTIE d'attention (étage qui gate) — option B ratifiée (plan 04eaa8d)",
                    "out_max_abs_vs_wrong_hypothesis": discrim_out, "threshold": DISCRIM_THR,
                    "v_norm_max_abs_vs_wrong_hypothesis_documentaire": discrim_vnorm,
                    "verdict": "DISCRIMINANTE"},
                "points_fixes": {
                    "out[0,0,:3]": [float(x) for x in out[0, 0, :3]],
                    "out[0,-1,3837:]": [float(x) for x in out[0, -1, 3837:]],
                    "q[0,-1,0,:3]": [float(x) for x in q_states[0, -1, 0, :3]],
                    "k[0,-1,0,:3]": [float(x) for x in k_states[0, -1, 0, :3]],
                    "v[0,0,0,:3]": [float(x) for x in v_states[0, 0, 0, :3]],
                    "cos[-1,:3]": [float(x) for x in cos[0, -1, :3]],
                    "sin[-1,:3]": [float(x) for x in sin[0, -1, :3]],
                },
            }
            print(f"S={s} : masqués {n_masked} (causal pur), étages a/b/v/c capturés par hooks, "
                  f"témoin K=V (entrée v_norm == kp brut) OK, eager témoin OK", flush=True)
    finally:
        mg.apply_rotary_pos_emb = orig_rope
        mg.eager_attention_forward = orig_eager
        h_kp.remove()
        h_v.remove()
        h_o.remove()

    os.makedirs(FX, exist_ok=True)
    save_file(tensors, os.path.join(FX, "u_full.safetensors"))

    manifest = {
        "source": "66_u5_full_oracle.py (Task 5, plan J2 amendé — Amendement 2 2026-07-25)",
        "oracle": "Gemma4TextAttention(text_config, layer_idx=5) RÉEL, poids export dq (D9), "
                  "module f32 (flux moteur bf16->f32, précédent déclaré U3) ; intermédiaires par "
                  "HOOKS (apply_rotary_pos_emb x2, k_proj, v_norm entrée+sortie, o_proj), témoin "
                  "eager asserté ; V = v_norm(k_proj brut) prouvé par bit-égalité entrée v_norm == kp",
        "mask": "create_causal_mask (transformers 5.14, mask_kwargs du forward Gemma4TextModel) "
                "— causal PUR (couche full), additive f32 [1,1,S,S]",
        "layouts": {
            "hidden": "[1,S,3840] bf16 (graine consignée, module nourri en f32)",
            "q": "[1,S,16,512] f32 post q_proj+q_norm+p-rope, PRE-transpose (unsqueeze_dim=2)",
            "k": "[1,S,1,512] f32 key_states FINAL post k_proj+k_norm+p-rope, PRE-transpose",
            "v": "[1,S,1,512] f32 value_states == v_norm(k_proj BRUT), PRE-transpose",
            "out": "[1,S,3840] f32 sortie o_proj (== sortie module, asserté)",
            "cos/sin": "[S,512] f32 du module réel Gemma4TextRotaryEmbedding (layer_type "
                       "full_attention) — le gate compare ses cos/sin HOST (formule runner "
                       "w4auto ropeFull) : tripwire positions {708,1030} borne 1e-3",
        },
        "geometry": {"mqa_groups": 16, "nh": 16, "kvh": 1, "hd": 512, "scaling": 1.0,
                     "k_eq_v": True, "rope": "proportional theta 1e6, partial 0.25 sur 512 "
                     "(64 freq actives + 192 nope)", "layer": LAYER},
        "seed": SEED, "seq_lens": list(SEQ_LENS), "tripwire_positions": list(TRIPWIRE_POS),
        "thresholds": {"u5_max_abs": MAX_ABS_THR, "u5_mean_abs": MEAN_ABS_THR,
                       "u5_stage_c_max_abs": MAX_ABS_THR,
                       "discriminability_min_on_output": DISCRIM_THR,
                       "tripwire_max_abs": 1.0e-3,
                       "note": "seuils f32 (§3/Amendement 2) — comparaisons f32, PAS bit-exact ; "
                               "discriminabilité : critère sur la SORTIE (option B, plan 04eaa8d)"},
        "checkpoint_fact_uniform_full_norms": uniform_norms_fact,
        "sanity_prope": {"pos0_identity_strict": True, "nope_384_identity_all_pos": True,
                         "inv_freq_active": 64, "inv_freq_zeros": 192, "attention_scaling": 1.0},
        "k_norm_weight_spread": k_norm_spread,
        "cases": manifest_cases,
        "pass": "gate u5 (Amendement 2, périmètre U5 explicite) : gating S=8 étages a/b/v aux "
                "seuils max_abs<=1e-4 mean_abs<=1e-6 ; sanity p-RoPE pos 0 identité STRICTE "
                "(oracle : cos/sin HF ; gate : cos/sin host) ; tripwire positions {708,1030} "
                "borne 1e-3 (cos/sin host vs HF + étages ropés a/b à ces positions, informatif, "
                "sans mean_abs) ; étage (c) attention complète S=8 ET S=1040 max_abs<=1e-4 ; "
                "discriminabilité K=V >= 10x seuil sur la SORTIE, câblée ICI (exit 1 sinon — "
                "option B ratifiée, plan 04eaa8d), témoin structurel bit-exact entrée "
                "v_norm == kp asserté",
        "versions": {"transformers": transformers.__version__, "torch": torch.__version__},
    }
    with open(os.path.join(FX, "u_full_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    sz = os.path.getsize(os.path.join(FX, "u_full.safetensors")) / 1.0e6
    print(f"PASS fixture U5 écrite : S={SEQ_LENS} -> fixtures/u_full.safetensors ({sz:.1f} Mo)", flush=True)


if __name__ == "__main__":
    main()
