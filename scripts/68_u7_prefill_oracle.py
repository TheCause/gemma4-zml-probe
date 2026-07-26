#!/usr/bin/env python3
"""U7 — oracle prefill 12B COMPLET + logits softcap : forward HF CPU du MODÈLE entier
sur l'export dq (piège 12 : le modèle réel, jamais une recomposition).

Un seul mode `--u7` (le `--u7b` de la v2.1 est SUPPRIMÉ par l'Amendement du plan —
voie GPU-HF morte). Prompt canonique EXACT du périmètre Amendement 2 §U7 :
« What is the capital of France? Answer in one word. » via le chat template du
SNAPSHOT 12B (templates_match=False vs E2B, U_12B_CONTRACT §6 — sha asserté ici),
pattern BatchEncoding 5.14 (`return_dict=True`, contrat §8) — ici un FORWARD, pas
de generate.

Modèle : `AutoModelForCausalLM.from_pretrained(export_dq, dtype=bf16,
low_cpu_mem_usage=True)` (mmap safetensors) — le forward complet est en bf16 (le
modèle réel) ; PAS de .float() ici, contrairement aux oracles unitaires 65-67 (flux
par étage) : U7 compare le bout-en-bout au modèle HF tel quel. Critère REQUALIFIÉ
(Amendement 2 §U7, décision Régis 25 juil) : discriminant = top-5 ensemble+ordre+
marges + softcap ; max_abs documentaire sous garde-fou 0.5 (enveloppe G2 bf16-réel,
mesuré 0.376 au run 5 du gate). Chrono consigné (donnée D6).

Critères pré-enregistrés ASSERTÉS (Amendement 2 §U7, §3-U7) sur les logits de la
DERNIÈRE position :
  - softcap exercé : max|logits| <= 30 ET > 25 quelque part (le softcap doit mordre) ;
  - top-5 + valeurs -> fixture (le gate Zig compare en ENSEMBLE + tie rule |dlogit| < 1e-4) ;
  - marge top1-top2 et écart rang5-rang6 consignés (piège 17 : lus AVANT tout verdict).

Sorties : fixtures/u_prefill.safetensors (ids i32 + logits f32 dernière position +
top5_ids/top5_vals) + fixtures/u_prefill_manifest.json.

Venv : /data/venvs/g12b (VM) ou ~/ml-venvs/g12b (M4, repli D6 si chrono VM prohibitif).
Lancement (VM) : /data/venvs/g12b/bin/python3 scripts/68_u7_prefill_oracle.py --u7
"""
import argparse
import hashlib
import json
import os
import time

os.environ.setdefault("HF_HOME", "/data/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from safetensors.torch import save_file

ROOT = "/data/gemma4-zml-probe"


def install_fp32_hooks(backbone, extra=()) -> int:
    """--compute-fp32 : calcul fp32 sur STOCKAGE bf16 — même mécanique que le 69 (Amendement 3,
    resserrage U7 : le « forward f32 complet = 52 Go infaisable » du bloc Amendement 2 confondait
    stockage et arithmétique). Chaque module à paramètres directs passe en fp32 le temps de SON
    forward (bf16→fp32→bf16 sans perte) ; activations fp32 de bout en bout. ⚠ RAM résidente
    inchangée (~24 Go bf16 + ~1 Go transient) : hôte M4 (32 Gio), pas la VM (23 Gio)."""
    mods = [m for m in backbone.modules() if any(True for _ in m.parameters(recurse=False))]
    mods.extend(extra)

    def pre(mod, _inp):
        mod.to(torch.float32)

    def post(mod, _inp, _out):
        mod.to(torch.bfloat16)

    for m in mods:
        m.register_forward_pre_hook(pre)
        m.register_forward_hook(post)
    return len(mods)

# Prompt canonique EXACT (Amendement 2 §U7) — ne pas reformuler.
PROMPT = "What is the capital of France? Answer in one word."
# sha256 du chat_template.jinja du snapshot 12B (U_12B_CONTRACT §6, fait U0 constaté).
TEMPLATE_SHA_12B = "ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4"
VOCAB = 262144
SOFTCAP = 30.0
SOFTCAP_BITE = 25.0  # « ET > 25 quelque part » (§3-U7)
TIE_THR = 1.0e-4  # règle de tie du gate (consignée au manifest, appliquée côté Zig)
# Seuil max_abs côté gate : 1e-2 (originel §3-U7, RESTAURÉ 26 juil contre la fixture fp32 —
# mesuré 9.365e-4) ; 0.5 = ancien garde-fou de l'ère oracle bf16 (Amendement 2, SUPERSEDED,
# gardé ici pour le manifest du mode bf16 historique).
MAX_ABS_GUARD_FP32 = 1.0e-2
MAX_ABS_GUARD = 5.0e-1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--u7", action="store_true", help="mode unique (obligatoire, cf plan Task 7)")
    ap.add_argument("--weights", default=os.path.join(ROOT, "weights_12b_dq"),
                    help="export dq (D9 : oracles Python = export)")
    ap.add_argument("--fixtures-dir", default=os.path.join(ROOT, "fixtures"))
    ap.add_argument("--host-label", default="VM",
                    help="étiquette du chrono D6 (VM|M4) — JAMAIS le hostname réel "
                         "(anonymisation repo public)")
    ap.add_argument("--compute-fp32", action="store_true",
                    help="calcul fp32 sur stockage bf16 (hooks par-module, même mécanique que "
                         "le 69) — resserrage U7 post-Amendement 3 ; sorties suffixées _fp32 ; "
                         "hôte M4 requis (RAM)")
    args = ap.parse_args()
    assert args.u7, "mode --u7 requis (seul mode : le --u7b v2.1 est SUPPRIMÉ, Amendement)"

    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dq = args.weights

    # --- chat template : celui du SNAPSHOT 12B, copié dans l'export par le 63 — sha asserté ---
    tpl_path = os.path.join(dq, "chat_template.jinja")
    tpl_sha = hashlib.sha256(open(tpl_path, "rb").read()).hexdigest()
    assert tpl_sha == TEMPLATE_SHA_12B, \
        f"chat template export != snapshot 12B (U0 §6) : {tpl_sha}"

    tok = AutoTokenizer.from_pretrained(dq)
    msgs = [{"role": "user", "content": PROMPT}]
    # Pattern BatchEncoding 5.14 (contrat §8) : return_dict=True, **enc au forward.
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True)
    ids = enc["input_ids"][0]
    S = int(ids.shape[0])
    assert S >= 4, f"prompt templaté suspect : S={S}"
    print(f"prompt canonique templaté : S={S} ids={ids.tolist()}", flush=True)

    # --- témoin masques (même témoin que l'oracle 67) : à ce S < 1024, le masque sliding HF
    # == masque causal HF BIT-ÉGAL -> le gate moteur (two_masks=false, UNE table causale)
    # est licite. Asserté sur la mécanique HF réelle (masking_utils), pas recomposé.
    from transformers import AutoConfig
    from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
    cfg_mask = AutoConfig.from_pretrained(dq).text_config
    cfg_mask._attn_implementation = "eager"
    dummy = torch.zeros(1, S, cfg_mask.hidden_size, dtype=torch.float32)
    pos = torch.arange(S, dtype=torch.long).unsqueeze(0)
    mask_kw = dict(config=cfg_mask, inputs_embeds=dummy, attention_mask=None,
                   past_key_values=None, position_ids=pos)
    mask_sl = create_sliding_window_causal_mask(**mask_kw)
    mask_fl = create_causal_mask(**mask_kw)
    assert torch.equal(mask_sl, mask_fl), \
        f"S={S} : masque sliding != causal (fenêtre mordante ?) — two_masks=false du gate illicite"
    print(f"témoin masques : sliding == causal BIT-ÉGAL à S={S} (fenêtre 1024 non mordante) "
          f"-> gate two_masks=false licite", flush=True)

    # --- forward HF CPU complet, bf16, mmap (chrono consigné — donnée D6) ---
    t0 = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        dq, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.train(False)
    t_load = time.monotonic() - t0
    cfg_t = model.config.text_config
    assert cfg_t.num_hidden_layers == 48 and cfg_t.hidden_size == 3840
    assert float(cfg_t.final_logit_softcapping) == SOFTCAP, \
        f"final_logit_softcapping {cfg_t.final_logit_softcapping} != {SOFTCAP}"
    print(f"modèle chargé (bf16, low_cpu_mem_usage) en {t_load:.1f} s", flush=True)

    compute_dtype = "bfloat16"
    if args.compute_fp32:
        n_hooked = install_fp32_hooks(model.model, extra=(model.lm_head,))
        compute_dtype = "float32"
        print(f"calcul fp32 armé : {n_hooked} sous-modules hookés dont lm_head "
              f"(stockage bf16 intact, activations fp32)", flush=True)

    t1 = time.monotonic()
    with torch.no_grad():
        out = model(**enc)
    t_fwd = time.monotonic() - t1
    logits_all = out.logits
    assert list(logits_all.shape) == [1, S, VOCAB], f"logits {list(logits_all.shape)}"
    want_dtype = torch.float32 if args.compute_fp32 else torch.bfloat16
    assert logits_all.dtype == want_dtype, f"logits {logits_all.dtype} != {want_dtype}"
    last = logits_all[0, -1, :].float()
    assert torch.isfinite(last).all(), "logits dernière position non finis"
    print(f"forward prefill S={S} en {t_fwd:.1f} s (chrono D6)", flush=True)

    # --- softcap exercé (§3-U7) : max|logits| <= 30 ET > 25 quelque part — ASSERTÉ ---
    abs_last = last.abs()
    max_abs_logit = float(abs_last.max())
    n_over_25 = int((abs_last > SOFTCAP_BITE).sum())
    assert max_abs_logit <= SOFTCAP, f"softcap VIOLÉ : max|logits|={max_abs_logit} > {SOFTCAP}"
    assert n_over_25 > 0, \
        f"softcap ne MORD pas : aucun |logit| > {SOFTCAP_BITE} (max={max_abs_logit})"
    print(f"softcap exercé : max|logits|={max_abs_logit:.4f} <= 30, "
          f"{n_over_25} logits > {SOFTCAP_BITE} en |.|", flush=True)

    # --- top-5 + marges (piège 17 : marges lues AVANT tout verdict) ---
    top6_vals, top6_ids = torch.topk(last, 6)
    top5_ids = top6_ids[:5]
    top5_vals = top6_vals[:5]
    top5_tokens = [tok.decode([int(i)]) for i in top5_ids]
    margin_12 = float(top5_vals[0] - top5_vals[1])
    gap_56 = float(top6_vals[4] - top6_vals[5])
    print(f"top-5 : {[(int(i), repr(s), round(float(v), 4)) for i, s, v in zip(top5_ids, top5_tokens, top5_vals)]}",
          flush=True)
    print(f"marge top1-top2={margin_12:.6f} ; écart rang5-rang6={gap_56:.6f} "
          f"(tie rule gate : |dlogit| < {TIE_THR})", flush=True)

    # --- fixture + manifest (suffixe _fp32 en mode --compute-fp32 : les deux coexistent) ---
    suffix = "_fp32" if args.compute_fp32 else ""
    os.makedirs(args.fixtures_dir, exist_ok=True)
    save_file({
        "ids": ids.to(torch.int32).contiguous(),
        "logits": last.contiguous(),  # f32, dernière position (bf16->f32 exact en mode bf16)
        "top5_ids": top5_ids.to(torch.int32).contiguous(),
        "top5_vals": top5_vals.contiguous(),
    }, os.path.join(args.fixtures_dir, f"u_prefill{suffix}.safetensors"))

    manifest = {
        "source": "68_u7_prefill_oracle.py --u7 (Task 7, plan J2 — Amendement 2 §U7)",
        "oracle": "forward HF CPU COMPLET (AutoModelForCausalLM sur l'export dq, bf16, "
                  "low_cpu_mem_usage/mmap) — le modèle réel de bout en bout, PAS de .float() "
                  "(contrairement aux oracles unitaires 65-67) ; logits de la DERNIÈRE position "
                  "(bf16 -> f32 exact dans la fixture)",
        "prompt": PROMPT,
        "chat_template_sha256": tpl_sha,
        "seq_len": S,
        "ids": [int(i) for i in ids],
        "softcap": {"cap": SOFTCAP, "max_abs_logit": max_abs_logit,
                    "n_abs_over_25": n_over_25,
                    "asserted": "max|logits| <= 30 ET > 25 quelque part (§3-U7)"},
        "top5": {"ids": [int(i) for i in top5_ids],
                 "tokens": top5_tokens,
                 "vals": [float(v) for v in top5_vals]},
        "margins": {"top1_top2": margin_12, "rank5_rank6_gap": gap_56,
                    "note": "piège 17 — marges consignées AVANT tout verdict gate"},
        "chrono_s": {"load": round(t_load, 1), "forward_prefill": round(t_fwd, 1),
                     "host": args.host_label, "note": "donnée D6"},
        "thresholds": {"u7_max_abs_guard": MAX_ABS_GUARD_FP32 if args.compute_fp32 else MAX_ABS_GUARD,
                       "tie_dlogit": TIE_THR,
                       "note": "§3-U7 REQUALIFIÉ (Amendement 2 §U7, décision Régis 25 juil) : "
                               "critère discriminant = top-5 en ENSEMBLE + ordre + marges "
                               "(tie rule |dlogit| < 1e-4) + softcap ; max_abs documentaire "
                               "sous garde-fou 0.5 CÂBLÉ (enveloppe G2 bf16-réel, dépassement "
                               "= FAIL) — appliqués côté gate Zig (mode u7)",
                       "anomalie_zml_consignee": "crash reproductible du gate au 31e compileFn "
                               "du process (stage-major K=1, runs 1/3 : segfault émission MLIR "
                               "manualRope, swap quasi vide) + OOM du graphe complet 48 couches "
                               "en exécution (run 4, anon-rss 23,6 Go) -> voie retenue : "
                               "stage-major CHUNKÉ 8x6 (repli prescrit du plan Task 7) ; "
                               "à investiguer côté socle ZML"},
        "seed": "sans objet (forward déterministe, aucun tirage)",
        "dtype": f"stockage bf16, calcul {compute_dtype} ; fixture logits f32",
        "compute_dtype": compute_dtype,
        "versions": {"transformers": transformers.__version__, "torch": torch.__version__},
    }
    with open(os.path.join(args.fixtures_dir, f"u_prefill{suffix}_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    print(f"PASS fixture U7 écrite : S={S} -> {os.path.join(args.fixtures_dir, f'u_prefill{suffix}.safetensors')} ; "
          f"top1={top5_tokens[0]!r} marge={margin_12:.4f}", flush=True)


if __name__ == "__main__":
    main()
