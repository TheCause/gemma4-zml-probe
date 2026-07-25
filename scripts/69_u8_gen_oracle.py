#!/usr/bin/env python3
"""U8/U9-iv — oracle de génération 12B Unified sur l'export dq : décode greedy 48 tokens
(fixture u8_gen48) + mode --teacher-force (U9-iv : UN prefill de la séquence donnée).

Plan Task 9 Step 9.1 / Task 10 Step 10.3 (docs/superpowers/plans/2026-07-24-w4-j2-12b-unified.md),
D6 plan A : oracle CPU M4 (venv g12b), l'export dq mmap (~26 Go, 1er token = warm-up, la mesure
= médiane des tokens 2-4). Modèle RÉEL bf16 de bout en bout (AutoModelForCausalLM,
low_cpu_mem_usage — même chemin que le smoke U0b-décode Task 1b et l'oracle U7 68).

Format fixture (mode décode) : le mode --oracle de gemma4_g12auto lit EXACTEMENT deux clés
(gemma4_g12auto.zig l.904-936) : `positions` (i32, positions[0] == ids.len du prompt rendu,
check anti-mismatch prompt/fixture) et `fed` (i32, la séquence de référence [s0, t1, …] comparée
à `generated`). `expected`/`top5_ids`/`top5_vals` = diagnostic (non lus par le runner), miroir
du producteur J1 56_w4_gen_oracle.py (fed = seq[:n], expected = seq[1:n+1]).

Chat template : celui du SNAPSHOT 12B copié dans l'export par le 63 — sha asserté (même garde
que le 68) ; pattern BatchEncoding 5.14 (contrat §8) : return_dict=True, **enc au forward.
Softcap : appliqué par HF DANS ForCausalLM.forward (modeling_gemma4.py : lm_head → /cap →
tanh → ×cap) — out.logits est déjà softcappé, asserté ≤ 30 à chaque step.

Mode --teacher-force <ids.safetensors> (U9-iv, D10) : lit la clé `ids` (i32, écrite par
--out-ids du runner), fait UN prefill de [prompt templaté ++ ids[:-1]], puis argmax + top-5 +
marge PAR POSITION au fil de l'eau : hidden states du backbone (self.model) → lm_head + softcap
par CHUNKS de positions (miroir exact modeling_gemma4.py l.2592-2597) — jamais les ~600 Mo de
logits d'un coup. Self-check câblé : sur un préfixe court, tête manuelle == forward HF complet
(torch.equal). ⚠ --prompt reste REQUIS dans ce mode : --out-ids n'écrit que les ids GÉNÉRÉS,
le contexte du prefill = prompt templaté + ids (décision d'interprétation, rapport Task 9).

CLI :
  décode  : 69_u8_gen_oracle.py --weights <export_dq> --prompt "..." --n-tokens 48 --out u8_gen48.safetensors
  U9-iv   : 69_u8_gen_oracle.py --weights <export_dq> --prompt "..." --teacher-force u9_ids.safetensors --out u9_tf.json
Venv : ~/ml-venvs/g12b (M4, plan A) ou /data/venvs/g12b (VM, repli D6-B).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time

if os.path.isdir("/data"):
    os.environ.setdefault("HF_HOME", "/data/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = "/data/gemma4-zml-probe"
# sha256 du chat_template.jinja du snapshot 12B (U_12B_CONTRACT §6, même garde que le 68).
TEMPLATE_SHA_12B = "ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4"
VOCAB = 262144
SOFTCAP = 30.0
TOPK = 5
HEAD_CHUNK = 64  # teacher-force : positions par chunk (64 × 262144 × bf16 ≈ 32 Mo)


def anon_path(p: str) -> str:
    """Anonymisation (règle repo public — feedback revue U8) : le préfixe HOME de l'hôte oracle
    ne doit JAMAIS apparaître dans un manifest committable — même mécanique que --host-label.
    Les chemins /data/... (VM) sont déjà publics dans le repo, laissés tels quels."""
    home = os.path.expanduser("~")
    return "<oracle-host>" + p[len(home):] if home not in ("~", "/") and p.startswith(home) else p


def install_fp32_hooks(backbone, extra=()) -> int:
    """--compute-fp32 : calcul fp32 sur STOCKAGE bf16 (parité oracles J1 46/56 « poids fp32,
    oracle = source de vérité », impossible en chargement monolithique 12B : ~48 Go > 32 Gio M4).
    Chaque sous-module passe en fp32 le temps de SON forward (pré-hook) puis revient en bf16
    (post-hook) — bf16→fp32→bf16 est sans perte, pic RAM = +1 copie de couche (~1 Go).
    Les activations restent fp32 de bout en bout (les post-hooks ne touchent que les poids).
    `extra` : modules hors backbone (lm_head) — hooks transitoires AUSSI, car en tied (D7)
    une conversion permanente serait annulée par le post-hook d'embed_tokens au forward suivant.
    Granularité : tout module à PARAMÈTRES DIRECTS (Linear/Norm/Embedding), structure-agnostique
    (les DecoderLayers du 12B Unified ne sont pas des enfants directs de model.model — appris
    au smoke 16 ids : q_proj bf16 face à des activations fp32)."""
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


def load_model_and_tok(dq: str):
    """Chargement commun (chemin U0b-décode/U7) : sha template asserté, modèle réel bf16 mmap."""
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tpl_sha = hashlib.sha256(open(os.path.join(dq, "chat_template.jinja"), "rb").read()).hexdigest()
    assert tpl_sha == TEMPLATE_SHA_12B, f"chat template export != snapshot 12B (U0 §6) : {tpl_sha}"
    tok = AutoTokenizer.from_pretrained(dq)
    t0 = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(dq, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.train(False)
    t_load = time.monotonic() - t0
    cfg_t = model.config.get_text_config()
    assert cfg_t.num_hidden_layers == 48 and cfg_t.hidden_size == 3840
    assert float(cfg_t.final_logit_softcapping) == SOFTCAP
    print(f"modèle chargé (bf16, low_cpu_mem_usage) en {t_load:.1f} s", flush=True)
    versions = {"transformers": transformers.__version__, "torch": torch.__version__}
    return model, tok, tpl_sha, t_load, versions


def encode_prompt(tok, prompt: str):
    """Pattern BatchEncoding 5.14 (contrat §8) : return_dict=True, **enc au forward."""
    msgs = [{"role": "user", "content": prompt}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True)
    ids = enc["input_ids"][0]
    S = int(ids.shape[0])
    assert S >= 4, f"prompt templaté suspect : S={S}"
    print(f"prompt canonique templaté : S={S} ids={ids.tolist()}", flush=True)
    return enc, ids, S


def step_top5(logits_1d: torch.Tensor):
    """top-5 (+marge top1−top2) d'un vecteur logits [V] — f32 pour les marges."""
    lg = logits_1d.float()
    assert torch.isfinite(lg).all(), "logits non finis"
    max_abs = float(lg.abs().max())
    assert max_abs <= SOFTCAP, f"softcap VIOLÉ : max|logits|={max_abs} > {SOFTCAP}"
    vals, idxs = torch.topk(lg, TOPK)
    return idxs.to(torch.int32), vals, float(vals[0] - vals[1]), max_abs


def mode_decode(args, model, tok, tpl_sha, t_load, versions):
    enc, prompt_ids, S = encode_prompt(tok, args.prompt)
    n = int(args.n_tokens)
    eot_id = 106  # <turn|> (mesuré, cf gemma4_g12auto.zig) — informatif seulement : en mode
    # --oracle le runner NE s'arrête PAS à l'EOT (limite = fed.len), l'oracle non plus.

    # --- prefill (token 0 = s0, chrono à part : chargement pages mmap + warm-up) ---
    t0 = time.monotonic()
    with torch.no_grad():
        out = model(**enc, use_cache=True)
    t_prefill = time.monotonic() - t0
    assert out.logits.dtype == torch.bfloat16, f"logits {out.logits.dtype} != bf16 (modèle réel)"
    pkv = out.past_key_values
    idxs, vals, margin, max_abs = step_top5(out.logits[0, -1, :])
    print(f"prefill S={S} en {t_prefill:.1f} s ; s0={int(idxs[0])} marge={margin:.6f}", flush=True)

    seq = [int(idxs[0])]
    top5_ids = torch.zeros(n, TOPK, dtype=torch.int32)
    top5_vals = torch.zeros(n, TOPK, dtype=torch.float32)
    margins, times, max_abs_all = [margin], [t_prefill], [max_abs]
    top5_ids[0], top5_vals[0] = idxs, vals

    # --- décode greedy : n-1 steps produisent seq[1..n-1], +1 step pour `expected` (miroir 56) ---
    for k in range(1, n + 1):
        tk = time.monotonic()
        with torch.no_grad():
            out = model(input_ids=torch.tensor([[seq[-1]]], dtype=torch.long),
                        past_key_values=pkv, use_cache=True)
        dt = time.monotonic() - tk
        idxs, vals, margin, max_abs = step_top5(out.logits[0, -1, :])
        seq.append(int(idxs[0]))
        times.append(dt)
        if k < n:  # le step n ne produit que expected[n-1], pas un step de génération du gate
            top5_ids[k], top5_vals[k] = idxs, vals
            margins.append(margin)
            max_abs_all.append(max_abs)
            print(f"  marge top1-top2 @ gen={k} : {margin:.6f} (top1={int(idxs[0])} top2={int(idxs[1])})"
                  f"  [{dt:.2f} s]", flush=True)
        else:
            print(f"  step expected-only (produit expected[{n - 1}]={seq[-1]}) [{dt:.2f} s]", flush=True)

    fed = seq[:n]
    expected = seq[1:n + 1]
    # Chrono D6 : médiane des tokens 2-4 (le token 0 = prefill mmap/warm-up, le 1 = encore chaud).
    med_2_4 = statistics.median(times[2:5])
    min_margin = min(margins)
    argmin = margins.index(min_margin)
    print(f"chrono : médiane tokens 2-4 = {med_2_4:.2f} s/token (D6) ; total {sum(times):.1f} s", flush=True)
    print(f"marges : min={min_margin:.6f} @ gen={argmin} (piège 17 : lues AVANT tout verdict)", flush=True)
    eot_pos = fed.index(eot_id) if eot_id in fed else None
    print(f"EOT ({eot_id}) dans fed : {'@ gen=' + str(eot_pos) if eot_pos is not None else 'absent'}", flush=True)
    print(f"aperçu réponse HF : {tok.decode(fed, skip_special_tokens=True)[:300]!r}", flush=True)

    positions = torch.arange(S, S + n, dtype=torch.int32)  # positions[0] == S (check runner)
    save_file({
        "positions": positions.contiguous(),
        "fed": torch.tensor(fed, dtype=torch.int32),
        "expected": torch.tensor(expected, dtype=torch.int32),
        "top5_ids": top5_ids.contiguous(),
        "top5_vals": top5_vals.contiguous(),
    }, args.out)

    manifest = {
        "source": "69_u8_gen_oracle.py (mode décode — Task 9 Step 9.1, plan J2)",
        "oracle": "décode greedy HF CPU, modèle RÉEL bf16 (AutoModelForCausalLM sur l'export dq, "
                  "low_cpu_mem_usage/mmap) — chemin U0b-décode/U7 ; logits softcappés par HF",
        "prompt": args.prompt,
        "prompt_ids": prompt_ids.tolist(),
        "seq_len": S,
        "n_decode": n,
        "fed_head": fed[:8], "expected_head": expected[:8],
        "eot_id": eot_id, "eot_pos_in_fed": eot_pos,
        "reponse_hf": tok.decode(fed, skip_special_tokens=True),
        "margins": {"per_step": [round(m, 6) for m in margins],
                    "min": round(min_margin, 6), "argmin": argmin,
                    "note": "piège 17 — marges consignées AVANT tout verdict gate"},
        "softcap": {"cap": SOFTCAP, "max_abs_per_step_max": round(max(max_abs_all), 4),
                    "asserted": "max|logits| <= 30 à chaque step"},
        "chrono_s": {"load": round(t_load, 1), "per_token": [round(t, 2) for t in times],
                     "median_tokens_2_4": round(med_2_4, 2), "host": args.host_label,
                     "note": "D6 — token 0 = prefill (mmap+warm-up), médiane 2-4 = la mesure"},
        "chat_template_sha256": tpl_sha,
        "weights": anon_path(args.weights),
        "fixture_keys_lues_par_le_runner": ["positions", "fed"],
        "versions": versions,
    }
    with open(args.out + ".manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    print(f"PASS fixture U8 écrite : {args.out} (S={S}, n={n}, marge_min={min_margin:.4f} @ {argmin})", flush=True)


def mode_teacher_force(args, model, tok, tpl_sha, t_load, versions):
    """U9-iv (D10) : UN prefill de [prompt ++ ids[:-1]] ; argmax+top-5 par position au fil de l'eau."""
    _, prompt_ids, S = encode_prompt(tok, args.prompt)
    with safe_open(args.teacher_force, framework="pt") as f:
        gen = f.get_tensor("ids").to(torch.long)
    N = int(gen.shape[0])
    assert N >= 2, f"ids fixture suspecte : N={N}"
    print(f"teacher-force : {N} ids générés (runner --out-ids), prefill de S+N-1={S + N - 1} positions", flush=True)

    full = torch.cat([prompt_ids.to(torch.long), gen[:-1]]).unsqueeze(0)
    T = int(full.shape[1])

    compute_dtype = "bfloat16"
    if args.compute_fp32:
        n_hooked = install_fp32_hooks(model.model, extra=(model.lm_head,))
        compute_dtype = "float32"
        print(f"calcul fp32 armé : {n_hooked} sous-modules hookés dont lm_head "
              f"(stockage bf16 intact, activations fp32)", flush=True)

    # --- témoin fenêtre (D10/plan Step 10.3) : à T > sliding_window le masque sliding HF doit
    # DIFFÉRER du causal (fenêtre mordante) — même mécanique réelle que le témoin du 68.
    cfg_t = model.config.get_text_config()
    sw = int(cfg_t.sliding_window)
    window_bites = None
    if T > sw:
        from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
        import copy
        cfg_mask = copy.deepcopy(cfg_t)
        cfg_mask._attn_implementation = "eager"
        dummy = torch.zeros(1, T, cfg_t.hidden_size, dtype=torch.float32)
        pos = torch.arange(T, dtype=torch.long).unsqueeze(0)
        mask_kw = dict(config=cfg_mask, inputs_embeds=dummy, attention_mask=None,
                       past_key_values=None, position_ids=pos)
        window_bites = not torch.equal(create_sliding_window_causal_mask(**mask_kw),
                                       create_causal_mask(**mask_kw))
        assert window_bites, f"T={T} > fenêtre {sw} mais masque sliding == causal : fenêtre NON mordante en prefill"
        print(f"témoin fenêtre : masque sliding != causal à T={T} (fenêtre {sw} mordante en prefill)", flush=True)

    # --- backbone : UN prefill (use_cache=False), hidden states post-norm ---
    t1 = time.monotonic()
    with torch.no_grad():
        hs = model.model(input_ids=full, use_cache=False).last_hidden_state  # [1, T, 3840] bf16
    t_fwd = time.monotonic() - t1
    print(f"prefill backbone T={T} en {t_fwd:.1f} s", flush=True)

    cap = float(cfg_t.final_logit_softcapping)

    def head(h):  # miroir EXACT modeling_gemma4.py l.2592-2597 : lm_head -> /cap -> tanh -> ×cap
        lg = model.lm_head(h)
        return torch.tanh(lg / cap) * cap

    # --- self-check tête manuelle == forward HF complet (préfixe court) ---
    # bf16 : torch.equal (l'arrondi bf16 final gomme l'ordre d'accumulation BLAS).
    # fp32 : max|d| ≤ 1e-3 — l'accumulation fp32 dépend de la forme du matmul
    # (logits_to_keep=1 slice AVANT lm_head), bruit ~1e-4 mesuré (précédent E2B P4.3 :
    # même classe, 1.5e-5) ; le bit-exact inter-formes n'existe pas en fp32.
    with torch.no_grad():
        want = model(input_ids=full[:, :8], logits_to_keep=1).logits[0, -1, :]
        got = head(model.model(input_ids=full[:, :8], use_cache=False).last_hidden_state)[0, -1, :]
    if args.compute_fp32:
        d = float((got.float() - want.float()).abs().max())
        assert d <= 1e-3, f"self-check tête manuelle != forward HF (max|d|={d} > 1e-3)"
        print(f"self-check tête : manuelle == forward HF (fp32, max|d|={d:.2e} ≤ 1e-3, préfixe 8)", flush=True)
    else:
        assert torch.equal(got, want), \
            f"self-check tête manuelle != forward HF (max|d|={float((got.float() - want.float()).abs().max())})"
        print("self-check tête : manuelle == forward HF (bit-égal, préfixe 8)", flush=True)

    # --- balayage au fil de l'eau : positions S-1 .. T-1 prédisent gen[0..N-1] ---
    t2 = time.monotonic()
    n_match = 0
    margins, mismatches, top5_all = [], [], []
    for lo in range(S - 1, T, HEAD_CHUNK):
        hi = min(lo + HEAD_CHUNK, T)
        with torch.no_grad():
            lg = head(hs[:, lo:hi, :]).float()[0]  # [c, V]
        vals, idxs = torch.topk(lg, TOPK, dim=-1)
        for j in range(hi - lo):
            k = lo - (S - 1) + j
            got_id, want_id = int(idxs[j, 0]), int(gen[k])
            margin = float(vals[j, 0] - vals[j, 1])
            margins.append(round(margin, 6))
            top5_all.append({"ids": idxs[j].tolist(), "vals": [round(float(v), 4) for v in vals[j]]})
            if got_id == want_id:
                n_match += 1
            else:
                mismatches.append({"pos_gen": k, "got": got_id, "want": want_id,
                                   "margin_top1_top2": round(margin, 6),
                                   "top5_ids": idxs[j].tolist(),
                                   "top5_vals": [round(float(v), 6) for v in vals[j]]})
    t_head = time.monotonic() - t2

    print(f"U9-iv : {n_match}/{N} argmax == ids runner ; {len(mismatches)} divergence(s)", flush=True)
    for mm in mismatches[:16]:
        print(f"  divergence @ gen={mm['pos_gen']} : got={mm['got']} want={mm['want']} "
              f"marge={mm['margin_top1_top2']}", flush=True)
    print(f"marges : min={min(margins):.6f} @ gen={margins.index(min(margins))}", flush=True)

    report = {
        "source": "69_u8_gen_oracle.py (mode --teacher-force — Task 10 Step 10.3, plan J2, D10)",
        "oracle": f"UN prefill HF CPU (calcul {compute_dtype}, stockage bf16) de [prompt templaté "
                  "++ ids[:-1]] ; tête lm_head+softcap par chunks (miroir modeling_gemma4.py, "
                  "self-check bit-égal câblé)"
                  + (" ; --compute-fp32 : hooks par-couche, parité oracles J1 46/56"
                     if args.compute_fp32 else ""),
        "compute_dtype": compute_dtype,
        "prompt": args.prompt, "prompt_ids": prompt_ids.tolist(), "seq_len": S,
        "ids_fixture": anon_path(args.teacher_force), "n_ids": N, "prefill_len": T,
        "window": {"sliding_window": sw, "bites_in_prefill": window_bites},
        "verdict_data": {"n_match": n_match, "n_total": N, "mismatches": mismatches,
                         "note": "N<total : protocole de marge J1 par position — décision Régis, "
                                 "pas ce script (Amendement 2)"},
        "margins_per_pos": margins,
        "top5_per_pos": top5_all,
        "chrono_s": {"load": round(t_load, 1), "prefill_backbone": round(t_fwd, 1),
                     "head_sweep": round(t_head, 1), "host": args.host_label},
        "chat_template_sha256": tpl_sha, "weights": anon_path(args.weights),
        "versions": versions,
    }
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"rapport écrit : {args.out} ({n_match}/{N})", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights", default=os.path.join(ROOT, "weights_12b_dq"),
                    help="export dq (D9 : oracles Python = export, jamais le packé)")
    ap.add_argument("--prompt", required=True,
                    help="prompt user (REQUIS aussi en --teacher-force : contexte du prefill)")
    ap.add_argument("--n-tokens", type=int, default=48, help="mode décode : tokens greedy (48 = U8)")
    ap.add_argument("--out", required=True,
                    help="décode : fixture .safetensors (+ .manifest.json) ; teacher-force : rapport .json")
    ap.add_argument("--teacher-force", default=None, metavar="IDS_SAFETENSORS",
                    help="U9-iv : safetensors clé 'ids' (i32, --out-ids du runner) — UN prefill, "
                         "argmax+top-5 par position au fil de l'eau")
    ap.add_argument("--compute-fp32", action="store_true",
                    help="teacher-force uniquement : calcul fp32 sur stockage bf16 (hooks "
                         "par-couche) — restaure la parité d'instrument avec les oracles J1 "
                         "(46/56 : poids fp32) que le 12B monolithique interdit (RAM)")
    ap.add_argument("--host-label", default="M4",
                    help="étiquette chrono D6 (M4|VM) — JAMAIS le hostname réel (anonymisation)")
    args = ap.parse_args()
    if args.compute_fp32 and not args.teacher_force:
        ap.error("--compute-fp32 n'existe qu'en --teacher-force (le mode décode U8 reste bf16 tel que consigné)")

    model, tok, tpl_sha, t_load, versions = load_model_and_tok(args.weights)
    if args.teacher_force:
        mode_teacher_force(args, model, tok, tpl_sha, t_load, versions)
    else:
        mode_decode(args, model, tok, tpl_sha, t_load, versions)


if __name__ == "__main__":
    main()
