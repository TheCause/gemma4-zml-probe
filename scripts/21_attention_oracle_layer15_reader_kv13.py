"""P5.2.E.0 — PyTorch oracle : première attention effective.

Layer 15 (reader, sliding) lit le KV partagé produit par layer 13 (writer, sliding).
Calcul : Q15 × K13ᵀ → (+ masque causal) → softmax fp32 → × V13 → context.

Oracle PyTorch UNIQUEMENT. Aucun ZML. Aucun chargement de modèle : on relit des
fixtures déjà validées (q_final de P5.2.C, k_final/v_final de P5.2.D corrigé) et on
reproduit `eager_attention_forward` de `modeling_gemma4.py` (transformers 5.9.0,
lignes ~982-1015) :

    key_states  = repeat_kv(key,   num_key_value_groups)   # n_kv 1 -> 8 (GQA)
    value_states= repeat_kv(value, num_key_value_groups)
    attn_weights= torch.matmul(query, key_states.transpose(2, 3)) * scaling   # scaling = 1.0
    # (pas de softcap d'attention sur ce path)
    attn_weights= attn_weights + attention_mask                              # masque ADDITIF
    attn_weights= softmax(attn_weights, dim=-1, dtype=torch.float32)
    attn_output = torch.matmul(attn_weights, value_states)

Faits Gemma4 verrouillés (cf PLANNING.md §7, source de vérité modeling_gemma4) :
- `Gemma4TextAttention.scaling = 1.0` (PAS 1/√head_dim — la norm passe par q_norm).
- Pas de softcap d'attention (le `final_logit_softcapping` n'intervient qu'en P7).
- GQA via `repeat_kv`, `num_key_value_groups = n_heads / n_kv = 8 / 1 = 8`.
- Masque additif, softmax en fp32.
- V vient de `v_final` = `v_after_norm` (RMSNorm sans scale, D.0b) — JAMAIS le V brut.

Masque : layer_type = sliding_attention, sliding_window = 512. On construit le VRAI
masque sliding causal et on PROUVE qu'à S=4 < 512 il est strictement identique au
masque causal pur (aucune position supplémentaire masquée). E.0 ne valide donc PAS
le comportement sliding-window réel — celui-ci sera testé en E.mask avec un cas
synthétique S=8, window=3.

Interdits E.0 : ZML, modification de runners ZML, E.1, layer 14 (full attention),
softcap d'attention, scaling 1/√head_dim, V brut (non normé), vrai sliding masking
prétendu validé.
"""
from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file


ROOT = Path(__file__).resolve().parents[1]
Q_FIXTURE = ROOT / "fixtures" / "q_only_reader_layer15.safetensors"          # P5.2.C (q_final)
KV_FIXTURE = ROOT / "fixtures" / "p5_2_d0_kv_oracle_layer13.pt"              # P5.2.D.0b (k/v_final + v brut)
KV_SLIM = ROOT / "fixtures" / "p5_2_d5_kv_slot_layer13.safetensors"          # D.5 canonical (cross-check)
OUT_FIXTURE = ROOT / "fixtures" / "p5_2_e0_attention_oracle_layer15_kv13.pt"

# --- Config (depuis manifests C.0 / D.0b, source de vérité modeling_gemma4) ---
READER_LAYER_IDX = 15
WRITER_LAYER_IDX = 13
LAYER_TYPE = "sliding_attention"
HEAD_DIM = 256
N_HEADS = 8
N_KV = 1
NUM_KEY_VALUE_GROUPS = N_HEADS // N_KV          # 8 (GQA)
SEQ_LEN = 4
SLIDING_WINDOW = 512
SCALING = 1.0                                   # Gemma4TextAttention.scaling — PAS 1/sqrt(head_dim)
SOFTCAP = None                                  # pas de softcap d'attention sur ce path
TOL = 1.0e-4

EXPECTED_Q_SHAPE = (1, N_HEADS, SEQ_LEN, HEAD_DIM)   # [1, 8, 4, 256]
EXPECTED_K_SHAPE = (1, N_KV, SEQ_LEN, HEAD_DIM)      # [1, 1, 4, 256]
EXPECTED_V_SHAPE = (1, N_KV, SEQ_LEN, HEAD_DIM)      # [1, 1, 4, 256]


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Réplique modeling_gemma4.repeat_kv : [b, n_kv, s, d] -> [b, n_kv*n_rep, s, d]."""
    b, n_kv, s, d = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :]
        .expand(b, n_kv, n_rep, s, d)
        .reshape(b, n_kv * n_rep, s, d)
    )


def profile(name: str, t: torch.Tensor) -> dict:
    p = {
        "shape": list(t.shape),
        "dtype": str(t.dtype).replace("torch.", ""),
        "min": t.min().item(),
        "max": t.max().item(),
        "mean": t.mean().item(),
        "std": t.std().item(),
        "sum": t.sum().item(),
    }
    print(
        f"  {name:<14} shape={tuple(t.shape)!s:<18} dtype={p['dtype']} "
        f"min={p['min']: .6e} max={p['max']: .6e} mean={p['mean']: .6e} "
        f"std={p['std']: .6e} sum={p['sum']: .6e}"
    )
    return p


def main() -> None:
    torch.manual_seed(1337)  # déterminisme nominal (aucun RNG utilisé ici)

    print("=" * 70)
    print("P5.2.E.0 — PyTorch attention oracle : layer 15 reader × KV layer 13")
    print("=" * 70)
    print(f"scaling = {SCALING} (NOT 1/sqrt({HEAD_DIM}) = {1.0/HEAD_DIM**0.5:.6f})")
    print(f"softcap = {SOFTCAP}  |  layer_type = {LAYER_TYPE}  |  sliding_window = {SLIDING_WINDOW}")
    print(f"GQA num_key_value_groups = {NUM_KEY_VALUE_GROUPS}  (n_heads {N_HEADS} / n_kv {N_KV})")
    print()

    # === 1-3. Chargement fixtures validées ===
    assert Q_FIXTURE.exists(), f"missing Q fixture {Q_FIXTURE}"
    assert KV_FIXTURE.exists(), f"missing KV fixture {KV_FIXTURE}"
    q_final = load_file(str(Q_FIXTURE))["q_final"].to(torch.float32).contiguous()

    kv_blob = torch.load(str(KV_FIXTURE), map_location="cpu", weights_only=False)
    k_final = kv_blob["k_final"].to(torch.float32).contiguous()
    v_final = kv_blob["v_final"].to(torch.float32).contiguous()
    v_after_reshape = kv_blob["v_after_reshape"].to(torch.float32).contiguous()  # V brut [1,4,1,256]

    print(f"Loaded q_final  from {Q_FIXTURE.name}")
    print(f"Loaded k_final/v_final from {KV_FIXTURE.name} (D.0b corrected, V RMSNorm no-scale)")

    # Cross-check optionnel vs fixture D.5 slim canonical (k_final/v_final identiques).
    if KV_SLIM.exists():
        slim = load_file(str(KV_SLIM))
        d_k = (slim["k_final"].to(torch.float32) - k_final).abs().max().item()
        d_v = (slim["v_final"].to(torch.float32) - v_final).abs().max().item()
        print(f"Cross-check vs D.5 slim : |Δk_final|={d_k:.3e}  |Δv_final|={d_v:.3e}  (expected 0.0)")
        assert d_k == 0.0 and d_v == 0.0, "D.0b et D.5 slim divergent — fixtures incohérentes"
    print()

    # === 4. Assertions de forme ===
    assert tuple(q_final.shape) == EXPECTED_Q_SHAPE, f"q_final {tuple(q_final.shape)} != {EXPECTED_Q_SHAPE}"
    assert tuple(k_final.shape) == EXPECTED_K_SHAPE, f"k_final {tuple(k_final.shape)} != {EXPECTED_K_SHAPE}"
    assert tuple(v_final.shape) == EXPECTED_V_SHAPE, f"v_final {tuple(v_final.shape)} != {EXPECTED_V_SHAPE}"
    assert q_final.dtype == k_final.dtype == v_final.dtype == torch.float32

    # === V correction check : V doit être normé, pas brut. ===
    # v_final = transpose(v_after_norm) ; v_after_reshape = V brut. max|.| ~ 0.777 (D.2b/D.5).
    v_norm_vs_raw = (v_final[0, 0, :, :] - v_after_reshape[0, :, 0, :]).abs().max().item()
    print(f"V correction check : max|v_final - v_raw_final| = {v_norm_vs_raw:.6e}  "
          f"(expected ~0.777, V RMSNormed sans scale — sinon le bug V brut est revenu)")
    assert v_norm_vs_raw > 1e-2, (
        f"v_final ~ v_raw : V non normé, bug D.0 revenu (got {v_norm_vs_raw})"
    )
    print()

    # === 5. GQA : repeat_kv 1 -> 8 (fidèle à modeling_gemma4) ===
    key_states = repeat_kv(k_final, NUM_KEY_VALUE_GROUPS)     # [1, 8, 4, 256]
    value_states = repeat_kv(v_final, NUM_KEY_VALUE_GROUPS)   # [1, 8, 4, 256]
    assert tuple(key_states.shape) == EXPECTED_Q_SHAPE
    assert tuple(value_states.shape) == EXPECTED_Q_SHAPE
    # Sanity GQA : chaque head répliqué porte le même contenu que l'unique kv head.
    assert torch.equal(key_states[0, 0], key_states[0, 7]), "repeat_kv K incohérent"
    assert torch.equal(value_states[0, 0], value_states[0, 7]), "repeat_kv V incohérent"

    # === 6-7. scores_raw = Q @ Kᵀ * scaling  (scaling = 1.0, PAS de /sqrt(head_dim)) ===
    scores_raw = torch.matmul(q_final, key_states.transpose(2, 3)) * SCALING   # [1, 8, 4, 4]
    assert tuple(scores_raw.shape) == (1, N_HEADS, SEQ_LEN, SEQ_LEN)

    # === 8. Masque causal additif pour S=4 + PREUVE de dégénérescence du sliding ===
    min_val = torch.finfo(torch.float32).min
    idx = torch.arange(SEQ_LEN)
    causal_bool = idx.view(SEQ_LEN, 1) >= idx.view(1, SEQ_LEN)            # j <= i visible
    # vrai masque sliding : causal ET (i - j) < sliding_window
    sliding_bool = causal_bool & ((idx.view(SEQ_LEN, 1) - idx.view(1, SEQ_LEN)) < SLIDING_WINDOW)
    causal_mask = torch.where(causal_bool, torch.zeros(()), torch.full((), min_val)).view(1, 1, SEQ_LEN, SEQ_LEN)
    sliding_mask = torch.where(sliding_bool, torch.zeros(()), torch.full((), min_val)).view(1, 1, SEQ_LEN, SEQ_LEN)
    degenerates = torch.equal(sliding_mask, causal_mask)
    print(f"Sliding mask dégénère en causal (S={SEQ_LEN} < window={SLIDING_WINDOW}) : {degenerates}")
    assert degenerates, "sliding != causal : E.0 suppose S < window, hypothèse violée"
    print("  -> E.0 ne valide PAS le sliding-window réel (testé plus tard en E.mask, S=8 window=3).")
    print("  Masque causal additif (0 visible / finfo.min masqué), forme [1,1,4,4] :")
    for i in range(SEQ_LEN):
        row = ["  0  " if causal_bool[i, j] else " min " for j in range(SEQ_LEN)]
        print(f"    q{i}: [{' '.join(row)}]  -> voit tokens 0..{i}")
    print()

    # === 9. scores_masked = scores_raw + masque (additif) ===
    scores_masked = scores_raw + sliding_mask        # broadcast [1,1,4,4] sur [1,8,4,4]

    # === 10. probs = softmax(scores_masked, dim=-1) en fp32 ===
    probs = torch.softmax(scores_masked, dim=-1, dtype=torch.float32)     # [1, 8, 4, 4]

    # === 11. context = probs @ V ===
    context = torch.matmul(probs, value_states)      # [1, 8, 4, 256]
    assert tuple(context.shape) == (1, N_HEADS, SEQ_LEN, HEAD_DIM)

    # === Checks masque / probabilités ===
    print("Mask & probability checks:")
    prob_sums = probs.sum(dim=-1)                    # [1, 8, 4] -> doit valoir 1
    max_sum_err = (prob_sums - 1.0).abs().max().item()
    print(f"  max|sum(probs, dim=-1) - 1| = {max_sum_err:.3e}  (expected < 1e-6)")
    assert max_sum_err < 1e-6, f"probs ne somment pas à 1 (err {max_sum_err})"

    # Positions futures masquées -> proba ~ 0
    future = ~causal_bool                            # [4,4] True où j > i (futur)
    future_prob_max = probs[:, :, future].abs().max().item() if future.any() else 0.0
    print(f"  max proba sur positions futures masquées = {future_prob_max:.3e}  (expected ~0)")
    assert future_prob_max < 1e-9, f"fuite d'attention sur le futur (max {future_prob_max})"

    # Causalité explicite : q0 ne voit que t0, q1 voit t0..t1, etc.
    for i in range(SEQ_LEN):
        visible_mass = probs[0, 0, i, : i + 1].sum().item()
        future_mass = probs[0, 0, i, i + 1 :].sum().item()
        print(f"  q{i}: masse visible(0..{i})={visible_mass:.6f}  masse futur={future_mass:.3e}")
        assert abs(visible_mass - 1.0) < 1e-6 and future_mass < 1e-9
    print()

    # === 12. Profils + fixed points ===
    print("Tensor profiles:")
    stats = {
        "q_final": profile("q_final", q_final),
        "k_final": profile("k_final", k_final),
        "v_final": profile("v_final", v_final),
        "scores_raw": profile("scores_raw", scores_raw),
        "scores_masked": profile("scores_masked", scores_masked),
        "probs": profile("probs", probs),
        "context": profile("context", context),
    }
    print()

    def fp(t: torch.Tensor, idx_tuple, n: int) -> str:
        sl = t[idx_tuple][:n].tolist()
        return "[" + ", ".join(f"{v:.10f}" for v in sl) + "]"

    print("Fixed points:")
    fixed = {}
    for name, t, pts, n in [
        ("scores_raw", scores_raw, [(0, 0, 0), (0, 0, 3), (0, 7, 3)], 4),
        ("probs", probs, [(0, 0, 0), (0, 0, 3), (0, 7, 3)], 4),
        ("context", context, [(0, 0, 0), (0, 0, 3), (0, 7, 3)], 8),
    ]:
        for pt in pts:
            key = f"{name}[{','.join(map(str, pt))},:{n}]"
            val = fp(t, pt, n)
            fixed[key] = t[pt][:n].tolist()
            print(f"  {key} = {val}")
    print()

    # === 13. Sauvegarde fixture oracle (.pt) — tenseurs + meta embarquée ===
    meta = {
        "source": "P5.2.E.0 PyTorch attention oracle (reader layer 15 sliding × KV writer layer 13 sliding)",
        "spec_refs": [
            "transformers/models/gemma4/modeling_gemma4.py (5.9.0) eager_attention_forward L~982-1015",
            "Gemma4TextAttention.scaling = 1.0 (L772) ; num_key_value_groups = 8",
        ],
        "reader_layer_idx": READER_LAYER_IDX,
        "writer_layer_idx": WRITER_LAYER_IDX,
        "layer_type": LAYER_TYPE,
        "config": {
            "head_dim": HEAD_DIM, "n_heads": N_HEADS, "n_kv": N_KV,
            "num_key_value_groups": NUM_KEY_VALUE_GROUPS, "seq_len": SEQ_LEN,
            "sliding_window": SLIDING_WINDOW, "scaling": SCALING, "softcap": SOFTCAP,
            "rms_norm_eps": 1e-06,
        },
        "inputs": {
            "q_final": "fixture q_only_reader_layer15.safetensors (P5.2.C)",
            "k_final": "fixture p5_2_d0_kv_oracle_layer13.pt (P5.2.D.0b, V RMSNorm no-scale)",
            "v_final": "fixture p5_2_d0_kv_oracle_layer13.pt (P5.2.D.0b, V RMSNorm no-scale)",
        },
        "pipeline": [
            "key_states   = repeat_kv(k_final, 8)            [1,8,4,256]",
            "value_states = repeat_kv(v_final, 8)            [1,8,4,256]",
            "scores_raw   = matmul(q_final, key_states.T(2,3)) * 1.0   [1,8,4,4]",
            "scores_masked= scores_raw + causal_mask(additif)         [1,8,4,4]",
            "probs        = softmax(scores_masked, dim=-1, fp32)      [1,8,4,4]",
            "context      = matmul(probs, value_states)               [1,8,4,256]",
        ],
        "mask": {
            "effective": "causal only",
            "reason": f"sliding_window={SLIDING_WINDOW} > seq_len={SEQ_LEN} -> dégénère en causal",
            "sliding_equals_causal": degenerates,
            "additive_min": min_val,
            "warning": "E.0 ne valide PAS le sliding-window réel ; à tester en E.mask (S=8, window=3).",
        },
        "checks": {
            "max_prob_sum_err": max_sum_err,
            "future_prob_max": future_prob_max,
            "v_final_vs_v_raw_max": v_norm_vs_raw,
        },
        "stats": stats,
        "fixed_points": fixed,
        "tolerance_for_zml_le": TOL,
        "interdits_p5_2_e0": [
            "ZML", "runners ZML", "E.1", "layer 14 (full attention)",
            "softcap d'attention", "scaling 1/sqrt(head_dim)", "V brut non normé",
            "vrai sliding masking prétendu validé",
        ],
        "next": "P5.2.E.1 = ZML QK scores only, à comparer contre scores_raw de cette fixture.",
    }
    blob = {
        # Inputs (traçabilité / réutilisation E.1)
        "q_final": q_final,
        "k_final": k_final,
        "v_final": v_final,
        "key_states": key_states,
        "value_states": value_states,
        "causal_mask": causal_mask,
        "sliding_mask": sliding_mask,
        # Outputs attendus (contrat)
        "scores_raw": scores_raw,
        "scores_masked": scores_masked,
        "probs": probs,
        "context": context,
        # Métadonnées embarquées (pas de manifest .json séparé pour E.0)
        "meta": meta,
    }
    OUT_FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, str(OUT_FIXTURE))
    tensor_bytes = sum(
        v.numel() * v.element_size() for v in blob.values() if isinstance(v, torch.Tensor)
    )
    print(f"wrote {OUT_FIXTURE}  ({tensor_bytes} bytes of tensor payload)")
    print()
    print("meta (embedded in .pt, keys):", ", ".join(meta.keys()))
    print()
    print("P5.2.E.0 attention oracle PASS.")


if __name__ == "__main__":
    main()
