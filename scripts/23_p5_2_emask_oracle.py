"""P5.2.E.mask — Oracle PyTorch du masque sliding RÉEL (cas synthétique S=8, window=3).

But : fermer le trou de couverture laissé par E.0/E.1. À S=4 et sliding_window=512,
le masque sliding dégénère en masque causal pur (la fenêtre ne mord jamais). Ici on
prend un cas synthétique S=8, window=3 où sliding ≠ causal, pour valider la VRAIE
logique de fenêtrage.

Convention (source de vérité, NON inférée) :
  transformers/masking_utils.py sliding_window_overlay : `kv_idx > q_idx - sliding_window`
  composé (and_masks) avec causal `kv_idx <= q_idx`.
  -> visible  ⟺  (k <= q)  ET  (k > q - window)  ⟺  q - window < k <= q.
  Identique au helper ZML zml.nn.causalAttnMask (k_idx.cmp(.LE,q) AND q.cmp(.LT, k+window)).

Table de visibilité attendue (window=3) :
  q=0:[0]  q=1:[0,1]  q=2:[0,1,2]  q=3:[1,2,3]  q=4:[2,3,4]
  q=5:[3,4,5]  q=6:[4,5,6]  q=7:[5,6,7]
  -> 21 positions visibles, 43 masquées (sur 64). q=7 masqué : [0,1,2,3,4].

Scope STRICT E.mask : masque additif + application sur scores synthétiques. Comparaison
de scores_masked. PAS de softmax, PAS de context, PAS de dot(V), PAS de layer 14.

Sortie : `fixtures/p5_2_emask_sliding_layer_synthetic.safetensors` (scores_synth,
sliding_mask, scores_masked) + .pt + manifest.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]
OUT_ST = ROOT / "fixtures" / "p5_2_emask_sliding_layer_synthetic.safetensors"
OUT_PT = ROOT / "fixtures" / "p5_2_emask_sliding_layer_synthetic.pt"
OUT_MANIFEST = ROOT / "fixtures" / "p5_2_emask_sliding_layer_synthetic_manifest.json"

S = 8          # séquence synthétique (q = k = 8)
WINDOW = 3     # sliding_window mordant (window < S)
NH = 2         # n heads synthétiques (pour vérifier le broadcast du masque [1,1,S,S] -> heads)
SEED = 1337

# Table de visibilité de référence (q -> set des k visibles), window=3.
EXPECTED_VISIBLE = {
    0: {0},
    1: {0, 1},
    2: {0, 1, 2},
    3: {1, 2, 3},
    4: {2, 3, 4},
    5: {3, 4, 5},
    6: {4, 5, 6},
    7: {5, 6, 7},
}


def main() -> None:
    torch.manual_seed(SEED)
    min_val = torch.finfo(torch.float32).min

    # === Masque sliding additif [S, S] dérivé de la convention source. ===
    q_idx = torch.arange(S).view(S, 1)
    k_idx = torch.arange(S).view(1, S)
    causal = k_idx <= q_idx                 # k <= q
    window = k_idx > (q_idx - WINDOW)       # k > q - window
    visible = causal & window               # [S, S] bool
    sliding_mask = torch.where(
        visible, torch.zeros(()), torch.full((), min_val)
    ).to(torch.float32)                     # [S, S] additif (0 / finfo.min)

    # === Vérification stricte de la table (assertions = la table dérivée == cible). ===
    print(f"Sliding-window mask S={S}, window={WINDOW} (visible 'o' / masqué '.') :")
    n_visible = 0
    for q in range(S):
        row_vis = {k for k in range(S) if visible[q, k].item()}
        assert row_vis == EXPECTED_VISIBLE[q], (
            f"q={q}: dérivé {sorted(row_vis)} != attendu {sorted(EXPECTED_VISIBLE[q])}"
        )
        n_visible += len(row_vis)
        row = "".join("o" if visible[q, k] else "." for k in range(S))
        print(f"  q={q}: [{row}]  visible k={sorted(row_vis)}")
    n_masked = S * S - n_visible
    assert n_visible == 21, f"n_visible {n_visible} != 21"
    assert n_masked == 43, f"n_masked {n_masked} != 43"
    # Cas-clé de Régis : q=7 masqué exactement [0,1,2,3,4].
    masked_q7 = {k for k in range(S) if not visible[7, k].item()}
    assert masked_q7 == {0, 1, 2, 3, 4}, f"q=7 masqué {sorted(masked_q7)} != [0,1,2,3,4]"
    print(f"  -> {n_visible} visibles, {n_masked} masquées ; q=7 masqué = {sorted(masked_q7)} ✓")
    print()

    # === Scores synthétiques [1, NH, S, S] déterministes + application du masque. ===
    scores_synth = torch.randn(1, NH, S, S, dtype=torch.float32)
    scores_masked = scores_synth + sliding_mask.view(1, 1, S, S)   # broadcast additif

    # Sanity application : visible -> inchangé (bit-exact) ; masqué -> finfo.min absorbe le score.
    vis_b = visible.view(1, 1, S, S)
    diff_visible = (scores_masked - scores_synth)[vis_b.expand_as(scores_masked)].abs().max().item()
    assert diff_visible == 0.0, f"positions visibles altérées (diff {diff_visible})"
    masked_min = scores_masked[(~vis_b).expand_as(scores_masked)].max().item()
    assert masked_min < -1e30, f"positions masquées pas assez négatives (max {masked_min})"
    print(f"Application : visible inchangé (Δ={diff_visible}) ; masqué max={masked_min:.3e} (< -1e30) ✓")
    print()

    # Fixed points (informational).
    print("Fixed points (scores_masked[0,0,7,:8], q=7) — attendu masqué sur k=0..4 :")
    print("  " + ", ".join(f"{v:.4e}" for v in scores_masked[0, 0, 7, :].tolist()))
    print()

    tensors = {
        "scores_synth": scores_synth.contiguous(),
        "sliding_mask": sliding_mask.contiguous(),       # [S, S] additif (oracle du masque ZML)
        "scores_masked": scores_masked.contiguous(),     # [1, NH, S, S]
    }
    OUT_ST.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(OUT_ST))
    torch.save(tensors, str(OUT_PT))
    total = sum(t.numel() * t.element_size() for t in tensors.values())
    print(f"wrote {OUT_ST} and {OUT_PT.name} ({total} bytes payload)")

    manifest = {
        "source": "P5.2.E.mask oracle — sliding window mask réel, cas synthétique S=8 window=3",
        "spec_refs": [
            "transformers/masking_utils.py sliding_window_overlay (kv_idx > q_idx - sliding_window) + causal",
            "zml/nn.zig causalAttnMask (k.cmp(.LE,q) AND q.cmp(.LT,k+window)) — helper ZML utilisé par E.mask",
        ],
        "S": S, "window": WINDOW, "n_heads": NH, "seed": SEED,
        "convention": "visible <=> (k <= q) AND (k > q - window) <=> q-window < k <= q",
        "n_visible": n_visible, "n_masked": n_masked,
        "min_val_finfo_f32": min_val,
        "tensors": {name: {"shape": list(t.shape), "dtype": "float32"} for name, t in tensors.items()},
        "zml_pipeline_hint": [
            "mask = zml.nn.causalAttnMask(.{ .q = 8, .k = 8 }, .f32, 3)   -> {.q,.k} additif",
            "compare mask vs sliding_mask oracle (structure 0/finfo.min)",
            "scores_masked = scores_synth.add(mask.broad(scores_synth.shape()))",
            "compare scores_masked vs oracle (visible bit-exact, masqué < -1e30)",
        ],
        "scope": "mask only — pas de softmax, pas de context, pas de dot(V), pas de layer 14",
        "interdits_p5_2_emask": ["softmax", "context", "dot(V)", "layer 14", "full attention"],
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {OUT_MANIFEST}")
    print()
    print("P5.2.E.mask oracle PASS.")


if __name__ == "__main__":
    main()
