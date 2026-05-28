"""P5.2.D.5 — Export safetensors fixture pour la sous-gate ZML KV slot mock.

Lit `fixtures/p5_2_d0_kv_oracle_layer13.pt` (artefact figé par D.0) et
extrait un sous-ensemble minimal pour D.5 :

    hidden_input     [B,S,H]            = [1, 4, 1536]
    k_proj_weight    [n_kv*hd, H]       = [256, 1536]
    k_norm_weight    [head_dim]         = [256]
    v_proj_weight    [n_kv*hd, H]       = [256, 1536]
    k_final          [B,n_kv,S,hd]      = [1, 1, 4, 256]   (oracle PyTorch — cache layout)
    v_final          [B,n_kv,S,hd]      = [1, 1, 4, 256]   (oracle PyTorch — cache layout)

Sortie : `fixtures/p5_2_d5_kv_slot_layer13.safetensors` + manifest JSON.

Décision layout : on cible le **cache layout** `[1, 1, 4, 256]` = `{b, kvh, s, hd}`
(option B), miroir des `k_final` / `v_final` D.0 (transpose(1,2) appliqué côté
PyTorch). Le runner ZML calculera d'abord en compute layout `[1, 4, 1, 256]`
puis appliquera un `transpose(.{.b, .kvh, .s, .hd})` final. Avec `n_kv=1`,
cette transposition est un **no-op en mémoire** (dim singleton ne réordonne
pas), mais reste explicite pour préparer les futures couches non-singleton.

Garde-fous D.5 : pas d'attention, pas de Q path, pas de reader, pas de
layer 14, pas de sliding mask, pas de cache dynamique réel.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]
IN_FIXTURE = ROOT / "fixtures" / "p5_2_d0_kv_oracle_layer13.pt"
OUT_FIXTURE = ROOT / "fixtures" / "p5_2_d5_kv_slot_layer13.safetensors"
OUT_MANIFEST = ROOT / "fixtures" / "p5_2_d5_kv_slot_layer13_manifest.json"

LAYER_IDX = 13
LAYER_TYPE = "sliding_attention"
RMS_EPS = 1.0e-6
ROPE_THETA = 10000.0
EXPECTED_HIDDEN_SHAPE = (1, 4, 1536)
EXPECTED_K_PROJ_W_SHAPE = (256, 1536)
EXPECTED_K_NORM_W_SHAPE = (256,)
EXPECTED_V_PROJ_W_SHAPE = (256, 1536)
EXPECTED_K_FINAL_SHAPE = (1, 1, 4, 256)
EXPECTED_V_FINAL_SHAPE = (1, 1, 4, 256)


def main() -> None:
    assert IN_FIXTURE.exists(), f"missing D.0 fixture {IN_FIXTURE}"
    blob = torch.load(str(IN_FIXTURE), map_location="cpu", weights_only=False)

    hidden_input = blob["hidden_input"]
    k_proj_weight = blob["k_proj_weight"]
    k_norm_weight = blob["k_norm_weight"]
    v_proj_weight = blob["v_proj_weight"]
    k_final = blob["k_final"]
    v_final = blob["v_final"]

    # Sanity shape & dtype.
    for name, t, expected in [
        ("hidden_input", hidden_input, EXPECTED_HIDDEN_SHAPE),
        ("k_proj_weight", k_proj_weight, EXPECTED_K_PROJ_W_SHAPE),
        ("k_norm_weight", k_norm_weight, EXPECTED_K_NORM_W_SHAPE),
        ("v_proj_weight", v_proj_weight, EXPECTED_V_PROJ_W_SHAPE),
        ("k_final", k_final, EXPECTED_K_FINAL_SHAPE),
        ("v_final", v_final, EXPECTED_V_FINAL_SHAPE),
    ]:
        assert tuple(t.shape) == expected, (
            f"{name} shape {tuple(t.shape)} != {expected}"
        )
        assert t.dtype == torch.float32, f"{name} dtype {t.dtype} != float32"

    # Cross-check : avec n_kv=1, k_final[0,0,s,:] == k_after_rope[0,s,0,:]
    # bit-exact (transpose dim singleton = no-op mémoire).
    k_after_rope = blob["k_after_rope"]  # [1, 4, 1, 256] {b, s, kvh, hd}
    v_after_reshape = blob["v_after_reshape"]  # [1, 4, 1, 256] {b, s, kvh, hd}
    # k_final = k_after_rope.transpose(1, 2).contiguous() côté D.0.
    # avec n_kv=1, doit donner les mêmes valeurs réorganisées.
    k_compute_vs_final = (k_after_rope[0, :, 0, :] - k_final[0, 0, :, :]).abs().max().item()
    v_compute_vs_final = (v_after_reshape[0, :, 0, :] - v_final[0, 0, :, :]).abs().max().item()
    print(
        f"Sanity transpose K (compute -> cache) |diff|_max : {k_compute_vs_final:.6e}  "
        f"(expected 0.0 strict, n_kv=1 transpose no-op)"
    )
    print(
        f"Sanity transpose V (compute -> cache) |diff|_max : {v_compute_vs_final:.6e}  "
        f"(expected 0.0 strict, n_kv=1 transpose no-op)"
    )
    assert k_compute_vs_final == 0.0, (
        f"K compute vs cache should be bit-exact for n_kv=1, got {k_compute_vs_final}"
    )
    assert v_compute_vs_final == 0.0, (
        f"V compute vs cache should be bit-exact for n_kv=1, got {v_compute_vs_final}"
    )
    print()

    # === Fixed-point blocks (informational). ===
    # Cache layout [1,1,4,256] : [0,0,s,:8] = flat s*256.
    print("Fixed points (cache layout k_final / v_final, kvh=0):")
    for s in (0, 3):
        kblk = k_final[0, 0, s, :8].tolist()
        vblk = v_final[0, 0, s, :8].tolist()
        kstr = ", ".join(f"{v:.10f}" for v in kblk)
        vstr = ", ".join(f"{v:.10f}" for v in vblk)
        print(f"  k_final[0, 0, {s}, :8] = [{kstr}]")
        print(f"  v_final[0, 0, {s}, :8] = [{vstr}]")
    print()

    # Stats.
    print("Stats:")
    for name, t in [
        ("hidden_input", hidden_input),
        ("k_proj_weight", k_proj_weight),
        ("k_norm_weight", k_norm_weight),
        ("v_proj_weight", v_proj_weight),
        ("k_final", k_final),
        ("v_final", v_final),
    ]:
        print(
            f"  {name:<14} shape={tuple(t.shape)!s:<20} dtype={t.dtype} "
            f"mean={t.mean().item(): .6e} std={t.std().item(): .6e} "
            f"min={t.min().item(): .6e} max={t.max().item(): .6e}"
        )

    # === Write safetensors fixture (6 tensors). ===
    tensors = {
        "hidden_input": hidden_input.contiguous(),
        "k_proj_weight": k_proj_weight.contiguous(),
        "k_norm_weight": k_norm_weight.contiguous(),
        "v_proj_weight": v_proj_weight.contiguous(),
        "k_final": k_final.contiguous(),
        "v_final": v_final.contiguous(),
    }
    OUT_FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(OUT_FIXTURE))
    total_bytes = sum(t.numel() * t.element_size() for t in tensors.values())
    print()
    print(f"wrote {OUT_FIXTURE}  ({total_bytes} bytes of tensor payload)")

    manifest = {
        "source": "P5.2.D.5 ZML KV slot mock fixture (slim from D.0)",
        "derived_from": str(IN_FIXTURE.name),
        "layer_idx": LAYER_IDX,
        "layer_type": LAYER_TYPE,
        "is_writer_producer": True,
        "rms_eps": RMS_EPS,
        "rope_theta": ROPE_THETA,
        "slot_layout_decision": {
            "chosen": "cache layout [1, 1, 4, 256] = {b, kvh, s, hd}",
            "rationale": "Mirror PyTorch k_final / v_final D.0 (transpose(1,2) appliqué). "
            "Plus proche du futur cache réel. Avec n_kv=1, no-op mémoire mais "
            "explicite pour préparer les futures couches.",
            "compute_layout": "[1, 4, 1, 256] = {b, s, kvh, hd}",
            "cache_layout": "[1, 1, 4, 256] = {b, kvh, s, hd}",
            "transition": "tensor.transpose(.{.b, .kvh, .s, .hd}) côté ZML",
        },
        "tensors": {
            name: {
                "shape": list(t.shape),
                "dtype": str(t.dtype).replace("torch.", ""),
            }
            for name, t in tensors.items()
        },
        "pipeline": [
            "hidden_input [B,S,H]",
            "k_after_proj = hidden_input @ k_proj_weight.T  [B,S,n_kv*hd]",
            "k_4d         = k_after_proj.reshape(B,S,n_kv,hd).withTags(.{.b,.s,.kvh,.hd})",
            "k_normalized = rmsNorm(k_4d, .hd, 1e-6)",
            "k_after_norm = k_normalized * k_norm_weight.broad(shape)",
            "k_after_rope = zml.nn.rope(k_after_norm, null, default sequential theta=10000)",
            "k_slot       = k_after_rope.transpose(.{.b, .kvh, .s, .hd})  [B,n_kv,S,hd]",
            "v_after_proj = hidden_input @ v_proj_weight.T  [B,S,n_kv*hd]",
            "v_4d         = v_after_proj.reshape(B,S,n_kv,hd).withTags(.{.b,.s,.kvh,.hd})",
            "v_slot       = v_4d.transpose(.{.b, .kvh, .s, .hd})  [B,n_kv,S,hd]   (V non normé, non roté)",
            "return (k_slot, v_slot)",
        ],
        "transpose_sanity_pytorch": {
            "k_compute_vs_cache_max_abs": k_compute_vs_final,
            "v_compute_vs_cache_max_abs": v_compute_vs_final,
            "explanation": "With n_kv=1, the singleton dim transpose is a memory no-op (bit-exact).",
        },
        "expected_zml_max_abs_le": 1.0e-4,
        "expected_zml_residual_hint": "K side ~ D.4 résidu ~5e-7 (RoPE orthogonale), V side ~ D.2 résidu ~5e-6 (matmul brut)",
        "interdits_p5_2_d5": [
            "attention",
            "Q path",
            "reader (layer 15-34)",
            "layer 14 (full attention proportional)",
            "sliding mask",
            "cache dynamique réel (scatter / dynamicSlice)",
        ],
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {OUT_MANIFEST}")
    print()
    print("P5.2.D.5 fixture export PASS.")


if __name__ == "__main__":
    main()
