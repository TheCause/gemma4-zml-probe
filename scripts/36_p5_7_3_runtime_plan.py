"""P5.7.3 — Plan de dispatch runtime des 35 couches (host-side, no compute).

Consolide, depuis config + policy table P5.1, la spec de dispatch par couche que le runtime
ZML (P5.7.4+) consomme : type sliding/full, producer/reader, source KV (YOCO), dims (head_dim
256/512, q/kv width, intermediate 6144/12288), RoPE (theta, partial_rotary, manuelle ou helper),
masque. Validé contre config et la policy. Émet fixtures/p5_7_3_runtime_plan.json.

Tout est dérivable (et déjà validé : P5.2.A policy lookup, P5.7.2 shapes). Ce gate fige la
spec de dispatch en un artefact unique pour le dispatcher et la revue cross-LLM.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "gemma4-e2b-it-meta" / "config.json"
POLICY = ROOT / "fixtures" / "yoco_policy_table.json"
OUT = ROOT / "fixtures" / "p5_7_3_runtime_plan.json"


def main() -> None:
    cfg = json.loads(CONFIG.read_text())["text_config"]
    policy = json.loads(POLICY.read_text())

    n_layers = int(cfg["num_hidden_layers"])
    n_heads = int(cfg["num_attention_heads"])
    n_kv = int(cfg["num_key_value_heads"])
    base_inter = int(cfg["intermediate_size"])
    dwm = bool(cfg["use_double_wide_mlp"])
    rope_params = cfg["rope_parameters"]

    by_idx = {int(r["layer_idx"]): r for r in policy["table"]}
    writers = {"sliding_attention": 13, "full_attention": 14}  # cf P5.1

    plan = []
    for i in range(n_layers):
        row = by_idx[i]
        lt = row["layer_type"]
        is_reader = bool(row["is_reader"])
        full = lt == "full_attention"
        head_dim = int(cfg["global_head_dim"]) if full else int(cfg["head_dim"])
        rp = rope_params[lt]
        intermediate = base_inter * 2 if (dwm and is_reader) else base_inter
        plan.append({
            "layer_idx": i,
            "layer_type": lt,
            "is_full_attention": full,
            "is_reader": is_reader,
            "is_producer": not is_reader,
            "is_kv_writer": (i == writers[lt]) and not is_reader,
            "target_kv_layer": int(row["target_kv_layer"]),
            "kv_source": ("self" if not is_reader else f"layer_{writers[lt]}"),
            "head_dim": head_dim,
            "q_width": n_heads * head_dim,
            "kv_width": n_kv * head_dim,
            "mlp_intermediate": intermediate,
            "mlp_double_wide": (dwm and is_reader),
            "rope_theta": float(rp["rope_theta"]),
            "rope_type": rp["rope_type"],
            "partial_rotary_factor": float(rp.get("partial_rotary_factor", 1.0)),
            "rope_manual": full,  # full = proportional partial -> RoPE manuelle (P5.6) ; sliding = zml.nn.rope
            "mask_type": "sliding_window" if lt == "sliding_attention" else "causal",
            "load_kv_weights": not is_reader,  # YOCO : readers ne chargent pas K/V au runtime
        })

    # === Assertions ===
    producers = [p["layer_idx"] for p in plan if p["is_producer"]]
    readers = [p["layer_idx"] for p in plan if p["is_reader"]]
    fulls = [p["layer_idx"] for p in plan if p["is_full_attention"]]
    writers_idx = sorted({p["layer_idx"] for p in plan if p["is_kv_writer"]})
    assert producers == list(range(15)), producers
    assert readers == list(range(15, 35)), readers
    assert fulls == [4, 9, 14, 19, 24, 29, 34], fulls
    assert writers_idx == [13, 14], writers_idx
    for p in plan:
        if p["is_full_attention"]:
            assert p["head_dim"] == 512 and p["q_width"] == 4096 and p["rope_theta"] == 1e6 and p["rope_manual"]
            assert abs(p["partial_rotary_factor"] - 0.25) < 1e-9
        else:
            assert p["head_dim"] == 256 and p["q_width"] == 2048 and p["rope_theta"] == 1e4 and not p["rope_manual"]
        if p["is_reader"]:
            assert p["mlp_intermediate"] == 12288 and not p["load_kv_weights"]
            assert p["target_kv_layer"] == (14 if p["is_full_attention"] else 13)
        else:
            assert p["mlp_intermediate"] == 6144 and p["load_kv_weights"]

    runtime_tensors = sum(17 if not p["load_kv_weights"] else 17 for p in plan)  # 17 keys/layer ; readers skip 3 k/v at runtime
    runtime_load = sum(14 if p["is_reader"] else 17 for p in plan)

    out = {
        "source": "P5.7.3 runtime dispatch plan (host-side, no compute)",
        "spec_refs": ["P5.1 yoco_policy_table.json", "P5.7.0 loader manifest", "config text_config"],
        "config": {"num_layers": n_layers, "n_heads": n_heads, "n_kv": n_kv,
                   "head_dim_sliding": int(cfg["head_dim"]), "head_dim_full": int(cfg["global_head_dim"]),
                   "intermediate": base_inter, "use_double_wide_mlp": dwm},
        "summary": {
            "producers": producers, "readers": readers, "full_attention_layers": fulls,
            "kv_writers": writers_idx, "runtime_tensors_loaded": runtime_load,
            "kv_writer_sliding": 13, "kv_writer_full": 14,
        },
        "layers": plan,
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print("=== P5.7.3 runtime plan ===")
    print(f"wrote {OUT}")
    print(f"producers={len(producers)} readers={len(readers)} full={fulls} writers={writers_idx}")
    print(f"runtime tensors to load: {runtime_load} (readers skip 3 K/V each via YOCO)")
    print("dispatch spec : full=head_dim512/rope manuelle θ1e6 partial0.25 ; sliding=256/zml.nn.rope θ1e4")
    print("P5.7.3 PASS")


if __name__ == "__main__":
    main()
