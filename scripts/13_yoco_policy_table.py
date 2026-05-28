"""P5.1 — YOCO policy table (statique, lecture seule).

Produit la table d'accès cache K/V de 35 entrées pour Gemma 4 E2B :
- layers 0..14 = producers (own KV)
- layers 15..34 = readers (target = last producer of same layer_type)

Reproduit la formule Transformers `Gemma4TextAttention.__init__`
(modeling_gemma4.py:777-782) et vLLM `Gemma4Attention.__init__`
(gemma4.py:469-489), strictement identique dans les deux runtimes.

Pas d'attention, pas de RoPE, pas de model load. Config seule.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from transformers import AutoConfig


REPO_ID = "google/gemma-4-E2B-it"
OUT_PATH = Path(__file__).parent.parent / "fixtures" / "yoco_policy_table.json"


def compute_policy_table(
    num_hidden_layers: int,
    num_kv_shared_layers: int,
    layer_types: list[str],
) -> tuple[list[dict], dict]:
    """Returns (rows, summary). Mirrors Transformers + vLLM logic verbatim.

    For each layer i:
      first_kv_shared_layer_idx = num_hidden_layers - num_kv_shared_layers
      is_reader = i >= first_kv_shared_layer_idx (and num_kv_shared_layers > 0)
      target_kv_layer:
        if not is_reader: i
        else: last index j < first_kv_shared_layer_idx with layer_types[j] == layer_types[i]
    """
    assert len(layer_types) == num_hidden_layers, (
        f"layer_types length {len(layer_types)} != num_hidden_layers {num_hidden_layers}"
    )
    first_kv_shared = num_hidden_layers - num_kv_shared_layers
    prev = layer_types[:first_kv_shared]

    rows: list[dict] = []
    for i, t in enumerate(layer_types):
        is_reader = num_kv_shared_layers > 0 and i >= first_kv_shared
        if is_reader:
            # mirror prev_layers[::-1].index(t) → last occurrence of t in prev
            try:
                target = len(prev) - 1 - prev[::-1].index(t)
            except ValueError as e:
                raise RuntimeError(
                    f"layer {i} type {t!r} has no producer in prev_layers[0..{first_kv_shared})"
                ) from e
        else:
            target = i
        rows.append(
            {
                "layer_idx": i,
                "layer_type": t,
                "is_reader": is_reader,
                "target_kv_layer": target,
            }
        )

    producers = [r for r in rows if not r["is_reader"]]
    readers = [r for r in rows if r["is_reader"]]
    # 2 designated writers per Transformers store_full_length_kv:
    # the last non-reader of each unique layer_type encountered in prev.
    writers: dict[str, int] = {}
    for r in producers:
        writers[r["layer_type"]] = r["layer_idx"]  # last wins

    summary = {
        "first_kv_shared_layer_idx": first_kv_shared,
        "num_producers": len(producers),
        "num_readers": len(readers),
        "writers_per_layer_type": writers,
        "producers_by_layer_type": {
            t: [r["layer_idx"] for r in producers if r["layer_type"] == t]
            for t in sorted(set(layer_types))
        },
        "readers_by_layer_type": {
            t: [r["layer_idx"] for r in readers if r["layer_type"] == t]
            for t in sorted(set(layer_types))
        },
        "readers_target_distribution": {
            str(target): sorted(
                r["layer_idx"] for r in readers if r["target_kv_layer"] == target
            )
            for target in sorted(set(r["target_kv_layer"] for r in readers))
        },
    }
    return rows, summary


def validate_e2b(rows: list[dict], summary: dict) -> None:
    """Assertions à partir des findings P5.0 pour gemma-4-E2B-it."""
    errors: list[str] = []

    # Counts
    if summary["num_producers"] != 15:
        errors.append(f"expected 15 producers, got {summary['num_producers']}")
    if summary["num_readers"] != 20:
        errors.append(f"expected 20 readers, got {summary['num_readers']}")
    if summary["first_kv_shared_layer_idx"] != 15:
        errors.append(
            f"expected first_kv_shared_layer_idx=15, got {summary['first_kv_shared_layer_idx']}"
        )

    # Writers
    if summary["writers_per_layer_type"].get("full_attention") != 14:
        errors.append(
            f"full writer expected 14, got {summary['writers_per_layer_type'].get('full_attention')}"
        )
    if summary["writers_per_layer_type"].get("sliding_attention") != 13:
        errors.append(
            f"sliding writer expected 13, got {summary['writers_per_layer_type'].get('sliding_attention')}"
        )

    # Readers target only 13 or 14
    targets = set(summary["readers_target_distribution"].keys())
    if targets != {"13", "14"}:
        errors.append(f"reader targets expected {{13, 14}}, got {targets}")

    # Each producer i has target==i (identity)
    for r in rows:
        if not r["is_reader"] and r["target_kv_layer"] != r["layer_idx"]:
            errors.append(
                f"producer {r['layer_idx']} should target self, got {r['target_kv_layer']}"
            )
        if r["is_reader"]:
            expected = 14 if r["layer_type"] == "full_attention" else 13
            if r["target_kv_layer"] != expected:
                errors.append(
                    f"reader {r['layer_idx']} ({r['layer_type']}) target {r['target_kv_layer']} != {expected}"
                )

    if errors:
        print("VALIDATION FAILED:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        sys.exit(1)
    print("validation: PASS")


def main() -> None:
    cfg = AutoConfig.from_pretrained(REPO_ID)
    tc = cfg.text_config

    num_hidden_layers = tc.num_hidden_layers
    num_kv_shared_layers = getattr(tc, "num_kv_shared_layers", 0)
    layer_types = list(tc.layer_types)
    sliding_window = getattr(tc, "sliding_window", None)

    print(f"repo                  : {REPO_ID}")
    print(f"num_hidden_layers     : {num_hidden_layers}")
    print(f"num_kv_shared_layers  : {num_kv_shared_layers}")
    print(f"sliding_window        : {sliding_window}")
    print()

    rows, summary = compute_policy_table(
        num_hidden_layers, num_kv_shared_layers, layer_types
    )

    # Pretty-print table
    print("layer_idx | layer_type        | is_reader | target_kv_layer")
    print("----------|-------------------|-----------|----------------")
    for r in rows:
        print(
            f"{r['layer_idx']:>9} | {r['layer_type']:<17} | {str(r['is_reader']):<9} | {r['target_kv_layer']:>15}"
        )
    print()
    print("=== summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print()

    validate_e2b(rows, summary)

    payload = {
        "repo": REPO_ID,
        "source": "P5.1 policy table from AutoConfig (no model load)",
        "spec_refs": [
            "transformers/models/gemma4/modeling_gemma4.py L777-L782 (Gemma4TextAttention.__init__)",
            "vllm/model_executor/models/gemma4.py L469-L489 (Gemma4Attention.__init__)",
        ],
        "config": {
            "num_hidden_layers": num_hidden_layers,
            "num_kv_shared_layers": num_kv_shared_layers,
            "sliding_window": sliding_window,
            "first_kv_shared_layer_idx": summary["first_kv_shared_layer_idx"],
        },
        "summary": summary,
        "table": rows,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
