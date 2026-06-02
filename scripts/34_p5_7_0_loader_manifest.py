"""P5.7.0 - Build the Gemma 4 E2B text loader manifest.

This gate does not compute a forward pass and does not load tensor payloads.
It materializes the full text weight plan needed by P5.7:

- top-level embeddings / PLE / final norm / tied lm_head alias;
- 35 decoder layer weight keys and expected shapes;
- runtime KV ownership from the P5.1 YOCO policy table;
- disk-only K/V weights on reader layers, which exist in the raw checkpoint
  but must not be loaded/executed by the runtime.

If `weights/model.safetensors` is present, the script also validates key
presence and shapes through safetensors metadata/slices only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "gemma4-e2b-it-meta" / "config.json"
POLICY_PATH = ROOT / "fixtures" / "yoco_policy_table.json"
DEFAULT_WEIGHTS = ROOT / "weights" / "model.safetensors"
OUT_MANIFEST = ROOT / "fixtures" / "p5_7_0_loader_manifest.json"

REPO_ID = "google/gemma-4-E2B-it"
TEXT_PREFIX = "model.language_model"
DTYPE = "bfloat16"


def key_entry(
    logical_name: str,
    key: str | None,
    shape: list[int] | None,
    *,
    role: str,
    load_for_runtime: bool,
    disk_expected: bool = True,
    note: str | None = None,
    alias_of: str | None = None,
) -> dict[str, Any]:
    return {
        "logical_name": logical_name,
        "key": key,
        "shape": shape,
        "dtype": DTYPE if shape is not None else None,
        "role": role,
        "disk_expected": disk_expected,
        "load_for_runtime": load_for_runtime,
        **({"alias_of": alias_of} if alias_of else {}),
        **({"note": note} if note else {}),
    }


def attention_dims(tc: dict[str, Any], layer_type: str) -> dict[str, int]:
    if layer_type == "full_attention":
        head_dim = int(tc["global_head_dim"])
    elif layer_type == "sliding_attention":
        head_dim = int(tc["head_dim"])
    else:
        raise ValueError(f"unknown layer_type {layer_type!r}")
    return {
        "head_dim": head_dim,
        "q_width": int(tc["num_attention_heads"]) * head_dim,
        "kv_width": int(tc["num_key_value_heads"]) * head_dim,
    }


def layer_intermediate(tc: dict[str, Any], is_reader: bool) -> int:
    base = int(tc["intermediate_size"])
    if bool(tc.get("use_double_wide_mlp")) and is_reader:
        return base * 2
    return base


def layer_entries(tc: dict[str, Any], row: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    layer_idx = int(row["layer_idx"])
    layer_type = str(row["layer_type"])
    is_reader = bool(row["is_reader"])
    target_kv_layer = int(row["target_kv_layer"])
    hidden = int(tc["hidden_size"])
    ple_dim = int(tc["hidden_size_per_layer_input"])
    dims = attention_dims(tc, layer_type)
    intermediate = layer_intermediate(tc, is_reader)
    prefix = f"{TEXT_PREFIX}.layers.{layer_idx}"

    common = [
        key_entry("input_layernorm", f"{prefix}.input_layernorm.weight", [hidden], role="decoder_norm", load_for_runtime=True),
        key_entry("q_proj", f"{prefix}.self_attn.q_proj.weight", [dims["q_width"], hidden], role="attention_q", load_for_runtime=True),
        key_entry("q_norm", f"{prefix}.self_attn.q_norm.weight", [dims["head_dim"]], role="attention_q", load_for_runtime=True),
        key_entry("o_proj", f"{prefix}.self_attn.o_proj.weight", [hidden, dims["q_width"]], role="attention_o", load_for_runtime=True),
        key_entry("post_attention_layernorm", f"{prefix}.post_attention_layernorm.weight", [hidden], role="decoder_norm", load_for_runtime=True),
        key_entry("pre_feedforward_layernorm", f"{prefix}.pre_feedforward_layernorm.weight", [hidden], role="decoder_norm", load_for_runtime=True),
        key_entry("mlp.gate_proj", f"{prefix}.mlp.gate_proj.weight", [intermediate, hidden], role="mlp", load_for_runtime=True),
        key_entry("mlp.up_proj", f"{prefix}.mlp.up_proj.weight", [intermediate, hidden], role="mlp", load_for_runtime=True),
        key_entry("mlp.down_proj", f"{prefix}.mlp.down_proj.weight", [hidden, intermediate], role="mlp", load_for_runtime=True),
        key_entry("post_feedforward_layernorm", f"{prefix}.post_feedforward_layernorm.weight", [hidden], role="decoder_norm", load_for_runtime=True),
        key_entry("per_layer_input_gate", f"{prefix}.per_layer_input_gate.weight", [ple_dim, hidden], role="ple_injection", load_for_runtime=True),
        key_entry("per_layer_projection", f"{prefix}.per_layer_projection.weight", [hidden, ple_dim], role="ple_injection", load_for_runtime=True),
        key_entry("post_per_layer_input_norm", f"{prefix}.post_per_layer_input_norm.weight", [hidden], role="ple_injection", load_for_runtime=True),
        key_entry("layer_scalar", f"{prefix}.layer_scalar", [1], role="decoder_scale", load_for_runtime=True),
    ]

    kv_note = None
    if is_reader:
        kv_note = (
            f"disk-only on YOCO reader; runtime reads K/V from layer {target_kv_layer} "
            f"({layer_type}) and must not instantiate these K/V weights"
        )
    kv_load = not is_reader
    kv = [
        key_entry("k_proj", f"{prefix}.self_attn.k_proj.weight", [dims["kv_width"], hidden], role="attention_k", load_for_runtime=kv_load, note=kv_note),
        key_entry("k_norm", f"{prefix}.self_attn.k_norm.weight", [dims["head_dim"]], role="attention_k", load_for_runtime=kv_load, note=kv_note),
        key_entry("v_proj", f"{prefix}.self_attn.v_proj.weight", [dims["kv_width"], hidden], role="attention_v", load_for_runtime=kv_load, note=kv_note),
        key_entry(
            "v_norm",
            None,
            None,
            role="attention_v",
            load_for_runtime=not is_reader,
            disk_expected=False,
            note="Gemma4RMSNorm with_scale=False: runtime op has no checkpoint weight",
        ),
    ]

    meta = {
        "layer_idx": layer_idx,
        "layer_type": layer_type,
        "is_reader": is_reader,
        "target_kv_layer": target_kv_layer,
        "runtime_kv_role": "reader" if is_reader else "producer",
        "attention": dims,
        "mlp_intermediate": intermediate,
    }
    return meta, common[:3] + kv + common[3:]


def build_manifest(config: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    tc = config["text_config"]
    hidden = int(tc["hidden_size"])
    num_layers = int(tc["num_hidden_layers"])
    ple_dim = int(tc["hidden_size_per_layer_input"])
    packed_ple = num_layers * ple_dim
    vocab = int(tc["vocab_size"])

    top_level = [
        key_entry("embed_tokens", f"{TEXT_PREFIX}.embed_tokens.weight", [vocab, hidden], role="embedding", load_for_runtime=True),
        key_entry("lm_head", f"{TEXT_PREFIX}.embed_tokens.weight", [vocab, hidden], role="lm_head", load_for_runtime=False, alias_of="embed_tokens", note="tied weights; no second load"),
        key_entry("final_norm", f"{TEXT_PREFIX}.norm.weight", [hidden], role="final_norm", load_for_runtime=True),
        key_entry("embed_tokens_per_layer", f"{TEXT_PREFIX}.embed_tokens_per_layer.weight", [vocab, packed_ple], role="ple_frontend", load_for_runtime=True),
        key_entry("per_layer_model_projection", f"{TEXT_PREFIX}.per_layer_model_projection.weight", [packed_ple, hidden], role="ple_frontend", load_for_runtime=True),
        key_entry("per_layer_projection_norm", f"{TEXT_PREFIX}.per_layer_projection_norm.weight", [ple_dim], role="ple_frontend", load_for_runtime=True),
    ]

    layers = []
    disk_expected_entries: list[dict[str, Any]] = []
    runtime_entries: list[dict[str, Any]] = []
    runtime_tensor_load_entries: list[dict[str, Any]] = []

    for row in policy["table"]:
        meta, entries = layer_entries(tc, row)
        layers.append({**meta, "weights": entries})
        for entry in entries:
            if entry["disk_expected"]:
                disk_expected_entries.append(entry)
            if entry["load_for_runtime"]:
                runtime_entries.append(entry)
                if entry["key"]:
                    runtime_tensor_load_entries.append(entry)

    for entry in top_level:
        if entry["disk_expected"] and entry["logical_name"] != "lm_head":
            disk_expected_entries.append(entry)
        if entry["load_for_runtime"]:
            runtime_entries.append(entry)
            if entry["key"]:
                runtime_tensor_load_entries.append(entry)

    ignored_reader_kv = [
        entry
        for layer in layers
        if layer["is_reader"]
        for entry in layer["weights"]
        if entry["key"] and entry["logical_name"] in {"k_proj", "k_norm", "v_proj"}
    ]

    manifest = {
        "source": "P5.7.0 loader manifest only (no compute, no tensor payload load)",
        "repo": REPO_ID,
        "checkpoint_prefix": TEXT_PREFIX,
        "spec_refs": [
            "P5.1 yoco_policy_table.json",
            "docs/P5_0_yoco_shared_kv_cartography.md: disk has 35 K/V sets, runtime loads 15",
            "docs/P5_6_closeout.md: complete component coverage before P5.7",
        ],
        "config": {
            "num_hidden_layers": num_layers,
            "num_kv_shared_layers": int(tc["num_kv_shared_layers"]),
            "first_kv_shared_layer_idx": int(policy["summary"]["first_kv_shared_layer_idx"]),
            "hidden_size": hidden,
            "head_dim": int(tc["head_dim"]),
            "global_head_dim": int(tc["global_head_dim"]),
            "num_attention_heads": int(tc["num_attention_heads"]),
            "num_key_value_heads": int(tc["num_key_value_heads"]),
            "intermediate_size": int(tc["intermediate_size"]),
            "use_double_wide_mlp": bool(tc["use_double_wide_mlp"]),
            "hidden_size_per_layer_input": ple_dim,
            "vocab_size": vocab,
            "sliding_window": int(tc["sliding_window"]),
            "final_logit_softcapping": float(tc["final_logit_softcapping"]),
            "tie_word_embeddings": bool(tc["tie_word_embeddings"]),
            "enable_moe_block": bool(tc["enable_moe_block"]),
        },
        "top_level": top_level,
        "layers": layers,
        "summary": {
            "top_level_disk_keys": len([e for e in top_level if e["disk_expected"] and e["logical_name"] != "lm_head"]),
            "layer_disk_keys": len(disk_expected_entries) - len([e for e in top_level if e["disk_expected"] and e["logical_name"] != "lm_head"]),
            "total_disk_keys_expected": len(disk_expected_entries),
            "runtime_weighted_ops": len(runtime_entries),
            "runtime_tensors_to_load": len(runtime_tensor_load_entries),
            "reader_kv_disk_keys_ignored_at_runtime": len(ignored_reader_kv),
            "no_weight_runtime_ops": ["v_norm (RMSNorm with_scale=False)", "embed_scale sqrt(hidden)", "PLE fusion scales", "lm_head tied alias", "rotary tables/masks generated in later gates"],
            "producers": [r["layer_idx"] for r in policy["table"] if not r["is_reader"]],
            "readers": [r["layer_idx"] for r in policy["table"] if r["is_reader"]],
            "writers_per_layer_type": policy["summary"]["writers_per_layer_type"],
        },
    }
    return manifest


def expected_key_shapes(manifest: dict[str, Any]) -> dict[str, list[int]]:
    expected: dict[str, list[int]] = {}
    for entry in manifest["top_level"]:
        if entry["disk_expected"] and entry["key"] and entry["logical_name"] != "lm_head":
            expected[entry["key"]] = entry["shape"]
    for layer in manifest["layers"]:
        for entry in layer["weights"]:
            if entry["disk_expected"] and entry["key"]:
                expected[entry["key"]] = entry["shape"]
    return expected


def validate_weights(weights_path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        return {
            "enabled": False,
            "status": "SKIPPED",
            "reason": f"safetensors unavailable: {exc}",
        }

    expected = expected_key_shapes(manifest)
    missing: list[str] = []
    shape_mismatches: list[dict[str, Any]] = []

    with safe_open(str(weights_path), framework="pt", device="cpu") as f:
        keys = set(f.keys())
        for key, shape in expected.items():
            if key not in keys:
                missing.append(key)
                continue
            tensor_slice = f.get_slice(key)
            actual_shape = list(tensor_slice.get_shape())
            if actual_shape != shape:
                shape_mismatches.append(
                    {"key": key, "expected": shape, "actual": actual_shape}
                )

    status = "PASS" if not missing and not shape_mismatches else "FAIL"
    return {
        "enabled": True,
        "status": status,
        "weights_path": str(weights_path),
        "checked_keys": len(expected),
        "missing": missing,
        "shape_mismatches": shape_mismatches,
    }


def write_manifest(payload: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--policy", type=Path, default=POLICY_PATH)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--out", type=Path, default=OUT_MANIFEST)
    parser.add_argument("--require-weights", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    policy = json.loads(args.policy.read_text())
    manifest = build_manifest(config, policy)

    if args.weights.exists():
        validation = validate_weights(args.weights, manifest)
    else:
        validation = {
            "enabled": False,
            "status": "SKIPPED",
            "weights_path": str(args.weights),
            "reason": "checkpoint not present locally; static loader plan emitted",
        }
    manifest["weights_validation"] = validation

    write_manifest(manifest, args.out)

    print("=== P5.7.0 loader manifest ===")
    print(f"wrote: {args.out}")
    print(f"layers: {manifest['config']['num_hidden_layers']}")
    print(f"disk keys expected: {manifest['summary']['total_disk_keys_expected']}")
    print(f"runtime tensors to load: {manifest['summary']['runtime_tensors_to_load']}")
    print(f"reader K/V disk keys ignored at runtime: {manifest['summary']['reader_kv_disk_keys_ignored_at_runtime']}")
    print(f"weights validation: {validation['status']}")

    if validation["enabled"] and validation["status"] != "PASS":
        print(json.dumps(validation, indent=2), file=sys.stderr)
        raise SystemExit(1)
    if args.require_weights and not validation["enabled"]:
        raise SystemExit("BLOCK: --require-weights set but checkpoint validation was skipped")
    print("P5.7.0 PASS")


if __name__ == "__main__":
    main()
