"""P4.4.0 — Export fixture PLE au format safetensors pour le runtime ZML.

Charge les 6 .npy produits par 06_export_ple_fixture.py et les empaquete dans
un unique fichier safetensors lisible nativement par ZML
(`zml.safetensors.TensorRegistry.fromPath`).

Met aussi a jour fixture_manifest.json avec la cle `primary_fixture` pointant
sur le .safetensors. Les .npy restent comme intermediaires verifiables.

Gate :
  PASS si toutes les shapes/dtypes attendues sont confirmees et que le fichier
  est ecrit, sinon BLOCK.
"""
from pathlib import Path
import json
import numpy as np
from safetensors.numpy import save_file

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "fixtures"
OUT = FIX / "ple_fixture.safetensors"
MANIFEST = FIX / "fixture_manifest.json"

FILES = {
    "ple_input_ids": "ple_input_ids.npy",
    "embed_tokens_slice": "embed_tokens_slice.npy",
    "embed_tokens_per_layer_slice": "embed_tokens_per_layer_slice.npy",
    "per_layer_model_projection": "per_layer_model_projection.npy",
    "per_layer_projection_norm": "per_layer_projection_norm.npy",
    "ple_reference_final": "ple_reference_final.npy",
}

EXPECTED = {
    "ple_input_ids": ((1, 4), "int64"),
    "embed_tokens_slice": ((4, 1536), "float32"),
    "embed_tokens_per_layer_slice": ((4, 8960), "float32"),
    "per_layer_model_projection": ((8960, 1536), "float32"),
    "per_layer_projection_norm": ((256,), "float32"),
    "ple_reference_final": ((1, 4, 35, 256), "float32"),
}


def main():
    tensors = {}
    print("=== Export PLE fixture to safetensors ===")
    for name, filename in FILES.items():
        path = FIX / filename
        if not path.exists():
            raise SystemExit(f"BLOCK: missing {path}")
        arr = np.load(path)
        expected_shape, expected_dtype = EXPECTED[name]
        if arr.shape != expected_shape:
            raise SystemExit(
                f"BLOCK: {name} shape mismatch: got {arr.shape}, expected {expected_shape}"
            )
        if str(arr.dtype) != expected_dtype:
            raise SystemExit(
                f"BLOCK: {name} dtype mismatch: got {arr.dtype}, expected {expected_dtype}"
            )
        tensors[name] = arr
        print(f"  {name:32s} shape={arr.shape} dtype={arr.dtype}")

    save_file(tensors, str(OUT))
    print(f"\nWrote: {OUT}")
    print(f"Size : {OUT.stat().st_size / (1024**2):.2f} MiB")

    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text())
    else:
        manifest = {}
    manifest["primary_fixture"] = {
        "format": "safetensors",
        "path": "fixtures/ple_fixture.safetensors",
        "tensor_names": list(FILES.keys()),
        "note": "Primary fixture for P4.4 ZML PLE-only runner. NPY files remain source/export intermediates.",
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"Updated manifest: {MANIFEST}")

    print("\nPASS: safetensors fixture export complete")


if __name__ == "__main__":
    main()
