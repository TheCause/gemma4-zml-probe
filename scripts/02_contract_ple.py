from collections import Counter
from math import prod
from huggingface_hub import get_safetensors_metadata
import huggingface_hub
import sys

repo = "google/gemma-4-E2B-it"

print("=== P-1 / PLE CONTRACT CHECK ===")
print("huggingface_hub:", huggingface_hub.__version__)
print("repo:", repo)

try:
    meta = get_safetensors_metadata(repo)
except Exception as e:
    print("\nERROR while reading safetensors metadata:")
    print(type(e).__name__, e)
    print("\nPossible causes:")
    print("- Hugging Face auth missing")
    print("- Gemma license not accepted")
    print("- huggingface_hub too old")
    print("- network/proxy issue")
    sys.exit(1)

print("\n--- repo metadata ---")
print("sharded:", meta.sharded)
print("files:", list(meta.files_metadata.keys()))
print("parameter_count:", getattr(meta, "parameter_count", "n/a"))

keys = sorted(meta.weight_map.keys())

print("\n--- key prefix summary ---")
prefixes = Counter(".".join(k.split(".")[:3]) for k in keys)
for prefix, count in prefixes.most_common(30):
    print(f"{count:4d}  {prefix}")

exact_terms = (
    "embed_tokens_per_layer",
    "per_layer_model_projection",
    "per_layer_projection_norm",
)

print("\n--- exact PLE keys ---")
exact_keys = [k for k in keys if any(term in k for term in exact_terms)]
print("count:", len(exact_keys))
for k in exact_keys:
    print(k)

print("\n--- broader per_layer keys ---")
broad_keys = [k for k in keys if "per_layer" in k]
print("count:", len(broad_keys))
for k in broad_keys:
    print(k)

print("\n--- exact PLE tensor shapes + param counts ---")
total_ple_params = 0
found = {}

for filename, fmeta in meta.files_metadata.items():
    for name, info in sorted(fmeta.tensors.items()):
        if any(term in name for term in exact_terms):
            n_params = prod(info.shape)
            total_ple_params += n_params
            bytes_bf16 = n_params * 2

            print(
                filename,
                name,
                info.dtype,
                info.shape,
                f"params={n_params:,}",
                f"bf16_bytes≈{bytes_bf16:,}",
            )

            if "embed_tokens_per_layer" in name:
                found[name] = list(info.shape)

print("\n--- exact PLE total ---")
print("params:", f"{total_ple_params:,}")
print("bf16_gib≈", f"{total_ple_params * 2 / (1024**3):.2f}")
print("bf16_gb≈", f"{total_ple_params * 2 / 1e9:.2f}")

print("\n--- contract check ---")
expected_shape = [262144, 8960]

if not found:
    print("BLOCK: no embed_tokens_per_layer tensor found")
    sys.exit(1)

contract_failed = False
for name, shape in found.items():
    if shape != expected_shape:
        print(f"BLOCK: unexpected shape for {name}: {shape}")
        contract_failed = True
    else:
        print(f"OK: {name} shape = {shape}")

if contract_failed:
    sys.exit(1)

print("PASS: PLE embedding shape matches expected [262144, 8960]")
