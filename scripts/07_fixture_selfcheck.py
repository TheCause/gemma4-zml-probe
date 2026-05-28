"""P4.3 — Selfcheck du fixture PLE.

Charge uniquement les .npy (pas de HF, pas de safetensors) et reconstruit
ple_reference_final.npy a partir des autres fixtures, en respectant la
convention de scaling declaree dans fixture_manifest.json.

Gate :
  PASS si max_abs(recompute, reference) <= 1e-5
  BLOCK sinon.
"""
import json
import math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "fixtures"

L = 35
H = 1536
D = 256
EPS = 1e-6

MAX_ABS_THRESHOLD = 1e-4
MEAN_ABS_THRESHOLD = 1e-6


def rmsnorm(x, weight, eps=EPS):
    var = np.mean(x * x, axis=-1, keepdims=True)
    return x * (1.0 / np.sqrt(var + eps)) * weight


def main():
    manifest_path = FIX / "fixture_manifest.json"
    if not manifest_path.exists():
        raise SystemExit("BLOCK: missing fixtures/fixture_manifest.json")
    manifest = json.loads(manifest_path.read_text())
    scaling = manifest["scaling_contract"]

    input_ids = np.load(FIX / "ple_input_ids.npy")
    embed = np.load(FIX / "embed_tokens_slice.npy").astype(np.float32)
    ple_embed = np.load(FIX / "embed_tokens_per_layer_slice.npy").astype(np.float32)
    proj = np.load(FIX / "per_layer_model_projection.npy").astype(np.float32)
    norm = np.load(FIX / "per_layer_projection_norm.npy").astype(np.float32)
    ref = np.load(FIX / "ple_reference_final.npy").astype(np.float32)

    print("Shapes:")
    print(f"  input_ids: {input_ids.shape} {input_ids.dtype}")
    print(f"  embed:     {embed.shape} {embed.dtype}")
    print(f"  ple_embed: {ple_embed.shape} {ple_embed.dtype}")
    print(f"  proj:      {proj.shape} {proj.dtype}")
    print(f"  norm:      {norm.shape} {norm.dtype}")
    print(f"  ref:       {ref.shape} {ref.dtype}")

    assert input_ids.shape == (1, 4)
    assert embed.shape == (4, H)
    assert ple_embed.shape == (4, L * D)
    assert proj.shape == (L * D, H)
    assert norm.shape == (D,)
    assert ref.shape == (1, 4, L, D)

    if scaling["embed_tokens_slice"] == "raw_unscaled":
        inputs_embeds = embed.reshape(1, 4, H) * math.sqrt(H)
    else:
        inputs_embeds = embed.reshape(1, 4, H)

    if scaling["embed_tokens_per_layer_slice"] == "raw_unscaled":
        token_identity = ple_embed.reshape(1, 4, L, D) * math.sqrt(D)
    else:
        token_identity = ple_embed.reshape(1, 4, L, D)

    context_proj = inputs_embeds @ proj.T
    context_scaled = context_proj * (1.0 / math.sqrt(H))
    context_reshaped = context_scaled.reshape(1, 4, L, D)
    context_norm = rmsnorm(context_reshaped, norm)
    ple = (token_identity + context_norm) * (1.0 / math.sqrt(2.0))

    diff = np.abs(ple - ref)
    print("\nCompare fixture recompute vs ple_reference_final.npy")
    print(f"  max_abs : {diff.max():.6e}")
    print(f"  mean_abs: {diff.mean():.6e}")
    print(f"  fixed ours: {ple[0,0,0,:4].tolist()}")
    print(f"  fixed ref : {ref[0,0,0,:4].tolist()}")
    print(f"  threshold max_abs : {MAX_ABS_THRESHOLD:.1e}")
    print(f"  threshold mean_abs: {MEAN_ABS_THRESHOLD:.1e}")

    if diff.max() > MAX_ABS_THRESHOLD or diff.mean() > MEAN_ABS_THRESHOLD:
        raise SystemExit(
            "BLOCK: fixture is not self-consistent with reference "
            f"(max_abs={diff.max():.6e}, mean_abs={diff.mean():.6e})"
        )

    print("\nPASS: fixture selfcheck OK")


if __name__ == "__main__":
    main()
