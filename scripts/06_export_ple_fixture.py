"""P4 prep — Export d'un fixture PLE minimal pour le portage ZML.

But : produire un jeu de tenseurs autonomes (.npy) que le mini-module ZML
peut charger sans depender du safetensors HF ni du vocabulaire complet.

Sortie dans ./fixtures/ :
    ple_input_ids.npy                    [1, 4]      int64
    embed_tokens_slice.npy               [4, 1536]   fp32   (slice sur les 4 tokens)
    embed_tokens_per_layer_slice.npy     [4, 8960]   fp32   (slice sur les 4 tokens)
    per_layer_model_projection.npy       [8960, 1536] fp32  (complet)
    per_layer_projection_norm.npy        [256]       fp32   (complet)
    ple_reference_final.npy              [1, 4, 35, 256] fp32  (verite algorithmique P3)

Cible totale ~55 MB en fp32 (dominee par la projection 8960x1536).
"""
import math
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import safe_open

WEIGHTS = Path("./weights/model.safetensors")
FIXTURE_DIR = Path("./fixtures")
FIXTURE_DIR.mkdir(exist_ok=True)

# Input figé (decode = 'ZML test prompt')
input_ids = torch.tensor([[236953, 3620, 1594, 11172]], dtype=torch.long)

# Invariants Gemma4 E2B
L, H, D = 35, 1536, 256
EPS = 1e-6


def find_key(keys, suffix):
    matches = [k for k in keys if k.endswith(suffix)]
    assert len(matches) == 1, f"key {suffix}: {matches}"
    return matches[0]


def rmsnorm(x, weight, eps=EPS):
    var = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    y = x.to(torch.float32) * torch.rsqrt(var + eps)
    return y * weight.to(torch.float32)


def main():
    if not WEIGHTS.exists():
        raise SystemExit(f"BLOCK: poids absents : {WEIGHTS}")

    print(f"=== Export PLE fixture -> {FIXTURE_DIR} ===\n")

    with safe_open(str(WEIGHTS), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        embed_w     = f.get_tensor(find_key(keys, "embed_tokens.weight"))
        ple_embed_w = f.get_tensor(find_key(keys, "embed_tokens_per_layer.weight"))
        proj_w      = f.get_tensor(find_key(keys, "per_layer_model_projection.weight"))
        norm_w      = f.get_tensor(find_key(keys, "per_layer_projection_norm.weight"))

    # 1. Slices vocab restreintes aux 4 tokens
    ids_flat        = input_ids.view(-1)               # [4]
    embed_slice     = embed_w[ids_flat]                # [4, 1536]
    ple_embed_slice = ple_embed_w[ids_flat]            # [4, 8960]

    # 2. Recalcul PLE_FINAL_BLOCK via l'algo P3 valide (fp32 verite math)
    inputs_embeds   = embed_slice.to(torch.float32) * math.sqrt(H)              # [4, 1536]
    token_identity  = (ple_embed_slice.to(torch.float32) * math.sqrt(D)).view(4, L, D)
    context_proj    = inputs_embeds @ proj_w.to(torch.float32).T                # [4, 8960]
    context_scaled  = context_proj * (1.0 / math.sqrt(H))
    context_reshape = context_scaled.view(4, L, D)
    context_norm    = rmsnorm(context_reshape, norm_w)
    ple_final       = (token_identity + context_norm) * (1.0 / math.sqrt(2.0))  # [4, L, D]
    ple_final_batch = ple_final.view(1, 4, L, D)

    # 3. Sauvegardes .npy (fp32 portable : pas de bf16 natif en numpy)
    fixtures = {
        "ple_input_ids":                 input_ids.cpu().numpy().astype(np.int64),
        "embed_tokens_slice":            embed_slice.to(torch.float32).cpu().numpy(),
        "embed_tokens_per_layer_slice":  ple_embed_slice.to(torch.float32).cpu().numpy(),
        "per_layer_model_projection":    proj_w.to(torch.float32).cpu().numpy(),
        "per_layer_projection_norm":     norm_w.to(torch.float32).cpu().numpy(),
        "ple_reference_final":           ple_final_batch.cpu().numpy(),
    }

    total = 0
    for name, arr in fixtures.items():
        path = FIXTURE_DIR / f"{name}.npy"
        np.save(path, arr)
        size = path.stat().st_size
        total += size
        print(f"  {name:32s} shape={str(tuple(arr.shape)):20s} dtype={str(arr.dtype):8s} {size/1024:8.1f} KB")

    print(f"\n  TOTAL fixture : {total/1024/1024:.2f} MB")

    # 4. Verification immediate vs P2 fixed point (PLE_FINAL_BLOCK[0,0,0,:4])
    ref_0004 = np.array([-0.012451171875, 0.087890625, -1.515625, 0.080078125], dtype=np.float32)
    ours_0004 = ple_final_batch[0, 0, 0, :4].cpu().numpy()
    diff = np.abs(ours_0004 - ref_0004)
    print(f"\nVerif PLE_FINAL_BLOCK[0,0,0,:4] (fixed point P2) :")
    print(f"  ours = {ours_0004.tolist()}")
    print(f"  ref  = {ref_0004.tolist()}")
    print(f"  max_abs = {diff.max():.6e}   mean_abs = {diff.mean():.6e}")
    if diff.max() > 1e-2:
        raise SystemExit("BLOCK: fixed point divergence vs P2 (algo P3 casse ?)")
    print("\nPASS: fixture PLE exporte, fixed point aligne avec la reference P2")


if __name__ == "__main__":
    main()
