"""P3 — Reproduction du PLE de gemma-4-E2B-it depuis les poids bruts safetensors,
sans aucun module haut niveau Transformers. Pont vers ZML.

Pipeline reproduit :
    input_ids
      -> raw embed_tokens lookup x sqrt(H)
      -> raw embed_tokens_per_layer lookup x sqrt(D)
      -> raw per_layer_model_projection (Linear, weight [out, in])
      -> scale 1/sqrt(H)
      -> reshape [B,S,L,D]
      -> raw RMSNorm sur D (formule : y = x * rsqrt(mean(x^2)+eps) * weight)
      -> fusion (token_identity + context_normalized) / sqrt(2)
"""
import math
from pathlib import Path
import torch
from safetensors.torch import safe_open

WEIGHTS = Path("./weights/model.safetensors")
REF_PT = Path("logs/03_ple_reference_tensors.pt")

# Input P2 fige (decode = 'ZML test prompt', tokenizer Gemma 4)
input_ids = torch.tensor([[236953, 3620, 1594, 11172]], dtype=torch.long)

# Invariants Gemma4 E2B (confirmes par 02_contract_ple.py)
L = 35           # num_hidden_layers
H = 1536         # hidden_size
D = 256          # hidden_size_per_layer_input
PACKED = L * D   # 8960
EPS = 1e-6       # rms_norm_eps (verifie dans config.json text_config)

# Valeurs de reference P2 (extraites de logs/03_ple_reference.log)
REF_PLE_SUM = -9.8641
REF_PLE_0004 = torch.tensor(
    [-0.012451171875, 0.087890625, -1.515625, 0.080078125],
    dtype=torch.float32,
)


def find_key(keys, suffix):
    matches = [k for k in keys if k.endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one key ending with {suffix}, got {matches}")
    return matches[0]


def rmsnorm(x, weight, eps=EPS):
    """RMSNorm Gemma 4 : x * rsqrt(mean(x^2)+eps) * weight  (PAS la variante (1+weight))."""
    var = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    y = x.to(torch.float32) * torch.rsqrt(var + eps)
    return y * weight.to(torch.float32)


def compare_tensor(name, ours, ref, strict_threshold=1e-3, bf16_threshold=0.16):
    """Comparaison BF16-aware : la ref est en bf16 (Transformers cast post-RMSNorm),
    on ne peut pas exiger 1e-3 vs fp32. On evalue aussi notre fp32 quantifie en bf16."""
    ours_f32 = ours.detach().cpu().to(torch.float32)
    ref_f32 = ref.detach().cpu().to(torch.float32)
    diff = (ours_f32 - ref_f32).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()

    ours_bf16 = ours_f32.to(torch.bfloat16).to(torch.float32)
    bf16_diff = (ours_bf16 - ref_f32).abs()
    bf16_max_abs = bf16_diff.max().item()
    bf16_mean_abs = bf16_diff.mean().item()

    print(f"\n=== COMPARE {name} ===")
    print(f"  fp32_vs_ref_bf16   max_abs  : {max_abs:.6e}   mean_abs : {mean_abs:.6e}")
    print(f"  bf16quant_vs_ref   max_abs  : {bf16_max_abs:.6e}   mean_abs : {bf16_mean_abs:.6e}")

    if max_abs <= strict_threshold:
        print("  -> PASS_STRICT")
        return "PASS_STRICT"
    if bf16_max_abs <= bf16_threshold:
        print(f"  -> PASS_BF16_AWARE (seuil ULP-friendly = {bf16_threshold})")
        return "PASS_BF16_AWARE"
    print("  -> BLOCK")
    return "BLOCK"


def main():
    if not WEIGHTS.exists():
        raise SystemExit(f"BLOCK: poids absents : {WEIGHTS}  (lancer `hf download ... --local-dir ./weights`)")

    with safe_open(str(WEIGHTS), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        k_embed     = find_key(keys, "embed_tokens.weight")
        k_ple_embed = find_key(keys, "embed_tokens_per_layer.weight")
        k_proj      = find_key(keys, "per_layer_model_projection.weight")
        k_norm      = find_key(keys, "per_layer_projection_norm.weight")
        print("Keys safetensors:")
        print("  embed_tokens              :", k_embed)
        print("  embed_tokens_per_layer    :", k_ple_embed)
        print("  per_layer_model_projection:", k_proj)
        print("  per_layer_projection_norm :", k_norm)
        embed_w     = f.get_tensor(k_embed)
        ple_embed_w = f.get_tensor(k_ple_embed)
        proj_w      = f.get_tensor(k_proj)
        norm_w      = f.get_tensor(k_norm)

    print("\nShapes / dtypes:")
    print(f"  embed_w     : {tuple(embed_w.shape)} {embed_w.dtype}")
    print(f"  ple_embed_w : {tuple(ple_embed_w.shape)} {ple_embed_w.dtype}")
    print(f"  proj_w      : {tuple(proj_w.shape)} {proj_w.dtype}")
    print(f"  norm_w      : {tuple(norm_w.shape)} {norm_w.dtype}")

    assert ple_embed_w.shape == (262144, PACKED), f"shape PLE inattendue: {ple_embed_w.shape}"
    assert proj_w.shape == (PACKED, H),           f"shape proj inattendue: {proj_w.shape}"
    assert norm_w.shape == (D,),                  f"shape norm inattendue: {norm_w.shape}"
    assert embed_w.shape[1] == H,                 f"embed_w second dim != H: {embed_w.shape}"

    # 1. Flux principal brut : lookup + scaling sqrt(H)
    inputs_embeds = embed_w[input_ids].to(torch.float32) * math.sqrt(H)
    # 2. Token identity PLE brut : lookup + scaling sqrt(D)
    token_identity = ple_embed_w[input_ids].to(torch.float32) * math.sqrt(D)
    token_identity = token_identity.view(1, 4, L, D)
    # 3. Projection context-aware : Linear avec weight [out, in] -> y = x @ W.T
    context_proj = torch.matmul(inputs_embeds, proj_w.to(torch.float32).T)
    # 4. Scaling officiel 1/sqrt(H)
    context_scaled = context_proj * (1.0 / math.sqrt(H))
    # 5. Reshape [B,S,L,D]
    context_reshaped = context_scaled.view(1, 4, L, D)
    # 6. RMSNorm brute sur D=256
    context_normalized = rmsnorm(context_reshaped, norm_w)
    # 7. Fusion PLE
    ple_final = (token_identity + context_normalized) * (1.0 / math.sqrt(2.0))

    # 8. Sortie brute + scalaires P2 (informatif)
    ple_sum = ple_final.sum().item()
    ple_0004 = ple_final[0, 0, 0, :4].detach().cpu()
    print("\nRaw PyTorch outputs:")
    print(f"  PLE_FINAL shape : {tuple(ple_final.shape)} {ple_final.dtype}")
    print(f"  PLE_FINAL sum   : {ple_sum:.4f}   (ref P2: {REF_PLE_SUM})")
    print(f"  PLE_FINAL[0,0,0,:4]: {ple_0004.tolist()}")
    print(f"  ref  [0,0,0,:4]    : {REF_PLE_0004.tolist()}")
    fixed_max = (ple_0004 - REF_PLE_0004).abs().max().item()
    print(f"  fixed_max vs P2 ref: {fixed_max:.6e}")

    # 9. Comparaison BF16-aware vs .pt de reference
    if not REF_PT.exists():
        raise SystemExit(f"BLOCK: .pt absent ({REF_PT}) — relancer 03 d'abord pour produire la reference tensorielle.")

    ref = torch.load(str(REF_PT), map_location="cpu", weights_only=True)
    results = {
        "inputs_embeds":      compare_tensor("inputs_embeds",      inputs_embeds,      ref["inputs_embeds"]),
        "token_identity":     compare_tensor("token_identity",     token_identity,     ref["token_identity"]),
        "context_normalized": compare_tensor("context_normalized", context_normalized, ref["context_normalized"]),
        "ple_final":          compare_tensor("ple_final",          ple_final,          ref["ple_final"]),
    }

    # 10. Verdict final
    print("\n=== VERDICT P3 ===")
    for k, v in results.items():
        print(f"  {k:22s} : {v}")

    # token_identity DOIT etre PASS (c'est l'invariant n.1 : scaling sqrt(D) = x16).
    if results["token_identity"] not in {"PASS_STRICT", "PASS_BF16_AWARE"}:
        print("\nSuspects token_identity :")
        print("  - scaling embed_tokens_per_layer oublie : x sqrt(256)=16")
        print("  - reshape [L,D] vs [D,L]")
        raise SystemExit("BLOCK: token_identity failed")

    if results["ple_final"] == "BLOCK":
        print("\nSuspects ple_final (avec token_identity OK) :")
        print("  - Linear transpose dans le mauvais sens")
        print("  - RMSNorm epsilon (config dit 1e-6)")
        print("  - RMSNorm formule (Gemma 4 fait * weight, PAS (1+weight))")
        print("  - scaling 1/sqrt(H) oublie")
        print("  - fusion /sqrt(2) oubliee")
        raise SystemExit("BLOCK: ple_final failed")

    print("\nPASS: raw PyTorch PLE reproduction is BF16-aware compatible with reference")


if __name__ == "__main__":
    main()
