import math
import sys
import torch
import transformers
import numpy as np


def compute_checksums(tensor, name):
    """Profil numérique multicritère pour validation future ZML."""
    t = tensor.detach().cpu().to(torch.float32).numpy()

    print(f"\n=== VERIFICATION TENSEUR: {name} ===")
    print(f"  Shape       : {t.shape}")
    print(f"  Dtype       : {tensor.dtype}")
    print(f"  Min / Max   : {t.min():.6f} / {t.max():.6f}")
    print(f"  Mean / Std  : {t.mean():.6f} / {t.std():.6f}")
    print(f"  Sum (Gross) : {t.sum():.4f}")

    if len(t.shape) == 3:  # [B, S, H]
        print(f"  Fixed Points [0, 0, :4] : {t[0, 0, :4].tolist()}")
    elif len(t.shape) == 4:  # [B, S, L, D]
        print(f"  Fixed Points [0, 0, 0, :4] : {t[0, 0, 0, :4].tolist()}")
        print(f"  Fixed Points [0, -1, -1, :4]: {t[0, -1, -1, :4].tolist()}")


def find_ple_text_model(root):
    """Localise le sous-module texte par capacites, pas par nom fragile."""
    required = (
        "embed_tokens",
        "embed_tokens_per_layer",
        "per_layer_model_projection",
        "per_layer_projection_norm",
    )

    candidates = [root]
    candidates.extend(module for _, module in root.named_modules())

    for module in candidates:
        if all(hasattr(module, name) for name in required):
            return module

    raise RuntimeError(
        "Divergence topologique : aucun sous-module contenant le pipeline PLE complet.\n"
        f"Sous-modules explores : {[type(m).__name__ for m in candidates]}"
    )


def load_model_cpu(repo_id):
    """Charge le modele sur CPU avec fallback dtype/torch_dtype."""
    try:
        return transformers.AutoModelForCausalLM.from_pretrained(
            repo_id,
            dtype="auto",
            device_map={"": "cpu"},
        )
    except TypeError as e:
        print("\nFallback: cette version Transformers ne supporte peut-etre pas dtype=.")
        print("Erreur initiale:", repr(e))
        print("Nouvelle tentative avec torch_dtype='auto'.")
        return transformers.AutoModelForCausalLM.from_pretrained(
            repo_id,
            torch_dtype="auto",
            device_map={"": "cpu"},
        )


def main():
    repo_id = "google/gemma-4-E2B-it"

    print("[LOGS ENVIRONNEMENT DE REFERENCE]")
    print(f"  torch             : {torch.__version__}")
    print(f"  transformers      : {transformers.__version__}")
    print(f"  transformers file : {transformers.__file__}")
    print(f"  Cible             : {repo_id}")

    print("\nChargement de la configuration et du tokenizer...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(repo_id)

    print("\nChargement du modele sur CPU...")
    model = load_model_cpu(repo_id)
    model.eval()

    print("\nRecherche du sous-module texte avec pipeline PLE...")
    text_model = find_ple_text_model(model)

    print("\n[DIAGNOSTIC CAPACITES PLE]")
    print(f"  Classe cible detectee         : {type(text_model)}")
    print(f"  Type embed_tokens             : {type(text_model.embed_tokens)}")
    print(f"  Type embed_tokens_per_layer   : {type(text_model.embed_tokens_per_layer)}")
    print(f"  Type per_layer_model_projection: {type(text_model.per_layer_model_projection)}")
    print(f"  Type per_layer_projection_norm : {type(text_model.per_layer_projection_norm)}")
    print("  NOTE : si embed_tokens_per_layer est Gemma4TextScaledWordEmbedding,")
    print("         le scaling implicite *sqrt(hidden_size_per_layer_input) est deja inclus.")
    print("         Depuis les poids bruts ZML, il faudra le refaire explicitement.")

    cfg = text_model.config
    if hasattr(cfg, "text_config"):
        cfg = cfg.text_config

    print("\n[CONFIGURATION TEXTUELLE]")
    print(f"  Layers (L)      : {cfg.num_hidden_layers}")
    print(f"  Hidden Size (H) : {cfg.hidden_size}")
    print(f"  PLE Dimension D : {cfg.hidden_size_per_layer_input}")

    # Contrat mini-lot : B=1, S=4.
    text = "ZML test prompt"
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    input_ids = inputs["input_ids"][:, :4]

    if input_ids.shape[1] != 4:
        print(f"  Ajustement : tokenizer S={input_ids.shape[1]}.")
        print("  Forcage d'une sequence deterministe.")
        input_ids = torch.tensor([[2, 235285, 1234, 5678]], dtype=torch.long)

    assert input_ids.shape[1] == 4, f"Contrat S=4 viole: {input_ids.shape}"

    print(f"\n[INPUT IDS]")
    print(f"  Input IDs retenus : {input_ids.tolist()}")
    print(f"  Decodage des IDs  : '{tokenizer.decode(input_ids[0].tolist())}'")

    with torch.no_grad():
        # A. Flux principal
        inputs_embeds = text_model.embed_tokens(input_ids)
        compute_checksums(inputs_embeds, "inputs_embeds (Flux Principal)")

        # B. Flux auxiliaire token identity
        # IMPORTANT : le module Transformers peut inclure un scaling interne.
        token_identity = text_model.embed_tokens_per_layer(input_ids)
        token_identity_reshaped = token_identity.view(
            input_ids.shape[0],
            input_ids.shape[1],
            cfg.num_hidden_layers,
            cfg.hidden_size_per_layer_input,
        )
        compute_checksums(
            token_identity_reshaped,
            "token_identity (Reshaped - scaling interne inclus)",
        )

        # C. Flux auxiliaire context-aware
        # Ordre critique :
        # projection -> scaling 1/sqrt(hidden_size) -> reshape -> RMSNorm
        context_proj = text_model.per_layer_model_projection(inputs_embeds)

        context_scaled = context_proj * (1.0 / math.sqrt(cfg.hidden_size))

        context_reshaped = context_scaled.view(
            input_ids.shape[0],
            input_ids.shape[1],
            cfg.num_hidden_layers,
            cfg.hidden_size_per_layer_input,
        )

        context_normalized = text_model.per_layer_projection_norm(context_reshaped)
        compute_checksums(
            context_normalized,
            "context_normalized (Scaled -> Reshaped -> RMSNorm)",
        )

        # D. Fusion finale PLE
        ple_final = (token_identity_reshaped + context_normalized) * (
            1.0 / math.sqrt(2.0)
        )

        compute_checksums(
            ple_final,
            f"PLE_FINAL_BLOCK [B=1, S=4, L={cfg.num_hidden_layers}, D={cfg.hidden_size_per_layer_input}]",
        )

        compute_checksums(
            ple_final[:, :, 0, :],
            "PLE_LAYER_0_INJECTION",
        )

        compute_checksums(
            ple_final[:, :, -1, :],
            "PLE_LAYER_FINAL_INJECTION",
        )

        # Sauvegarde des tenseurs de reference pour comparaison directe en P3.
        import os
        os.makedirs("logs", exist_ok=True)
        torch.save(
            {
                "input_ids": input_ids.detach().cpu(),
                "ple_final": ple_final.detach().cpu(),
                "token_identity": token_identity_reshaped.detach().cpu(),
                "context_normalized": context_normalized.detach().cpu(),
                "inputs_embeds": inputs_embeds.detach().cpu(),
                "config": {
                    "L": cfg.num_hidden_layers,
                    "H": cfg.hidden_size,
                    "D": cfg.hidden_size_per_layer_input,
                    "rms_norm_eps": getattr(cfg, "rms_norm_eps", None),
                },
            },
            "logs/03_ple_reference_tensors.pt",
        )
        print("\nsave: logs/03_ple_reference_tensors.pt")

    print("\nPASS: PLE reference extraction completed")


if __name__ == "__main__":
    main()
