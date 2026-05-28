"""P5.0 — Cartographie YOCO / Shared KV (lecture seule config).

Sans dépendance au modèle complet : AutoConfig lit depuis le cache HF
(ou télécharge la config seule). Pas d'allocation de poids.
"""
from transformers import AutoConfig


def main() -> None:
    repo_id = "google/gemma-4-E2B-it"
    cfg = AutoConfig.from_pretrained(repo_id)
    tc = cfg.text_config

    print("=== Gemma 4 E2B text_config / YOCO map ===")
    print(f"transformers config class : {type(cfg).__name__}")
    print(f"text_config class         : {type(tc).__name__}")
    print(f"num_hidden_layers         : {tc.num_hidden_layers}")
    print(f"num_kv_shared_layers      : {getattr(tc, 'num_kv_shared_layers', None)}")
    print(f"sliding_window            : {getattr(tc, 'sliding_window', None)}")
    print(f"num_attention_heads       : {tc.num_attention_heads}")
    print(f"num_key_value_heads       : {tc.num_key_value_heads}")
    print(f"head_dim                  : {getattr(tc, 'head_dim', None)}")
    print(f"hidden_size               : {tc.hidden_size}")
    print(f"hidden_size_per_layer_input: {getattr(tc, 'hidden_size_per_layer_input', None)}")
    print(f"final_logit_softcapping   : {getattr(tc, 'final_logit_softcapping', None)}")
    print()

    layer_types = list(tc.layer_types)
    full = [i for i, t in enumerate(layer_types) if t == "full_attention"]
    sliding = [i for i, t in enumerate(layer_types) if t == "sliding_attention"]
    print(f"layer_types length        : {len(layer_types)}")
    print(f"full_attention layers     : {full}")
    print(f"sliding_attention layers  : {sliding}")
    print(f"|full|                    : {len(full)}")
    print(f"|sliding|                 : {len(sliding)}")
    print()

    print("Layer table:")
    print("  idx  type")
    print("  ---  ----")
    for i, t in enumerate(layer_types):
        print(f"  {i:02d}   {t}")


if __name__ == "__main__":
    main()
