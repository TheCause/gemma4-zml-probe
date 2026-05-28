"""P5.0 — Carte des poids attention par couche (lecture metadata safetensors).

Hypothèse YOCO : certaines couches manquent leurs propres k_proj / v_proj /
k_norm car elles réutilisent les K/V partagés. Le nombre de couches sans poids
KV doit matcher `num_kv_shared_layers`.
"""
from collections import defaultdict

from huggingface_hub import get_safetensors_metadata


INTERESTING = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "self_attn.q_norm",
    "self_attn.k_norm",
]


def main() -> None:
    repo_id = "google/gemma-4-E2B-it"
    meta = get_safetensors_metadata(repo_id)
    keys = sorted(meta.weight_map.keys())

    by_layer: dict[int, list[str]] = defaultdict(list)
    for k in keys:
        if ".layers." not in k:
            continue
        parts = k.split(".")
        try:
            idx = parts.index("layers")
            layer_id = int(parts[idx + 1])
        except (ValueError, IndexError):
            continue
        if any(term in k for term in INTERESTING):
            by_layer[layer_id].append(k)

    print("=== YOCO / Attention weight ownership map ===")
    print(f"repo: {repo_id}")
    print(f"total weight keys: {len(keys)}")
    print()

    print(
        "layer | q_proj k_proj v_proj o_proj q_norm k_norm | own_kv?"
    )
    print(
        "------+---------------------------------------------+--------"
    )

    own_kv_layers: list[int] = []
    no_kv_layers: list[int] = []

    for layer in sorted(by_layer):
        names = by_layer[layer]
        has = {term: any(term in x for x in names) for term in INTERESTING}
        own_kv = has["self_attn.k_proj"] and has["self_attn.v_proj"]
        owns_kn = has["self_attn.k_norm"]
        marker = "OWN" if (own_kv and owns_kn) else "shared/yoco"
        if own_kv and owns_kn:
            own_kv_layers.append(layer)
        else:
            no_kv_layers.append(layer)
        print(
            f"  {layer:02d}  |  {int(has['self_attn.q_proj'])}     "
            f" {int(has['self_attn.k_proj'])}     "
            f" {int(has['self_attn.v_proj'])}     "
            f" {int(has['self_attn.o_proj'])}     "
            f" {int(has['self_attn.q_norm'])}     "
            f" {int(has['self_attn.k_norm'])}    | {marker}"
        )

    print()
    print(f"Layers WITH own k_proj+v_proj+k_norm : {own_kv_layers}")
    print(f"  count = {len(own_kv_layers)}")
    print(f"Layers WITHOUT own KV (YOCO consumers): {no_kv_layers}")
    print(f"  count = {len(no_kv_layers)}")

    print()
    print("Cross-check with config:")
    print("  Expected num_kv_shared_layers = 20 (config text_config).")
    print(f"  Observed YOCO consumers      = {len(no_kv_layers)}")
    print(
        "  Match expected = "
        f"{'YES' if len(no_kv_layers) == 20 else 'NO (investigate)'}"
    )


if __name__ == "__main__":
    main()
