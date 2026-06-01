"""P5.4 — PyTorch oracle + fixture : embedding lookup (gather) + scale.

Entrée du modèle : `Gemma4TextScaledWordEmbedding` = `embed_tokens(input_ids) * embed_scale`,
embed_scale = sqrt(hidden_size) = sqrt(1536) (buffer non-persistant). P4.4 Gate B avait déjà
validé le SCALE bit-exact sur un slice pré-gathered ; P5.4 valide en plus le GATHER en ZML
(`weight.gather(.{.voc = tokens})`, cf qwen3_5/lfm2/zml.nn.embed).

La table complète embed_tokens.weight [262144,1536] = 1.6 GB (impraticable en fixture). On teste
le gather sur un SLICE de vocab (4096 lignes) + des token ids dans [0,4096). Le gather pleine
table est mécaniquement identique (juste plus de lignes).

Pipeline oracle :
    embed_slice = embed_tokens.weight[:VOCAB_SLICE]         [4096,1536]
    gathered    = embed_slice[input_ids]                    [1,4,1536]
    embed_out   = gathered * sqrt(1536)

Fixture (3 tenseurs) : embed_slice, input_ids (i32) (inputs), embed_out (oracle).
Interdits : couches décodeur, PLE auxiliaire, lm_head.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights" / "model.safetensors"
OUT_FIXTURE = ROOT / "fixtures" / "p5_4_embed.safetensors"
OUT_MANIFEST = ROOT / "fixtures" / "p5_4_embed_manifest.json"

HIDDEN = 1536
VOCAB_SLICE = 4096
SEQ_LEN = 4
EMBED_SCALE = math.sqrt(HIDDEN)   # sqrt(1536) ~ 39.191835884
EMBED_KEY = "model.language_model.embed_tokens.weight"
INPUT_IDS = [2, 105, 2048, 4095]  # dans [0, VOCAB_SLICE)


def main() -> None:
    assert WEIGHTS.exists(), f"missing weights {WEIGHTS}"
    with safe_open(str(WEIGHTS), framework="pt") as s:
        assert EMBED_KEY in s.keys(), f"missing {EMBED_KEY}"
        full = s.get_tensor(EMBED_KEY).to(torch.float32)
    embed_slice = full[:VOCAB_SLICE].contiguous()           # [4096,1536]
    assert tuple(embed_slice.shape) == (VOCAB_SLICE, HIDDEN)

    input_ids = torch.tensor(INPUT_IDS, dtype=torch.int32).view(1, SEQ_LEN)   # [1,4]
    gathered = embed_slice[input_ids.long()]                # [1,4,1536]
    embed_out = (gathered * EMBED_SCALE).contiguous()
    assert tuple(embed_out.shape) == (1, SEQ_LEN, HIDDEN)

    print("=" * 70)
    print(f"P5.4 — PyTorch oracle embedding gather + scale (vocab slice {VOCAB_SLICE})")
    print("=" * 70)
    print(f"embed_scale = sqrt({HIDDEN}) = {EMBED_SCALE:.9f} (P4.4 Gate B bit-exact)")
    print(f"input_ids = {INPUT_IDS}")
    # Sanity : gather == sélection de lignes.
    for i, tid in enumerate(INPUT_IDS):
        d = (gathered[0, i] - embed_slice[tid]).abs().max().item()
        assert d == 0.0, f"gather mismatch id {tid}"
    print("Sanity gather: gathered[i] == embed_slice[input_ids[i]] bit-exact OK")
    print()
    print("Fixed points (embed_out):")
    for q in [0, 3]:
        vals = embed_out[0, q, :8].tolist()
        print(f"  embed_out[0,{q},:8] = [{', '.join(f'{v:.10f}' for v in vals)}]")
    print()
    print(f"Stats embed_out: mean={embed_out.mean():.4e} std={embed_out.std():.4e} "
          f"min={embed_out.min():.4e} max={embed_out.max():.4e}")

    tensors = {
        "embed_slice": embed_slice,
        "input_ids": input_ids,
        "embed_out": embed_out,
    }
    OUT_FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(OUT_FIXTURE))
    print(f"\nwrote {OUT_FIXTURE}  ({sum(t.numel()*t.element_size() for t in tensors.values())} bytes)")

    manifest = {
        "source": "P5.4 PyTorch oracle embedding gather + scale (vocab slice 4096)",
        "spec_refs": ["Gemma4TextScaledWordEmbedding : embed_tokens(ids) * sqrt(hidden_size)",
                      "P4.4 Gate B : scale sqrt(1536) bit-exact ; ici on ajoute le gather ZML"],
        "config": {"hidden": HIDDEN, "vocab_slice": VOCAB_SLICE, "seq_len": SEQ_LEN,
                   "embed_scale": EMBED_SCALE, "input_ids": INPUT_IDS},
        "tensors": {n: {"shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", "")} for n, t in tensors.items()},
        "zml_pipeline_hint": [
            "embed_slice = load {.voc=4096,.d=1536} ; input_ids = load {.b=1,.s=4} i32",
            "gathered = embed_slice.gather(.{.voc = input_ids}, .{})   {.b,.s,.d}",
            "embed_out = gathered.scale(sqrt(1536))",
            "compare vs embed_out oracle [1,4,1536], tol 1e-4 (attendu bit-exact : gather+scale exacts)",
        ],
        "expected_zml_max_abs_le": 1.0e-4,
        "interdits_p5_4": ["couches décodeur", "PLE auxiliaire", "lm_head"],
        "note": "table complète 262144 impraticable en fixture ; gather pleine table = mécaniquement identique.",
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {OUT_MANIFEST}")
    print("\nP5.4 oracle + fixture export PASS.")


if __name__ == "__main__":
    main()
