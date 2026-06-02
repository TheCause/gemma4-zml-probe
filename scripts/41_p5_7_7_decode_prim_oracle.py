"""P5.7.7 — Oracle des PRIMITIVES decode isolées (scatterSlices + rope(pos)).

Ferme la dette de validation notée par la revue adversariale de decode-1 : les 2 primitives ZML neuves
(scatterSlices append, zml.nn.rope avec pos_idx) avaient été introduites ensemble dans un composite.
On les isole, chacune testée séparément, AVANT decode-3.

Oracle déterministe (pas de modèle) :
  - scatterSlices : copie host (numpy) = vérité triviale de l'op. cache {1,1,5,4} de valeurs connues,
    update {1,1,1,4}, attendus pour scatter à pos=4 ET pos=2 (ciblage dynamique + override + passthrough).
  - rope(pos) : on exporte juste l'entrée x {1,5,1,8} ; le runner fait l'auto-cohérence
    rope(x, arange)[4] == rope(x[4:5], pos=[4]) (prouve que pos est utilisé ; ≡ HF via P5.2.C.3).

Fixture : fixtures/p5_7_7_decode_prim.safetensors
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
OUT_FIXTURE = ROOT / "fixtures" / "p5_7_7_decode_prim.safetensors"
OUT_MANIFEST = ROOT / "fixtures" / "p5_7_7_decode_prim_manifest.json"

K, HD4 = 5, 4   # scatter : cache {1,1,5,4}
S, HD8 = 5, 8   # rope    : x {1,5,1,8}


def main() -> None:
    # ---- scatterSlices : cache + update + attendus (copie déterministe) ----
    k_idx = np.arange(K).reshape(1, 1, K, 1)
    h_idx = np.arange(HD4).reshape(1, 1, 1, HD4)
    cache0 = (10.0 * k_idx + h_idx).astype(np.float32)            # {1,1,5,4} val = 10*k + hd
    update = (900.0 + np.arange(HD4)).astype(np.float32).reshape(1, 1, 1, HD4)
    exp_pos4 = cache0.copy(); exp_pos4[:, :, 4, :] = update[:, :, 0, :]
    exp_pos2 = cache0.copy(); exp_pos2[:, :, 2, :] = update[:, :, 0, :]

    # ---- rope : entrée x déterministe non triviale ----
    x = (np.arange(S * HD8).astype(np.float32) * 0.1).reshape(1, S, 1, HD8)

    tensors = {
        "scat_cache": torch.from_numpy(cache0).contiguous(),
        "scat_update": torch.from_numpy(update).contiguous(),
        "scat_exp_pos4": torch.from_numpy(exp_pos4).contiguous(),
        "scat_exp_pos2": torch.from_numpy(exp_pos2).contiguous(),
        "scat_pos4": torch.tensor([4], dtype=torch.int32),
        "scat_pos2": torch.tensor([2], dtype=torch.int32),
        "rope_x": torch.from_numpy(x).contiguous(),
        "rope_pos4": torch.tensor([4], dtype=torch.int32),
    }
    OUT_FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(OUT_FIXTURE))
    print("wrote", OUT_FIXTURE)
    print("  scat: cache[*,*,4,:] avant =", cache0[0, 0, 4].tolist(), "-> apres pos4 =", exp_pos4[0, 0, 4].tolist())
    print("  scat: cache[*,*,2,:] avant =", cache0[0, 0, 2].tolist(), "-> apres pos2 =", exp_pos2[0, 0, 2].tolist())
    print("  scat: lignes non touchees conservees (passthrough) — ex [0]:", exp_pos4[0, 0, 0].tolist())

    manifest = {
        "source": "P5.7.7 primitives decode isolées (scatterSlices + rope pos)",
        "purpose": "ferme la dette d'isolation notée par la revue adversariale de decode-1",
        "scatter": "cache {1,1,5,4} val=10k+hd ; update=900+hd ; attendus pos4/pos2 (copie numpy = vérité)",
        "rope": "auto-cohérence rope(x,arange)[4]==rope(x[4],pos=[4]) (≡ HF via P5.2.C.3)",
        "tensors": {n: {"shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", "")} for n, t in tensors.items()},
        "pass_threshold": {"scatter": "max_abs == 0 (copie exacte)", "rope": "max_abs == 0 (même calcul, prouve pos utilisé)"},
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print("wrote", OUT_MANIFEST, "\nP5.7.7 decode-prim oracle OK.")


if __name__ == "__main__":
    main()
