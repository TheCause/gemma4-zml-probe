"""L1b — Oracle ring-buffer 512 + masque circulaire (fixture, SANS ré-exécuter HF).

Construit la fixture L1b (`gen_long_ring.safetensors`) en RÉUTILISANT la fixture L0 (`gen_long.safetensors`)
pour tout ce qui ne dépend pas du token (expected, fed, embeds, embptls, cos_full, sin_full, positions,
cache_fl_*, cache0_full) et en RECONSTRUISANT seulement ce qui change en ring 512 :
  - `masks_sliding` : CIRCULAIRE, shape {N,1,1,1,KMAX_SLIDING=512} (au lieu de .k=L_MAX linéaire).
  - `cache_sl_k/v`  : re-packed à .k=512 (prefill 0..3 aux slots 0..3, reste 0).

La séquence greedy HF (expected/fed) est IDENTIQUE à L0/L1a : le ring 512 est un encodage mémoire du
MÊME attention (les 512 dernières positions visibles), donc tokens bit-identiques. L1b valide que le
moteur ZML reproduit HF avec le vrai ring + masque circulaire (gate L1b, cf GENERATION_LONGUE_PLAN).

Émet AUSSI `gen_long_ring_naive.safetensors` (mêmes tensors sauf `masks_sliding` NAIVE = bande non
remappée sur le ring) pour le CONTRE-TEST de non-vacuité L1b (PLAN step 277) : le runner L1b sur cette
fixture NAIVE doit DIVERGER à partir de p≈512 (la bande dépasse le ring → slots masqués à tort).

CLI : python3 scripts/47_gen_long_ring_oracle.py   (3090 ; lit gen_long.safetensors, n'a PAS besoin de GPU/HF)
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
L0 = ROOT / "gen_long.safetensors"
OUT_RING = ROOT / "gen_long_ring.safetensors"
OUT_NAIVE = ROOT / "gen_long_ring_naive.safetensors"
MANIFEST = ROOT / "gen_long_ring_manifest.json"

SEQ_LEN = 4
L_MAX = 1024
SLIDING_WINDOW = 512
KMAX_SLIDING = 512  # ring
N_DECODE = L_MAX - SEQ_LEN  # 1020
MIN = torch.finfo(torch.float32).min
HD_S = 256


def j_max(s: int, p: int) -> int | None:
    """Position la plus récente écrite au slot s du ring (mod 512) pour la position courante p, ou None."""
    if s > p:  # slot jamais écrit (aucune position ≤ p congrue à s mod 512 car s≤511 et s>p)
        return None
    k = (p - s) // 512
    return s + 512 * k


def circular_mask(p: int) -> torch.Tensor:
    """Masque circulaire {KMAX_SLIDING} : slot s visible ssa la position qu'il contient est dans la bande."""
    m = torch.full((KMAX_SLIDING,), MIN, dtype=torch.float32)
    lo = max(0, p - (SLIDING_WINDOW - 1))
    for s in range(KMAX_SLIDING):
        j = j_max(s, p)
        if j is not None and lo <= j <= p:
            m[s] = 0.0
    return m


def naive_mask(p: int) -> torch.Tensor:
    """Contre-test : bande non-remappée (slot index ≡ position). Diverge pour p≥512 (bande déborde le ring)."""
    m = torch.full((KMAX_SLIDING,), MIN, dtype=torch.float32)
    lo = max(0, p - (SLIDING_WINDOW - 1))
    hi = min(p, KMAX_SLIDING - 1)
    for s in range(KMAX_SLIDING):
        if lo <= s <= hi:
            m[s] = 0.0
    return m


def main() -> None:
    assert L0.exists(), f"L0 fixture manquante : {L0} (lancer scripts/46_gen_long_oracle.py d'abord)"
    print(f"Lecture L0 : {L0}")

    # Reuse tout sauf masks_sliding + cache_sl_k/v (reconstruits).
    reuse_names = [
        "embeds", "embptls", "cos_full", "sin_full", "positions",
        "masks_full", "cache_fl_k", "cache_fl_v", "expected", "fed",
    ]
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(L0), framework="pt") as s:
        # cache_sl originals (linéaire .k=L_MAX) pour re-pack
        sl_k_lin = s.get_tensor("cache_sl_k")  # [n_slots,1,1,L_MAX,HD_S]
        sl_v_lin = s.get_tensor("cache_sl_v")
        for n in reuse_names:
            tensors[n] = s.get_tensor(n).clone()

    n = tensors["expected"].shape[0]
    assert n == N_DECODE, f"expected len {n} != {N_DECODE}"
    positions = tensors["positions"]
    print(f"N={n}, positions {positions[0].item()}..{positions[-1].item()}, KMAX_SLIDING={KMAX_SLIDING}")

    # masks_sliding circulaire (correct) + naive (contre-test), shape {N,1,1,1,KMAX_SLIDING}.
    masks_circ = torch.zeros(n, 1, 1, 1, KMAX_SLIDING, dtype=torch.float32)
    masks_naiv = torch.zeros(n, 1, 1, 1, KMAX_SLIDING, dtype=torch.float32)
    for k in range(n):
        p = int(positions[k].item())
        masks_circ[k, 0, 0, 0, :] = circular_mask(p)
        masks_naiv[k, 0, 0, 0, :] = naive_mask(p)

    # cache_sl re-packed ring .k=512 : prefill positions 0..SEQ_LEN-1 aux slots 0..3, reste 0.
    n_slots = sl_k_lin.shape[0]
    cache_sl_k = torch.zeros(n_slots, 1, 1, KMAX_SLIDING, HD_S, dtype=torch.float32)
    cache_sl_v = torch.zeros_like(cache_sl_k)
    cache_sl_k[:, :, :, :SEQ_LEN, :] = sl_k_lin[:, :, :, :SEQ_LEN, :].float()
    cache_sl_v[:, :, :, :SEQ_LEN, :] = sl_v_lin[:, :, :, :SEQ_LEN, :].float()

    # Vérif : pour p<512, circulaire == naive (sanity) ; pour p>=512, ils diffèrent.
    for k in (0, 5, 100):
        p = int(positions[k].item())
        assert torch.equal(masks_circ[k], masks_naiv[k]), f"p<{SLIDING_WINDOW} devraient coïncider (p={p})"
    k_512 = int((512 - SEQ_LEN))  # p=512
    assert not torch.equal(masks_circ[k_512], masks_naiv[k_512]), "p=512 : circulaire vs naive doivent différer"

    # === Fixture L1b (correcte) ===
    tensors_ring = dict(tensors)
    tensors_ring["masks_sliding"] = masks_circ.contiguous()
    tensors_ring["cache_sl_k"] = cache_sl_k.contiguous()
    tensors_ring["cache_sl_v"] = cache_sl_v.contiguous()
    for k, t in tensors_ring.items():
        assert not torch.isnan(t.float()).any(), f"NaN dans {k}"
    save_file(tensors_ring, str(OUT_RING))
    print("wrote", OUT_RING)

    # === Fixture contre-test (naive) : mêmes tensors sauf masks_sliding naive ===
    tensors_naive = dict(tensors)
    tensors_naive["masks_sliding"] = masks_naiv.contiguous()
    tensors_naive["cache_sl_k"] = cache_sl_k.contiguous()
    tensors_naive["cache_sl_v"] = cache_sl_v.contiguous()
    save_file(tensors_naive, str(OUT_NAIVE))
    print("wrote", OUT_NAIVE, "(contre-test : masque non-remappé)")

    manifest = {
        "source": "L1b oracle ring 512 + masque circulaire (reuse L0, rebuild masks_sliding + cache_sl ring)",
        "l0_fixture": str(L0.name), "seq_len": SEQ_LEN, "n_decode": N_DECODE, "l_max": L_MAX,
        "kmax_sliding": KMAX_SLIDING, "kmax_full": L_MAX, "sliding_window": SLIDING_WINDOW,
        "fixtures": {
            "gen_long_ring.safetensors": "L1b correct (masque circulaire) — runner attend PASS argmax==HF",
            "gen_long_ring_naive.safetensors": "contre-test non-vacuité (masque non-remappé) — runner attend DIVERGENCE ~p=512",
        },
        "tensors": {n_: {"shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", "")}
                    for n_, t in tensors_ring.items()},
        "pass_criterion_L1b": "argmax ZML[k] == expected[k] pour tout k (ring 512 + masque circulaire)",
        "counter_test_L1b": "runner sur gen_long_ring_naive.safetensors doit DIVERGER à partir de p≈512",
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print("wrote", MANIFEST, "\nL1b oracle ring OK.")


if __name__ == "__main__":
    main()
