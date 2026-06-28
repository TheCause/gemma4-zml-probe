#!/usr/bin/env python3
"""48 — Détokenisation + VALIDATION (round-trip) de la séquence générée.

Le banc valide le MODÈLE op-par-op (sortie == HuggingFace, en token ids). La détokenisation (token ids
-> texte) est déléguée au tokenizer HF (SentencePiece) — hors du périmètre op-par-op. Ce script la rend
REPRODUCTIBLE et, surtout, la VALIDE par round-trip, dans l'esprit gated du projet :

    ids  --tok.decode-->  texte  --tok.encode-->  ids'        gate : (ids' == ids)

Un round-trip EXACT prouve que le texte affiché représente FIDÈLEMENT les tokens générés (bijection sur
la séquence) — on ne fait pas confiance au tokenizer en boîte noire, on le vérifie. Affiche aussi le
texte lisible (skip_special_tokens) pour inspecter l'inférence.

CLI : python3 scripts/48_detokenize.py [fixture.safetensors] [--field fed|expected] [--max-chars N]
Prérequis : tokenizer google/gemma-4-E2B-it accessible (HF_HOME / cache), transformers.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/data/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from safetensors import safe_open
from transformers import AutoTokenizer

MODEL = "google/gemma-4-E2B-it"
ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description="Détokenise + valide (round-trip) une séquence de tokens.")
    ap.add_argument("fixture", nargs="?", default=str(ROOT / "gen_long.safetensors"))
    ap.add_argument("--field", default="fed", choices=["fed", "expected"],
                    help="tensor de tokens à décoder (défaut: fed = séquence feedée == générée)")
    ap.add_argument("--max-chars", type=int, default=2000, help="tronque le texte affiché")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL)
    with safe_open(args.fixture, framework="pt") as s:
        ids = s.get_tensor(args.field).tolist()

    # --- Détokenisation ---
    text_clean = tok.decode(ids, skip_special_tokens=True)
    text_full = tok.decode(ids, skip_special_tokens=False)

    # --- VALIDATION round-trip : re-tokeniser le texte doit redonner les mêmes ids ---
    reids = tok.encode(text_full, add_special_tokens=False)
    exact = reids == ids
    n = min(len(ids), len(reids))
    n_match = sum(1 for a, b in zip(ids, reids) if a == b)
    first_mismatch = next((i for i, (a, b) in enumerate(zip(ids, reids)) if a != b), None)

    print(f"=== Détokenisation : {args.fixture} (champ '{args.field}', {len(ids)} tokens) ===\n")
    print("--- TEXTE GÉNÉRÉ (lisible, skip_special_tokens) ---")
    print(text_clean.strip()[: args.max_chars])
    print("\n--- VALIDATION round-trip : tokenize(detokenize(ids)) == ids ---")
    print(f"len(ids)={len(ids)}  len(reids)={len(reids)}  match={n_match}/{n}")
    if exact:
        print("ROUND-TRIP PASS — détokenisation fidèle (bijection exacte sur la séquence).")
        return 0
    print(f"ROUND-TRIP FAIL — 1er mismatch index {first_mismatch}")
    if first_mismatch is not None:
        lo = max(0, first_mismatch - 2)
        hi = first_mismatch + 3
        print(f"  ids  [{lo}:{hi}] = {ids[lo:hi]}")
        print(f"  reids[{lo}:{hi}] = {reids[lo:hi]}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
