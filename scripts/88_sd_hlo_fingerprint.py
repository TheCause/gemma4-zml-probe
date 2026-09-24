#!/usr/bin/env python3
"""88 — Empreinte du module HLO pré-optimisation principal d'un dump XLA (C-SD-C).

Réutilise les neutralisations éprouvées de 53_g2_3_hlo_check.py (chemins de dump, --xla_dump_to,
id numérique éphémère du nom de module). Module principal = le plus gros *before_optimizations*.txt.
Refuse bruyamment : dump absent, aucun pré-opt, décodage corrompu, module principal ambigu.

Usage : python3 scripts/88_sd_hlo_fingerprint.py <dump_dir>  → imprime `sha256 <hex> <fichier>`
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("hlo53", HERE / "53_g2_3_hlo_check.py")
hlo53 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hlo53)


def main() -> int:
    if len(sys.argv) != 2:
        sys.exit("usage: 88_sd_hlo_fingerprint.py <dump_dir>")
    d = sys.argv[1]
    files = hlo53.collect_before_opt(d, "dump")          # exit 1 explicite si rien
    txt = [p for p in files if p.suffix == ".txt"]
    if not txt:
        sys.exit(f"[erreur] aucun pré-opt .txt dans {d}")
    main_mod, ambiguous = hlo53.pick_main(txt)
    if ambiguous:
        sys.exit(f"[erreur] module principal ambigu dans {d} (2e fichier ≥ 90 % du 1er) — trancher à la main")
    h, corrupted = hlo53.normalized_hash(main_mod, [str(Path(d).resolve()), d])
    if corrupted:
        sys.exit(f"[erreur] décodage UTF-8 corrompu : {main_mod}")
    print(f"sha256 {h} {main_mod.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
