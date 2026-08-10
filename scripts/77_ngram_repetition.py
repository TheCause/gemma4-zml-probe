#!/usr/bin/env python3
"""RP7 (requalifiée) — métrique de répétition sur une trajectoire d'ids.

Décision Régis du 10 août 2026 (D4) : RP7 est une **mesure publiée, sans PASS/FAIL**. Le
symptôme de « récitation » n'a jamais été reproduit sur ce modèle (3 témoins greedy sans
boucle, dette D5) ; un gate binaire y serait vacue. On publie donc deux chiffres comparables,
penalty ON et OFF, et on laisse le lecteur juger.

Métrique : **longueur du plus long n-gramme répété** (plus longue sous-séquence apparaissant
au moins deux fois), plus le nombre de bigrammes/trigrammes distincts répétés. Aucune de ces
grandeurs n'est un seuil : elles décrivent, elles ne tranchent pas.

Entrée : les `.safetensors` produits par `--out-ids` du runner (clé `ids`) **ou** les fixtures
oracle de `69_u8_gen_oracle.py` (clé `fed`) — ce qui rend la mesure comparative exécutable
côté HF sans GPU.
"""
import json
import struct
import sys


def read_ids(path: str) -> list[int]:
    """Lecture safetensors en Python pur — pas de dépendance torch pour lire un vecteur i32."""
    with open(path, "rb") as f:
        blob = f.read()
    n = struct.unpack("<Q", blob[:8])[0]
    header = json.loads(blob[8 : 8 + n])
    # `ids` = sortie --out-ids du runner ; `fed` = trajectoire d'une fixture oracle HF.
    key = next((k for k in ("ids", "fed") if k in header), None)
    if key is None:
        raise SystemExit(f"{path} : ni 'ids' ni 'fed' (clés : {sorted(header)})")
    start, end = header[key]["data_offsets"]
    raw = blob[8 + n + start : 8 + n + end]
    return list(struct.unpack("<%di" % (len(raw) // 4), raw))


def longest_repeated(seq: list[int]) -> tuple[int, int]:
    """(longueur du plus long n-gramme répété, position de sa 1re occurrence).

    O(n²) assumé : n = 200. Un suffix automaton serait plus rapide et moins relisible.
    """
    n = len(seq)
    best_len, best_pos = 0, -1
    for i in range(n):
        for j in range(i + 1, n):
            k = 0
            while j + k < n and seq[i + k] == seq[j + k]:
                k += 1
                if i + k >= j:  # occurrences non chevauchantes : on s'arrête au contact
                    break
            if k > best_len:
                best_len, best_pos = k, i
    return best_len, best_pos


def repeated_ngrams(seq: list[int], size: int) -> int:
    seen, rep = set(), set()
    for i in range(len(seq) - size + 1):
        g = tuple(seq[i : i + size])
        if g in seen:
            rep.add(g)
        seen.add(g)
    return len(rep)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: 77_ngram_repetition.py <ids.safetensors> [<ids.safetensors> ...]")
    for path in sys.argv[1:]:
        ids = read_ids(path)
        tail = ids[-200:]
        ln, pos = longest_repeated(tail)
        print(
            f"{path} : n={len(ids)} (fenêtre {len(tail)}) "
            f"plus_long_ngramme_répété={ln} @pos={pos} "
            f"bigrammes_répétés={repeated_ngrams(tail, 2)} "
            f"trigrammes_répétés={repeated_ngrams(tail, 3)} "
            f"ids_distincts={len(set(tail))}/{len(tail)}"
        )


if __name__ == "__main__":
    main()
