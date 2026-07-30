#!/usr/bin/env python3
"""D10 (C8/AL-RSS) : fabrique la fixture oracle 200 depuis le témoin witness_long_before.

positions = i32[200] avec positions[0] = <P> (longueur du prompt rendu, MESURÉE — plan Task 4.1) ;
le reste du tableau n'est pas consommé par le runner (seul positions[0] est lu, cf
gemma4_g12auto.zig, garde OraclePromptMismatch) mais le format oracle exige le tenseur complet.
fed = les 200 ids du témoin (relus depuis son safetensors, clé "ids").
Usage: python3 scripts/73_d10_oracle200_fixture.py <witness.safetensors> <P> <out.safetensors>
"""
import json
import struct
import sys


def read_ids(path):
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hlen))
        info = header["ids"]
        assert info["dtype"] == "I32", info
        n = info["shape"][0]
        data = f.read()
        off = info["data_offsets"]
        return list(struct.unpack(f"<{n}i", data[off[0]:off[1]]))


def write_fixture(path, positions, fed):
    tensors = {"positions": positions, "fed": fed}
    header = {}
    blobs = []
    off = 0
    for name, vals in tensors.items():
        raw = struct.pack(f"<{len(vals)}i", *vals)
        header[name] = {"dtype": "I32", "shape": [len(vals)],
                        "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    hj = json.dumps(header, separators=(",", ":")).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        for b in blobs:
            f.write(b)


def main():
    witness, p_len, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    fed = read_ids(witness)
    assert len(fed) == 200, f"témoin attendu à 200 ids, lu {len(fed)}"
    positions = [p_len] + [0] * 199
    write_fixture(out, positions, fed)
    print(f"OK: {out} (positions[0]={p_len}, fed={len(fed)} ids)")


if __name__ == "__main__":
    main()
