#!/usr/bin/env python3
"""Inspection et FABRICATION DE MUTANTS pour les dumps de KV-cache (spec kvdump §6, livrable 3).

Le dump est un safetensors auto-décrivant : 8 octets de longueur (u64 LE) + header JSON
(`__metadata__` + entrées de tenseurs) + données brutes contiguës.

⚠ TOUT est fait à BAS NIVEAU (struct + json + copie par blocs), JAMAIS via l'API haut niveau
`safetensors.torch.save_file` : celle-ci réordonne les clés et recalcule les offsets, ce qui
détruirait précisément ce qu'un mutant doit conserver (un manifest INTACT sur des données
altérées). `safe_open` n'est utilisé qu'en LECTURE, dans `inspect`.

Sous-commandes :
  inspect <f>                     clés, metadata, recalcul des 5 xxh64 ; exit 0 ssi tous ok
  mutate-flip <f> <out> <t> <off> flip d'1 octet dans les DONNÉES du tenseur t, manifest INTACT
  make-zeroed <f> <out>           4 caches zérotés + xxh64 du manifest RECALCULÉS (manifest
                                  VALIDE : c'est le MORDANT qu'on teste, pas le checksum)
  set-meta <f> <out> <k> <v>      réécrit UNE clé de metadata (instruments DC5 b/d/f)
  mutate-shape <f> <out> <t>      +1 sur la 1re dim du tenseur t dans le header, data intacte
  shift-fwd <src> <dst>           K5/PF2 : le dump MENT d'une position (step_next+1, ids_fed
                                  étendu d'un id fantôme, caches INTACTS) — le manifest reste
                                  cohérent, mais déclare un token que le cache ne porte pas
  make-fixture <n> <p0> <out>     fixture oracle de n ids (tuilage de u9_ids.safetensors) avec
                                  positions[0] == p0 (la garde du runner l'exige) ; écrit les
                                  DEUX clés lues par --oracle : `positions` et `fed`
"""
import argparse
import json
import os
import re
import struct
import sys

import numpy as np
import xxhash

BLOCK = 64 << 20  # 64 MiB : les caches pèsent jusqu'à 1,3 Gio par tenseur
CACHE_KEYS = ("sl_k", "sl_v", "fl_k", "fl_v")


def read_header(path):
    """-> (header_len, header_dict, data_base). Ne lit AUCUNE donnée."""
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise SystemExit(f"{path}: tronqué (moins de 8 octets)")
        n = struct.unpack("<Q", raw)[0]
        hdr = json.loads(f.read(n))
    return n, hdr, 8 + n


def entries(hdr):
    """Entrées de tenseurs, dans l'ordre des data_offsets (l'ordre du fichier)."""
    items = [(k, v) for k, v in hdr.items() if k != "__metadata__"]
    items.sort(key=lambda kv: kv[1]["data_offsets"][0])
    return items


def hash_range(f, start, length):
    h = xxhash.xxh64()
    f.seek(start)
    left = length
    while left:
        chunk = f.read(min(BLOCK, left))
        if not chunk:
            raise SystemExit("lecture courte : fichier tronqué")
        h.update(chunk)
        left -= len(chunk)
    return h.intdigest()


def write_file(out, hdr, payload_writer):
    """Écrit len+header+données. `payload_writer(fout)` écrit les données, dans l'ordre du header."""
    blob = json.dumps(hdr, separators=(",", ":")).encode()
    with open(out, "wb") as fo:
        fo.write(struct.pack("<Q", len(blob)))
        fo.write(blob)
        payload_writer(fo)


def copy_payload(src, data_base, total):
    def writer(fo):
        with open(src, "rb") as fi:
            fi.seek(data_base)
            left = total
            while left:
                chunk = fi.read(min(BLOCK, left))
                if not chunk:
                    raise SystemExit("lecture courte pendant la copie")
                fo.write(chunk)
                left -= len(chunk)

    return writer


def total_bytes(hdr):
    return max(v["data_offsets"][1] for _, v in entries(hdr))


def cmd_inspect(a):
    n, hdr, base = read_header(a.file)
    meta = hdr.get("__metadata__", {})
    print(f"fichier      : {a.file}")
    print(f"header       : {n} octets, data_base = {base}")
    print(f"taille réelle: {os.path.getsize(a.file)} octets")
    print("tenseurs     :")
    for k, v in entries(hdr):
        off0, off1 = v["data_offsets"]
        print(f"  {k:8s} {v['dtype']:4s} shape={v['shape']} octets={off1 - off0}")
    print("manifest     :")
    for k in sorted(meta):
        print(f"  {k} = {meta[k]}")
    ok = True
    with open(a.file, "rb") as f:
        for k, v in entries(hdr):
            want = meta.get(f"{k}_xxh64")
            if want is None:
                print(f"  ⚠ {k}: aucun checksum au manifest")
                ok = False
                continue
            off0, off1 = v["data_offsets"]
            got = hash_range(f, base + off0, off1 - off0)
            good = f"{got:x}" == want
            ok = ok and good
            print(f"  xxh64 {k:8s} : {got:x} {'== manifest' if good else '!= manifest (' + want + ')'}")
    print("VERDICT      :", "tous les checksums concordent" if ok else "DISCORDANCE")
    return 0 if ok else 1


def cmd_mutate_flip(a):
    _, hdr, base = read_header(a.file)
    ent = dict(entries(hdr))
    if a.tensor not in ent:
        raise SystemExit(f"tenseur inconnu : {a.tensor}")
    off0, off1 = ent[a.tensor]["data_offsets"]
    if not (0 <= a.offset < off1 - off0):
        raise SystemExit(f"offset {a.offset} hors du tenseur ({off1 - off0} octets)")
    total = total_bytes(hdr)
    write_file(a.out, hdr, copy_payload(a.file, base, total))
    # flip APRÈS copie : manifest (donc checksums) INTACT, une donnée altérée.
    pos = 8 + len(json.dumps(hdr, separators=(",", ":")).encode()) + off0 + a.offset
    with open(a.out, "r+b") as f:
        f.seek(pos)
        b = f.read(1)
        f.seek(pos)
        f.write(bytes([b[0] ^ 0xFF]))
    print(f"mutate-flip : {a.tensor}[{a.offset}] {b[0]:#04x} -> {b[0] ^ 0xFF:#04x} dans {a.out} (manifest intact)")
    return 0


def cmd_make_zeroed(a):
    _, hdr, base = read_header(a.file)
    meta = hdr["__metadata__"]
    zero_h = {}
    for k, v in entries(hdr):
        if k in CACHE_KEYS:
            length = v["data_offsets"][1] - v["data_offsets"][0]
            h = xxhash.xxh64()
            left = length
            while left:
                m = min(BLOCK, left)
                h.update(b"\0" * m)
                left -= m
            zero_h[k] = f"{h.intdigest():x}"
    for k, v in zero_h.items():
        meta[f"{k}_xxh64"] = v  # manifest VALIDE : DC4 teste le mordant, pas le checksum

    def writer(fo):
        with open(a.file, "rb") as fi:
            for k, v in entries(hdr):
                off0, off1 = v["data_offsets"]
                length = off1 - off0
                if k in CACHE_KEYS:
                    left = length
                    while left:
                        m = min(BLOCK, left)
                        fo.write(b"\0" * m)
                        left -= m
                else:
                    fi.seek(base + off0)
                    left = length
                    while left:
                        chunk = fi.read(min(BLOCK, left))
                        fo.write(chunk)
                        left -= len(chunk)

    write_file(a.out, hdr, writer)
    print(f"make-zeroed : 4 caches à zéro, checksums RECALCULÉS (manifest valide) -> {a.out}")
    return 0


def cmd_shift_fwd(a):
    """K5/PF2 — forge un dump qui MENT d'UNE position.

    step_next+1, `ids_fed` étendu d'un id fantôme (fed_next recopié — un id valide par
    construction), fed_next inchangé, checksum `ids_fed_xxh64` RECALCULÉ, les 4 caches
    intacts avec leurs checksums d'origine. Le manifest reste donc COHÉRENT (l'invariant
    ids_fed.len == step_next tient, tous les xxh64 concordent) : ce qu'il déclare, en
    revanche, est un token que le cache ne porte PAS — la position step_next du cache est
    restée aux zéros de l'allocation. C'est le mécanisme DC4 (cache zéroté) localisé à UNE
    seule position, donc la corruption la plus fine qu'un bug de position produirait.

    ⚠ shift-BACK (tronquer d'une position) a été analysé et REJETÉ à la spec (C-K5-B) :
    l'état tronqué est AUTO-COHÉRENT — le run forgé réécrirait ce slot à l'identique, et son
    mordant nul serait sain, donc ininterprétable comme gate."""
    _, hdr, base = read_header(a.src)
    meta = hdr["__metadata__"]
    ent = dict(entries(hdr))
    if "ids_fed" not in ent:
        raise SystemExit("ids_fed absent du dump — ce n'est pas un g12-kvdump-v1")
    off0, off1 = ent["ids_fed"]["data_offsets"]
    n_old = (off1 - off0) // 4
    if n_old != int(meta["step_next"]):
        raise SystemExit(f"invariant rompu AVANT mutation : ids_fed={n_old} != step_next={meta['step_next']}")
    with open(a.src, "rb") as f:
        f.seek(base + off0)
        ids = np.frombuffer(f.read(off1 - off0), dtype=np.int32)
    phantom = np.int32(int(meta["fed_next"]))
    new_ids = np.append(ids, phantom)
    new_blob = new_ids.tobytes()

    # Offsets RECALCULÉS séquentiellement dans l'ordre du fichier : ids_fed est le dernier
    # tenseur du format v1, mais un recalcul générique survit à un changement d'ordre.
    cursor = 0
    order = [k for k, _ in entries(hdr)]
    for k in order:
        length = len(new_blob) if k == "ids_fed" else (
            hdr[k]["data_offsets"][1] - hdr[k]["data_offsets"][0])
        hdr[k]["data_offsets"] = [cursor, cursor + length]
        cursor += length
    hdr["ids_fed"]["shape"] = [len(new_ids)]
    meta["step_next"] = str(int(meta["step_next"]) + 1)
    meta["ids_fed_xxh64"] = f"{xxhash.xxh64(new_blob).intdigest():x}"

    def writer(fo):
        with open(a.src, "rb") as fi:
            for k in order:
                if k == "ids_fed":
                    fo.write(new_blob)
                    continue
                o0, o1 = ent[k]["data_offsets"]  # offsets d'ORIGINE pour la lecture
                fi.seek(base + o0)
                left = o1 - o0
                while left:
                    chunk = fi.read(min(BLOCK, left))
                    if not chunk:
                        raise SystemExit("lecture courte pendant la copie")
                    fo.write(chunk)
                    left -= len(chunk)

    write_file(a.dst, hdr, writer)
    print(f"shift-fwd : {a.src} -> {a.dst} step_next={meta['step_next']} "
          f"ids_fed {n_old}->{len(new_ids)} phantom={int(phantom)} "
          f"(caches INTACTS : la position {n_old} du cache reste aux zéros)")
    return 0


def cmd_set_meta(a):
    _, hdr, base = read_header(a.file)
    old = hdr["__metadata__"].get(a.key)
    hdr["__metadata__"][a.key] = a.value
    write_file(a.out, hdr, copy_payload(a.file, base, total_bytes(hdr)))
    print(f"set-meta : {a.key} : {old!r} -> {a.value!r} dans {a.out}")
    return 0


def cmd_mutate_shape(a):
    _, hdr, base = read_header(a.file)
    if a.tensor not in hdr:
        raise SystemExit(f"tenseur inconnu : {a.tensor}")
    old = list(hdr[a.tensor]["shape"])
    hdr[a.tensor]["shape"][0] = old[0] + 1  # data INTACTE : c'est la shape déclarée qui ment
    write_file(a.out, hdr, copy_payload(a.file, base, total_bytes(hdr)))
    print(f"mutate-shape : {a.tensor} {old} -> {hdr[a.tensor]['shape']} dans {a.out} (données intactes)")
    return 0


def cmd_make_fixture(a):
    src = a.source
    with open(src, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        ent = hdr["ids"]
        off0, off1 = ent["data_offsets"]
        f.seek(8 + n + off0)
        ids = np.frombuffer(f.read(off1 - off0), dtype=np.int32)
    if ids.size == 0:
        raise SystemExit(f"{src}: 'ids' vide")
    reps = (a.n + ids.size - 1) // ids.size
    fed = np.tile(ids, reps)[: a.n].astype(np.int32)
    # positions[0] == p0 : la garde du runner (`positions[0] == ids.len`) l'EXIGE, et elle
    # dépend de la géométrie du run consommateur — d'où un paramètre, pas une constante.
    positions = (np.arange(a.n, dtype=np.int32) + a.p0).astype(np.int32)
    out_hdr = {
        "positions": {"dtype": "I32", "shape": [a.n], "data_offsets": [0, 4 * a.n]},
        "fed": {"dtype": "I32", "shape": [a.n], "data_offsets": [4 * a.n, 8 * a.n]},
    }

    def writer(fo):
        fo.write(positions.tobytes())
        fo.write(fed.tobytes())

    write_file(a.out, out_hdr, writer)
    print(f"make-fixture : {a.n} ids (tuilage de {os.path.basename(src)}), positions[0]={a.p0} -> {a.out}")
    return 0


def parse_generated(log_path):
    """Extrait la ligne `generated = { a, b, ... }` du log du runner (mode libre)."""
    txt = open(log_path, encoding="utf-8", errors="replace").read()
    m = re.search(r"generated = \{([^}]*)\}", txt)
    if not m:
        raise SystemExit(f"{log_path}: aucune ligne `generated = {{...}}` (le run est-il allé au bout ?)")
    body = m.group(1).strip()
    return [int(x) for x in body.split(",") if x.strip()] if body else []


def cmd_verdict(a):
    """Juge un run de continuation contre la référence DC2.

    dc4 (MORDANT, claim C-E)  : PASS ssi la 1re divergence d'ids est à un index < 4.
    dc3 (INTER-PROCESS, C-C)  : PASS ssi ids[0] identique ET (aucune divergence, ou 1re
                                divergence à une marge <= 1,873e-3 = 2 x le bruit U7, seuil
                                HÉRITÉ de la spec generation-config §2bis, publiée comme tie).
    """
    ref = json.load(open(a.ref, encoding="utf-8"))
    ids_ref, marges = ref["ids"], ref["marges"]
    got = parse_generated(a.log)
    n = min(len(ids_ref), len(got))
    first = next((j for j in range(n) if ids_ref[j] != got[j]), None)
    n_match = sum(1 for j in range(n) if ids_ref[j] == got[j])
    print(f"mode         : {a.mode}")
    print(f"ids référence: {len(ids_ref)}  ids run: {len(got)}  comparés: {n}")
    print(f"n_match      : {n_match}/{n}")
    if first is None:
        print("1re divergence: AUCUNE")
    else:
        mg = marges[first]
        print(f"1re divergence: @gen={first}  ref={ids_ref[first]} got={got[first]}  marge={mg}")
    if a.mode == "dc4":
        ok = first is not None and first < 4
        print(f"VERDICT DC4  : {'PASS' if ok else 'FAIL'} — divergence attendue dans les 4 premiers tokens")
        if first is None:
            print("  ⚠ continuation IDENTIQUE : le cache ne porterait RIEN — tous les gates")
            print("    d'équivalence de ce chantier seraient VIDES (spec C-E : STOP diagnostic).")
        return 0 if ok else 1
    # dc3
    if got and ids_ref and got[0] != ids_ref[0]:
        print(f"VERDICT DC3  : FAIL — argmax du 1er step différent (marge de référence {marges[0]})")
        return 1
    if first is None:
        print(f"VERDICT DC3  : PASS — 1er step identique, aucune divergence sur {n} tokens")
        return 0
    mg = marges[first]
    tie = mg is not None and mg <= 1.873e-3
    print(f"VERDICT DC3  : {'PASS (tie de bistabilité publié)' if tie else 'FAIL'} — seuil 1.873e-3")
    if not tie:
        print("  ⚠ divergence à marge > seuil : signature d'un ÉTAT CORROMPU, pas d'un tie.")
    return 0 if tie else 1


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("inspect")
    s.add_argument("file")
    s.set_defaults(fn=cmd_inspect)

    s = sub.add_parser("mutate-flip")
    s.add_argument("file")
    s.add_argument("out")
    s.add_argument("tensor")
    s.add_argument("offset", type=int)
    s.set_defaults(fn=cmd_mutate_flip)

    s = sub.add_parser("make-zeroed")
    s.add_argument("file")
    s.add_argument("out")
    s.set_defaults(fn=cmd_make_zeroed)

    s = sub.add_parser("set-meta")
    s.add_argument("file")
    s.add_argument("out")
    s.add_argument("key")
    s.add_argument("value")
    s.set_defaults(fn=cmd_set_meta)

    s = sub.add_parser("mutate-shape")
    s.add_argument("file")
    s.add_argument("out")
    s.add_argument("tensor")
    s.set_defaults(fn=cmd_mutate_shape)

    s = sub.add_parser("shift-fwd")
    s.add_argument("src")
    s.add_argument("dst")
    s.set_defaults(fn=cmd_shift_fwd)

    s = sub.add_parser("verdict")
    s.add_argument("--log", required=True)
    s.add_argument("--ref", required=True)
    s.add_argument("--mode", choices=["dc3", "dc4"], required=True)
    s.set_defaults(fn=cmd_verdict)

    s = sub.add_parser("make-fixture")
    s.add_argument("n", type=int)
    s.add_argument("p0", type=int)
    s.add_argument("out")
    s.add_argument("--source", default="/data/gemma4-zml-probe/u9_ids.safetensors")
    s.set_defaults(fn=cmd_make_fixture)

    a = p.parse_args()
    sys.exit(a.fn(a))


if __name__ == "__main__":
    main()
