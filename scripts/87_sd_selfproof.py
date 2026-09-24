#!/usr/bin/env python3
"""87 — Contre-épreuve du dépouilleur 86 (spec §4.6). Pour chaque mutant, copie les preuves dans un
dossier temporaire, applique la mutation, VÉRIFIE que le fichier muté diffère du nominal (sinon
INEXÉCUTABLE — leçon 81_selfproof_80.py), relance 86, et exige le verdict attendu du contrôle visé.
"""
from __future__ import annotations

import importlib.util
import io
import json
import shutil
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("r86", HERE / "86_sd_report.py")
r86 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r86)
EV = Path("docs/evidence/sd")


def rows(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def write(path, rs):
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rs))


def run86(d):
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = r86.main(["--letter", str(d / "r_letter.jsonl"), "--json", str(d / "r_json.jsonl"),
                         "--gen-text", str(EV / "gen_text.json"), "--out", str(d / "out.md"),
                         "--logs", str(EV / "run_letter.log"), str(EV / "run_json.log")])
    return code, buf.getvalue()


def verdict(out, cid):
    for l in out.splitlines():
        if l.startswith(cid + " ") or l.startswith(cid.ljust(12)):
            return l.split()[1]
    return None


def mutant_a(rs, ora):   # zc permutés d'un cran sur un cas où le décalage mord (> 2×TOL)
    for r in rs:
        if r["order"] == "fwd" and r["rep"] == 1:
            ids = list(r["zc"].keys())
            zh = [ora[f"{r['case_id']}/{r['perm']}"]["zc"][i] for i in ids]
            if max(abs(zh[(k + 1) % 4] - zh[k]) for k in range(4)) > 0.1:
                vals = [r["zc"][i] for i in ids]
                r["zc"] = {ids[k]: vals[(k + 1) % 4] for k in range(4)}
                return True
    return False


def mutant_b(rs, _):     # passe inverse altérée d'1 ulp sur un cas
    for r in rs:
        if r["order"] == "rev":
            k = next(iter(r["zc_bits"]))
            r["zc_bits"][k] = f"{(int(r['zc_bits'][k], 16) ^ 1):08x}"
            return True
    return False


def mutant_c(rs, _):     # option orig appliquée à un cas rev
    man = json.load(open("fixtures/sd_manifest.json"))
    orig = {e["case_id"]: e for e in man["letter"] if e["perm"] == "orig"}
    for r in rs:
        if r["perm"] == "rev" and r.get("decision", {}).get("kind") == "decision":
            o = orig[r["case_id"]]
            want_orig = r86.softmax_option(r["zc"], o["label_ids"], o["classes"])
            if want_orig != r["decision"]["option"]:
                r["decision"]["option"] = want_orig
                return True
    return False


def mutant_d(rs, _):     # un cas manquant
    for k, r in enumerate(rs):
        if r["order"] == "fwd" and r["rep"] == 3:
            del rs[k]
            return True
    return False


def mutant_e(rs, _):     # step0 ≠ 0
    rs[5]["step0"] = 17
    return True


MUTANTS = [("a", mutant_a, "C-SD-B", {"FAIL"}), ("b", mutant_b, "C-SD-D", {"FAIL"}),
           ("c", mutant_c, "C-SD-F(86)", {"FAIL"}), ("d", mutant_d, None, {3}),
           ("e", mutant_e, "C-SD-D", {"FAIL"})]


def main() -> int:
    ora = json.load(open("fixtures/sd_oracle.json"))["cases"]
    nominal = "".join(json.dumps(r) + "\n" for r in rows(EV / "r_letter.jsonl"))  # même sérialisation que write()
    bad = 0
    for name, fn, cid, want in MUTANTS:
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            shutil.copy(EV / "r_json.jsonl", d / "r_json.jsonl")
            rs = rows(EV / "r_letter.jsonl")
            applied = fn(rs, ora)
            write(d / "r_letter.jsonl", rs)
            if not applied or (d / "r_letter.jsonl").read_text() == nominal:
                print(f"mutant {name} : INEXÉCUTABLE (mutation vide)")
                bad += 1
                continue
            code, out = run86(d)
            got = code if cid is None else verdict(out, cid)
            ok = got in want
            print(f"mutant {name} : {cid or 'exit'} = {got} (attendu {want}) → {'CONDAMNÉ' if ok else 'NON CONDAMNÉ'}")
            bad += not ok
    print("87 PASS — le dépouilleur condamne les 5 mutants" if bad == 0 else f"87 FAIL — {bad} mutant(s) non condamné(s) ou inexécutable(s)")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
