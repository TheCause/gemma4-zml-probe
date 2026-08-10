#!/usr/bin/env python3
"""K5 — contre-épreuve du dépouilleur 80 : le juge doit être VU condamner.

Chaque mutant est appliqué au log RÉEL puis CONTRÔLÉ : si le log muté est identique au
nominal, le cas est déclaré INEXÉCUTABLE et non « PASS ». C'est le piège qui a mordu à la
première tentative — un `sed '0,/re/s//repl/'` silencieusement inopérant sur BSD faisait
passer le mutant pour un juge aveugle. Un mutant qui ne mute pas ne prouve rien.

Usage : 81_selfproof_80.py <pf1.json> <runB.err.log>"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

JSON, LOG = sys.argv[1], sys.argv[2]
nominal = Path(LOG).read_text()
JUDGE = str(Path(__file__).with_name("80_pf1_bridge.py"))


def judge(text, extra=()):
    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as fh:
        fh.write(text)
        path = fh.name
    r = subprocess.run([sys.executable, JUDGE, JSON, path, *extra],
                       capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr).strip()


def case(name, mutate, expect_fail=True):
    muted = mutate(nominal)
    print(f"--- {name} ---")
    if muted == nominal:
        print("INEXECUTABLE : le mutant n'a RIEN changé au log — le cas ne prouve rien")
        return False
    rc, out = judge(muted)
    print(out)
    ok = (rc != 0) if expect_fail else (rc == 0)
    print(f"exit={rc} -> {'ATTENDU' if ok else '⚠ NON ATTENDU'}")
    return ok


print("=== 80_selfproof — le juge PF1 doit être VU condamner ===")
print("--- (0) nominal : PASS attendu ---")
rc0, out0 = judge(nominal)
print(out0)
print(f"exit={rc0} -> {'ATTENDU' if rc0 == 0 else '⚠ NON ATTENDU'}")

results = [rc0 == 0]

# (a) un argmax de ligne ctx altéré : le contenu ment, le comptage reste bon.
results.append(case(
    "(a) idx de tête d'une ligne ctx altéré (563 -> 999999) : MISMATCH ctx attendu",
    lambda s: re.sub(r"(top5 @ ctx=55 : idx=\{ )\d+", r"\g<1>999999", s, count=1)))

# (b) une ligne ctx supprimée : le comptage ment.
results.append(case(
    "(b) une ligne ctx supprimée : n_ctx FAIL attendu",
    lambda s: "\n".join(l for l in s.splitlines() if "top5 @ ctx=60 " not in l)))

# (c) un id de generated altéré : régime POLICY.
results.append(case(
    "(c) 1er id de generated altéré : MISMATCH gen[0] attendu",
    lambda s: re.sub(r"(generated = \{ )\d+", r"\g<1>999999", s, count=1)))

# (d) plus aucune ligne ctx : un PASS serait VIDE.
results.append(case(
    "(d) toutes les lignes ctx retirées : refus bruyant attendu",
    lambda s: "\n".join(l for l in s.splitlines() if "top5 @ ctx=" not in l)))

# (e) bannière de build absente : INEXECUTABLE, jamais PASS.
results.append(case(
    "(e) bannière BUILD retirée : INEXECUTABLE attendu",
    lambda s: "\n".join(l for l in s.splitlines() if "BUILD: mode=" not in l)))

print()
print(f"BILAN : {sum(results)}/{len(results)} cas conformes")
sys.exit(0 if all(results) else 1)
