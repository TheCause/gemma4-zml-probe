#!/usr/bin/env python3
"""K5/PF6 — le rendu Zig du tour 2 == le suffixe HF mesuré (Task 2), en ids, littéralement.

Le seul juge du littéral `renderChatTemplateTurn2`. Un FAIL sur l'id de TÊTE ne se corrige
PAS en retranchant cet id du littéral : il se diagnostique contre `closure_tail_ids` du
même JSON (la frontière clôture-du-tour-1 / tour-2 — spec §4.4.1).

Usage : 79_t2_render_check.py <rendu_tour2_hf.json> <ids_only_turn2.log>"""
import json
import re
import sys

j = json.load(open(sys.argv[1]))
log = open(sys.argv[2]).read()

mode = re.search(r"BUILD: mode=(\w+)", log)
if not mode:
    sys.exit("DEPOUILLEMENT IMPOSSIBLE : 'BUILD: mode=' absent du log — jamais un PASS")
if mode.group(1) != "ReleaseFast":
    sys.exit(f"INEXECUTABLE : BUILD mode={mode.group(1)}")

m = re.search(r"ids_turn2 = \{ ([\d, ]+) \}", log)
if not m:
    sys.exit("DEPOUILLEMENT IMPOSSIBLE : ids_turn2 absent du log")
zig = [int(x) for x in m.group(1).split(",")]
hf = j["suffix_ids"]

if zig == hf:
    print(f"PF6 PASS — {len(zig)} ids identiques")
    print(f"  (frontière : closure_tail_ids={j['closure_tail_ids']} appartient à l'injection D-K5-5)")
    sys.exit(0)

k = next((i for i, (a, b) in enumerate(zip(zig, hf)) if a != b), min(len(zig), len(hf)))
diag = ""
if k == 0 and j.get("closure_tail_ids") and zig and zig[0] in j["closure_tail_ids"]:
    diag = (f"\n  DIAGNOSTIC : l'id de tête {zig[0]} appartient à closure_tail_ids="
            f"{j['closure_tail_ids']} — c'est l'INJECTION de clôture qui est mal placée, "
            f"PAS le littéral du tour 2. Ne pas retrancher l'id du littéral.")
sys.exit(f"PF6 FAIL — 1er écart à l'index {k} : zig={zig[k:k + 4]} hf={hf[k:k + 4]} "
         f"(len {len(zig)} vs {len(hf)}){diag}")
