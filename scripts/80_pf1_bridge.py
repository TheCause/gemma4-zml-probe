#!/usr/bin/env python3
"""K5/PF1 — verdict machine : oracle context (69 --context-ids) vs log du runner.

Le binaire publie des mesures BRUTES ; le juge est ICI, avec ses seuils pré-enregistrés
(spec 2026-08-10 §5). Rien ne se lit à l'œil.

CANAUX (spec §4.5) — les mélanger ferait échouer à tort tout scénario où une suppression
mord (cas réel : docs/evidence/kvdump/dc4.err.log:16, argmax brut 258882 supprimé à gen=0) :
  * régime ctx  : les lignes `top5 @ ctx=` du runner sont des logits BRUTS in-graph
                  -> comparées à `argmax_raw` de l'oracle ;
  * régime gen  : `generated` sort de policy.select (POST-politique)
                  -> comparé à `argmax_policy`.

Usage : 80_pf1_bridge.py <pf1.json> <runB.err.log> [--expect-fail]
Sortie : exit 0 ssi PASS (ou, avec --expect-fail, ssi le mutant mord GRAS)."""
import json
import re
import sys

TIE_MARGIN = 1.873e-3   # spec generation-config §2bis (2× le bruit U7) — hérité, pas réglé ici
FAT_FACTOR = 10.0       # PF2 : un mismatch « gras » a une marge > 10× TIE_MARGIN

RE_BUILD = re.compile(r"BUILD: mode=(\w+)")
RE_KVLOAD = re.compile(r"KVLOAD: .*step_next=(\d+) fed_next=(\d+)")
RE_CTX = re.compile(r"top5 @ ctx=(\d+) : idx=\{ ([\d, ]+) \}")
RE_GEN = re.compile(r"generated = \{ ([\d, ]+) \}")


def need(m, what):
    if not m:
        sys.exit(f"DEPOUILLEMENT IMPOSSIBLE : {what} absent du log — jamais un zéro, jamais un PASS")
    return m


log = open(sys.argv[2]).read()
mode = need(RE_BUILD.search(log), "BUILD: mode=").group(1)
if mode != "ReleaseFast":
    sys.exit(f"INEXECUTABLE : BUILD mode={mode} (un log de gate sans ReleaseFast n'est pas un PASS)")
step_next = int(need(RE_KVLOAD.search(log), "KVLOAD:").group(1))
ctx_lines = [(int(p), [int(x) for x in idx.split(",")]) for p, idx in RE_CTX.findall(log)]
generated = [int(x) for x in need(RE_GEN.search(log), "generated =").group(1).split(",")]
if not ctx_lines:
    sys.exit("DEPOUILLEMENT IMPOSSIBLE : aucune ligne 'top5 @ ctx=' — le prefill de reprise n'a "
             "rien émis (--dump-top5 oublié ?), un PASS ici serait VIDE")

j = json.load(open(sys.argv[1]))
C = j["C"]
pos = {int(e["p"]): e for e in j["positions"]}
fails = []

# --- Régime (i) ctx : BRUT <-> BRUT. Appariement ORDINAL : le mutant PF2 décale les positions
# absolues d'une unité, l'ordinal compare quand même et c'est le mismatch de CONTENU qui fait foi.
oracle_ctx = [pos[p] for p in sorted(pos) if p <= C - 2]
if len(ctx_lines) != len(oracle_ctx):
    fails.append(f"n_ctx runner={len(ctx_lines)} != oracle={len(oracle_ctx)} margin=999.0")
for k, ((p_run, idx_run), e) in enumerate(zip(ctx_lines, oracle_ctx)):
    if idx_run[0] != e["argmax_raw"]:
        fails.append(f"ctx[{k}] p_run={p_run} p_hf={e['p']} zml={idx_run[0]} "
                     f"hf={e['argmax_raw']} margin={e['margin_raw']:.6f}")

# --- Régime (i) suite : la PREMIÈRE génération (produite à la position C-1) — POLICY <-> POLICY.
e0 = pos.get(C - 1)
if e0 is None:
    sys.exit("DEPOUILLEMENT IMPOSSIBLE : position C-1 absente du rapport oracle")
if generated[0] != e0["argmax_policy"]:
    fails.append(f"gen[0] zml={generated[0]} hf={e0['argmax_policy']} "
                 f"margin={e0['margin_policy']:.6f}")

# --- Régime (ii) : générations suivantes — première divergence à marge <= TIE tolérée et
# PUBLIÉE (bistabilité, régime DC3 ; aucune claim de trajectoire complète n'est énonçable).
tie_note = None
for k in range(1, len(generated)):
    e = pos.get(C - 1 + k)
    if e is None:
        break
    if generated[k] != e["argmax_policy"]:
        if e["margin_policy"] <= TIE_MARGIN:
            tie_note = (f"tie @ gen={k} margin={e['margin_policy']:.6f} <= {TIE_MARGIN} "
                        f"(publié, régime DC3)")
        else:
            fails.append(f"gen[{k}] zml={generated[k]} hf={e['argmax_policy']} "
                         f"margin={e['margin_policy']:.6f} > TIE")
        break

verdict_fail = bool(fails)
print(f"step_next={step_next} C={C} n_ctx={len(ctx_lines)} (oracle {len(oracle_ctx)}) "
      f"gen={len(generated)}")
if tie_note:
    print(tie_note)
for f_ in fails:
    print("MISMATCH:", f_)

if "--expect-fail" in sys.argv:
    fat = any(float(f_.split("margin=")[1].split()[0]) > FAT_FACTOR * TIE_MARGIN for f_ in fails)
    if verdict_fail and fat:
        print("PF2 : le mutant MORD (mismatches gras) — attendu")
        sys.exit(0)
    sys.exit("PF2 : le mutant NE MORD PAS — le gate ne prouve rien. STOP diagnostic : un mordant "
             "nul condamne l'ANTÉCÉDENT (le scénario), pas la corruption (leçon RP4c).")

print("PF1 : PASS — aucun mismatch" if not verdict_fail else "PF1 : FAIL")
sys.exit(1 if verdict_fail else 0)
