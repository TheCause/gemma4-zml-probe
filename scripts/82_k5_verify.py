#!/usr/bin/env python3
"""K5 — VÉRIFICATEUR D'ENSEMBLE : rejoue les 7 gates depuis les preuves versionnées.

Tourne sur M1, **sans GPU, sans M4, en quelques secondes**. Il ne refait pas les runs : il
re-dépouille les logs et rapports committés dans `docs/evidence/k5/` avec les mêmes juges que
le jour J, et vérifie que les 7 tags git sont bien des ancêtres de HEAD.

Ce qu'il NE prouve PAS : que le binaire d'aujourd'hui produirait encore ces logs. Pour cela il
faut rejouer les runs sur la VM (cf `docs/K5_RESULTS.md` §9) — c'est une autre opération, et
elle est nommée ici pour qu'on ne confonde pas les deux.

Toute preuve manquante est un ÉCHEC BRUYANT, jamais un gate silencieusement sauté.

Usage : python3 scripts/82_k5_verify.py [--verbose]
Exit  : 0 ssi les 7 gates PASSENT et les 2 contre-épreuves tiennent."""
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EV = ROOT / "docs" / "evidence" / "k5"
SC = ROOT / "scripts"
VERBOSE = "--verbose" in sys.argv
HLO_REF = "297679847aa04b719942d75d093adf2b"

results = []


def need_file(name):
    p = EV / name
    if not p.exists():
        raise FileNotFoundError(f"preuve absente : docs/evidence/k5/{name}")
    return p


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    return r.returncode, (r.stdout + r.stderr).strip()


def gate(name, fn):
    try:
        detail = fn()
        results.append((name, True, detail))
    except Exception as e:                                    # noqa: BLE001
        results.append((name, False, f"{type(e).__name__}: {e}"))


# ---------------------------------------------------------------- PF0
def pf0():
    av = need_file("hlo_witness_avant.md5").read_text().split()[0]
    ap = need_file("hlo_witness_apres.md5").read_text().split()[0]
    assert av == HLO_REF, f"md5 AVANT {av} != témoin de référence {HLO_REF}"
    assert ap == HLO_REF, f"md5 APRÈS {ap} != témoin de référence {HLO_REF}"
    rc, out = run(["git", "diff", "main", "--stat", "--", "zml_runner/engine.zig"])
    assert out == "", f"engine.zig a bougé :\n{out}"
    return f"md5 HLO avant == après == {HLO_REF[:12]}… ; engine.zig 0 octet de diff"


# ---------------------------------------------------------------- PF1 / PF3
def pf_equiv(stem_json, stem_log, label, expect_window=None):
    j, lg = need_file(stem_json), need_file(stem_log)
    rc, out = run([sys.executable, str(SC / "80_pf1_bridge.py"), str(j), str(lg)])
    assert rc == 0, f"le juge 80 REFUSE {label} :\n{out}"
    assert "PASS" in out, f"verdict inattendu :\n{out}"
    d = json.loads(j.read_text())
    if expect_window is not None:
        w = d["window"]["bites_in_prefill"]
        assert w is expect_window, f"témoin fenêtre = {w}, attendu {expect_window} (gate INEXECUTABLE sinon)"
    n = re.search(r"n_ctx=(\d+) \(oracle (\d+)\) gen=(\d+)", out)
    return (f"{n.group(1)} ctx (oracle {n.group(2)}) + {n.group(3)} gen, 0 mismatch"
            + (f" ; fenêtre mordante={d['window']['bites_in_prefill']}" if expect_window is not None else ""))


# ---------------------------------------------------------------- PF2
def pf2():
    j, lg = need_file("pf1.json"), need_file("pf2_shift2.err.log")
    rc, out = run([sys.executable, str(SC / "80_pf1_bridge.py"), str(j), str(lg), "--expect-fail"])
    assert rc == 0, f"le mutant canonique (--n 2) NE MORD PAS :\n{out}"
    assert "MORD" in out, f"verdict inattendu :\n{out}"
    # Et le nominal doit re-passer sur le même rapport : sans ça, « ça mord » ne
    # distinguerait pas un mutant mordant d'un juge qui condamne tout.
    rc2, out2 = run([sys.executable, str(SC / "80_pf1_bridge.py"), str(j),
                     str(need_file("pf1_runB.err.log"))])
    assert rc2 == 0, "le NOMINAL échoue aussi — le juge condamne tout, le mordant ne prouve rien"
    mm = re.search(r"MISMATCH: (ctx\[\d+\][^\n]*)", out)
    # La réfutation à N=1 fait partie du dossier : elle doit rester consultable.
    need_file("DECISION_C-K5-B_requalifiee.md")
    need_file("pf2_sensibilite.md")
    return f"mutant --n 2 mord ({mm.group(1)[:60]}…), nominal re-passe ; requalification archivée"


# ---------------------------------------------------------------- PF4
def pf4():
    txt = need_file("pf4_refus.log").read_text()
    attendus = {"a": "PromptTooLong", "b": "SequenceTooLong", "c": "IgnorePromptWithLoadCache",
                "d": "PromptTooLong", "e": "OraclePromptMismatch"}
    blocs = re.split(r"^### ", txt, flags=re.M)[1:]
    assert len(blocs) == 5, f"{len(blocs)} cas dans le log, 5 attendus"
    for b in blocs:
        m_cas = re.match(r"\(([a-e])\)", b)   # le bloc s'ouvre sur « (a) … », pas sur « a »
        assert m_cas, f"cas non identifiable en tête de bloc : {b[:40]!r}"
        cas = m_cas.group(1)
        rc = re.search(rf"rc\({cas}\)=(\d+)", b)
        assert rc and rc.group(1) != "0", f"cas ({cas}) : rc={rc.group(1) if rc else '?'} — un refus qui n'a pas refusé"
        assert attendus[cas] in b, f"cas ({cas}) : erreur '{attendus[cas]}' absente — crash non qualifié ?"
    return "5/5 refus VUS échouer avec leur erreur nommée"


# ---------------------------------------------------------------- PF5
def pf5():
    for v in ("1280", "4k"):
        t = need_file(f"pf5_{v}.err.log").read_text()
        assert "KVEQ: 32/32 bit-identiques -> PASS" in t, f"variante {v} : 32/32 absent"
        assert "BUILD: mode=ReleaseFast" in t, f"variante {v} : build non ReleaseFast"
    r = need_file("pf5_resume.err.log").read_text()
    assert "KVLOAD:" in r, "reprise simple : KVLOAD absent"
    n_k5 = len([l for l in r.splitlines() if "K5:" in l])
    assert n_k5 == 0, f"reprise simple : {n_k5} ligne(s) K5: — le chemin s'arme sans prompt !"
    assert "--out-ids : 8 ids écrits" in r, "reprise simple : le log --out-ids a changé de forme"
    return "32/32 sur 1280 ET 4k ; reprise simple : 0 ligne K5:, log --out-ids historique"


# ---------------------------------------------------------------- PF6
def pf6():
    rc, out = run([sys.executable, str(SC / "79_t2_render_check.py"),
                   str(need_file("rendu_tour2_hf.json")), str(need_file("pf6.err.log"))])
    assert rc == 0, f"PF6 :\n{out}"
    d = json.loads((EV / "rendu_tour2_hf.json").read_text())
    assert d["history_rewritten"] is False, "le jinja réécrit l'historique — cas C-K5-F, voir la spec"
    assert d["closure_tail_ids"] == [106, 107], f"frontière changée : {d['closure_tail_ids']}"
    return f"{len(d['suffix_ids'])}/{len(d['suffix_ids'])} ids ; frontière closure=[106,107] intacte"


# ---------------------------------------------------------------- contre-épreuves
def selfproofs():
    rc1, o1 = run([sys.executable, str(SC / "81_selfproof_80.py"),
                   str(need_file("pf1.json")), str(need_file("pf1_runB.err.log"))])
    assert rc1 == 0, f"le juge PF1 n'est plus vu condamner :\n{o1}"
    b = re.search(r"BILAN : (\d+)/(\d+)", o1)
    sp = need_file("79_selfproof.log").read_text()
    assert sp.count("PF6 FAIL") >= 2, "la contre-épreuve PF6 archivée ne montre plus de condamnation"
    return f"juge PF1 : {b.group(0)} cas conformes ; juge PF6 : condamnations archivées"


# ---------------------------------------------------------------- tags git
def tags():
    rc, out = run(["git", "tag", "--merged", "HEAD"])
    presents = {t for t in out.splitlines() if t.startswith("gate/pf")}
    attendus = {f"gate/pf{i}-pass" for i in range(7)}
    manquants = attendus - presents
    assert not manquants, f"tags absents ou non ancêtres de HEAD : {sorted(manquants)}"
    return f"7/7 tags gate/pf*-pass, tous ancêtres de HEAD"


gate("PF0  graphe intact", pf0)
gate("PF1  équivalence frontière", lambda: pf_equiv("pf1.json", "pf1_runB.err.log", "PF1"))
gate("PF2  le mordant", pf2)
gate("PF3  fenêtre à travers", lambda: pf_equiv("pf3.json", "pf3_runB.err.log", "PF3", expect_window=True))
gate("PF4  refus bruyants", pf4)
gate("PF5  non-régression", pf5)
gate("PF6  rendu tour 2", pf6)
gate("---  contre-épreuves des juges", selfproofs)
gate("---  tags git", tags)

print("=" * 78)
print("K5 — vérification d'ensemble (preuves versionnées, sans GPU)")
print("=" * 78)
ok = True
for name, passed, detail in results:
    mark = "✅ PASS" if passed else "❌ FAIL"
    ok = ok and passed
    print(f"{mark}  {name:32s} {detail if (passed and VERBOSE) or not passed else ''}")
    if passed and not VERBOSE and detail:
        print(f"          {detail}")
print("=" * 78)
print("VERDICT : les 7 gates tiennent sur les preuves versionnées" if ok
      else "VERDICT : ÉCHEC — au moins un gate ne tient plus")
print("Rejouer les runs eux-mêmes (GPU + M4) : docs/K5_RESULTS.md §9")
sys.exit(0 if ok else 1)
