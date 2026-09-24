#!/usr/bin/env python3
"""86 — Dépouilleur SD (spec §4.6, §6, §7, §8). Stdlib seule.

Entrées : --manifest, --oracle, --letter r_letter.jsonl, --json r_json.jsonl, --gen-text gen_text.json
(ids→texte du bras A, produit sur la 3090 par `--decode-only`), --logs run_letter.log run_json.log.
Sortie : docs/SD_RESULTS.md + code retour : 0 = tous contrôles PASS ; 1 = au moins un FAIL ;
3 = INEXÉCUTABLE (preuve manquante, témoin rouge…). Jamais PASS par défaut.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

TOL = 0.05


def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


class Verdicts:
    def __init__(self):
        self.rows = []

    def add(self, cid, status, detail):
        assert status in ("PASS", "FAIL", "INEXÉCUTABLE", "DESCRIPTIF")
        self.rows.append((cid, status, detail))

    def code(self):
        s = {r[1] for r in self.rows}
        return 3 if "INEXÉCUTABLE" in s else (1 if "FAIL" in s else 0)


def softmax_option(zc_by_id, label_ids, classes):
    z = [zc_by_id[str(i)] for i in label_ids]
    best = max(range(4), key=lambda k: (z[k], -k))  # égalité → plus petite lettre (comme sd_policy)
    return classes[best]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="fixtures/sd_manifest.json")
    ap.add_argument("--oracle", default="fixtures/sd_oracle.json")
    ap.add_argument("--letter", default="docs/evidence/sd/r_letter.jsonl")
    ap.add_argument("--json", default="docs/evidence/sd/r_json.jsonl")
    ap.add_argument("--gen-text", default="docs/evidence/sd/gen_text.json")
    ap.add_argument("--logs", nargs="+", default=["docs/evidence/sd/run_letter.log", "docs/evidence/sd/run_json.log"])
    ap.add_argument("--out", default="docs/SD_RESULTS.md")
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args(argv)

    v = Verdicts()
    for p in [a.manifest, a.oracle, a.letter, a.json, a.gen_text, *a.logs]:
        if not Path(p).is_file():
            print(f"INEXÉCUTABLE : preuve absente {p}")
            return 3
    man = json.load(open(a.manifest))
    ora = json.load(open(a.oracle))["cases"]
    L = load_jsonl(a.letter)
    J = load_jsonl(a.json)
    gen_text = json.load(open(a.gen_text))
    mcase = {f"{e['case_id']}/{e['perm']}": e for e in man["letter"]}

    # --- C-SD-E : bannière dans chaque log chronométré
    bad = [p for p in a.logs if "BUILD: mode=ReleaseFast" not in open(p, errors="replace").read()]
    v.add("C-SD-E", "INEXÉCUTABLE" if bad else "PASS", f"logs sans bannière : {bad}" if bad else "bannière présente dans tous les logs chronométrés")

    # --- complétude
    fwd = [r for r in L if r["order"] == "fwd" and r["rep"] >= 1]
    rev = [r for r in L if r["order"] == "rev"]
    keys = set(mcase)
    for rep in range(1, a.reps + 1):
        got = {f"{r['case_id']}/{r['perm']}" for r in fwd if r["rep"] == rep}
        if got != keys:
            print(f"INEXÉCUTABLE : rep {rep} incomplète ({len(got)}/{len(keys)})")
            return 3
    if {f"{r['case_id']}/{r['perm']}" for r in rev} != keys:
        print("INEXÉCUTABLE : passe inverse incomplète")
        return 3
    jkeys = {e["case_id"] for e in man["json"]}
    for rep in range(1, a.reps + 1):
        if {r["case_id"] for r in J if r["rep"] == rep} != jkeys:
            print(f"INEXÉCUTABLE : bras JSON rep {rep} incomplète")
            return 3

    # --- C-SD-B : écarts ZML − HF (rep 1)
    max_dz, max_dl, tight = 0.0, 0.0, []
    for r in fwd:
        if r["rep"] != 1:
            continue
        k = f"{r['case_id']}/{r['perm']}"
        o = ora[k]
        for i, zh in o["zc"].items():
            max_dz = max(max_dz, abs(r["zc"][i] - zh))
        max_dl = max(max_dl, abs(r["lse"] - o["lse"]))
        zs = sorted(o["zc"].values(), reverse=True)
        if zs[0] - zs[1] < 0.1:
            tight.append(k)
    ok_b = max_dz <= TOL and max_dl <= TOL
    v.add("C-SD-B", "PASS" if ok_b else "FAIL", f"max|Δzc|={max_dz:.3e}, max|Δlse|={max_dl:.3e} (seuil {TOL}) ; marge HF < 0,1 : {len(tight)} cas {tight[:6]}")

    # --- C-SD-D : step0, témoin rep1/rep2, contrôle rep1/rev (bits)
    if any(r["step0"] != 0 for r in L + J):
        v.add("C-SD-D", "FAIL", "step0 ≠ 0 sur au moins une ligne (remise à 0 de step absente)")
    else:
        by = defaultdict(dict)
        for r in L:
            if r["order"] == "fwd" and r["rep"] in (1, 2):
                by[r["rep"]][f"{r['case_id']}/{r['perm']}"] = (r["zc_bits"], r["lse_bits"])
            elif r["order"] == "rev":
                by["rev"][f"{r['case_id']}/{r['perm']}"] = (r["zc_bits"], r["lse_bits"])
        wit = [k for k in keys if by[1][k] != by[2][k]]
        ctl = [k for k in keys if by[1][k] != by["rev"][k]]
        if wit:
            v.add("C-SD-D", "INEXÉCUTABLE", f"témoin rouge : {len(wit)} cas diffèrent entre rep1 et rep2 (non-déterminisme GPU) {wit[:4]}")
        else:
            v.add("C-SD-D", "FAIL" if ctl else "PASS", f"rep1 vs passe inverse : {len(ctl)} cas diffèrent {ctl[:4]}" if ctl else "bits identiques rep1/rep2 et rep1/inverse sur les 48 prompts")

    # --- C-SD-F (partie dépouillement) : option runner == option recalculée
    mism = []
    for r in fwd + rev:
        m = mcase[f"{r['case_id']}/{r['perm']}"]
        want = softmax_option(r["zc"], m["label_ids"], m["classes"])
        d = r.get("decision") or {}
        if d.get("kind") == "err" or d.get("option") != want:
            mism.append(f"{r['case_id']}/{r['perm']}/rep{r['rep']}:{d.get('option')}≠{want}")
    v.add("C-SD-F(86)", "FAIL" if mism else "PASS", f"{len(mism)} désaccords {mism[:4]}" if mism else "option runner == option recalculée sur toutes les lignes")

    # --- P1
    med_c = st.median([(r["t_read_c_ns"] + r["t_policy_ns"]) for r in fwd]) / 1e6
    med_b = st.median([r["t_read_b_ns"] for r in fwd]) / 1e6
    p1 = abs(med_c - med_b) < 1.0
    v.add("P1", "PASS" if p1 else "FAIL", f"médiane(t_read_C+t_policy)={med_c:.3f} ms, médiane(t_read_B)={med_b:.3f} ms, écart {abs(med_c-med_b):.3f} ms (seuil 1 ms)")

    # --- P2 (population : cas dont le bras A finit par EOT)
    C_ABS, C_GEN = 14.0, 9.0
    tC = defaultdict(list)
    lenL = defaultdict(list)
    for r in fwd:
        tC[r["case_id"]].append((r["t_absorb_ns"] + r["t_read_c_ns"] + r["t_policy_ns"]) / 1e6)
        lenL[r["case_id"]].append(r["n_prompt"])
    tA, nA, lenJ, stops = defaultdict(list), {}, {}, defaultdict(set)
    for r in J:
        if r["rep"] < 1:
            continue
        tA[r["case_id"]].append((r["t_absorb_ns"] + r["t_gen_ns"]) / 1e6)
        nA[r["case_id"]] = r["n_gen"]
        lenJ[r["case_id"]] = r["n_prompt"]
        stops[r["case_id"]].add(r["stop"])
    ngen_all = defaultdict(set)
    for r in J:
        if r["rep"] >= 1:
            ngen_all[r["case_id"]].add(r["n_gen"])
    unstable = [c for c, st_ in ngen_all.items() if len(st_) != 1]
    if unstable:
        print(f"INEXÉCUTABLE : n_gen du bras A varie entre répétitions pour {unstable[:4]}")
        return 3
    pop = sorted(c for c in jkeys if stops[c] == {"eot"})
    G, F, part = {}, {}, {}
    for c in pop:
        G[c] = st.median(tA[c]) - st.median(tC[c])
        F[c] = (lenJ[c] - st.mean(lenL[c])) * C_ABS + (nA[c] - 1) * C_GEN
        part[c] = G[c] / st.median(tA[c])
    if not pop:
        v.add("P2", "INEXÉCUTABLE", "aucun cas du bras A terminé par EOT")
    else:
        a_ok = all(G[c] > 0 for c in pop)
        dev = st.median([abs(G[c] - F[c]) for c in pop])
        b_ok = dev <= max(0.3 * st.median(F.values()), 20.0)
        c_ok = st.median(part.values()) < 0.25
        v.add("P2", "PASS" if (a_ok and b_ok and c_ok) else "FAIL",
              f"population {len(pop)}/{len(jkeys)} ; (a) G>0 partout : {a_ok} ; (b) médiane|G−F|={dev:.1f} ms vs tol {max(0.3*st.median(F.values()),20.0):.1f} : {b_ok} ; (c) médiane part={100*st.median(part.values()):.1f} % : {c_ok}")

    # --- Qualité (descriptif)
    acc = defaultdict(lambda: [0, 0])
    for r in fwd:
        m = mcase[f"{r['case_id']}/{r['perm']}"]
        opt = (r.get("decision") or {}).get("option")
        acc[("C", m["gold"])][0] += opt == m["gold"]
        acc[("C", m["gold"])][1] += 1
        top1 = r["top5_idx"][0]
        lab = m["classes"][m["label_ids"].index(top1)] if top1 in m["label_ids"] else "off_label"
        acc[("B", m["gold"])][0] += lab == m["gold"]
        acc[("B", m["gold"])][1] += 1
    fmt_err = 0
    for r in J:
        if r["rep"] != 1:
            continue
        gold = next(e["gold"] for e in man["json"] if e["case_id"] == r["case_id"])
        txt = gen_text.get(r["case_id"], "")
        try:
            choice = json.loads(txt.strip().strip("`").removeprefix("json").strip())["choice"]
        except Exception:
            choice, fmt_err = None, fmt_err + 1
        acc[("A", gold)][0] += choice == gold
        acc[("A", gold)][1] += 1
    flips = 0
    by_case = defaultdict(dict)
    for r in fwd:
        if r["rep"] == 1:
            by_case[r["case_id"]][r["perm"]] = (r.get("decision") or {}).get("option")
    flips = sum(1 for c in by_case.values() if c.get("orig") != c.get("rev"))
    mass = sorted((r.get("decision") or {}).get("mass_in", float("nan")) for r in fwd if r["rep"] == 1)
    v.add("Qualité", "DESCRIPTIF", f"bascules orig↔rev (bras C) : {flips}/24 ; format_error bras A : {fmt_err}/24 ; mass_in min/méd/max = {mass[0]:.3g}/{st.median(mass):.3g}/{mass[-1]:.3g}")

    # --- Descriptifs exigés par la spec (§4.4, §6, §8, §9) — aucun n'influe sur un verdict
    def p95(xs):
        xs = sorted(xs)
        return xs[min(len(xs) - 1, math.ceil(0.95 * len(xs)) - 1)]
    tB = [(r["t_absorb_ns"] + r["t_read_b_ns"]) / 1e6 for r in fwd]
    tCl = [(r["t_absorb_ns"] + r["t_read_c_ns"] + r["t_policy_ns"]) / 1e6 for r in fwd]
    tAl = [(r["t_absorb_ns"] + r["t_gen_ns"]) / 1e6 for r in J if r["rep"] >= 1]
    desc = [f"bras A : médiane {st.median(tAl):.1f} ms, p95 {p95(tAl):.1f} ms",
            f"bras B : médiane {st.median(tB):.1f} ms, p95 {p95(tB):.1f} ms",
            f"bras C : médiane {st.median(tCl):.1f} ms, p95 {p95(tCl):.1f} ms"]
    c_abs_m = st.median([r["t_absorb_ns"] / 1e6 / r["n_prompt"] for r in fwd])
    c_gen_m = st.median([r["t_gen_ns"] / 1e6 / (r["n_gen"] - 1) for r in J if r["rep"] >= 1 and r["n_gen"] > 1])
    desc.append(f"coûts MESURÉS : c_abs = {c_abs_m:.2f} ms/pas (publié 14), c_gen = {c_gen_m:.2f} ms/pas (publié 9)")
    desc.append(f"t_reset médian {st.median([r['t_reset_ns'] for r in fwd])/1e6:.2f} ms (hors temps de décision)")
    warm = [r for r in L if r["order"] == "warmup"]
    if warm:
        desc.append(f"échauffement (froid) : absorption {warm[0]['t_absorb_ns']/1e6:.1f} ms sur {warm[0]['n_prompt']} pas")
    for lg in a.logs:
        for line in open(lg, errors="replace"):
            # seule la ligne « compile: <durée> » ; « [mem] post-compile: RSS=… » n'est pas un temps de compile
            body = line.split("info:", 1)[-1].strip()
            if body.startswith("compile:"):
                desc.append(f"compile ({Path(lg).name}) : {body.split('compile:', 1)[1].strip()}")
    off = sum(1 for r in fwd if r["top5_idx"][0] not in mcase[f"{r['case_id']}/{r['perm']}"]["label_ids"])
    desc.append(f"bras B hors étiquettes : {off}/{len(fwd)}")
    theta_rows = []
    for th in (0.0, 0.1, 0.2, 0.4, 0.6, 0.8):
        kept = [r for r in fwd if (r.get("decision") or {}).get("margin", 0) >= th]
        err = sum(1 for r in kept if (r["decision"]["option"] != mcase[f"{r['case_id']}/{r['perm']}"]["gold"]))
        theta_rows.append(f"| {th} | {len(kept)}/{len(fwd)} | {err}/{len(kept) if kept else 0} |")
    thinking = man["meta"].get("enable_thinking_passed")
    desc.append(f"rendu du template : enable_thinking {'passé à False' if thinking else 'ABSENT du template (rendu sans le paramètre)'}")

    # --- rapport
    lines = ["# SD — Résultats (généré par scripts/86_sd_report.py, ne pas éditer à la main)", "",
             "| Contrôle | Verdict | Détail |", "|---|---|---|"]
    lines += [f"| {c} | **{s}** | {d} |" for c, s, d in v.rows]
    lines += ["", "## Exactitude par bras et par classe", "", "| Bras | Classe | Justes / total |", "|---|---|---|"]
    for (arm, cl), (ok, n) in sorted(acc.items()):
        lines.append(f"| {arm} | {cl} | {ok}/{n} |")
    lines += ["", "## Descriptifs", ""] + [f"- {d}" for d in desc]
    lines += ["", "## Couverture / erreur du bras C en fonction de θ (descriptif, spec §4.4)", "",
              "| θ | décisions gardées | erreurs parmi gardées |", "|---|---|---|"] + theta_rows
    lines += ["", "## Gain par cas (bras A − bras C, ms)", "", "| Cas | len JSON | len lettre (moy.) | n_gen A | G mesuré | F prédit | part de t_A |", "|---|---|---|---|---|---|---|"]
    for c in pop:
        lines.append(f"| {c} | {lenJ[c]} | {st.mean(lenL[c]):.1f} | {nA[c]} | {G[c]:.1f} | {F[c]:.1f} | {100*part[c]:.1f} % |")
    Path(a.out).write_text("\n".join(lines) + "\n")
    for c, s, d in v.rows:
        print(f"{c:12s} {s:13s} {d}")
    return v.code()


if __name__ == "__main__":
    raise SystemExit(main())
