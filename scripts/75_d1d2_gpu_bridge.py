#!/usr/bin/env python3
"""Dépouillement des gates G-D0 / G-D1 / G-D2 — couverture GPU de `applyTopP` et
`applyTemperature` (dettes D1/D2). Spec : `docs/superpowers/specs/2026-08-10-d1d2-gpu-coverage.md`.

POURQUOI UN DÉPOUILLEUR SÉPARÉ. Le binaire publie des compteurs BRUTS et ne rend aucun verdict :
le juge est ici, avec les seuils pré-enregistrés de la spec. Un binaire qui s'auto-déclarerait
PASS mélangerait la mesure et son interprétation.

Usage :
  75_d1d2_gpu_bridge.py --d1-log F --d2-log F --hlo-witness-md5 M --hlo-after-md5 M
Sortie : un verdict par gate, puis exit 0 si les TROIS passent, 1 sinon.
"""
import argparse
import re
import sys

# Seuils PRÉ-ENREGISTRÉS (spec §4). Ils ne se règlent pas après coup sur les chiffres obtenus.
MIN_STEPS = 300          # G-D1 : taille minimale de l'échantillon
MIN_STEPS_WITH_CUT = 30  # G-D1 : antécédent non vide — sinon le gate est passé À VIDE
MIN_TEMP_APPLIED = 300   # G-D2 (i) : la ligne s'exécute réellement sur GPU

RE_D1 = re.compile(
    r"G-D1: steps=(\d+) désaccords=(\d+) 1er_id_en_désaccord=(-?\d+) \| "
    r"ANTÉCÉDENT steps_avec_coupe=(\d+) ids_coupés=(\d+) \| "
    r"frontière_serrée=(\d+) ex_æquo_frontière=(\d+)"
)
RE_D2 = re.compile(
    r"G-D2: temp_appliquée=(\d+) steps \| mutant_a_division_vs_mul=(\d+) logits \| "
    r"mutant_b_ordre=(\d+) ids sur (\d+) steps"
)
RE_ALLOC = re.compile(r"ALLOC-LOOP: alloc=(\d+) resize=(\d+) remap=(\d+) free=(\d+) bytes=(\d+)")
RE_BUILD = re.compile(r"BUILD: mode=(\w+)")


def read(path: str) -> str:
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def need(rx: re.Pattern, text: str, path: str, what: str):
    """Refus BRUYANT : une ligne absente est un dépouillement impossible, jamais un zéro."""
    m = rx.search(text)
    if m is None:
        sys.exit(f"FATAL : ligne « {what} » introuvable dans {path} — le run a-t-il abouti ? "
                 f"(le gate ne peut pas conclure sur une mesure absente)")
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d1-log", required=True, help="stderr du run G-D1 (top_k/top_p armés, T=1.0)")
    ap.add_argument("--d2-log", required=True, help="stderr du run G-D2 (T=0.7)")
    ap.add_argument("--hlo-witness-md5", required=True, help="md5 du HLO before_optimizations AVANT le pont")
    ap.add_argument("--hlo-after-md5", required=True, help="md5 du HLO before_optimizations APRÈS le pont")
    a = ap.parse_args()

    d1_txt, d2_txt = read(a.d1_log), read(a.d2_log)
    verdicts = {}

    # --- G-D0 : le graphe n'a pas bougé, et l'interdit D10 tient ------------------------------
    hlo_ok = a.hlo_witness_md5.strip().lower() == a.hlo_after_md5.strip().lower()
    alloc_lines = []
    for name, txt in (("d1", d1_txt), ("d2", d2_txt)):
        m = need(RE_ALLOC, txt, name, "ALLOC-LOOP")
        alloc_lines.append((name, tuple(int(g) for g in m.groups())))
    alloc_ok = all(all(v == 0 for v in vals) for _, vals in alloc_lines)
    modes = [need(RE_BUILD, t, n, "BUILD: mode=").group(1) for n, t in (("d1", d1_txt), ("d2", d2_txt))]
    mode_ok = all(m == "ReleaseFast" for m in modes)
    verdicts["G-D0"] = hlo_ok and alloc_ok and mode_ok
    print("=== G-D0 — le graphe n'a pas bougé, l'interdit D10 tient ===")
    print(f"  md5 HLO avant  : {a.hlo_witness_md5}")
    print(f"  md5 HLO après  : {a.hlo_after_md5}")
    print(f"  identiques     : {'OUI' if hlo_ok else 'NON  <-- le pont a bougé le graphe'}")
    for name, vals in alloc_lines:
        print(f"  ALLOC-LOOP {name} : alloc={vals[0]} resize={vals[1]} remap={vals[2]} free={vals[3]} bytes={vals[4]}")
    print(f"  mode de build  : {', '.join(modes)}")
    print(f"  G-D0 : {'PASS' if verdicts['G-D0'] else 'FAIL'}\n")

    # --- G-D1 : applyTopP == référence descendante f64, sur antécédent non vide ---------------
    g = need(RE_D1, d1_txt, a.d1_log, "G-D1").groups()
    steps, disagree, first_bad, with_cut, cut_total, tight, ties = (int(x) for x in g)
    enough = steps >= MIN_STEPS
    antecedent = with_cut >= MIN_STEPS_WITH_CUT
    verdicts["G-D1"] = enough and antecedent and disagree == 0
    print("=== G-D1 — applyTopP contre une référence écrite AUTREMENT (tri descendant, f64) ===")
    print(f"  steps comparés          : {steps}   (seuil {MIN_STEPS}) {'OK' if enough else 'INSUFFISANT'}")
    print(f"  désaccords              : {disagree}   (prédiction : 0)")
    if disagree:
        print(f"  1er id en désaccord     : {first_bad}   <-- bug possible de applyTopP, jamais exercé sur GPU")
    print(f"  ANTÉCÉDENT : steps où top-p coupe : {with_cut} (seuil {MIN_STEPS_WITH_CUT}) "
          f"{'OK' if antecedent else 'VIDE — le gate ne prouve RIEN, changer top_p/prompt'}")
    print(f"  ids coupés au total     : {cut_total}")
    print(f"  frontière serrée (|cum-p|<1e-9) : {tight}   ex æquo à la frontière : {ties}")
    if disagree and tight:
        print("  ⚠ des désaccords ET des cas frontière : vérifier s'ils coïncident avant de "
              "conclure à un bug structurel (le bruit f32/f64 est une explication concurrente)")
    print(f"  G-D1 : {'PASS' if verdicts['G-D1'] else 'FAIL'}\n")

    # --- G-D2 : la température s'exécute, divise, et l'ordre de la chaîne compte ---------------
    g2 = need(RE_D2, d2_txt, a.d2_log, "G-D2").groups()
    temp_applied, mut_a, mut_b_ids, mut_b_steps = (int(x) for x in g2)
    executed = temp_applied >= MIN_TEMP_APPLIED
    # Les MUTANTS doivent MORDRE : un mutant qui ne mord pas ne prouve rien (et le dire).
    mut_a_bites = mut_a > 0
    mut_b_bites = mut_b_ids > 0
    verdicts["G-D2"] = executed and mut_a_bites and mut_b_bites
    print("=== G-D2 — applyTemperature exercée sur GPU, + 2 mutants qui doivent MORDRE ===")
    print(f"  (i)  steps où la température s'exécute : {temp_applied} (seuil {MIN_TEMP_APPLIED}) "
          f"{'OK' if executed else 'VACUITÉ — la ligne n a pas tourné'}")
    print(f"  (ii) mutant a — x/t contre x*(1/t)     : {mut_a} logits diffèrent "
          f"{'-> le gate DISCRIMINE la division' if mut_a_bites else '-> NE MORD PAS : G-D2 ne prouve rien sur la division'}")
    print(f"  (iii) mutant b — ordre TopK->TopP->Temp : {mut_b_ids} ids sur {mut_b_steps} steps "
          f"{'-> l ORDRE de la chaîne a un effet observable' if mut_b_bites else '-> NE MORD PAS : G-D2 ne prouve pas l ordre'}")
    print(f"  G-D2 : {'PASS' if verdicts['G-D2'] else 'FAIL'}\n")

    print("===== VERDICT =====")
    for k, v in verdicts.items():
        print(f"  {k} : {'PASS' if v else 'FAIL'}")
    ok = all(verdicts.values())
    print("TOUS VERTS" if ok else "AU MOINS UN GATE ROUGE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
