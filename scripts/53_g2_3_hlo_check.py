#!/usr/bin/env python3
"""G2.3 — check HLO par run : non-vacuité structurelle + anti-câblage-croisé (§5.2/§5.3).

Compare le dump XLA du run (posé par l'orchestrateur via XLA_FLAGS=--xla_dump_to=<dir>) au
dump de D0 (baseline tout-null), sur les modules **PRÉ-optimisation** (`*before_optimizations*`)
— jamais le post-opt : XLA fold les chaînes de converts à l'optimisation, le compte post-opt
ne correspondrait plus à l'oracle des émissions (mismatchs spurieux vs
fixtures/g2_3_expected_converts.json).

Deux preuves, dans l'ordre :
  1. `differs_from_d0` — le graphe HLO du run diffère-t-il structurellement de celui de D0 ?
     Hash du contenu des modules pré-opt, après neutralisation des différences éphémères entre
     deux compilations propres (philosophie E1 `diff -rq`, cf docs/ENGINE_LOG.md l.92 : entre
     deux compiles du même graphe, seuls le chemin de dump — debug_options — et des noms de
     codegen diffèrent). Neutralisé : les chemins des deux dossiers de dump, tout
     `--xla_dump_to=...`, et l'id numérique éphémère du nom de module (`HloModule foo.123`).
  2. `convert_delta_observed` — recensement des ops `convert` du **module principal** (le plus
     gros fichier pré-opt : le runner sweep ne compile QUE `.forward`, mono-graphe qui domine
     tout module auxiliaire) dans le run ET dans D0, par type de transition (f32→bf16,
     bf16→f32, autres) ; delta TOTAL = total_run − total_d0, comparé à la somme (familles
     dédupliquées — cohérent avec parse_families du 52) des `delta_converts_vs_d0` attendus.

Verdicts (contrat FIGÉ, consommé par 52_g2_3_analyze.py::preflight_hlo/evaluate_hlo) :
  IDENTICAL — HLO identique à D0 (le 52 en fait un INVALID structurel : l'arrondi n'a rien
              câblé, cf §5.2/§5.3).
  INVALID   — delta observé != attendu (invalid_reason porte LES DEUX nombres ; le 52 recalcule
              l'attendu de son côté, les deux figurent aussi en clair dans le rapport).
  OK        — differs + delta == attendu.
  Mode dégradé (dernier recours à raison PRÉCISE, jamais un attrape-tout) : `degraded: true` +
  `differs_from_d0` + `degraded_reason`, SANS `convert_delta_observed` — le 52 ne l'accepte
  qu'en multi-familles (§5.3). Cas couverts, chacun documenté dans degraded_reason :
    (a) nombre de modules pré-opt TEXTE différent entre run et D0 (structure de dump
        inattendue : l'appariement du module principal n'est plus fiable) ;
    (b) modules pré-opt présents mais aucun sous forme texte (.txt) — le diff de hash reste
        concluant, le comptage non ;
    (c) fichier pré-opt non-UTF8/corrompu (U+FFFD au décodage — course d'écriture pendant le
        flush XLA ?) : JAMAIS avalé en silence, un IDENTICAL/INVALID calculé sur du texte
        remplacé serait spurieux.
  En dégradé, verdict="OK" si differs (la seule preuve vérifiable a passé), "IDENTICAL" sinon ;
  c'est le flag `degraded` qui porte la limitation, le 52 tranche selon mono/multi.
  Signal NON-bloquant `main_module_ambiguous: true` (+ WARNING) : les 2 plus gros modules
  pré-opt sont à <10 % l'un de l'autre — l'heuristique « module principal = le plus gros »
  mérite un œil au debug, sans changer le verdict.

Exit codes : 0 = rapport rendu (le verdict EST le résultat, même INVALID/IDENTICAL — c'est le
52 qui le consigne au manifest, l'orchestrateur ne doit pas s'arrêter là-dessus) ;
1 = erreur d'usage/structure (famille inconnue, dossier absent, aucun *before_optimizations*).

CLI (appelée par scripts/g2_3_sweep.sh, plan Task 10) :
  python3 scripts/53_g2_3_hlo_check.py \
    --run-dir /data/g2_3_hlo_mlp --d0-dir /data/g2_3_hlo_none --family mlp \
    --expected fixtures/g2_3_expected_converts.json --out /tmp/hlo_g2_3_mlp.json
  Multi-familles : --family "qkv_proj,qk_scores" → attendu = somme des deltas (dédupliqué).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

# Une op convert HLO pré-opt a la forme (dump texte plein, opérandes typés) :
#   %convert.7 = bf16[8,1536]{1,0} convert(f32[8,1536]{1,0} %param.3), metadata={...}
# Nom avec ou sans '%', layout {...} optionnel. `\s+convert\(` exige un blanc devant le nom
# d'op : `bitcast-convert(` (précédé de '-') n'est PAS compté — ce n'est pas un arrondi.
# Groupe 1 = dtype résultat ; groupe 2 = dtype opérande si imprimé (sinon None → "?").
RE_CONVERT = re.compile(
    r"=\s*([a-z][a-z0-9]*)\[[^\]]*\](?:\{[^}]*\})?\s+convert\(\s*(?:([a-z][a-z0-9]*)\[)?")
# Neutralisations pour le hash (différences éphémères entre deux compiles propres, cf E1) :
RE_DUMP_FLAG = re.compile(r"--xla_dump_to=[^\s,\"']+")
RE_MODULE_NAME = re.compile(r"HloModule\s+([^\s,]+)")


# ---------------------------------------------------------------- collecte des dumps

def collect_before_opt(dir_s: str, what: str) -> list[Path]:
    """Fichiers `*before_optimizations*` du dossier (récursif — xla_dump_to est plat, mais on
    ne parie pas dessus). Aucun ⇒ exit 1 EXPLICITE avec ce qui a été trouvé : un dump sans
    pré-opt signifie un XLA_FLAGS mal posé ou une purge prématurée, pas un run à valider."""
    d = Path(dir_s)
    if not d.is_dir():
        sys.exit(f"[erreur] {what}: {d} absent ou pas un dossier")
    files = sorted(p for p in d.rglob("*before_optimizations*") if p.is_file())
    if not files:
        found = sorted(p.name for p in d.rglob("*") if p.is_file())
        sys.exit(f"[erreur] {what}: aucun fichier *before_optimizations* dans {d} — "
                 f"dump post-opt seul ? XLA_FLAGS mal posé ? Fichiers trouvés "
                 f"({len(found)}): {found[:20]}")
    return files


def normalized_hash(path: Path, strip_paths: list[str]) -> tuple[str, bool]:
    """Hash du contenu, différences éphémères neutralisées (fichiers texte seulement — les
    non-texte, ex .pb, sont hashés bruts : pas de chemin normalisable dedans de façon fiable,
    et ils ne portent le chemin de dump que via debug_options texte).
    Retourne (hash, corrompu) : corrompu=True si le décodage UTF-8 a produit un U+FFFD —
    signalé à l'appelant plutôt qu'avalé (un dump lu pendant le flush XLA donnerait sinon
    un IDENTICAL/INVALID spurieux en silence)."""
    raw = path.read_bytes()
    if path.suffix != ".txt":
        return hashlib.sha256(raw).hexdigest(), False
    text = raw.decode("utf-8", errors="replace")
    corrupted = "�" in text
    for p in strip_paths:                     # chemins des deux dossiers de dump (debug_options)
        text = text.replace(p, "<DUMP_DIR>")
    text = RE_DUMP_FLAG.sub("--xla_dump_to=<DUMP_DIR>", text)
    # Id de compilation éphémère du nom de module : si `HloModule foo.123`, le token `foo.123`
    # peut réapparaître dans le corps (ENTRY, computations) — on neutralise TOUTES ses
    # occurrences, pas seulement l'en-tête (sinon deux compiles identiques divergeraient).
    m = RE_MODULE_NAME.search(text)
    if m and (mid := re.match(r"(.+)\.\d+$", m.group(1))):
        text = re.sub(re.escape(m.group(1)) + r"(?![0-9])", mid.group(1) + ".<ID>", text)
    return hashlib.sha256(text.encode()).hexdigest(), corrupted


def pick_main(txt_files: list[Path]) -> tuple[Path, bool]:
    """Module principal = le plus gros fichier pré-opt texte (point de vérité du comptage).
    Tie-break : tri stable sur l'ordre déjà trié de collect_before_opt → déterministe.
    Retourne (module, ambigu) : ambigu=True si le 2e plus gros est à <10 % du 1er —
    l'heuristique « le plus gros » mérite alors un œil (signal, pas un verdict)."""
    ordered = sorted(txt_files, key=lambda p: p.stat().st_size, reverse=True)
    main = ordered[0]
    ambiguous = len(ordered) > 1 and ordered[1].stat().st_size >= 0.9 * main.stat().st_size
    return main, ambiguous


# ---------------------------------------------------------------- comptage des converts

def count_converts(path: Path) -> dict:
    """Recensement des ops `convert` d'un module HLO texte, par transition dtype→dtype."""
    trans: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = RE_CONVERT.search(line)
        if m:
            dst, src = m.group(1), m.group(2) or "?"
            trans[f"{src}->{dst}"] = trans.get(f"{src}->{dst}", 0) + 1
    return {"total": sum(trans.values()), "by_transition": dict(sorted(trans.items())),
            "module": path.name}


# ---------------------------------------------------------------- familles

def parse_families(family_arg: str, expected_path: str) -> tuple[list[str], int, dict]:
    """Familles de --family (séparateur ','), validées contre la table ; attendu = somme des
    deltas, DÉDUPLIQUÉ (ordre préservé) — même règle que parse_families du script 52 :
    'mlp,mlp' ne double-compte pas."""
    ep = Path(expected_path)
    if not ep.exists():
        sys.exit(f"[erreur] --expected: {ep} absent")
    try:
        table = json.loads(ep.read_text())
    except json.JSONDecodeError as e:
        sys.exit(f"[erreur] --expected: json invalide ({e})")
    known = [k for k in table if not k.startswith("_")]
    fams: list[str] = []
    for f in (s.strip() for s in family_arg.split(",")):
        if f not in known:
            sys.exit(f"[erreur] famille inconnue '{f}' dans --family '{family_arg}' "
                     f"(familles: {sorted(known)})")
        if f not in fams:
            fams.append(f)
    expected = sum(int(table[f]["delta_converts_vs_d0"]) for f in fams)
    return fams, expected, table


# ---------------------------------------------------------------- check

def do_check(args) -> int:
    fams, expected, _ = parse_families(args.family, args.expected)
    run_files = collect_before_opt(args.run_dir, "--run-dir")
    d0_files = collect_before_opt(args.d0_dir, "--d0-dir")

    # 1. Diff structurel — multiset des hashes normalisés (l'ordre/le nom de fichier peut
    #    varier d'une compile à l'autre via le compteur module_NNNN, le CONTENU fait foi).
    strip = [str(Path(args.run_dir).resolve()), str(Path(args.d0_dir).resolve()),
             args.run_dir.rstrip("/"), args.d0_dir.rstrip("/")]
    corrupted: list[str] = []
    h_run, h_d0 = [], []
    for files, acc in ((run_files, h_run), (d0_files, h_d0)):
        for p in files:
            h, bad = normalized_hash(p, strip)
            acc.append(h)
            if bad:
                corrupted.append(p.name)
    differs = sorted(h_run) != sorted(h_d0)

    report: dict = {
        "differs_from_d0": differs,
        "convert_delta_expected": expected,
        "families": fams,
        "files_compared": {
            "run": {"dir": args.run_dir, "before_optimizations": [p.name for p in run_files]},
            "d0": {"dir": args.d0_dir, "before_optimizations": [p.name for p in d0_files]},
            "hash_normalization": "chemins de dump + --xla_dump_to + id numérique HloModule "
                                  "neutralisés (méthode E1, docs/ENGINE_LOG.md)",
        },
    }

    # 2. Comptage — module principal = le plus gros fichier pré-opt TEXTE (le sweep ne
    #    compile que .forward : le module principal domine tout module auxiliaire).
    txt_run = [p for p in run_files if p.suffix == ".txt"]
    txt_d0 = [p for p in d0_files if p.suffix == ".txt"]
    degraded_reason = None
    if corrupted:
        # Garde anti-échec-silencieux : comptage ET hash texte non fiables sur du U+FFFD.
        degraded_reason = (f"dump non-UTF8/corrompu : {sorted(corrupted)} — course d'écriture "
                           "pendant le dump ? comptage non fiable, diff de hash indicatif seul")
    elif not txt_run or not txt_d0:
        degraded_reason = ("modules before_optimizations présents mais aucun sous forme texte "
                           f"(.txt) — comptage impossible, diff de hash concluant seul "
                           f"(run: {[p.name for p in run_files]}, d0: {[p.name for p in d0_files]})")
    elif len(txt_run) != len(txt_d0):
        degraded_reason = (f"nombre de modules pré-opt texte différent (run {len(txt_run)} vs "
                           f"D0 {len(txt_d0)}) — structure de dump inattendue, appariement du "
                           f"module principal non fiable ; diff de hash concluant seul")

    counts_run = counts_d0 = None
    delta = None
    warnings: list[str] = []
    if degraded_reason is None:
        main_run, amb_run = pick_main(txt_run)
        main_d0, amb_d0 = pick_main(txt_d0)
        for tag, amb, files in (("run", amb_run, txt_run), ("d0", amb_d0, txt_d0)):
            if amb:
                top2 = sorted(files, key=lambda p: p.stat().st_size, reverse=True)[:2]
                warnings.append(f"WARNING: module principal ambigu côté {tag} — 2 plus gros "
                                "à <10 % : "
                                + ", ".join(f"{p.name} ({p.stat().st_size} o)" for p in top2))
        if amb_run or amb_d0:
            report["main_module_ambiguous"] = True
        counts_run, counts_d0 = count_converts(main_run), count_converts(main_d0)
        delta = counts_run["total"] - counts_d0["total"]
        report["converts_run"] = counts_run
        report["converts_d0"] = counts_d0
        report["files_compared"]["run"]["main_module"] = main_run.name
        report["files_compared"]["d0"]["main_module"] = main_d0.name

    # 3. Verdict — dégradé d'abord (une preuve non fiable ne doit pas produire un
    #    IDENTICAL/INVALID propre en silence), puis identité (§5.2), puis oracle §5.3.
    if degraded_reason is not None:
        report["degraded"] = True
        report["degraded_reason"] = degraded_reason
        # differs → seule la preuve structurelle a passé ; le 52 tranche mono/multi.
        # not differs → le 52 l'invalide de toute façon (differs_from_d0 false).
        report["verdict"] = "OK" if differs else "IDENTICAL"
    elif not differs:
        # Graphes identiques ⇒ comptes identiques par construction : delta 0 en clair.
        report["verdict"] = "IDENTICAL"
        report["convert_delta_observed"] = 0 if delta is None else delta
    elif delta != expected:
        report["convert_delta_observed"] = delta
        report["verdict"] = "INVALID"
        report["invalid_reason"] = (f"delta converts observé {delta} != attendu {expected} "
                                    f"(familles {fams}) — câblage-croisé suspecté, run à corriger")
    else:
        report["convert_delta_observed"] = delta
        report["verdict"] = "OK"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    # Print lisible (style repo) — le rapport json est le contrat, ceci est pour l'humain/log.
    print("=" * 72)
    print(f"G2.3 §5.3 — check HLO : familles {fams} (attendu Δ{expected})")
    print("=" * 72)
    print(f"  run : {args.run_dir} ({len(run_files)} module(s) pré-opt)")
    print(f"  d0  : {args.d0_dir} ({len(d0_files)} module(s) pré-opt)")
    print(f"  differs_from_d0 : {differs}")
    for w in warnings:
        print(f"  ⚠ {w}")
    if counts_run is not None:
        print(f"  converts run : {counts_run['total']} {counts_run['by_transition']} "
              f"[{counts_run['module']}]")
        print(f"  converts d0  : {counts_d0['total']} {counts_d0['by_transition']} "
              f"[{counts_d0['module']}]")
        print(f"  delta observé Δ{delta} vs attendu Δ{expected}")
    if report.get("degraded"):
        print(f"  ⚠ MODE DÉGRADÉ : {degraded_reason}")
    print(f"  verdict : {report['verdict']}"
          + (f" — {report['invalid_reason']}" if "invalid_reason" in report else ""))
    print(f"  rapport → {out}")
    return 0


# ---------------------------------------------------------------- CLI

def main() -> None:
    p = argparse.ArgumentParser(
        description="G2.3 — check HLO par run : non-vacuité + anti-câblage-croisé (cf docstring)")
    p.add_argument("--run-dir", required=True, help="dossier XLA dump du run (--xla_dump_to)")
    p.add_argument("--d0-dir", required=True, help="dossier XLA dump de D0 (baseline tout-null)")
    p.add_argument("--family", required=True,
                   help="famille du run, ou liste 'fam1,fam2' (multi : attendu = somme dédupliquée)")
    p.add_argument("--expected", required=True,
                   help="fixtures/g2_3_expected_converts.json (oracle des deltas)")
    p.add_argument("--out", required=True, help="chemin du rapport json (contrat du script 52)")
    args = p.parse_args()
    sys.exit(do_check(args))


if __name__ == "__main__":
    main()
