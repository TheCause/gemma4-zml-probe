#!/usr/bin/env python3
"""G2.3 — Analyse par run du sweep de sensibilité bf16 par-op (docs/G2_3_OP_SENSITIVITY.md).

Généralise 51_g2_2_analyze.py : pour chaque run Dᵢ du sweep (dump logits .bin f32 brut),
applique les gates PRÉ-ENREGISTRÉS §5 dans l'ordre — custody → sanité → non-vacuité →
anti-câblage-croisé (delta converts HLO, rapport du script 53) → métriques vs D0 ET vs A →
verdict bucket (SAFE ≤1× / TOLERABLE ≤2× / SENSITIVE >2× de l'enveloppe B sur KL p50 vs A,
départage pré-enregistré : le pire de {KL p50, max_abs p50}) — puis UPSERT de l'entrée au
manifest (clé = run-name, compteur re_run + historique md5, jamais de doublon silencieux)
et npz de courbes par-step `<run-name>_metrics.npz` (ce qui survit à la purge des dumps).

Formats hétérogènes des références (protocole §2 — piège capital) :
  A  = `.npy` memmap f32 [N, 262144] (np.load mmap_mode='r') — JAMAIS lu comme du brut.
  B  = `.npy` uint16 = bit-patterns bf16 (calibration seule) — réinterprété .view(bfloat16).
  D* = `.bin` f32 brut [steps × 262144] (np.memmap + reshape).
Le loader choisit par extension.

KL — convention pré-enregistrée §4 (= ligne 65 du script 51, qui fait foi) : KL(ref‖run) =
(ls_ref.exp() * (ls_ref - ls_run)).sum(), ref ∈ {A, D0} → kl_vs_A = KL(A‖Dᵢ), kl_vs_D0 = KL(D0‖Dᵢ).
Source normative des seuils : valeurs NON-ARRONDIES du manifest --envelope (§4).

Modes :
  (défaut)              analyse complète d'un run one-hot / combiné.
  --ref-a none          runs S49 : métriques vs D0 seul, verdict "diagnostic-only", pas de
                        buckets (pas d'enveloppe S49), sanité INFORMATIVE (loggée, non bloquante).
  --register-reference  run D0 : custody (md5 A si fourni + D0) + sanité de D0 + entrée de type
                        référence — PAS d'auto-analyse vs soi-même, pas de verdict bucket.
  --calibrate-sanity    calcule les seuils de sanité §5.1 depuis les memmaps A ET B → manifest.
  --selfcheck           rejoue les ratios publiés G2.2 depuis fixtures/ (numpy pur, M1-ok) —
                        garde anti-régression de formule.

CLI (appelé par scripts/g2_3_sweep.sh, cf plan Task 10) :
  python3 scripts/52_g2_3_analyze.py \
    --run-logits <D.bin> --run-name <nom déjà préfixé> \
    --ref-a /data/gemma4-zml-probe/g2_logits_a_f32.npy --ref-d0 <D0.bin> \
    --envelope fixtures/g2_envelope_manifest.json \
    --expected-converts fixtures/g2_3_expected_converts.json --hlo-report <json du 53> \
    --manifest fixtures/g2_3_manifest.json --bin-sha <sha> --zml-rev <rev> --repo-rev <rev>

Exit codes : 0 = analyse rendue (le verdict EST le résultat, même FAIL-SANITY/VACUOUS/INVALID) ;
2 = custody REFUSÉE (référence au md5 non conforme — régénérer, jamais réutiliser) ;
1 = erreur d'usage/config (dont seuils de sanité absents hors --calibrate-sanity).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

VOC = 262144                                 # vocabulaire Gemma 4 (invariant projet, cf script 50)
REF_B_DEFAULT = "/data/gemma4-zml-probe/g2_logits_b_bf16u16.npy"
# Garde du selfcheck : ratios G2.2 publiés (fixtures/g2_2_manifest.json, arrondis protocole).
SELFCHECK_PUBLISHED = {"max_abs_p50": 0.44, "kl_p50": 0.28}
SELFCHECK_TOL = 0.01
BUCKET_ORDER = {"SAFE": 0, "TOLERABLE": 1, "SENSITIVE": 2}

# torch importé PARESSEUSEMENT : le selfcheck tourne en numpy pur (M1 sans torch OK).
_torch = None
_device = None


def torch_mod():
    global _torch, _device
    if _torch is None:
        import torch
        _torch = torch
        _device = "cuda" if torch.cuda.is_available() else "cpu"
    return _torch, _device


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- formules (identiques au 51)

def pct(a) -> dict:
    return {"p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)), "max": float(a.max())}


def ratios(d: dict, env: dict) -> dict:
    """Ratios run/enveloppe sur p50/p95/max — enveloppe NON-ARRONDIE du manifest (normative, §4)."""
    return {q: (d[q] / env[q] if env[q] > 0 else float("inf")) for q in ("p50", "p95", "max")}


def bucket(r: float) -> str:
    return "SAFE" if r <= 1.0 else ("TOLERABLE" if r <= 2.0 else "SENSITIVE")


# ---------------------------------------------------------------- loaders (choix par extension)

def open_logits(path_s: str, what: str):
    """A = .npy memmap f32 (np.load) ; D0/Dᵢ = .bin f32 brut (np.memmap). JAMAIS l'inverse :
    lire le .npy comme du brut décalerait tout du header numpy (métriques poubelle)."""
    path = Path(path_s)
    if not path.exists():
        sys.exit(f"[erreur] {what}: {path} absent")
    if path.suffix == ".npy":
        arr = np.load(path, mmap_mode="r")
        if arr.dtype != np.float32 or arr.ndim != 2 or arr.shape[1] != VOC:
            sys.exit(f"[erreur] {what}: .npy inattendu (dtype={arr.dtype}, shape={arr.shape}) — "
                     f"attendu f32 [N,{VOC}] (bras A, script 50)")
        return arr
    raw = np.memmap(path, dtype=np.float32, mode="r")
    if raw.size % VOC:
        sys.exit(f"[erreur] {what}: taille {raw.size} non multiple de VOC {VOC} — pas un dump f32 brut ?")
    return raw.reshape(-1, VOC)


def row_f32(arr, k, torch, device):
    return torch.from_numpy(np.asarray(arr[k])).to(device)


def row_b_u16(arr, k, torch, device):
    """Bras B : bit-patterns bf16 stockés en uint16 — réinterprétation OBLIGATOIRE (§2)."""
    u16 = np.asarray(arr[k])
    try:
        t = torch.from_numpy(u16.copy())
    except TypeError:  # torch sans support uint16 : mêmes bits via int16
        t = torch.from_numpy(u16.view(np.int16).copy())
    return t.view(torch.bfloat16).float().to(device)


# ---------------------------------------------------------------- passes streaming (step par step)

def max_run_length(a) -> int:
    if len(a) == 0:
        return 0
    best = cur = 1
    for i in range(1, len(a)):
        cur = cur + 1 if a[i] == a[i - 1] else 1
        best = max(best, cur)
    return best


def scan_pass(arr, n_steps: int, label: str, row_fn=row_f32) -> dict:
    """Sanité §5.1 / calibration : finitude, entropie par step, argmax par step (streaming)."""
    torch, device = torch_mod()
    entropy = np.zeros(n_steps, dtype=np.float32)
    am = np.full(n_steps, -1, dtype=np.int64)
    nf_steps = 0
    nf_elems = 0
    for k in range(n_steps):
        x = row_fn(arr, k, torch, device)
        bad = int((~torch.isfinite(x)).sum())
        if bad:
            nf_steps += 1
            nf_elems += bad
            entropy[k] = np.nan
            continue
        ls = torch.log_softmax(x, dim=-1)
        entropy[k] = float(-(ls.exp() * ls).sum())
        am[k] = int(x.argmax())
        if (k + 1) % 256 == 0:
            print(f"    {label} {k + 1}/{n_steps}")
    ent = entropy[np.isfinite(entropy)]
    return {"nonfinite_steps": nf_steps, "nonfinite_elems": nf_elems,
            "entropy": entropy, "argmax": am,
            "entropy_mean": float(ent.mean()) if ent.size else float("nan"),
            "entropy_std": float(ent.std()) if ent.size else float("nan"),
            "argmax_repeat_max": max_run_length(am)}


def metrics_pass(ref, run, n_steps: int, label: str) -> dict:
    """Formules du 51 : max_abs, KL(ref‖run) = (ls_ref.exp()*(ls_ref-ls_run)).sum(), match argmax."""
    torch, device = torch_mod()
    max_abs = np.zeros(n_steps, dtype=np.float32)
    kl = np.zeros(n_steps, dtype=np.float32)
    match = np.zeros(n_steps, dtype=bool)
    for k in range(n_steps):
        r = row_f32(ref, k, torch, device)
        d = row_f32(run, k, torch, device)
        max_abs[k] = float((d - r).abs().max())
        lsr = torch.log_softmax(r, dim=-1)
        lsd = torch.log_softmax(d, dim=-1)
        kl[k] = float((lsr.exp() * (lsr - lsd)).sum())
        match[k] = bool(int(d.argmax()) == int(r.argmax()))
        if (k + 1) % 256 == 0:
            print(f"    {label} {k + 1}/{n_steps}")
    n_mism = int(n_steps - match.sum())
    first = int(np.argmin(match)) if n_mism else -1
    return {"max_abs": max_abs, "kl": kl, "match": match,
            "mismatches": n_mism, "first_bifurcation": first}


# ---------------------------------------------------------------- manifest, custody, upsert

def load_manifest(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"_meta": {"source": "G2.3 manifest par run (docs/G2_3_OP_SENSITIVITY.md §4-§5)",
                      "created": now()},
            "custody": {}, "sanity_thresholds": None, "runs": {}}


def save_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def custody_check(manifest: dict, manifest_path: Path, path_s: str, role: str, prov: dict) -> str:
    """Chaîne de custody §2 : 1re fois = enregistre (md5+taille+params) ; ensuite REFUSE au
    mismatch (exit 2) — au moindre doute, RÉGÉNÉRER (script 50 pour A/B, run `none` pour D0)."""
    path = Path(path_s)
    if not path.exists():
        sys.exit(f"[erreur] custody {role}: {path} absent")
    m = md5_file(path)
    cust = manifest.setdefault("custody", {})
    key = str(path)
    if key in cust:
        if cust[key]["md5"] != m:
            print("=" * 72)
            print(f"CUSTODY REFUSÉE — {role} = {path}")
            print(f"  md5 enregistré : {cust[key]['md5']} ({cust[key].get('registered_at', '?')})")
            print(f"  md5 observé    : {m}")
            print("  La référence a changé depuis son enregistrement : TOUT verdict serait")
            print("  incomparable. RÉGÉNÉRER (script 50 pour A/B, run `none` pour D0) plutôt")
            print("  que réutiliser (protocole §2), puis ré-enregistrer.")
            sys.exit(2)
        print(f"  custody {role:6s} OK  : {path.name} md5={m}")
    else:
        cust[key] = {"role": role, "md5": m, "size_bytes": path.stat().st_size,
                     "registered_at": now(), "params": prov}
        save_manifest(manifest_path, manifest)  # persiste immédiatement l'enregistrement
        print(f"  custody {role:6s} REG : {path.name} md5={m} (1re fois — enregistré)")
    return m


def upsert_run(manifest: dict, run_name: str, entry: dict, dump_md5: str) -> None:
    """UPSERT clé = run-name : ré-analyse → compteur re_run + md5 des passages, jamais de
    doublon silencieux."""
    runs = manifest.setdefault("runs", {})
    prev = runs.get(run_name)
    if prev is not None:
        entry["re_run"] = int(prev.get("re_run", 0)) + 1
        hist = prev.get("md5_history") or ([prev["dump_md5"]] if prev.get("dump_md5") else [])
        entry["md5_history"] = list(hist) + [dump_md5]
        entry["previous_verdict"] = prev.get("verdict")
        print(f"  [upsert] ré-analyse de '{run_name}' (re_run={entry['re_run']}, "
              f"md5 précédents: {hist})")
    else:
        entry["re_run"] = 0
        entry["md5_history"] = [dump_md5]
    runs[run_name] = entry


# ---------------------------------------------------------------- anti-câblage-croisé (§5.3)

def parse_families(run_name: str, known: list[str]) -> list[str]:
    """Extrait les familles d'un run-name préfixé par l'orchestrateur (ex 'g2_3_s49_norms+mlp').
    1er segment : suffixe famille (le + long qui matche) ; suivants : noms exacts."""
    segs = run_name.split("+")
    cand = [f for f in known if segs[0] == f or segs[0].endswith("_" + f)]
    if not cand:
        sys.exit(f"[erreur] impossible d'extraire une famille connue de '{segs[0]}' "
                 f"(familles: {sorted(known)}) — run-name mal formé ?")
    fams = [max(cand, key=len)]
    for s in segs[1:]:
        if s not in known:
            sys.exit(f"[erreur] famille inconnue '{s}' dans run-name '{run_name}'")
        fams.append(s)
    return fams


def evaluate_hlo(args) -> tuple[dict, bool]:
    """Gate §5.3 : delta de converts observé (rapport du script 53) == somme des deltas attendus
    des familles actives (fixtures/g2_3_expected_converts.json). Mismatch ⇒ INVALID."""
    if not args.expected_converts or not args.hlo_report:
        print("  ⚠ gate anti-câblage-croisé NON évaluée (--expected-converts/--hlo-report absents)")
        return {"status": "not-provided",
                "note": "gate §5.3 non évaluée — hlo-report/expected-converts absents"}, False
    exp = json.loads(Path(args.expected_converts).read_text())
    known = [k for k in exp if not k.startswith("_")]
    fams = parse_families(args.run_name, known)
    expected = sum(int(exp[f]["delta_converts_vs_d0"]) for f in fams)
    rep = json.loads(Path(args.hlo_report).read_text())
    differs = bool(rep.get("differs_from_d0"))
    observed = rep.get("convert_delta_observed", rep.get("convert_ops_observed"))
    degraded = bool(rep.get("degraded")) or observed is None
    check = {"families": fams, "expected_delta": expected, "observed_delta": observed,
             "differs_from_d0": differs, "report": str(args.hlo_report),
             "report_verdict": rep.get("verdict")}
    if not differs:
        check["invalid_reason"] = "graphe HLO identique à D0 (câblage-croisé : la famille n'a rien changé)"
        return check, True
    if rep.get("verdict") == "INVALID":
        check["invalid_reason"] = f"verdict INVALID du script 53 : {rep}"
        return check, True
    if degraded:
        # §5.3 : si la somme n'est pas vérifiable (multi-familles), 53 dégrade en
        # differs_from_d0 seul — consigné, pas INVALID.
        check["degraded"] = "differs_from_d0 seul (§5.3, comptage non vérifiable)"
        return check, False
    if int(observed) != expected:
        check["invalid_reason"] = (f"delta converts observé {observed} != attendu {expected} "
                                   f"(familles {fams})")
        return check, True
    return check, False


# ---------------------------------------------------------------- modes

def do_selfcheck() -> int:
    """Garde anti-régression de formule : rejoue les percentiles/ratios G2.2 depuis les fixtures
    du repo avec LES fonctions de ce script (numpy pur — tourne sur M1 sans torch ni gros dumps)."""
    fx = Path(__file__).resolve().parent.parent / "fixtures"
    npz_p, env_p, g22_p = fx / "g2_2_metrics.npz", fx / "g2_envelope_manifest.json", fx / "g2_2_manifest.json"
    for p in (npz_p, env_p):
        if not p.exists():
            print(f"SELFCHECK FAIL — fixture absente : {p}")
            return 1
    d = np.load(npz_p)
    env = json.loads(env_p.read_text())
    r_ma = ratios(pct(d["max_abs"]), env["max_abs"])
    r_kl = ratios(pct(d["kl"]), env["kl_a_b"])
    print("=" * 72)
    print("SELFCHECK — reproduction des ratios G2.2 (g2_2_metrics.npz vs enveloppe)")
    print("=" * 72)
    print(f"  ratios max_abs : {r_ma}")
    print(f"  ratios KL      : {r_kl}")
    ok_ma = abs(r_ma["p50"] - SELFCHECK_PUBLISHED["max_abs_p50"]) <= SELFCHECK_TOL
    ok_kl = abs(r_kl["p50"] - SELFCHECK_PUBLISHED["kl_p50"]) <= SELFCHECK_TOL
    ok_bucket = bucket(r_kl["p50"]) == "SAFE" and bucket(r_ma["p50"]) == "SAFE" \
        and bucket(1.5) == "TOLERABLE" and bucket(2.5) == "SENSITIVE"
    ok_exact = True
    if g22_p.exists():  # garde secondaire : reproduction exacte des ratios publiés au manifest G2.2
        pub = json.loads(g22_p.read_text())["ratios_vs_envelope"]
        ok_exact = all(abs(r_ma[q] - pub["max_abs"][q]) < 1e-6 and abs(r_kl[q] - pub["kl"][q]) < 1e-6
                       for q in ("p50", "p95", "max"))
        print(f"  reproduction exacte vs g2_2_manifest.json : {'OK' if ok_exact else 'ÉCART'}")
    if ok_ma and ok_kl and ok_bucket and ok_exact:
        print(f"\nSELFCHECK PASS — ratios G2.2 reproduits (max_abs p50 {r_ma['p50']:.2f}x, "
              f"KL p50 {r_kl['p50']:.2f}x, tolérance ±{SELFCHECK_TOL}x)")
        return 0
    print(f"\nSELFCHECK FAIL — attendu max_abs p50 {SELFCHECK_PUBLISHED['max_abs_p50']}±{SELFCHECK_TOL}, "
          f"KL p50 {SELFCHECK_PUBLISHED['kl_p50']}±{SELFCHECK_TOL} "
          f"(obtenu {r_ma['p50']:.4f} / {r_kl['p50']:.4f} ; buckets {'OK' if ok_bucket else 'KO'})")
    return 1


def do_calibrate(args) -> int:
    """§5.1 : seuils de sanité depuis les memmaps A (f32) ET B (u16→bf16) — AVANT le sweep.
    Entropie : borne basse = min(moyennes A,B) − 3σ (σ = max des écarts-types par step).
    Répétition argmax : max observé sur A/B avec marge (×2, plancher +8). NaN/Inf : 0 toléré."""
    if not args.ref_a or args.ref_a.lower() == "none" or not args.manifest:
        sys.exit("[erreur] --calibrate-sanity exige --ref-a (réel) et --manifest")
    mpath = Path(args.manifest)
    manifest = load_manifest(mpath)
    prov = {"bin_sha": args.bin_sha, "zml_rev": args.zml_rev, "repo_rev": args.repo_rev}
    print("Calibration sanité — memmaps A (f32) et B (bf16-as-u16)")
    custody_check(manifest, mpath, args.ref_a, "ref_a", prov)
    a = open_logits(args.ref_a, "ref A")
    b_path = Path(args.ref_b)
    if not b_path.exists():
        sys.exit(f"[erreur] ref B absent : {b_path} (relancer scripts/50_bf16_envelope_oracle.py)")
    custody_check(manifest, mpath, str(b_path), "ref_b", prov)
    b = np.load(b_path, mmap_mode="r")
    if b.dtype != np.uint16:
        sys.exit(f"[erreur] ref B: dtype {b.dtype} != uint16 (bit-patterns bf16 attendus, script 50)")
    sa = scan_pass(a, a.shape[0], "calib A", row_f32)
    sb = scan_pass(b, b.shape[0], "calib B", row_b_u16)
    if sa["nonfinite_elems"] or sb["nonfinite_elems"]:
        sys.exit(f"[erreur] NaN/Inf dans les références (A:{sa['nonfinite_elems']}, "
                 f"B:{sb['nonfinite_elems']}) — références corrompues, régénérer (script 50)")
    ent_min = min(sa["entropy_mean"], sb["entropy_mean"]) - 3.0 * max(sa["entropy_std"], sb["entropy_std"])
    rep_obs = max(sa["argmax_repeat_max"], sb["argmax_repeat_max"])
    rep_max = max(2 * rep_obs, rep_obs + 8)
    manifest["sanity_thresholds"] = {
        "nan_inf_tolerated": 0,
        "entropy_mean_min": float(ent_min),
        "argmax_repeat_max": int(rep_max),
        "calibration": {
            "ref_a": str(args.ref_a), "ref_b": str(b_path),
            "entropy_mean_a": sa["entropy_mean"], "entropy_std_a": sa["entropy_std"],
            "entropy_mean_b": sb["entropy_mean"], "entropy_std_b": sb["entropy_std"],
            "argmax_repeat_a": sa["argmax_repeat_max"], "argmax_repeat_b": sb["argmax_repeat_max"],
            "formula_entropy": "min(mean_A, mean_B) - 3*max(std_A, std_B)",
            "formula_repeat": "max(2*max_obs(A,B), max_obs+8)",
            "calibrated_at": now(),
        },
    }
    save_manifest(mpath, manifest)
    print("=" * 72)
    print("Seuils de sanité calibrés (§5.1) — à recopier dans docs/G2_3_OP_SENSITIVITY.md §5.1")
    print("=" * 72)
    print(f"  entropie moyenne A     : {sa['entropy_mean']:.4f} (σ {sa['entropy_std']:.4f})")
    print(f"  entropie moyenne B     : {sb['entropy_mean']:.4f} (σ {sb['entropy_std']:.4f})")
    print(f"  → entropie min tolérée : {ent_min:.4f}")
    print(f"  répétition argmax A/B  : {sa['argmax_repeat_max']}/{sb['argmax_repeat_max']}")
    print(f"  → répétition max tolérée : {rep_max}")
    print(f"  NaN/Inf                : 0 toléré")
    print(f"\nwrote {mpath}")
    return 0


def finish(manifest: dict, mpath: Path, args, entry: dict, dump_md5: str, npz_payload: dict) -> int:
    out_npz = mpath.parent / f"{args.run_name}_metrics.npz"
    np.savez(out_npz, **npz_payload)
    entry["metrics_npz"] = str(out_npz)
    upsert_run(manifest, args.run_name, entry, dump_md5)
    save_manifest(mpath, manifest)
    print(f"\n  VERDICT {args.run_name} : {entry['verdict']}")
    print(f"\nwrote {out_npz}\nwrote {mpath}")
    return 0


def do_analyze(args) -> int:
    for req in ("run_logits", "run_name", "ref_d0", "manifest"):
        if not getattr(args, req):
            sys.exit(f"[erreur] --{req.replace('_', '-')} requis")
    ref_a_none = (not args.ref_a) or args.ref_a.lower() == "none"
    if not args.register_reference and not ref_a_none and not args.envelope:
        sys.exit("[erreur] --envelope requis pour l'analyse vs A (source normative des seuils, §4)")
    mpath = Path(args.manifest)
    manifest = load_manifest(mpath)
    prov = {"bin_sha": args.bin_sha, "zml_rev": args.zml_rev, "repo_rev": args.repo_rev}

    mode = "register-reference" if args.register_reference else ("vs-D0 seul (S49)" if ref_a_none else "complet")
    print("=" * 72)
    print(f"G2.3 — analyse run '{args.run_name}' (mode {mode}) — docs/G2_3_OP_SENSITIVITY.md §5")
    print("=" * 72)

    # --- 1. custody (§2) : A (si fourni) et D0 — REFUS exit 2 au mismatch
    if not ref_a_none:
        custody_check(manifest, mpath, args.ref_a, "ref_a", prov)
    md5_d0 = custody_check(manifest, mpath, args.ref_d0, "ref_d0", prov)

    run_path = Path(args.run_logits)
    run = open_logits(args.run_logits, "run")
    n_steps = run.shape[0]
    same_as_d0 = run_path.resolve() == Path(args.ref_d0).resolve()
    dump_md5 = md5_d0 if same_as_d0 else md5_file(run_path)
    print(f"  run = {run_path.name} [{n_steps},{VOC}] md5={dump_md5}")

    th = manifest.get("sanity_thresholds")
    if not th:
        sys.exit("[erreur] seuils de sanité absents du manifest — lancer d'abord "
                 "`52_g2_3_analyze.py --calibrate-sanity` (protocole §5.1, AVANT le sweep)")

    entry: dict = {"type": "reference" if args.register_reference else "run",
                   "run_name": args.run_name, "run_logits": str(run_path),
                   "dump_md5": dump_md5, "n_steps": n_steps,
                   "provenance": {**prov, "analyzed_at": now()}}

    # --- 2. sanité (§5.1) — informative (loggée, non bloquante) pour les runs S49 (ref-a none)
    informative = ref_a_none
    scan = scan_pass(run, n_steps, "sanité")
    sane = (scan["nonfinite_elems"] == 0
            and scan["entropy_mean"] >= th["entropy_mean_min"]
            and scan["argmax_repeat_max"] <= th["argmax_repeat_max"])
    entry["sanity"] = {"ok": bool(sane), "informative": informative,
                       "nonfinite_steps": scan["nonfinite_steps"],
                       "nonfinite_elems": scan["nonfinite_elems"],
                       "entropy_mean": scan["entropy_mean"],
                       "argmax_repeat_max": scan["argmax_repeat_max"],
                       "thresholds": {"entropy_mean_min": th["entropy_mean_min"],
                                      "argmax_repeat_max": th["argmax_repeat_max"],
                                      "nan_inf_tolerated": 0}}
    print(f"  sanité : NaN/Inf={scan['nonfinite_elems']} ; entropie moy={scan['entropy_mean']:.4f} "
          f"(min {th['entropy_mean_min']:.4f}) ; répétition argmax={scan['argmax_repeat_max']} "
          f"(max {th['argmax_repeat_max']}) → {'OK' if sane else 'ÉCHEC'}"
          f"{' [informative, non bloquante — S49]' if informative else ''}")
    npz_payload = {"entropy": scan["entropy"], "argmax": scan["argmax"]}

    # --- mode register-reference : custody + sanité seulement, PAS d'auto-analyse vs soi-même
    if args.register_reference:
        entry["verdict"] = "REFERENCE" if (sane or informative) else "FAIL-SANITY"
        if not sane and informative:
            entry["sanity"]["note"] = "échec de sanité informative-only (seuils calibrés S46, §7)"
        if entry["verdict"] == "FAIL-SANITY":
            print("  ⚠ D0 échoue la sanité : référence INUTILISABLE, régénérer avant tout sweep.")
        return finish(manifest, mpath, args, entry, dump_md5, npz_payload)

    if not sane and not informative:
        entry["verdict"] = "FAIL-SANITY"  # publié tel quel : le verdict EST le résultat
        return finish(manifest, mpath, args, entry, dump_md5, npz_payload)

    # --- 3. non-vacuité (§5.2) + métriques vs D0 (classement)
    d0 = open_logits(args.ref_d0, "ref D0")
    if d0.shape[0] != n_steps:
        sys.exit(f"[erreur] steps run ({n_steps}) != D0 ({d0.shape[0]}) — mauvaise référence ?")
    m0 = metrics_pass(d0, run, n_steps, "vs D0")
    npz_payload.update(max_abs_vs_d0=m0["max_abs"], kl_vs_d0=m0["kl"], match_vs_d0=m0["match"])
    entry["metrics_vs_D0"] = {"max_abs": pct(m0["max_abs"]), "kl": pct(m0["kl"]),
                              "argmax_mismatches": m0["mismatches"],
                              "first_bifurcation_step": m0["first_bifurcation"]}
    print(f"  vs D0 : max_abs {entry['metrics_vs_D0']['max_abs']}")
    print(f"          KL(D0‖run) {entry['metrics_vs_D0']['kl']}")
    print(f"          mismatches {m0['mismatches']}/{n_steps} ; 1re bifurcation "
          f"{'aucune' if m0['first_bifurcation'] < 0 else m0['first_bifurcation']}")
    if float(m0["max_abs"].max()) == 0.0:
        entry["verdict"] = "VACUOUS"  # bit-identique à D0 : l'arrondi n'a rien arrondi (§5.2)
        print("  ⚠ run bit-identique à D0 — VACUOUS, investigation requise (jamais un SAFE silencieux)")
        return finish(manifest, mpath, args, entry, dump_md5, npz_payload)

    # --- 4. anti-câblage-croisé (§5.3)
    hlo_check, invalid = evaluate_hlo(args)
    entry["hlo_check"] = hlo_check
    if "families" in hlo_check:
        entry["families"] = hlo_check["families"]
        print(f"  converts : familles {hlo_check['families']} — attendu Δ{hlo_check['expected_delta']}, "
              f"observé Δ{hlo_check.get('observed_delta')}, differs_from_d0={hlo_check['differs_from_d0']}")
    if invalid:
        entry["verdict"] = "INVALID"
        print(f"  ⚠ INVALID : {hlo_check.get('invalid_reason')}")
        return finish(manifest, mpath, args, entry, dump_md5, npz_payload)

    # --- 5. métriques vs A + verdict bucket (§4) — sauf mode S49 (vs D0 seul)
    if ref_a_none:
        entry["metrics_vs_A"] = None
        entry["verdict"] = "diagnostic-only"  # pas d'enveloppe S49 → pas de buckets (§7)
        return finish(manifest, mpath, args, entry, dump_md5, npz_payload)

    a = open_logits(args.ref_a, "ref A")
    if n_steps > a.shape[0]:
        sys.exit(f"[erreur] steps run ({n_steps}) > A ({a.shape[0]})")
    ma = metrics_pass(a, run, n_steps, "vs A ")
    npz_payload.update(max_abs_vs_a=ma["max_abs"], kl_vs_a=ma["kl"], match_vs_a=ma["match"])
    entry["metrics_vs_A"] = {"max_abs": pct(ma["max_abs"]), "kl": pct(ma["kl"]),
                             "argmax_mismatches": ma["mismatches"],
                             "first_bifurcation_step": ma["first_bifurcation"]}
    env = json.loads(Path(args.envelope).read_text())
    r_ma = ratios(entry["metrics_vs_A"]["max_abs"], env["max_abs"])
    r_kl = ratios(entry["metrics_vs_A"]["kl"], env["kl_a_b"])
    b_kl, b_ma = bucket(r_kl["p50"]), bucket(r_ma["p50"])
    verdict = max((b_kl, b_ma), key=lambda v: BUCKET_ORDER[v])  # départage §4 : le pire l'emporte
    entry["ratios_vs_envelope"] = {"max_abs": r_ma, "kl": r_kl}
    entry["buckets"] = {"kl_p50": b_kl, "max_abs_p50": b_ma}
    entry["verdict"] = verdict
    print(f"  vs A  : max_abs {entry['metrics_vs_A']['max_abs']}")
    print(f"          KL(A‖run) {entry['metrics_vs_A']['kl']}")
    print(f"          mismatches {ma['mismatches']}/{n_steps} ; 1re bifurcation "
          f"{'aucune' if ma['first_bifurcation'] < 0 else ma['first_bifurcation']}")
    print(f"  ratios vs enveloppe B : KL {r_kl} → {b_kl} ; max_abs {r_ma} → {b_ma}")
    return finish(manifest, mpath, args, entry, dump_md5, npz_payload)


# ---------------------------------------------------------------- CLI

def main() -> None:
    p = argparse.ArgumentParser(
        description="G2.3 — analyse par run du sweep de sensibilité bf16 (cf docstring)")
    p.add_argument("--run-logits", help="dump logits du run, .bin f32 brut [steps×262144]")
    p.add_argument("--run-name", help="nom du run (déjà préfixé par l'orchestrateur)")
    p.add_argument("--ref-a", help=".npy memmap f32 (bras A) ; 'none' = mode vs-D0 seul (S49)")
    p.add_argument("--ref-d0", help="baseline ZML fp32, .bin f32 brut")
    p.add_argument("--ref-b", default=REF_B_DEFAULT,
                   help=f".npy uint16 bf16-bits (bras B, calibration) [défaut {REF_B_DEFAULT}]")
    p.add_argument("--envelope", help="fixtures/g2_envelope_manifest.json (seuils NON-ARRONDIS, normatif)")
    p.add_argument("--expected-converts", help="fixtures/g2_3_expected_converts.json (gate §5.3)")
    p.add_argument("--hlo-report", help="rapport json du script 53 (gate §5.3)")
    p.add_argument("--manifest", help="fixtures/g2_3_manifest.json (custody + seuils + runs)")
    p.add_argument("--bin-sha", default="n/a", help="sha256 du binaire sweep (provenance)")
    p.add_argument("--zml-rev", default="n/a", help="rev git du workspace ZML (provenance)")
    p.add_argument("--repo-rev", default="n/a", help="rev git du repo probe (provenance)")
    p.add_argument("--register-reference", action="store_true",
                   help="run D0 : custody + sanité seulement, pas d'auto-analyse")
    p.add_argument("--calibrate-sanity", action="store_true",
                   help="calibre les seuils §5.1 depuis les memmaps A et B → manifest")
    p.add_argument("--selfcheck", action="store_true",
                   help="rejoue les ratios G2.2 depuis fixtures/ (garde de formule, numpy pur)")
    args = p.parse_args()

    if args.selfcheck:
        sys.exit(do_selfcheck())
    if args.calibrate_sanity:
        sys.exit(do_calibrate(args))
    sys.exit(do_analyze(args))


if __name__ == "__main__":
    main()
