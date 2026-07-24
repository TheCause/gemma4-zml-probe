#!/usr/bin/env python3
"""U 1b — export DEQUANTISÉ bf16 du checkpoint 12B packé + selfchecks câblés.

Streaming sur les clés de weights_12b/model.safetensors (snapshot figé, cf. U_12B_CONTRACT §6) :
chaque `weight_packed` -> unpack_from_int32 (IMPORTÉ de compressed-tensors, piège 12 — jamais
recodé) + dequant groupé -> tenseur bf16 sous la clé `.weight` ; les clés bf16 passent telles
quelles ; les `weight_shape`/`weight_scale` sont consommées, jamais exportées. Sortie en 2 shards
HF standard (model-0000X-of-0000N.safetensors + index) pour borner la RAM (~14 Go par shard vs
23 Gi VM), + config.json ÉPURÉ (sans `quantization_config`) et fichiers tokenizer — l'export se
charge par transformers SANS compressed-tensors (M4 n'a pas ct).

DÉVIATION déclarée vs plan Task 1b : les 11 clés multimodales (model.vision_embedder.*,
model.embed_vision.*, model.embed_audio.*, ~105 Mo) sont CONSERVÉES au lieu d'être exclues —
Gemma4UnifiedForConditionalGeneration les instancie sans liste d'ignore (transformers 5.14.1,
conversion_mapping.py renomme vision_embedder->embed_vision au load) : les exclure rendrait le
selfcheck census « 0 missing » infaisable. Le census strict 0/0 prime (c'est le gate).

Selfchecks (assertions, échec bruyant, ordre = RAM croissante) :
  (ii) 328/328 modules packés décompressés par la VOIE OFFICIELLE ct
       (PackedQuantizationCompressor.decompress + QuantizationScheme parsé du config) et
       comparés bit (`torch.equal`) au tenseur de l'export — l'export utilise la recomposition
       manuelle (formule J1, script 57) : deux voies INDÉPENDANTES, le check peut échouer.
  (i)  census : from_pretrained sur l'export (bf16, low_cpu_mem_usage, output_loading_info)
       -> assert 0 missing / 0 unexpected / 0 mismatched.

Manifest : fixtures/u_export_manifest.json (écrit APRÈS les selfchecks — son existence = PASS).

Option `--fixture-u1b` (Task 3, ne PAS lancer en Task 1b) : extrait 3 matrices dequantées
(q_proj L0/L5, down_proj L0) -> fixtures/u_mats12.safetensors + manifest (critère fixture :
« dequant ZML == bit-exact bf16 »). Option `--skip-export` : rejoue uniquement les selfchecks
sur un export déjà écrit.

Venv : /data/venvs/g12b. Lancement (VM) :
  HF_HOME=/data/hf_cache /data/venvs/g12b/bin/python3 scripts/63_u_dequant_export.py
"""
import argparse
import gc
import json
import os
import shutil

os.environ.setdefault("HF_HOME", "/data/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# Piège 12 : imports de RÉFÉRENCE compressed-tensors 0.17.1 — chemins vérifiés sur le venv g12b.
from compressed_tensors.compressors.pack_quantized import (
    PackedQuantizationCompressor,
    unpack_from_int32,
)
from compressed_tensors.quantization import QuantizationConfig, QuantizationScheme

ROOT = "/data/gemma4-zml-probe"
CKPT = os.path.join(ROOT, "weights_12b", "model.safetensors")
OUT_DIR = os.path.join(ROOT, "weights_12b_dq")
FX = os.path.join(ROOT, "fixtures")
SNAPSHOT_SHA = "1d2c2d7f2466070e69d6fb3fd5ce9a7d75f2f6ee"
SHARD_BYTES = 14_000_000_000  # seuil d'accumulation -> ~2 shards pour ~26 Go
N_PACKED_EXPECTED = 328  # contrat U0 : 48x6 + 40 v_proj

TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "generation_config.json",
                   "chat_template.jinja")


def snapshot_dir() -> str:
    # PAS realpath : il traverse les 2 hops du cache HF et atterrit dans blobs/ (bug déjà mordu).
    snap = os.path.dirname(os.readlink(CKPT))
    assert SNAPSHOT_SHA in snap, f"snapshot inattendu : {snap}"
    return snap


def _shard_sizes(weight_map: dict) -> list:
    """Tailles des shards depuis le DISQUE — source de vérité unique pour le manifest,
    utilisée par les deux branches (export et --skip-export) : elles convergent par construction."""
    return [{"file": fn, "bytes": os.path.getsize(os.path.join(OUT_DIR, fn))}
            for fn in sorted(set(weight_map.values()))]


def dequant_manual(wp: torch.Tensor, ws: torch.Tensor, shape: list) -> torch.Tensor:
    """Recomposition manuelle (formule J1 prouvée, script 57) — contrôlée par le selfcheck (ii)."""
    out_d, in_d = int(shape[0]), int(shape[1])
    assert ws.dtype == torch.bfloat16, f"scale {ws.dtype} != bf16"
    assert list(ws.shape) == [out_d, in_d // 32], f"scale {list(ws.shape)} vs [{out_d},{in_d//32}]"
    q = unpack_from_int32(wp, num_bits=4, shape=torch.Size([out_d, in_d]))
    return (q.reshape(out_d, in_d // 32, 32).to(torch.bfloat16) * ws.unsqueeze(-1)).reshape(out_d, in_d)


def plan_output(f) -> list:
    """[(clé export, nbytes, kind, base)] dans l'ordre trié des clés du checkpoint.
    kind = 'dequant' (base = module packé) | 'pass'. Shapes lues sans charger les tenseurs."""
    keys = sorted(f.keys())
    plan = []
    n_packed = 0
    for k in keys:
        if k.endswith(".weight_shape") or k.endswith(".weight_scale"):
            continue  # consommées avec le weight_packed correspondant
        if k.endswith(".weight_packed"):
            base = k[: -len("weight_packed")]
            shape = f.get_tensor(base + "weight_shape").tolist()
            plan.append((base + "weight", int(shape[0]) * int(shape[1]) * 2, "dequant", base))
            n_packed += 1
        else:
            sl = f.get_slice(k)
            assert str(sl.get_dtype()) == "BF16", f"{k}: dtype {sl.get_dtype()} != BF16"
            n = 1
            for d in sl.get_shape():
                n *= d
            plan.append((k, n * 2, "pass", k))
    assert n_packed == N_PACKED_EXPECTED, f"{n_packed} weight_packed != {N_PACKED_EXPECTED}"
    return plan


def run_export() -> dict:
    snap = snapshot_dir()
    os.makedirs(OUT_DIR, exist_ok=True)

    with safe_open(CKPT, framework="pt") as f:
        plan = plan_output(f)

        # Découpage en shards : greedy dans l'ordre trié, flush au seuil.
        shards, cur, cur_bytes = [], [], 0
        for entry in plan:
            if cur and cur_bytes + entry[1] > SHARD_BYTES:
                shards.append(cur)
                cur, cur_bytes = [], 0
            cur.append(entry)
            cur_bytes += entry[1]
        if cur:
            shards.append(cur)

        n_shards = len(shards)
        weight_map, total = {}, 0
        for i, entries in enumerate(shards, start=1):
            fname = f"model-{i:05d}-of-{n_shards:05d}.safetensors"
            tensors = {}
            for out_key, nbytes, kind, base in entries:
                if kind == "dequant":
                    t = dequant_manual(
                        f.get_tensor(base + "weight_packed"),
                        f.get_tensor(base + "weight_scale"),
                        f.get_tensor(base + "weight_shape").tolist(),
                    )
                else:
                    t = f.get_tensor(out_key)
                assert t.dtype == torch.bfloat16, f"{out_key}: {t.dtype} != bf16"
                assert not torch.isnan(t.float()).any(), f"NaN dans {out_key}"
                tensors[out_key] = t.contiguous()
                weight_map[out_key] = fname
                total += nbytes
            save_file(tensors, os.path.join(OUT_DIR, fname), metadata={"format": "pt"})
            print(f"  shard {i}/{n_shards} : {len(tensors)} tenseurs, "
                  f"{sum(e[1] for e in entries)/1e9:.2f} Go -> {fname}", flush=True)
            del tensors
            gc.collect()

    with open(os.path.join(OUT_DIR, "model.safetensors.index.json"), "w") as fh:
        json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, fh, indent=2)

    # config ÉPURÉ : sans quantization_config, sinon from_pretrained exigerait ct (absent de M4).
    cfg = json.load(open(os.path.join(snap, "config.json")))
    cfg.pop("quantization_config")
    with open(os.path.join(OUT_DIR, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)
    for name in TOKENIZER_FILES:
        shutil.copy2(os.path.join(snap, name), os.path.join(OUT_DIR, name))

    print(f"export : {len(weight_map)} clés, {total/1e9:.2f} Go, {n_shards} shards", flush=True)
    # Manifest depuis le DISQUE (même logique que --skip-export : convergence par construction).
    cfg_written = json.load(open(os.path.join(OUT_DIR, "config.json")))
    return {"n_keys": len(weight_map), "total_bytes": total,
            "shards": _shard_sizes(weight_map),
            "quantization_config_removed": "quantization_config" not in cfg_written}


def selfcheck_official_328() -> int:
    """(ii) chaque module packé décompressé par la voie OFFICIELLE ct, comparé bit à l'export."""
    snap = snapshot_dir()
    cfg = json.load(open(os.path.join(snap, "config.json")))
    qconf = QuantizationConfig.model_validate(cfg["quantization_config"])
    scheme = qconf.config_groups["group_0"]
    assert isinstance(scheme, QuantizationScheme) and scheme.weights.num_bits == 4 \
        and scheme.weights.group_size == 32 and scheme.weights.symmetric, f"scheme inattendu : {scheme}"

    index = json.load(open(os.path.join(OUT_DIR, "model.safetensors.index.json")))
    by_shard = {}  # shard -> [clé .weight d'un module packé]
    with safe_open(CKPT, framework="pt") as f:
        packed_bases = {k[: -len("weight_packed")] for k in f.keys() if k.endswith(".weight_packed")}
    for base in packed_bases:
        key = base + "weight"
        by_shard.setdefault(index["weight_map"][key], []).append(base)

    n_ok = 0
    with safe_open(CKPT, framework="pt") as src:
        for shard, bases in sorted(by_shard.items()):
            with safe_open(os.path.join(OUT_DIR, shard), framework="pt") as exp:
                for base in sorted(bases):
                    sd = {"weight_packed": src.get_tensor(base + "weight_packed"),
                          "weight_scale": src.get_tensor(base + "weight_scale"),
                          "weight_shape": src.get_tensor(base + "weight_shape")}
                    official = PackedQuantizationCompressor.decompress(sd, scheme)["weight"]
                    got = exp.get_tensor(base + "weight")
                    assert torch.equal(official, got), \
                        f"selfcheck (ii) : {base}weight export != décompression officielle ct"
                    n_ok += 1
    assert n_ok == N_PACKED_EXPECTED, f"selfcheck (ii) : {n_ok} modules vérifiés != {N_PACKED_EXPECTED}"
    print(f"selfcheck (ii) : {n_ok}/{N_PACKED_EXPECTED} bit-exact (voie officielle ct)", flush=True)
    return n_ok


def selfcheck_census() -> dict:
    """(i) l'export se charge par transformers seul : 0 missing / 0 unexpected / 0 mismatched."""
    from transformers import AutoModelForCausalLM
    model, info = AutoModelForCausalLM.from_pretrained(
        OUT_DIR, dtype=torch.bfloat16, low_cpu_mem_usage=True, output_loading_info=True)
    missing, unexpected = info["missing_keys"], info["unexpected_keys"]
    mismatched = info.get("mismatched_keys", [])
    del model
    gc.collect()
    assert not missing, f"census : {len(missing)} missing, ex. {missing[:5]}"
    assert not unexpected, f"census : {len(unexpected)} unexpected, ex. {unexpected[:5]}"
    assert not mismatched, f"census : mismatched {mismatched[:5]}"
    print("selfcheck (i) : census 0 missing / 0 unexpected / 0 mismatched", flush=True)
    return {"missing": 0, "unexpected": 0, "mismatched": 0}


def fixture_u1b() -> None:
    """Task 3 (U1b) : 3 matrices dequantées -> fixtures/u_mats12.safetensors (+ manifest)."""
    LM = "model.language_model."
    MATS = {  # nom fixture -> (module, [out, in]) — 3 familles de shape du contrat U0 §3
        "deq_q0": (LM + "layers.0.self_attn.q_proj", [4096, 3840]),   # sliding, 16x256
        "deq_q5": (LM + "layers.5.self_attn.q_proj", [8192, 3840]),   # full, 16x512
        "deq_dn0": (LM + "layers.0.mlp.down_proj", [3840, 15360]),    # down, in large
    }
    os.makedirs(FX, exist_ok=True)
    out_t = {}
    with safe_open(CKPT, framework="pt") as f:
        for name, (mod, shp) in MATS.items():
            shape = f.get_tensor(mod + ".weight_shape").tolist()
            assert [int(s) for s in shape] == shp, f"{mod}: weight_shape {shape} != {shp}"
            out_t[name] = dequant_manual(
                f.get_tensor(mod + ".weight_packed"), f.get_tensor(mod + ".weight_scale"), shp
            ).contiguous()
    save_file(out_t, os.path.join(FX, "u_mats12.safetensors"))
    json.dump(
        {
            "source": "63_u_dequant_export.py --fixture-u1b (snapshot " + SNAPSHOT_SHA + ")",
            "pass": "dequant ZML de weight_packed/weight_scale (lus du checkpoint) == deq_*, "
                    "bit-exact bf16, 3/3",
            "matrices": {k: v[1] for k, v in MATS.items()},
            "modules": {k: v[0] for k, v in MATS.items()},
        },
        open(os.path.join(FX, "u_mats12_manifest.json"), "w"), indent=2,
    )
    print("PASS fixture U1b écrite (fixtures/u_mats12.safetensors)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = ap.add_mutually_exclusive_group()  # les deux modes sont incompatibles, refus explicite
    group.add_argument("--fixture-u1b", action="store_true",
                       help="écrit uniquement la fixture U1b (Task 3), pas d'export")
    group.add_argument("--skip-export", action="store_true",
                       help="rejoue les selfchecks sur un export existant")
    args = ap.parse_args()

    if args.fixture_u1b:
        fixture_u1b()
        return

    if args.skip_export:
        # Source de vérité DISQUE, même logique que la branche export (convergence par construction).
        index = json.load(open(os.path.join(OUT_DIR, "model.safetensors.index.json")))
        cfg_written = json.load(open(os.path.join(OUT_DIR, "config.json")))
        export_info = {"n_keys": len(index["weight_map"]),
                       "total_bytes": index["metadata"]["total_size"],
                       "shards": _shard_sizes(index["weight_map"]),
                       "quantization_config_removed": "quantization_config" not in cfg_written}
    else:
        export_info = run_export()

    n_official = selfcheck_official_328()  # (ii) d'abord : RAM faible
    census = selfcheck_census()            # (i) ensuite : charge le modèle entier

    import transformers
    import compressed_tensors
    os.makedirs(FX, exist_ok=True)
    manifest = {
        "source": "63_u_dequant_export.py (Task 1b, plan J2 amendé 2026-07-24)",
        "snapshot_sha": SNAPSHOT_SHA,
        "versions": {"transformers": transformers.__version__, "torch": torch.__version__,
                     "compressed_tensors": compressed_tensors.__version__},
        "export": export_info,
        "deviation": "11 clés multimodales CONSERVÉES (~105 Mo) : le modèle les instancie sans "
                     "liste d'ignore -> les exclure rendrait le census 0-missing infaisable ; "
                     "le census strict 0/0 prime.",
        "selfchecks": {
            "official_ct_bit_exact": f"{n_official}/{N_PACKED_EXPECTED}",
            "census": census,
        },
        "verdict": "PASS",
    }
    with open(os.path.join(FX, "u_export_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"PASS U-export : {export_info['n_keys']} clés, "
          f"{export_info['total_bytes']/1e9:.2f} Go, selfchecks (i)+(ii) OK -> "
          f"{os.path.join(FX, 'u_export_manifest.json')}", flush=True)


if __name__ == "__main__":
    main()
