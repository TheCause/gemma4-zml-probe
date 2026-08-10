#!/usr/bin/env python3
"""U8/U9-iv — oracle de génération 12B Unified sur l'export dq : décode greedy 48 tokens
(fixture u8_gen48) + mode --teacher-force (U9-iv : UN prefill de la séquence donnée).

Plan Task 9 Step 9.1 / Task 10 Step 10.3 (docs/superpowers/plans/2026-07-24-w4-j2-12b-unified.md),
D6 plan A : oracle CPU M4 (venv g12b), l'export dq mmap (~26 Go, 1er token = warm-up, la mesure
= médiane des tokens 2-4). Modèle RÉEL bf16 de bout en bout (AutoModelForCausalLM,
low_cpu_mem_usage — même chemin que le smoke U0b-décode Task 1b et l'oracle U7 68).

Format fixture (mode décode) : le mode --oracle de gemma4_g12auto lit EXACTEMENT deux clés
(gemma4_g12auto.zig l.904-936) : `positions` (i32, positions[0] == ids.len du prompt rendu,
check anti-mismatch prompt/fixture) et `fed` (i32, la séquence de référence [s0, t1, …] comparée
à `generated`). `expected`/`top5_ids`/`top5_vals` = diagnostic (non lus par le runner), miroir
du producteur J1 56_w4_gen_oracle.py (fed = seq[:n], expected = seq[1:n+1]).

Chat template : celui du SNAPSHOT 12B copié dans l'export par le 63 — sha asserté (même garde
que le 68) ; pattern BatchEncoding 5.14 (contrat §8) : return_dict=True, **enc au forward.
Softcap : appliqué par HF DANS ForCausalLM.forward (modeling_gemma4.py : lm_head → /cap →
tanh → ×cap) — out.logits est déjà softcappé, asserté ≤ 30 à chaque step.

Mode --teacher-force <ids.safetensors> (U9-iv, D10) : lit la clé `ids` (i32, écrite par
--out-ids du runner), fait UN prefill de [prompt templaté ++ ids[:-1]], puis argmax + top-5 +
marge PAR POSITION au fil de l'eau : hidden states du backbone (self.model) → lm_head + softcap
par CHUNKS de positions (miroir exact modeling_gemma4.py l.2592-2597) — jamais les ~600 Mo de
logits d'un coup. Self-check câblé : sur un préfixe court, tête manuelle == forward HF complet
(torch.equal). ⚠ --prompt reste REQUIS dans ce mode : --out-ids n'écrit que les ids GÉNÉRÉS,
le contexte du prefill = prompt templaté + ids (décision d'interprétation, rapport Task 9).

CLI :
  décode  : 69_u8_gen_oracle.py --weights <export_dq> --prompt "..." --n-tokens 48 --out u8_gen48.safetensors
  U9-iv   : 69_u8_gen_oracle.py --weights <export_dq> --prompt "..." --teacher-force u9_ids.safetensors --out u9_tf.json
Venv : ~/ml-venvs/g12b (M4, plan A) ou /data/venvs/g12b (VM, repli D6-B).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time

if os.path.isdir("/data"):
    os.environ.setdefault("HF_HOME", "/data/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = "/data/gemma4-zml-probe"
# sha256 du chat_template.jinja du snapshot 12B (U_12B_CONTRACT §6, même garde que le 68).
TEMPLATE_SHA_12B = "ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4"
VOCAB = 262144
SOFTCAP = 30.0
TOPK = 5
HEAD_CHUNK = 64  # teacher-force : positions par chunk (64 × 262144 × bf16 ≈ 32 Mo)
# Version minimale de transformers exigée : la sémantique de SuppressTokensLogitsProcessor et la
# normalisation `eos_token_id: int → [int]` sont lues à cette version (spec §4.1/§4.3). Un oracle
# qui tournerait sur une autre version validerait une sémantique qu'on n'a pas relue.
TRANSFORMERS_MIN = (5, 14, 1)


def script_md5() -> str:
    """md5 de CE fichier. GC7 l'exige au rapport : la VM n'est pas un dépôt git et sa copie du
    script a DÉJÀ été périmée une fois (tf_probe/tf200.json produit par un script absent du chemin
    canonique). Un rapport qui ne dit pas quel code l'a produit n'est pas traçable."""
    with open(os.path.abspath(__file__), "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def build_gen_policy(model, disabled: bool):
    """Politique de décodage, prise à la SOURCE : le processor de transformers, construit depuis
    `model.generation_config` peuplé par `from_pretrained`. Jamais une réimplémentation locale.

    POURQUOI CE BLOC EXISTE — finding du 27 juil (docs/FINDING_GENERATION_CONFIG.md) : cet oracle
    faisait un argmax NU, exactement comme le runner ZML. L'instrument était donc aveugle au MÊME
    endroit que son sujet, et aucun gate ne pouvait détecter que `suppress_tokens` n'était appliqué
    nulle part. Runner et oracle changent dans le même mouvement — c'est la condition pour que le
    mode --oracle reste une comparaison valide (spec §4.4, corollaire non négociable).

    Rend `(processor|None, description)`.
    """
    import transformers  # importé localement, comme dans load_model_and_tok (HF_HOME d'abord)

    ver = tuple(int(x) for x in transformers.__version__.split(".")[:3])
    assert ver >= TRANSFORMERS_MIN, (
        f"transformers {transformers.__version__} < {'.'.join(map(str, TRANSFORMERS_MIN))} : "
        "la sémantique de la politique de décodage n'a été relue qu'à partir de cette version"
    )
    gc = model.generation_config
    sup = list(gc.suppress_tokens) if getattr(gc, "suppress_tokens", None) else []
    eos = getattr(gc, "eos_token_id", None)
    eos = [eos] if isinstance(eos, int) else (list(eos) if eos else [])
    assert getattr(gc, "begin_suppress_tokens", None) in (None, []), (
        "begin_suppress_tokens présente : sémantique temporelle non implémentée (spec §4.1)"
    )
    desc = {
        "applied": (not disabled) and bool(sup),
        "suppress_tokens": sup,
        "eos_token_id": eos,
        "source": "model.generation_config (from_pretrained)",
        "processor": "transformers.generation.logits_process.SuppressTokensLogitsProcessor",
        "note": "SUPPRESSION seulement — l'arrêt EOS n'est PAS appliqué par cet oracle "
                "(mode --oracle du runner : suppression ON, arrêt OFF, décision Régis n°3)",
    }
    if disabled or not sup:
        return None, desc
    from transformers.generation.logits_process import SuppressTokensLogitsProcessor
    return SuppressTokensLogitsProcessor(sup, device="cpu"), desc


def apply_policy(lg2d: torch.Tensor, proc):
    """Applique la politique sur une COPIE — l'ordre est imposé (plan 4.2) : les asserts de
    l'appelant portent sur les logits BRUTS, la politique vient après, et le topk après elle.
    Inverser reviendrait à asserter sur des `-inf` fabriqués par nous."""
    if proc is None:
        return None
    return proc(torch.zeros((lg2d.shape[0], 1), dtype=torch.long), lg2d.clone())


def anon_path(p: str) -> str:
    """Anonymisation (règle repo public — feedback revue U8) : le préfixe HOME de l'hôte oracle
    ne doit JAMAIS apparaître dans un manifest committable — même mécanique que --host-label.
    Les chemins /data/... (VM) sont déjà publics dans le repo, laissés tels quels."""
    home = os.path.expanduser("~")
    return "<oracle-host>" + p[len(home):] if home not in ("~", "/") and p.startswith(home) else p


def install_fp32_hooks(backbone, extra=()) -> int:
    """--compute-fp32 : calcul fp32 sur STOCKAGE bf16 (parité oracles J1 46/56 « poids fp32,
    oracle = source de vérité », impossible en chargement monolithique 12B : ~48 Go > 32 Gio M4).
    Chaque sous-module passe en fp32 le temps de SON forward (pré-hook) puis revient en bf16
    (post-hook) — bf16→fp32→bf16 est sans perte, pic RAM = +1 copie de couche (~1 Go).
    Les activations restent fp32 de bout en bout (les post-hooks ne touchent que les poids).
    `extra` : modules hors backbone (lm_head) — hooks transitoires AUSSI, car en tied (D7)
    une conversion permanente serait annulée par le post-hook d'embed_tokens au forward suivant.
    Granularité : tout module à PARAMÈTRES DIRECTS (Linear/Norm/Embedding), structure-agnostique
    (les DecoderLayers du 12B Unified ne sont pas des enfants directs de model.model — appris
    au smoke 16 ids : q_proj bf16 face à des activations fp32)."""
    mods = [m for m in backbone.modules() if any(True for _ in m.parameters(recurse=False))]
    mods.extend(extra)

    def pre(mod, _inp):
        mod.to(torch.float32)

    def post(mod, _inp, _out):
        mod.to(torch.bfloat16)

    for m in mods:
        m.register_forward_pre_hook(pre)
        m.register_forward_hook(post)
    return len(mods)


def load_model_and_tok(dq: str):
    """Chargement commun (chemin U0b-décode/U7) : sha template asserté, modèle réel bf16 mmap."""
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tpl_sha = hashlib.sha256(open(os.path.join(dq, "chat_template.jinja"), "rb").read()).hexdigest()
    assert tpl_sha == TEMPLATE_SHA_12B, f"chat template export != snapshot 12B (U0 §6) : {tpl_sha}"
    tok = AutoTokenizer.from_pretrained(dq)
    t0 = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(dq, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.train(False)
    t_load = time.monotonic() - t0
    cfg_t = model.config.get_text_config()
    assert cfg_t.num_hidden_layers == 48 and cfg_t.hidden_size == 3840
    assert float(cfg_t.final_logit_softcapping) == SOFTCAP
    print(f"modèle chargé (bf16, low_cpu_mem_usage) en {t_load:.1f} s", flush=True)
    versions = {"transformers": transformers.__version__, "torch": torch.__version__}
    return model, tok, tpl_sha, t_load, versions


def encode_prompt(tok, prompt: str):
    """Pattern BatchEncoding 5.14 (contrat §8) : return_dict=True, **enc au forward."""
    msgs = [{"role": "user", "content": prompt}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True)
    ids = enc["input_ids"][0]
    S = int(ids.shape[0])
    assert S >= 4, f"prompt templaté suspect : S={S}"
    print(f"prompt canonique templaté : S={S} ids={ids.tolist()}", flush=True)
    return enc, ids, S


def step_top5(logits_1d: torch.Tensor, proc=None, penalty_proc=None, hist_ids=None):
    """top-5 (+marge top1−top2) d'un vecteur logits [V] — f32 pour les marges.

    Ordre IMPOSÉ (plan 4.2) : les asserts portent sur les logits BRUTS, la chaîne s'applique
    ENSUITE sur une copie, le topk après elle. Asserter après la politique reviendrait à tester la
    finitude de `-inf` qu'on a nous-mêmes écrits.

    ⚠ La PENALTY est appliquée ICI et pas avant l'appel, pour une raison mesurable : diviser un
    logit positif par 0,8 le porte à 37,5 et ferait sauter l'assert `max_abs <= SOFTCAP`, qui
    parle du modèle et non de la chaîne. L'ordre obtenu est exactement celui de HF, mesuré (F8) :
    RepetitionPenalty(4) → SuppressTokens(15).

    ⚠ Le processor est APPELÉ (spec C5) : retranscrire son `torch.where` des deux côtés ferait
    passer une faute commune à l'oracle et au runner.

    Rend `(idxs_bruts, vals_bruts, marge_brute, max_abs, chaîne|None)`. Le 5ᵉ élément est le
    top-5 après CHAÎNE COMPLÈTE (penalty puis politique) — c'est lui qui décide du token retenu.
    """
    lg = logits_1d.float()
    assert torch.isfinite(lg).all(), "logits non finis"
    max_abs = float(lg.abs().max())
    assert max_abs <= SOFTCAP, f"softcap VIOLÉ : max|logits|={max_abs} > {SOFTCAP}"
    vals, idxs = torch.topk(lg, TOPK)  # BRUT — schéma inchangé, les gates historiques le lisent
    lg_chain, touched = lg, False
    if penalty_proc is not None:
        assert hist_ids is not None and hist_ids.numel() > 0, \
            "penalty armée sans historique : le processor recevrait un input_ids vide"
        lg_chain = penalty_proc(hist_ids.unsqueeze(0), lg.clone().unsqueeze(0))[0]
        touched = bool((lg_chain != lg).any())
    pol = None
    if proc is not None:
        lg_p = apply_policy(lg_chain.unsqueeze(0), proc)[0]
        pvals, pidxs = torch.topk(lg_p, TOPK)
        # Avec V=262144 et |S|=2, il reste toujours ≥ 5 candidats finis : un -inf ici signalerait
        # une liste de suppression aberrante, et json.dump écrirait `-Infinity` (JSON non standard).
        assert torch.isfinite(pvals).all(), "top-5 POST-politique contient un -inf"
        pol = (pidxs.to(torch.int32), pvals)
    elif penalty_proc is not None:
        # Penalty armée SANS politique (--no-gen-policy) : la sélection doit quand même suivre la
        # chaîne, sinon la penalty serait calculée puis jetée et le gate passerait à vide.
        pvals, pidxs = torch.topk(lg_chain, TOPK)
        pol = (pidxs.to(torch.int32), pvals)
    return idxs.to(torch.int32), vals, float(vals[0] - vals[1]), max_abs, pol, touched


def mode_decode(args, model, tok, tpl_sha, t_load, versions):
    proc, gen_policy = build_gen_policy(model, args.no_gen_policy)
    print(f"GENPOLICY: {gen_policy}", flush=True)
    # Prérequis GC8 : l'instrument fp32 doit être disponible ICI aussi, sinon le seuil de kill
    # dérivé en fp32 ne s'appliquerait pas au gate qui l'utilise (plan 6.1).
    compute_dtype = "bfloat16"
    if args.compute_fp32:
        n_hooked = install_fp32_hooks(model.model, extra=(model.lm_head,))
        compute_dtype = "float32"
        print(f"calcul fp32 armé (mode décode) : {n_hooked} sous-modules hookés", flush=True)
    enc, prompt_ids, S = encode_prompt(tok, args.prompt)
    n = int(args.n_tokens)
    # --- Phase 1 : repetition penalty, par le VRAI processor (spec C5) ---
    # Historique = prompt ++ tokens générés, qui est le défaut HF (`input_ids` complet). Le runner
    # tient le même contrat côté Zig ; c'est cette égalité de définition qui rend RP3 concluant.
    penalty_proc = None
    if args.repetition_penalty != 1.0:
        from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor
        penalty_proc = RepetitionPenaltyLogitsProcessor(penalty=float(args.repetition_penalty))
        print(f"PENALTY: armée rp={args.repetition_penalty} (processor HF, historique = prompt ++ généré)", flush=True)
    n_penalty_touched = 0
    eot_id = 106  # <turn|> (mesuré, cf gemma4_g12auto.zig) — informatif seulement : en mode
    # --oracle le runner NE s'arrête PAS à l'EOT (limite = fed.len), l'oracle non plus.

    # --- prefill (token 0 = s0, chrono à part : chargement pages mmap + warm-up) ---
    t0 = time.monotonic()
    with torch.no_grad():
        out = model(**enc, use_cache=True)
    t_prefill = time.monotonic() - t0
    # Le dtype ATTENDU dépend de l'instrument : bf16 par défaut (modèle réel), fp32 sous
    # --compute-fp32 (hooks par-module). L'assert restait câblé sur bf16 — il a mordu au premier
    # run fp32 en mode décode, ce qui est exactement son rôle : il refusait un dtype qu'il ne
    # connaissait pas plutôt que de laisser passer un instrument non déclaré.
    want_logits_dtype = torch.float32 if args.compute_fp32 else torch.bfloat16
    assert out.logits.dtype == want_logits_dtype, \
        f"logits {out.logits.dtype} != {want_logits_dtype} (compute_fp32={args.compute_fp32})"
    pkv = out.past_key_values
    # ⚠ s0 est produit par le prefill, HORS boucle : la penalty s'y applique AUSSI, sinon la
    # divergence avec le runner serait garantie dès le token 0. À ce point HF a vu exactement le
    # prompt — l'historique est donc `prompt_ids`.
    idxs, vals, margin, max_abs, pol, touched = step_top5(out.logits[0, -1, :], proc, penalty_proc, prompt_ids)
    if touched:
        n_penalty_touched += 1
    # Le token retenu est celui d'APRÈS politique quand elle s'applique — c'est ce que fait le
    # runner en mode --oracle (suppression ON). Sans cela, la fixture serait produite par un
    # décodage que le runner ne reproduit plus, et le mode --oracle comparerait deux politiques
    # différentes en croyant comparer deux implémentations.
    chosen = int(pol[0][0]) if pol is not None else int(idxs[0])
    print(f"prefill S={S} en {t_prefill:.1f} s ; s0={chosen} marge={margin:.6f}"
          f"{'' if pol is None else f' (brut={int(idxs[0])})'}", flush=True)

    seq = [chosen]
    top5_ids = torch.zeros(n, TOPK, dtype=torch.int32)
    top5_vals = torch.zeros(n, TOPK, dtype=torch.float32)
    top5_pol_ids = torch.zeros(n, TOPK, dtype=torch.int32)
    top5_pol_vals = torch.zeros(n, TOPK, dtype=torch.float32)
    n_policy_bites = 0
    stopped_at, stop_eos = None, None
    margins, times, max_abs_all = [margin], [t_prefill], [max_abs]
    top5_ids[0], top5_vals[0] = idxs, vals
    if pol is not None:
        top5_pol_ids[0], top5_pol_vals[0] = pol
        if chosen != int(idxs[0]):
            n_policy_bites += 1

    # --- décode greedy : n-1 steps produisent seq[1..n-1], +1 step pour `expected` (miroir 56) ---
    for k in range(1, n + 1):
        tk = time.monotonic()
        with torch.no_grad():
            out = model(input_ids=torch.tensor([[seq[-1]]], dtype=torch.long),
                        past_key_values=pkv, use_cache=True)
        dt = time.monotonic() - tk
        # Historique au step k = prompt ++ les k tokens déjà produits — exactement l'`input_ids`
        # que HF passerait à son processor à ce point (`seq` contient k éléments AVANT l'append).
        hist_k = torch.cat([prompt_ids, torch.tensor(seq, dtype=prompt_ids.dtype)]) if penalty_proc is not None else None
        idxs, vals, margin, max_abs, pol, touched = step_top5(out.logits[0, -1, :], proc, penalty_proc, hist_k)
        if touched:
            n_penalty_touched += 1
        chosen_k = int(pol[0][0]) if pol is not None else int(idxs[0])
        seq.append(chosen_k)
        times.append(dt)
        if k < n:  # le step n ne produit que expected[n-1], pas un step de génération du gate
            top5_ids[k], top5_vals[k] = idxs, vals
            if pol is not None:
                top5_pol_ids[k], top5_pol_vals[k] = pol
                if chosen_k != int(idxs[0]):
                    n_policy_bites += 1
            margins.append(margin)
            max_abs_all.append(max_abs)
            print(f"  marge top1-top2 @ gen={k} : {margin:.6f} (top1={int(idxs[0])} top2={int(idxs[1])})"
                  f"{'' if pol is None or chosen_k == int(idxs[0]) else f' SUPPRIMÉ -> {chosen_k}'}"
                  f"  [{dt:.2f} s]", flush=True)
        else:
            print(f"  step expected-only (produit expected[{n - 1}]={seq[-1]}) [{dt:.2f} s]", flush=True)
        # Arrêt EOS (prérequis GC8). Sémantique HF (stopping_criteria.py:534, 579-581) : « any of »
        # sans priorité, le token est concaténé PUIS testé — il FAIT DONC PARTIE de la sortie.
        if args.gen_policy_stop and chosen_k in gen_policy["eos_token_id"]:
            stopped_at, stop_eos = k, chosen_k
            print(f"  arrêt EOS à gen={k} (id={chosen_k}, eos={gen_policy['eos_token_id']})", flush=True)
            break

    # Troncature quand l'arrêt a mordu. ⚠ Ce mode ne produit PAS une fixture d'oracle standard :
    # `fed`/`expected` y sont bornés par l'arrêt et non par `--n-tokens`. C'est voulu — GC8 compare
    # des trajectoires ARRÊT COMPRIS — et c'est la raison pour laquelle le drapeau est off par
    # défaut : `u8_gen48` a été produite sans arrêt et doit le rester.
    n_eff = (stopped_at + 1) if stopped_at is not None else n
    if stopped_at is not None:
        top5_ids, top5_vals = top5_ids[:n_eff], top5_vals[:n_eff]
        top5_pol_ids, top5_pol_vals = top5_pol_ids[:n_eff], top5_pol_vals[:n_eff]
    fed = seq[:n_eff]
    expected = seq[1:n_eff + 1]
    # Chrono D6 : médiane des tokens 2-4 (le token 0 = prefill mmap/warm-up, le 1 = encore chaud).
    med_2_4 = statistics.median(times[2:5])
    min_margin = min(margins)
    argmin = margins.index(min_margin)
    print(f"chrono : médiane tokens 2-4 = {med_2_4:.2f} s/token (D6) ; total {sum(times):.1f} s", flush=True)
    print(f"marges : min={min_margin:.6f} @ gen={argmin} (piège 17 : lues AVANT tout verdict)", flush=True)
    eot_pos = fed.index(eot_id) if eot_id in fed else None
    print(f"EOT ({eot_id}) dans fed : {'@ gen=' + str(eot_pos) if eot_pos is not None else 'absent'}", flush=True)
    print(f"aperçu réponse HF : {tok.decode(fed, skip_special_tokens=True)[:300]!r}", flush=True)

    positions = torch.arange(S, S + n, dtype=torch.int32)  # positions[0] == S (check runner)
    print(f"politique : a mordu {n_policy_bites} fois sur {n} tokens "
          f"({'ACTIVE' if proc is not None else 'INACTIVE'})", flush=True)
    # Non-vacuité de la penalty, côté oracle : une fixture produite par une penalty qui n'a
    # jamais rien changé ferait passer RP3 sans rien prouver (elle serait identique à la fixture
    # neutre). On le VOIT ici, avant que le runner ne soit comparé à elle.
    if penalty_proc is not None:
        print(f"PENALTY: n_penalty_touched={n_penalty_touched} sur {n_eff} steps", flush=True)
        assert n_penalty_touched > 0, \
            "penalty armée mais AUCUN logit changé sur toute la trajectoire : fixture inexploitable"
    tensors = {
        "positions": positions.contiguous(),
        "fed": torch.tensor(fed, dtype=torch.int32),
        "expected": torch.tensor(expected, dtype=torch.int32),
        # ⚠ prompt_ids en TENSEUR, pas seulement au manifest JSON : le runner ne lit pas le JSON,
        # et sans ce tenseur il ne peut pas vérifier LITTÉRALEMENT qu'il a été templaté sur le
        # même prompt que l'oracle. Un désaccord de template produirait un mismatch d'ids
        # attribué à tort à la penalty.
        "prompt_ids": prompt_ids.to(torch.int32).contiguous(),
        # `top5_ids`/`top5_vals` restent LE BRUT (schéma inchangé, plan 4.3) : les gates
        # historiques les lisent, et la marge brute est l'instrument de requalification
        # pré-enregistré de U8/W4g. Le post-politique s'AJOUTE, il ne remplace pas.
        "top5_ids": top5_ids.contiguous(),
        "top5_vals": top5_vals.contiguous(),
    }
    if proc is not None:
        tensors["top5_policy_ids"] = top5_pol_ids.contiguous()
        tensors["top5_policy_vals"] = top5_pol_vals.contiguous()
    save_file(tensors, args.out)

    manifest = {
        "source": "69_u8_gen_oracle.py (mode décode — Task 9 Step 9.1, plan J2)",
        "oracle": "décode greedy HF CPU, modèle RÉEL bf16 (AutoModelForCausalLM sur l'export dq, "
                  "low_cpu_mem_usage/mmap) — chemin U0b-décode/U7 ; logits softcappés par HF",
        "prompt": args.prompt,
        "prompt_ids": prompt_ids.tolist(),
        "seq_len": S,
        "n_decode": n,
        # Phase 1 : le réglage qui a produit CETTE trajectoire. Sans lui au manifest, deux
        # fixtures de penalties différentes seraient indiscernables une fois sur le disque.
        "repetition_penalty": float(args.repetition_penalty),
        "n_penalty_touched": n_penalty_touched,
        "fed_head": fed[:8], "expected_head": expected[:8],
        "eot_id": eot_id, "eot_pos_in_fed": eot_pos,
        "reponse_hf": tok.decode(fed, skip_special_tokens=True),
        "margins": {"per_step": [round(m, 6) for m in margins],
                    "min": round(min_margin, 6), "argmin": argmin,
                    "note": "piège 17 — marges consignées AVANT tout verdict gate"},
        "softcap": {"cap": SOFTCAP, "max_abs_per_step_max": round(max(max_abs_all), 4),
                    "asserted": "max|logits| <= 30 à chaque step"},
        "chrono_s": {"load": round(t_load, 1), "per_token": [round(t, 2) for t in times],
                     "median_tokens_2_4": round(med_2_4, 2), "host": args.host_label,
                     "note": "D6 — token 0 = prefill (mmap+warm-up), médiane 2-4 = la mesure"},
        "chat_template_sha256": tpl_sha,
        "weights": anon_path(args.weights),
        "fixture_keys_lues_par_le_runner": ["positions", "fed"],
        "gen_policy": {**gen_policy, "n_bites": n_policy_bites,
                       "stop_applied": bool(args.gen_policy_stop),
                       "stopped_at_gen": stopped_at, "stop_eos_id": stop_eos},
        "compute_dtype": compute_dtype,
        "script_md5": script_md5(),
        "versions": versions,
    }
    with open(args.out + ".manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    print(f"PASS fixture U8 écrite : {args.out} (S={S}, n={n}, marge_min={min_margin:.4f} @ {argmin})", flush=True)


def mode_teacher_force(args, model, tok, tpl_sha, t_load, versions):
    """U9-iv (D10) : UN prefill de [prompt ++ ids[:-1]] ; argmax+top-5 par position au fil de l'eau."""
    proc, gen_policy = build_gen_policy(model, args.no_gen_policy)
    print(f"GENPOLICY: {gen_policy}", flush=True)
    _, prompt_ids, S = encode_prompt(tok, args.prompt)
    with safe_open(args.teacher_force, framework="pt") as f:
        gen = f.get_tensor("ids").to(torch.long)
    N = int(gen.shape[0])
    assert N >= 2, f"ids fixture suspecte : N={N}"
    print(f"teacher-force : {N} ids générés (runner --out-ids), prefill de S+N-1={S + N - 1} positions", flush=True)

    full = torch.cat([prompt_ids.to(torch.long), gen[:-1]]).unsqueeze(0)
    T = int(full.shape[1])

    compute_dtype = "bfloat16"
    if args.compute_fp32:
        n_hooked = install_fp32_hooks(model.model, extra=(model.lm_head,))
        compute_dtype = "float32"
        print(f"calcul fp32 armé : {n_hooked} sous-modules hookés dont lm_head "
              f"(stockage bf16 intact, activations fp32)", flush=True)

    # --- témoin fenêtre (D10/plan Step 10.3) : à T > sliding_window le masque sliding HF doit
    # DIFFÉRER du causal (fenêtre mordante) — même mécanique réelle que le témoin du 68.
    cfg_t = model.config.get_text_config()
    sw = int(cfg_t.sliding_window)
    window_bites = None
    if T > sw:
        from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
        import copy
        cfg_mask = copy.deepcopy(cfg_t)
        cfg_mask._attn_implementation = "eager"
        dummy = torch.zeros(1, T, cfg_t.hidden_size, dtype=torch.float32)
        pos = torch.arange(T, dtype=torch.long).unsqueeze(0)
        mask_kw = dict(config=cfg_mask, inputs_embeds=dummy, attention_mask=None,
                       past_key_values=None, position_ids=pos)
        window_bites = not torch.equal(create_sliding_window_causal_mask(**mask_kw),
                                       create_causal_mask(**mask_kw))
        assert window_bites, f"T={T} > fenêtre {sw} mais masque sliding == causal : fenêtre NON mordante en prefill"
        print(f"témoin fenêtre : masque sliding != causal à T={T} (fenêtre {sw} mordante en prefill)", flush=True)

    # --- backbone : UN prefill (use_cache=False), hidden states post-norm ---
    t1 = time.monotonic()
    with torch.no_grad():
        hs = model.model(input_ids=full, use_cache=False).last_hidden_state  # [1, T, 3840] bf16
    t_fwd = time.monotonic() - t1
    print(f"prefill backbone T={T} en {t_fwd:.1f} s", flush=True)

    cap = float(cfg_t.final_logit_softcapping)

    def head(h):  # miroir EXACT modeling_gemma4.py l.2592-2597 : lm_head -> /cap -> tanh -> ×cap
        lg = model.lm_head(h)
        return torch.tanh(lg / cap) * cap

    # --- self-check tête manuelle == forward HF complet (préfixe court) ---
    # bf16 : torch.equal (l'arrondi bf16 final gomme l'ordre d'accumulation BLAS).
    # fp32 : max|d| ≤ 1e-3 — l'accumulation fp32 dépend de la forme du matmul
    # (logits_to_keep=1 slice AVANT lm_head), bruit ~1e-4 mesuré (précédent E2B P4.3 :
    # même classe, 1.5e-5) ; le bit-exact inter-formes n'existe pas en fp32.
    with torch.no_grad():
        want = model(input_ids=full[:, :8], logits_to_keep=1).logits[0, -1, :]
        got = head(model.model(input_ids=full[:, :8], use_cache=False).last_hidden_state)[0, -1, :]
    if args.compute_fp32:
        d = float((got.float() - want.float()).abs().max())
        assert d <= 1e-3, f"self-check tête manuelle != forward HF (max|d|={d} > 1e-3)"
        print(f"self-check tête : manuelle == forward HF (fp32, max|d|={d:.2e} ≤ 1e-3, préfixe 8)", flush=True)
    else:
        assert torch.equal(got, want), \
            f"self-check tête manuelle != forward HF (max|d|={float((got.float() - want.float()).abs().max())})"
        print("self-check tête : manuelle == forward HF (bit-égal, préfixe 8)", flush=True)

    # --- balayage au fil de l'eau : positions S-1 .. T-1 prédisent gen[0..N-1] ---
    t2 = time.monotonic()
    n_match = 0
    margins, mismatches, top5_all = [], [], []
    top5_policy_all, n_policy_bites = [], 0
    for lo in range(S - 1, T, HEAD_CHUNK):
        hi = min(lo + HEAD_CHUNK, T)
        with torch.no_grad():
            lg = head(hs[:, lo:hi, :]).float()[0]  # [c, V]
        vals, idxs = torch.topk(lg, TOPK, dim=-1)
        # Politique APRÈS le topk brut, sur une COPIE (plan 4.2) : `top5_per_pos` doit rester le
        # brut, sinon les gates historiques liraient autre chose que ce qu'ils lisaient hier.
        lg_p = apply_policy(lg, proc)
        pvals, pidxs = None, None
        if lg_p is not None:
            pvals, pidxs = torch.topk(lg_p, TOPK, dim=-1)
            assert torch.isfinite(pvals).all(), "top-5 POST-politique contient un -inf"
        for j in range(hi - lo):
            k = lo - (S - 1) + j
            # `got_id` = l'argmax de la politique EFFECTIVE. C'est lui qui doit être confronté aux
            # ids du runner : le runner corrigé applique la même politique, et c'est tout l'objet
            # du chantier que les deux côtés cessent de faire un argmax nu chacun de son côté.
            got_id = int(pidxs[j, 0]) if lg_p is not None else int(idxs[j, 0])
            want_id = int(gen[k])
            margin = float(vals[j, 0] - vals[j, 1])
            margins.append(round(margin, 6))
            top5_all.append({"ids": idxs[j].tolist(), "vals": [round(float(v), 4) for v in vals[j]]})
            if lg_p is not None:
                top5_policy_all.append({"ids": pidxs[j].tolist(),
                                        "vals": [round(float(v), 4) for v in pvals[j]]})
                if int(pidxs[j, 0]) != int(idxs[j, 0]):
                    n_policy_bites += 1
            if got_id == want_id:
                n_match += 1
            else:
                mismatches.append({"pos_gen": k, "got": got_id, "want": want_id,
                                   "margin_top1_top2": round(margin, 6),
                                   "top5_ids": idxs[j].tolist(),
                                   "top5_vals": [round(float(v), 6) for v in vals[j]]})
    t_head = time.monotonic() - t2

    print(f"U9-iv : {n_match}/{N} argmax == ids runner ; {len(mismatches)} divergence(s)", flush=True)
    for mm in mismatches[:16]:
        print(f"  divergence @ gen={mm['pos_gen']} : got={mm['got']} want={mm['want']} "
              f"marge={mm['margin_top1_top2']}", flush=True)
    print(f"marges : min={min(margins):.6f} @ gen={margins.index(min(margins))}", flush=True)

    report = {
        "source": "69_u8_gen_oracle.py (mode --teacher-force — Task 10 Step 10.3, plan J2, D10)",
        "oracle": f"UN prefill HF CPU (calcul {compute_dtype}, stockage bf16) de [prompt templaté "
                  "++ ids[:-1]] ; tête lm_head+softcap par chunks (miroir modeling_gemma4.py, "
                  "self-check bit-égal câblé)"
                  + (" ; --compute-fp32 : hooks par-couche, parité oracles J1 46/56"
                     if args.compute_fp32 else ""),
        "compute_dtype": compute_dtype,
        "prompt": args.prompt, "prompt_ids": prompt_ids.tolist(), "seq_len": S,
        "ids_fixture": anon_path(args.teacher_force), "n_ids": N, "prefill_len": T,
        "window": {"sliding_window": sw, "bites_in_prefill": window_bites},
        "verdict_data": {"n_match": n_match, "n_total": N, "mismatches": mismatches,
                         "note": "N<total : protocole de marge J1 par position — décision Régis, "
                                 "pas ce script (Amendement 2)"},
        "margins_per_pos": margins,
        "top5_per_pos": top5_all,
        # Ajouté, jamais substitué : `top5_per_pos` reste le BRUT (plan 4.3).
        **({"top5_policy_per_pos": top5_policy_all} if proc is not None else {}),
        "gen_policy": {**gen_policy, "n_bites": n_policy_bites},
        "script_md5": script_md5(),
        "chrono_s": {"load": round(t_load, 1), "prefill_backbone": round(t_fwd, 1),
                     "head_sweep": round(t_head, 1), "host": args.host_label},
        "chat_template_sha256": tpl_sha, "weights": anon_path(args.weights),
        "versions": versions,
    }
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"rapport écrit : {args.out} ({n_match}/{N})", flush=True)


def mode_context(args, model, tok, tpl_sha, t_load, versions):
    """K5/PF1 — teacher-forcé À TRAVERS LA FRONTIÈRE d'une reprise de cache.

    Différence unique avec --teacher-force : le contexte n'est PAS un prompt templaté mais la
    séquence d'ids `ctx_ids` que le runner a réellement feedée (sous reprise, aucun --prompt ne
    peut l'exprimer — elle contient le prompt du run A, ses tokens générés, fed_next, la clôture
    injectée et le tour 2). Tout le reste est le mode teacher-force : UN prefill, tête manuelle
    par chunks, self-check, témoin fenêtre, marges consignées AVANT tout verdict.

    L'oracle PUBLIE, il ne juge pas : les DEUX canaux sont écrits pour chaque position, et c'est
    le dépouilleur 80 qui choisit lequel confronter à quoi. Raison mesurée (dc4.err.log:16) : le
    top-5 du runner est BRUT (in-graph), tandis que `generated` est POST-politique — à gen=0
    d'une reprise réelle, l'argmax brut 258882 était supprimé et le choisi valait 4509. Un canal
    unique produirait un FAIL gras purement spurieux."""
    proc, gen_policy = build_gen_policy(model, args.no_gen_policy)
    print(f"GENPOLICY: {gen_policy}", flush=True)
    with safe_open(args.context_ids, framework="pt") as f:
        keys = set(f.keys())
        assert {"ids", "ctx_ids"} <= keys, \
            f"--context-ids : clés {sorted(keys)} — 'ids' ET 'ctx_ids' requises (runner K5 --out-ids sous reprise AVEC prompt)"
        ctx = f.get_tensor("ctx_ids").to(torch.long)
        gen = f.get_tensor("ids").to(torch.long)
    C, m = int(ctx.shape[0]), int(gen.shape[0])
    assert m >= 2, f"ids fixture suspecte : m={m}"
    assert args.ctx_from is not None and 0 < args.ctx_from < C, \
        f"--ctx-from (step_next) requis et < len(ctx_ids)={C} (reçu {args.ctx_from})"

    # position p prédit full[p+1] ; la dernière (p = T-1) prédit gen[-1], qui n'est pas dans full.
    full = torch.cat([ctx, gen[:-1]]).unsqueeze(0)
    T = int(full.shape[1])
    print(f"context : C={C} ids de contexte, m={m} générés, prefill de T={T} positions ; "
          f"rapport à partir de p={args.ctx_from}", flush=True)

    compute_dtype = "bfloat16"
    if args.compute_fp32:
        n_hooked = install_fp32_hooks(model.model, extra=(model.lm_head,))
        compute_dtype = "float32"
        print(f"calcul fp32 armé : {n_hooked} sous-modules hookés dont lm_head", flush=True)

    # --- témoin fenêtre (identique au mode teacher-force) : à T > sliding_window le masque
    # sliding doit DIFFÉRER du causal, sinon PF3 serait INEXÉCUTABLE (pas PASS).
    cfg_t = model.config.get_text_config()
    sw = int(cfg_t.sliding_window)
    window_bites = None
    if T > sw:
        from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
        import copy
        cfg_mask = copy.deepcopy(cfg_t)
        cfg_mask._attn_implementation = "eager"
        dummy = torch.zeros(1, T, cfg_t.hidden_size, dtype=torch.float32)
        pos = torch.arange(T, dtype=torch.long).unsqueeze(0)
        mask_kw = dict(config=cfg_mask, inputs_embeds=dummy, attention_mask=None,
                       past_key_values=None, position_ids=pos)
        window_bites = not torch.equal(create_sliding_window_causal_mask(**mask_kw),
                                       create_causal_mask(**mask_kw))
        assert window_bites, f"T={T} > fenêtre {sw} mais masque sliding == causal"
        print(f"témoin fenêtre : masque sliding != causal à T={T} (fenêtre {sw} mordante)", flush=True)

    t1 = time.monotonic()
    with torch.no_grad():
        hs = model.model(input_ids=full, use_cache=False).last_hidden_state
    t_fwd = time.monotonic() - t1
    print(f"prefill backbone T={T} en {t_fwd:.1f} s", flush=True)

    cap = float(cfg_t.final_logit_softcapping)

    def head(h):  # miroir EXACT modeling_gemma4.py : lm_head -> /cap -> tanh -> ×cap
        lg = model.lm_head(h)
        return torch.tanh(lg / cap) * cap

    with torch.no_grad():
        want = model(input_ids=full[:, :8], logits_to_keep=1).logits[0, -1, :]
        got = head(model.model(input_ids=full[:, :8], use_cache=False).last_hidden_state)[0, -1, :]
    if args.compute_fp32:
        d = float((got.float() - want.float()).abs().max())
        assert d <= 1e-3, f"self-check tête manuelle != forward HF (max|d|={d} > 1e-3)"
        print(f"self-check tête : manuelle == forward HF (fp32, max|d|={d:.2e}, préfixe 8)", flush=True)
    else:
        assert torch.equal(got, want), "self-check tête manuelle != forward HF (bf16)"
        print("self-check tête : manuelle == forward HF (bit-égal, préfixe 8)", flush=True)

    t2 = time.monotonic()
    positions = []
    for lo in range(args.ctx_from, T, HEAD_CHUNK):
        hi = min(lo + HEAD_CHUNK, T)
        with torch.no_grad():
            lg = head(hs[:, lo:hi, :]).float()[0]  # [c, V]
        vals, idxs = torch.topk(lg, TOPK, dim=-1)
        lg_p = apply_policy(lg, proc)
        pvals, pidxs = (None, None)
        if lg_p is not None:
            pvals, pidxs = torch.topk(lg_p, TOPK, dim=-1)
            assert torch.isfinite(pvals).all(), "top-5 POST-politique contient un -inf"
        for j in range(hi - lo):
            p = lo + j
            e = {"p": p,
                 "argmax_raw": int(idxs[j, 0]),
                 "margin_raw": round(float(vals[j, 0] - vals[j, 1]), 6),
                 "top5_raw_ids": idxs[j].tolist(),
                 "top5_raw_vals": [round(float(v), 4) for v in vals[j]],
                 # `fed_at_next` : le token que le runner a REELLEMENT feedé/produit en p+1.
                 # Publié pour le diagnostic seulement — en prefill il est IMPOSÉ, l'exiger
                 # égal à l'argmax serait exiger que le modèle prédise le prompt.
                 "fed_at_next": int(full[0, p + 1]) if p + 1 < T else int(gen[-1])}
            if lg_p is not None:
                e["argmax_policy"] = int(pidxs[j, 0])
                e["margin_policy"] = round(float(pvals[j, 0] - pvals[j, 1]), 6)
            else:
                e["argmax_policy"] = e["argmax_raw"]
                e["margin_policy"] = e["margin_raw"]
            positions.append(e)
    t_head = time.monotonic() - t2

    raw_ctx = [e["margin_raw"] for e in positions if e["p"] <= C - 2]
    pol_gen = [e["margin_policy"] for e in positions if e["p"] >= C - 1]
    # Marges AVANT tout verdict (piège 17) — et ce script n'en rend aucun.
    if raw_ctx:
        print(f"marges ctx (raw)    : min={min(raw_ctx):.6f} médiane={statistics.median(raw_ctx):.6f} "
              f"sur {len(raw_ctx)} positions", flush=True)
    if pol_gen:
        print(f"marges gen (policy) : min={min(pol_gen):.6f} médiane={statistics.median(pol_gen):.6f} "
              f"sur {len(pol_gen)} positions", flush=True)

    report = {
        "source": "69_u8_gen_oracle.py (mode --context-ids — K5/PF1, spec 2026-08-10 §4.5)",
        "oracle": "UN prefill HF CPU de [ctx_ids ++ ids[:-1]] ; tête lm_head+softcap par chunks ; "
                  "DEUX canaux publiés par position (brut in-graph / post-politique) — le juge "
                  "est le script 80, pas celui-ci",
        "mode": "context", "compute_dtype": compute_dtype,
        "ctx_from": args.ctx_from, "C": C, "m": m, "prefill_len": T,
        "context_fixture": anon_path(args.context_ids),
        "window": {"sliding_window": sw, "bites_in_prefill": window_bites},
        "positions": positions,
        "gen_policy": gen_policy,
        "script_md5": script_md5(),
        "chrono_s": {"load": round(t_load, 1), "prefill_backbone": round(t_fwd, 1),
                     "head_sweep": round(t_head, 1), "host": args.host_label},
        "chat_template_sha256": tpl_sha, "weights": anon_path(args.weights),
        "versions": versions,
    }
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"rapport écrit : {args.out} ({len(positions)} positions, aucun verdict)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights", default=os.path.join(ROOT, "weights_12b_dq"),
                    help="export dq (D9 : oracles Python = export, jamais le packé)")
    # K5 : `required=True` est LEVÉ — le mode --context-ids tire son contexte des ids du runner,
    # pas d'un prompt templaté (sous reprise, aucun --prompt ne peut exprimer ce contexte).
    # La garde est déplacée après le parse : les DEUX autres modes l'exigent toujours.
    ap.add_argument("--prompt", default=None,
                    help="prompt user (REQUIS en mode décode et en --teacher-force : contexte du "
                         "prefill ; INTERDIT en --context-ids, où le contexte EST ctx_ids)")
    ap.add_argument("--n-tokens", type=int, default=48, help="mode décode : tokens greedy (48 = U8)")
    ap.add_argument("--out", required=True,
                    help="décode : fixture .safetensors (+ .manifest.json) ; teacher-force : rapport .json")
    ap.add_argument("--teacher-force", default=None, metavar="IDS_SAFETENSORS",
                    help="U9-iv : safetensors clé 'ids' (i32, --out-ids du runner) — UN prefill, "
                         "argmax+top-5 par position au fil de l'eau")
    ap.add_argument("--context-ids", default=None, metavar="OUT_IDS_SAFETENSORS",
                    help="K5/PF1 : safetensors du runner à DEUX clés (ids = générés, ctx_ids = "
                         "séquence complète feedée) — teacher-forcé à travers la frontière d'une "
                         "reprise ; exclusif de --prompt/--teacher-force")
    ap.add_argument("--ctx-from", type=int, default=None,
                    help="K5 : step_next — première position ABSOLUE à rapporter ; requis avec "
                         "--context-ids")
    ap.add_argument("--compute-fp32", action="store_true",
                    help="teacher-force uniquement : calcul fp32 sur stockage bf16 (hooks "
                         "par-couche) — restaure la parité d'instrument avec les oracles J1 "
                         "(46/56 : poids fp32) que le 12B monolithique interdit (RAM)")
    # Défaut = politique ON. C'est le sens du chantier : l'oracle sans politique était l'ANGLE
    # MORT (il faisait le même argmax nu que le runner). Le flag existe pour le contre-test à un
    # seul facteur de GC5 — deux runs du même script, même md5, un seul drapeau de différence.
    ap.add_argument("--no-gen-policy", action="store_true",
                    help="désactive la politique de décodage (suppress_tokens) — contre-test "
                         "GC5 : restaure l'argmax nu d'avant le chantier")
    # Prérequis GC8 (plan 6.2) : `mode_decode` n'a AUCUN arrêt EOS — l'eot_id y est « informatif
    # seulement » (le mode --oracle du runner ne s'arrête pas non plus). GC8 compare des
    # trajectoires ARRÊT COMPRIS, il lui faut donc un arrêt. Défaut OFF pour ne pas casser la
    # reproduction de `u8_gen48`, qui a été produite sans arrêt.
    ap.add_argument("--gen-policy-stop", action="store_true",
                    help="mode décode : s'arrêter sur les eos_token_id de la politique "
                         "(défaut off — u8_gen48 a été produite sans arrêt)")
    ap.add_argument("--host-label", default="M4",
                    help="étiquette chrono D6 (M4|VM) — JAMAIS le hostname réel (anonymisation)")
    ap.add_argument("--repetition-penalty", type=float, default=1.0,
                    help="phase 1 : RepetitionPenaltyLogitsProcessor HF appliqué en tête de "
                         "chaîne (avant suppress_tokens). 1.0 = neutre. Historique = prompt ++ "
                         "généré, le défaut HF.")
    args = ap.parse_args()
    # ⚠ La garde « --compute-fp32 n'existe qu'en --teacher-force » est LEVÉE (prérequis GC8,
    # plan 6.1). Raison : GC8 compare la trajectoire libre du runner corrigé à un décodage HF
    # greedy appliquant la même politique, ARRÊT COMPRIS, et son seuil de kill (1,873e-3) a été
    # dérivé en fp32. Le laisser tourner en bf16 aurait été un CHANGEMENT D'INSTRUMENT
    # SILENCIEUX — précisément la faute qui a produit les requalifications en cascade de J2.
    # Le mode décode par DÉFAUT reste bf16 : la reproduction de `u8_gen48` n'est pas touchée.
    # K5 : `--context-ids` est un mode teacher-forcé, pas un décodage — l'avertissement ci-dessous
    # y serait FAUX, et un log de gate qui se décrit mal trompe sa propre relecture.
    if args.compute_fp32 and not args.teacher_force and not args.context_ids:
        print("⚠ --compute-fp32 en mode DÉCODE : instrument fp32 (hooks par-module) sur un "
              "décodage autonome. Consigné au manifest ; NE reproduit PAS `u8_gen48` (bf16).",
              flush=True)

    # K5 : exclusivité et prérequis des modes, refusés AVANT le chargement du modèle (2 minutes
    # de mmap et 24 Gio de RAM ne doivent pas être dépensés pour découvrir un flag manquant).
    if args.context_ids and args.teacher_force:
        sys.exit("--context-ids et --teacher-force sont exclusifs (deux définitions du contexte)")
    if args.context_ids and args.prompt is not None:
        sys.exit("--context-ids : --prompt est INTERDIT — le contexte EST ctx_ids ; un prompt "
                 "templaté ne peut pas exprimer une reprise de cache")
    if not args.context_ids and args.prompt is None:
        sys.exit("--prompt est requis (modes décode et --teacher-force)")

    model, tok, tpl_sha, t_load, versions = load_model_and_tok(args.weights)
    if args.context_ids:
        mode_context(args, model, tok, tpl_sha, t_load, versions)
    elif args.teacher_force:
        mode_teacher_force(args, model, tok, tpl_sha, t_load, versions)
    else:
        mode_decode(args, model, tok, tpl_sha, t_load, versions)


if __name__ == "__main__":
    main()
