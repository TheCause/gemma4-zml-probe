#!/usr/bin/env python3
"""
measure_k_distribution_gemma4.py

Mesure la distribution du tenseur K de google/gemma-4-E2B-it au POINT CANONIQUE
(post-k_norm, post-RoPE, AVANT repeat_kv, AVANT cache-write) et produit un VERDICT
comparable a la grille TurboQuant :
    Qwen2.5-7B  layer0/head0 : var=1699  amax=172   post-Had=76.4   -> CATASTROPHE
    Gemma3-4B   layer0/head0 : var=2.80  amax=20    post-Had=7.0    -> STABLE

Methodologie (synthese de 3 analyses) :
  - Point de capture = sortie de apply_rotary_pos_emb sur K (var k_after_rope),
    layout [B, S, n_kv, head_dim] AVANT transpose/repeat_kv. C'est exactement le K
    ecrit dans le KV-cache (cf P5_0 §5.3, P5_2 L500).
  - Capture par MONKEYPATCH de transformers...modeling_gemma4.apply_rotary_pos_emb.
    Robuste face aux DEUX signatures possibles (Gemma4 single-tensor +unsqueeze_dim,
    et HF-standard q,k -> q_embed,k_embed). On distingue K de Q par n_kv=1 vs n_q=8.
  - Producers uniquement (layers 0..14). Layers 15..34 sont des readers YOCO sans
    k_proj/k_norm instancie : aucun K propre a mesurer.
  - head_dim heterogene : 256 (sliding) vs 512 (full attention {4,9,14}). La capture
    lit head_dim depuis la shape reelle ; aucune hypothese 256 codee en dur.
  - GQA 8:1, num_key_value_heads=1 : on mesure l'unique tete KV, JAMAIS post-repeat_kv.
  - Metriques IDENTIQUES a profile_qwen_l0.py :
        var       = (samples**2).mean()          (moment d'ordre 2 NON centre)
        amax      = samples.abs().max()           (scalaire global tokens x canaux)
        post-Had  = (samples @ H).abs().max()     H = Sylvester /sqrt(d), PAS de norm L2
  - dtype : modele charge bf16 (reflete le runtime reel quantifie), K capture force
    en fp32 pour les stats (comme bench_v4.capture_kv). Voir note dtype plus bas.

Output : print detaille + JSON dans /data/gemma4-zml-probe/k_distribution_gemma4_e2b.json
RIEN n'est ecrit sur / (interdit, / a 90%).
"""
from __future__ import annotations

import json
import math
import os
import sys
import traceback
from pathlib import Path

# --- Environnement : cache HF sur /data, jamais sur / ---------------------
os.environ.setdefault("HF_HOME", "/data/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")        # modele deja en cache
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

REPO = "google/gemma-4-E2B-it"
OUT_DIR = Path("/data/gemma4-zml-probe")
OUT_JSON = OUT_DIR / "k_distribution_gemma4_e2b.json"

# Grille de reference TurboQuant (layer0/head0), source turboquant.md L136-139.
REFERENCE_GRID = {
    "Qwen2.5-7B": {"var": 1699.0, "amax": 172.0, "post_had_amax": 76.4,
                   "verdict": "CATASTROPHE (skip_0 fp16 / PolarQuant requis)"},
    "Gemma3-4B": {"var": 2.80, "amax": 20.0, "post_had_amax": 7.0,
                  "verdict": "STABLE (V4_Had standard viable)"},
}

# Corpus de calibration : Pride and Prejudice, VERBATIM le CORPUS de
# bench_end2end.py utilise pour generer la grille (comparabilite stricte).
CORPUS = (
    "It is a truth universally acknowledged, that a single man in possession of a "
    "good fortune, must be in want of a wife. However little known the feelings or "
    "views of such a man may be on his first entering a neighbourhood, this truth is "
    "so well fixed in the minds of the surrounding families, that he is considered as "
    "the rightful property of some one or other of their daughters.\n\n"
    "My dear Mr. Bennet, said his lady to him one day, have you heard that Netherfield "
    "Park is let at last? Mr. Bennet replied that he had not. But it is, returned she; "
    "for Mrs. Long has just been here, and she told me all about it. Mr. Bennet made no "
    "answer. Do not you want to know who has taken it? cried his wife impatiently. You "
    "want to tell me, and I have no objection to hearing it. This was invitation enough.\n\n"
    "Why, my dear, you must know, Mrs. Long says that Netherfield is taken by a young man "
    "of large fortune from the north of England; that he came down on Monday in a chaise "
    "and four to see the place, and was so much delighted with it, that he agreed with Mr. "
    "Morris immediately; that he is to take possession before Michaelmas, and some of his "
    "servants are to be in the house by the end of next week. What is his name? Bingley. Is "
    "he married or single? Oh! Single, my dear, to be sure! A single man of large fortune; "
    "four or five thousand a year. What a fine thing for our girls!\n\n"
    "How so? How can it affect them? My dear Mr. Bennet, replied his wife, how can you be so "
    "tiresome! You must know that I am thinking of his marrying one of them. Is that his "
    "design in settling here? Design! Nonsense, how can you talk so! But it is very likely "
    "that he may fall in love with one of them, and therefore you must visit him as soon as "
    "he comes. I see no occasion for that.\n\n"
    "But consider your daughters. Only think what an establishment it would be for one of "
    "them. Sir William and Lady Lucas are determined to go, merely on that account, for in "
    "general, you know they visit no new comers. Indeed you must go, for it will be "
    "impossible for us to visit him, if you do not."
)

N_TOKENS = 256
PRODUCER_LAYERS = list(range(15))   # 0..14 (first_kv_shared = 35 - 20 = 15)
FULL_LAYERS = {4, 9, 14, 19, 24, 29, 34}  # full_attention; producers full = {4,9,14}


# ==========================================================================
# Metriques TurboQuant — replique EXACTE de profile_qwen_l0.py
# ==========================================================================
def build_hadamard(d: int) -> torch.Tensor:
    """Sylvester Hadamard normalisee 1/sqrt(d). Identique a build_hadamard de
    profile_qwen_l0.py (L28-32). d DOIT etre puissance de 2."""
    if d <= 0 or (d & (d - 1)) != 0:
        raise ValueError(f"Hadamard requires d power of 2, got d={d}")
    H = torch.tensor([[1.0]])
    while H.shape[0] < d:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(d)


def channel_stats(samples_2d: torch.Tensor):
    """samples_2d: [N, d] fp32. Stats par canal (dim=0 = sur les tokens)."""
    abs_s = samples_2d.abs()
    amax = abs_s.max(dim=0)[0]   # [d]
    p50 = abs_s.median(dim=0)[0]
    return amax, p50


def topk_outlier_channels(amax, p50, k=5):
    ratio = amax / p50.clamp_min(1e-9)
    top_idx = torch.topk(ratio, k=min(k, ratio.numel())).indices
    return [(int(i), float(ratio[i]), float(amax[i]), float(p50[i]))
            for i in top_idx.tolist()]


def head_metrics(samples: torch.Tensor) -> dict:
    """samples: [N, d] fp32 (une tete KV). Calcule var/amax/post-Had EXACTEMENT
    comme la grille TurboQuant. var = mean-square NON centre."""
    d = samples.shape[-1]
    H_mat = build_hadamard(d).to(samples.device, samples.dtype)
    samples_rot = samples @ H_mat   # AUCUNE normalisation L2 (diagnostic grille)

    amax_pre, p50_pre = channel_stats(samples)
    amax_post, p50_post = channel_stats(samples_rot)

    var = float((samples ** 2).mean())             # moment 2 NON centre
    amax = float(samples.abs().max())              # scalaire global
    post_had_amax = float(samples_rot.abs().max())

    return {
        "var": var,
        "amax": amax,
        "post_had_amax": post_had_amax,
        "amax_max_ratio_pre": float(amax_pre.max() / amax_pre.median().clamp_min(1e-9)),
        "amax_max_ratio_post": float(amax_post.max() / amax_post.median().clamp_min(1e-9)),
        "top5_outlier_channels_pre": topk_outlier_channels(amax_pre, p50_pre, k=5),
        "top5_outlier_channels_post": topk_outlier_channels(amax_post, p50_post, k=5),
        "std_codebook_ref": 1.0 / math.sqrt(d),  # N(0,1/d) ref pour situer post-Had
        "n_samples": int(samples.shape[0]),
        "d": int(d),
    }


# ==========================================================================
# Capture K au point canonique via monkeypatch apply_rotary_pos_emb
# ==========================================================================
class KCapture:
    """Intercepte la sortie K de apply_rotary_pos_emb dans modeling_gemma4.

    Robustesse :
      - Gere la signature Gemma4 single-tensor `apply_rotary_pos_emb(x, cos, sin,
        unsqueeze_dim=2)` (un appel Q [.,.,8,d] puis un appel K [.,.,1,d]).
      - Gere la signature HF-standard `apply_rotary_pos_emb(q, k, cos, sin)` ->
        (q_embed, k_embed) si une variante de transformers l'utilise.
      - On ne retient que les tenseurs K, identifies par n_heads==num_key_value_heads
        a l'axe -2 (=1) AVANT transpose, distinct de Q (n_q=8). Si jamais q et k
        partagent n=1 (pathologique), on departage par ordre d'appel.
      - Chaque appel = un layer (ordre 0,1,2,...). On ne garde que les producers.
    """
    def __init__(self, mod, n_kv: int, n_q: int):
        self.mod = mod
        self.n_kv = n_kv
        self.n_q = n_q
        self.orig = mod.apply_rotary_pos_emb
        self._k_calls = []   # liste ordonnee des K captures [B,S,n_kv,d] (post-RoPE)
        self._raw_call_log = []

    def _is_k_tensor(self, t):
        # K pre-transpose : [B, S, n_kv, head_dim], axe -2 == n_kv (=1), != n_q (=8)
        if not torch.is_tensor(t) or t.dim() != 4:
            return False
        return t.shape[-2] == self.n_kv and self.n_kv != self.n_q

    def __enter__(self):
        orig = self.orig
        self_ref = self

        def patched(*args, **kwargs):
            out = orig(*args, **kwargs)
            try:
                # Cas A : retour single-tensor (Gemma4 style) -> out est K ou Q
                if torch.is_tensor(out):
                    if self_ref._is_k_tensor(out):
                        self_ref._k_calls.append(out.detach().to(torch.float32).cpu())
                        self_ref._raw_call_log.append(("single", list(out.shape), "K"))
                    else:
                        self_ref._raw_call_log.append(("single", list(out.shape), "Q/other"))
                # Cas B : retour tuple (q_embed, k_embed) (HF standard)
                elif isinstance(out, tuple) and len(out) == 2:
                    q_e, k_e = out
                    if torch.is_tensor(k_e):
                        # k_e peut etre [B,S,n_kv,d] OU deja transpose [B,n_kv,S,d]
                        kt = k_e
                        if kt.dim() == 4 and (kt.shape[-2] == self_ref.n_kv or
                                              kt.shape[1] == self_ref.n_kv):
                            self_ref._k_calls.append(kt.detach().to(torch.float32).cpu())
                            self_ref._raw_call_log.append(("tuple", list(kt.shape), "K"))
            except Exception as e:  # ne jamais casser le forward
                self_ref._raw_call_log.append(("error", str(e), ""))
            return out

        self.mod.apply_rotary_pos_emb = patched
        return self

    def __exit__(self, *exc):
        self.mod.apply_rotary_pos_emb = self.orig
        return False

    def k_per_layer(self):
        return self._k_calls


class VCapture:
    """Forward hooks sur les v_norm des producers. V n'a PAS de RoPE dans Gemma4
    (pipeline v_proj -> view -> v_norm -> transpose -> cache). Le point canonique
    = sortie de v_norm (post-norm, pre-transpose, pre-cache-write), layout
    [B, S, n_kv, head_dim]. Seuls les producers (0..14) possedent v_norm."""
    def __init__(self, model):
        self.model = model
        self.handles = []
        self._v = {}   # layer_idx -> V post-vnorm [B,S,n_kv,d] fp32 cpu

    def __enter__(self):
        import re
        for name, mod in self.model.named_modules():
            if not name.endswith("self_attn"):
                continue
            vnorm = getattr(mod, "v_norm", None)
            if vnorm is None:
                continue
            m = re.search(r"layers\.(\d+)\.self_attn$", name)
            if not m:
                continue
            L = int(m.group(1))

            def make_hook(layer_idx):
                def hook(module, inp, out):
                    try:
                        t = out[0] if isinstance(out, tuple) else out
                        self._v[layer_idx] = t.detach().to(torch.float32).cpu()
                    except Exception:
                        pass
                return hook

            self.handles.append(vnorm.register_forward_hook(make_hook(L)))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        return False

    def v_per_layer(self):
        return self._v


def normalize_k_layout(k: torch.Tensor, n_kv: int):
    """Ramene K a [n_kv, S, head_dim] quel que soit le layout capture.
    Accepte [B,S,n_kv,d] (pre-transpose) ou [B,n_kv,S,d] (post-transpose)."""
    assert k.dim() == 4, f"K dim attendu 4, got {k.dim()} (shape {tuple(k.shape)})"
    B = k.shape[0]
    assert B == 1, f"batch attendu 1, got {B}"
    if k.shape[-2] == n_kv:          # [B, S, n_kv, d]
        k = k.squeeze(0).permute(1, 0, 2).contiguous()  # -> [n_kv, S, d]
    elif k.shape[1] == n_kv:         # [B, n_kv, S, d]
        k = k.squeeze(0).contiguous()                   # -> [n_kv, S, d]
    else:
        raise AssertionError(
            f"K layout inattendu {tuple(k.shape)} : aucun axe == n_kv={n_kv}"
        )
    return k   # [n_kv, S, head_dim]


# ==========================================================================
# Chargement modele
# ==========================================================================
def load_model(device: str, dtype):
    cfg = AutoConfig.from_pretrained(REPO)
    tc = cfg.text_config
    # Asserts contrat config (analyses 1/3)
    assert tc.num_key_value_heads == 1, f"n_kv attendu 1, got {tc.num_key_value_heads}"
    assert tc.num_attention_heads == 8, f"n_q attendu 8, got {tc.num_attention_heads}"
    assert tc.num_hidden_layers == 35, f"35 layers attendus, got {tc.num_hidden_layers}"
    assert tc.num_kv_shared_layers == 20, f"20 shared attendus, got {tc.num_kv_shared_layers}"
    assert tc.head_dim == 256, f"head_dim sliding attendu 256, got {tc.head_dim}"
    assert getattr(tc, "global_head_dim", None) == 512, \
        f"global_head_dim (full) attendu 512, got {getattr(tc,'global_head_dim',None)}"
    first_kv = tc.num_hidden_layers - tc.num_kv_shared_layers
    assert first_kv == 15, f"first_kv_shared attendu 15, got {first_kv}"

    tok = AutoTokenizer.from_pretrained(REPO)

    # attn_implementation='eager' : evite tout chemin SDPA fuse, garde
    # apply_rotary_pos_emb au niveau Python (le point de capture). dtype bf16
    # reflete le runtime reel (ce qui est quantifie). On force le K en fp32 a la
    # capture pour des stats fideles.
    #
    # Gemma-4-E2B-it = Gemma4ForConditionalGeneration (multimodal) : AutoModelForCausalLM
    # peut ne pas reconnaitre cette classe -> on essaie plusieurs classes dans l'ordre.
    # Chargement puis .to(device) (pas de device_map -> aucune dependance accelerate).
    def _load(cls):
        # feedback HF runtime fallback : dtype -> torch_dtype selon la version
        try:
            return cls.from_pretrained(REPO, dtype=dtype, attn_implementation="eager")
        except TypeError:
            return cls.from_pretrained(REPO, torch_dtype=dtype, attn_implementation="eager")

    candidates = []
    try:
        from transformers import AutoModelForCausalLM as _A
        candidates.append(("AutoModelForCausalLM", _A))
    except Exception:
        pass
    try:
        from transformers import AutoModelForImageTextToText as _B
        candidates.append(("AutoModelForImageTextToText", _B))
    except Exception:
        pass
    try:
        from transformers import Gemma4ForConditionalGeneration as _C
        candidates.append(("Gemma4ForConditionalGeneration", _C))
    except Exception:
        pass

    model = None
    errs = []
    for _name, _cls in candidates:
        try:
            model = _load(_cls)
            print(f"# modele charge via {_name}")
            break
        except Exception as e:
            errs.append(f"{_name}: {type(e).__name__}: {e}")
    if model is None:
        raise RuntimeError("Echec chargement modele:\n  " + "\n  ".join(errs))

    if device.startswith("cuda"):
        model = model.to(device)
    model.eval()
    return tok, model, cfg, tc


def build_input_ids(tok, device):
    """Input TEXTE PUR (pas de pixel_values/input_features). Chat template natif
    Gemma-4 (modele -it) puis troncature aux 256 premiers tokens. Si le template
    echoue, fallback texte brut (toujours 256 tokens)."""
    used_template = False
    ids = None
    # transformers 5.9 : apply_chat_template(return_tensors='pt')[0] peut renvoyer
    # un tokenizers.Encoding (BatchEncoding indexee par int). On force return_dict
    # et on extrait input_ids -> tensor garanti.
    try:
        messages = [{"role": "user", "content": CORPUS}]
        enc = tok.apply_chat_template(
            messages, add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        )
        ids = enc["input_ids"]
        if ids.dim() == 2:
            ids = ids[0]
        used_template = True
    except Exception:
        ids = None
    if not torch.is_tensor(ids):
        # fallback texte brut
        ids = tok(CORPUS, return_tensors="pt").input_ids[0]
        used_template = False
    ids = ids[:N_TOKENS].unsqueeze(0).to(device)
    return ids, used_template


# ==========================================================================
# Verdict
# ==========================================================================
def verdict_for(var_l0: float) -> tuple[str, str]:
    if var_l0 > 100:
        return ("QWEN-LIKE", (
            "QWEN-LIKE: pathologie catastrophe attendue. Le design quant DOIT gerer "
            "layer 0 (skip_0 fp16, ou PolarQuant). vonly seul insuffisant si on veut "
            "quantifier K."))
    if var_l0 < 10:
        return ("GEMMA3-LIKE", (
            "GEMMA3-LIKE: quant KV standard (V4_Had) probablement viable sans skip. "
            "On fonce."))
    return ("INTERMEDIAIRE", (
        "INTERMEDIAIRE: mesurer NIAH (exact-match passkey, PAS ROUGE-1/MSE) avant "
        "de trancher."))


# ==========================================================================
# Main
# ==========================================================================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    force_fp32 = os.environ.get("FORCE_FP32") == "1"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # bf16 = runtime reel quantifie ; FORCE_FP32=1 -> reference fp32 (exclut artefact bf16)
    dtype = torch.float32 if (force_fp32 or device == "cpu") else torch.bfloat16
    print("=" * 78)
    print("Gemma-4-E2B-it — mesure distribution K (point canonique post-k_norm post-RoPE)")
    print("=" * 78)
    print(f"device={device}  dtype={dtype}")
    print(f"HF_HOME={os.environ.get('HF_HOME')}  out={OUT_JSON}")

    tok, model, cfg, tc = load_model(device, dtype)
    n_kv = tc.num_key_value_heads
    n_q = tc.num_attention_heads
    layer_types = list(tc.layer_types)
    print(f"n_kv_heads={n_kv}  n_q_heads={n_q}  head_dim(sliding)={tc.head_dim} "
          f"global_head_dim(full)={tc.global_head_dim}")
    print(f"producers (mesurables) = layers {PRODUCER_LAYERS[0]}..{PRODUCER_LAYERS[-1]}")

    ids, used_template = build_input_ids(tok, model.device if hasattr(model, "device") else device)
    print(f"input: {ids.shape[1]} tokens  chat_template={used_template}")

    # Localiser le module modeling_gemma4 (point de monkeypatch)
    try:
        import transformers.models.gemma4.modeling_gemma4 as gm
    except Exception as e:
        print(f"FATAL: impossible d'importer modeling_gemma4 : {e}")
        sys.exit(2)
    assert hasattr(gm, "apply_rotary_pos_emb"), \
        "modeling_gemma4 ne contient pas apply_rotary_pos_emb"

    # Forward unique avec capture K (monkeypatch RoPE) ET V (hooks v_norm)
    with KCapture(gm, n_kv=n_kv, n_q=n_q) as cap, VCapture(model) as vcap:
        with torch.no_grad():
            _ = model(ids)
        k_calls = cap.k_per_layer()
        call_log = cap._raw_call_log
        v_by_layer = vcap.v_per_layer()
    print(f"# V captures (hooks v_norm) sur {len(v_by_layer)} producers")

    print(f"\n# apply_rotary_pos_emb : {len(call_log)} appels totaux, "
          f"{len(k_calls)} identifies K")
    if len(k_calls) == 0:
        print("FATAL: aucun tenseur K capture. Log des appels (echantillon):")
        for entry in call_log[:10]:
            print("  ", entry)
        print("  -> verifier signature apply_rotary_pos_emb / attn_implementation.")
        sys.exit(3)

    # Le i-eme appel K correspond au layer i (ordre du forward).
    # On ne garde que les producers (0..14). Defensive : il doit y en avoir 15.
    n_k = len(k_calls)
    print(f"# K captures sur {n_k} appels (attendu >= 15 producers ; "
          f"les readers 15..34 ne re-RoPE pas K et n'en produisent pas).")

    results_layers = []
    var_l0_head0 = None
    var_l0_max = None
    v_var_l0_head0 = None

    for L in PRODUCER_LAYERS:
        if L >= n_k:
            print(f"  WARN: layer {L} absent des captures (n_k={n_k}) — skip")
            continue
        k_raw = k_calls[L]                       # [B,S,n_kv,d] ou [B,n_kv,S,d]
        lt = layer_types[L]
        expected_d = tc.global_head_dim if L in FULL_LAYERS else tc.head_dim

        # --- Asserts defensifs de shape (refus si layout inattendu) ---
        try:
            k = normalize_k_layout(k_raw, n_kv)  # [n_kv, S, d]
        except AssertionError as e:
            print(f"  REFUS layer {L}: {e}")
            continue
        got_nkv, S, got_d = k.shape
        assert got_nkv == n_kv, \
            f"layer {L}: n_kv capture {got_nkv} != attendu {n_kv} " \
            f"(piege repeat_kv ? on aurait du capturer AVANT repeat_kv)"
        assert got_d == expected_d, \
            f"layer {L} ({lt}): head_dim capture {got_d} != attendu {expected_d} " \
            f"(sliding=256 / full=512)"

        layer_entry = {
            "layer": L,
            "layer_type": lt,
            "is_full": L in FULL_LAYERS,
            "head_dim": int(got_d),
            "n_kv_heads": int(got_nkv),
            "seq_len": int(S),
            "captured_layout": list(k_raw.shape),
            "heads": [],
        }

        # --- K : une seule tete KV (n_kv=1) mais on boucle generiquement ---
        head_vars = []
        for h in range(got_nkv):
            samples = k[h]                         # [S, d]
            m = head_metrics(samples)
            m["head"] = h
            layer_entry["heads"].append(m)
            head_vars.append(m["var"])
        layer_entry["var_max_over_heads"] = float(max(head_vars))
        layer_entry["var_head0"] = float(layer_entry["heads"][0]["var"])

        # --- V : post-v_norm, PAS de RoPE (hook v_norm) ---
        layer_entry["v_heads"] = []
        if L in v_by_layer:
            try:
                vt = normalize_k_layout(v_by_layer[L], n_kv)   # [n_kv, S, d]
                assert vt.shape[-1] == expected_d, \
                    f"layer {L}: V head_dim {vt.shape[-1]} != attendu {expected_d}"
                v_vars = []
                for h in range(vt.shape[0]):
                    mv = head_metrics(vt[h])
                    mv["head"] = h
                    layer_entry["v_heads"].append(mv)
                    v_vars.append(mv["var"])
                layer_entry["v_var_head0"] = float(layer_entry["v_heads"][0]["var"])
                layer_entry["v_var_max_over_heads"] = float(max(v_vars))
            except AssertionError as e:
                layer_entry["v_error"] = str(e)
                print(f"  WARN V layer {L}: {e}")
        else:
            layer_entry["v_error"] = "v_norm non capture"

        results_layers.append(layer_entry)

        if L == 0:
            var_l0_head0 = layer_entry["var_head0"]
            var_l0_max = layer_entry["var_max_over_heads"]
            v_var_l0_head0 = layer_entry.get("v_var_head0")

    # ----------------------------------------------------------------------
    # VERDICT (sur layer 0 head 0, comme la grille ; aussi max-over-heads)
    # ----------------------------------------------------------------------
    if var_l0_head0 is None:
        print("FATAL: layer 0 non mesure — impossible de produire un verdict.")
        sys.exit(4)

    # La grille = head0. n_kv=1 donc head0 == max-over-heads ; on garde les deux.
    var_decision = var_l0_head0
    vkey, vtext = verdict_for(var_decision)

    l0 = results_layers[0]
    l0h0 = l0["heads"][0]

    # ----------------------------------------------------------------------
    # PRINT — tableau comparatif vs grille
    # ----------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("TABLEAU COMPARATIF — layer 0 / head 0 (KV head unique)")
    print("=" * 78)
    hdr = f"{'modele':<22}{'var':>12}{'amax':>10}{'post-Had':>12}{'verdict'}"
    print(hdr)
    print("-" * 78)
    print(f"{'Qwen2.5-7B (ref)':<22}{1699.0:>12.2f}{172.0:>10.2f}{76.4:>12.2f}  CATASTROPHE")
    print(f"{'Gemma3-4B (ref)':<22}{2.80:>12.2f}{20.0:>10.2f}{7.0:>12.2f}  STABLE")
    print("-" * 78)
    print(f"{'Gemma-4-E2B K (MES.)':<22}{l0h0['var']:>12.2f}"
          f"{l0h0['amax']:>10.2f}{l0h0['post_had_amax']:>12.2f}  {vkey}")
    if l0.get("v_heads"):
        v0 = l0["v_heads"][0]
        print(f"{'Gemma-4-E2B V (MES.)':<22}{v0['var']:>12.2f}"
              f"{v0['amax']:>10.2f}{v0['post_had_amax']:>12.2f}  (V, sans RoPE)")
    print("-" * 78)
    print(f"layer0 head_dim={l0['head_dim']}  n_kv={l0['n_kv_heads']}  "
          f"seq={l0['heads'][0]['n_samples']}  layout_capture={l0['captured_layout']}")
    print(f"std_codebook_ref N(0,1/d) = {l0h0['std_codebook_ref']:.5f} "
          f"(echelle attendue du codebook gaussien ; post-Had={l0h0['post_had_amax']:.2f} "
          f"a comparer a ~3-4x cet ecart-type)")
    print(f"amax_max_ratio pre/post-Had = {l0h0['amax_max_ratio_pre']:.2f} / "
          f"{l0h0['amax_max_ratio_post']:.2f}  "
          f"(post proche de 1-2 = Hadamard a bien etale les pics)")
    print("  top5 canaux outliers (pre-Had)  [ch, ratio amax/med, amax, med]:")
    for c, r, a, md in l0h0["top5_outlier_channels_pre"]:
        print(f"    ch{c:>4}  ratio={r:>9.2f}  amax={a:>9.3f}  med={md:.5f}")

    # Profil var par layer 0..14
    print("\n" + "=" * 78)
    print("PROFIL var(K) par layer producteur (0..14)")
    print("=" * 78)
    print(f"{'layer':>6} {'type':>18} {'d':>5} {'varK(h0)':>11} {'varV(h0)':>11} "
          f"{'amaxK':>8} {'amaxV':>8}")
    for le in results_layers:
        h0 = le["heads"][0]
        vv = le.get("v_var_head0", float("nan"))
        va = le["v_heads"][0]["amax"] if le.get("v_heads") else float("nan")
        print(f"{le['layer']:>6} {le['layer_type']:>18} {le['head_dim']:>5} "
              f"{le['var_head0']:>11.4f} {vv:>11.4f} {h0['amax']:>8.3f} {va:>8.3f}")

    # ----------------------------------------------------------------------
    # VERDICT final
    # ----------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"var_k_l0_head0 = {var_l0_head0:.4f}   (var_max_over_heads = {var_l0_max:.4f})")
    if v_var_l0_head0 is not None:
        print(f"v_var_l0_head0 = {v_var_l0_head0:.4f}   (V sans RoPE, normalise par v_norm)")
    print(f">>> {vtext}")
    print("=" * 78)

    # ----------------------------------------------------------------------
    # DUMP JSON
    # ----------------------------------------------------------------------
    out = {
        "model": REPO,
        "device": device,
        "model_dtype": str(dtype).replace("torch.", ""),
        "k_capture_dtype": "float32",
        "capture_point": (
            "apply_rotary_pos_emb output on K (post-k_norm, post-RoPE, "
            "pre-transpose, pre-repeat_kv, pre-cache-write) — monkeypatch "
            "modeling_gemma4.apply_rotary_pos_emb, K identifie par n_kv=1"
        ),
        "attn_implementation": "eager",
        "calibration": {
            "corpus": "Pride and Prejudice (bench_end2end.CORPUS verbatim)",
            "n_tokens": int(ids.shape[1]),
            "chat_template": bool(used_template),
        },
        "config": {
            "num_hidden_layers": tc.num_hidden_layers,
            "num_kv_shared_layers": tc.num_kv_shared_layers,
            "first_kv_shared_layer_idx": tc.num_hidden_layers - tc.num_kv_shared_layers,
            "num_key_value_heads": n_kv,
            "num_attention_heads": n_q,
            "head_dim_sliding": tc.head_dim,
            "global_head_dim_full": tc.global_head_dim,
            "rms_norm_eps": tc.rms_norm_eps,
            "producer_layers": PRODUCER_LAYERS,
            "full_producer_layers": sorted(FULL_LAYERS & set(PRODUCER_LAYERS)),
        },
        "metrics_definition": {
            "var": "(samples**2).mean() — moment d'ordre 2 NON centre (mean-square)",
            "amax": "samples.abs().max() — scalaire global (tokens x canaux, 1 tete KV)",
            "post_had_amax": "(samples @ H).abs().max(), H=Sylvester/sqrt(d), PAS de norm L2",
        },
        "reference_grid": REFERENCE_GRID,
        "verdict": {
            "key": vkey,
            "text": vtext,
            "var_k_l0_head0": var_l0_head0,
            "var_k_l0_max_over_heads": var_l0_max,
            "v_var_l0_head0": v_var_l0_head0,
            "thresholds": {"qwen_like": ">100", "gemma3_like": "<10",
                           "intermediate": "10..100"},
        },
        "layers": results_layers,
        "apply_rotary_call_log_sample": call_log[:40],
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
