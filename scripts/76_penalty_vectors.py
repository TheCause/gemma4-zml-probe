#!/usr/bin/env python3
"""RP1 — vecteurs de référence de la repetition penalty, produits par le VRAI processor HF.
Interdiction de retranscrire le torch.where (spec C5) : une faute commune aux deux
implémentations passerait le gate.

Numéroté 76 et non 71 : `scripts/71_gc1_fixture.py` occupe déjà le 71 (la collision a
déjà coûté une fois, script 74). 76 = premier numéro libre vérifié au 10 août 2026.
"""
import json, torch
from safetensors.torch import save_file
from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor
import transformers

VOCAB = 512  # petit, suffisant : la formule est indépendante de la taille
PENALTIES = [0.8, 1.0, 1.15, 1.5]

def main():
    g = torch.Generator().manual_seed(20260727)
    # logits des DEUX signes, amplitude comparable au post-softcap réel (±30)
    logits = (torch.rand(VOCAB, generator=g, dtype=torch.float32) * 60.0) - 30.0
    # historique AVEC doublons (exerce la déduplication) et couvrant les deux signes
    hist = torch.tensor([7, 7, 7, 42, 100, 100, 3, 511, 0], dtype=torch.int64)

    # Vecteur à TIES f32 EXACTS — sans lui, le critère tie-break de RP1 est inexécutable :
    # `torch.rand * 60 - 30` ne produit jamais deux f32 rigoureusement égaux.
    ties = torch.full((16,), -1.0, dtype=torch.float32)
    ties[3] = 5.5; ties[9] = 5.5; ties[14] = 5.5   # max atteint 3 fois → attendu = 3 (le PREMIER)

    out = {"logits_in": logits, "hist": hist.to(torch.int32), "logits_ties": ties}
    meta = {"transformers_version": transformers.__version__, "penalties": PENALTIES,
            "ties_expected_argmax": 3}
    for p in PENALTIES:
        proc = RepetitionPenaltyLogitsProcessor(penalty=float(p))
        got = proc(hist.unsqueeze(0), logits.clone().unsqueeze(0)).squeeze(0)
        out[f"logits_out_{p}"] = got.contiguous()
        # métadonnées de non-vacuité : le test Zig les vérifiera aussi
        touched = (got != logits).sum().item()
        meta[f"touched_{p}"] = touched
    save_file(out, "fixtures/penalty_vectors.safetensors")
    print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
