import os, sys
os.environ.setdefault("HF_HOME", "/data/hf_cache")
import torch
from safetensors.torch import save_file
sys.path.insert(0, "/data/gemma4-zml-probe")
from turboquant import TurboQuantMSE

B_BITS = 4  # K = 16 niveaux (V-only 4 bits)
OUT = "/data/gemma4-zml-probe/turboquant_constants.safetensors"
state = {}
for d in (256, 512):
    mse = TurboQuantMSE(d, B_BITS, device="cpu", rotation="hadamard")
    state[f"codebook_{d}"] = mse.codebook.to(torch.float32).contiguous()   # [K]
    state[f"hadamard_{d}"] = mse.Pi.to(torch.float32).contiguous()         # [d,d], Pi[e,d]
save_file(state, OUT)
print("Wrote", OUT, {k: tuple(v.shape) for k, v in state.items()})
