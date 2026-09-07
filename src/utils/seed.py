import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Seeds Python's random module, NumPy, and PyTorch (CPU + all CUDA devices).

    Controls: weight initialisation (Xavier init in CompGCN/Generator/Critic),
    dropout masks, torch.randperm batch shuffling, torch.randn generator noise,
    and NegativeSampler's torch.rand/torch.randint/torch.multinomial calls.

    Does NOT guarantee bit-for-bit determinism on GPU: CompGCNLayer's
    scatter_add_ (compgcn.py) uses a non-deterministic CUDA reduction kernel
    regardless of seed. Call torch.use_deterministic_algorithms(True) if exact
    determinism is required -- expect a slowdown, and some ops may not have a
    deterministic CUDA implementation at all.

    Call this immediately before constructing/training each model you want to
    compare -- seeding once at the top of a long notebook does not put two
    later models at the same RNG starting point if anything random happened
    in between them.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
