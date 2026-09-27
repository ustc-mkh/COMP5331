import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True, device="cpu"):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed also seeds every CUDA device; this run uses only one.
    torch.random.default_generator.manual_seed(seed)
    device = torch.device(device)
    if device.type == "cuda":
        with torch.cuda.device(device):
            torch.cuda.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic


def random_state(device="cpu"):
    device = torch.device(device)
    cuda = None
    if device.type == "cuda":
        cuda = {"device": str(device), "state": torch.cuda.get_rng_state(device)}
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": cuda}


def restore_random_state(state, device="cpu"):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    device = torch.device(device)
    cuda = state["cuda"]
    if cuda is not None and device.type == "cuda":
        if isinstance(cuda, list):
            # Legacy checkpoints stored every visible GPU; restore only this run's GPU.
            index = device.index if device.index is not None else torch.cuda.current_device()
            value = cuda[index]
        else:
            value = cuda["state"]
        torch.cuda.set_rng_state(value.cpu(), device)
