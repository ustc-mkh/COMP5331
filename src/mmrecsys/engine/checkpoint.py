from copy import deepcopy

import torch

from ..experiment.artifacts import atomic_torch_save
from ..experiment.seed import random_state, restore_random_state


def compatible_config(config):
    result = deepcopy(config)
    # Only epoch budget and artifact locations may change for a resumed trajectory.
    result["train"].pop("epochs")
    result["train"].setdefault("preload_to_device", False)
    result["runtime"].pop("output_root")
    result["runtime"].pop("cache_root")
    return result


def save_checkpoint(path, model, optimizer, scheduler, sampler, state, config, fingerprint):
    atomic_torch_save({"format_version": 1, "model": model.state_dict(),
                       "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                       "sampler": sampler.state_dict(), "state": state.copy(),
                       "random": random_state(next(model.parameters()).device), "config": config, "fingerprint": fingerprint}, path)


def load_checkpoint(path, model, config, fingerprint, *, optimizer=None, scheduler=None, sampler=None):
    # These are locally generated experiment files containing Python/NumPy RNG state.
    record = torch.load(path, map_location="cpu", weights_only=False)
    if record.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    if record["fingerprint"] != fingerprint:
        raise ValueError("Checkpoint data/feature fingerprint mismatch")
    if compatible_config(record["config"]) != compatible_config(config):
        raise ValueError("Checkpoint configuration is incompatible with this experiment")
    model.load_state_dict(record["model"])
    if optimizer is not None:
        optimizer.load_state_dict(record["optimizer"])
        scheduler.load_state_dict(record["scheduler"])
        sampler.load_state_dict(record["sampler"])
        restore_random_state(record["random"], next(model.parameters()).device)
    return record["state"]
