import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from . import ROOT


def load_config(path=None, overrides=None):
    path = Path(path) if path else ROOT / "configs" / "baby.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a mapping")
    default_keys = set(yaml.safe_load((ROOT / "configs" / "baby.yaml").read_text(encoding="utf-8")))
    if set(config) != default_keys:
        raise ValueError("Unexpected or missing configuration keys: " + str(set(config) ^ default_keys))
    config.update({k: v for k, v in (overrides or {}).items() if v is not None})
    if config["model"] != "BM3" or config["dataset"] not in {"baby", "sports", "clothing", "elec", "microlens"}:
        raise ValueError("This package supports original BM3 on the five released datasets")
    if config["inter_file_name"] != config["dataset"] + ".inter":
        raise ValueError("Interaction filename must match the configured dataset")
    if config["use_neg_sampling"] or config["use_full_sampling"] or config["end2end"]:
        raise ValueError("Original BM3 uses positive pairs and pre-extracted features")
    if config["valid_metric"] != "recall@20" or sorted(config["topk"]) != [10, 20]:
        raise ValueError("The reproduction protocol fixes Recall@20 selection and K=10,20")
    for key in ("n_layers", "embedding_size", "epochs", "train_batch_size", "eval_batch_size", "threads", "checkpoint_every"):
        if not isinstance(config[key], int) or isinstance(config[key], bool) or config[key] < 1:
            raise ValueError(key + " must be a positive integer")
    if not 0 <= config["dropout"] < 1 or config["reg_weight"] < 0 or config["learning_rate"] <= 0:
        raise ValueError("Invalid dropout, regularization, or learning rate")
    return config


def configure_process(seed, device, threads):
    # CPU work does not probe or initialize CUDA. GPU execution is explicit.
    torch.set_num_threads(threads)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if str(device).startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable; no CPU fallback")
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


def fingerprint(files):
    raw = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()
