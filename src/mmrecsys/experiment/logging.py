import importlib.metadata
import json
import platform
from pathlib import Path
import subprocess

import torch


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def append_metrics(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def manifest(root, config, data_metadata, device):
    def git(*args):
        try:
            return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.DEVNULL,
                                           text=True).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
    return {"data": data_metadata, "git_commit": git("rev-parse", "HEAD"),
            "git_status": git("status", "--porcelain"), "python": platform.python_version(),
            "dependencies": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "scipy", "PyYAML")},
            "device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
            "evaluation": {**config["eval"], "ranking": "all_items", "tie_break": "item_id_ascending",
                           "padding": "excluded", "aggregation": "macro_user_mean"}}
