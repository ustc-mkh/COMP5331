from copy import deepcopy
from dataclasses import asdict
import math
from pathlib import Path

import yaml

from .registry import get_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def read_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        result = yaml.safe_load(stream)
    if not isinstance(result, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return result


def merge(base: dict, overlay: dict) -> dict:
    result = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def parse_overrides(values: list[str]) -> dict:
    result = {}
    for expression in values:
        key, separator, raw = expression.partition("=")
        if not separator or any(not part for part in key.split(".")):
            raise ValueError(f"Expected dotted.key=value, got {expression!r}")
        target = result
        parts = key.split(".")
        for part in parts[:-1]:
            if part in target and not isinstance(target[part], dict):
                raise ValueError(f"Conflicting override: {expression}")
            target = target.setdefault(part, {})
        target[parts[-1]] = yaml.safe_load(raw)
    return result


def _check_keys(config: dict, template: dict, prefix=""):
    if not isinstance(config, dict):
        raise ValueError(f"{prefix or 'config'} must be a mapping")
    unknown = config.keys() - template.keys()
    if unknown:
        raise ValueError(f"Unknown configuration fields: {[prefix + str(k) for k in unknown]}")
    for key, value in config.items():
        if key == "model" or prefix + key == "data.features":
            if not isinstance(value, dict):
                raise ValueError(f"{prefix + key} must be a mapping")
            continue
        if isinstance(template[key], dict):
            _check_keys(value, template[key], prefix + key + ".")


def validate(config: dict) -> dict:
    template = read_yaml(PROJECT_ROOT / "configs/default.yaml")
    _check_keys(config, template)
    for group, fields in template.items():
        if group not in config:
            raise ValueError(f"Missing config field: {group}")
        if isinstance(fields, dict) and fields.keys() - config[group].keys():
            raise ValueError(f"Missing fields in {group}: {fields.keys() - config[group].keys()}")
    config = deepcopy(config)
    config["model"] = asdict(get_model(config["model"]["name"]).parse_config(config["model"]))
    for section, keys in {"train": ("epochs", "batch_size", "eval_every", "patience"),
                          "eval": ("user_batch_size", "item_chunk_size"),
                          "runtime": ("num_threads",)}.items():
        for key in keys:
            if type(config[section][key]) is not int or config[section][key] < 1:
                raise ValueError(f"{section}.{key} must be a positive integer")
    if type(config["seed"]) is not int or not 0 <= config["seed"] < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    topk = config["eval"]["topk"]
    if not isinstance(topk, list) or not topk or any(type(k) is not int or k < 1 for k in topk):
        raise ValueError("eval.topk must be a nonempty list of positive integers")
    config["eval"]["topk"] = sorted(set(topk))
    if config["eval"]["monitor"] not in {f"{m}@{k}" for m in ("Recall", "NDCG") for k in topk}:
        raise ValueError("eval.monitor must be one of the computed metrics")
    if config["eval"]["mode"] not in ("min", "max"):
        raise ValueError("eval.mode must be min or max")
    if config["eval"]["history"] not in ("train", "train_valid"):
        raise ValueError("eval.history must be train or train_valid")
    for group, key in (("eval", "require_train_user"), ("runtime", "deterministic")):
        if type(config[group][key]) is not bool:
            raise ValueError(f"{group}.{key} must be boolean")
    if config["optimizer"]["name"] != "adam":
        raise ValueError("Only the Adam optimizer is currently supported")
    for key in ("lr", "weight_decay", "decay_base", "decay_steps"):
        value = config["optimizer"][key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (key != "weight_decay" and value == 0):
            raise ValueError(f"Invalid optimizer.{key}")
    if not isinstance(config["data"]["separator"], str) or len(config["data"]["separator"]) != 1:
        raise ValueError("data.separator must be one character")
    for group, key in (("data", "root"), ("runtime", "output_root"), ("runtime", "cache_root")):
        config[group][key] = str((PROJECT_ROOT / config[group][key]).resolve())
    return config


def load_config(path=None, model=None, dataset=None, overrides=()):
    experiment = read_yaml(PROJECT_ROOT / path) if path else {}
    cli = parse_overrides(list(overrides))
    if model:
        cli = merge(cli, {"model": {"name": model}})
    if dataset:
        cli = merge(cli, {"data": {"name": dataset}})
    selected = merge(experiment, cli)
    model_name = selected.get("model", {}).get("name", "mgcn")
    dataset_name = selected.get("data", {}).get("name", "baby")
    get_model(model_name)
    config = read_yaml(PROJECT_ROOT / "configs/default.yaml")
    for layer in (read_yaml(PROJECT_ROOT / f"configs/datasets/{dataset_name}.yaml"),
                  read_yaml(PROJECT_ROOT / f"configs/models/{model_name}.yaml"), experiment, cli):
        # Check each layer, so a later valid override cannot hide an earlier typo.
        _check_keys(layer, read_yaml(PROJECT_ROOT / "configs/default.yaml"))
        if "model" in layer:
            get_model(model_name).parse_config(merge({"name": model_name}, layer["model"]))
        config = merge(config, layer)
    return validate(config)
