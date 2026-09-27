from datetime import datetime, timezone
from pathlib import Path
import uuid

import torch
import yaml

from ..config import PROJECT_ROOT, read_yaml, validate
from ..data.dataset import load_dataset
from ..data.sampling import TrainingSampler
from ..engine.checkpoint import load_checkpoint
from ..engine.evaluator import Evaluator
from ..engine.trainer import Trainer
from ..registry import get_model
from .artifacts import GraphCache
from .logging import manifest, write_json
from .seed import seed_everything


def resolve_device(value):
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; set runtime.device=cpu")
    return device


def assemble(config):
    device = resolve_device(config["runtime"]["device"])
    if device.type == "cuda":
        # Explicit tensor placement does not change CUDA's current device.
        torch.cuda.set_device(device)
    seed_everything(config["seed"], config["runtime"]["deterministic"], device)
    torch.set_num_threads(config["runtime"]["num_threads"])
    entry = get_model(config["model"]["name"])
    print(f"Loading {config['data']['name']} on {device}", flush=True)
    train, views, metadata = load_dataset(config["data"], entry.modalities, config["eval"])
    print(f"users={train.n_users} items={train.n_items} splits={metadata['interactions']}; building/loading graphs", flush=True)
    model = entry.factory(entry.parse_config(config["model"]), train,
                          GraphCache(config["runtime"]["cache_root"])).to(device)
    evaluator = Evaluator(config["eval"], train.n_items, device)
    return train, views, metadata, model, evaluator, device


def train_experiment(config, resume=None):
    if resume is None:
        identifier = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
        run_dir = Path(config["runtime"]["output_root"]) / f"{config['model']['name']}-{config['data']['name']}-{identifier}"
        run_dir.mkdir(parents=True, exist_ok=False)
    else:
        resume = Path(resume).resolve()
        if resume.name != "last.pt" or not (resume.parent / "best.pt").exists():
            raise ValueError("Resume from last.pt in its original run directory with best.pt present")
        run_dir = resume.parent
    print(f"run_dir={run_dir}", flush=True)
    train, views, metadata, model, evaluator, device = assemble(config)
    entry = get_model(config["model"]["name"])
    sampler = TrainingSampler(train, entry.batch_spec, config["train"]["batch_size"], config["seed"])
    options = config["optimizer"]
    optimizer = torch.optim.Adam(model.parameters(), lr=options["lr"], weight_decay=options["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: options["decay_base"] ** (epoch / options["decay_steps"]))
    if resume is not None:
        # Check compatibility before overwriting any prior artifacts.
        state = load_checkpoint(resume, model, config, metadata["fingerprint"])
        if config["train"]["epochs"] < state["epoch"]:
            raise ValueError("train.epochs cannot be smaller than the resumed epoch")
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    if resume is None:
        write_json(run_dir / "manifest.json", manifest(PROJECT_ROOT, config, metadata, device))
    trainer = Trainer(model, optimizer, scheduler, sampler, evaluator, config["train"], run_dir,
                      device, config, metadata["fingerprint"])
    result = trainer.fit(views["valid"], views["test"], resume)
    return run_dir, result


def evaluate_experiment(run, split="test", device=None):
    run_dir = (PROJECT_ROOT / run).resolve()
    config = validate(read_yaml(run_dir / "config.yaml"))
    checkpoint_config = config
    if device:
        from copy import deepcopy
        config = deepcopy(config)
        config["runtime"]["device"] = device
    _, views, metadata, model, evaluator, _ = assemble(config)
    load_checkpoint(run_dir / "best.pt", model, checkpoint_config, metadata["fingerprint"])
    metrics = evaluator.evaluate(model, views[split])
    write_json(run_dir / f"evaluation-{split}.json", {"split": split, "metrics": metrics,
               "evaluation_protocol": config["eval"], "tie_break": "item_id_ascending"})
    return metrics
