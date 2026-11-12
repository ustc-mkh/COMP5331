"""Validation-selected training, restart checkpoints, and isolated test evaluation."""
import csv
import importlib.metadata
import json
import os
import platform
import random
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import ROOT, UPSTREAM_COMMIT
from .config import configure_process, write_json
from .data import inspect_data, runtime_config
from .model import BM3, make_loaders, evaluate, early_stopping


def environment_info(device):
    packages = {}
    for name in ("torch", "numpy", "scipy", "pandas", "PyYAML"):
        packages[name] = importlib.metadata.version(name)
    info = {"python": platform.python_version(), "platform": platform.platform(),
            "packages": packages, "device": str(device), "torch_cuda_build": torch.version.cuda}
    if str(device).startswith("cuda"):
        info["gpu"] = torch.cuda.get_device_name(torch.device(device))
        info["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES", "not set")
    return info


def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def save_checkpoint(path, state):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    def payload_bytes(value):
        if torch.is_tensor(value):
            return value.numel() * value.element_size()
        if isinstance(value, np.ndarray):
            return value.nbytes
        if isinstance(value, dict):
            return sum(payload_bytes(item) for item in value.values())
        if isinstance(value, (tuple, list)):
            return sum(payload_bytes(item) for item in value)
        return 0

    required = payload_bytes(state) + 16 * 1024 * 1024
    free = shutil.disk_usage(path.parent).free
    if free < required:
        raise RuntimeError("Insufficient disk space for {}: need about {:.0f} MiB free, have {:.0f} MiB".format(
            path.name, required / 1024 ** 2, free / 1024 ** 2))
    try:
        torch.save(state, temporary)
        os.replace(str(temporary), str(path))
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def load_checkpoint(path, device="cpu"):
    return torch.load(str(path), map_location=device, weights_only=False)


def save_history(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rng_state(device):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if str(device).startswith("cuda") else None}


def restore_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if str(device).startswith("cuda") and state["cuda"] is not None:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def _check_compatible(config, stored_config):
    # Extending the epoch budget is allowed. All optimization/evaluation settings stay fixed.
    for key in config:
        if key not in {"epochs", "checkpoint_every"} and config[key] != stored_config[key]:
            raise ValueError("Resume configuration mismatch: " + key)


def train(config, data_dir, run_dir, device="cpu", smoke=False, test_at_end=True,
          resume=None, require_baby=True):
    run_dir = Path(run_dir).resolve()
    data_dir = Path(data_dir).resolve()
    device = torch.device(device)
    report, splits = inspect_data(data_dir, require_baby=require_baby, dataset=config["dataset"])
    if splits[0].item_num < max(config["topk"]):
        raise ValueError("Catalog is smaller than evaluation K")
    kind = "smoke" if smoke or not require_baby else "formal"
    resumed = None
    if resume:
        resume = Path(resume).resolve()
        if resume.parent != run_dir or not (run_dir / "best.pt").is_file():
            raise ValueError("Resume in the original run directory; keep both best.pt and last.pt")
        resumed = load_checkpoint(resume, device)
        _check_compatible(config, resumed["config"])
        if resumed["data_fingerprint"] != report["fingerprint"] or resumed["kind"] != kind:
            raise ValueError("Resume dataset/kind mismatch")
        if resumed["source_commit"] != UPSTREAM_COMMIT:
            raise ValueError("Resume source version mismatch")
        if resumed["epoch"] >= config["epochs"]:
            raise ValueError("The checkpoint has already reached the requested epoch budget")
        if resumed["bad_epochs"] > config["stopping_step"]:
            raise ValueError("This run already met its early-stopping criterion")
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
    configure_process(config["seed"], device, config["threads"])
    runtime = runtime_config(config, data_dir, device, run_dir)
    train_loader, valid_loader, test_loader = make_loaders(runtime, splits, 32 if smoke else None)
    train_loader.pretrain_setup()
    model = BM3(runtime, train_loader).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    best_score, best_epoch, bad_epochs = -1.0, 0, 0
    best_valid, history, start_epoch = None, [], 1
    if resumed:
        model.load_state_dict(resumed["model_state_dict"])
        optimizer.load_state_dict(resumed["optimizer_state_dict"])
        best_score, best_epoch, bad_epochs = resumed["best_score"], resumed["best_epoch"], resumed["bad_epochs"]
        best_valid, history = resumed["best_valid"], resumed["history"]
        start_epoch = resumed["epoch"] + 1
        train_loader.dataset = splits[0].copy(pd.DataFrame(resumed["train_order"], columns=["userID", "itemID"]))
        restore_rng(resumed["rng"], device)
    metadata = {"kind": kind, "config": config, "environment": environment_info(device),
                "data": report, "source_commit": UPSTREAM_COMMIT,
                "source_repository": "https://github.com/enoche/BM3",
                "protocol": "original BM3; validation Recall@20; test only after selection",
                "smoke_limits": {"train_batches_per_epoch": 2, "eval_users": 32} if smoke else None}
    write_json(run_dir / "metadata.json", metadata)
    print("RUN", run_dir, "KIND", kind, "DEVICE", device, flush=True)
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    start_time = time.perf_counter()
    for epoch in range(start_epoch, config["epochs"] + 1):
        model.train()
        train_loader.pr = train_loader.inter_pr = 0
        sync(device)
        epoch_start = time.perf_counter()
        total_loss, batch_count = 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad()
            loss = model.calculate_loss(batch)
            if not torch.isfinite(loss).all():
                raise FloatingPointError("Non-finite loss at epoch {}".format(epoch))
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            batch_count += 1
            if smoke and batch_count == 2:
                break
        train_loader.pr = train_loader.inter_pr = 0
        sync(device)
        train_seconds = time.perf_counter() - epoch_start
        eval_start = time.perf_counter()
        valid = evaluate(model, valid_loader, runtime)
        sync(device)
        validation_seconds = time.perf_counter() - eval_start
        score = valid[config["valid_metric"]]
        best_score, bad_epochs, should_stop, improved = early_stopping(
            score, best_score, bad_epochs, max_step=config["stopping_step"], bigger=True)
        row = {"epoch": epoch, "train_loss_sum": total_loss, "train_loss_mean": total_loss / batch_count,
               "train_seconds": train_seconds, "validation_seconds": validation_seconds}
        row.update({"valid_" + key: value for key, value in valid.items()})
        history.append(row)
        base_state = {"kind": kind, "config": dict(config), "epoch": epoch,
                      "model_state_dict": model.state_dict(), "data_fingerprint": report["fingerprint"],
                      "source_commit": UPSTREAM_COMMIT, "validation": valid}
        if improved:
            best_epoch, best_valid = epoch, valid
            save_checkpoint(run_dir / "best.pt", base_state)
        if epoch == start_epoch or epoch % config["checkpoint_every"] == 0 or should_stop or epoch == config["epochs"]:
            state = dict(base_state, optimizer_state_dict=optimizer.state_dict(), best_score=best_score,
                         best_epoch=best_epoch, bad_epochs=bad_epochs, best_valid=best_valid,
                         history=history, rng=rng_state(device),
                         train_order=train_loader.dataset.df[["userID", "itemID"]].to_numpy())
            save_checkpoint(run_dir / "last.pt", state)
        save_history(run_dir / "history.csv", history)
        print("epoch={} loss={:.6f} valid_R20={:.4f} best_epoch={} train_s={:.2f}".format(
            epoch, total_loss, score, best_epoch, train_seconds), flush=True)
        if should_stop:
            break
    best = load_checkpoint(run_dir / "best.pt", device)
    model.load_state_dict(best["model_state_dict"])
    # No test scores enter early stopping or hyperparameter selection.
    test_result = evaluate(model, test_loader, runtime, run_dir / "test_top20.npz") if test_at_end else None
    result = {"kind": kind, "seed": config["seed"], "config": config, "best_epoch": best_epoch,
              "epochs_completed": history[-1]["epoch"], "best_validation": best_valid,
              "test": test_result, "test_evaluated": test_at_end,
              "data_fingerprint": report["fingerprint"], "source_commit": UPSTREAM_COMMIT,
              "session_wall_seconds": time.perf_counter() - start_time,
              "mean_train_seconds_per_epoch": float(np.mean([row["train_seconds"] for row in history])),
              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if str(device).startswith("cuda") else None}
    write_json(run_dir / "result.json", result)
    print(json.dumps({"kind": kind, "best_epoch": best_epoch, "validation": best_valid, "test": test_result}), flush=True)
    return result


def evaluate_checkpoint(checkpoint, data_dir, device="cpu", require_baby=True, output=None):
    checkpoint = Path(checkpoint).resolve()
    stored = load_checkpoint(checkpoint, "cpu")
    config = stored["config"]
    report, splits = inspect_data(data_dir, require_baby=require_baby, dataset=config["dataset"])
    if stored["data_fingerprint"] != report["fingerprint"]:
        raise ValueError("Checkpoint and evaluation data differ")
    device = torch.device(device)
    configure_process(config["seed"], device, config["threads"])
    runtime = runtime_config(config, data_dir, device, checkpoint.parent)
    train_loader, valid_loader, test_loader = make_loaders(runtime, splits, 32 if stored["kind"] == "smoke" else None)
    train_loader.pretrain_setup()
    model = BM3(runtime, train_loader).to(device)
    model.load_state_dict(stored["model_state_dict"])
    result = {"kind": stored["kind"], "epoch": stored["epoch"], "checkpoint": str(checkpoint),
              "validation": evaluate(model, valid_loader, runtime), "test": evaluate(model, test_loader, runtime),
              "data_fingerprint": report["fingerprint"]}
    if output:
        write_json(output, result)
    return result
