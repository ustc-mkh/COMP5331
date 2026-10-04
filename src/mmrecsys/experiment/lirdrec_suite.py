"""Resume an ordered LIRDRec suite on one explicitly selected GPU UUID."""
import argparse
from collections import deque
from contextlib import contextmanager
from copy import deepcopy
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
from statistics import mean, stdev
import subprocess
import sys
import tarfile
import time

import yaml

from ..config import PROJECT_ROOT, load_config, read_yaml, validate
from .ablation import VARIANTS, variant_config


FORMAT_VERSION = 1
GPU_PATTERN = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\Z")


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def source_identity(dataset):
    """Freeze training dependencies, excluding scheduling, docs, and tests."""
    package = PROJECT_ROOT / "src/mmrecsys"
    paths = [package / name for name in ("__init__.py", "config.py", "registry.py",
             "models/__init__.py", "models/base.py", "models/lirdrec.py",
             "experiment/__init__.py", "experiment/artifacts.py", "experiment/logging.py",
             "experiment/runner.py", "experiment/seed.py")]
    for name in ("data", "engine", "nn"):
        paths.extend((package / name).rglob("*.py"))
    paths.extend(PROJECT_ROOT / name for name in ("configs/default.yaml", "configs/models/lirdrec.yaml",
                                                 f"configs/datasets/{dataset}.yaml"))
    return {str(path.relative_to(PROJECT_ROOT)): file_digest(path) for path in sorted(set(paths))}


def verify_source_archive(archive, expected):
    """Read hashes without extracting an existing source snapshot."""
    with tarfile.open(archive, "r:*") as stream:
        members = {member.name.removeprefix("./"): member for member in stream.getmembers() if member.isfile()}
        different = []
        for name, wanted in expected.items():
            member = members.get(name)
            if member is None:
                different.append(name)
                continue
            with stream.extractfile(member) as source:
                actual = hashlib.sha256(source.read()).hexdigest()
            if actual != wanted:
                different.append(name)
    if different:
        raise ValueError(f"Reuse source snapshot differs: {', '.join(different)}")


def comparable_config(config):
    """Exact effective settings except artifact locations; epoch budget is included."""
    result = deepcopy(validate(config))
    for name in ("output_root", "cache_root"):
        result["runtime"].pop(name)
    return result


def data_files(config):
    data = config["data"]
    root = Path(data["root"]) / data["name"]
    return [root / name for name in (data["interactions"], *data["features"].values())]


def data_identity(config):
    files = data_files(config)
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing dataset files: " + ", ".join(missing))
    return {str(path): {"sha256": file_digest(path), "bytes": path.stat().st_size} for path in files}


def build_slots(datasets, seeds, output, overrides=()):
    if not datasets or len(datasets) != len(set(datasets)):
        raise ValueError("Datasets must be nonempty and unique")
    if not seeds or seeds[0] != 999 or len(seeds) != len(set(seeds)):
        raise ValueError("Seeds must be unique and start with 999")
    reserved = ("seed", "model.name", "data.name", "runtime.device", "runtime.num_threads", "runtime.output_root")
    if any(expression.partition("=")[0] in reserved for expression in overrides):
        raise ValueError("Use suite options for datasets/seeds; GPU, CPU threads, and output are fixed by the suite")
    configs = {dataset: load_config(model="lirdrec", dataset=dataset,
                                   overrides=[*overrides, "runtime.device=cuda:0", "runtime.num_threads=2"])
               for dataset in datasets}
    ordered = ([("baby", 999, 1)] if "baby" in datasets else [])
    ordered += [(dataset, 999, 2) for dataset in datasets if dataset != "baby"]
    ordered += [(dataset, seed, 3) for seed in seeds[1:] for dataset in datasets]
    slots = []
    for dataset, seed, stage in ordered:
        for variant in VARIANTS:
            identifier = f"{dataset}-{seed}-{variant}"
            config = variant_config(configs[dataset], variant, seed)
            config["runtime"]["output_root"] = str(Path(output) / "slots" / identifier / "runs")
            slots.append({"id": identifier, "stage": stage, "dataset": dataset, "seed": seed,
                          "variant": variant, "config": config, "config_digest": digest(comparable_config(config)),
                          "source": source_identity(dataset), "status": "pending"})
    for slot in slots:
        slot["source_digest"] = digest(slot["source"])
    return slots


def gpu_processes(gpu_uuid):
    if not GPU_PATTERN.fullmatch(gpu_uuid):
        raise ValueError("Provide one full physical GPU UUID (GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx)")
    query = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True)
    if gpu_uuid not in {line.strip() for line in query.splitlines()}:
        raise ValueError(f"Requested GPU UUID is unavailable: {gpu_uuid}")
    query = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name",
                                     "--format=csv,noheader,nounits"], text=True)
    return [line.strip() for line in query.splitlines() if line.split(",", 1)[0].strip() == gpu_uuid]


def worker_environment(gpu_uuid):
    if not GPU_PATTERN.fullmatch(gpu_uuid):
        raise ValueError("A full physical GPU UUID is required")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_uuid
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "2"
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return env


@contextmanager
def exclusive_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another suite or its surviving worker holds {path}") from error
        # Do not explicitly unlock: a worker inherits this fd and must keep the lock
        # when its supervisor is killed before it can stop that worker.
        yield stream.fileno()


@contextmanager
def suite_resources(output, plan, gpu_uuid=None, wait_for_gpu=None, max_parallel=None):
    """A surviving worker retains the suite lock, including across GPU migration."""
    output = Path(output)
    with exclusive_lock(output / ".lock") as suite_fd:
        saved_path = output / "suite.json"
        if saved_path.is_file():
            # A previous supervisor may have advanced after this caller read its
            # plan. Never replace newly completed slots with that stale copy.
            saved = json.loads(saved_path.read_text())
            if saved.get("format_version") != FORMAT_VERSION:
                raise ValueError("Unsupported suite format")
            plan.clear()
            plan.update(saved)
        old_gpu = plan["gpu_uuid"]
        selected = gpu_uuid or old_gpu
        if not GPU_PATTERN.fullmatch(selected):
            raise ValueError("A full physical GPU UUID is required")
        gpu_lock = PROJECT_ROOT / "cache" / f"lirdrec-suite-{selected}.lock"
        with exclusive_lock(gpu_lock) as gpu_fd:
            if selected != old_gpu:
                plan.setdefault("gpu_migrations", []).append({"from": old_gpu, "to": selected,
                                                               "at": now(), "reason": "explicit_resume_override"})
                plan["gpu_uuid"] = selected
            if wait_for_gpu is not None:
                plan["wait_for_gpu"] = wait_for_gpu
            plan.setdefault("wait_for_gpu", False)
            concurrency = plan.get("max_parallel", 1) if max_parallel is None else max_parallel
            if type(concurrency) is not int or not 1 <= concurrency <= 10:
                raise ValueError("max_parallel must be an integer between 1 and 10")
            plan["max_parallel"] = concurrency
            save_progress(output, plan)
            yield suite_fd, gpu_fd, selected


def wait_until_gpu_available(output, plan, slot, gpu_uuid):
    previous = None
    while True:
        active = gpu_processes(gpu_uuid)
        if not active:
            return True
        slot.update(status="waiting_gpu", error="GPU has active processes: " + "; ".join(active))
        save_progress(output, plan)
        if active != previous:
            print(f"WAITING_GPU {slot['id']}: {slot['error']}", flush=True)
        if not plan.get("wait_for_gpu", False):
            return False
        previous = active
        time.sleep(30)


def read_result(run, config=None):
    result = json.loads((Path(run) / "result.json").read_text())
    config = config or read_yaml(Path(run) / "config.yaml")
    metrics = result.get("test", {})
    required = {f"{metric}@{cutoff}" for metric in ("Recall", "NDCG") for cutoff in config["eval"]["topk"]}
    if not metrics or not all(isinstance(value, (float, int)) and math.isfinite(value) for value in metrics.values()):
        raise ValueError("Completed result has missing or nonfinite test metrics")
    if not required.issubset(metrics) or any(not 0 <= metrics[name] <= 1 for name in required):
        raise ValueError("Completed result must contain all configured Recall and NDCG metrics in [0, 1]")
    if type(metrics.get("n_users")) is not int or metrics["n_users"] < 1:
        raise ValueError("Completed result has no evaluated users")
    if not isinstance(result.get("best_epoch"), int) or result["best_epoch"] < 1:
        raise ValueError("Completed result has no best validation epoch")
    if result.get("monitor") != config["eval"]["monitor"] or not math.isfinite(result.get("best_validation_metric", float("nan"))):
        raise ValueError("Completed result has inconsistent validation selection metadata")
    return result


def verify_run(slot, run, *, check_data=False):
    """Check effective config and canonical data identity, never select on test."""
    run = Path(run).resolve()
    for name in ("config.yaml", "manifest.json", "last.pt", "best.pt"):
        if not (run / name).is_file():
            raise ValueError(f"Run lacks {name}: {run}")
    if comparable_config(read_yaml(run / "config.yaml")) != comparable_config(slot["config"]):
        raise ValueError(f"Run configuration differs from slot {slot['id']}")
    manifest = json.loads((run / "manifest.json").read_text())
    import torch
    for name in ("last.pt", "best.pt"):
        checkpoint = torch.load(run / name, map_location="cpu", weights_only=False)
        if checkpoint.get("format_version") != 1:
            raise ValueError(f"Unsupported checkpoint format in {run / name}")
        if comparable_config(checkpoint["config"]) != comparable_config(slot["config"]):
            raise ValueError(f"Checkpoint configuration differs from slot {slot['id']}")
        if checkpoint["fingerprint"] != manifest["data"]["fingerprint"]:
            raise ValueError("Checkpoint data identity differs from its run manifest")
    if check_data:
        from ..data.dataset import load_dataset
        _, _, metadata = load_dataset(slot["config"]["data"], ("image", "text"), slot["config"]["eval"])
        if metadata["fingerprint"] != manifest["data"]["fingerprint"]:
            raise ValueError("Run data/feature fingerprint differs from current dataset")
    if slot.get("training_data_fingerprint") not in (None, manifest["data"]["fingerprint"]):
        raise ValueError("Run data fingerprint differs from recorded slot")
    return manifest["data"]["fingerprint"]


def discover_resume(slot):
    root = Path(slot["config"]["runtime"]["output_root"])
    runs = sorted(root.glob("*/config.yaml"), key=lambda path: path.parent.name, reverse=True)
    for config_path in runs:
        run = config_path.parent
        if (run / "last.pt").is_file() and (run / "best.pt").is_file():
            verify_run(slot, run)
            return run
    return None


def run_or_resume(slot, train):
    previous = discover_resume(slot)
    if previous is not None and (previous / "result.json").is_file():
        try:
            return previous, read_result(previous)
        except (OSError, ValueError, KeyError, TypeError):
            # An interruption can truncate the shared runner's non-atomic result
            # write. Epoch checkpoints are atomic, so replay final evaluation.
            print(f"Recovering incomplete result file in {previous}", flush=True)
    return train(slot["config"], previous / "last.pt" if previous else None, evaluate_test=True)


def summarize(slots):
    rows = []
    for dataset in dict.fromkeys(slot["dataset"] for slot in slots):
        done = [slot for slot in slots if slot["dataset"] == dataset and slot["status"] == "completed"]
        baseline = {slot["seed"]: slot for slot in done if slot["variant"] == "baseline"}
        for variant in VARIANTS:
            runs = [slot for slot in done if slot["variant"] == variant]
            if not runs:
                continue
            paired = [slot for slot in runs if slot["seed"] in baseline] if variant != "baseline" else []
            for metric in runs[0]["result"]["test"]:
                if metric == "n_users":
                    continue
                values = [slot["result"]["test"][metric] for slot in runs]
                deltas = [slot["result"]["test"][metric] - baseline[slot["seed"]]["result"]["test"][metric]
                          for slot in paired]
                reference = [baseline[slot["seed"]]["result"]["test"][metric] for slot in paired]
                rows.append({"dataset": dataset, "variant": variant, "metric": metric,
                             "n_runs": len(runs), "seeds": [slot["seed"] for slot in runs],
                             "mean": mean(values), "sample_std": stdev(values) if len(values) > 1 else None,
                             "n_pairs": len(paired), "paired_seeds": [slot["seed"] for slot in paired],
                             "delta_mean": mean(deltas) if deltas else None,
                             "delta_sample_std": stdev(deltas) if len(deltas) > 1 else None,
                             "relative_gain_pct": 100 * mean(deltas) / mean(reference)
                             if reference and mean(reference) != 0 else None})
    return rows


def save_progress(output, plan):
    plan["updated_at"] = now()
    plan["complete"] = all(slot["status"] == "completed" for slot in plan["slots"])
    atomic_json(Path(output) / "suite.json", plan)
    rows = [{key: slot.get(key) for key in ("id", "stage", "status", "run_dir", "result", "error")}
            for slot in plan["slots"]]
    atomic_json(Path(output) / "results.json", {"complete": plan["complete"],
                "selection": "Each run uses its best validation checkpoint; test is evaluated only after training",
                "slots": rows})
    metrics = summarize(plan["slots"])
    atomic_json(Path(output) / "summary.json", {"complete": plan["complete"],
                "selection": "Validation-selected checkpoints; gains include only complete baseline/variant seed pairs",
                "statuses": {status: sum(slot["status"] == status for slot in plan["slots"])
                             for status in sorted({slot["status"] for slot in plan["slots"]})}, "metrics": metrics})
    temporary = Path(output) / "summary.csv.tmp"
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]) if metrics else ["dataset", "variant", "metric", "n_runs"])
        writer.writeheader()
        writer.writerows(metrics)
    temporary.replace(Path(output) / "summary.csv")


def worker(slot_path):
    """One fresh process owns one training task and exits to release its GPU state."""
    import torch
    from .runner import train_experiment

    slot_path = Path(slot_path)
    record = json.loads(slot_path.read_text())
    slot, gpu_uuid = record["slot"], record["gpu_uuid"]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != gpu_uuid or torch.cuda.device_count() != 1:
        raise RuntimeError("Worker must see exactly the requested GPU")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    if source_identity(slot["dataset"]) != slot["source"]:
        raise ValueError("Training source changed after scheduling")
    if data_identity(slot["config"]) != slot["data"]:
        raise ValueError("Dataset changed after scheduling")
    run, result = run_or_resume(slot, train_experiment)
    fingerprint = verify_run(slot, run)
    atomic_json(Path(run) / "suite-provenance.json", {"slot_id": slot["id"],
                "source": slot["source"], "source_digest": slot["source_digest"],
                "config_digest": slot["config_digest"], "data": slot["data"],
                "training_data_fingerprint": fingerprint, "gpu_uuid": gpu_uuid})
    atomic_json(slot_path.parent / "completed.json", {"run_dir": str(run), "result": result,
                "training_data_fingerprint": fingerprint})


def prepare_slot(output, plan, slot, reuse, reuse_source):
    task = output / "slots" / slot["id"]
    task.mkdir(parents=True, exist_ok=True)
    try:
        current_data = data_identity(slot["config"])
    except FileNotFoundError as error:
        slot.update(status="waiting_data", error=str(error))
        save_progress(output, plan)
        print(f"WAITING_DATA {slot['id']}: {error}; resume after providing data", flush=True)
        return None
    if slot.get("data") not in (None, current_data):
        raise ValueError(f"Dataset changed for {slot['id']}; use a new suite")
    slot["data"] = current_data
    if slot["status"] == "completed":
        verify_run(slot, slot["run_dir"])
        if read_result(slot["run_dir"]) != slot["result"]:
            raise ValueError(f"Completed results changed for {slot['id']}")
        print(f"SKIP {slot['id']} (verified completed run)", flush=True)
        return False
    if slot["id"] in reuse:
        verify_source_archive(reuse_source, slot["source"])
        run = Path(reuse[slot["id"]]).resolve()
        fingerprint = verify_run(slot, run, check_data=True)
        slot.update(status="completed", run_dir=str(run), result=read_result(run),
                    training_data_fingerprint=fingerprint,
                    imported_source_snapshot=str(Path(reuse_source).resolve()), completed_at=now())
        slot.pop("error", None)
        save_progress(output, plan)
        print(f"REUSE {slot['id']} from {run}", flush=True)
        return False
    return True


def external_gpu_processes(gpu_uuid, owned_pids):
    """Only PIDs returned by this supervisor's live Popen objects are allowed."""
    external = []
    for line in gpu_processes(gpu_uuid):
        fields = line.split(",")
        try:
            owned = int(fields[1].strip()) in owned_pids
        except (IndexError, ValueError):
            owned = False
        if not owned:
            external.append(line)
    return external


def launch_worker(output, plan, slot, gpu_uuid, lock_fds, active):
    task = output / "slots" / slot["id"]
    previous = discover_resume(slot)
    slot.update(status="running", started_at=now(), attempt=slot.get("attempt", 0) + 1,
                gpu_uuid=gpu_uuid)
    slot.pop("error", None)
    slot.pop("pid", None)
    if previous:
        slot["resume_checkpoint"] = str(previous / "last.pt")
    atomic_json(task / "worker.json", {"slot": slot, "gpu_uuid": gpu_uuid})
    (task / "completed.json").unlink(missing_ok=True)
    (task / "input.yaml").write_text(yaml.safe_dump(slot["config"], sort_keys=False))
    save_progress(output, plan)
    command = [sys.executable, "-m", "mmrecsys.experiment.lirdrec_suite", "--worker", str(task / "worker.json")]
    log_path = task / f"attempt-{slot['attempt']:03d}.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=worker_environment(gpu_uuid),
                                   stdout=log, stderr=subprocess.STDOUT, pass_fds=lock_fds)
        # Register before any status write or print can fail, so the group finally
        # block owns every launched child even when post-launch bookkeeping fails.
        active[process.pid] = {"process": process, "slot": slot, "task": task,
                               "started": time.monotonic(), "log": log_path}
    slot.update(pid=process.pid, log=str(log_path))
    save_progress(output, plan)
    print(f"START {slot['id']} pid={process.pid} log={log_path}", flush=True)


def finish_worker(record, code):
    slot, task = record["slot"], record["task"]
    slot.pop("pid", None)
    slot["worker_seconds"] = time.monotonic() - record["started"]
    try:
        if code != 0:
            raise ValueError(f"worker exit={code}")
        completed = json.loads((task / "completed.json").read_text())
        verify_run(slot, completed["run_dir"])
        if read_result(completed["run_dir"]) != completed["result"]:
            raise ValueError("Worker result differs from its run artifact")
        slot.update(completed, status="completed", completed_at=now())
        slot.pop("error", None)
    except Exception as error:
        slot.update(status="failed", error=f"{error}; inspect {record['log']}; last.pt is retained")
    print(f"{slot['status'].upper()} {slot['id']}", flush=True)
    return slot["status"] == "completed"


def stop_workers(output, plan, active):
    """Terminate all owned children first, then reap; keep completed peers intact."""
    if not active:
        return
    terminated = set()
    for record in active.values():
        process = record["process"]
        if process.poll() is None:
            process.terminate()
            terminated.add(process.pid)
    deadline = time.monotonic() + 20
    for record in active.values():
        process, slot = record["process"], record["slot"]
        try:
            code = process.wait(timeout=max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            process.kill()
            code = process.wait()
        if process.pid not in terminated or (code == 0 and (record["task"] / "completed.json").is_file()):
            finish_worker(record, code)
        else:
            slot.update(status="interrupted", error="Group stopped; last.pt is retained",
                        worker_seconds=time.monotonic() - record["started"])
            slot.pop("pid", None)
    active.clear()
    try:
        save_progress(output, plan)
    except OSError:
        pass  # Children are stopped even if the disk cannot save status.


def run_seed_group(output, plan, slots, gpu_uuid, lock_fds, reuse, reuse_source):
    pending, active, prepared = deque(slots), {}, set()
    next_gpu_check = 0.0
    stop_after_drain = False
    try:
        while pending or active:
            # Always reap first. Waiting for someone else's GPU job must never
            # prevent recording completion or failure of our own workers.
            for pid, record in list(active.items()):
                code = record["process"].poll()
                if code is None:
                    continue
                del active[pid]
                success = finish_worker(record, code)
                save_progress(output, plan)
                if not success:
                    return False
            if stop_after_drain and not active:
                return False
            while pending and len(active) < plan["max_parallel"] and not stop_after_drain:
                slot = pending[0]
                if slot["id"] not in prepared:
                    needed = prepare_slot(output, plan, slot, reuse, reuse_source)
                    if needed is None:
                        return False
                    if not needed:
                        pending.popleft()
                        continue
                    prepared.add(slot["id"])
                if time.monotonic() < next_gpu_check:
                    break
                external = external_gpu_processes(gpu_uuid, set(active))
                if external:
                    message = "GPU has external processes: " + "; ".join(external)
                    if slot.get("status") != "waiting_gpu" or slot.get("error") != message:
                        slot.update(status="waiting_gpu", error=message)
                        save_progress(output, plan)
                        print(f"WAITING_GPU {slot['id']}: {message}", flush=True)
                    next_gpu_check = time.monotonic() + 30
                    if not plan.get("wait_for_gpu", False):
                        stop_after_drain = True
                    break
                pending.popleft()
                try:
                    launch_worker(output, plan, slot, gpu_uuid, lock_fds, active)
                except Exception as error:
                    # If Popen succeeded, finally owns its child; otherwise this
                    # slot failed to launch and no checkpoint is discarded.
                    if not any(record["slot"] is slot for record in active.values()):
                        slot.update(status="failed", error=f"Worker launch failed: {error}")
                        try:
                            save_progress(output, plan)
                        except OSError:
                            pass
                    raise
            if stop_after_drain and not active:
                return False
            if active:
                time.sleep(0.2)
            elif pending:
                time.sleep(max(0.1, min(30, next_gpu_check - time.monotonic())))
        return True
    finally:
        stop_workers(output, plan, active)


def run_suite(output, plan, *, reuse=None, reuse_source=None, gpu_uuid=None, wait_for_gpu=None, max_parallel=None):
    output = Path(output).resolve()
    reuse = reuse or {}
    unknown = reuse.keys() - {slot["id"] for slot in plan["slots"]}
    if unknown:
        raise ValueError(f"Unknown reuse slots: {sorted(unknown)}")
    if reuse and not reuse_source:
        raise ValueError("Importing an existing run requires --reuse-source snapshot.tar.gz")
    with suite_resources(output, plan, gpu_uuid, wait_for_gpu, max_parallel) as (suite_fd, gpu_fd, gpu_uuid):
        for slot in plan["slots"]:
            if source_identity(slot["dataset"]) != slot["source"]:
                raise ValueError(f"Training source changed for {slot['id']}; use a new suite")
            if digest(comparable_config(slot["config"])) != slot["config_digest"]:
                raise ValueError(f"Saved slot configuration changed: {slot['id']}")
        # Seed barriers are strict. Within one seed, datasets/variants may overlap
        # to fill the authorized worker limit, always on the same physical GPU.
        for seed in dict.fromkeys(slot["seed"] for slot in plan["slots"]):
            slots = [slot for slot in plan["slots"] if slot["seed"] == seed]
            if not run_seed_group(output, plan, slots, gpu_uuid, (suite_fd, gpu_fd), reuse, reuse_source):
                return False
        save_progress(output, plan)
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    parser.add_argument("--resume")
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--wait-for-gpu", action=argparse.BooleanOptionalAction, default=None,
                        help="Wait on the selected GPU when busy; saved for subsequent resumes")
    parser.add_argument("--max-parallel", type=int, choices=range(1, 11), default=None,
                        help="Concurrent independent workers on the one GPU (1-10); saved on resume")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--reuse", action="append", default=[], metavar="SLOT_ID=RUN_DIR")
    parser.add_argument("--reuse-source", help="Source snapshot associated with imported runs")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        worker(args.worker)
        return 0
    if args.resume:
        if args.output or args.datasets or args.seeds or args.overrides:
            parser.error("--resume uses saved output, datasets, seeds and configs; --gpu-uuid may migrate an inactive suite")
        output = Path(args.resume).resolve()
        plan = json.loads((output / "suite.json").read_text())
        if plan.get("format_version") != FORMAT_VERSION:
            parser.error("Unsupported suite format")
    else:
        if not args.output or not args.gpu_uuid:
            parser.error("New suites require --output and --gpu-uuid")
        if not GPU_PATTERN.fullmatch(args.gpu_uuid):
            parser.error("--gpu-uuid requires one full GPU UUID; indices and auto are forbidden")
        output = Path(args.output).resolve()
        datasets, seeds = args.datasets or ["baby", "sports", "clothing", "elec"], args.seeds or [999, 2024, 2025]
        plan = {"format_version": FORMAT_VERSION, "created_at": now(), "gpu_uuid": args.gpu_uuid,
                "datasets": datasets, "seeds": seeds, "slots": build_slots(datasets, seeds, output, args.overrides)}
    reuse = {}
    for expression in args.reuse:
        key, separator, path = expression.partition("=")
        if not separator or not key or not path or key in reuse:
            parser.error("Each --reuse must be a unique SLOT_ID=RUN_DIR")
        reuse[key] = path
    if args.dry_run:
        print(json.dumps({"output": str(output), **plan, "requested_gpu_uuid": args.gpu_uuid,
                          "requested_wait_for_gpu": args.wait_for_gpu, "requested_max_parallel": args.max_parallel}, indent=2))
        return 0
    if not args.resume:
        output.mkdir(parents=True, exist_ok=False)
        save_progress(output, plan)
    # SIGTERM gets the same cleanup as Ctrl-C; never leave a child training alone.
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    success = run_suite(output, plan, reuse=reuse, reuse_source=args.reuse_source,
                        gpu_uuid=args.gpu_uuid, wait_for_gpu=args.wait_for_gpu, max_parallel=args.max_parallel)
    print(f"suite={output}; completed={sum(slot['status'] == 'completed' for slot in plan['slots'])}/{len(plan['slots'])}", flush=True)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
