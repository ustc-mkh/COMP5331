"""Compare paired MGCN/DAMPS runs, scheduling one subprocess per device."""
import argparse
from collections import deque
from copy import deepcopy
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from statistics import mean, stdev
import subprocess
import sys
import time
import uuid

import torch
import yaml

from ..config import PROJECT_ROOT, load_config
from .ablation import variant_config
from .logging import write_json

DATASETS = ("baby", "sports", "clothing", "elec")


def devices_for(gpus):
    if gpus == ["cpu"]:
        return ["cpu"]
    if gpus == ["auto"]:
        return [f"cuda:{i}" for i in range(torch.cuda.device_count())] or ["cpu"]
    if not gpus or any(not value.isdecimal() for value in gpus):
        raise ValueError("--gpus expects auto, cpu, or space-separated visible GPU indices")
    indices = [int(value) for value in gpus]
    if len(set(indices)) != len(indices):
        raise ValueError("GPU indices must be unique")
    if any(index >= torch.cuda.device_count() for index in indices):
        raise ValueError("Requested GPU is unavailable (indices respect CUDA_VISIBLE_DEVICES)")
    return [f"cuda:{index}" for index in indices]


def build_jobs(datasets, seeds, mode, overrides):
    if not datasets or len(set(datasets)) != len(datasets):
        raise ValueError("Datasets must be nonempty and unique")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be nonempty and unique")
    # Overrides apply equally to both members of every pair.
    defaults = ["train.epochs=20", "train.patience=5"] if mode == "quick" else []
    jobs = []
    for dataset in datasets:
        base = load_config(f"configs/experiments/damps_mgcn_{dataset}.yaml",
                           overrides=defaults + overrides)
        if base["data"]["name"] != dataset or base["model"]["name"] != "mgcn":
            raise ValueError("Use --datasets to select datasets; the comparison requires MGCN")
        for seed in seeds:
            for variant in ("baseline", "full"):
                config = variant_config(base, variant, seed)
                jobs.append({"id": f"{dataset}-{seed}-{variant}", "dataset": dataset,
                             "seed": seed, "variant": variant, "config": config,
                             "status": "pending"})
    return jobs


def preflight(jobs):
    missing = set()
    for job in jobs:
        data = job["config"]["data"]
        root = Path(data["root"]) / data["name"]
        for filename in (data["interactions"], *data["features"].values()):
            if not (root / filename).is_file():
                missing.add(str(root / filename))
    if missing:
        raise ValueError("Missing data files:\n" + "\n".join(sorted(missing)))


def stats(values):
    return {"mean": mean(values), "sample_std": stdev(values) if len(values) > 1 else None}


def summarize(jobs):
    """Only complete seed pairs contribute to aggregate comparisons."""
    rows = []
    for dataset in dict.fromkeys(job["dataset"] for job in jobs):
        selected = [job for job in jobs if job["dataset"] == dataset]
        pairs = []
        for seed in dict.fromkeys(job["seed"] for job in selected):
            pair = {job["variant"]: job for job in selected
                    if job["seed"] == seed and job["status"] == "completed"}
            if pair.keys() == {"baseline", "full"}:
                pairs.append((seed, pair["baseline"]["result"]["test"], pair["full"]["result"]["test"]))
        if not pairs:
            continue
        for metric in pairs[0][1]:
            if metric == "n_users":
                continue
            baseline = [b[metric] for _, b, _ in pairs]
            damps = [d[metric] for _, _, d in pairs]
            if not all(math.isfinite(v) for v in baseline + damps):
                raise ValueError(f"Nonfinite metric: {dataset} {metric}")
            deltas = [d - b for b, d in zip(baseline, damps)]
            b, d, delta = stats(baseline), stats(damps), stats(deltas)
            paired_percent = [100 * (dv - bv) / bv for bv, dv in zip(baseline, damps) if bv != 0]
            rows.append({"dataset": dataset, "metric": metric, "n_pairs": len(pairs),
                         "seeds": [seed for seed, _, _ in pairs],
                         "baseline_mean": b["mean"], "baseline_std": b["sample_std"],
                         "damps_mean": d["mean"], "damps_std": d["sample_std"],
                         "delta_mean": delta["mean"], "delta_std": delta["sample_std"],
                         "relative_gain_pct": 100 * delta["mean"] / b["mean"] if b["mean"] else None,
                         "paired_gain_pct_mean": mean(paired_percent) if paired_percent else None,
                         "paired_gain_defined_count": len(paired_percent)})
    return rows


def save_reports(output, jobs, mode):
    # Atomic progress snapshots remain usable after interruption.
    temporary = output / "runs.json.tmp"
    write_json(temporary, {"mode": mode, "jobs": jobs})
    temporary.replace(output / "runs.json")
    rows = summarize(jobs)
    write_json(output / "summary.json", {"mode": mode,
               "selection": "Per-run best validation checkpoint; test never used for selection",
               "complete": all(job["status"] == "completed" for job in jobs),
               "metrics": rows})
    fields = list(rows[0]) if rows else ["dataset", "metric", "n_pairs"]
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [f"Mode: {mode}. Completed runs: {sum(j['status'] == 'completed' for j in jobs)}/{len(jobs)}.",
             "Only complete seed pairs are included. Checkpoints selected on validation.",
             "", "| Dataset | Metric | Pairs | MGCN | MGCN+DAMPS | Delta | Gain |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        gain = "N/A" if row["relative_gain_pct"] is None else f"{row['relative_gain_pct']:+.2f}%"
        lines.append(f"| {row['dataset']} | {row['metric']} | {row['n_pairs']} | "
                     f"{row['baseline_mean']:.6f} | {row['damps_mean']:.6f} | "
                     f"{row['delta_mean']:+.6f} | {gain} |")
    if mode == "quick":
        lines += ["", "Short-budget smoke comparison; not evidence of convergence or paper reproduction."]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_jobs(jobs, devices, output, mode):
    """Launch fresh interpreters; never fork an initialized CUDA context."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    pending = deque(job for job in jobs if job["status"] != "completed")
    active = {}
    save_reports(output, jobs, mode)
    try:
        while pending or active:
            for device in devices:
                if device in active or not pending:
                    continue
                job = pending.popleft()
                task_dir = output / job["id"]
                task_dir.mkdir(exist_ok=True)
                config = deepcopy(job["config"])
                config["runtime"]["device"] = device
                # Every attempt has its own directory, including retries of interrupted jobs.
                attempt = task_dir / uuid.uuid4().hex[:8]
                attempt.mkdir()
                config["runtime"]["output_root"] = str(attempt)
                config_path = attempt / "input.yaml"
                config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
                log_path = attempt / "train.log"
                env = os.environ.copy()
                env["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
                env["OMP_NUM_THREADS"] = str(config["runtime"]["num_threads"])
                env["MKL_NUM_THREADS"] = env["OMP_NUM_THREADS"]
                with log_path.open("w", encoding="utf-8") as log:
                    process = subprocess.Popen([sys.executable, "-m", "mmrecsys.cli", "train",
                                                "--config", str(config_path)], cwd=PROJECT_ROOT,
                                               env=env, stdout=log, stderr=subprocess.STDOUT)
                job.update(status="running", device=device, log=str(log_path))
                job.pop("error", None)
                active[device] = (process, job, attempt, time.monotonic())
                print(f"START {job['id']} on {device}; log={log_path}", flush=True)
                save_reports(output, jobs, mode)
            for device, (process, job, attempt, started) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                job["seconds"] = time.monotonic() - started
                results = list(attempt.glob("*/result.json"))
                if code == 0 and len(results) == 1:
                    try:
                        result = json.loads(results[0].read_text())
                        values = result["test"]
                        if not values or not all(math.isfinite(v) for v in values.values()):
                            raise ValueError("Empty or nonfinite test metrics")
                        job.update(status="completed", run_dir=str(results[0].parent), result=result)
                    except (ValueError, KeyError, TypeError) as error:
                        job.update(status="failed", error=str(error))
                else:
                    job.update(status="failed", error=f"exit code {code}; found {len(results)} result files")
                print(f"{job['status'].upper()} {job['id']} on {device}", flush=True)
                del active[device]
                save_reports(output, jobs, mode)
            if active:
                time.sleep(0.2)
    finally:
        # Terminate all children before waiting, so Ctrl-C does not leave GPU jobs behind.
        for process, _, _, _ in active.values():
            if process.poll() is None:
                process.terminate()
        for process, job, _, _ in active.values():
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            job["status"] = "interrupted"
        save_reports(output, jobs, mode)
    return all(job["status"] == "completed" for job in jobs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--gpus", nargs="+", default=["auto"], help="auto, cpu, or visible GPU indices, e.g. 0 1 2 3")
    parser.add_argument("--mode", choices=("quick", "full"))
    parser.add_argument("--set", action="append", default=[], dest="overrides", metavar="KEY=VALUE")
    parser.add_argument("--output", help="New comparison directory")
    parser.add_argument("--resume", help="Reuse completed runs; retry incomplete runs from scratch")
    parser.add_argument("--dry-run", action="store_true", help="Show configs without training or checking GPU/data availability")
    args = parser.parse_args(argv)
    if args.resume and (args.output or args.overrides or args.datasets or args.seeds or args.mode):
        parser.error("--resume uses saved settings; only --gpus and --dry-run may accompany it")
    if args.resume:
        output = (PROJECT_ROOT / args.resume).resolve()
        saved = json.loads((output / "runs.json").read_text())
        jobs, mode = saved["jobs"], saved["mode"]
    else:
        mode = args.mode or "quick"
        jobs = build_jobs(args.datasets or list(DATASETS), args.seeds or [999], mode, args.overrides)
        identifier = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
        output = (PROJECT_ROOT / args.output).resolve() if args.output else PROJECT_ROOT / "runs" / f"compare-{identifier}"
    if args.dry_run:
        print(json.dumps({"output": str(output), "mode": mode, "gpus": args.gpus, "jobs": jobs}, indent=2))
        return 0
    devices = devices_for(args.gpus)
    preflight(jobs)
    if not args.resume:
        output.mkdir(parents=True, exist_ok=False)
    print(f"comparison_dir={output}\ndevices={devices}; runs={len(jobs)}; mode={mode}", flush=True)
    success = run_jobs(jobs, devices, output, mode)
    print((output / "summary.md").read_text(), flush=True)
    print(f"Reports: {output / 'summary.csv'}\nResume: --resume {output}", flush=True)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
