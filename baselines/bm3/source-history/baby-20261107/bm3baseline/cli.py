import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

from . import ROOT
from .config import load_config, write_json
from .data import inspect_data


def verify_upstream():
    subprocess.run([sys.executable, str(ROOT / "scripts" / "fetch_upstream.py"), "--verify"], check=True)


def build_parser():
    parser = argparse.ArgumentParser(description="Original BM3 / Baby baseline preparation and experiments")
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="environment/source inspection; CPU by default")
    doctor.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    doctor.add_argument("--output", type=Path)
    check = sub.add_parser("check-data", help="validate released splits, feature rows, finite values and hashes")
    check.add_argument("--data-dir", type=Path, default=ROOT / "data" / "baby")
    check.add_argument("--output", type=Path, default=ROOT / "validation" / "baby_data.json")
    for name in ("train", "smoke"):
        task = sub.add_parser(name, help="formal training" if name == "train" else "two-epoch CPU smoke test, not report results")
        task.add_argument("--config", type=Path)
        task.add_argument("--data-dir", type=Path, default=ROOT / "data" / "baby")
        task.add_argument("--run-dir", type=Path, required=True)
        task.add_argument("--seed", type=int)
        if name == "train":
            task.add_argument("--device", choices=["cpu", "cuda"], required=True)
            task.add_argument("--epochs", type=int)
            task.add_argument("--resume", type=Path)
            task.add_argument("--validation-only", action="store_true")
        else:
            task.set_defaults(device="cpu", epochs=2, resume=None, validation_only=False)
    evaluate = sub.add_parser("evaluate", help="reload a checkpoint and independently calculate validation/test metrics")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--data-dir", type=Path, default=ROOT / "data" / "baby")
    evaluate.add_argument("--device", choices=["cpu", "cuda"], required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    sweep = sub.add_parser("sweep", help="prepare original 8-setting grid; execution requires --execute")
    sweep.add_argument("--config", type=Path)
    sweep.add_argument("--grid", type=Path, default=ROOT / "configs" / "baby_grid.json")
    sweep.add_argument("--data-dir", type=Path, default=ROOT / "data" / "baby")
    sweep.add_argument("--run-dir", type=Path, required=True)
    sweep.add_argument("--seed", type=int, default=999)
    sweep.add_argument("--device", choices=["cpu", "cuda"], required=True)
    sweep.add_argument("--execute", action="store_true")
    sweep.add_argument("--index", type=int, help="run one zero-based grid job, for independent GPUs/machines")
    select = sub.add_parser("select-sweep", help="select completed grid by validation Recall@20; never by test")
    select.add_argument("--run-dir", type=Path, required=True)
    aggregate = sub.add_parser("aggregate", help="mean/sample std and paper comparison; excludes smoke runs")
    aggregate.add_argument("--runs", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    plot = sub.add_parser("plot", help="plot training loss and validation Recall@20")
    plot.add_argument("--run-dir", type=Path, required=True)
    plot.add_argument("--output", type=Path)
    return parser


def sweep(args):
    config = load_config(args.config, {"seed": args.seed})
    grid = json.loads(args.grid.read_text(encoding="utf-8"))
    if set(grid) != {"n_layers", "dropout", "reg_weight"} or any(not values for values in grid.values()):
        raise ValueError("Grid must specify n_layers, dropout and reg_weight")
    jobs = []
    for index, values in enumerate(itertools.product(*grid.values())):
        candidate = load_config(args.config, dict(zip(grid.keys(), values), seed=args.seed))
        jobs.append({"index": index, "directory": "job-{:02d}".format(index), "config": candidate})
    args.run_dir.mkdir(parents=True, exist_ok=True)
    plan = {"base_config": config, "jobs": jobs, "selection": "validation recall@20, first job on ties"}
    plan_path = args.run_dir / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text(encoding="utf-8")) != plan:
        raise ValueError("A different sweep plan exists in this directory")
    write_json(plan_path, plan)
    print("Prepared {} jobs at {}".format(len(jobs), args.run_dir.resolve()), flush=True)
    if not args.execute:
        print("Plan only. No training started. Add --execute when a GPU is available.")
        return
    from .engine import train
    data_report, _ = inspect_data(args.data_dir)
    if args.index is not None:
        if not 0 <= args.index < len(jobs):
            raise ValueError("Grid index is out of range")
        jobs = [jobs[args.index]]
    for job in jobs:
        run_dir = args.run_dir / job["directory"]
        existing = run_dir / "result.json"
        if existing.exists():
            result = json.loads(existing.read_text(encoding="utf-8"))
            if result["config"] != job["config"] or result["test"] is not None or result["data_fingerprint"] != data_report["fingerprint"]:
                raise ValueError("Existing result does not match this validation-only job")
            print("Already completed:", run_dir, flush=True)
            continue
        restart = run_dir / "last.pt"
        train(job["config"], args.data_dir, run_dir, args.device, test_at_end=False,
              resume=restart if restart.exists() else None)


def select_sweep(directory):
    import yaml
    plan = json.loads((directory / "plan.json").read_text(encoding="utf-8"))
    candidates = []
    fingerprint = None
    for job in plan["jobs"]:
        result = json.loads((directory / job["directory"] / "result.json").read_text(encoding="utf-8"))
        if result["kind"] != "formal" or result["test"] is not None or result["config"] != job["config"]:
            raise ValueError("Grid selection requires completed formal validation-only jobs")
        if fingerprint is not None and result["data_fingerprint"] != fingerprint:
            raise ValueError("Grid jobs used different datasets")
        fingerprint = result["data_fingerprint"]
        candidates.append((result["best_validation"]["recall@20"], job))
    score, selected = max(candidates, key=lambda pair: pair[0])
    (directory / "selected.yaml").write_text(yaml.safe_dump(selected["config"], sort_keys=False), encoding="utf-8")
    write_json(directory / "selection.json", {"job": selected, "validation_recall@20": score,
                                              "test_used_for_selection": False, "data_fingerprint": fingerprint})
    print("Selected", selected["directory"], "by validation Recall@20:", score)
    print("Config:", (directory / "selected.yaml").resolve())


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.command in {"doctor", "train", "smoke", "evaluate", "sweep"}:
            verify_upstream()
        if args.command == "doctor":
            import torch
            from .engine import environment_info
            info = environment_info(args.device)
            if args.device == "cuda":
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA is unavailable")
                x = torch.ones(1, device="cuda")
                info["cuda_tensor_probe"] = float(x.cpu().item())
            info["expected_torch"] = "2.4.1"
            info["torch_version_matches"] = torch.__version__.split("+")[0] == "2.4.1"
            if args.output:
                write_json(args.output, info)
            print(json.dumps(info, indent=2))
            if not info["torch_version_matches"]:
                raise RuntimeError("Use the prepared PyTorch 2.4.1 environment")
        elif args.command == "check-data":
            report, _ = inspect_data(args.data_dir)
            write_json(args.output, report)
            print(json.dumps(report, indent=2))
        elif args.command in {"train", "smoke"}:
            from .engine import train
            config = load_config(args.config, {"seed": args.seed, "epochs": args.epochs})
            train(config, args.data_dir, args.run_dir, args.device, smoke=args.command == "smoke",
                  test_at_end=not args.validation_only, resume=args.resume)
        elif args.command == "evaluate":
            from .engine import evaluate_checkpoint
            result = evaluate_checkpoint(args.checkpoint, args.data_dir, args.device, output=args.output)
            print(json.dumps(result, indent=2))
        elif args.command == "sweep":
            sweep(args)
        elif args.command == "select-sweep":
            select_sweep(args.run_dir)
        elif args.command == "aggregate":
            from .report import aggregate
            print(json.dumps(aggregate(args.runs, args.output), indent=2))
        elif args.command == "plot":
            from .report import plot_history
            print(plot_history(args.run_dir, args.output))
    except (ValueError, RuntimeError, FileNotFoundError, FloatingPointError) as exc:
        parser.exit(1, "ERROR: {}\n".format(exc))
