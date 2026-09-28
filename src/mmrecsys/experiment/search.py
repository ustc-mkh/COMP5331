"""Search paired MGCN/DAMPS validation improvements, without testing candidates."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
from pathlib import Path
from statistics import mean, stdev
import uuid

import yaml

from ..config import PROJECT_ROOT, load_config, read_yaml, validate
from ..data.dataset import load_dataset
from .ablation import variant_config
from .logging import write_json
from .runner import train_experiment, evaluate_experiment


ALLOWED = {"model.n_ui_layers", "model.n_item_layers", "model.knn_k", "model.cl_weight",
           "model.reg_weight", "model.embedding_dim", "model.temperature", "optimizer.lr"}


def candidates(base, grid, seeds):
    if base["model"]["name"] != "mgcn" or base["eval"]["mode"] != "max":
        raise ValueError("Search requires MGCN and an increasing validation metric")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be nonempty and unique")
    if not grid or grid.keys() - ALLOWED:
        raise ValueError(f"Grid keys must be selected from {sorted(ALLOWED)}")
    for values in grid.values():
        if not isinstance(values, list) or not values:
            raise ValueError("Each grid dimension must be a nonempty list")
        if len({json.dumps(v, sort_keys=True) for v in values}) != len(values):
            raise ValueError("Duplicate grid values")
    result = []
    for values in product(*grid.values()):
        settings = dict(zip(grid, values))
        config = deepcopy(base)
        for key, value in settings.items():
            group, field = key.split(".")
            config[group][field] = value
        for seed in seeds:
            variant_config(config, "full", seed)
        result.append((settings, validate(config)))
    return result


def code_digest():
    digest = hashlib.sha256()
    for path in sorted((PROJECT_ROOT / "src/mmrecsys").rglob("*.py")):
        digest.update(str(path.relative_to(PROJECT_ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    write_json(temporary, value)
    temporary.replace(path)


def score_trial(settings, pairs, objective, minimum):
    baseline = [pair["baseline"]["validation"] for pair in pairs]
    full = [pair["full"]["validation"] for pair in pairs]
    absolute = [d - b for b, d in zip(baseline, full)]
    relative = [100 * (d - b) / b for b, d in zip(baseline, full)] if all(b > 0 for b in baseline) else None
    score = mean(relative) if objective == "relative" and relative is not None else (
        mean(absolute) if objective == "absolute" else None)
    return {"settings": settings, "pairs": pairs, "baseline_mean": mean(baseline),
            "damps_mean": mean(full), "absolute_gain_mean": mean(absolute),
            "relative_gain_percent_mean": mean(relative) if relative is not None else None,
            "relative_gain_percent_std": stdev(relative) if relative is not None and len(relative) > 1 else None,
            "positive_seed_count": sum(x > 0 for x in absolute), "score": score,
            "eligible": score is not None and mean(baseline) >= minimum}


def run_slot(config, slot):
    config = deepcopy(config)
    config["runtime"]["output_root"] = str(slot)
    existing = sorted(slot.glob("*/config.yaml")) if slot.exists() else []
    resume = None
    if existing:
        run = existing[-1].parent
        saved = read_yaml(run / "config.yaml")
        if saved != config:
            raise ValueError(f"Existing run configuration differs: {run}")
        result_path = run / "validation_result.json"
        if result_path.exists():
            result = json.loads(result_path.read_text())
            return {"run_dir": str(run), "validation": result["best_validation_metric"],
                    "best_epoch": result["best_epoch"]}
        if (run / "last.pt").exists() and (run / "best.pt").exists():
            resume = run / "last.pt"
    run, result = train_experiment(config, resume=resume, evaluate_test=False)
    return {"run_dir": str(run), "validation": result["best_validation_metric"],
            "best_epoch": result["best_epoch"]}


def run_search(base=None, grid=None, seeds=None, *, resume=None, objective="relative",
               min_baseline=0.0, limit=None, evaluate_best=False):
    if resume:
        output = (PROJECT_ROOT / resume).resolve()
        plan = json.loads((output / "plan.json").read_text())
        base, grid, seeds = plan["base"], plan["grid"], plan["seeds"]
        objective, min_baseline, limit = plan["objective"], plan["min_baseline"], plan["limit"]
    if objective not in ("relative", "absolute") or not 0 <= min_baseline < float('inf'):
        raise ValueError("Invalid objective or minimum baseline")
    all_trials = candidates(base, grid, seeds)
    if limit is not None:
        if type(limit) is not int or limit < 1:
            raise ValueError("limit must be a positive integer")
        all_trials = all_trials[:limit]
    _, _, metadata = load_dataset(base["data"], ("image", "text"), base["eval"])
    if any(config["model"]["knn_k"] > metadata["n_items"] - int(not config["model"]["knn_self_loops"])
           for _, config in all_trials):
        raise ValueError("Grid k exceeds available item neighbors")
    fingerprint, code = metadata["fingerprint"], code_digest()
    if resume:
        if plan["fingerprint"] != fingerprint or plan["code_digest"] != code:
            raise ValueError("Data or source code changed; start a new search")
    else:
        identifier = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
        output = Path(base["runtime"]["output_root"]) / f"search-{base['data']['name']}-{identifier}"
        output.mkdir(parents=True, exist_ok=False)
        plan = {"base": base, "grid": grid, "seeds": seeds, "objective": objective,
                "min_baseline": min_baseline, "limit": limit, "fingerprint": fingerprint,
                "code_digest": code, "selection": "paired validation gain; no candidate test evaluation"}
        atomic_json(output / "plan.json", plan)
    print(f"search_dir={output}; settings={len(all_trials)}; training_runs={2 * len(seeds) * len(all_trials)}", flush=True)
    ranked = []
    for index, (settings, config) in enumerate(all_trials):
        pairs = []
        for seed in seeds:
            pair = {"seed": seed}
            for variant in ("baseline", "full"):
                print(f"trial={index + 1}/{len(all_trials)} seed={seed} variant={variant} settings={settings}", flush=True)
                slot = output / f"trial-{index:04d}" / f"seed-{seed}" / variant
                pair[variant] = run_slot(variant_config(config, variant, seed), slot)
            pairs.append(pair)
        record = score_trial(settings, pairs, objective, min_baseline)
        record["trial_index"] = index
        ranked.append(record)
        ranked.sort(key=lambda r: (r["eligible"], r["score"] if r["score"] is not None else -float('inf'),
                                  r["damps_mean"], -r["trial_index"]), reverse=True)
        atomic_json(output / "leaderboard.json", ranked)
        if ranked[0]["eligible"]:
            atomic_json(output / "best.json", ranked[0])
            _, winner = all_trials[ranked[0]["trial_index"]]
            for variant in ("baseline", "full"):
                selected = variant_config(winner, variant, seeds[0])
                (output / f"best_{variant}.yaml").write_text(yaml.safe_dump(selected, sort_keys=False))
        print(f"validation gain={record['score']} ({objective}); completed settings={len(ranked)}", flush=True)
    eligible = [r for r in ranked if r["eligible"]]
    if not eligible:
        print("No eligible setting: check zero baseline metrics or --min-baseline", flush=True)
    elif evaluate_best:
        test_path = output / "best_test.json"
        if not test_path.exists():
            tests = [{"seed": pair["seed"], **{variant: evaluate_experiment(pair[variant]["run_dir"])
                     for variant in ("baseline", "full")}} for pair in eligible[0]["pairs"]]
            atomic_json(test_path, {"settings": eligible[0]["settings"], "tests": tests})
    if eligible:
        winner = eligible[0]
        print(f"best_settings={winner['settings']}; validation_baseline={winner['baseline_mean']:.6f}; "
              f"validation_damps={winner['damps_mean']:.6f}; "
              f"mean_relative_gain_percent={winner['relative_gain_percent_mean']}", flush=True)
        if winner["score"] <= 0:
            print("No positive validation gain found among eligible settings.", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--grid")
    parser.add_argument("--device")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--objective", choices=("relative", "absolute"), default=None)
    parser.add_argument("--min-baseline", type=float)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--resume")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--evaluate-best", action="store_true")
    args = parser.parse_args()
    if args.resume:
        if any(v is not None for v in (args.config, args.grid, args.device, args.seeds, args.objective, args.min_baseline, args.limit)) or args.overrides or args.dry_run:
            parser.error("--resume uses the saved plan; only --evaluate-best may be added")
        output = run_search(resume=args.resume, evaluate_best=args.evaluate_best)
    else:
        overrides = args.overrides + ([f"runtime.device={args.device}"] if args.device else [])
        base = load_config(args.config or "configs/experiments/damps_mgcn_baby.yaml", overrides=overrides)
        grid = read_yaml(PROJECT_ROOT / (args.grid or "configs/search/damps_mgcn.yaml"))
        seeds = args.seeds or [999]
        if args.dry_run:
            trials = candidates(base, grid, seeds)
            if args.limit is not None and args.limit < 1:
                parser.error("--limit must be positive")
            count = min(len(trials), args.limit) if args.limit is not None else len(trials)
            print(json.dumps({"settings": count, "training_runs": count * 2 * len(seeds), "seeds": seeds,
                              "objective": args.objective or "relative", "grid": grid}, indent=2))
            return
        output = run_search(base, grid, seeds, objective=args.objective or "relative",
                            min_baseline=args.min_baseline if args.min_baseline is not None else 0.0,
                            limit=args.limit, evaluate_best=args.evaluate_best)
    print(f"search_dir={output}")


if __name__ == "__main__":
    main()
