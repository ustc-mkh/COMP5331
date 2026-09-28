"""Paired MGCN/DAMPS experiments with fixed backbone settings across variants."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
import uuid

from ..config import load_config, validate
from .logging import write_json
from .runner import train_experiment


VARIANTS = {
    "baseline": (False, True, True, True),
    "full": (True, True, True, True),
    "no_apc": (True, False, True, True),
    "no_avrf": (True, True, False, True),
    "no_imcf": (True, True, True, False),
}


def variant_config(base, variant, seed):
    config = deepcopy(base)
    config["seed"] = seed
    for name, flag in zip(("damps_enabled", "damps_apc", "damps_avrf", "damps_imcf"), VARIANTS[variant]):
        config["model"][name] = flag
    return validate(config)


def run_ablation(base, seeds, variants):
    if not seeds or not variants or len(set(seeds)) != len(seeds) or len(set(variants)) != len(variants):
        raise ValueError("Provide nonempty, unique seeds and variants")
    # Validate all experiments before starting any training.
    experiments = [(variant, seed, variant_config(base, variant, seed))
                   for seed in seeds for variant in variants]
    identifier = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    output = Path(base["runtime"]["output_root"]) / f"ablation-{base['data']['name']}-{identifier}"
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for variant, seed, config in experiments:
        config["runtime"]["output_root"] = str(output / variant)
        print(f"variant={variant} seed={seed}", flush=True)
        run_dir, result = train_experiment(config)
        records.append({"variant": variant, "seed": seed, "run_dir": str(run_dir), "result": result})
        # Keep completed run locations even if a subsequent training is interrupted.
        write_json(output / "runs.json", records)
    summary = {}
    for variant in variants:
        results = [record["result"] for record in records if record["variant"] == variant]
        metrics = {"validation_best": [r["best_validation_metric"] for r in results]}
        metrics.update({name: [r["test"][name] for r in results]
                        for name in results[0]["test"] if name != "n_users"})
        summary[variant] = {name: {"mean": mean(values), "sample_std": stdev(values) if len(values) > 1 else None}
                            for name, values in metrics.items()}
    write_json(output / "summary.json", {"seeds": seeds, "variants": variants,
                                         "selection": "validation checkpoint per run; no selection across test results",
                                         "metrics": summary})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/damps_mgcn_baby.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[999, 2024, 2025])
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    args = parser.parse_args()
    overrides = args.overrides + ([f"runtime.device={args.device}"] if args.device else [])
    config = load_config(args.config, overrides=overrides)
    print(f"ablation_dir={run_ablation(config, args.seeds, args.variants)}")


if __name__ == "__main__":
    main()
