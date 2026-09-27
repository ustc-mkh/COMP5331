import argparse
import json

from .config import PROJECT_ROOT, load_config, merge, parse_overrides, read_yaml, validate
from .experiment.runner import evaluate_experiment, train_experiment


def main(argv=None):
    parser = argparse.ArgumentParser(description="Independent multimedia recommendation experiments")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--config")
    train.add_argument("--model")
    train.add_argument("--dataset")
    train.add_argument("--seed", type=int)
    train.add_argument("--device")
    train.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    train.add_argument("--resume", help="Path to last.pt; reloads that run's saved configuration")
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--run", required=True)
    evaluate.add_argument("--split", choices=("valid", "test"), default="test")
    evaluate.add_argument("--device")
    args = parser.parse_args(argv)
    if args.command == "evaluate":
        result = evaluate_experiment(args.run, args.split, args.device)
    else:
        overrides = list(args.set)
        if args.seed is not None:
            overrides.append(f"seed={args.seed}")
        if args.device:
            overrides.append(f"runtime.device={args.device}")
        if args.resume:
            if args.config or args.model or args.dataset:
                parser.error("--resume uses the saved config; use --set for compatible overrides")
            resume = (PROJECT_ROOT / args.resume).resolve()
            config = validate(merge(read_yaml(resume.parent / "config.yaml"), parse_overrides(overrides)))
        else:
            resume = None
            config = load_config(args.config, args.model, args.dataset, overrides)
        run_dir, result = train_experiment(config, resume)
        result = {"run_dir": str(run_dir), **result}
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
