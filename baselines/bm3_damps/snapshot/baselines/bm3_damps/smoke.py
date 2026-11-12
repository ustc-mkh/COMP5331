"""Two epochs/two batches on real data; never used as report results."""
import argparse
import json
from pathlib import Path

from model import BM3Damps
from bm3baseline import engine


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", default="baby")
    parser.add_argument("--variants", nargs="+", default=["full", "no_apc", "no_avrf", "no_imcf"])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    runs = json.loads((root / "docs/results/bm3/runs.json").read_text())["runs"]
    original = next(r for r in runs if r["dataset"] == args.dataset and r["seed"] == 999)
    engine.BM3 = BM3Damps
    for variant in args.variants:
        config = dict(original["config"], damps_variant=variant, epochs=2)
        directory = args.output / (args.dataset + "-" + variant)
        result = engine.train(config, args.data / args.dataset, directory, "cuda", smoke=True)
        check = engine.evaluate_checkpoint(directory / "best.pt", args.data / args.dataset,
                                           "cuda", output=directory / "reloaded.json")
        assert result["data_fingerprint"] == original["data_fingerprint"]
        assert result["best_epoch"] == check["epoch"]
        assert result["best_validation"] == check["validation"] and result["test"] == check["test"]
        print("SMOKE VERIFIED", args.dataset, variant, flush=True)
