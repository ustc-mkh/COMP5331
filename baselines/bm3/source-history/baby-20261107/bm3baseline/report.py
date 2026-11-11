import csv
import json
from pathlib import Path
import statistics

from .config import write_json

TARGET = {"recall@10": 0.0556, "recall@20": 0.0865, "ndcg@10": 0.0293, "ndcg@20": 0.0372}


def aggregate(directory, output):
    paths = sorted(Path(directory).rglob("result.json"))
    runs = [(path, json.loads(path.read_text(encoding="utf-8"))) for path in paths]
    runs = [(path, run) for path, run in runs if run["kind"] == "formal" and run.get("test") is not None]
    if not runs:
        raise ValueError("No completed formal test runs; smoke/sweep-only results are excluded")
    seeds, identity = set(), None
    for path, run in runs:
        config = {k: v for k, v in run["config"].items() if k != "seed"}
        current = (json.dumps(config, sort_keys=True), run["data_fingerprint"], run["source_commit"])
        if identity is not None and current != identity:
            raise ValueError("Cannot combine different configurations/data/versions: " + str(path))
        if run["seed"] in seeds:
            raise ValueError("Duplicate seed; choose a directory containing one run per seed")
        identity = current
        seeds.add(run["seed"])
    rows = []
    for metric, target in TARGET.items():
        values = [run["test"][metric] for _, run in runs]
        mean = statistics.mean(values)
        rows.append({"metric": metric, "n": len(values), "mean": mean,
                     "sample_std": statistics.stdev(values) if len(values) > 1 else None,
                     "paper_target": target, "relative_gap_percent": 100 * (mean - target) / target})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "summary.json", {"seeds": sorted(seeds), "metrics": rows,
                                         "run_files": [str(path.resolve()) for path, _ in runs],
                                         "paper": "10.1145/3770854.3780208, Table 2, Baby/BM3/raw"})
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def plot_history(run_dir, output=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    run_dir = Path(run_dir)
    with (run_dir / "history.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    epochs = [int(row["epoch"]) for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.2), constrained_layout=True)
    axes[0].plot(epochs, [float(row["train_loss_mean"]) for row in rows])
    axes[0].set(xlabel="Epoch", ylabel="Mean batch loss", title="BM3 training")
    axes[1].plot(epochs, [float(row["valid_recall@20"]) for row in rows], label="Validation Recall@20")
    axes[1].axvline(result["best_epoch"], color="tab:orange", linestyle="--", label="Selected epoch")
    axes[1].set(xlabel="Epoch", ylabel="Recall@20")
    axes[1].legend()
    if result["kind"] != "formal":
        figure.suptitle("SMOKE TEST - NOT A REPRODUCTION RESULT")
    output = Path(output) if output else run_dir / "learning_curve.png"
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output
