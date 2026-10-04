"""Check the portable result bundle using only the Python standard library."""
import csv
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parent
METRICS = ("Recall@10", "Recall@20", "NDCG@10", "NDCG@20")
DATASETS = ("baby", "sports", "clothing", "elec", "microlens")
VARIANTS = ("baseline", "full", "no_apc", "no_avrf", "no_imcf")
SEEDS = (999, 2024, 2025)


def read_json(name):
    return json.loads((ROOT / name).read_text())


def read_csv(name):
    with (ROOT / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def same(actual, expected, label):
    if expected is None:
        assert actual is None, label
    else:
        assert math.isclose(float(actual), expected, rel_tol=1e-12, abs_tol=1e-15), label


def main():
    payload = read_json("runs.json")
    runs = payload["runs"]
    by_run = {(r["dataset"], r["variant"], r["seed"]): r for r in runs}
    expected_runs = {(d, v, s) for d in DATASETS for v in VARIANTS for s in SEEDS}
    assert payload["complete"] and payload["n_runs"] == len(runs) == len(by_run) == 75
    assert set(by_run) == expected_runs
    for (dataset, variant, seed), run in by_run.items():
        assert run["id"] == f"{dataset}-{seed}-{variant}"
        assert 1 <= run["best_epoch"] <= 1000
        assert run["validation"]["metric"] == "Recall@20"
        assert 0 <= run["validation"]["value"] <= 1
        assert set(run["test"]) == {*METRICS, "n_users"}
        assert all(0 <= run["test"][m] <= 1 for m in METRICS)

    summary = read_json("summary.json")
    rows = summary["metrics"]
    by_metric = {(r["dataset"], r["variant"], r["metric"]): r for r in rows}
    assert summary["complete"] and len(rows) == len(by_metric) == 100
    assert set(by_metric) == {(d, v, m) for d in DATASETS for v in VARIANTS for m in METRICS}
    for (dataset, variant, metric), row in by_metric.items():
        values = [by_run[dataset, variant, seed]["test"][metric] for seed in SEEDS]
        baseline = [by_run[dataset, "baseline", seed]["test"][metric] for seed in SEEDS]
        assert row["n_runs"] == 3 and row["seeds"] == list(SEEDS)
        same(row["mean"], mean(values), "mean")
        same(row["sample_std"], stdev(values), "sample std (n-1)")
        if variant == "baseline":
            assert row["n_pairs"] == 0 and row["paired_seeds"] == []
            assert all(row[k] is None for k in ("delta_mean", "delta_sample_std", "relative_gain_pct"))
        else:
            differences = [a - b for a, b in zip(values, baseline)]
            assert row["n_pairs"] == 3 and row["paired_seeds"] == list(SEEDS)
            same(row["delta_mean"], mean(differences), "paired mean")
            same(row["delta_sample_std"], stdev(differences), "paired sample std")
            same(row["relative_gain_pct"], mean(differences) / mean(baseline) * 100, "gain")
    csv_summary = read_csv("summary.csv")
    assert len(csv_summary) == len(rows)
    for row, expected in zip(csv_summary, rows):
        assert set(row) == set(expected)
        assert all(row[k] == ("" if v is None else str(v)) for k, v in expected.items())

    table2, table3 = read_json("paper-table2.json"), read_json("paper-table3.json")
    for paper, page in ((table2, 7), (table3, 8)):
        source = paper["source"]
        assert source["physical_pdf_page_1based"] == page
        pdf = ROOT / source["pdf_path"]
        assert hashlib.sha256(pdf.read_bytes()).hexdigest() == source["pdf_sha256"]
    for dataset in DATASETS:
        assert table2["datasets"][dataset]["full"] == table3["datasets"][dataset]["full"]
    comparisons = read_csv("three-seeds-vs-paper.csv")
    assert len(comparisons) == 100
    assert {(r["dataset"], r["variant"], r["metric"]) for r in comparisons} == set(by_metric)
    for row in comparisons:
        dataset, variant, metric = row["dataset"], row["variant"], row["metric"]
        local = by_metric[dataset, variant, metric]
        paper = (table2 if variant in ("baseline", "full") else table3)["datasets"][dataset][variant][metric]
        assert row["paper_table"] == ("Table 2" if variant in ("baseline", "full") else "Table 3")
        assert int(row["n_runs"]) == 3
        same(row["paper_value"], paper, "paper value")
        same(row["our_mean"], local["mean"], "comparison mean")
        same(row["our_sample_std"], local["sample_std"], "comparison sample std")
        for seed in SEEDS:
            same(row[f"our_seed_{seed}"], by_run[dataset, variant, seed]["test"][metric], "per-seed value")
        same(row["absolute_difference"], local["mean"] - paper, "paper difference")
        same(row["relative_difference_pct"], (local["mean"] / paper - 1) * 100, "paper relative difference")
    gains = read_csv("three-seeds-damps-gains.csv")
    assert len(gains) == 20
    assert {(r["dataset"], r["metric"]) for r in gains} == {(d, m) for d in DATASETS for m in METRICS}
    for row in gains:
        dataset, metric = row["dataset"], row["metric"]
        full, baseline = by_metric[dataset, "full", metric], by_metric[dataset, "baseline", metric]
        pfull, pbase = (table2["datasets"][dataset][v][metric] for v in ("full", "baseline"))
        same(row["paper_full"], pfull, "paper full")
        same(row["paper_baseline"], pbase, "paper baseline")
        same(row["paper_relative_gain_pct"], (pfull / pbase - 1) * 100, "paper gain")
        same(row["our_full_mean"], full["mean"], "full mean")
        same(row["our_baseline_mean"], baseline["mean"], "baseline mean")
        same(row["our_relative_gain_pct"], full["relative_gain_pct"], "local gain")
        same(row["our_paired_difference_mean"], full["delta_mean"], "local paired mean")
        same(row["our_paired_difference_sample_std"], full["delta_sample_std"], "local paired std")
        differences = [by_run[dataset, "full", s]["test"][metric] - by_run[dataset, "baseline", s]["test"][metric] for s in SEEDS]
        assert int(row["n_pairs"]) == 3
        assert int(row["positive_pairs"]) == sum(d > 0 for d in differences)
        assert int(row["negative_pairs"]) == sum(d < 0 for d in differences)

    source = read_json("training-source.json")
    files = {}
    for group in source["datasets"].values():
        actual = hashlib.sha256(json.dumps(group["files"], sort_keys=True).encode()).hexdigest()
        assert actual == group["source_digest"]
        for path, value in group["files"].items():
            assert path not in files or files[path] == value
            files[path] = value
    assert len(files) == source["n_unique_files"]
    data = read_json("data-fingerprints.json")["datasets"]
    assert set(data) == set(DATASETS)
    assert sum(len(v["files"]) for v in data.values()) == 15
    for dataset in DATASETS:
        n_users = data[dataset]["loaded_data"]["n_users"]
        assert all(r["test"]["n_users"] == n_users for r in runs if r["dataset"] == dataset)
    machine_identity = re.compile(r"/(?:home|ssddata)/|" + "GPU" + r"-[0-9a-fA-F]{8}-")
    for path in ROOT.iterdir():
        if path.is_file():
            assert not machine_identity.search(path.read_text()), f"Machine identity in {path.name}"
    print("Verified 75 runs, 100 means/sample standard deviations, 100 paper comparisons, 20 paired gains, source digests and portable paths.")


if __name__ == "__main__":
    main()
