"""Export and independently check complete, portable BM3+DAMPS evidence."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ("baby", "sports", "clothing", "elec", "microlens")
SEEDS = (999, 2024, 2025)
VARIANTS = ("baseline", "full")
METRICS = ("recall@10", "recall@20", "ndcg@10", "ndcg@20")
EXTRA_CONFIG = {"damps_variant", "integration", "extension_digest"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def csv_save(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compare_rows(actual, expected):
    """Allow last-bit statistics.sqrt differences between Python versions."""
    assert len(actual) == len(expected)
    for observed, reference in zip(actual, expected):
        assert set(observed) == set(reference)
        for key, value in reference.items():
            if isinstance(value, float):
                assert math.isclose(float(observed[key]), value, rel_tol=1e-12, abs_tol=1e-15), (key, observed[key], value)
            else:
                assert str(observed[key]) == str(value), (key, observed[key], value)


def check_history(path, run):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert [int(r["epoch"]) for r in rows] == list(range(1, run["epochs_completed"] + 1))
    best = max(rows, key=lambda r: float(r["valid_recall@20"]))
    assert int(best["epoch"]) == run["best_epoch"]
    for metric in METRICS:
        assert float(best["valid_" + metric]) == run["best_validation"][metric]
    assert run["epochs_completed"] == min(run["best_epoch"] + 21, run["config"]["epochs"])
    for row in rows:
        assert math.isfinite(float(row["train_loss_mean"]))


def derive(runs, variants=VARIANTS):
    assert tuple(variants) == VARIANTS, "Only the requested baseline/full comparison is publishable"
    index = {(r["dataset"], r["variant"], r["seed"]): r for r in runs}
    assert len(index) == len(runs) == len(DATASETS)*len(variants)*len(SEEDS)
    assert set(index) == {(d, v, s) for d in DATASETS for v in variants for s in SEEDS}
    summary, contrasts = [], []
    for dataset in DATASETS:
        for variant in variants:
            for metric in METRICS:
                values = [index[dataset, variant, s]["test"][metric] for s in SEEDS]
                summary.append(dict(dataset=dataset, variant=variant, metric=metric,
                                    n_runs=len(SEEDS), mean=mean(values), sample_std=stdev(values)))
        # Match seeds before computing each contrast; retain variation in paired deltas.
        for comparison, control in (("full", "baseline"), ("no_apc", "full"),
                                    ("no_avrf", "full"), ("no_imcf", "full")):
            if comparison not in variants:
                continue
            for metric in METRICS:
                a = [index[dataset, comparison, s]["test"][metric] for s in SEEDS]
                b = [index[dataset, control, s]["test"][metric] for s in SEEDS]
                deltas = [x-y for x,y in zip(a,b)]
                contrasts.append(dict(dataset=dataset, comparison=comparison, control=control,
                                      metric=metric, n_pairs=len(SEEDS), paired_mean_delta=mean(deltas),
                                      paired_delta_sample_std=stdev(deltas),
                                      relative_mean_delta_pct=100*(mean(a)/mean(b)-1),
                                      positive_pairs=sum(x > 0 for x in deltas)))
    return summary, contrasts


def export(suite, output, stage="main"):
    from run_suite import digest
    plan = read(suite / "plan.json")
    assert stage == "main", "Ablation experiments were cancelled by the user"
    assert plan["training_runs"] == 15 and plan["digest"] == digest()
    assert plan["seeds"] == list(SEEDS) and plan["variants"] == ["full"]
    variants = VARIANTS
    jobs = plan["jobs"]
    expected = 15
    assert len(jobs) == expected
    assert all(j["config"]["damps_variant"] == "full" and j["config"]["seed"] in SEEDS for j in jobs)
    assert all((suite/j["id"]/"verified.json").is_file() for j in jobs), "Requested experiment group incomplete"
    raw = [r for r in read(ROOT / "docs/results/bm3/runs.json")["runs"] if r["seed"] in SEEDS]
    assert len(raw) == 15
    baseline = {(r["dataset"], r["seed"]): r for r in raw}
    output.mkdir(parents=True, exist_ok=True)
    (output / "histories").mkdir(exist_ok=True)
    runs = [dict(r, history="../bm3/"+r["history"]) for r in raw]
    artifacts = {}
    for job in jobs:
        directory = suite/job["id"]/"run"
        verified = read(suite/job["id"]/"verified.json")
        result, reloaded = verified["result"], verified["reloaded"]
        assert result == read(directory/"result.json") and reloaded == read(directory/"reloaded.json")
        assert result["kind"] == reloaded["kind"] == "formal" and result["test_evaluated"]
        assert result["config"] == job["config"]
        config = result["config"]
        old = baseline[config["dataset"], config["seed"]]
        assert {k:v for k,v in config.items() if k not in EXTRA_CONFIG} == old["config"]
        assert result["data_fingerprint"] == reloaded["data_fingerprint"] == old["data_fingerprint"]
        assert result["best_epoch"] == reloaded["epoch"]
        assert result["best_validation"] == reloaded["validation"] and result["test"] == reloaded["test"]
        check_history(directory/"history.csv", result)
        metadata = read(directory/"metadata.json")
        assert metadata["extension_digest"] == plan["digest"]
        assert metadata["integration"] == "projected_trainable_v1"
        history = "histories/" + job["id"] + ".csv"
        shutil.copyfile(directory/"history.csv", output/history)
        portable = dict(result, id=job["id"], dataset=config["dataset"], variant=config["damps_variant"],
                        reloaded={k:v for k,v in reloaded.items() if k != "checkpoint"},
                        history=history, completed=verified["completed"], environment=metadata["environment"])
        runs.append(portable)
        artifacts[job["id"]] = {name:sha(directory/name) for name in
                               ("result.json", "reloaded.json", "metadata.json", "history.csv", "best.pt", "last.pt")}
    summary, contrasts = derive(runs, variants)
    save(output/"runs.json", dict(complete=True, stage=stage, variants=list(variants),
                                 seeds=list(SEEDS), n_runs=len(runs), new_runs=expected, reused_baseline_runs=15, runs=runs))
    save(output/"summary.json", dict(complete=True, seeds=list(SEEDS), metrics=summary))
    save(output/"paired-contrasts.json", dict(direction="comparison minus control", metrics=contrasts))
    csv_save(output/"summary.csv", summary)
    csv_save(output/"paired-contrasts.csv", contrasts)
    files = list((ROOT/"baselines/bm3_damps").rglob("*.py"))
    files += list((ROOT/"baselines/bm3/runtime/bm3baseline").glob("*.py"))
    save(output/"provenance.json", dict(training_digest=plan["digest"], base_commit=plan["source_commit"],
          author_reference_tag=plan["author_reference_tag"], source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)},
          original_artifact_sha256=artifacts))
    portable_plan = dict(plan)
    portable_plan.update(export_stage=stage, exported_training_runs=expected)
    portable_plan["jobs"] = [{k:v for k,v in j.items() if k != "data"} for j in jobs]
    save(output/"training-plan.json", portable_plan)
    (output/".gitattributes").write_text("histories/*.csv -text -whitespace\n", encoding="utf-8")
    index = {(r["dataset"],r["variant"],r["metric"]):r for r in summary}
    with (output/"main-sheet-column.tsv").open("w", encoding="utf-8", newline="") as stream:
        stream.write("BM3+DAMPS\n")
        for dataset in DATASETS:
            for metric in METRICS:
                stream.write(str(index[dataset,"full",metric]["mean"])+"\n")
    # Recompute the original BM3 column over the same requested three seeds.
    with (output/"paired-main-sheet.tsv").open("w",encoding="utf-8",newline="") as stream:
        stream.write("BM3\tBM3+DAMPS\n")
        for dataset in DATASETS:
            for metric in METRICS:
                stream.write("\t".join(str(index[dataset,v,metric]["mean"]) for v in VARIANTS)+"\n")
    verify(output)
    print("EXPORTED", output, flush=True)


def verify(output):
    evidence = read(output/"runs.json")
    assert evidence["stage"] == "main" and evidence["seeds"] == list(SEEDS)
    variants = VARIANTS
    expected = 15
    assert evidence["complete"] and evidence["n_runs"] == 30 and evidence["new_runs"] == expected
    assert evidence["reused_baseline_runs"] == 15
    assert evidence["variants"] == list(variants)
    provenance = read(output/"provenance.json")
    for path, expected_hash in provenance["source_sha256"].items():
        assert sha(ROOT/path) == expected_hash, path
    baseline = {(r["dataset"],r["seed"]):r for r in read(ROOT/"docs/results/bm3/runs.json")["runs"]}
    for run in evidence["runs"]:
        config = run["config"]
        old = baseline[run["dataset"],run["seed"]]
        assert {k:v for k,v in config.items() if k not in EXTRA_CONFIG} == old["config"]
        assert run["test"] == run["reloaded"]["test"] and run["best_validation"] == run["reloaded"]["validation"]
        assert run["best_epoch"] == run["reloaded"]["epoch"]
        assert run["data_fingerprint"] == run["reloaded"]["data_fingerprint"] == old["data_fingerprint"]
        assert set(run["test"]) == set(METRICS) and all(0 <= v <= 1 for v in run["test"].values())
        history = output/run["history"]
        check_history(history, run)
        if run["variant"] == "baseline":
            assert run["test"] == old["test"]
        else:
            assert sha(history) == provenance["original_artifact_sha256"][run["id"]]["history.csv"]
            assert config["integration"] == "projected_trainable_v1"
            assert config["extension_digest"] == provenance["training_digest"]
    summary, contrasts = derive(evidence["runs"], variants)
    compare_rows(read(output/"summary.json")["metrics"], summary)
    compare_rows(read(output/"paired-contrasts.json")["metrics"], contrasts)
    for name, rows in (("summary.csv",summary),("paired-contrasts.csv",contrasts)):
        with (output/name).open(newline="",encoding="utf-8") as stream:
            compare_rows(list(csv.DictReader(stream)), rows)
    assert read(output/"summary.json")["seeds"] == list(SEEDS)
    print("PASS: 15 new runs + 15 paired baselines, seeds 999/2024/2025; reloads, histories, configurations, data, source and statistics")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--stage", choices=("main",), default="main",
                        help="Export only the complete 15-run, three-seed main experiment")
    args = parser.parse_args()
    if args.verify:
        verify(args.output.resolve())
    else:
        assert args.suite is not None
        export(args.suite.resolve(), args.output.resolve(), args.stage)
