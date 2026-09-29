from copy import deepcopy
import json
import os

import pytest

from mmrecsys.experiment.compare import build_jobs, devices_for, main, run_jobs, summarize
from mmrecsys.experiment.ablation import variant_config


def test_paired_configs_and_budgets():
    jobs = build_jobs(["baby", "sports", "clothing", "elec"], [999, 2024], "quick", ["train.epochs=3"])
    assert len(jobs) == 16
    for baseline, full in zip(jobs[::2], jobs[1::2]):
        b, d = deepcopy(baseline["config"]), deepcopy(full["config"])
        assert b["train"]["epochs"] == d["train"]["epochs"] == 3
        assert not b["model"].pop("damps_enabled")
        assert d["model"].pop("damps_enabled")
        assert b == d
    assert build_jobs(["baby"], [999], "full", [])[0]["config"]["train"]["epochs"] == 1000
    with pytest.raises(ValueError):
        build_jobs(["baby"], [999, 999], "quick", [])


def test_summary_pairs_zero_baseline_and_negative_gain():
    jobs = []
    for seed, baseline, full in [(1, .2, .3), (2, .4, .2), (3, 0, 0)]:
        for variant, value in [("baseline", baseline), ("full", full)]:
            jobs.append({"dataset": "baby", "seed": seed, "variant": variant,
                         "status": "completed", "result": {"test": {"Recall@20": value, "NDCG@20": 0, "n_users": 5}}})
    jobs.append({"dataset": "baby", "seed": 4, "variant": "baseline", "status": "completed",
                 "result": {"test": {"Recall@20": 1, "NDCG@20": 1}}})
    recall, ndcg = summarize(jobs)
    assert recall["n_pairs"] == 3
    assert recall["relative_gain_pct"] == pytest.approx(-100 / 6)
    assert recall["delta_mean"] == pytest.approx(-.1 / 3)
    assert recall["paired_gain_pct_mean"] == pytest.approx(0)
    assert recall["paired_gain_defined_count"] == 2
    assert ndcg["relative_gain_pct"] is None
    assert ndcg["paired_gain_pct_mean"] is None


def test_device_validation(monkeypatch):
    monkeypatch.setattr("torch.cuda.device_count", lambda: 2)
    assert devices_for(["auto"]) == ["cuda:0", "cuda:1"]
    assert devices_for(["1", "0"]) == ["cuda:1", "cuda:0"]
    assert devices_for(["cpu"]) == ["cpu"]
    for values in (["2"], ["0", "0"], ["cpu", "0"]):
        with pytest.raises(ValueError):
            devices_for(values)


def test_subprocess_training_reports_and_resume(tiny_config, tmp_path):
    devices = devices_for(os.environ.get("MMRECSYS_TEST_GPUS", "cpu").split())
    jobs = [{"id": variant, "dataset": "baby", "seed": 999, "variant": variant,
             "config": variant_config(tiny_config, variant, 999), "status": "pending"}
            for variant in ("baseline", "full")]
    output = tmp_path / "comparison"
    assert run_jobs(jobs, devices, output, "quick")
    summary = json.loads((output / "summary.json").read_text())
    assert summary["complete"]
    assert len(summary["metrics"]) == 6
    assert all(row["n_pairs"] == 1 for row in summary["metrics"])
    assert (output / "summary.csv").exists()
    original = deepcopy(jobs)
    assert main(["--resume", str(output), "--gpus", "cpu"]) == 0
    assert json.loads((output / "runs.json").read_text())["jobs"] == original
    for job in jobs:
        assert "epoch=2" in open(job["log"]).read()
        assert job["result"]["test"]["n_users"] == 4


def test_failed_child_preserves_other_results(tiny_config, tmp_path):
    jobs = [{"id": variant, "dataset": "baby", "seed": 999, "variant": variant,
             "config": variant_config(tiny_config, variant, 999), "status": "pending"}
            for variant in ("baseline", "full")]
    jobs[0]["config"]["data"]["interactions"] = "missing.inter"
    output = tmp_path / "comparison"
    assert not run_jobs(jobs, ["cpu"], output, "quick")
    assert [job["status"] for job in jobs] == ["failed", "completed"]
    assert summarize(jobs) == []
    assert not json.loads((output / "summary.json").read_text())["complete"]
