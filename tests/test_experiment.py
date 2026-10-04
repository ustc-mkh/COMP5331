from copy import deepcopy
import random
import json

import numpy as np
import pytest
import torch

from mmrecsys.config import PROJECT_ROOT, load_config
from mmrecsys.engine.checkpoint import load_checkpoint
from mmrecsys.experiment.runner import assemble, evaluate_experiment, train_experiment
from mmrecsys.experiment.seed import random_state, restore_random_state


def test_config_precedence_unknown_fields_and_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = load_config("configs/experiments/mgcn_baby.yaml", overrides=["model.cl_weight=0.1"])
    assert config["model"]["fusion"] == "author" and config["model"]["cl_weight"] == 0.1
    assert config["data"]["root"] == str((PROJECT_ROOT / "data").resolve())
    for expression in ("model.cl_weigth=1", "train.bach_size=1", "eval.topk=[]", "train.epochs=-1", "train=false"):
        with pytest.raises(ValueError):
            load_config(overrides=[expression])


def test_rng_state_restore():
    state = random_state()
    first = (random.random(), np.random.rand(), torch.rand(3))
    restore_random_state(state)
    second = (random.random(), np.random.rand(), torch.rand(3))
    assert first[:2] == second[:2]
    assert torch.equal(first[2], second[2])


@pytest.mark.parametrize("damps_enabled", [False, True])
@pytest.mark.parametrize("eval_every", [1, 3])
def test_train_evaluate_and_exact_epoch_resume(tiny_config, eval_every, damps_enabled):
    tiny_config["model"]["damps_enabled"] = damps_enabled
    tiny_config["train"]["eval_every"] = eval_every
    tiny_config["train"]["epochs"] = 4
    full_run, full_result = train_experiment(tiny_config)
    partial = deepcopy(tiny_config)
    partial["train"]["epochs"] = 2
    resumed_run, _ = train_experiment(partial)
    _, resumed_result = train_experiment(tiny_config, resumed_run / "last.pt")
    assert resumed_result == full_result
    logs = [json.loads(line) for line in (full_run / "metrics.jsonl").read_text().splitlines()]
    assert all("gate/image/saturated_fraction" in row["diagnostics"] for row in logs)
    if damps_enabled:
        assert all("gradient/damps/phase_residual/l2" in row["diagnostics"] for row in logs)
    resumed_logs = [json.loads(line) for line in (resumed_run / "metrics.jsonl").read_text().splitlines()]
    assert [row["diagnostics"] for row in logs] == [row["diagnostics"] for row in resumed_logs]
    full = torch.load(full_run / "last.pt", weights_only=False)
    resumed = torch.load(resumed_run / "last.pt", weights_only=False)
    for key in full["model"]:
        torch.testing.assert_close(full["model"][key], resumed["model"][key], rtol=0, atol=0)
    assert full["state"] == resumed["state"]
    assert full["scheduler"] == resumed["scheduler"]
    assert torch.equal(full["sampler"]["generator"], resumed["sampler"]["generator"])
    for name in ("config.yaml", "manifest.json", "metrics.jsonl", "best.pt", "last.pt", "result.json"):
        assert (resumed_run / name).exists()
    assert evaluate_experiment(resumed_run) == resumed_result["test"]
    _, _, metadata, model, _, _ = assemble(tiny_config)
    with pytest.raises(ValueError, match="fingerprint"):
        load_checkpoint(full_run / "best.pt", model, tiny_config, "changed-data")
    changed = deepcopy(tiny_config)
    changed["model"]["temperature"] = 0.1
    with pytest.raises(ValueError, match="incompatible"):
        load_checkpoint(full_run / "best.pt", model, changed, metadata["fingerprint"])


def test_ablation_suite(tiny_config):
    from mmrecsys.experiment.ablation import run_ablation, VARIANTS
    output = run_ablation(tiny_config, [999], list(VARIANTS))
    records = json.loads((output / "runs.json").read_text())
    summary = json.loads((output / "summary.json").read_text())
    assert len(records) == 5
    assert set(summary["metrics"]) == set(VARIANTS)
    for record in records:
        assert record["seed"] == 999
        assert summary["metrics"][record["variant"]]["Recall@3"]["mean"] == record["result"]["test"]["Recall@3"]
        assert summary["metrics"][record["variant"]]["Recall@3"]["sample_std"] is None


def test_resident_setting_legacy_config_compatibility(tiny_config):
    from mmrecsys.config import validate
    from mmrecsys.engine.checkpoint import compatible_config
    old = deepcopy(tiny_config)
    old['train'].pop('preload_to_device', None)
    normalized = validate(old)
    assert normalized['train']['preload_to_device'] is False
    assert compatible_config(old) == compatible_config(normalized)
    assert load_config()['train']['preload_to_device'] is True
    invalid = deepcopy(tiny_config)
    invalid['train']['preload_to_device'] = 'yes'
    with pytest.raises(ValueError, match='preload_to_device'):
        validate(invalid)
