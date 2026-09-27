from copy import deepcopy
import random

import numpy as np
import pytest
import torch

from mmrecsys.config import load_config
from mmrecsys.engine.checkpoint import load_checkpoint
from mmrecsys.experiment.runner import assemble, evaluate_experiment, train_experiment
from mmrecsys.experiment.seed import random_state, restore_random_state


def test_config_precedence_unknown_fields_and_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = load_config("configs/experiments/mgcn_baby_author.yaml", overrides=["model.cl_weight=0.1"])
    assert config["model"]["fusion"] == "author" and config["model"]["cl_weight"] == 0.1
    assert config["data"]["root"].endswith("COMP5331/data")
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


@pytest.mark.parametrize("eval_every", [1, 3])
def test_train_evaluate_and_exact_epoch_resume(tiny_config, eval_every):
    tiny_config["train"]["eval_every"] = eval_every
    tiny_config["train"]["epochs"] = 4
    full_run, full_result = train_experiment(tiny_config)
    partial = deepcopy(tiny_config)
    partial["train"]["epochs"] = 2
    resumed_run, _ = train_experiment(partial)
    _, resumed_result = train_experiment(tiny_config, resumed_run / "last.pt")
    assert resumed_result == full_result
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
