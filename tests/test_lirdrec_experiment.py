from copy import deepcopy
from dataclasses import asdict
import json

import pytest
import torch

from mmrecsys.experiment.runner import evaluate_experiment, train_experiment
from mmrecsys.models.lirdrec import LIRDRecConfig


@pytest.mark.parametrize("damps_enabled", [False, True])
def test_tiny_lirdrec_train_evaluate_and_exact_resume(tiny_config, damps_enabled):
    config = deepcopy(tiny_config)
    config["model"] = asdict(LIRDRecConfig(embedding_dim=4, knn_k=2, knn_chunk_size=2,
                                           damps_enabled=damps_enabled))
    config["train"]["epochs"] = 4
    full_run, full_result = train_experiment(config)
    partial = deepcopy(config)
    partial["train"]["epochs"] = 2
    resumed_run, _ = train_experiment(partial)
    _, resumed_result = train_experiment(config, resumed_run / "last.pt")
    assert resumed_result == full_result
    assert evaluate_experiment(resumed_run, device="cpu") == resumed_result["test"]
    assert resumed_result["test"]["n_users"] == 4
    assert all(0 <= metric <= 1 for name, metric in resumed_result["test"].items()
               if name.startswith(("Recall@", "NDCG@")))

    full = torch.load(full_run / "last.pt", weights_only=False)
    resumed = torch.load(resumed_run / "last.pt", weights_only=False)
    assert full["state"] == resumed["state"]
    assert full["state"]["epoch"] == 4
    assert full["scheduler"] == resumed["scheduler"]
    for name, expected in full["model"].items():
        torch.testing.assert_close(resumed["model"][name], expected, rtol=0, atol=0)
    assert torch.equal(full["sampler"]["generator"], resumed["sampler"]["generator"])
    full_logs = [json.loads(row) for row in (full_run / "metrics.jsonl").read_text().splitlines()]
    resumed_logs = [json.loads(row) for row in (resumed_run / "metrics.jsonl").read_text().splitlines()]
    assert [row["loss"] for row in resumed_logs] == [row["loss"] for row in full_logs]
    assert [row.get("valid") for row in resumed_logs] == [row.get("valid") for row in full_logs]
