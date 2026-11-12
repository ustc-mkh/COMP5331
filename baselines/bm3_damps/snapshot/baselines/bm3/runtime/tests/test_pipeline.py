import json
import math
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from bm3baseline import ROOT
from bm3baseline.config import load_config, configure_process, write_json
from bm3baseline.data import inspect_data, runtime_config, Interactions
from bm3baseline.model import BM3, OriginalBM3, make_loaders, evaluate
from bm3baseline.engine import train, load_checkpoint, evaluate_checkpoint
from bm3baseline.report import aggregate


@pytest.fixture
def fixture_data(tmp_path):
    directory = tmp_path / "baby"
    directory.mkdir()
    rows = []
    for user in range(32):
        for offset in range(9):
            label = 0 if offset < 5 else (1 if offset < 7 else 2)
            rows.append((user, (user + offset) % 32, label))
    pd.DataFrame(rows, columns=["userID", "itemID", "x_label"]).to_csv(directory / "baby.inter", sep="\t", index=False)
    rng = np.random.RandomState(17)
    np.save(directory / "image_feat.npy", rng.randn(32, 8).astype(np.float64))
    np.save(directory / "text_feat.npy", rng.randn(32, 6).astype(np.float32))
    return directory


@pytest.fixture
def small_config():
    return load_config(overrides={"embedding_size": 4, "feat_embed_dim": 4,
                                  "train_batch_size": 16, "eval_batch_size": 16,
                                  "epochs": 3, "threads": 1, "checkpoint_every": 1})


def construct(config, directory, output):
    _, splits = inspect_data(directory, require_baby=False)
    runtime = runtime_config(config, directory, torch.device("cpu"), output)
    configure_process(config["seed"], "cpu", config["threads"])
    loaders = make_loaders(runtime, splits)
    loaders[0].pretrain_setup()
    return BM3(runtime, loaders[0]), loaders, runtime


def test_upstream_integrity_and_core_method_identity():
    subprocess.run([sys.executable, str(ROOT / "scripts" / "fetch_upstream.py"), "--verify"], check=True)
    assert BM3.forward is OriginalBM3.forward
    assert BM3.calculate_loss is OriginalBM3.calculate_loss
    assert BM3.full_sort_predict is OriginalBM3.full_sort_predict


def test_binary_graph_normalization_and_no_heldout_edges(fixture_data, small_config, tmp_path):
    model, loaders, _ = construct(small_config, fixture_data, tmp_path)
    graph = model.norm_adj.to_dense().numpy()
    train_pairs = loaders[0].dataset.df
    reference = np.zeros((64, 64), dtype=np.float32)
    for user, item in train_pairs.itertuples(index=False, name=None):
        reference[user, item + 32] = reference[item + 32, user] = 1
    degree = (reference > 0).sum(axis=1) + 1e-7
    expected = reference / np.sqrt(degree[:, None] * degree[None, :])
    np.testing.assert_allclose(graph, expected, rtol=1e-6, atol=1e-7)
    assert graph[0, 32 + 5] == 0  # validation edge
    assert graph[0, 32 + 7] == 0  # test edge


def test_original_loss_backpropagates_to_both_modalities(fixture_data, small_config, tmp_path):
    model, loaders, _ = construct(small_config, fixture_data, tmp_path)
    batch = next(iter(loaders[0]))
    assert batch.shape[0] == 2  # no negative sampling
    loss = model.calculate_loss(batch)
    loss.backward()
    for name in ("user_embedding.weight", "item_id_embedding.weight", "image_embedding.weight",
                 "text_embedding.weight", "predictor.weight"):
        gradient = dict(model.named_parameters())[name].grad
        assert gradient is not None and torch.isfinite(gradient).all() and torch.any(gradient != 0)
    assert not any("damps" in name.lower() for name, _ in model.named_modules())


def test_known_recall_and_ndcg():
    from utils.metrics import recall_, ndcg_
    hits = np.array([[True, False, True], [False, True, False]])
    positives = np.array([2, 1])
    assert recall_(hits, positives)[0] == 0.25
    assert recall_(hits, positives)[2] == 1.0
    ideal_two = 1 + 1 / math.log2(3)
    expected = ((1 + 1 / math.log2(4)) / ideal_two + 1 / math.log2(3)) / 2
    assert ndcg_(hits, positives)[2] == pytest.approx(expected)


def test_masking_preserves_item_zero_and_validation_items_at_test(small_config, tmp_path):
    splits = [Interactions(pd.DataFrame({"userID": [0], "itemID": [item]}), 1, 32) for item in (1, 0, 2)]
    runtime = runtime_config(small_config, tmp_path / "baby", torch.device("cpu"), tmp_path)
    runtime["topk"] = [1, 20]
    _, valid, test = make_loaders(runtime, splits)

    class Scores(torch.nn.Module):
        def full_sort_predict(self, batch):
            scores = torch.zeros(len(batch[0]), 32)
            scores[:, 1] = 1000  # training item must be excluded
            scores[:, 0] = 100   # valid item is a real item, not padding
            scores[:, 2] = 99
            return scores

    assert evaluate(Scores(), valid, runtime)["recall@1"] == 1.0
    assert evaluate(Scores(), test, runtime)["recall@1"] == 0.0
    assert evaluate(Scores(), test, runtime)["recall@20"] == 1.0


@pytest.mark.parametrize("problem", ["duplicate", "bad_label", "nan_feature", "wrong_rows"])
def test_data_rejects_invalid_inputs(fixture_data, problem):
    if problem in {"duplicate", "bad_label"}:
        df = pd.read_csv(fixture_data / "baby.inter", sep="\t")
        if problem == "duplicate":
            df = pd.concat([df, df.iloc[[0]].assign(x_label=2)])
        else:
            df.loc[0, "x_label"] = 3
        df.to_csv(fixture_data / "baby.inter", sep="\t", index=False)
    else:
        features = np.load(fixture_data / "image_feat.npy")
        if problem == "nan_feature":
            features[0, 0] = np.nan
        else:
            features = features[:-1]
        np.save(fixture_data / "image_feat.npy", features)
    with pytest.raises(ValueError):
        inspect_data(fixture_data, require_baby=False)


def test_checkpoint_uses_best_validation_not_last_epoch(fixture_data, small_config, tmp_path, monkeypatch):
    import bm3baseline.engine as engine
    calls = []
    scores = iter([0.5, 0.1, 0.2, 0.9])

    def fake_evaluate(model, loader, config, save_topk=None):
        calls.append(save_topk is not None)
        value = next(scores)
        return {key: value for key in ("recall@10", "recall@20", "ndcg@10", "ndcg@20")}

    monkeypatch.setattr(engine, "evaluate", fake_evaluate)
    result = train(small_config, fixture_data, tmp_path / "selection", require_baby=False)
    assert calls == [False, False, False, True]
    assert result["best_epoch"] == 1
    best = load_checkpoint(tmp_path / "selection" / "best.pt")
    last = load_checkpoint(tmp_path / "selection" / "last.pt")
    assert best["epoch"] == 1 and last["epoch"] == 3
    assert not torch.equal(best["model_state_dict"]["user_embedding.weight"],
                           last["model_state_dict"]["user_embedding.weight"])


def test_resume_matches_uninterrupted_training(fixture_data, small_config, tmp_path):
    full = tmp_path / "full"
    split = tmp_path / "resumed"
    expected = train(small_config, fixture_data, full, require_baby=False, test_at_end=False)
    train(dict(small_config, epochs=1), fixture_data, split, require_baby=False, test_at_end=False)
    actual = train(small_config, fixture_data, split, require_baby=False, test_at_end=False, resume=split / "last.pt")
    full_state = load_checkpoint(full / "last.pt")
    split_state = load_checkpoint(split / "last.pt")
    for name in full_state["model_state_dict"]:
        torch.testing.assert_close(full_state["model_state_dict"][name], split_state["model_state_dict"][name], rtol=0, atol=0)
    assert expected["best_validation"] == actual["best_validation"]
    assert [row["train_loss_sum"] for row in full_state["history"]] == [row["train_loss_sum"] for row in split_state["history"]]


def test_checkpoint_reload_and_data_identity(fixture_data, small_config, tmp_path):
    directory = tmp_path / "reload"
    result = train(dict(small_config, epochs=1), fixture_data, directory, require_baby=False)
    reloaded = evaluate_checkpoint(directory / "best.pt", fixture_data, require_baby=False)
    assert result["test"] == reloaded["test"]
    values = np.load(fixture_data / "text_feat.npy")
    values[0, 0] += 1
    np.save(fixture_data / "text_feat.npy", values)
    with pytest.raises(ValueError, match="data differ"):
        evaluate_checkpoint(directory / "best.pt", fixture_data, require_baby=False)


def test_aggregation_excludes_smoke_and_rejects_duplicate_seeds(tmp_path, small_config):
    from bm3baseline.report import TARGET
    run = {"kind": "smoke", "config": small_config, "seed": 999, "test": TARGET,
           "data_fingerprint": "fixture", "source_commit": "fixture"}
    write_json(tmp_path / "runs" / "smoke" / "result.json", run)
    with pytest.raises(ValueError, match="No completed formal"):
        aggregate(tmp_path / "runs", tmp_path / "summary")
    run["kind"] = "formal"
    write_json(tmp_path / "runs" / "seed999" / "result.json", run)
    assert aggregate(tmp_path / "runs", tmp_path / "summary")[0]["n"] == 1
    write_json(tmp_path / "runs" / "duplicate" / "result.json", run)
    with pytest.raises(ValueError, match="Duplicate seed"):
        aggregate(tmp_path / "runs", tmp_path / "summary")


def test_config_rejects_negative_sampling():
    with pytest.raises(ValueError, match="positive pairs"):
        load_config(overrides={"use_neg_sampling": True})


def test_checkpoint_write_failure_preserves_previous_file(tmp_path, monkeypatch):
    import bm3baseline.engine as engine
    target = tmp_path / "last.pt"
    target.write_bytes(b"previous checkpoint")

    def fail_save(state, path):
        path.write_bytes(b"incomplete replacement")
        raise OSError("simulated disk write failure")

    monkeypatch.setattr(engine.torch, "save", fail_save)
    with pytest.raises(OSError, match="simulated disk"):
        engine.save_checkpoint(target, {"weight": torch.zeros(2)})
    assert target.read_bytes() == b"previous checkpoint"
    assert not (tmp_path / "last.pt.tmp").exists()


def test_low_disk_space_fails_before_checkpoint_write(tmp_path, monkeypatch):
    import bm3baseline.engine as engine
    from types import SimpleNamespace
    monkeypatch.setattr(engine.shutil, "disk_usage", lambda path: SimpleNamespace(free=1))
    with pytest.raises(RuntimeError, match="Insufficient disk space"):
        engine.save_checkpoint(tmp_path / "last.pt", {"weight": torch.zeros(2)})
    assert not list(tmp_path.glob("*.pt*"))


def test_shell_scripts_use_lf():
    for path in (ROOT / "scripts").glob("*.sh"):
        assert b"\r\n" not in path.read_bytes(), path
