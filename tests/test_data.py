from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from mmrecsys.data.dataset import load_dataset
from mmrecsys.data.sampling import BatchSpec, TrainingSampler


def test_training_view_and_eval_history(tiny_data, tiny_config):
    train, views, metadata = tiny_data
    assert train.n_users == 4 and train.n_items == 6
    assert len(train.edges) == 6
    assert not hasattr(train, "targets")
    assert train.history[0, 2] == 0  # Validation edge must never enter training.
    assert views["test"].history[0, 2] == 0
    tiny_config["eval"]["history"] = "train_valid"
    _, changed, _ = load_dataset(tiny_config["data"], ("image", "text"), tiny_config["eval"])
    assert changed["test"].history[0, 2] == 1
    assert changed["valid"].history[0, 2] == 0


@pytest.mark.parametrize("corruption", ["duplicate", "label", "negative", "gap", "nan", "rows", "field"])
def test_invalid_published_data_fails(tiny_config, corruption):
    root = Path(tiny_config["data"]["root"]) / "baby"
    path = root / "baby.inter"
    text = path.read_text()
    if corruption == "duplicate":
        path.write_text(text + "0\t0\t2\n")
    elif corruption == "label":
        path.write_text(text.replace("0\t0\t0", "0\t0\t9"))
    elif corruption == "negative":
        path.write_text(text.replace("0\t0\t0", "-1\t0\t0"))
    elif corruption == "gap":
        path.write_text(text.replace("3\t4\t0", "9\t4\t0"))
    elif corruption == "field":
        path.write_text(text.replace("userID", "userTypo"))
    else:
        values = np.ones((5 if corruption == "rows" else 6, 3), dtype=np.float32)
        if corruption == "nan":
            values[0, 0] = np.nan
        np.save(root / "image_feat.npy", values)
    with pytest.raises(ValueError):
        load_dataset(tiny_config["data"], ("image", "text"), tiny_config["eval"])


def test_sampling_reproducible_and_training_only(tiny_data):
    train = tiny_data[0]
    sampler = TrainingSampler(train, BatchSpec("pairwise", 50), 4, 17)
    first = list(sampler.batches(2))
    second = list(sampler.batches(2))
    for left, right in zip(first, second):
        assert torch.equal(left.users, right.users)
        assert torch.equal(left.negative_items, right.negative_items)
        for user, negatives in zip(left.users, left.negative_items):
            assert not train.history[int(user), negatives.numpy()].toarray().any()
    # A held-out positive is still a legal training negative.
    assert any(2 in batch.negative_items[row].tolist()
               for batch in first for row, user in enumerate(batch.users) if user == 0)
    full = replace(train, history=sp.csr_matrix(np.ones((4, 6))))
    with pytest.raises(ValueError, match="every item"):
        TrainingSampler(full, BatchSpec("pairwise"), 4, 17)
