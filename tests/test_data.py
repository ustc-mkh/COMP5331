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


def test_batched_sampler_dense_complements_and_uniformity(tiny_data):
    # Exercise one legal item, empty history, and holes at both ends of a row.
    history = sp.csr_matrix([[1, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0],
                             [0, 1, 0, 1, 0, 0], [1, 0, 0, 0, 0, 1]])
    train = replace(tiny_data[0], history=history,
                    edges=np.array([[0, 0], [1, 1], [2, 1], [3, 0]]))
    sampler = TrainingSampler(train, BatchSpec("pairwise", 12000), 4, 17)
    batch = next(sampler.batches(1))
    for user, negatives in zip(batch.users.numpy(), batch.negative_items.numpy()):
        legal = np.flatnonzero(history[user].toarray().ravel() == 0)
        counts = np.bincount(negatives, minlength=6)
        assert counts.sum() == 12000
        assert set(np.flatnonzero(counts)) == set(legal)
        expected = 12000 / len(legal)
        assert np.max(np.abs(counts[legal] - expected)) < expected * .12


def test_sampler_legacy_checkpoint_preserves_sequence(tiny_data):
    train = tiny_data[0]
    sampler = TrainingSampler(train, BatchSpec("pairwise", 3), 4, 17)
    sampler.load_state_dict({"generator": torch.Generator().get_state()})
    generator = torch.Generator().manual_seed(17 + 2 * 1_000_003)
    order = torch.randperm(len(train.edges), generator=generator).numpy()
    expected = []
    for user in train.edges[order, 0]:
        positives = sampler.positives[user]
        ranks = torch.randint(train.n_items - len(positives), (3,), generator=generator).numpy()
        expected.append(ranks + np.searchsorted(positives - np.arange(len(positives)), ranks, side="right"))
    actual = torch.cat([batch.negative_items for batch in sampler.batches(2)]).numpy()
    np.testing.assert_array_equal(actual, expected)
    assert sampler.state_dict()["algorithm_version"] == 1
    fresh = TrainingSampler(train, BatchSpec("pairwise"), 4, 17)
    assert fresh.state_dict()["algorithm_version"] == 2


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda:1', marks=pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason='CUDA device 1 unavailable'))])
def test_resident_sampler_complements_reproducibility_and_restore(tiny_data, device):
    from mmrecsys.data.sampling import DeviceTrainingSampler
    history = sp.csr_matrix([[1, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0],
                             [0, 1, 0, 1, 0, 0], [1, 0, 0, 0, 0, 1]])
    train = replace(tiny_data[0], history=history,
                    edges=np.array([[0, 0], [1, 1], [2, 1], [3, 0]]))
    sampler = DeviceTrainingSampler(train, BatchSpec('pairwise', 12000), 3, 17, device)
    batches = list(sampler.batches(2))
    assert [len(b.users) for b in batches] == [3, 1]
    for batch in batches:
        assert batch.users.device == torch.device(device)
        for user, negatives in zip(batch.users.cpu().numpy(), batch.negative_items.cpu().numpy()):
            legal = np.flatnonzero(history[user].toarray().ravel() == 0)
            counts = np.bincount(negatives, minlength=6)
            assert set(np.flatnonzero(counts)) == set(legal)
            expected = 12000 / len(legal)
            assert np.max(np.abs(counts[legal] - expected)) < expected * .12
    for left, right in zip(batches, sampler.batches(2)):
        assert torch.equal(left.users, right.users)
        assert torch.equal(left.negative_items, right.negative_items)
    restored = DeviceTrainingSampler(train, sampler.spec, 3, 17, device)
    restored.load_state_dict(sampler.state_dict())
    for left, right in zip(sampler.batches(3), restored.batches(3)):
        assert torch.equal(left.negative_items, right.negative_items)
    with pytest.raises(ValueError, match='same sampling algorithm'):
        restored.load_state_dict({'algorithm_version': 2})
    positive = DeviceTrainingSampler(train, BatchSpec('positive'), 3, 17, device)
    assert all(b.negative_items is None for b in positive.batches(1))
    empty_history = replace(train, history=sp.csr_matrix((4, 6)))
    empty = DeviceTrainingSampler(empty_history, BatchSpec('pairwise'), 3, 17, device)
    assert all(((b.negative_items >= 0) & (b.negative_items < 6)).all() for b in empty.batches(1))
