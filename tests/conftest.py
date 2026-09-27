import csv

import numpy as np
import pytest
import torch

from mmrecsys.config import load_config
from mmrecsys.data.dataset import load_dataset


@pytest.fixture(autouse=True)
def limited_threads():
    torch.set_num_threads(1)


@pytest.fixture
def tiny_config(tmp_path):
    root = tmp_path / "data" / "baby"
    root.mkdir(parents=True)
    # All six item IDs occur globally; item 5 is absent from training.
    records = [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 2, 0), (2, 3, 0), (3, 4, 0),
               (0, 2, 1), (1, 3, 1), (2, 4, 1), (3, 5, 1),
               (0, 3, 2), (1, 4, 2), (2, 5, 2), (3, 0, 2)]
    with (root / "baby.inter").open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(["userID", "itemID", "x_label"])
        writer.writerows(records)
    rng = np.random.default_rng(42)
    np.save(root / "image_feat.npy", rng.uniform(0.1, 1, (6, 5)).astype(np.float32))
    np.save(root / "text_feat.npy", rng.uniform(0.1, 1, (6, 3)).astype(np.float32))
    return load_config(overrides=[f"data.root={tmp_path / 'data'}", "model.knn_k=2",
                       "model.embedding_dim=4", "model.knn_chunk_size=2", "runtime.device=cpu",
                       f"runtime.output_root={tmp_path / 'runs'}", f"runtime.cache_root={tmp_path / 'cache'}",
                       "runtime.num_threads=1", "train.epochs=2", "train.batch_size=4",
                       "eval.topk=[1, 3, 10]", "eval.monitor=Recall@3", "eval.item_chunk_size=2"])


@pytest.fixture
def tiny_data(tiny_config):
    return load_dataset(tiny_config["data"], ("image", "text"), tiny_config["eval"])
