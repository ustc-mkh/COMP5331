import csv
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from .views import EvalData, TrainData
from ..experiment.artifacts import array_digest, content_key


def load_dataset(config: dict, modalities: tuple[str, ...], eval_config: dict):
    root = Path(config["root"]) / config["name"]
    path = root / config["interactions"]
    fields = [config["user_field"], config["item_field"], config["split_field"]]
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter=config["separator"])
        if not reader.fieldnames or not set(fields).issubset(reader.fieldnames):
            raise ValueError(f"{path}: required fields {fields}, found {reader.fieldnames}")
        records = []
        for line, row in enumerate(reader, 2):
            try:
                records.append(tuple(int(row[field]) for field in fields))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{line}: invalid integer ID or split") from exc
    values = np.asarray(records, dtype=np.int64)
    if values.ndim != 2 or len(values) == 0:
        raise ValueError(f"{path}: empty interactions")
    if (values[:, :2] < 0).any() or not np.isin(values[:, 2], [0, 1, 2]).all():
        raise ValueError(f"{path}: IDs must be nonnegative and split labels must be 0/1/2")
    n_users, n_items = (int(values[:, axis].max()) + 1 for axis in (0, 1))
    for axis, size in ((0, n_users), (1, n_items)):
        if not np.array_equal(np.unique(values[:, axis]), np.arange(size)):
            raise ValueError(f"{path}: non-contiguous global {'user' if axis == 0 else 'item'} IDs")
    pairs, counts = np.unique(values[:, :2], axis=0, return_counts=True)
    if (counts > 1).any():
        raise ValueError(f"{path}: duplicate interaction (including across splits): {pairs[counts > 1][0].tolist()}")
    edges = [values[values[:, 2] == label, :2].copy() for label in (0, 1, 2)]
    if any(len(part) == 0 for part in edges):
        raise ValueError(f"{path}: train, valid and test splits must all be nonempty")
    matrices = [sp.csr_matrix((np.ones(len(part), dtype=np.float32), (part[:, 0], part[:, 1])),
                              shape=(n_users, n_items)) for part in edges]
    features, fingerprints = {}, {}
    for modality in modalities:
        if modality not in config["features"]:
            raise ValueError(f"Missing required modality: {modality}")
        feature_path = root / config["features"][modality]
        array = np.load(feature_path, mmap_mode="r", allow_pickle=False)
        if array.ndim != 2 or array.shape[0] != n_items or array.shape[1] == 0:
            raise ValueError(f"{feature_path}: expected [{n_items}, feature_dim], got {array.shape}")
        if not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all():
            raise ValueError(f"{feature_path}: features must contain finite floating point values")
        fingerprints[modality] = array_digest(array)
        converted = np.array(array, dtype=np.float32, copy=True)
        if not np.isfinite(converted).all():
            raise ValueError(f"{feature_path}: features overflow float32")
        features[modality] = torch.from_numpy(converted)
    fingerprint = content_key({"n_users": n_users, "n_items": n_items,
                               "train_edges": array_digest(edges[0]), "features": fingerprints})
    train = TrainData(n_users, n_items, edges[0], matrices[0], features, fingerprint, fingerprints)
    views = {}
    for label, split in ((1, "valid"), (2, "test")):
        history = matrices[0]
        if split == "test" and eval_config["history"] == "train_valid":
            history = (history + matrices[1]).tocsr()
        eligible = np.diff(matrices[label].indptr) > 0
        if eval_config["require_train_user"]:
            eligible &= np.diff(matrices[0].indptr) > 0
        users = np.flatnonzero(eligible)
        if len(users) == 0:
            raise ValueError(f"No eligible evaluation users in {split}")
        views[split] = EvalData(split, users, matrices[label], history)
    # Full split identity belongs to the experiment, never the model's TrainData.
    metadata = {"fingerprint": content_key({"interactions": array_digest(values), "train": fingerprint}),
                "train_fingerprint": fingerprint, "features": fingerprints,
                "n_users": n_users, "n_items": n_items,
                "interactions": dict(zip(("train", "valid", "test"), map(len, edges)))}
    return train, views, metadata
