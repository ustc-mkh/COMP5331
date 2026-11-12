"""Pre-extracted-feature dataset adapter, preserving upstream split and shuffle semantics."""
from pathlib import Path

import numpy as np
import pandas as pd

from . import ROOT
from .config import fingerprint, sha256_file

EXPECTED = {"users": 19445, "items": 7050, "interactions": 160792}
REQUIRED = ("baby.inter", "image_feat.npy", "text_feat.npy")


class Interactions:
    """The subset of upstream RecDataset needed by its unchanged dataloaders.

    Removes unused raw-image/lmdb dependencies; does not remap IDs or re-split data.
    """
    uid_field = "userID"
    iid_field = "itemID"

    def __init__(self, df, user_num, item_num):
        self.df = df
        self.user_num = user_num
        self.item_num = item_num
        self.inter_num = len(df)

    def copy(self, df):
        return Interactions(df.copy(), self.user_num, self.item_num)

    def get_user_num(self):
        return self.user_num

    def get_item_num(self):
        return self.item_num

    def shuffle(self):
        # Same NumPy RNG and pandas sampling operation as upstream.
        self.df = self.df.sample(frac=1, replace=False).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        return self.df.iloc[index]


def inspect_data(directory, require_baby=True, verify_reference=True):
    directory = Path(directory).resolve()
    for name in REQUIRED:
        if not (directory / name).is_file():
            raise FileNotFoundError("Missing {}; run scripts/download_baby.py".format(directory / name))
    df = pd.read_csv(directory / "baby.inter", sep="\t", usecols=["userID", "itemID", "x_label"])
    if df.empty or df.isna().any().any():
        raise ValueError("Empty interactions or missing IDs/split labels")
    for name in ("userID", "itemID", "x_label"):
        if not pd.api.types.is_integer_dtype(df[name]):
            raise ValueError(name + " must contain integer IDs/labels")
    if min(df.userID.min(), df.itemID.min()) < 0 or set(df.x_label.unique()) != {0, 1, 2}:
        raise ValueError("Expected nonnegative IDs and x_label=0/1/2")
    if df.duplicated(["userID", "itemID"]).any():
        raise ValueError("Repeated user-item pairs, including possible cross-split leakage")
    users, items = int(df.userID.max()) + 1, int(df.itemID.max()) + 1
    if df.userID.nunique() != users or df.itemID.nunique() != items:
        raise ValueError("IDs must be contiguous, zero-based indices into the released features")
    statistics = {"users": users, "items": items, "interactions": len(df)}
    if require_baby and statistics != EXPECTED:
        raise ValueError("Dataset differs from the paper: {} != {}".format(statistics, EXPECTED))
    feature_info = {}
    for name in REQUIRED[1:]:
        array = np.load(directory / name, mmap_mode="r", allow_pickle=False)
        if array.ndim != 2 or array.shape[0] != items or not np.issubdtype(array.dtype, np.floating):
            raise ValueError("Invalid feature rows/type: " + name)
        zero_rows = 0
        for start in range(0, len(array), 1024):
            block = array[start:start + 1024]
            if not np.isfinite(block).all():
                raise ValueError("Non-finite features: " + name)
            zero_rows += int(np.all(block == 0, axis=1).sum())
        feature_info[name] = {"shape": list(array.shape), "dtype": str(array.dtype), "zero_rows": zero_rows}
    files = {name: sha256_file(directory / name) for name in REQUIRED}
    reference = ROOT / "configs" / "baby_files.json"
    if require_baby and verify_reference and reference.exists():
        import json
        expected_files = json.loads(reference.read_text(encoding="utf-8"))
        for name in REQUIRED:
            if files[name] != expected_files[name]["sha256"]:
                raise ValueError("Dataset bytes differ from verified official Baby: " + name)
    splits = []
    split_info = {}
    train_users = set(df.loc[df.x_label == 0, "userID"])
    for label, name in enumerate(("train", "valid", "test")):
        split = df.loc[df.x_label == label, ["userID", "itemID"]].copy()
        before = len(split)
        if label:
            split = split.loc[split.userID.isin(train_users)].copy()
        if split.empty:
            raise ValueError("Empty evaluated split after upstream cold-user filtering: " + name)
        split_info[name] = {"interactions": len(split), "users": int(split.userID.nunique()),
                            "dropped_cold_user_interactions": before - len(split)}
        splits.append(Interactions(split, users, items))
    report = dict(statistics, features=feature_info, splits=split_info, sha256=files,
                  fingerprint=fingerprint(files), source="official Baby" if require_baby else "synthetic test fixture",
                  protocol={"split": "released x_label: 0 train, 1 valid, 2 test",
                            "ranking": "all items; mask training positives only in BOTH valid and test",
                            "cold_users": "remove valid/test interactions whose user is absent from train",
                            "graph": "training interactions only"})
    return report, splits


def runtime_config(config, directory, device, output):
    directory = Path(directory).resolve()
    # Upstream concatenates data_path + dataset; keep an explicit trailing separator.
    import os
    runtime = dict(config)
    runtime.update(dataset=directory.name, data_path=str(directory.parent) + os.sep,
                   device=device, recommend_topk=str(Path(output) / "recommendations"))
    return runtime
