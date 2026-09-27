from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import torch


@dataclass(frozen=True)
class TrainData:
    n_users: int
    n_items: int
    edges: np.ndarray
    history: sp.csr_matrix
    features: dict[str, torch.Tensor]
    fingerprint: str
    feature_fingerprints: dict[str, str]


@dataclass(frozen=True)
class EvalData:
    split: str
    users: np.ndarray
    targets: sp.csr_matrix
    history: sp.csr_matrix


@dataclass(frozen=True)
class TrainBatch:
    users: torch.Tensor
    positive_items: torch.Tensor
    negative_items: torch.Tensor | None = None

    def to(self, device: torch.device) -> "TrainBatch":
        return TrainBatch(self.users.to(device), self.positive_items.to(device),
                          None if self.negative_items is None else self.negative_items.to(device))
