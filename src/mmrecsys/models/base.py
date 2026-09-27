from dataclasses import dataclass
from typing import Protocol

import torch
from torch import nn

from ..data.views import TrainBatch


@dataclass
class LossOutput:
    total: torch.Tensor
    components: dict[str, torch.Tensor]
    batch_size: int


class Scorer(Protocol):
    def score(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor: ...


@dataclass
class DotProductScorer:
    users: torch.Tensor
    items: torch.Tensor

    def score(self, users, items):
        return self.users[users] @ self.items[items].T


class Recommender(nn.Module):
    def compute_loss(self, batch: TrainBatch) -> LossOutput:
        raise NotImplementedError

    def make_scorer(self) -> Scorer:
        raise NotImplementedError

    def on_epoch_start(self, epoch: int) -> None:
        pass

    def on_epoch_end(self, epoch: int) -> None:
        pass
