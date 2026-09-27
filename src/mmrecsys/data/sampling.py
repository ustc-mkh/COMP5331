from dataclasses import dataclass

import numpy as np
import torch

from .views import TrainBatch, TrainData


@dataclass(frozen=True)
class BatchSpec:
    kind: str
    negatives: int = 1


class TrainingSampler:
    """Single-process batches; randomness is independently derived per epoch."""

    def __init__(self, data: TrainData, spec: BatchSpec, batch_size: int, seed: int):
        if spec.kind not in ("positive", "pairwise") or spec.negatives < 1:
            raise ValueError("Invalid batch specification")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.data, self.spec, self.batch_size, self.seed = data, spec, batch_size, seed
        self.generator = torch.Generator()
        self.positives = [data.history.indices[data.history.indptr[u]:data.history.indptr[u + 1]]
                          for u in range(data.n_users)]
        active = np.unique(data.edges[:, 0])
        if spec.kind == "pairwise" and any(len(self.positives[u]) == data.n_items for u in active):
            raise ValueError("Cannot sample negatives: a training user has interacted with every item")

    def batches(self, epoch: int):
        self.generator.manual_seed((self.seed + epoch * 1_000_003) % (2**63 - 1))
        order = torch.randperm(len(self.data.edges), generator=self.generator).numpy()
        for start in range(0, len(order), self.batch_size):
            edges = self.data.edges[order[start:start + self.batch_size]]
            negatives = None
            if self.spec.kind == "pairwise":
                # Uniform ranks in the complement avoid rejection loops, even for dense users.
                negatives = np.empty((len(edges), self.spec.negatives), dtype=np.int64)
                for row, user in enumerate(edges[:, 0]):
                    positive = self.positives[user]
                    ranks = torch.randint(self.data.n_items - len(positive), (self.spec.negatives,),
                                          generator=self.generator).numpy()
                    adjusted = positive - np.arange(len(positive))
                    negatives[row] = ranks + np.searchsorted(adjusted, ranks, side="right")
                negatives = torch.from_numpy(negatives)
            yield TrainBatch(torch.from_numpy(edges[:, 0].copy()),
                             torch.from_numpy(edges[:, 1].copy()), negatives)

    def state_dict(self):
        return {"generator": self.generator.get_state()}

    def load_state_dict(self, state):
        self.generator.set_state(state["generator"].cpu())
