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
        self._legacy_sampling = False
        self._offsets = data.history.indptr
        self._counts = np.diff(self._offsets)
        self._adjusted = data.history.indices - (
            np.arange(len(data.history.indices)) - np.repeat(self._offsets[:-1], self._counts))
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
                if self._legacy_sampling:
                    # Preserve the random trajectory of checkpoints predating batched sampling.
                    for row, user in enumerate(edges[:, 0]):
                        positive = self.positives[user]
                        ranks = torch.randint(self.data.n_items - len(positive), (self.spec.negatives,),
                                              generator=self.generator).numpy()
                        adjusted = positive - np.arange(len(positive))
                        negatives[row] = ranks + np.searchsorted(adjusted, ranks, side="right")
                else:
                    users = edges[:, 0]
                    counts = self._counts[users]
                    # Equal-sized complements share a single unbiased integer draw.
                    for count in np.unique(counts):
                        rows = np.flatnonzero(counts == count)
                        negatives[rows] = torch.randint(
                            self.data.n_items - int(count), (len(rows), self.spec.negatives),
                            generator=self.generator).numpy()
                    # Batched upper_bound over each user's adjusted CSR row.
                    low = np.zeros_like(negatives)
                    high = np.broadcast_to(counts[:, None], negatives.shape).copy()
                    offsets = np.broadcast_to(self._offsets[users, None], negatives.shape)
                    while np.any(low < high):
                        active = low < high
                        middle = (low[active] + high[active]) // 2
                        right = self._adjusted[offsets[active] + middle] <= negatives[active]
                        low[active] = np.where(right, middle + 1, low[active])
                        high[active] = np.where(right, high[active], middle)
                    negatives += low
                negatives = torch.from_numpy(negatives)
            yield TrainBatch(torch.from_numpy(edges[:, 0].copy()),
                             torch.from_numpy(edges[:, 1].copy()), negatives)

    def state_dict(self):
        return {"generator": self.generator.get_state(),
                "algorithm_version": 1 if self._legacy_sampling else 2}

    def load_state_dict(self, state):
        version = state.get("algorithm_version", 1)
        if version not in (1, 2):
            raise ValueError("Unsupported sampling algorithm version")
        self._legacy_sampling = version == 1
        self.generator.set_state(state["generator"].cpu())


class DeviceTrainingSampler:
    """Resident interactions and epoch-wide exact complement sampling on the device.

    Grouped integer draws avoid floating-point rounding bias and rejection loops.
    Only training history is uploaded; held-out positives remain legal negatives.
    """

    def __init__(self, data: TrainData, spec: BatchSpec, batch_size: int, seed: int, device):
        # Reuse validation and CSR complement construction without changing CPU sampling.
        source = TrainingSampler(data, spec, batch_size, seed)
        self.device = torch.device(device)
        self.spec, self.batch_size, self.seed = spec, batch_size, seed
        self.generator = torch.Generator(device=self.device)
        self.edges = torch.as_tensor(data.edges, dtype=torch.long, device=self.device)
        self.n_edges = len(data.edges)
        if data.n_users * data.n_items > np.iinfo(np.int64).max:
            raise ValueError("Training history keys exceed int64 capacity")
        users = data.edges[:, 0]
        self.keys = torch.as_tensor(
            source._adjusted + np.repeat(np.arange(data.n_users, dtype=np.int64),
                                         source._counts) * data.n_items,
            dtype=torch.long, device=self.device)
        self.query_base = torch.as_tensor(users * data.n_items, dtype=torch.long, device=self.device)
        self.offsets = torch.as_tensor(source._offsets[users], dtype=torch.long, device=self.device)
        counts = source._counts[users]
        self.groups = []
        if spec.kind == "pairwise":
            for count in np.unique(counts):
                rows = np.flatnonzero(counts == count)
                self.groups.append((data.n_items - int(count), len(rows),
                                    torch.as_tensor(rows, dtype=torch.long, device=self.device)))

    def batches(self, epoch: int):
        self.generator.manual_seed((self.seed + epoch * 1_000_003) % (2**63 - 1))
        order = torch.randperm(self.n_edges, generator=self.generator, device=self.device)
        edges = self.edges[order]
        negatives = None
        if self.spec.kind == "pairwise":
            ranks = torch.empty((self.n_edges, self.spec.negatives), dtype=torch.long, device=self.device)
            for upper, count, rows in self.groups:
                draws = torch.randint(upper, (count, self.spec.negatives),
                                      generator=self.generator, device=self.device)
                ranks.index_copy_(0, rows, draws)
            # User-disjoint sorted key ranges implement per-user upper_bound in one call.
            queries = (ranks + self.query_base[:, None]).contiguous()
            skipped = torch.searchsorted(self.keys, queries, right=True) - self.offsets[:, None]
            negatives = (ranks + skipped)[order]
        for start in range(0, self.n_edges, self.batch_size):
            batch = edges[start:start + self.batch_size]
            yield TrainBatch(batch[:, 0], batch[:, 1],
                             None if negatives is None else negatives[start:start + self.batch_size])

    def state_dict(self):
        return {"generator": self.generator.get_state(), "algorithm_version": 3,
                "device_type": self.device.type}

    def load_state_dict(self, state):
        if state.get("algorithm_version") != 3 or state.get("device_type") != self.device.type:
            raise ValueError("Resident sampler checkpoint requires the same sampling algorithm and device type")
        self.generator.set_state(state["generator"].cpu())
