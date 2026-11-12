"""FREEDOM with optional initialization-only shared DAMPS calibration.

Adapted from Task 6 / enoche/FREEDOM; see docs/models.md.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn

from .base import DotProductScorer, LossOutput, Recommender
from ._shared import cached_ui, normalize_with_epsilon, parse_config, validate_config
from ..experiment.artifacts import array_digest
from ..nn.damps import DAMPS
from ..nn.graph import knn_graph
from ..nn.losses import bpr_loss


@dataclass(frozen=True)
class FREEDOMConfig:
    name: str = "freedom"
    embedding_dim: int = 64
    n_ui_layers: int = 2
    n_item_layers: int = 1
    knn_k: int = 10
    knn_chunk_size: int = 512
    mm_image_weight: float = 0.1
    dropout: float = 0.8
    reg_weight: float = 0.0001
    damps_enabled: bool = False
    damps_apc: bool = True
    damps_avrf: bool = True
    damps_imcf: bool = True
    damps_eps: float = 1e-6

    def __post_init__(self):
        validate_config(self, "freedom", positive=("embedding_dim", "knn_k", "knn_chunk_size"),
                        nonnegative_int=("n_ui_layers", "n_item_layers"),
                        nonnegative=("reg_weight",), probabilities=("dropout",))
        if (type(self.mm_image_weight) not in (int, float)
                or not math.isfinite(self.mm_image_weight) or not 0 <= self.mm_image_weight <= 1):
            raise ValueError("model.mm_image_weight must be in [0, 1]")

    @classmethod
    def parse(cls, values):
        return parse_config(cls, values)


class FREEDOM(Recommender):
    def __init__(self, config, data, cache):
        super().__init__()
        self.config = config
        self.n_users, self.n_items = data.n_users, data.n_items
        self.register_buffer("ui_graph", cached_ui(data, cache), persistent=False)
        self.register_buffer("masked_graph", self.ui_graph, persistent=False)
        edges = torch.as_tensor(data.edges.copy(), dtype=torch.long).T.contiguous()
        self.register_buffer("edge_indices", edges, persistent=False)
        self.register_buffer("edge_weights", self._edge_weights(edges), persistent=False)
        dim = config.embedding_dim
        self.user_embedding = nn.Embedding(data.n_users, dim)
        self.item_embedding = nn.Embedding(data.n_items, dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        raw = {m: data.features[m].clone() for m in ("image", "text")}
        self.damps = None
        if config.damps_enabled:
            self.damps = DAMPS(dim, apc=config.damps_apc, avrf=config.damps_avrf,
                               imcf=config.damps_imcf, eps=config.damps_eps,
                               raw_image=raw["image"], raw_text=raw["text"])
            # Task 6 calibrates once, then trains independent feature Parameters.
            self.damps.requires_grad_(False)
            with torch.no_grad():
                raw["image"], raw["text"] = self.damps()
        self.features = nn.ModuleDict()
        self.projections = nn.ModuleDict()
        graphs = []
        for modality in ("image", "text"):
            features = raw[modality].detach().clone()
            self.features[modality] = nn.Embedding.from_pretrained(features, freeze=False)
            self.projections[modality] = nn.Linear(features.shape[1], dim)
            # Calibration depends on initialization; hash its actual output, not only the data.
            fingerprint = (array_digest(features.cpu().numpy()) if config.damps_enabled
                           else data.feature_fingerprints[modality])
            metadata = {"kind": "freedom_item_knn", "feature": fingerprint,
                        "n_items": data.n_items, "numbering": "feature_row_is_item_id",
                        "similarity": "cosine", "tie_break": "item_id_ascending",
                        "k": config.knn_k, "self_loops": True, "symmetrize": False,
                        "edge_weight": "binary", "normalization": "symmetric", "epsilon": 1e-7}
            graphs.append(cache.get_or_build(metadata, lambda features=features: normalize_with_epsilon(
                knn_graph(features, config.knn_k, chunk_size=config.knn_chunk_size,
                          self_loops=True, symmetrize=False, edge_weight="binary", normalization="none"))))
        graph = config.mm_image_weight * graphs[0] + (1 - config.mm_image_weight) * graphs[1]
        # A calibrated graph is initialization-dependent state and must survive reload exactly.
        self.register_buffer("item_graph", graph.coalesce(), persistent=config.damps_enabled)

    def _edge_weights(self, edges):
        user_degree = torch.bincount(edges[0], minlength=self.n_users).float() + 1e-7
        item_degree = torch.bincount(edges[1], minlength=self.n_items).float() + 1e-7
        return user_degree[edges[0]].rsqrt() * item_degree[edges[1]].rsqrt()

    @torch.no_grad()
    def on_epoch_start(self, epoch):
        if self.config.dropout == 0:
            self.masked_graph = self.ui_graph
            return
        count = int(self.edge_weights.numel() * (1 - self.config.dropout))
        selected = (torch.multinomial(self.edge_weights, count) if count
                    else self.edge_indices.new_empty(0))
        kept = self.edge_indices[:, selected].clone()
        weights = self._edge_weights(kept)
        kept[1] += self.n_users
        self.masked_graph = torch.sparse_coo_tensor(
            torch.cat((kept, kept.flip(0)), dim=1), torch.cat((weights, weights)),
            self.ui_graph.shape, check_invariants=True).coalesce()

    def encode(self):
        item_side = self.item_embedding.weight
        for _ in range(self.config.n_item_layers):
            item_side = torch.sparse.mm(self.item_graph, item_side)
        hidden = torch.cat((self.user_embedding.weight, self.item_embedding.weight))
        layers = [hidden]
        graph = self.masked_graph if self.training else self.ui_graph
        for _ in range(self.config.n_ui_layers):
            hidden = torch.sparse.mm(graph, hidden)
            layers.append(hidden)
        final = torch.stack(layers).mean(0)
        return final[:self.n_users], final[self.n_users:] + item_side

    def compute_loss(self, batch):
        if batch.negative_items is None or batch.negative_items.shape[1] != 1:
            raise ValueError("FREEDOM requires one negative item per positive")
        users, items = self.encode()
        u = users[batch.users]
        ranking = bpr_loss(u, items[batch.positive_items], items[batch.negative_items])
        modal = ranking.new_zeros(())
        for modality in ("text", "image"):
            features = self.projections[modality](self.features[modality].weight)
            modal = modal + bpr_loss(u, features[batch.positive_items], features[batch.negative_items])
        regularization = self.config.reg_weight * modal
        return LossOutput(ranking + regularization,
                          {"bpr": ranking, "modality_bpr": regularization}, len(u))

    @torch.no_grad()
    def make_scorer(self):
        if self.training:
            raise RuntimeError("make_scorer requires model.eval()")
        users, items = self.encode()
        return DotProductScorer(users, items)
