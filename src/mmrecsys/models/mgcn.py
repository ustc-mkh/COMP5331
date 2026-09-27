"""MGCN (ACM MM 2023), equations (1)-(19).

See docs/mgcn.md for the explicit paper/author-code differences.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn

from .base import DotProductScorer, LossOutput, Recommender
from ..data.views import TrainBatch, TrainData
from ..experiment.artifacts import GraphCache
from ..nn.graph import interaction_block, interaction_graph, knn_graph
from ..nn.losses import bpr_loss, info_nce


@dataclass(frozen=True)
class MGCNConfig:
    name: str = "mgcn"
    embedding_dim: int = 64
    n_ui_layers: int = 2
    n_item_layers: int = 1
    knn_k: int = 10
    knn_chunk_size: int = 512
    knn_self_loops: bool = True
    knn_symmetrize: bool = False
    knn_edge_weight: str = "cosine"
    knn_normalization: str = "symmetric"
    temperature: float = 0.2
    cl_weight: float = 0.01
    reg_weight: float = 0.0001
    fusion: str = "paper"
    regularization: str = "parameters"
    trainable_features: bool = False

    def __post_init__(self):
        for name in ("embedding_dim", "knn_k", "knn_chunk_size", "n_item_layers"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"model.{name} must be a positive integer")
        if type(self.n_ui_layers) is not int or self.n_ui_layers < 0:
            raise ValueError("model.n_ui_layers must be a nonnegative integer")
        for name in ("temperature", "cl_weight", "reg_weight"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid model.{name}")
        if self.temperature == 0:
            raise ValueError("temperature must be positive")
        for name in ("knn_self_loops", "knn_symmetrize", "trainable_features"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"model.{name} must be boolean")
        if self.fusion not in ("paper", "author") or self.regularization not in ("parameters", "batch_final"):
            raise ValueError("Invalid fusion or regularization mode")
        if self.knn_edge_weight not in ("cosine", "binary") or self.knn_normalization not in ("symmetric", "none"):
            raise ValueError("Invalid kNN weighting or normalization")

    @classmethod
    def parse(cls, values):
        unknown = values.keys() - cls.__dataclass_fields__.keys()
        if unknown:
            raise ValueError(f"Unknown model fields: {sorted(unknown)}")
        result = cls(**values)
        if result.name != "mgcn":
            raise ValueError("Expected model.name=mgcn")
        return result


class MGCN(Recommender):
    def __init__(self, config: MGCNConfig, data: TrainData, cache: GraphCache):
        super().__init__()
        self.config = config
        self.n_users, self.n_items = data.n_users, data.n_items
        self.modalities = ("image", "text")
        if not set(self.modalities).issubset(data.features):
            raise ValueError("MGCN requires image and text features")
        dim = config.embedding_dim
        self.user_embedding = nn.Embedding(data.n_users, dim)
        self.item_embedding = nn.Embedding(data.n_items, dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        ui = cache.get_or_build({"kind": "interaction", "train": data.fingerprint,
                                 "n_users": data.n_users, "n_items": data.n_items,
                                 "self_loops": False, "normalization": "symmetric"},
                                lambda: interaction_graph(data.edges, data.n_users, data.n_items))
        self.register_buffer("ui_graph", ui, persistent=False)
        self.register_buffer("ui_block", interaction_block(ui, data.n_users), persistent=False)
        self.feature_embeddings = nn.ModuleDict()
        self.projections = nn.ModuleDict()
        self.purifiers = nn.ModuleDict()
        self.preferences = nn.ModuleDict()
        for modality in self.modalities:
            features = data.features[modality]
            if config.trainable_features:
                self.feature_embeddings[modality] = nn.Embedding.from_pretrained(features.clone(), freeze=False)
            else:
                self.register_buffer(f"{modality}_features", features.clone(), persistent=False)
            self.projections[modality] = nn.Linear(features.shape[1], dim)
            self.purifiers[modality] = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
            self.preferences[modality] = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
            options = {"k": config.knn_k, "self_loops": config.knn_self_loops,
                       "symmetrize": config.knn_symmetrize, "edge_weight": config.knn_edge_weight,
                       "normalization": config.knn_normalization}
            metadata = {"kind": "item_knn", "feature": data.feature_fingerprints[modality],
                        "numbering": "feature_row_is_item_id", "n_items": data.n_items,
                        "similarity": "cosine", "tie_break": "item_id_ascending", **options}
            graph = cache.get_or_build(metadata, lambda: knn_graph(features, chunk_size=config.knn_chunk_size,
                                                                   **options))
            self.register_buffer(f"{modality}_graph", graph, persistent=False)
        self.common_attention = nn.Sequential(nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, 1, bias=False))

    def encode(self):
        # Eq. (3)-(5): ID-only LightGCN, including layer zero in the mean.
        hidden = torch.cat((self.user_embedding.weight, self.item_embedding.weight))
        behavior = hidden
        for _ in range(self.config.n_ui_layers):
            hidden = torch.sparse.mm(self.ui_graph, hidden)
            behavior = behavior + hidden
        behavior = behavior / (self.config.n_ui_layers + 1)
        views = []
        for modality in self.modalities:
            raw = (self.feature_embeddings[modality].weight if self.config.trainable_features
                   else getattr(self, f"{modality}_features"))
            # Eq. (1)-(2): ID embeddings multiplied by the modality-derived gate.
            items = self.item_embedding.weight * self.purifiers[modality](self.projections[modality](raw))
            for _ in range(self.config.n_item_layers):
                items = torch.sparse.mm(getattr(self, f"{modality}_graph"), items)
            views.append(torch.cat((torch.sparse.mm(self.ui_block, items), items)))
        # Eq. (11)-(15): shared attention and behavior-gated modality residuals.
        weights = torch.softmax(torch.cat([self.common_attention(view) for view in views], dim=1), dim=1)
        common = sum(weights[:, index:index + 1] * view for index, view in enumerate(views))
        residual = sum(self.preferences[modality](behavior) * (view - common)
                       for modality, view in zip(self.modalities, views))
        if self.config.fusion == "paper":
            multimodal = common + residual / len(views)
        else:
            multimodal = (common + residual) / (len(views) + 1)
        final = behavior + multimodal
        return final[:self.n_users], final[self.n_users:], behavior, multimodal

    def compute_loss(self, batch: TrainBatch) -> LossOutput:
        if batch.negative_items is None or batch.negative_items.ndim != 2:
            raise ValueError("MGCN requires negative_items with shape [B, K]")
        users, items, behavior, multimodal = self.encode()
        u, p, n = users[batch.users], items[batch.positive_items], items[batch.negative_items]
        ranking = bpr_loss(u, p, n)
        cl = info_nce(multimodal[batch.users], behavior[batch.users], self.config.temperature)
        positive_nodes = batch.positive_items + self.n_users
        cl = cl + info_nce(multimodal[positive_nodes], behavior[positive_nodes], self.config.temperature)
        if self.config.regularization == "parameters":
            regularizer = sum(parameter.square().sum() for parameter in self.parameters())
        else:
            regularizer = (u.square().sum() + p.square().sum() + n.square().sum()) / (2 * len(u))
        weighted_cl = self.config.cl_weight * cl
        weighted_reg = self.config.reg_weight * regularizer
        total = ranking + weighted_cl + weighted_reg
        return LossOutput(total, {"bpr": ranking, "contrastive": weighted_cl,
                                  "regularization": weighted_reg}, len(u))

    @torch.no_grad()
    def make_scorer(self):
        if self.training:
            raise RuntimeError("make_scorer requires model.eval()")
        users, items, _, _ = self.encode()
        return DotProductScorer(users, items)
