"""BM3, adapted from enoche/BM3 (ef82a333), under GPL-3.0.

See docs/models.md for attribution and protocol.
"""
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .base import DotProductScorer, LossOutput, Recommender
from ._shared import cached_ui, parse_config, validate_config
from ..nn.damps import DAMPS


@dataclass(frozen=True)
class BM3Config:
    name: str = "bm3"
    embedding_dim: int = 64
    n_ui_layers: int = 1
    dropout: float = 0.5
    reg_weight: float = 0.1
    cl_weight: float = 2.0
    damps_enabled: bool = False
    damps_apc: bool = True
    damps_avrf: bool = True
    damps_imcf: bool = True
    damps_eps: float = 1e-6

    def __post_init__(self):
        validate_config(self, "bm3", positive=("embedding_dim",),
                        nonnegative_int=("n_ui_layers",), nonnegative=("reg_weight", "cl_weight"),
                        probabilities=("dropout",))

    @classmethod
    def parse(cls, values):
        return parse_config(cls, values)


class BM3(Recommender):
    def __init__(self, config, data, cache):
        super().__init__()
        self.config = config
        self.n_users, self.n_items = data.n_users, data.n_items
        dim = config.embedding_dim
        self.register_buffer("ui_graph", cached_ui(data, cache), persistent=False)
        self.user_embedding = nn.Embedding(data.n_users, dim)
        self.item_embedding = nn.Embedding(data.n_items, dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        self.predictor = nn.Linear(dim, dim)
        nn.init.xavier_normal_(self.predictor.weight)
        self.features = nn.ModuleDict()
        self.projections = nn.ModuleDict()
        for modality in ("image", "text"):
            raw = data.features[modality]
            self.features[modality] = nn.Embedding.from_pretrained(raw.clone(), freeze=False)
            self.projections[modality] = nn.Linear(raw.shape[1], dim)
            nn.init.xavier_normal_(self.projections[modality].weight)
        self.damps = None
        if config.damps_enabled:
            self.damps = DAMPS(dim, apc=config.damps_apc, avrf=config.damps_avrf,
                               imcf=config.damps_imcf, eps=config.damps_eps)
            with torch.no_grad():
                self.damps.initialize(*self.project_modalities())

    def project_modalities(self):
        return tuple(self.projections[m](self.features[m].weight) for m in ("image", "text"))

    def encode(self):
        hidden = torch.cat((self.user_embedding.weight, self.item_embedding.weight))
        layers = [hidden]
        for _ in range(self.config.n_ui_layers):
            hidden = torch.sparse.mm(self.ui_graph, hidden)
            layers.append(hidden)
        final = torch.stack(layers).mean(0)
        return final[:self.n_users], final[self.n_users:] + self.item_embedding.weight

    def compute_loss(self, batch):
        users, items = self.encode()
        image, text = self.project_modalities()
        if self.damps is not None:
            image, text = self.damps(image, text)
        # The target branch is a stopped-gradient dropout view, with no EMA model.
        with torch.no_grad():
            target_u = F.dropout(users, self.config.dropout)
            target_i = F.dropout(items, self.config.dropout)
            target_t = F.dropout(text, self.config.dropout)
            target_v = F.dropout(image, self.config.dropout)
        u, p = batch.users, batch.positive_items
        online_u = self.predictor(users)[u]
        online_i = self.predictor(items)[p]
        online_t = self.predictor(text)[p]
        online_v = self.predictor(image)[p]

        def distance(first, second):
            return 1 - F.cosine_similarity(first, second, dim=-1).mean()

        collaborative = distance(online_u, target_i[p]) + distance(online_i, target_u[u])
        modal = (distance(online_t, target_i[p]) + distance(online_t, target_t[p])
                 + distance(online_v, target_i[p]) + distance(online_v, target_v[p]))
        # Original EmbLoss uses full-representation norms, not squared L2.
        regularization = self.config.reg_weight * (users.norm() + items.norm()) / self.n_items
        contrastive = self.config.cl_weight * modal
        return LossOutput(collaborative + contrastive + regularization,
                          {"bootstrap": collaborative, "contrastive": contrastive,
                           "regularization": regularization}, len(u))

    @torch.no_grad()
    def make_scorer(self):
        if self.training:
            raise RuntimeError("make_scorer requires model.eval()")
        users, items = self.encode()
        return DotProductScorer(self.predictor(users), self.predictor(items))
