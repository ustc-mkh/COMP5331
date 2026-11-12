"""SMORE frequency fusion, adapted from the Task 4 implementation.

Reference: kennethorq/SMORE, WSDM 2025; see docs/models.md.
"""
from dataclasses import dataclass

import torch
from torch import nn

from .base import DotProductScorer, LossOutput, Recommender
from ._shared import cached_ui, parse_config, validate_config
from ..nn.damps import DAMPS
from ..nn.graph import interaction_block, knn_graph
from ..nn.losses import bpr_loss, info_nce


@dataclass(frozen=True)
class SMOREConfig:
    name: str = "smore"
    embedding_dim: int = 64
    n_ui_layers: int = 4
    n_item_layers: int = 1
    image_knn_k: int = 40
    text_knn_k: int = 15
    knn_chunk_size: int = 512
    dropout: float = 0.1
    cl_weight: float = 0.01
    reg_weight: float = 0.0001
    temperature: float = 0.2
    damps_enabled: bool = False
    damps_apc: bool = True
    damps_avrf: bool = True
    damps_imcf: bool = True
    damps_eps: float = 1e-6

    def __post_init__(self):
        validate_config(self, "smore",
                        positive=("embedding_dim", "image_knn_k", "text_knn_k", "knn_chunk_size"),
                        nonnegative_int=("n_ui_layers", "n_item_layers"),
                        nonnegative=("cl_weight", "reg_weight", "temperature"), probabilities=("dropout",))
        if self.temperature == 0:
            raise ValueError("model.temperature must be positive")

    @classmethod
    def parse(cls, values):
        return parse_config(cls, values)


def max_fusion_graph(image, text):
    """Union of edges, taking the maximum weight where both modalities select it."""
    image, text = image.coalesce(), text.coalesce()
    size = image.shape[0]
    indices = torch.cat((image.indices(), text.indices()), dim=1)
    keys, inverse = torch.unique(indices[0] * size + indices[1], return_inverse=True)
    values = image.values().new_full((len(keys),), -torch.inf)
    values.scatter_reduce_(0, inverse, torch.cat((image.values(), text.values())),
                           reduce="amax", include_self=True)
    return torch.sparse_coo_tensor(torch.stack((keys // size, keys % size)), values,
                                  image.shape, check_invariants=True).coalesce()


class SMORE(Recommender):
    def __init__(self, config, data, cache):
        super().__init__()
        self.config = config
        self.n_users, self.n_items = data.n_users, data.n_items
        dim = config.embedding_dim
        self.user_embedding = nn.Embedding(data.n_users, dim)
        self.item_embedding = nn.Embedding(data.n_items, dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        self.register_buffer("ui_graph", cached_ui(data, cache, epsilon=0), persistent=False)
        self.register_buffer("ui_block", interaction_block(self.ui_graph, data.n_users), persistent=False)
        self.features = nn.ModuleDict()
        for modality, k in (("image", config.image_knn_k), ("text", config.text_knn_k)):
            raw = data.features[modality]
            self.features[modality] = nn.Embedding.from_pretrained(raw.clone(), freeze=False)
            metadata = {"kind": "item_knn", "feature": data.feature_fingerprints[modality],
                        "n_items": data.n_items, "numbering": "feature_row_is_item_id",
                        "similarity": "cosine", "tie_break": "item_id_ascending", "k": k,
                        "self_loops": True, "symmetrize": False, "edge_weight": "cosine",
                        "normalization": "symmetric"}
            graph = cache.get_or_build(metadata, lambda raw=raw, k=k: knn_graph(
                raw, k, chunk_size=config.knn_chunk_size, self_loops=True,
                symmetrize=False, edge_weight="cosine", normalization="symmetric"))
            self.register_buffer(f"{modality}_graph", graph, persistent=False)
        self.register_buffer("fusion_graph", max_fusion_graph(self.image_graph, self.text_graph),
                             persistent=False)
        self.projections = nn.ModuleDict({m: nn.Linear(data.features[m].shape[1], dim)
                                          for m in ("image", "text")})
        self.queries = nn.ModuleDict({m: nn.Sequential(nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, dim, bias=False))
                                     for m in ("image", "text")})
        self.gates = nn.ModuleDict({m: nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
                                   for m in ("image", "text", "fusion")})
        self.preferences = nn.ModuleDict({m: nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
                                         for m in ("image", "text", "fusion")})
        self.spectral_weights = nn.ParameterDict({m: nn.Parameter(torch.randn(1, dim // 2 + 1, 2))
                                                 for m in ("image", "text", "fusion")})
        self.dropout = nn.Dropout(config.dropout)
        self.damps = None
        # Initialize after backbone parameters to preserve the paired baseline initialization.
        if config.damps_enabled:
            self.damps = DAMPS(dim, apc=config.damps_apc, avrf=config.damps_avrf,
                               imcf=config.damps_imcf, eps=config.damps_eps)
            with torch.no_grad():
                self.damps.initialize(*self.project_modalities())

    def project_modalities(self):
        return tuple(self.projections[m](self.features[m].weight) for m in ("image", "text"))

    def spectrum_convolution(self, image, text):
        zi, zt = torch.fft.rfft(image, dim=-1, norm="ortho"), torch.fft.rfft(text, dim=-1, norm="ortho")
        return tuple(torch.fft.irfft(spectrum * torch.view_as_complex(self.spectral_weights[m]),
                                     n=self.config.embedding_dim, dim=-1, norm="ortho")
                     for m, spectrum in (("image", zi), ("text", zt), ("fusion", zi * zt)))

    def encode(self):
        image, text = self.project_modalities()
        if self.damps is not None:
            image, text = self.damps(image, text)
        convolutions = self.spectrum_convolution(image, text)
        hidden = torch.cat((self.user_embedding.weight, self.item_embedding.weight))
        layers = [hidden]
        for _ in range(self.config.n_ui_layers):
            hidden = torch.sparse.mm(self.ui_graph, hidden)
            layers.append(hidden)
        content = torch.stack(layers).mean(0)
        views = {}
        for modality, convolution in zip(("image", "text", "fusion"), convolutions):
            items = self.item_embedding.weight * self.gates[modality](convolution)
            for _ in range(self.config.n_item_layers):
                items = torch.sparse.mm(getattr(self, f"{modality}_graph"), items)
            views[modality] = torch.cat((torch.sparse.mm(self.ui_block, items), items))
        # Source attention is a softmax over feature dimensions, separately per modality.
        image_view = self.queries["image"](views["fusion"]).softmax(-1) * views["image"]
        text_view = self.queries["text"](views["fusion"]).softmax(-1) * views["text"]
        side = torch.stack([self.dropout(self.preferences[m](content)) * view
                            for m, view in (("image", image_view), ("text", text_view),
                                            ("fusion", views["fusion"]))]).mean(0)
        final = content + side
        return final[:self.n_users], final[self.n_users:], side, content

    def compute_loss(self, batch):
        if batch.negative_items is None or batch.negative_items.shape[1] != 1:
            raise ValueError("SMORE requires one negative item per positive")
        users, items, side, content = self.encode()
        u, p, n = users[batch.users], items[batch.positive_items], items[batch.negative_items]
        ranking = bpr_loss(u, p, n)
        # Unified batches use their actual size, including the final partial batch.
        regularization = self.config.reg_weight * (u.square().sum() + p.square().sum()
                                                   + n.square().sum()) / (2 * len(u))
        nodes = batch.positive_items + self.n_users
        contrastive = self.config.cl_weight * (
            info_nce(side[nodes], content[nodes], self.config.temperature)
            + info_nce(side[batch.users], content[batch.users], self.config.temperature))
        return LossOutput(ranking + regularization + contrastive,
                          {"bpr": ranking, "regularization": regularization,
                           "contrastive": contrastive}, len(u))

    @torch.no_grad()
    def make_scorer(self):
        if self.training:
            raise RuntimeError("make_scorer requires model.eval()")
        users, items, _, _ = self.encode()
        return DotProductScorer(users, items)
