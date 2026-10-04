"""LIRDRec backbone adapted from enoche/LIRDRec/src/models/lirdrec.py.

Preserve the baseline's 4*d hidden projections, fixed DCT features, binary
item graphs, summed UI layers, PWC recurrence, and full-representation penalty.
DAMPS calibrates the two projected modalities once; it does not change the
backbone dimensions, raw features, item graph, or shared DCT branch.
"""
from dataclasses import dataclass
import math

from scipy.fft import dct
import torch
from torch import nn
import torch.nn.functional as F

from .base import DotProductScorer, LossOutput, Recommender
from ..data.views import TrainBatch, TrainData
from ..experiment.artifacts import GraphCache
from ..nn.damps import DAMPS
from ..nn.graph import interaction_graph, knn_graph
from ..nn.losses import bpr_loss


@dataclass(frozen=True)
class LIRDRecConfig:
    name: str = "lirdrec"
    embedding_dim: int = 64
    n_ui_layers: int = 2
    n_item_layers: int = 1
    knn_k: int = 10
    knn_chunk_size: int = 512
    mm_image_weight: float = 0.1
    dropout: float = 0.0
    reg_weight: float = 0.0001
    decay_base: float = 0.9
    decay_weight: float = 0.9
    damps_enabled: bool = False
    damps_apc: bool = True
    damps_avrf: bool = True
    damps_imcf: bool = True
    damps_eps: float = 1e-6

    def __post_init__(self):
        for name in ("embedding_dim", "knn_k", "knn_chunk_size"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"model.{name} must be a positive integer")
        if self.embedding_dim < 4:
            raise ValueError("model.embedding_dim must be at least 4 for PWC")
        for name in ("n_ui_layers", "n_item_layers"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"model.{name} must be a nonnegative integer")
        for name in ("mm_image_weight", "dropout", "reg_weight", "decay_base", "decay_weight", "damps_eps"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"Invalid model.{name}")
        if not 0 <= self.mm_image_weight <= 1 or not 0 <= self.dropout < 1:
            raise ValueError("Require mm_image_weight in [0, 1] and dropout in [0, 1)")
        if self.reg_weight < 0 or self.damps_eps <= 0:
            raise ValueError("Require nonnegative reg_weight and positive damps_eps")
        if not 0 < self.decay_base <= 1 or not 0 <= self.decay_weight <= 1:
            raise ValueError("Require decay_base in (0, 1] and decay_weight in [0, 1]")
        for name in ("damps_enabled", "damps_apc", "damps_avrf", "damps_imcf"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"model.{name} must be boolean")

    @classmethod
    def parse(cls, values):
        unknown = values.keys() - cls.__dataclass_fields__.keys()
        if unknown:
            raise ValueError(f"Unknown model fields: {sorted(unknown)}")
        result = cls(**values)
        if result.name != "lirdrec":
            raise ValueError("Expected model.name=lirdrec")
        return result


class WeCopy(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, value):
        return self.fc2(F.leaky_relu(self.fc1(value)))


class PWC(nn.Module):
    """Author recurrence with checkpointed weights and detached attention memory."""

    def __init__(self, n_users, dim, base, weight):
        super().__init__()
        self.weco_a = WeCopy(dim, dim // 4)
        self.weco_b = WeCopy(dim, dim // 4)
        self.weco_c = WeCopy(dim, dim // 4)
        self.theta = nn.Parameter(nn.init.xavier_normal_(torch.empty(n_users, 3)))
        self.base = base
        self.register_buffer("blend_weights", torch.tensor([weight, 1 - weight], dtype=torch.float64))
        self.register_buffer("last_att", self.theta.detach().clone())

    @property
    def w1(self):
        return self.blend_weights[0]

    @property
    def w2(self):
        return self.blend_weights[1]

    @torch.no_grad()
    def update_w(self, epoch):
        # The author compounds base**epoch on the PREVIOUS normalized weights.
        # This is not a single base**epoch schedule from the initial weights.
        if self.w2.item() == 0:
            self.blend_weights.copy_(self.blend_weights.new_tensor([1, 0]))
        else:
            self.blend_weights[0].mul_(self.base ** epoch)
            self.blend_weights.div_(self.blend_weights.sum())
        self.theta.copy_(self.last_att)

    def forward(self, image, text, shared):
        scores = torch.cat((self.weco_a(image), self.weco_b(text), self.weco_c(shared)), dim=1)
        attention = (self.w1 * scores + self.w2 * self.theta).softmax(dim=1)
        # Evaluation must neither advance the recurrence nor retain an autograd graph.
        if self.training:
            with torch.no_grad():
                self.last_att.copy_(attention.detach())
        return torch.cat((attention[:, :1] * image, attention[:, 1:2] * text,
                          attention[:, 2:] * shared), dim=1)


def _author_normalize(graph, *, double_degrees=False):
    """Binary row-degree normalization, retaining the author's 1e-7 epsilon."""
    graph = graph.coalesce()
    row, col = graph.indices()
    degree = torch.zeros(graph.shape[0], dtype=torch.float64 if double_degrees else graph.dtype,
                         device=graph.device)
    degree.scatter_add_(0, row, graph.values().to(degree.dtype))
    inverse = (degree + 1e-7).rsqrt()
    values = (graph.values() * inverse[row] * inverse[col]).to(graph.dtype)
    return torch.sparse_coo_tensor(graph.indices(), values, graph.shape).coalesce()


class LIRDRec(Recommender):
    def __init__(self, config: LIRDRecConfig, data: TrainData, cache: GraphCache):
        super().__init__()
        self.config = config
        self.n_users, self.n_items = data.n_users, data.n_items
        self.embedding_dim = config.embedding_dim
        if not {"image", "text"}.issubset(data.features):
            raise ValueError("LIRDRec requires image and text features")
        # The baseline forward reads v_feat/t_feat, not its unused trainable embeddings.
        self.register_buffer("v_feat", data.features["image"].detach().clone(), persistent=False)
        self.register_buffer("t_feat", data.features["text"].detach().clone(), persistent=False)
        dim = config.embedding_dim
        self.v_preference = nn.Parameter(nn.init.xavier_normal_(torch.empty(data.n_users, dim)))
        self.v_MLP = nn.Linear(self.v_feat.shape[1], 4 * dim)
        self.v_MLP_1 = nn.Linear(4 * dim, dim, bias=False)
        self.t_preference = nn.Parameter(nn.init.xavier_normal_(torch.empty(data.n_users, dim)))
        self.t_MLP = nn.Linear(self.t_feat.shape[1], 4 * dim)
        self.t_MLP_1 = nn.Linear(4 * dim, dim, bias=False)
        self.id_preference = nn.Parameter(nn.init.xavier_normal_(torch.empty(data.n_users, dim)))
        self.s_MLP = nn.Linear(self.v_feat.shape[1] + self.t_feat.shape[1], 4 * dim)
        self.s_MLP_1 = nn.Linear(4 * dim, dim, bias=False)
        self.fusion_module = PWC(data.n_users, dim, config.decay_base, config.decay_weight)
        # Orthonormal DCT-II matches torch_dct without an additional dependency.
        dct_features = [torch.from_numpy(dct(feature.cpu().numpy(), type=2, axis=-1,
                                           norm="ortho", workers=1)).to(feature)
                        for feature in (self.v_feat, self.t_feat)]
        self.register_buffer("interleaved_feat", torch.cat(dct_features, dim=1), persistent=False)
        self.register_buffer("current_epoch", torch.tensor(0, dtype=torch.long))

        ui = cache.get_or_build(
            {"kind": "lirdrec_interaction", "train": data.fingerprint,
             "n_users": data.n_users, "n_items": data.n_items, "epsilon": 1e-7},
            lambda: _author_normalize(interaction_graph(data.edges, data.n_users, data.n_items,
                                                       normalization="none"), double_degrees=True))
        self.register_buffer("norm_adj", ui, persistent=False)
        self.register_buffer("masked_adj", ui, persistent=False)
        indices = torch.as_tensor(data.edges, dtype=torch.long).T.contiguous()
        self.register_buffer("edge_indices", indices, persistent=False)
        self.register_buffer("edge_values", self._edge_weights(indices), persistent=False)
        modality_graphs = []
        for modality, features in (("image", self.v_feat), ("text", self.t_feat)):
            metadata = {"kind": "lirdrec_item_knn", "feature": data.feature_fingerprints[modality],
                        "n_items": data.n_items, "numbering": "feature_row_is_item_id",
                        "k": config.knn_k, "similarity": "cosine", "edge_weight": "binary",
                        "self_loops": True, "symmetrize": False, "epsilon": 1e-7,
                        "tie_break": "item_id_ascending"}
            graph = cache.get_or_build(metadata, lambda features=features: _author_normalize(
                knn_graph(features, config.knn_k, chunk_size=config.knn_chunk_size,
                          self_loops=True, symmetrize=False, edge_weight="binary", normalization="none")))
            modality_graphs.append(graph)
        mm = config.mm_image_weight * modality_graphs[0] + (1 - config.mm_image_weight) * modality_graphs[1]
        self.register_buffer("mm_adj", mm.coalesce(), persistent=False)

        # Construct after the backbone so paired variants have identical initialization.
        self.damps = (DAMPS(dim, apc=config.damps_apc, avrf=config.damps_avrf,
                            imcf=config.damps_imcf, eps=config.damps_eps)
                      if config.damps_enabled else None)
        if self.damps is not None:
            with torch.no_grad():
                self.damps.initialize(*self.project_modalities())

    def _edge_weights(self, indices):
        user_degree = torch.bincount(indices[0], minlength=self.n_users).float() + 1e-7
        item_degree = torch.bincount(indices[1], minlength=self.n_items).float() + 1e-7
        return user_degree[indices[0]].rsqrt() * item_degree[indices[1]].rsqrt()

    @torch.no_grad()
    def on_epoch_start(self, epoch):
        if type(epoch) is not int or epoch != self.current_epoch.item() + 1:
            raise ValueError("LIRDRec epochs must advance consecutively from the checkpoint epoch")
        self.fusion_module.update_w(epoch)
        self.current_epoch.fill_(epoch)
        if self.config.dropout == 0:
            self.masked_adj = self.norm_adj
            return
        count = int(self.edge_values.numel() * (1 - self.config.dropout))
        selected = torch.multinomial(self.edge_values, count) if count else self.edge_indices.new_empty(0)
        kept = self.edge_indices[:, selected].clone()
        values = self._edge_weights(kept)
        kept[1] += self.n_users
        indices = torch.cat((kept, kept.flip(0)), dim=1)
        self.masked_adj = torch.sparse_coo_tensor(indices, torch.cat((values, values)),
                                                self.norm_adj.shape).coalesce()

    def project_modalities(self):
        return (self.v_MLP_1(F.leaky_relu(self.v_MLP(self.v_feat))),
                self.t_MLP_1(F.leaky_relu(self.t_MLP(self.t_feat))))

    def encode(self, adj=None):
        if adj is None:
            adj = self.masked_adj if self.training else self.norm_adj
        image, text = self.project_modalities()
        if self.damps is not None:
            image, text = self.damps(image, text)
        shared = self.s_MLP_1(F.leaky_relu(self.s_MLP(self.interleaved_feat)))
        hidden = torch.cat([F.normalize(torch.cat((preference, items), dim=0), dim=1)
                            for preference, items in ((self.v_preference, image),
                                                      (self.t_preference, text),
                                                      (self.id_preference, shared))], dim=1)
        layers = [hidden]
        for _ in range(self.config.n_ui_layers):
            hidden = torch.sparse.mm(adj, hidden)
            layers.append(hidden)
        representation = torch.stack(layers).sum(0)
        users = self.fusion_module(*representation[:self.n_users].split(self.embedding_dim, dim=1))
        items = representation[self.n_users:]
        propagated = items
        for _ in range(self.config.n_item_layers):
            propagated = torch.sparse.mm(self.mm_adj, propagated)
        return users, items + propagated

    def forward(self, adj=None):
        return self.encode(adj)

    def compute_loss(self, batch: TrainBatch):
        if batch.negative_items is None:
            raise ValueError("LIRDRec requires negative items")
        users, items = self.encode()
        negatives = batch.negative_items
        if negatives.ndim == 1:
            negatives = negatives[:, None]
        ranking = bpr_loss(users[batch.users], items[batch.positive_items], items[negatives])
        regularization = self.config.reg_weight * (users.square().mean() + items.square().mean())
        return LossOutput(ranking + regularization, {"bpr": ranking, "reg": regularization}, len(batch.users))

    @torch.no_grad()
    def make_scorer(self):
        if self.training:
            raise RuntimeError("make_scorer requires model.eval()")
        return DotProductScorer(*self.encode())

    @torch.no_grad()
    def training_diagnostics(self):
        result = {"pwc/network_weight": self.fusion_module.w1.detach(),
                  "pwc/history_weight": self.fusion_module.w2.detach()}
        if self.damps is not None:
            for name, parameter in self.damps.named_parameters():
                if parameter.grad is not None:
                    result[f"gradient/damps/{name}/norm"] = parameter.grad.norm()
            if self.damps.mix_logits is not None:
                weights = self.damps.mix_logits.softmax(0)
                result["damps/avrf_mix_weight"], result["damps/imcf_mix_weight"] = weights.unbind()
        return result
