"""Small configuration and graph helpers for the migrated backbones."""
from dataclasses import fields
import math

import torch

from ..nn.graph import interaction_graph


def parse_config(cls, values):
    unknown = values.keys() - {field.name for field in fields(cls)}
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {sorted(unknown)}")
    return cls(**values)


def validate_config(config, name, *, positive=(), nonnegative_int=(),
                    nonnegative=(), probabilities=()):
    if config.name != name:
        raise ValueError(f"Expected model.name={name}")
    for field in positive + nonnegative_int:
        value = getattr(config, field)
        if type(value) is not int or value < (1 if field in positive else 0):
            raise ValueError(f"Invalid model.{field}")
    for field in nonnegative + probabilities + ("damps_eps",):
        value = getattr(config, field)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid model.{field}")
        if field in probabilities and value >= 1:
            raise ValueError(f"model.{field} must be in [0, 1)")
    if config.damps_eps == 0:
        raise ValueError("model.damps_eps must be positive")
    for field in ("damps_enabled", "damps_apc", "damps_avrf", "damps_imcf"):
        if type(getattr(config, field)) is not bool:
            raise ValueError(f"model.{field} must be boolean")


def normalize_with_epsilon(graph, epsilon=1e-7):
    """Author binary-degree normalization, including the degree epsilon."""
    graph = graph.coalesce()
    row, col = graph.indices()
    degree = graph.values().new_zeros(graph.shape[0])
    degree.scatter_add_(0, row, graph.values())
    inverse = (degree + epsilon).clamp_min(torch.finfo(degree.dtype).tiny).rsqrt()
    inverse[degree == 0] = 0
    return torch.sparse_coo_tensor(graph.indices(), graph.values() * inverse[row] * inverse[col],
                                  graph.shape, check_invariants=True).coalesce()


def cached_ui(data, cache, *, epsilon=1e-7):
    return cache.get_or_build(
        {"kind": "interaction", "train": data.fingerprint,
         "n_users": data.n_users, "n_items": data.n_items,
         "self_loops": False, "normalization": "symmetric", "epsilon": epsilon},
        lambda: normalize_with_epsilon(interaction_graph(
            data.edges, data.n_users, data.n_items, normalization="none"), epsilon))
