import torch
import torch.nn.functional as F


def normalize_graph(graph: torch.Tensor, normalization: str = "symmetric") -> torch.Tensor:
    graph = graph.coalesce()
    if normalization == "none":
        return graph
    if normalization != "symmetric":
        raise ValueError(f"Unknown normalization: {normalization}")
    row, col = graph.indices()
    degree = torch.zeros(graph.shape[0], device=graph.device, dtype=graph.dtype)
    degree.scatter_add_(0, row, graph.values())
    if (degree < 0).any():
        raise ValueError("Negative weighted degrees cannot be inverse-square-root normalized")
    inverse = degree.clamp_min(torch.finfo(degree.dtype).tiny).rsqrt()
    inverse[degree == 0] = 0
    weights = graph.values() * inverse[row] * inverse[col]
    if not torch.isfinite(weights).all():
        raise ValueError("Non-finite normalized graph weights")
    return torch.sparse_coo_tensor(graph.indices(), weights, graph.shape, check_invariants=True).coalesce()


def interaction_graph(edges, n_users: int, n_items: int, *, self_loops: bool = False,
                      normalization: str = "symmetric") -> torch.Tensor:
    edges = torch.as_tensor(edges, dtype=torch.long)
    user, item = edges[:, 0], edges[:, 1] + n_users
    indices = torch.stack((torch.cat((user, item)), torch.cat((item, user))))
    size = n_users + n_items
    if self_loops:
        diagonal = torch.arange(size)
        indices = torch.cat((indices, torch.stack((diagonal, diagonal))), dim=1)
    graph = torch.sparse_coo_tensor(indices, torch.ones(indices.shape[1]), (size, size), check_invariants=True)
    return normalize_graph(graph, normalization)


def interaction_block(graph: torch.Tensor, n_users: int) -> torch.Tensor:
    graph = graph.coalesce()
    row, col = graph.indices()
    keep = (row < n_users) & (col >= n_users)
    indices = torch.stack((row[keep], col[keep] - n_users))
    return torch.sparse_coo_tensor(indices, graph.values()[keep],
                                  (n_users, graph.shape[0] - n_users), check_invariants=True).coalesce()


@torch.no_grad()
def knn_graph(features: torch.Tensor, k: int, *, chunk_size: int,
              self_loops: bool, symmetrize: bool, edge_weight: str,
              normalization: str) -> torch.Tensor:
    """Exact cosine kNN in row blocks; ties prefer smaller item IDs.

    self_loops=True permits self among the k neighbors, as in Eq. (7).
    symmetrize=True takes the union, preserving each selected cosine weight.
    Cosines are not silently clipped or replaced by binary adjacency.
    """
    size = features.shape[0]
    if not 1 <= k <= size - int(not self_loops):
        raise ValueError(f"knn_k={k} exceeds available neighbors for {size} items")
    if chunk_size < 1 or edge_weight not in ("cosine", "binary"):
        raise ValueError("Invalid kNN graph options")
    if features.ndim != 2 or not torch.isfinite(features).all():
        raise ValueError("kNN requires finite matrix features")
    unit = F.normalize(features.detach().float(), dim=1)
    rows, columns, weights = [], [], []
    # Never construct an N x N matrix, even when the configured block is larger than N.
    block_size = min(chunk_size, max(1, size - 1))
    for start in range(0, size, block_size):
        stop = min(start + block_size, size)
        similarity = unit[start:stop] @ unit.T
        if not self_loops:
            similarity[torch.arange(stop - start, device=unit.device),
                       torch.arange(start, stop, device=unit.device)] = -torch.inf
        # A stable sort defines the boundary-tie rule exactly, without score perturbations.
        selected = torch.argsort(similarity, dim=1, descending=True, stable=True)[:, :k]
        value = similarity.gather(1, selected)
        rows.append(torch.arange(start, stop, device=unit.device).repeat_interleave(k))
        columns.append(selected.flatten())
        weights.append(value.flatten() if edge_weight == "cosine" else torch.ones_like(value).flatten())
    row, col, value = torch.cat(rows), torch.cat(columns), torch.cat(weights)
    if symmetrize:
        forward = row * size + col
        reverse = col * size + row
        missing = ~torch.isin(reverse, forward)
        row, col, value = (torch.cat((row, col[missing])), torch.cat((col, row[missing])),
                           torch.cat((value, value[missing])))
    graph = torch.sparse_coo_tensor(torch.stack((row, col)), value, (size, size), check_invariants=True).coalesce()
    return normalize_graph(graph, normalization)
