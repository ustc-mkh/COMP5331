import numpy as np
import pytest
import torch

from mmrecsys.experiment.artifacts import GraphCache
from mmrecsys.nn.graph import interaction_block, interaction_graph, knn_graph


def test_interaction_normalization_and_offset():
    graph = interaction_graph(np.array([[0, 0], [0, 1], [1, 1]]), 3, 3)
    expected = torch.zeros(6, 6)
    expected[0, 3] = expected[3, 0] = 1 / np.sqrt(2)
    expected[0, 4] = expected[4, 0] = 0.5
    expected[1, 4] = expected[4, 1] = 1 / np.sqrt(2)
    torch.testing.assert_close(graph.to_dense(), expected)
    torch.testing.assert_close(interaction_block(graph, 3).to_dense(), expected[:3, 3:])


def test_knn_matches_dense_cosine_and_ties(tmp_path):
    features = torch.tensor([[1., 0.], [1., 0.], [1., 1.], [0., 1.]])
    unit = torch.nn.functional.normalize(features, dim=1)
    similarity = unit @ unit.T
    order = torch.argsort(similarity, dim=1, descending=True, stable=True)[:, :2]
    adjacency = torch.zeros(4, 4).scatter_(1, order, similarity.gather(1, order))
    inverse = adjacency.sum(1).rsqrt()
    expected = inverse[:, None] * adjacency * inverse[None, :]
    for chunk in (1, 2, 100):
        graph = knn_graph(features, 2, chunk_size=chunk, self_loops=True,
                          symmetrize=False, edge_weight="cosine", normalization="symmetric")
        torch.testing.assert_close(graph.to_dense(), expected)
    graph = knn_graph(features, 1, chunk_size=2, self_loops=False,
                      symmetrize=False, edge_weight="binary", normalization="none")
    assert graph.to_dense().diagonal().sum() == 0
    assert graph.to_dense()[0, 1] == 1
    with pytest.raises(ValueError, match="neighbors"):
        knn_graph(features, 5, chunk_size=2, self_loops=True,
                  symmetrize=False, edge_weight="cosine", normalization="symmetric")


def test_graph_cache_content_key(tmp_path):
    cache = GraphCache(tmp_path)
    graph = interaction_graph(np.array([[0, 0]]), 1, 1)
    cache.get_or_build({"feature": "a", "k": 1}, lambda: graph)
    cached = cache.get_or_build({"feature": "a", "k": 1}, lambda: pytest.fail("Cache miss"))
    torch.testing.assert_close(cached.to_dense(), graph.to_dense())
    cache.get_or_build({"feature": "b", "k": 1}, lambda: graph)
    cache.get_or_build({"feature": "a", "k": 2}, lambda: graph)
    assert len(list(tmp_path.glob("*.pt"))) == 3
