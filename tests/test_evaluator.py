import numpy as np
import scipy.sparse as sp
import torch

from mmrecsys.engine.evaluator import rank_users
from mmrecsys.engine.metrics import user_metrics


class MatrixScorer:
    def __init__(self, scores):
        self.scores = scores

    def score(self, users, items):
        return self.scores[users][:, items]


def test_chunked_ranking_mask_ties_and_padding():
    scores = torch.tensor([[9., 8., 7., 7., 6.], [2., 2., 2., 2., 2.], [1., 2., 3., 4., 5.]])
    history = sp.csr_matrix([[1, 1, 0, 0, 0], [0, 1, 0, 0, 0], [1, 1, 1, 1, 1]])
    expected = np.array([[2, 3, 4, -1, -1], [0, 2, 3, 4, -1], [-1, -1, -1, -1, -1]])
    for chunk in (1, 2, 5, 20):
        ranked = rank_users(MatrixScorer(scores), np.arange(3), history, 5, 10, chunk, "cpu")
        np.testing.assert_array_equal(ranked, expected)
    metrics = user_metrics(expected[0], np.array([2, 4]), [1, 3, 10])
    assert metrics["Recall@1"] == 0.5
    assert metrics["NDCG@1"] == 1
    assert metrics["Recall@3"] == 1
    assert np.isclose(metrics["NDCG@3"], 1.5 / (1 + 1 / np.log2(3)))
    assert user_metrics(expected[2], np.array([0]), [10])["Recall@10"] == 0
