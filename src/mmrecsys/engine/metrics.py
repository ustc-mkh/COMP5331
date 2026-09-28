import numpy as np


def user_metrics(recommended: np.ndarray, targets: np.ndarray, topk: list[int]) -> dict[str, float]:
    if len(targets) == 0:
        raise ValueError("Metrics require at least one target")
    hits = np.isin(recommended, targets) & (recommended >= 0)
    result = {}
    for k in topk:
        relevant = hits[:k]
        discount = 1 / np.log2(np.arange(len(relevant)) + 2)
        ideal = (1 / np.log2(np.arange(min(k, len(targets))) + 2)).sum()
        result[f"Recall@{k}"] = float(relevant.sum() / len(targets))
        result[f"NDCG@{k}"] = float((relevant * discount).sum() / ideal)
    return result


def batch_user_metrics(recommended: np.ndarray, targets, users: np.ndarray,
                       topk: list[int]) -> dict[str, np.ndarray]:
    """Per-user metrics using CSR membership queries for the entire batch."""
    counts = np.diff(targets.indptr)[users]
    if np.any(counts == 0):
        raise ValueError("Metrics require at least one target")
    valid = recommended >= 0
    rows = np.broadcast_to(users[:, None], recommended.shape)
    hits = np.zeros(recommended.shape, dtype=bool)
    hits[valid] = np.asarray(targets[rows[valid], recommended[valid]]).ravel() != 0
    discount = 1 / np.log2(np.arange(max(topk)) + 2)
    # Match the scalar reference's reduction for each possible ideal length.
    ideal = np.array([0.] + [discount[:length].sum() for length in range(1, max(topk) + 1)])
    result = {}
    for k in topk:
        relevant = hits[:, :k]
        result[f"Recall@{k}"] = relevant.sum(axis=1) / counts
        result[f"NDCG@{k}"] = (relevant * discount[:relevant.shape[1]]).sum(axis=1) / ideal[np.minimum(k, counts)]
    return result
