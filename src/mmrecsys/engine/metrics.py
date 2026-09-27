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
