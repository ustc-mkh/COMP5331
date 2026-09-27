import numpy as np
import torch

from .metrics import user_metrics


def stable_topk(scores, items, k):
    # Lexicographic ordering: score descending, exact ties by item ID ascending.
    by_id = torch.argsort(items, dim=1, stable=True)
    scores, items = scores.gather(1, by_id), items.gather(1, by_id)
    order = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :k]
    return scores.gather(1, order), items.gather(1, order)


@torch.no_grad()
def rank_users(scorer, users: np.ndarray, history, n_items: int, k: int,
               item_chunk_size: int, device):
    user_tensor = torch.as_tensor(users, dtype=torch.long, device=device)
    best_scores = torch.empty((len(users), 0), device=device)
    best_items = torch.empty((len(users), 0), device=device, dtype=torch.long)
    sparse_history = history[users].tocoo()
    for start in range(0, n_items, item_chunk_size):
        items = torch.arange(start, min(n_items, start + item_chunk_size), device=device)
        scores = scorer.score(user_tensor, items)
        if scores.shape != (len(users), len(items)) or not torch.isfinite(scores).all():
            raise ValueError("Scorer returned invalid shape or non-finite scores")
        scores = scores.clone()
        mask = (sparse_history.col >= start) & (sparse_history.col < start + len(items))
        rows = torch.as_tensor(sparse_history.row[mask], dtype=torch.long, device=device)
        columns = torch.as_tensor(sparse_history.col[mask] - start, dtype=torch.long, device=device)
        scores[rows, columns] = -torch.inf
        chunk_scores, chunk_items = stable_topk(scores, items.expand(len(users), -1), k)
        best_scores, best_items = stable_topk(torch.cat((best_scores, chunk_scores), dim=1),
                                             torch.cat((best_items, chunk_items), dim=1), k)
    best_items[~torch.isfinite(best_scores)] = -1
    return best_items.cpu().numpy()


class Evaluator:
    def __init__(self, config: dict, n_items: int, device):
        self.config, self.n_items, self.device = config, n_items, device

    @torch.no_grad()
    def evaluate(self, model, view):
        model.eval()
        scorer = model.make_scorer()
        totals = {f"{metric}@{k}": 0.0 for metric in ("Recall", "NDCG") for k in self.config["topk"]}
        if len(view.users) == 0:
            raise ValueError("Evaluation has no eligible users")
        for start in range(0, len(view.users), self.config["user_batch_size"]):
            users = view.users[start:start + self.config["user_batch_size"]]
            predictions = rank_users(scorer, users, view.history, self.n_items,
                                     max(self.config["topk"]), self.config["item_chunk_size"], self.device)
            for user, predictions_for_user in zip(users, predictions):
                target = view.targets.indices[view.targets.indptr[user]:view.targets.indptr[user + 1]]
                for name, value in user_metrics(predictions_for_user, target, self.config["topk"]).items():
                    totals[name] += value
        return {**{name: value / len(view.users) for name, value in totals.items()}, "n_users": len(view.users)}
