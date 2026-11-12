"""Reuse the exact upstream BM3 forward, loss, trainable features, and predictor."""
import sys

import numpy as np
import scipy.sparse as sp
import torch

from . import ROOT

sys.path.insert(0, str(ROOT / "upstream" / "BM3" / "src"))
from models.bm3 import BM3 as OriginalBM3  # noqa: E402
from utils.dataloader import TrainDataLoader, EvalDataLoader  # noqa: E402
from utils.topk_evaluator import TopKEvaluator  # noqa: E402
from utils.utils import early_stopping  # noqa: E402


class BM3(OriginalBM3):
    def get_norm_adj_mat(self, interaction_matrix):
        """Public SciPy equivalent of upstream's removed private DOK._update call.

        Retains binary symmetric edges, D^(-1/2) A D^(-1/2), and degree +1e-7.
        """
        inter = interaction_matrix.tocoo()
        rows = np.concatenate([inter.row, inter.col + self.n_users])
        cols = np.concatenate([inter.col + self.n_users, inter.row])
        adjacency = sp.coo_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                                  shape=(self.n_nodes, self.n_nodes)).tocsr()
        adjacency.data.fill(1.0)
        degree = np.asarray((adjacency > 0).sum(axis=1)).reshape(-1) + 1e-7
        diagonal = sp.diags(np.power(degree, -0.5))
        normalized = (diagonal @ adjacency @ diagonal).tocoo()
        indices = torch.from_numpy(np.vstack((normalized.row, normalized.col)).astype(np.int64))
        values = torch.from_numpy(normalized.data.astype(np.float32))
        return torch.sparse_coo_tensor(indices, values, (self.n_nodes, self.n_nodes))


def make_loaders(config, splits, max_eval_users=None):
    train = TrainDataLoader(config, splits[0], batch_size=config["train_batch_size"], shuffle=True)
    evaluated = []
    for split in splits[1:]:
        if max_eval_users is not None:
            users = split.df.userID.unique()[:max_eval_users]
            split = split.copy(split.df.loc[split.df.userID.isin(users)])
        evaluated.append(EvalDataLoader(config, split, additional_dataset=splits[0],
                                        batch_size=config["eval_batch_size"]))
    return train, evaluated[0], evaluated[1]


@torch.no_grad()
def evaluate(model, loader, config, save_topk=None):
    """Same full-sort scores, training-only mask, topk and rounded metrics as upstream."""
    model.eval()
    batch_matrices = []
    # Explicitly reset cursors after any interrupted/incomplete previous iteration.
    loader.pr = loader.inter_pr = 0
    for batch in loader:
        scores = model.full_sort_predict(batch)
        if not torch.isfinite(scores).all():
            raise FloatingPointError("Non-finite recommendation scores")
        mask = batch[1]
        scores[mask[0], mask[1]] = -1e10
        _, topk = torch.topk(scores, max(config["topk"]), dim=-1)
        batch_matrices.append(topk)
    result = TopKEvaluator(config).evaluate(batch_matrices, loader)
    if save_topk is not None:
        from pathlib import Path
        path = Path(save_topk)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, users=loader.get_eval_users().numpy(),
                            items=torch.cat(batch_matrices).cpu().numpy())
    return {key: float(value) for key, value in result.items()}
