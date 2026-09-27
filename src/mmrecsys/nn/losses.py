import torch
import torch.nn.functional as F


def bpr_loss(users, positives, negatives):
    positive_scores = (users * positives).sum(-1, keepdim=True)
    negative_scores = (users[:, None, :] * negatives).sum(-1)
    return F.softplus(negative_scores - positive_scores).mean()


def info_nce(first, second, temperature: float):
    """Directional in-batch InfoNCE; diagonal pairs are positives."""
    logits = F.normalize(first, dim=-1) @ F.normalize(second, dim=-1).T / temperature
    return F.cross_entropy(logits, torch.arange(len(first), device=first.device))
