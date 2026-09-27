from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from mmrecsys.data.views import TrainBatch
from mmrecsys.experiment.artifacts import GraphCache
from mmrecsys.models.mgcn import MGCN, MGCNConfig
from mmrecsys.nn.losses import info_nce


def dense_reference(model):
    """Independent dense evaluation of paper equations, including fusion coefficients."""
    c = model.config
    initial = torch.cat((model.user_embedding.weight, model.item_embedding.weight))
    adjacency = model.ui_graph.to_dense()
    behavior = sum(torch.linalg.matrix_power(adjacency, layer) @ initial
                   for layer in range(c.n_ui_layers + 1)) / (c.n_ui_layers + 1)
    modalities = []
    for name in ("image", "text"):
        raw = model.feature_embeddings[name].weight if c.trainable_features else getattr(model, f"{name}_features")
        projected = F.linear(raw, model.projections[name].weight, model.projections[name].bias)
        gate_layer = model.purifiers[name][0]
        purified = model.item_embedding.weight * torch.sigmoid(F.linear(projected, gate_layer.weight, gate_layer.bias))
        item_view = torch.linalg.matrix_power(getattr(model, f"{name}_graph").to_dense(), c.n_item_layers) @ purified
        modalities.append(torch.cat((adjacency[:model.n_users, model.n_users:] @ item_view, item_view)))
    scores = torch.cat([model.common_attention(view) for view in modalities], dim=1)
    weights = scores.exp() / scores.exp().sum(1, keepdim=True)
    common = modalities[0] * weights[:, :1] + modalities[1] * weights[:, 1:]
    specific = [torch.sigmoid(model.preferences[name][0](behavior)) * (view - common)
                for name, view in zip(("image", "text"), modalities)]
    side = common + (specific[0] + specific[1]) / 2 if c.fusion == "paper" else (common + sum(specific)) / 3
    return behavior + side, behavior, side


@pytest.mark.parametrize("fusion,regularization,trainable", [("paper", "parameters", False), ("author", "batch_final", True)])
def test_paper_equations_loss_and_gradients(tiny_data, tiny_config, tmp_path, fusion, regularization, trainable):
    config = replace(MGCNConfig.parse(tiny_config["model"]), fusion=fusion,
                     regularization=regularization, trainable_features=trainable)
    model = MGCN(config, tiny_data[0], GraphCache(tmp_path))
    u, i, behavior, multimodal = model.encode()
    expected, expected_behavior, expected_side = dense_reference(model)
    torch.testing.assert_close(torch.cat((u, i)), expected)
    torch.testing.assert_close(behavior, expected_behavior)
    torch.testing.assert_close(multimodal, expected_side)
    batch = TrainBatch(torch.tensor([0, 1, 2]), torch.tensor([0, 1, 3]), torch.tensor([[4], [5], [0]]))
    output = model.compute_loss(batch)
    eu, ep, en = expected[batch.users], expected[batch.positive_items + 4], expected[batch.negative_items[:, 0] + 4]
    bpr = -F.logsigmoid((eu * ep).sum(1) - (eu * en).sum(1)).mean()
    reg = (sum(p.square().sum() for p in model.parameters()) if regularization == "parameters"
           else (eu.square().sum() + ep.square().sum() + en.square().sum()) / 6)
    cl = info_nce(expected_side[batch.users], expected_behavior[batch.users], 0.2)
    cl += info_nce(expected_side[batch.positive_items + 4], expected_behavior[batch.positive_items + 4], 0.2)
    torch.testing.assert_close(output.total, bpr + config.reg_weight * reg + config.cl_weight * cl)
    output.total.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    assert model.projections["image"].weight.grad.abs().sum() > 0
    assert "ui_graph" not in model.state_dict()
    model.eval()
    scorer = model.make_scorer()
    torch.testing.assert_close(scorer.score(torch.tensor([0, 2]), torch.tensor([1, 3])), u[[0, 2]] @ i[[1, 3]].T)


def test_info_nce_matches_exp_definition_and_is_stable():
    x, y = torch.randn(4, 3), torch.randn(4, 3)
    cosine = F.normalize(x, dim=1) @ F.normalize(y, dim=1).T
    expected = -(cosine.diag() / 0.2 - torch.log(torch.exp(cosine / 0.2).sum(1))).mean()
    torch.testing.assert_close(info_nce(x, y, 0.2), expected)
    assert torch.isfinite(info_nce(x, y, 0.0001))
