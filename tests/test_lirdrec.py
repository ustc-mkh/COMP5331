"""LIRDRec numerical checks use dense NumPy equations on the six-item fixture."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch

from mmrecsys.data.views import TrainBatch
from mmrecsys.experiment.artifacts import GraphCache
from mmrecsys.models.lirdrec import LIRDRec, LIRDRecConfig


def small_config(**kwargs):
    return LIRDRecConfig(embedding_dim=4, knn_k=2, knn_chunk_size=2, **kwargs)


def as_numpy(value):
    return value.detach().cpu().numpy().astype(np.float64)


def linear_numpy(value, layer):
    result = value @ as_numpy(layer.weight).T
    return result if layer.bias is None else result + as_numpy(layer.bias)


def leaky_numpy(value):
    return np.where(value >= 0, value, 0.01 * value)


def dct_numpy(value):
    width = value.shape[1]
    basis = np.sqrt(2 / width) * np.cos(np.pi / width * np.arange(width)[:, None]
                                       * (np.arange(width)[None, :] + 0.5))
    basis[0] /= np.sqrt(2)
    return value @ basis.T


def dense_oracle(model, data):
    """Author baseline equations, independently constructing both graph families.

    This deliberately does not read model graph buffers, projected features,
    fusion output, or its loss helpers, so changes to normalization, DCT,
    layer aggregation, weighting, and regularization are observable.
    """
    config = model.config
    size = data.n_users + data.n_items
    adjacency = np.zeros((size, size))
    for user, item in data.edges:
        adjacency[user, item + data.n_users] = 1
        adjacency[item + data.n_users, user] = 1
    degrees = adjacency.sum(1) + 1e-7
    adjacency /= np.sqrt(degrees[:, None] * degrees[None, :])

    raw = [as_numpy(data.features[name]) for name in ("image", "text")]
    knn_graphs = []
    for features in raw:
        unit = features / np.linalg.norm(features, axis=1, keepdims=True)
        neighbors = np.argsort(-(unit @ unit.T), axis=1, kind="stable")[:, :config.knn_k]
        graph = np.zeros((data.n_items, data.n_items))
        graph[np.arange(data.n_items)[:, None], neighbors] = 1
        degree = graph.sum(1) + 1e-7
        knn_graphs.append(graph / np.sqrt(degree[:, None] * degree[None, :]))
    item_graph = config.mm_image_weight * knn_graphs[0] + (1 - config.mm_image_weight) * knn_graphs[1]

    inputs = [*raw, np.concatenate([dct_numpy(x) for x in raw], axis=1)]
    initial_views = []
    for prefix, features in zip(("v", "t", "s"), inputs):
        hidden = leaky_numpy(linear_numpy(features, getattr(model, f"{prefix}_MLP")))
        projected = linear_numpy(hidden, getattr(model, f"{prefix}_MLP_1"))
        preference = getattr(model, "id_preference" if prefix == "s" else f"{prefix}_preference")
        nodes = np.concatenate((as_numpy(preference), projected))
        initial_views.append(nodes / np.maximum(np.linalg.norm(nodes, axis=1, keepdims=True), 1e-12))
    initial = np.concatenate(initial_views, axis=1)
    representation = sum(np.linalg.matrix_power(adjacency, layer) @ initial
                         for layer in range(config.n_ui_layers + 1))

    fusion = model.fusion_module
    user_views = np.split(representation[:data.n_users], 3, axis=1)
    logits = np.concatenate([
        linear_numpy(leaky_numpy(linear_numpy(view, attention.fc1)), attention.fc2)
        for view, attention in zip(user_views, (fusion.weco_a, fusion.weco_b, fusion.weco_c))
    ], axis=1)
    logits = float(fusion.w1) * logits + float(fusion.w2) * as_numpy(fusion.theta)
    weights = np.exp(logits - logits.max(axis=1, keepdims=True))
    weights /= weights.sum(axis=1, keepdims=True)
    users = np.concatenate([view * weights[:, index:index + 1]
                            for index, view in enumerate(user_views)], axis=1)
    items = representation[data.n_users:]
    items = items + np.linalg.matrix_power(item_graph, config.n_item_layers) @ items
    return users, items, weights


@pytest.mark.parametrize("ui_layers,item_layers", [(0, 1), (2, 2)])
def test_baseline_embeddings_and_loss_match_independent_oracle(tiny_data, tmp_path, ui_layers, item_layers):
    torch.manual_seed(18)
    model = LIRDRec(small_config(n_ui_layers=ui_layers, n_item_layers=item_layers),
                    tiny_data[0], GraphCache(tmp_path))
    model.on_epoch_start(1)
    expected_users, expected_items, expected_attention = dense_oracle(model, tiny_data[0])
    users, items = model.encode()
    np.testing.assert_allclose(as_numpy(users), expected_users, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(as_numpy(items), expected_items, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(as_numpy(model.fusion_module.last_att), expected_attention,
                               rtol=2e-5, atol=2e-6)
    assert users.shape == (4, 12) and items.shape == (6, 12)
    batch = TrainBatch(torch.tensor([0, 1, 2]), torch.tensor([0, 1, 3]),
                       torch.tensor([[4], [5], [0]]))
    positive_scores = (expected_users[[0, 1, 2]] * expected_items[[0, 1, 3]]).sum(1)
    negative_scores = (expected_users[[0, 1, 2]] * expected_items[[4, 5, 0]]).sum(1)
    expected_bpr = np.logaddexp(0, negative_scores - positive_scores).mean()
    # Source regularizes all final user/item representations, including non-batch rows.
    expected_reg = model.config.reg_weight * (np.square(expected_users).mean() + np.square(expected_items).mean())
    loss = model.compute_loss(batch)
    assert loss.total.item() == pytest.approx(expected_bpr + expected_reg, rel=2e-5)
    loss.total.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    model.eval()
    torch.testing.assert_close(model.make_scorer().score(batch.users, batch.positive_items),
                               users[batch.users] @ items[batch.positive_items].T)


def test_pwc_epoch_schedule_and_checkpoint_restore(tiny_data, tmp_path):
    model = LIRDRec(small_config(), tiny_data[0], GraphCache(tmp_path))
    initial = model.config.decay_weight
    attention = model.fusion_module
    for epoch in range(1, 4):
        previous_attention = attention.last_att.clone()
        model.on_epoch_start(epoch)
        odds = initial / (1 - initial) * model.config.decay_base ** (epoch * (epoch + 1) / 2)
        assert float(attention.w1) == pytest.approx(odds / (1 + odds), abs=1e-14)
        assert float(attention.w2) == pytest.approx(1 / (1 + odds), abs=1e-14)
        torch.testing.assert_close(attention.theta, previous_attention, rtol=0, atol=0)
        model.encode()
        assert not attention.last_att.requires_grad
    state = deepcopy(model.state_dict())
    restored = LIRDRec(small_config(), tiny_data[0], GraphCache(tmp_path))
    restored.load_state_dict(state)
    assert restored.current_epoch.item() == 3
    for value in (model, restored):
        value.on_epoch_start(4)
    for actual, expected in zip(restored.encode(), model.encode()):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(restored.fusion_module.blend_weights, attention.blend_weights,
                               rtol=0, atol=0)

    # Validation must not change the state used by the next training epoch.
    model.eval()
    before = deepcopy(model.state_dict())
    model.make_scorer()
    for name, expected in before.items():
        torch.testing.assert_close(model.state_dict()[name], expected, rtol=0, atol=0)


def test_damps_bypass_preserves_paired_baseline(tiny_data, tmp_path):
    config = small_config()
    torch.manual_seed(203)
    baseline = LIRDRec(config, tiny_data[0], GraphCache(tmp_path))
    torch.manual_seed(203)
    bypass = LIRDRec(replace(config, damps_enabled=True, damps_apc=False,
                             damps_avrf=False, damps_imcf=False), tiny_data[0], GraphCache(tmp_path))
    for name, parameter in baseline.named_parameters():
        torch.testing.assert_close(dict(bypass.named_parameters())[name], parameter, rtol=0, atol=0)
    for model in (baseline, bypass):
        model.on_epoch_start(1)
    for actual, expected in zip(bypass.encode(), baseline.encode()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("disabled", [None, "apc", "avrf", "imcf"])
def test_damps_and_three_ablations_backpropagate(tiny_data, tmp_path, disabled):
    torch.manual_seed(81)
    config = small_config(damps_enabled=True)
    if disabled is not None:
        config = replace(config, **{f"damps_{disabled}": False})
    model = LIRDRec(config, tiny_data[0], GraphCache(tmp_path))
    model.on_epoch_start(1)
    batch = TrainBatch(torch.tensor([0, 1, 2]), torch.tensor([0, 1, 3]),
                       torch.tensor([[4], [5], [0]]))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    old_projection = model.v_MLP.weight.detach().clone()
    loss = model.compute_loss(batch)
    assert torch.isfinite(loss.total)
    loss.total.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    assert model.v_MLP.weight.grad.abs().sum() > 0
    assert model.t_MLP.weight.grad.abs().sum() > 0
    for name, parameter in model.damps.named_parameters():
        assert parameter.grad.abs().sum() > 0, name
    optimizer.step()
    assert not torch.equal(model.v_MLP.weight, old_projection)
    restored = LIRDRec(config, tiny_data[0], GraphCache(tmp_path))
    restored.load_state_dict(model.state_dict())
    model.eval()
    restored.eval()
    for actual, expected in zip(restored.encode(), model.encode()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
