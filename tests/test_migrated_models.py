"""Native model contracts, source equations, and shared training/recovery."""
from copy import deepcopy
import json

import pytest
import torch
import torch.nn.functional as F

from mmrecsys.config import load_config
from mmrecsys.data.sampling import TrainingSampler
from mmrecsys.data.views import TrainBatch
from mmrecsys.engine.checkpoint import load_checkpoint
from mmrecsys.experiment.ablation import run_ablation
from mmrecsys.experiment.runner import assemble, evaluate_experiment, train_experiment
from mmrecsys.models.smore import max_fusion_graph
from mmrecsys.registry import get_model


def model_config(tiny_config, name, enabled=False):
    overrides = [f"data.root={tiny_config['data']['root']}",
                 f"runtime.output_root={tiny_config['runtime']['output_root']}",
                 f"runtime.cache_root={tiny_config['runtime']['cache_root']}",
                 "runtime.device=cpu", "runtime.num_threads=1",
                 "model.embedding_dim=4", f"model.damps_enabled={str(enabled).lower()}",
                 "train.epochs=3", "train.batch_size=4", "train.patience=10",
                 "eval.topk=[1, 3]", "eval.monitor=Recall@3",
                 "eval.user_batch_size=2", "eval.item_chunk_size=2"]
    if name == "smore":
        overrides += ["model.image_knn_k=2", "model.text_knn_k=3", "model.knn_chunk_size=2"]
    if name == "freedom":
        overrides += ["model.knn_k=2", "model.knn_chunk_size=2", "model.dropout=0.4"]
    return load_config(model=name, dataset="baby", overrides=overrides)


@pytest.mark.parametrize("name", ["bm3", "smore", "freedom"])
@pytest.mark.parametrize("enabled", [False, True])
def test_native_batch_loss_gradients_and_scorer(tiny_config, name, enabled):
    cfg = model_config(tiny_config, name, enabled)
    data, _, _, model, _, _ = assemble(cfg)
    sampler = TrainingSampler(data, get_model(name).batch_spec, 4, 999)
    batch = next(sampler.batches(1))
    assert (batch.negative_items is None) == (name == "bm3")
    model.train()
    model.on_epoch_start(1)
    loss = model.compute_loss(batch)
    assert loss.total.ndim == 0 and torch.isfinite(loss.total)
    assert loss.batch_size == len(batch.users)
    torch.testing.assert_close(loss.total, sum(loss.components.values()))
    loss.total.backward()
    for parameter_name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, parameter_name
            assert torch.isfinite(parameter.grad).all(), parameter_name
        else:
            assert parameter.grad is None, parameter_name
    if enabled:
        if name == "freedom":
            assert not any(p.requires_grad for p in model.damps.parameters())
        else:
            assert model.damps.phase_residual.grad is not None
    model.eval()
    scorer = model.make_scorer()
    users, items = torch.arange(4), torch.arange(6)
    scores = scorer.score(users, items)
    assert scores.shape == (4, 6) and torch.isfinite(scores).all()
    torch.testing.assert_close(scores, torch.cat((scorer.score(users, items[:3]),
                                                scorer.score(users, items[3:])), dim=1))


@pytest.mark.parametrize("name", ["bm3", "smore", "freedom"])
@pytest.mark.parametrize("enabled", [False, True])
def test_train_reload_and_exact_epoch_resume(tiny_config, name, enabled):
    cfg = model_config(tiny_config, name, enabled)
    full_dir, full_result = train_experiment(cfg)
    partial = deepcopy(cfg)
    partial["train"]["epochs"] = 1
    resumed_dir, _ = train_experiment(partial)
    _, resumed_result = train_experiment(cfg, resumed_dir / "last.pt")
    assert full_result == resumed_result
    full = torch.load(full_dir / "last.pt", weights_only=False)
    resumed = torch.load(resumed_dir / "last.pt", weights_only=False)
    assert full["state"] == resumed["state"]
    for key, value in full["model"].items():
        other = resumed["model"][key]
        if value.is_sparse:
            value, other = value.to_dense(), other.to_dense()
        torch.testing.assert_close(value, other, rtol=0, atol=0)
    assert evaluate_experiment(resumed_dir) == resumed_result["test"]


def test_bm3_matches_original_bootstrap_loss_and_prediction(tiny_config):
    cfg = model_config(tiny_config, "bm3")
    _, _, _, model, _, _ = assemble(cfg)
    batch = TrainBatch(torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3]))
    users, items = model.encode()
    image, text = model.project_modalities()
    torch.manual_seed(17)
    with torch.no_grad():
        target_u = F.dropout(users.clone(), model.config.dropout)
        target_i = F.dropout(items.clone(), model.config.dropout)
        target_t = F.dropout(text.clone(), model.config.dropout)
        target_v = F.dropout(image.clone(), model.config.dropout)
    u, i = batch.users, batch.positive_items
    online_u, online_i = model.predictor(users)[u], model.predictor(items)[i]
    online_t, online_v = model.predictor(text)[i], model.predictor(image)[i]
    def distance(a, b):
        return 1 - F.cosine_similarity(a, b.detach(), dim=-1).mean()
    expected = distance(online_u, target_i[i]) + distance(online_i, target_u[u])
    expected += model.config.cl_weight * (distance(online_t, target_i[i])
                                         + distance(online_v, target_i[i])
                                         + distance(online_t, target_t[i])
                                         + distance(online_v, target_v[i]))
    expected += model.config.reg_weight * (users.norm() + items.norm()) / len(items)
    torch.manual_seed(17)
    actual = model.compute_loss(batch).total
    torch.testing.assert_close(actual, expected)
    expected_grads = torch.autograd.grad(expected, tuple(model.parameters()), retain_graph=True)
    actual_grads = torch.autograd.grad(actual, tuple(model.parameters()))
    for a, b in zip(actual_grads, expected_grads):
        torch.testing.assert_close(a, b)
    model.eval()
    torch.testing.assert_close(model.make_scorer().score(u, i),
                               model.predictor(users)[u] @ model.predictor(items)[i].T)


def test_smore_spectral_convolution_and_max_union(tiny_config):
    cfg = model_config(tiny_config, "smore")
    _, _, _, model, _, _ = assemble(cfg)
    image, text = torch.randn(6, 4), torch.randn(6, 4)
    with torch.no_grad():
        for weight in model.spectral_weights.values():
            weight[..., 0].fill_(1)
            weight[..., 1].zero_()
    a, b, fusion = model.spectrum_convolution(image, text)
    torch.testing.assert_close(a, image)
    torch.testing.assert_close(b, text)
    # Orthonormal FFT multiplication equals circular convolution divided by sqrt(D).
    circular = torch.stack([sum(image[:, j] * text[:, (k - j) % 4] for j in range(4))
                            for k in range(4)], dim=1) / 2
    torch.testing.assert_close(fusion, circular)
    first = torch.sparse_coo_tensor(torch.tensor([[0, 0], [0, 1]]), torch.tensor([0.2, -0.3]), (2, 2))
    second = torch.sparse_coo_tensor(torch.tensor([[0, 1], [0, 0]]), torch.tensor([0.7, 0.4]), (2, 2))
    torch.testing.assert_close(max_fusion_graph(first, second).to_dense(),
                               torch.tensor([[0.7, -0.3], [0.4, 0.0]]))


def test_freedom_pruning_and_initialization_state(tiny_config):
    cfg = model_config(tiny_config, "freedom", True)
    _, _, metadata, model, _, _ = assemble(cfg)
    model.on_epoch_start(1)
    graph = model.masked_graph.coalesce()
    count = int(model.edge_weights.numel() * (1 - model.config.dropout))
    assert graph._nnz() == 2 * count
    torch.testing.assert_close(graph.to_dense(), graph.to_dense().T)
    model.eval()
    scores = model.make_scorer().score(torch.arange(4), torch.arange(6))
    with torch.no_grad():
        for parameter in model.damps.parameters():
            parameter.add_(10)
    torch.testing.assert_close(scores, model.make_scorer().score(torch.arange(4), torch.arange(6)))
    assert "item_graph" in model.state_dict()
    run, _ = train_experiment(cfg)
    # Checkpoint contains the actual calibrated graph even if reconstruction's graph differs.
    with torch.no_grad():
        model.item_graph = model.item_graph * 0
    load_checkpoint(run / "best.pt", model, cfg, metadata["fingerprint"])
    assert model.item_graph.coalesce().values().abs().sum() > 0


@pytest.mark.parametrize("name", ["bm3", "smore", "freedom"])
def test_shared_component_ablation(tiny_config, name):
    cfg = model_config(tiny_config, name)
    cfg["train"]["epochs"] = 1
    output = run_ablation(cfg, [999], ["baseline", "full", "no_apc", "no_avrf", "no_imcf"])
    records = json.loads((output / "runs.json").read_text())
    assert len(records) == 5
    assert all(record["result"]["test"]["n_users"] == 4 for record in records)


@pytest.mark.parametrize("name", ["bm3", "smore", "freedom"])
@pytest.mark.parametrize("dataset", ["baby", "sports", "clothing", "elec", "microlens"])
def test_experiment_configs(name, dataset):
    for prefix, enabled in (("", False), ("damps_", True)):
        cfg = load_config(f"configs/experiments/{prefix}{name}_{dataset}.yaml")
        assert cfg["model"]["damps_enabled"] == enabled
    for override in ("model.embedding_dim=0", "model.dropout=1", "model.reg_weight=-1",
                     "model.damps_apc=1", "model.typo=1"):
        with pytest.raises(ValueError):
            load_config(model=name, dataset=dataset, overrides=[override])
