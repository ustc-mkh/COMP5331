import sys
from pathlib import Path
import copy
import pytest
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "bm3/runtime/tests"))
from model import BM3Damps, RawBM3, VARIANTS
from test_pipeline import fixture_data, small_config
from bm3baseline.config import configure_process
from bm3baseline.data import inspect_data, runtime_config
from bm3baseline.model import make_loaders
from bm3baseline import engine


def construct(config, directory, output, variant):
    _, splits = inspect_data(directory, require_baby=False)
    config = dict(config, damps_variant=variant)
    runtime = runtime_config(config, directory, torch.device("cpu"), output)
    configure_process(config["seed"], "cpu", 1)
    loaders = make_loaders(runtime, splits)
    loaders[0].pretrain_setup()
    return BM3Damps(runtime, loaders[0]), loaders, runtime


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_gradients_width_priors_and_reload(fixture_data, small_config, tmp_path, variant):
    model, loaders, _ = construct(small_config, fixture_data, tmp_path, variant)
    prior = model.damps.phase_prior.clone() if model.damps.apc else None
    batch = next(iter(loaders[0]))
    before = {n: p.detach().clone() for n, p in model.damps.named_parameters()}
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    loss = model.calculate_loss(batch)
    loss.backward()
    for name, p in model.damps.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, name
    optimizer.step()
    assert any(not torch.equal(before[n], p) for n, p in model.damps.named_parameters())
    if prior is not None:
        assert torch.equal(prior, model.damps.phase_prior)
    user, item = model.forward()
    assert user.shape[1] == item.shape[1] == small_config["embedding_size"]
    replica, _, _ = construct(small_config, fixture_data, tmp_path, variant)
    replica.load_state_dict(model.state_dict())
    model.eval(); replica.eval()
    assert torch.equal(model.full_sort_predict(batch), replica.full_sort_predict(batch))
    assert BM3Damps.forward is RawBM3.forward
    assert BM3Damps.full_sort_predict is RawBM3.full_sort_predict


def test_baseline_unchanged_and_paired_initialization(fixture_data, small_config, tmp_path):
    baseline, loaders, runtime = construct(small_config, fixture_data, tmp_path, "baseline")
    configure_process(small_config["seed"], "cpu", 1)
    raw = RawBM3(runtime, loaders[0])
    assert all(torch.equal(v, raw.state_dict()[k]) for k,v in baseline.state_dict().items())
    batch = next(iter(loaders[0]))
    torch.manual_seed(12); expected = raw.calculate_loss(batch)
    torch.manual_seed(12); actual = baseline.calculate_loss(batch)
    assert torch.equal(expected, actual)
    for variant in VARIANTS:
        candidate, _, _ = construct(small_config, fixture_data, tmp_path, variant)
        assert all(torch.equal(v, candidate.state_dict()[k]) for k,v in raw.state_dict().items())


def test_new_model_checkpoint_resume_exact(fixture_data, small_config, tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "BM3", BM3Damps)
    config = dict(small_config, damps_variant="full", checkpoint_every=1, epochs=3)
    direct = engine.train(config, fixture_data, tmp_path/"direct", require_baby=False)
    engine.train(dict(config, epochs=1), fixture_data, tmp_path/"resumed", require_baby=False)
    resumed = engine.train(config, fixture_data, tmp_path/"resumed", require_baby=False,
                           resume=tmp_path/"resumed/last.pt")
    for key in ("best_epoch", "best_validation", "test"):
        assert direct[key] == resumed[key]
    reloaded = engine.evaluate_checkpoint(tmp_path/"direct/best.pt", fixture_data, require_baby=False)
    assert reloaded["test"] == direct["test"]
