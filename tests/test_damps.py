from dataclasses import replace
from itertools import product

import numpy as np
import pytest
import torch

from mmrecsys.config import load_config
from mmrecsys.data.views import TrainBatch
from mmrecsys.experiment.artifacts import GraphCache
from mmrecsys.models.mgcn import MGCN, MGCNConfig
from mmrecsys.nn.damps import DAMPS


def numpy_reference(image, text, residual, logits, prior, masks):
    """Independent forward reference with fixed prior and learned masks."""
    zi, zt = np.fft.rfft(image, norm="ortho"), np.fft.rfft(text, norm="ortho")
    zi *= np.exp(-1j * (prior / 2 + residual))
    zt *= np.exp(1j * (prior / 2 + residual))
    weights = np.exp(logits) / np.exp(logits).sum()
    coherence = np.abs(zt * zi.conj()) ** 2 / (np.abs(zt) ** 2 * np.abs(zi) ** 2 + 1e-8)
    results = []
    for z, vrf in zip((zi, zt), masks):
        results.append(np.fft.irfft((weights[0] * vrf + weights[1] * coherence) * z, n=image.shape[1], norm="ortho"))
    return results


@pytest.mark.parametrize('dim', [5, 6])
def test_equations_against_numpy(dim):
    rng = np.random.default_rng(913)
    image, text = (rng.normal(size=(8, dim)) for _ in range(2))
    module = DAMPS(dim).double()
    module.initialize(torch.tensor(image), torch.tensor(text))
    with torch.no_grad():
        module.phase_residual.copy_(torch.linspace(-0.1, 0.2, dim // 2 + 1))
        module.mix_logits.copy_(torch.tensor([0.4, -0.3]))
    expected = numpy_reference(image, text, module.phase_residual.detach().numpy(),
                               module.mix_logits.detach().numpy(), module.phase_prior.numpy(),
                               [module.avrf_image.detach().numpy(), module.avrf_text.detach().numpy()])
    actual = module(torch.tensor(image), torch.tensor(text))
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, torch.tensor(reference), rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("offset", [0.4, -0.4, 2.9, -2.9])
def test_author_phase_sign_and_coherence_epsilon(offset):
    module = DAMPS(4).double()
    image = torch.ones(7, 3, dtype=torch.complex128)
    text = image * np.exp(offset * 1j)
    module.phase_prior.fill_(offset)
    zi, zt = module.calibrate_phase(image, text)
    torch.testing.assert_close(torch.angle(zt * zi.conj()), torch.full((7, 3), np.angle(np.exp(2j * offset)), dtype=torch.float64))
    torch.testing.assert_close(module.coherence_filter(zi, zt), torch.full((7, 3), 1 / (1 + 1e-8), dtype=torch.float64))
    torch.testing.assert_close(zi.abs(), image.abs())
    torch.testing.assert_close(zt.abs(), text.abs())
    with torch.no_grad():
        module.phase_residual.fill_(0.1)
    ri, rt = module.calibrate_phase(image, text)
    torch.testing.assert_close(torch.angle(rt * ri.conj()), torch.full((7, 3), np.angle(np.exp(1j * (2 * offset + 0.2))), dtype=torch.float64))
    weak = image * 1e-2
    torch.testing.assert_close(module.coherence_filter(weak, weak), torch.full((7, 3), 0.5, dtype=torch.float64))
    assert not module.coherence_filter(image * 0, text).any()
    torch.testing.assert_close(DAMPS(4).mix_logits.softmax(0), torch.tensor([0.549834, 0.450166]))


def test_initialization_matches_author_statistics_and_is_fixed():
    torch.manual_seed(41)
    image, text = torch.randn(12, 6), torch.randn(12, 6)
    module = DAMPS(6)
    module.initialize(image, text)
    for raw, weight in ((image, module.avrf_image), (text, module.avrf_text)):
        # Independent NumPy reference: lower median and sample std across bins.
        mag = torch.fft.rfft(raw, norm='ortho').abs().double().numpy()
        median = np.sort(mag, axis=0)[(len(mag) - 1) // 2]
        mad = np.sort(np.abs(mag - median), axis=0)[(len(mag) - 1) // 2]
        noise = (1.4826 * mad) ** 2
        signal = np.maximum(mag.var(axis=0) - noise, 0)
        vr = (signal / (signal + noise + 1e-6)).astype(np.float32)
        probability = 1 / (1 + np.exp(-(vr - vr.mean()) / (vr.std(ddof=1) + 1e-6)))
        expected = np.log(probability / (1 - probability + 1e-8))
        torch.testing.assert_close(weight, torch.from_numpy(expected), atol=1e-6, rtol=1e-5)
    phase = torch.angle(torch.fft.rfft(text, norm='ortho')) - torch.angle(torch.fft.rfft(image, norm='ortho'))
    torch.testing.assert_close(module.phase_prior, torch.atan2(phase.sin().mean(0), phase.cos().mean(0)))
    prior = module.phase_prior.clone()
    old_weight = module.avrf_image.detach().clone()
    optimizer = torch.optim.Adam(module.parameters(), lr=0.01)
    out = module(image * 2, text + 1)
    sum(x.square().mean() for x in out).backward()
    assert module.avrf_image.grad.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(old_weight, module.avrf_image)
    torch.testing.assert_close(module.phase_prior, prior, rtol=0, atol=0)
    assert not module.phase_prior.requires_grad
    restored = DAMPS(6)
    restored.load_state_dict(module.state_dict())
    for actual, expected in zip(restored(image, text), module(image, text)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    with pytest.raises(RuntimeError, match='already'):
        module.initialize(image, text)
    with pytest.raises(RuntimeError, match='Initialize'):
        DAMPS(6)(image, text)


@pytest.mark.parametrize('flags', list(product([False, True], repeat=3)))
@pytest.mark.parametrize('kind', ['random', 'zero', 'constant', 'single'])
def test_ablations_and_degenerate_inputs_have_finite_gradients(flags, kind):
    torch.manual_seed(7)
    module = DAMPS(6, apc=flags[0], avrf=flags[1], imcf=flags[2]).double()
    image, text = torch.randn(9, 6, dtype=torch.float64), torch.randn(9, 6, dtype=torch.float64)
    if kind == 'zero':
        image.zero_()
        text.zero_()
    elif kind == 'constant':
        image.fill_(2)
        text.fill_(3)
    elif kind == 'single':
        image, text = image[:1], text[:1]
    module.initialize(image, text)
    image.requires_grad_()
    text.requires_grad_()
    outputs = module(image, text)
    sum(x.square().mean() for x in outputs).backward()
    for value in (image, text, *module.parameters()):
        assert value.grad is not None and torch.isfinite(value.grad).all()
    if not any(flags):
        assert outputs[0] is image and outputs[1] is text


def test_mgcn_integration_and_internal_projection_bypass(tiny_data, tiny_config, tmp_path):
    base = MGCNConfig.parse(tiny_config['model'])
    torch.manual_seed(10)
    baseline = MGCN(base, tiny_data[0], GraphCache(tmp_path))
    torch.manual_seed(10)
    bypass = MGCN(replace(base, damps_enabled=True, damps_apc=False, damps_avrf=False, damps_imcf=False),
                  tiny_data[0], GraphCache(tmp_path))
    dummy = torch.zeros(tiny_data[0].n_items, base.embedding_dim)
    for actual, expected in zip(bypass.damps(dummy, dummy), bypass.damps.project_features()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    model = MGCN(replace(base, damps_enabled=True), tiny_data[0], GraphCache(tmp_path))
    batch = TrainBatch(torch.tensor([0, 1, 2]), torch.tensor([0, 1, 3]), torch.tensor([[4], [5], [0]]))
    output = model.compute_loss(batch)
    output.total.backward()
    for name, parameter in model.named_parameters():
        if name.startswith(("feature_embeddings.", "projections.")):
            assert parameter.grad is None, name
        else:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    for parameter in (model.damps.phase_residual, model.damps.mix_logits, model.damps.avrf_image, model.damps.avrf_text, model.damps.image_trs.weight, model.damps.image_embedding.weight):
        assert parameter.grad.abs().sum() > 0
    model.eval()
    users, items, _, _ = model.encode()
    torch.testing.assert_close(model.make_scorer().score(batch.users, batch.positive_items),
                               users[batch.users] @ items[batch.positive_items].T)


def test_experiment_configs_and_validation():
    for dataset in ('baby', 'sports', 'clothing', 'elec'):
        config = load_config(f'configs/experiments/damps_mgcn_{dataset}.yaml')
        assert config['model']['damps_enabled'] and config['data']['name'] == dataset
    for override in ('model.damps_enabled=1', 'model.damps_apc=nope', 'model.damps_eps=0', 'model.damps_eps=.nan'):
        with pytest.raises(ValueError):
            load_config(overrides=[override])


@pytest.mark.parametrize("flags", list(product([False, True], repeat=3)))
def test_variants_preserve_backbone_initialization_and_sampling(tiny_data, tiny_config, tmp_path, flags):
    from mmrecsys.data.sampling import TrainingSampler, BatchSpec
    base = MGCNConfig.parse(tiny_config["model"])
    torch.manual_seed(999)
    baseline = MGCN(base, tiny_data[0], GraphCache(tmp_path))
    torch.manual_seed(999)
    variant = MGCN(replace(base, damps_enabled=True, damps_apc=flags[0],
                           damps_avrf=flags[1], damps_imcf=flags[2]), tiny_data[0], GraphCache(tmp_path))
    for name, parameter in baseline.named_parameters():
        torch.testing.assert_close(dict(variant.named_parameters())[name], parameter, atol=0, rtol=0)
    first = next(TrainingSampler(tiny_data[0], BatchSpec("pairwise"), 4, 999).batches(1))
    second = next(TrainingSampler(tiny_data[0], BatchSpec("pairwise"), 4, 999).batches(1))
    assert torch.equal(first.negative_items, second.negative_items)
    assert torch.equal(first.users, second.users)
    assert torch.equal(first.positive_items, second.positive_items)


def test_diagnostics_are_detached_and_do_not_change_training(tiny_data, tiny_config, tmp_path):
    from copy import deepcopy
    config = replace(MGCNConfig.parse(tiny_config["model"]), damps_enabled=True)
    plain = MGCN(config, tiny_data[0], GraphCache(tmp_path))
    observed = deepcopy(plain)
    observed.on_epoch_start(1)
    batch = TrainBatch(torch.tensor([0, 1]), torch.tensor([0, 1]), torch.tensor([[4], [5]]))
    for model in (plain, observed):
        model.compute_loss(batch).total.backward()
    diagnostics = observed.training_diagnostics()
    assert "gate/image/saturated_fraction" in diagnostics
    assert "gradient/damps/avrf_image/l2" in diagnostics
    assert all(not x.requires_grad and torch.isfinite(x) for x in diagnostics.values())
    for x, y in zip(plain.parameters(), observed.parameters()):
        torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)


@pytest.mark.parametrize("trainable", [False, True])
def test_owned_features_update_and_checkpoint(tiny_data, tiny_config, tmp_path, trainable):
    config = replace(MGCNConfig.parse(tiny_config["model"]), damps_enabled=True,
                     trainable_features=trainable)
    model = MGCN(config, tiny_data[0], GraphCache(tmp_path))
    assert model.damps.image_embedding.weight.requires_grad == trainable
    if trainable:
        assert model.damps.image_embedding.weight is not model.feature_embeddings["image"].weight
        assert model.damps.image_embedding.weight.data_ptr() == model.feature_embeddings["image"].weight.data_ptr()
    prior = model.damps.phase_prior.clone()
    old = model.damps.image_trs.weight.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    batch = TrainBatch(torch.tensor([0, 1]), torch.tensor([0, 1]), torch.tensor([[4], [5]]))
    model.compute_loss(batch).total.backward()
    optimizer.step()
    assert not torch.equal(old, model.damps.image_trs.weight)
    torch.testing.assert_close(prior, model.damps.phase_prior, rtol=0, atol=0)
    restored = MGCN(config, tiny_data[0], GraphCache(tmp_path))
    restored.load_state_dict(model.state_dict())
    for actual, expected in zip(restored.encode(), model.encode()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    legacy = {k: v for k, v in model.state_dict().items()
              if not k.startswith(("damps.image_", "damps.text_"))}
    with pytest.raises(RuntimeError, match="Missing key"):
        restored.load_state_dict(legacy)


def test_owned_features_match_author_source():
    import importlib.util
    from pathlib import Path
    source = Path("KDD2026_DAMPS-v1.0.0/Wmhwxl-KDD2026_DAMPS-af90958/src/models/damps.py")
    if not source.exists():
        pytest.skip("Author release is not installed")
    spec = importlib.util.spec_from_file_location("author_damps", source)
    author = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(author)
    image, text = torch.randn(12, 9), torch.randn(12, 7)
    torch.manual_seed(42)
    expected = author.DAMPS(6, torch.device("cpu"), image.clone(), text.clone())
    torch.manual_seed(42)
    actual = DAMPS(6, raw_image=image.clone(), raw_text=text.clone())
    torch.testing.assert_close(actual.phase_prior, expected.avg_R)
    torch.testing.assert_close(actual.avrf_image, expected.AVRF_image)
    torch.testing.assert_close(actual.avrf_text, expected.AVRF_txt)
    with torch.no_grad():
        actual.phase_residual.uniform_(-0.2, 0.2)
        expected.psi.copy_(actual.phase_residual)
    dummy = torch.randn(12, 6, requires_grad=True)
    for left, right in zip(actual(dummy, dummy), expected(dummy, dummy)):
        torch.testing.assert_close(left, right)
    sum(x.square().sum() for x in actual(dummy, dummy)).backward()
    sum(x.square().sum() for x in expected(dummy, dummy)).backward()
    assert dummy.grad is None
    for name in ("image_embedding.weight", "text_embedding.weight", "image_trs.weight", "text_trs.weight"):
        torch.testing.assert_close(dict(actual.named_parameters())[name].grad,
                                   dict(expected.named_parameters())[name].grad)


def test_corrected_rotation_checkpoint_rejected():
    module = DAMPS(6)
    state = module.state_dict()
    state["implementation_version"] = torch.tensor(4)
    with pytest.raises(RuntimeError, match="phase-rotation version mismatch"):
        DAMPS(6).load_state_dict(state)


def test_owned_damps_skips_unused_backbone_projections(tiny_data, tiny_config, tmp_path):
    config = replace(MGCNConfig.parse(tiny_config["model"]), damps_enabled=True)
    model = MGCN(config, tiny_data[0], GraphCache(tmp_path))
    calls = []
    handles = [projection.register_forward_hook(lambda *args: calls.append(True))
               for projection in model.projections.values()]
    batch = TrainBatch(torch.tensor([0, 1]), torch.tensor([0, 1]), torch.tensor([[4], [5]]))
    model.compute_loss(batch).total.backward()
    for handle in handles:
        handle.remove()
    assert not calls
    assert model.damps.image_trs.weight.grad is not None
    assert all(p.grad is None for p in model.projections.parameters())
    dummy = torch.zeros(model.n_items, config.embedding_dim)
    for actual, expected in zip(model.damps(), model.damps(dummy, dummy)):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
