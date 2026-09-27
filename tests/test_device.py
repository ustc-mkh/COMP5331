from contextlib import nullcontext
from unittest.mock import Mock

import pytest
import torch

from mmrecsys.experiment import runner, seed


def test_cpu_rng_never_initializes_cuda(monkeypatch):
    monkeypatch.setattr(torch, 'use_deterministic_algorithms', Mock())
    def unexpected(*args, **kwargs):
        pytest.fail('CPU RNG handling must not access CUDA')

    for name in ('is_available', 'manual_seed', 'manual_seed_all', 'get_rng_state',
                 'get_rng_state_all', 'set_rng_state', 'set_rng_state_all'):
        monkeypatch.setattr(torch.cuda, name, unexpected)
    seed.seed_everything(999, device='cpu')
    state = seed.random_state('cpu')
    expected = torch.rand(3)
    seed.restore_random_state(state, 'cpu')
    torch.testing.assert_close(torch.rand(3), expected, rtol=0, atol=0)
    # Old CPU runs could save all GPU states; ignore these when resuming on CPU.
    state['cuda'] = [torch.zeros(1, dtype=torch.uint8)]
    seed.restore_random_state(state, 'cpu')


@pytest.mark.parametrize('legacy', [False, True])
def test_cuda_rng_only_saves_and_restores_selected_gpu(monkeypatch, legacy):
    monkeypatch.setattr(torch, 'use_deterministic_algorithms', Mock())
    device = torch.device('cuda:2')
    value = torch.tensor([1, 2, 3], dtype=torch.uint8)
    get_state, set_state, manual_seed = Mock(return_value=value), Mock(), Mock()
    monkeypatch.setattr(torch.cuda, 'get_rng_state', get_state)
    monkeypatch.setattr(torch.cuda, 'set_rng_state', set_state)
    monkeypatch.setattr(torch.cuda, 'manual_seed', manual_seed)
    context = Mock(return_value=nullcontext())
    monkeypatch.setattr(torch.cuda, 'device', context)

    def unexpected(*args, **kwargs):
        pytest.fail('Single-GPU runs must not access every CUDA generator')

    for name in ('manual_seed_all', 'get_rng_state_all', 'set_rng_state_all'):
        monkeypatch.setattr(torch.cuda, name, unexpected)
    seed.seed_everything(999, device=device)
    context.assert_called_once_with(device)
    manual_seed.assert_called_once_with(999)
    state = seed.random_state(device)
    get_state.assert_called_once_with(device)
    if legacy:
        state['cuda'] = [torch.zeros_like(value), torch.zeros_like(value), value]
    seed.restore_random_state(state, device)
    set_state.assert_called_once()
    restored_value, restored_device = set_state.call_args.args
    assert restored_device == device
    torch.testing.assert_close(restored_value, value)


def test_assemble_selects_cuda_before_seed_or_model_initialization(tiny_config, monkeypatch):
    events = []
    tiny_config['runtime']['device'] = 'cuda:2'
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'set_device', lambda device: events.append(('device', str(device))))
    monkeypatch.setattr(runner, 'seed_everything', lambda value, deterministic, device: events.append(('seed', str(device))))

    class ReachedDataLoading(Exception):
        pass

    def stop_before_loading(*args):
        assert events == [('device', 'cuda:2'), ('seed', 'cuda:2')]
        raise ReachedDataLoading

    monkeypatch.setattr(runner, 'load_dataset', stop_before_loading)
    with pytest.raises(ReachedDataLoading):
        runner.assemble(tiny_config)
