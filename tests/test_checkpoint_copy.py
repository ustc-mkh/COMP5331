import os

import pytest
import torch

from mmrecsys.experiment.artifacts import atomic_copy
from mmrecsys.experiment.runner import train_experiment


def test_atomic_copy_is_independent_and_replaces_destination(tmp_path):
    source, destination = tmp_path / 'best.pt', tmp_path / 'last.pt'
    source.write_bytes(b'new checkpoint')
    destination.write_bytes(b'old checkpoint')
    atomic_copy(source, destination)
    assert destination.read_bytes() == source.read_bytes()
    assert not os.path.samefile(source, destination)
    destination.write_bytes(b'later epoch')
    assert source.read_bytes() == b'new checkpoint'
    assert not list(tmp_path.glob('*.tmp'))


@pytest.mark.parametrize('failure', ['copy', 'replace'])
def test_failed_copy_keeps_existing_destination(tmp_path, monkeypatch, failure):
    from mmrecsys.experiment import artifacts

    source, destination = tmp_path / 'best.pt', tmp_path / 'last.pt'
    source.write_bytes(b'new checkpoint')
    destination.write_bytes(b'previous checkpoint')

    def fail(*args):
        raise OSError('simulated failure')

    if failure == 'copy':
        monkeypatch.setattr(artifacts.shutil, 'copyfile', fail)
    else:
        monkeypatch.setattr(artifacts.os, 'replace', fail)
    with pytest.raises(OSError, match='simulated failure'):
        atomic_copy(source, destination)
    assert destination.read_bytes() == b'previous checkpoint'
    assert source.read_bytes() == b'new checkpoint'
    assert not list(tmp_path.glob('*.tmp'))


def test_trainer_serializes_once_per_epoch(tiny_config, monkeypatch):
    from mmrecsys.engine import trainer

    original_save = trainer.save_checkpoint
    saved = []
    copied = []
    original_copy = trainer.atomic_copy

    def copy(source, destination):
        original_copy(source, destination)
        assert source.read_bytes() == destination.read_bytes()
        copied.append(destination.name)

    monkeypatch.setattr(trainer, 'atomic_copy', copy)

    def save(path, *args, **kwargs):
        saved.append(path.name)
        return original_save(path, *args, **kwargs)

    monkeypatch.setattr(trainer, 'save_checkpoint', save)
    # Force one improvement followed by a non-improvement.
    def evaluate(self, model, view):
        return {'Recall@3': 0.5}

    from mmrecsys.engine.evaluator import Evaluator
    monkeypatch.setattr(Evaluator, 'evaluate', evaluate)
    run, result = train_experiment(tiny_config)
    assert saved == ['best.pt', 'last.pt']
    assert copied == ['last.pt']
    assert result['best_epoch'] == 1
    best = torch.load(run / 'best.pt', weights_only=False)
    last = torch.load(run / 'last.pt', weights_only=False)
    assert best['state']['epoch'] == 1
    assert last['state']['epoch'] == 2
    assert not os.path.samefile(run / 'best.pt', run / 'last.pt')
