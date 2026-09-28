import json
from pathlib import Path

import pytest

from mmrecsys.engine.evaluator import Evaluator
from mmrecsys.experiment import search


def test_grid_and_gain_definition(tiny_config):
    trials = search.candidates(tiny_config, {"model.n_ui_layers": [1, 2], "model.cl_weight": [.01, .02]}, [999, 2024])
    assert len(trials) == 4
    for grid in ({"model.damps_enabled": [True]}, {"optimizer.lr": []}, {"model.cl_weight": [.01, .01]}):
        with pytest.raises(ValueError):
            search.candidates(tiny_config, grid, [999])
    pairs = [{"baseline": {"validation": .1}, "full": {"validation": .12}},
             {"baseline": {"validation": .2}, "full": {"validation": .21}}]
    record = search.score_trial({}, pairs, "relative", 0)
    assert record["score"] == pytest.approx(12.5)  # mean of 20% and 5%, not ratio of means
    assert record["absolute_gain_mean"] == pytest.approx(.015)
    assert record["positive_seed_count"] == 2
    assert not search.score_trial({}, pairs, "relative", .16)["eligible"]
    pairs[0]["baseline"]["validation"] = 0
    assert not search.score_trial({}, pairs, "relative", 0)["eligible"]
    assert search.score_trial({}, pairs, "absolute", 0)["eligible"]


def test_search_validation_only_and_completed_resume(tiny_config, monkeypatch):
    splits = []
    original = Evaluator.evaluate
    def observe(self, model, view):
        splits.append(view.split)
        assert view.split == 'valid', 'Candidate training must never evaluate test labels'
        return original(self, model, view)
    monkeypatch.setattr(Evaluator, 'evaluate', observe)
    output = search.run_search(tiny_config, {"model.cl_weight": [.01, .02]}, [999], objective='absolute')
    ranked = json.loads((output / 'leaderboard.json').read_text())
    best = json.loads((output / 'best.json').read_text())
    assert len(ranked) == 2 and ranked[0] == best
    assert best['score'] == max(r['score'] for r in ranked)
    assert splits and set(splits) == {'valid'}
    assert (output / 'best_full.yaml').exists()
    for record in ranked:
        for variant in ('baseline', 'full'):
            path = Path(record['pairs'][0][variant]['run_dir'])
            assert (path / 'validation_result.json').exists()
            assert not (path / 'result.json').exists()
    def unexpected(*args, **kwargs):
        raise AssertionError('Completed candidates must not be retrained')
    monkeypatch.setattr(search, 'train_experiment', unexpected)
    assert search.run_search(resume=output) == output
    assert json.loads((output / 'best.json').read_text()) == best
    monkeypatch.setattr(Evaluator, 'evaluate', original)
    search.run_search(resume=output, evaluate_best=True)
    report = json.loads((output / 'best_test.json').read_text())
    assert report['settings'] == best['settings']
    assert set(report['tests'][0]) == {'seed', 'baseline', 'full'}


def test_search_recovers_interrupted_run(tiny_config, monkeypatch):
    original = search.train_experiment
    interrupted = []
    def interrupt_full(config, resume=None, **kwargs):
        if config['model']['damps_enabled']:
            # Create a real resumable checkpoint, then emulate interruption before result publication.
            run, _ = original(config, resume=resume, **kwargs)
            (run / 'validation_result.json').unlink()
            interrupted.append(run)
            raise RuntimeError('simulated interruption')
        return original(config, resume=resume, **kwargs)
    monkeypatch.setattr(search, 'train_experiment', interrupt_full)
    with pytest.raises(RuntimeError, match='simulated'):
        search.run_search(tiny_config, {'model.cl_weight': [.01]}, [999], objective='absolute')
    output = next(Path(tiny_config['runtime']['output_root']).glob('search-*'))
    resumed = []
    def observe_resume(config, resume=None, **kwargs):
        resumed.append(resume)
        return original(config, resume=resume, **kwargs)
    monkeypatch.setattr(search, 'train_experiment', observe_resume)
    search.run_search(resume=output)
    assert resumed == [interrupted[0] / 'last.pt']
    assert (output / 'best.json').exists()
    monkeypatch.setattr(search, 'code_digest', lambda: 'changed')
    with pytest.raises(ValueError, match='source code changed'):
        search.run_search(resume=output)
