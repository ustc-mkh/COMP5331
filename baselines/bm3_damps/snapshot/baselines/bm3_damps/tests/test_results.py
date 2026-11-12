"""A complete main experiment is publishable independently; partial groups are not."""
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from results import DATASETS, SEEDS, METRICS, VARIANTS, derive


def observations(variants):
    return [dict(dataset=d,variant=v,seed=s,test={m:0.1+i*0.001 for m in METRICS})
            for d in DATASETS for v in variants for i,s in enumerate(SEEDS)]


def test_complete_main_without_ablations():
    variants=("baseline","full")
    rows=observations(variants)
    summary,contrasts=derive(rows,variants)
    assert len(summary)==40 and len(contrasts)==20
    assert len(rows)==30 and all(r["n_runs"]==3 for r in summary)
    assert all(r["n_pairs"]==3 for r in contrasts)
    assert all(r["paired_mean_delta"]==0 for r in contrasts)
    with pytest.raises(AssertionError):
        derive(rows[:-1],variants)
    with pytest.raises(AssertionError):
        derive(rows+rows[:1],variants)


def test_extra_seed_and_cancelled_ablations_are_not_publishable():
    rows=observations(VARIANTS)
    with pytest.raises(AssertionError):
        derive(rows+[dict(rows[0],seed=2026)])
    with pytest.raises(AssertionError):
        derive(rows+[dict(rows[0],variant="no_apc")])
    with pytest.raises(AssertionError):
        derive(rows,variants=("baseline","full","no_apc"))
