"""Recovery must never duplicate live workers or take another user's busy GPU."""
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from maintain import may_restart


def test_idle_interrupted_suite_can_resume():
    assert may_restart(8, [], 0, [], True, 100*1024**3)


@pytest.mark.parametrize("change", [
    {"processes":[{"pid":123,"job":"baby-full-999"}]},
    {"processes":[{"pid":123,"job":None}]},
    {"foreign_processes":["GPU-owned-by-another-job, 456"]},
    {"source_matches":False},
    {"free_bytes":1024**3},
    {"recovery_count":3},
    {"verified_count":15},
])
def test_unsafe_or_unnecessary_restart_is_blocked(change):
    args=dict(verified_count=8,processes=[],recovery_count=0,foreign_processes=[],
              source_matches=True,free_bytes=100*1024**3)
    args.update(change)
    assert not may_restart(**args)
