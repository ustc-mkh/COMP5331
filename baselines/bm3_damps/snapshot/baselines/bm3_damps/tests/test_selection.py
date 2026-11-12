"""The scope change must preserve completed-run/checkpoint compatibility."""
from pathlib import Path
import sys
import copy

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_suite import make_plan
from select_main_three import selected_plan, DATASETS, SEEDS


def test_selects_only_requested_full_runs_without_changing_training(tmp_path):
    original=make_plan(tmp_path,tmp_path/"data")
    before=copy.deepcopy(original)
    chosen=selected_plan(original)
    assert original==before and len(original["jobs"])==100
    assert chosen["training_runs"]==15 and chosen["raw_baseline_runs"]==15
    assert chosen["seeds"]==[999,2024,2025] and chosen["variants"]==["full"]
    assert chosen["digest"]==original["digest"]
    lookup={j["id"]:j for j in original["jobs"]}
    for job in chosen["jobs"]:
        assert job==lookup[job["id"]]
        assert job["config"]["damps_variant"]=="full"
    assert {(j["config"]["dataset"],j["config"]["seed"]) for j in chosen["jobs"]}=={(d,s) for d in DATASETS for s in SEEDS}
