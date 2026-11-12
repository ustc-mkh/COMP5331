"""Apply the user's scope change, preserving desired workers and all old evidence."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import time

from run_suite import read, save

SEEDS = (999, 2024, 2025)
DATASETS = ("baby", "sports", "clothing", "elec", "microlens")


def selected_plan(plan):
    jobs=[j for j in plan["jobs"] if j["config"]["damps_variant"]=="full" and j["config"]["seed"] in SEEDS]
    assert len(jobs)==15
    assert {(j["config"]["dataset"],j["config"]["seed"]) for j in jobs}=={(d,s) for d in DATASETS for s in SEEDS}
    return dict(plan,training_runs=15,raw_baseline_runs=15,seeds=list(SEEDS),variants=["full"],jobs=jobs,
                scope="BM3+DAMPS only; three requested seeds; no ablations")


def processes(suite):
    found=[]
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit(): continue
        try:
            if entry.stat().st_uid!=os.getuid(): continue
            args=[v.decode() for v in (entry/"cmdline").read_bytes().split(b"\0") if v]
        except (FileNotFoundError,PermissionError,ProcessLookupError,UnicodeError): continue
        if str(suite) not in args: continue
        names={Path(v).name for v in args}
        if not names.intersection({"run_suite.py","maintain.py","run_selected.py"}): continue
        job=args[args.index("--worker")+1] if "--worker" in args else None
        kind="worker" if job else "monitor" if "maintain.py" in names else "launcher"
        found.append(dict(pid=int(entry.name),kind=kind,job=job))
    return found


def terminate(record, suite):
    # Re-check identity immediately before signalling; never signal a recycled PID.
    if record not in processes(suite): return
    os.kill(record["pid"],signal.SIGTERM)
    deadline=time.monotonic()+5
    while time.monotonic()<deadline and record in processes(suite): time.sleep(.1)
    if record in processes(suite): raise RuntimeError("Process did not stop: "+str(record))


def apply(suite):
    suite=suite.resolve()
    original=(suite/"plan.json").read_bytes()
    plan=json.loads(original)
    assert plan["training_runs"] in (100,15)
    updated=selected_plan(plan)
    allowed={j["id"] for j in updated["jobs"]}
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive=suite.parent/"scope-history"/stamp
    archive.mkdir(parents=True,exist_ok=False)
    (archive/"plan-before.json").write_bytes(original)
    for path in (suite/"status.json",suite.parent/"maintenance/status.json",suite.parent/"maintenance/state.json"):
        if path.exists(): shutil.copyfile(path,archive/(path.parent.name+"-"+path.name))
    stopped=[]
    for kind in ("monitor","launcher"):
        for record in processes(suite):
            if record["kind"]==kind:
                terminate(record,suite); stopped.append(record)
    for record in processes(suite):
        if record["kind"]=="worker" and record["job"] not in allowed:
            terminate(record,suite); stopped.append(record)
    remaining=processes(suite)
    assert all(p["kind"]=="worker" and p["job"] in allowed for p in remaining)
    updated.update(scope_changed_at=datetime.now(timezone.utc).isoformat(),
                   prior_plan_sha256=hashlib.sha256(original).hexdigest(),
                   excluded_job_ids=[j["id"] for j in plan["jobs"] if j["id"] not in allowed])
    save(suite/"plan.json",updated)
    report=dict(changed_at=updated["scope_changed_at"],total=15,seeds=list(SEEDS),variant="full",
                ablations_cancelled=True,stopped=stopped,preserved_workers=remaining,
                verified=sum((suite/j["id"]/"verified.json").exists() for j in updated["jobs"]),
                archived_plan=str(archive/"plan-before.json"),all_existing_results_preserved=True)
    save(suite.parent/"scope-change.json",report)
    save(suite/"status.json",dict(updated=report["changed_at"],pid=None,total=15,verified=report["verified"],
                                 active={p["job"]:p for p in remaining},pending=15-report["verified"]-len(remaining),
                                 failed=[],status="restricted_queue_transition"))
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite",type=Path,required=True)
    apply(parser.parse_args().suite)
