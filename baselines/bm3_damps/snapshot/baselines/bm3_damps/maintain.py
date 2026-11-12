"""Maintain this fixed experiment every 30 minutes; exit after verified exports."""
import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from run_suite import HERE, ROOT, digest, read, save
from select_main_three import processes, selected_plan


def now():
    return datetime.now(timezone.utc)


def experiment_processes(suite):
    """Match this Unix user's exact suite argument, never unrelated processes."""
    return [dict(pid=p["pid"],job=p["job"]) for p in processes(suite) if p["kind"]!="monitor"]


def foreign_gpu_processes(gpus):
    # Only called when this experiment has no remaining training processes.
    query = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
                           text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30, check=True)
    return [line.strip() for line in query.stdout.splitlines() if line.split(",")[0].strip() in gpus]


def verified_groups(plan, suite):
    counts = Counter()
    datasets = {"full":Counter()}
    verified = set()
    for job in plan["jobs"]:
        marker = suite/job["id"]/"verified.json"
        if not marker.exists():
            continue
        stored = read(marker)
        result, reloaded = stored["result"], stored["reloaded"]
        assert stored["id"] == job["id"] and result["config"] == job["config"]
        assert result["kind"] == reloaded["kind"] == "formal"
        assert result["test"] == reloaded["test"] and result["best_validation"] == reloaded["validation"]
        assert result["best_epoch"] == reloaded["epoch"]
        assert result["data_fingerprint"] == reloaded["data_fingerprint"] == job["baseline_fingerprint"]
        counts[job["config"]["damps_variant"]] += 1
        datasets[job["config"]["damps_variant"]][job["config"]["dataset"]] += 1
        verified.add(job["id"])
    return counts, datasets, verified


def may_restart(verified_count, processes, recovery_count, foreign_processes, source_matches, free_bytes):
    return (verified_count < 15 and not processes and recovery_count < 3 and not foreign_processes
            and source_matches and free_bytes >= 20*1024**3)


def export_group(stage, output, suite, maintenance):
    logpath = maintenance/("export-"+stage+".log")
    with logpath.open("a", buffering=1) as log:
        subprocess.run([sys.executable,str(HERE/"results.py"),"--suite",str(suite),"--output",str(output),
                        "--stage",stage],stdout=log,stderr=subprocess.STDOUT,check=True)
        subprocess.run([sys.executable,str(HERE/"plot_results.py"),"--output",str(output)],
                       stdout=log,stderr=subprocess.STDOUT,check=True)
    save(maintenance/("ready-"+stage+".json"),dict(ready_at=now().isoformat(),stage=stage,
          output=str(output),verified=True,plotted=True,uploaded=False,
          publication="Use existing group format; upload only full DAMPS and the matching three-seed BM3 baseline"))


def check(args, state):
    suite=args.suite.resolve()
    maintenance=suite.parent/"maintenance"
    plan=read(suite/"plan.json")
    assert plan["training_runs"]==15 and plan["jobs"]==selected_plan(plan)["jobs"]
    matches=plan["digest"]==digest()
    counts,datasets,verified=verified_groups(plan,suite)
    processes=experiment_processes(suite)
    live_jobs=[]
    for process in processes:
        if process["job"] is None:
            continue
        log=suite/process["job"]/"worker.log"
        lines=log.read_text(errors="replace").splitlines() if log.exists() else []
        progress=next((line for line in reversed(lines) if line.startswith("epoch=")),"loading or evaluating")
        age=time.time()-log.stat().st_mtime if log.exists() else None
        live_jobs.append(dict(process,progress=progress,log_age_seconds=age,stale=age is not None and age>7200))
    free=shutil.disk_usage(suite).free
    summary=dict(checked_at=now().isoformat(),next_check_at=(now()+timedelta(seconds=args.interval)).isoformat(),
                 monitor_pid=os.getpid(),interval_seconds=args.interval,total=15,verified=len(verified),
                 main_verified=counts["full"],main_total=15,seeds=[999,2024,2025],
                 ablations_cancelled=True,
                 by_variant=dict(counts),by_dataset={v:dict(c) for v,c in datasets.items()},
                 launcher_pids=[p["pid"] for p in processes if p["job"] is None],active=live_jobs,
                 source_matches=matches,free_disk_bytes=free,recoveries=state.get("recoveries",0),actions=[])
    queue=read(suite/"status.json")
    summary["failed_jobs"]=[j for j in queue.get("failed",[]) if j["id"] not in verified]
    if not processes and len(verified)<15:
        foreign=foreign_gpu_processes(args.gpus)
        summary["foreign_gpu_processes"]=foreign
        if may_restart(len(verified),processes,state.get("recoveries",0),foreign,matches,free):
            with (maintenance/"recovery-launcher.log").open("a",buffering=1) as log:
                child=subprocess.Popen([sys.executable,"-u",str(HERE/"run_selected.py"),"--output",str(suite),
                                        "--data",str(args.data.resolve()),"--gpus",*args.gpus,"--resume"],
                                       stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                       start_new_session=True,cwd=ROOT)
            state["recoveries"]=state.get("recoveries",0)+1
            summary["actions"].append(dict(action="resumed_interrupted_suite",pid=child.pid))
            save(maintenance/"state.json",state)
        else:
            summary["actions"].append(dict(action="recovery_waits_for_investigation_or_free_resources"))
    if matches:
        for stage,complete,destination in (("main",counts["full"]==15,"bm3-damps"),):
            if complete and not (maintenance/("ready-"+stage+".json")).exists():
                try:
                    export_group(stage,ROOT/"docs/results"/destination,suite,maintenance)
                    summary["actions"].append(dict(action="verified_and_plotted",stage=stage))
                except subprocess.CalledProcessError as exc:
                    summary["actions"].append(dict(action="export_needs_attention",stage=stage,exit_code=exc.returncode))
    summary["main_ready"]=(maintenance/"ready-main.json").exists()
    summary["complete"]=len(verified)==15 and summary["main_ready"]
    save(maintenance/"status.json",summary)
    with (maintenance/"history.jsonl").open("a") as stream:
        stream.write(json.dumps(summary,allow_nan=False)+"\n")
    print(json.dumps(summary,allow_nan=False),flush=True)
    return summary["complete"]


def main(args):
    import fcntl
    maintenance=args.suite.resolve().parent/"maintenance"
    maintenance.mkdir(exist_ok=True)
    lock=(maintenance/"monitor.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=read(maintenance/"state.json") if (maintenance/"state.json").exists() else {"recoveries":0}
    while True:
        try:
            complete=check(args,state)
        except Exception as exc:
            complete=False
            error=dict(checked_at=now().isoformat(),error=repr(exc),next_check_at=(now()+timedelta(seconds=args.interval)).isoformat())
            save(maintenance/"last-error.json",error)
            print(json.dumps(error),flush=True)
        if complete or args.once:
            break
        time.sleep(args.interval)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite",type=Path,required=True)
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--gpus",nargs="+",required=True)
    parser.add_argument("--interval",type=int,default=1800)
    parser.add_argument("--once",action="store_true")
    args=parser.parse_args()
    if args.interval<60:
        parser.error("Use at least 60 seconds between maintenance checks")
    main(args)
