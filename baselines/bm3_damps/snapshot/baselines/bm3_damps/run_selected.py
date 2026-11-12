"""Run only the 15 requested full-DAMPS jobs, adopting existing desired workers."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from run_suite import HERE, ROOT, digest, make_plan, now, read, save
from select_main_three import processes, selected_plan


def live_workers(suite):
    workers=[]
    for record in processes(suite):
        if record["kind"]!="worker": continue
        try:
            entries=(Path("/proc")/str(record["pid"])/"environ").read_bytes().split(b"\0")
        except (FileNotFoundError,ProcessLookupError): continue
        gpu=next(v.split(b"=",1)[1].decode() for v in entries if v.startswith(b"CUDA_VISIBLE_DEVICES="))
        workers.append(dict(record,gpu=gpu))
    return workers


def gpu_processes():
    query=subprocess.run(["nvidia-smi","--query-compute-apps=gpu_uuid,pid","--format=csv,noheader,nounits"],
                         stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=30,check=True)
    return {line.split(",")[0].strip() for line in query.stdout.splitlines() if "," in line}


def launch(args):
    import fcntl
    suite=args.output.resolve()
    if args.resume: assert (suite/"plan.json").exists(), "Resume requires an existing plan"
    suite.mkdir(parents=True,exist_ok=True)
    lock=(suite/"launcher.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (suite/"plan.json").exists():
        plan=read(suite/"plan.json")
    else:
        assert args.data is not None, "Specify --data for a new suite"
        plan=selected_plan(make_plan(suite,args.data.resolve()))
        save(suite/"plan.json",plan)
    assert plan["training_runs"]==15 and plan["jobs"]==selected_plan(plan)["jobs"]
    assert plan["digest"]==digest(),"Frozen training source changed"
    assert len(set(args.gpus))==len(args.gpus)
    assert all(g.startswith("GPU-") for g in args.gpus), "Use GPU UUIDs for unambiguous ownership checks"
    jobs={j["id"]:j for j in plan["jobs"]}
    active,children,attempts,failed={},{},{},[]
    for existing in live_workers(suite):
        assert existing["job"] in jobs and existing["gpu"] in args.gpus
        assert existing["gpu"] not in active
        active[existing["gpu"]]=dict(id=existing["job"],pid=existing["pid"],adopted=True)
        attempts[existing["job"]]=1
        print(now(),"ADOPT",existing["job"],existing["pid"],flush=True)
    active_ids={v["id"] for v in active.values()}
    pending=[j["id"] for j in plan["jobs"] if not (suite/j["id"]/"verified.json").exists() and j["id"] not in active_ids]
    while pending or active:
        for pid,(child,log) in list(children.items()):
            if child.poll() is not None:
                log.close(); del children[pid]
        living={p["pid"] for p in live_workers(suite)}
        for gpu,job in list(active.items()):
            if job["pid"] in living: continue
            ok=(suite/job["id"]/"verified.json").exists()
            del active[gpu]
            print(now(),"FINISH",job["id"],"verified",ok,flush=True)
            if not ok:
                if attempts[job["id"]]<2: pending.append(job["id"])
                else: failed.append(dict(id=job["id"],reason="worker exited without verified result"))
        occupied=gpu_processes() if pending and len(active)<len(args.gpus) else set()
        for gpu in args.gpus:
            if gpu in active or gpu in occupied or not pending: continue
            job_id=pending.pop(0)
            if (suite/job_id/"verified.json").exists(): continue
            job=jobs[job_id]
            directory=suite/job_id; directory.mkdir(exist_ok=True)
            log=(directory/"worker.log").open("a",buffering=1)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,PYTHONHASHSEED=str(job["config"]["seed"]),
                     PYTHONUNBUFFERED="1",OMP_NUM_THREADS="4",MKL_NUM_THREADS="4")
            child=subprocess.Popen([sys.executable,str(HERE/"run_suite.py"),"--output",str(suite),"--worker",job_id],
                                   stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=HERE)
            active[gpu]=dict(id=job_id,pid=child.pid,adopted=False)
            children[child.pid]=(child,log)
            attempts[job_id]=attempts.get(job_id,0)+1
            print(now(),"START",job_id,"pid",child.pid,"gpu",gpu,flush=True)
        status=dict(updated=now(),pid=os.getpid(),total=15,seeds=[999,2024,2025],variants=["full"],
                    verified=sum((suite/j/"verified.json").exists() for j in jobs),
                    pending=len(pending),active=active,failed=failed,scope="full DAMPS only; ablations cancelled")
        save(suite/"status.json",status)
        if pending or active: time.sleep(10)
    for child,log in children.values(): child.wait(); log.close()
    status.update(finished=now(),success=not failed)
    save(suite/"status.json",status)
    return 1 if failed else 0


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--data",type=Path)  # Accepted by the maintenance restart command.
    parser.add_argument("--gpus",nargs="+",required=True)
    parser.add_argument("--resume",action="store_true")
    raise SystemExit(launch(parser.parse_args()))
