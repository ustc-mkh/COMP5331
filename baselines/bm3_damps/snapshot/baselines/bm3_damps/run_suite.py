"""Five datasets x five seeds x four DAMPS variants, reusing verified raw BM3."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "bm3/runtime"))


def read(path):
    return json.loads(Path(path).read_text())


def save(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")
    tmp.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def digest():
    h = hashlib.sha256()
    paths = sorted((HERE.parent/"bm3/runtime/bm3baseline").glob("*.py"))
    paths += [HERE/"model.py", HERE/"damps.py", Path(__file__).resolve()]
    for p in paths:
        h.update(p.relative_to(ROOT).as_posix().encode()); h.update(p.read_bytes())
    return h.hexdigest()


def make_plan(output, data, epochs=None):
    raw = read(ROOT/"docs/results/bm3/runs.json")
    assert raw["complete"] and raw["n_runs"] == 25
    jobs = []
    # Finish complete DAMPS first, then component-removal ablations.
    for variant in ("full", "no_apc", "no_avrf", "no_imcf"):
        for run in raw["runs"]:
            config = dict(run["config"])
            config.update(damps_variant=variant, integration="projected_trainable_v1", extension_digest=digest())
            if epochs is not None:
                config["epochs"] = epochs
            dataset, seed = config["dataset"], config["seed"]
            jobs.append({"id": "{}-{}-{}".format(dataset, variant, seed), "config": config,
                         "data": str(data/dataset), "baseline_fingerprint": run["data_fingerprint"]})
    return {"created": now(), "digest": digest(), "integration": "Trainable DAMPS on projected BM3 features",
            "source_commit": "13691bf6016939f68bf487ca547febca789f67e7", "training_runs": len(jobs),
            "raw_baseline_runs": 25, "author_reference_tag": "af90958f5fb43214caeda9729fefd5c9044e26ac", "jobs": jobs}


def worker(output, job_id):
    from model import BM3Damps
    from bm3baseline import engine
    engine.BM3 = BM3Damps
    plan = read(output/"plan.json")
    if plan["digest"] != digest():
        raise ValueError("Frozen experiment source changed")
    job = next(j for j in plan["jobs"] if j["id"] == job_id)
    directory = output/job_id/"run"
    if (directory/"result.json").exists():
        result = read(directory/"result.json")
    else:
        resume = directory/"last.pt" if (directory/"last.pt").exists() else None
        if directory.exists() and resume is None:
            directory.rename(directory.with_name("incomplete-"+str(time.time_ns())))
        result = engine.train(job["config"], job["data"], directory, "cuda", resume=resume)
    if result["data_fingerprint"] != job["baseline_fingerprint"]:
        raise ValueError("Data differ from original baseline")
    reloaded = engine.evaluate_checkpoint(directory/"best.pt", job["data"], "cuda", output=directory/"reloaded.json")
    assert result["best_epoch"] == reloaded["epoch"]
    assert result["best_validation"] == reloaded["validation"] and result["test"] == reloaded["test"]
    metadata = read(directory/"metadata.json")
    metadata.update(protocol="Original BM3 training/evaluation with trainable projected DAMPS", extension_digest=plan["digest"],
                    damps_variant=job["config"]["damps_variant"], integration="projected_trainable_v1")
    save(directory/"metadata.json", metadata)
    save(output/job_id/"verified.json", {"id": job_id, "completed": now(), "result": result, "reloaded": reloaded})


def launch(args):
    import fcntl
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    lock = (output/"launcher.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (output/"plan.json").exists():
        assert args.resume, "Use --resume for an existing suite"
        plan = read(output/"plan.json")
        assert plan["digest"] == digest(), "Source changed"
    else:
        assert not args.resume
        plan = make_plan(output, args.data.resolve())
        save(output/"plan.json", plan)
    pending = [j for j in plan["jobs"] if not (output/j["id"]/"verified.json").exists()]
    active, failed, attempts = {}, [], {}
    while pending or active:
        for gpu in args.gpus:
            if gpu in active or not pending: continue
            job = pending.pop(0); directory = output/job["id"]; directory.mkdir(exist_ok=True)
            attempts[job["id"]] = attempts.get(job["id"], 0)+1
            log = (directory/"worker.log").open("a", buffering=1)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONHASHSEED=str(job["config"]["seed"]),
                       PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
            p = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--output", str(output), "--worker", job["id"]],
                                 stdout=log, stderr=subprocess.STDOUT, env=env, cwd=HERE)
            active[gpu] = (job,p,log)
            print(now(), "START", job["id"], "pid", p.pid, "gpu", gpu, flush=True)
        for gpu,(job,p,log) in list(active.items()):
            code=p.poll()
            if code is None: continue
            log.close(); del active[gpu]
            ok=code==0 and (output/job["id"]/"verified.json").exists()
            print(now(), "FINISH", job["id"], "exit",code,"verified",ok,flush=True)
            if not ok:
                if attempts[job["id"]]<2: pending.append(job)
                else: failed.append({"id":job["id"],"exit_code":code})
        status={"updated":now(), "pid":os.getpid(), "total":len(plan["jobs"]),
                "verified":sum((output/j["id"]/"verified.json").exists() for j in plan["jobs"]),
                "pending":len(pending), "failed":failed,
                "active":{g:{"id":j["id"],"pid":p.pid} for g,(j,p,l) in active.items()}}
        save(output/"status.json",status)
        if pending or active: time.sleep(10)
    status.update(finished=now(),success=not failed); save(output/"status.json",status)
    return 1 if failed else 0


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--data",type=Path)
    parser.add_argument("--gpus",nargs="+",default=["0","1","2","3"])
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--worker")
    parser.add_argument("--dry-run",action="store_true")
    args=parser.parse_args()
    if args.worker: worker(args.output.resolve(),args.worker)
    elif args.dry_run: print(json.dumps(make_plan(args.output.resolve(),args.data.resolve()),indent=2))
    else: raise SystemExit(launch(args))
