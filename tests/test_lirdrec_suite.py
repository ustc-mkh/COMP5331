from copy import deepcopy
import io
import json
from pathlib import Path
import subprocess
import tarfile

import pytest

from mmrecsys.experiment import lirdrec_suite as suite


# Fictional UUIDs for mocked scheduling; never identify real lab hardware.
GPU = "GPU-00000000-0000-0000-0000-000000000001"


def test_five_dataset_order_and_immutable_resource_limits(tmp_path):
    datasets = ["baby", "sports", "clothing", "elec", "microlens"]
    slots = suite.build_slots(datasets, [999, 2024, 2025], tmp_path)
    expected = [("baby", 999)] + [(name, 999) for name in datasets[1:]]
    expected += [(name, seed) for seed in (2024, 2025) for name in datasets]
    assert [(slot["dataset"], slot["seed"]) for slot in slots[::5]] == expected
    assert len(slots) == 75
    for offset in range(0, len(slots), 5):
        assert [slot["variant"] for slot in slots[offset:offset + 5]] == list(suite.VARIANTS)
    assert all(slot["config"]["runtime"]["device"] == "cuda:0" for slot in slots)
    assert all(slot["config"]["runtime"]["num_threads"] == 2 for slot in slots)
    assert all(slot["config"]["train"]["epochs"] == 1000 for slot in slots)
    for expression in ("runtime.device=cuda:1", "runtime.num_threads=20", "model.name=mgcn"):
        with pytest.raises(ValueError, match="suite options"):
            suite.build_slots(["baby"], [999], tmp_path, [expression])


def test_source_archive_must_match_bytes(tmp_path):
    archive = tmp_path / "source.tar.gz"
    source = b"a verified training source\n"
    with tarfile.open(archive, "w:gz") as stream:
        member = tarfile.TarInfo("src/mmrecsys/models/lirdrec.py")
        member.size = len(source)
        stream.addfile(member, io.BytesIO(source))
    expected = {"src/mmrecsys/models/lirdrec.py": suite.hashlib.sha256(source).hexdigest()}
    suite.verify_source_archive(archive, expected)
    with pytest.raises(ValueError, match="source snapshot differs"):
        suite.verify_source_archive(archive, {**expected, "missing.py": "unknown"})
    with pytest.raises(ValueError, match="source snapshot differs"):
        suite.verify_source_archive(archive, {"src/mmrecsys/models/lirdrec.py": "changed"})


def test_reuse_config_comparison_includes_budget(tiny_config):
    moved = deepcopy(tiny_config)
    moved["runtime"]["output_root"] += "-different"
    moved["runtime"]["cache_root"] += "-different"
    assert suite.comparable_config(moved) == suite.comparable_config(tiny_config)
    moved["train"]["epochs"] += 1
    assert suite.comparable_config(moved) != suite.comparable_config(tiny_config)


def test_data_fingerprint_detects_content_change(tiny_config):
    before = suite.data_identity(tiny_config)
    path = suite.data_files(tiny_config)[0]
    path.write_text(path.read_text().replace("0\t0\t0", "0\t5\t0", 1))
    assert suite.data_identity(tiny_config) != before


def plan_for(config, tmp_path, monkeypatch):
    config = deepcopy(config)
    config["runtime"]["output_root"] = str(tmp_path / "slot-runs")
    slot = {"id": "baby-999-baseline", "dataset": "baby", "seed": 999,
            "variant": "baseline", "stage": 1, "config": config,
            "config_digest": suite.digest(suite.comparable_config(config)),
            "source": {}, "source_digest": suite.digest({}), "status": "pending"}
    plan = {"format_version": suite.FORMAT_VERSION, "gpu_uuid": GPU, "slots": [slot]}
    monkeypatch.setattr(suite, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(suite, "source_identity", lambda dataset: {})
    return plan, slot


def test_missing_dataset_stops_before_later_stages(tmp_path, tiny_config, monkeypatch):
    plan, first = plan_for(tiny_config, tmp_path, monkeypatch)
    first["config"]["data"]["root"] = str(tmp_path / "missing")
    first["config_digest"] = suite.digest(suite.comparable_config(first["config"]))
    later = deepcopy(first)
    later.update(id="baby-2024-baseline", seed=2024, stage=3)
    plan["slots"].append(later)
    monkeypatch.setattr(suite, "gpu_processes", lambda gpu: pytest.fail("Must stop before GPU access"))
    assert not suite.run_suite(tmp_path / "suite", plan)
    saved = json.loads((tmp_path / "suite/suite.json").read_text())
    assert [slot["status"] for slot in saved["slots"]] == ["waiting_data", "pending"]


def test_busy_requested_gpu_never_falls_back(tmp_path, tiny_config, monkeypatch):
    plan, slot = plan_for(tiny_config, tmp_path, monkeypatch)
    monkeypatch.setattr(suite, "gpu_processes", lambda gpu: [f"{GPU}, 123, other-training"])
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: pytest.fail("No worker may launch"))
    assert not suite.run_suite(tmp_path / "suite", plan)
    assert slot["status"] == "waiting_gpu"
    assert plan["gpu_uuid"] == GPU


def test_interrupt_terminates_only_own_child_and_persists_resume(tmp_path, tiny_config, monkeypatch):
    plan, slot = plan_for(tiny_config, tmp_path, monkeypatch)
    monkeypatch.setattr(suite, "gpu_processes", lambda gpu: [])
    calls = []

    class Child:
        pid = 314
        polls = 0

        def poll(self):
            self.polls += 1
            if self.polls == 1:
                raise KeyboardInterrupt()
            return None

        def wait(self, timeout=None):
            return -15

        def terminate(self):
            calls.append("terminate")

    def launch(command, **kwargs):
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == GPU
        assert kwargs["env"]["OMP_NUM_THREADS"] == "2"
        assert len(kwargs["pass_fds"]) == 2
        calls.append("launch")
        return Child()

    monkeypatch.setattr(subprocess, "Popen", launch)
    with pytest.raises(KeyboardInterrupt):
        suite.run_suite(tmp_path / "suite", plan)
    assert calls == ["launch", "terminate"]
    saved = json.loads((tmp_path / "suite/suite.json").read_text())
    assert saved["slots"][0]["status"] == "interrupted"
    assert "pid" not in saved["slots"][0]


def test_environment_replaces_multi_gpu_visibility(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    env = suite.worker_environment(GPU)
    assert env["CUDA_VISIBLE_DEVICES"] == GPU
    assert env["OMP_NUM_THREADS"] == env["MKL_NUM_THREADS"] == env["OPENBLAS_NUM_THREADS"] == "2"
    for unsafe in ("auto", "0", "0,1", GPU + ",GPU-other"):
        with pytest.raises(ValueError):
            suite.worker_environment(unsafe)


def test_truncated_final_result_replays_from_checkpoint(tmp_path, monkeypatch):
    previous = tmp_path / "run"
    previous.mkdir()
    (previous / "result.json").write_text('{"best_epoch":')
    monkeypatch.setattr(suite, "discover_resume", lambda slot: previous)
    calls = []
    config = {"sentinel": "saved-config"}

    def train(actual_config, resume, *, evaluate_test):
        calls.append((actual_config, resume, evaluate_test))
        return previous, {"regenerated": True}

    assert suite.run_or_resume({"config": config}, train) == (previous, {"regenerated": True})
    assert calls == [(config, previous / "last.pt", True)]


def test_post_launch_status_write_failure_reaps_worker(tmp_path, tiny_config, monkeypatch):
    plan, slot = plan_for(tiny_config, tmp_path, monkeypatch)
    monkeypatch.setattr(suite, "gpu_processes", lambda gpu: [])
    original_save = suite.save_progress
    calls = []

    class Child:
        pid = 315

        def poll(self):
            return None

        def wait(self, timeout=None):
            calls.append("reap")
            return -15

        def terminate(self):
            calls.append("terminate")

    def fail_after_launch(output, actual_plan):
        if actual_plan["slots"][0].get("pid") == Child.pid:
            raise OSError("simulated full disk after launch")
        return original_save(output, actual_plan)

    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: Child())
    monkeypatch.setattr(suite, "save_progress", fail_after_launch)
    with pytest.raises(OSError, match="full disk"):
        suite.run_suite(tmp_path / "suite", plan)
    assert calls == ["terminate", "reap"]
    assert slot["status"] == "interrupted"


def test_summary_includes_only_complete_seed_pairs():
    def slot(seed, variant, value, status="completed"):
        return {"dataset": "baby", "seed": seed, "variant": variant, "status": status,
                "result": {"test": {"Recall@20": value, "n_users": 4}}}

    rows = suite.summarize([slot(999, "baseline", 0.1), slot(999, "full", 0.15),
                            slot(2024, "baseline", 0.2), slot(2024, "full", 0.3),
                            slot(2025, "full", 0.9), slot(2025, "baseline", 0, "failed")])
    full = next(row for row in rows if row["variant"] == "full")
    assert full["n_runs"] == 3 and full["n_pairs"] == 2
    assert full["paired_seeds"] == [999, 2024]
    assert full["delta_mean"] == pytest.approx(0.075)
    assert full["delta_sample_std"] == pytest.approx(0.05 / 2 ** 0.5)
    assert full["relative_gain_pct"] == pytest.approx(50)


def test_partial_metrics_cannot_mark_run_completed(tmp_path, tiny_config):
    result = {"best_epoch": 1, "best_validation_metric": 0.5,
              "monitor": tiny_config["eval"]["monitor"],
              "test": {"Recall@1": 0.2, "NDCG@1": 0.1, "n_users": 4}}
    (tmp_path / "result.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="all configured"):
        suite.read_result(tmp_path, tiny_config)


NEW_GPU = "GPU-00000000-0000-0000-0000-000000000002"


def test_gpu_migration_rejected_while_suite_worker_holds_lock(tmp_path, tiny_config, monkeypatch):
    plan, _ = plan_for(tiny_config, tmp_path, monkeypatch)
    output = tmp_path / "suite"
    suite.save_progress(output, plan)
    original = (output / "suite.json").read_bytes()
    with suite.exclusive_lock(output / ".lock"):
        with pytest.raises(RuntimeError, match="surviving worker"):
            with suite.suite_resources(output, plan, NEW_GPU, True):
                pytest.fail("Active suite must not migrate")
    assert (output / "suite.json").read_bytes() == original
    assert plan["gpu_uuid"] == GPU


def test_gpu_migration_rejected_when_new_card_locked(tmp_path, tiny_config, monkeypatch):
    plan, _ = plan_for(tiny_config, tmp_path, monkeypatch)
    output = tmp_path / "suite"
    suite.save_progress(output, plan)
    with suite.exclusive_lock(tmp_path / "cache" / f"lirdrec-suite-{NEW_GPU}.lock"):
        with pytest.raises(RuntimeError, match="surviving worker"):
            with suite.suite_resources(output, plan, NEW_GPU, True):
                pytest.fail("Must hold the new GPU lock before recording a migration")
    assert json.loads((output / "suite.json").read_text())["gpu_uuid"] == GPU
    assert "gpu_migrations" not in plan


def test_gpu_migration_reloads_latest_progress_and_persists_policy(tmp_path, tiny_config, monkeypatch):
    plan, slot = plan_for(tiny_config, tmp_path, monkeypatch)
    original_config = deepcopy(slot["config"])
    original_digest = slot["config_digest"]
    output = tmp_path / "suite"
    suite.save_progress(output, plan)
    stale = deepcopy(plan)
    plan["slots"][0]["attempt"] = 3
    suite.save_progress(output, plan)
    with suite.suite_resources(output, stale, NEW_GPU, True) as (_, _, selected):
        assert selected == NEW_GPU
        assert stale["slots"][0]["attempt"] == 3
        assert stale["slots"][0]["config"] == original_config
        assert stale["slots"][0]["config_digest"] == original_digest
        # Once migrated, this suite reserves no lock on the previous GPU.
        with suite.exclusive_lock(tmp_path / "cache" / f"lirdrec-suite-{GPU}.lock"):
            pass
    saved = json.loads((output / "suite.json").read_text())
    assert saved["gpu_uuid"] == NEW_GPU and saved["wait_for_gpu"] is True
    assert [(move["from"], move["to"]) for move in saved["gpu_migrations"]] == [(GPU, NEW_GPU)]
    with suite.suite_resources(output, saved) as (_, _, selected):
        assert selected == NEW_GPU and saved["wait_for_gpu"] is True
    assert len(saved["gpu_migrations"]) == 1


def test_wait_for_busy_gpu_retries_only_saved_uuid_then_proceeds(tmp_path, tiny_config, monkeypatch):
    plan, slot = plan_for(tiny_config, tmp_path, monkeypatch)
    plan["wait_for_gpu"] = True
    queries, sleeps = [], []
    responses = iter([[f"{GPU}, 500, other-job"], [f"{GPU}, 500, other-job"], []])

    def processes(gpu):
        queries.append(gpu)
        return next(responses)

    monkeypatch.setattr(suite, "gpu_processes", processes)
    monkeypatch.setattr(suite.time, "sleep", sleeps.append)
    assert suite.wait_until_gpu_available(tmp_path / "suite", plan, slot, GPU)
    assert queries == [GPU, GPU, GPU] and sleeps == [30, 30]
    saved = json.loads((tmp_path / "suite/suite.json").read_text())
    assert saved["wait_for_gpu"] is True and saved["slots"][0]["status"] == "waiting_gpu"


def test_gpu_wait_is_interruptible_without_starting_training(tmp_path, tiny_config, monkeypatch):
    plan, slot = plan_for(tiny_config, tmp_path, monkeypatch)
    plan["wait_for_gpu"] = True
    monkeypatch.setattr(suite, "gpu_processes", lambda gpu: [f"{GPU}, 500, other-job"])

    def interrupt(seconds):
        raise KeyboardInterrupt()

    monkeypatch.setattr(suite.time, "sleep", interrupt)
    with pytest.raises(KeyboardInterrupt):
        suite.wait_until_gpu_available(tmp_path / "suite", plan, slot, GPU)
    assert slot["status"] == "waiting_gpu"
    assert "pid" not in slot


def queue_plan(tiny_config, tmp_path, monkeypatch, *, datasets=("baby", "sports", "clothing"), seeds=(999, 2024)):
    plan, template = plan_for(tiny_config, tmp_path, monkeypatch)
    slots = []
    for seed in seeds:
        for dataset in datasets:
            for variant in suite.VARIANTS:
                slot = deepcopy(template)
                slot.update(id=f"{dataset}-{seed}-{variant}", dataset=dataset, seed=seed, variant=variant)
                slot["config"]["seed"] = seed
                slot["config"]["data"]["name"] = dataset
                slot["config"]["runtime"]["output_root"] = str(tmp_path / "suite/slots" / slot["id"] / "runs")
                slot["config_digest"] = suite.digest(suite.comparable_config(slot["config"]))
                slots.append(slot)
    plan["slots"] = slots
    monkeypatch.setattr(suite, "data_identity", lambda config: {"fixture": {"sha256": "unchanged"}})
    monkeypatch.setattr(suite, "verify_run", lambda slot, run, **kwargs: "fixture-fingerprint")
    return plan


class QueueSimulation:
    """Exercise the production scheduling loop with timed, isolated fake workers."""
    def __init__(self, monkeypatch):
        self.clock = 0.0
        self.children = []
        self.live = {}
        self.events = []
        self.peak = 0
        self.durations = {}
        self.exit_codes = {}
        self.external = lambda: []
        self.interrupt_on_sleep = False
        monkeypatch.setattr(suite.time, "monotonic", lambda: self.clock)
        monkeypatch.setattr(suite.time, "sleep", self.sleep)
        monkeypatch.setattr(subprocess, "Popen", self.launch)
        monkeypatch.setattr(suite, "gpu_processes", self.gpu_processes)

    def sleep(self, seconds):
        if self.interrupt_on_sleep:
            raise KeyboardInterrupt()
        self.clock += seconds

    def gpu_processes(self, gpu):
        assert gpu == GPU
        return [f"{GPU}, {pid}, owned-worker" for pid in self.live] + self.external()

    def launch(self, command, **kwargs):
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == GPU
        assert kwargs["env"]["OMP_NUM_THREADS"] == "2"
        assert len(kwargs["pass_fds"]) == 2
        task = Path(command[-1]).parent
        spec = json.loads((task / "worker.json").read_text())
        assert spec["gpu_uuid"] == GPU
        child = SimulatedChild(self, spec["slot"], task)
        self.children.append(child)
        self.live[child.pid] = child
        self.peak = max(self.peak, len(self.live))
        self.events.append(("launch", child.slot["id"], self.clock))
        return child


class SimulatedChild:
    def __init__(self, simulation, slot, task):
        self.simulation, self.slot, self.task = simulation, slot, task
        self.pid = 10000 + len(simulation.children)
        self.resume = slot.get("resume_checkpoint")
        self.end = simulation.clock + simulation.durations.get(slot["id"], 1 + self.pid % 3)
        self.returncode = None
        self.run = Path(self.resume).parent if self.resume else Path(slot["config"]["runtime"]["output_root"]) / "run-0001"
        self.run.mkdir(parents=True, exist_ok=True)
        (self.run / "config.yaml").write_text(suite.yaml.safe_dump(slot["config"]))
        # Checkpoint existence drives the real discover_resume path; checkpoint
        # contents are covered by the separate CPU training/resume tests.
        (self.run / "last.pt").write_bytes(b"saved-epoch")
        (self.run / "best.pt").write_bytes(b"saved-best")

    def poll(self):
        if self.returncode is not None:
            return self.returncode
        if self.simulation.clock < self.end:
            return None
        self.returncode = self.simulation.exit_codes.get(self.slot["id"], 0)
        self.simulation.live.pop(self.pid)
        self.simulation.events.append(("done", self.slot["id"], self.simulation.clock))
        if self.returncode == 0:
            result = {"best_epoch": 1, "best_validation_metric": 0.5,
                      "monitor": self.slot["config"]["eval"]["monitor"],
                      "test": {f"{metric}@{k}": 0.5 for metric in ("Recall", "NDCG")
                               for k in self.slot["config"]["eval"]["topk"]}}
            result["test"]["n_users"] = 4
            suite.atomic_json(self.run / "result.json", result)
            suite.atomic_json(self.task / "completed.json", {"run_dir": str(self.run), "result": result,
                              "training_data_fingerprint": "fixture-fingerprint"})
        return self.returncode

    def terminate(self):
        self.returncode = -15
        self.simulation.live.pop(self.pid, None)
        self.simulation.events.append(("terminate", self.slot["id"], self.simulation.clock))

    def kill(self):
        self.terminate()
        self.returncode = -9

    def wait(self, timeout=None):
        code = self.poll()
        if code is None:
            if timeout is not None and self.simulation.clock + timeout < self.end:
                raise subprocess.TimeoutExpired("simulated worker", timeout)
            self.simulation.clock = self.end
            code = self.poll()
        return code


@pytest.mark.parametrize("limit", [1, 2, 10])
def test_queue_obeys_concurrency_limit_and_seed_barrier(tiny_config, tmp_path, monkeypatch, limit):
    plan = queue_plan(tiny_config, tmp_path, monkeypatch)
    simulation = QueueSimulation(monkeypatch)
    assert suite.run_suite(tmp_path / "suite", plan, max_parallel=limit)
    assert simulation.peak == limit
    assert not simulation.live
    assert all(slot["status"] == "completed" for slot in plan["slots"])
    end_999 = max(time for event, name, time in simulation.events if event == "done" and "-999-" in name)
    start_2024 = min(time for event, name, time in simulation.events if event == "launch" and "-2024-" in name)
    assert start_2024 >= end_999
    saved = json.loads((tmp_path / "suite/suite.json").read_text())
    assert saved["max_parallel"] == limit
    assert all("worker_seconds" in slot for slot in saved["slots"])
    if limit == 10:
        first_launches = [name for event, name, _ in simulation.events if event == "launch"][:10]
        assert any(name.startswith("sports-") for name in first_launches)


def test_external_gpu_job_blocks_dispatch_but_not_reaping(tiny_config, tmp_path, monkeypatch):
    plan = queue_plan(tiny_config, tmp_path, monkeypatch, datasets=("baby",), seeds=(999,))
    simulation = QueueSimulation(monkeypatch)
    first, second, third = [slot["id"] for slot in plan["slots"][:3]]
    simulation.durations.update({first: 1, second: 4})
    simulation.external = lambda: [f"{GPU}, 99999, external"] if 0.5 <= simulation.clock < 10 else []
    assert suite.run_suite(tmp_path / "suite", plan, max_parallel=2, wait_for_gpu=True)
    second_done = next(time for event, name, time in simulation.events if event == "done" and name == second)
    third_start = next(time for event, name, time in simulation.events if event == "launch" and name == third)
    assert second_done < 5 and third_start >= 10
    assert not any(event == "terminate" for event, _, _ in simulation.events)
    assert not simulation.live


def test_stale_plan_pid_is_never_whitelisted(tiny_config, tmp_path, monkeypatch):
    plan = queue_plan(tiny_config, tmp_path, monkeypatch, datasets=("baby",), seeds=(999,))
    plan["slots"][0]["pid"] = 99999
    simulation = QueueSimulation(monkeypatch)
    simulation.external = lambda: [f"{GPU}, 99999, another-user-now"]
    assert not suite.run_suite(tmp_path / "suite", plan, max_parallel=10)
    assert not simulation.children
    assert plan["slots"][0]["status"] == "waiting_gpu"


def test_worker_failure_reaps_peers_and_resume_uses_existing_checkpoints(tiny_config, tmp_path, monkeypatch):
    plan = queue_plan(tiny_config, tmp_path, monkeypatch, datasets=("baby",), seeds=(999, 2024))
    simulation = QueueSimulation(monkeypatch)
    first, second = [slot["id"] for slot in plan["slots"][:2]]
    simulation.durations.update({first: 1, second: 60})
    simulation.exit_codes[first] = 1
    assert not suite.run_suite(tmp_path / "suite", plan, max_parallel=2)
    assert [slot["status"] for slot in plan["slots"][:3]] == ["failed", "interrupted", "pending"]
    assert not simulation.live and len(simulation.children) == 2
    assert [name for event, name, _ in simulation.events if event == "terminate"] == [second]
    simulation.exit_codes.clear()
    simulation.durations[second] = 1
    assert suite.run_suite(tmp_path / "suite", plan)
    retried = simulation.children[2:4]
    assert all(child.resume and child.resume.endswith("last.pt") for child in retried)
    assert all(child.slot["attempt"] == 2 for child in retried)
    assert plan["max_parallel"] == 2 and not simulation.live
    assert all(slot["status"] == "completed" for slot in plan["slots"])


def test_interrupt_reaps_all_ten_workers_and_leaves_later_seeds_pending(tiny_config, tmp_path, monkeypatch):
    plan = queue_plan(tiny_config, tmp_path, monkeypatch)
    simulation = QueueSimulation(monkeypatch)
    simulation.interrupt_on_sleep = True
    with pytest.raises(KeyboardInterrupt):
        suite.run_suite(tmp_path / "suite", plan, max_parallel=10)
    assert simulation.peak == 10 and not simulation.live
    assert len([event for event, _, _ in simulation.events if event == "terminate"]) == 10
    assert all(slot["status"] == "interrupted" for slot in plan["slots"][:10])
    assert all(slot["status"] == "pending" for slot in plan["slots"][10:])
    assert all("pid" not in slot for slot in plan["slots"])


def test_parallel_setting_persists_and_rejects_over_ten(tmp_path, tiny_config, monkeypatch):
    plan, _ = plan_for(tiny_config, tmp_path, monkeypatch)
    output = tmp_path / "suite"
    with suite.suite_resources(output, plan, max_parallel=10):
        assert plan["max_parallel"] == 10
    with suite.suite_resources(output, plan):
        assert plan["max_parallel"] == 10
    for invalid in (0, 11, True, 1.5):
        with pytest.raises(ValueError, match="max_parallel"):
            with suite.suite_resources(output, plan, max_parallel=invalid):
                pytest.fail("Invalid concurrency must not be accepted")


@pytest.mark.parametrize("peer_code,peer_status", [(0, "completed"), (1, "failed")])
def test_failure_cleanup_preserves_peers_that_already_finished(tiny_config, tmp_path, monkeypatch, peer_code, peer_status):
    plan = queue_plan(tiny_config, tmp_path, monkeypatch, datasets=("baby",), seeds=(999,))
    simulation = QueueSimulation(monkeypatch)
    first, second = [slot["id"] for slot in plan["slots"][:2]]
    simulation.durations.update({first: 1, second: 1})
    simulation.exit_codes.update({first: 1, second: peer_code})
    assert not suite.run_suite(tmp_path / "suite", plan, max_parallel=2)
    assert [slot["status"] for slot in plan["slots"][:3]] == ["failed", peer_status, "pending"]
    assert not simulation.live
    assert not any(event == "terminate" for event, _, _ in simulation.events)


def test_no_wait_mode_drains_owned_workers_before_returning(tiny_config, tmp_path, monkeypatch):
    plan = queue_plan(tiny_config, tmp_path, monkeypatch, datasets=("baby",), seeds=(999,))
    simulation = QueueSimulation(monkeypatch)
    first, second = [slot["id"] for slot in plan["slots"][:2]]
    simulation.durations.update({first: 1, second: 4})
    simulation.external = lambda: [f"{GPU}, 99999, external"] if simulation.clock >= 0.5 else []
    assert not suite.run_suite(tmp_path / "suite", plan, max_parallel=2, wait_for_gpu=False)
    assert [slot["status"] for slot in plan["slots"][:3]] == ["completed", "completed", "waiting_gpu"]
    assert len(simulation.children) == 2 and not simulation.live
    assert not any(event == "terminate" for event, _, _ in simulation.events)


def test_launch_failure_stops_existing_peers(tiny_config, tmp_path, monkeypatch):
    plan = queue_plan(tiny_config, tmp_path, monkeypatch, datasets=("baby",), seeds=(999,))
    simulation = QueueSimulation(monkeypatch)
    original_launch = simulation.launch

    def fail_second(command, **kwargs):
        if simulation.children:
            raise OSError("simulated process creation failure")
        return original_launch(command, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", fail_second)
    with pytest.raises(OSError, match="process creation"):
        suite.run_suite(tmp_path / "suite", plan, max_parallel=10)
    assert [slot["status"] for slot in plan["slots"][:3]] == ["interrupted", "failed", "pending"]
    assert not simulation.live
