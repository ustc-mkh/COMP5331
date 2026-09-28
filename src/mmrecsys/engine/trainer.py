import time

import torch

from .checkpoint import load_checkpoint, save_checkpoint
from ..experiment.logging import append_metrics, write_json


class Trainer:
    def __init__(self, model, optimizer, scheduler, sampler, evaluator, train_config: dict,
                 run_dir, device, checkpoint_config: dict, fingerprint: str):
        self.model, self.optimizer, self.scheduler = model, optimizer, scheduler
        self.sampler, self.evaluator = sampler, evaluator
        self.config, self.run_dir, self.device = train_config, run_dir, device
        self.checkpoint_config, self.fingerprint = checkpoint_config, fingerprint

    def fit(self, valid, test, resume=None, *, evaluate_test=True):
        state = {"epoch": 0, "global_step": 0, "best_metric": None,
                 "best_epoch": 0, "bad_evaluations": 0}
        if resume is not None:
            state = load_checkpoint(resume, self.model, self.checkpoint_config, self.fingerprint,
                                    optimizer=self.optimizer, scheduler=self.scheduler, sampler=self.sampler)
        for epoch in range(state["epoch"] + 1, self.config["epochs"] + 1):
            if state["bad_evaluations"] >= self.config["patience"]:
                break
            if torch.device(self.device).type == "cuda":
                torch.cuda.synchronize(self.device)
            started = time.monotonic()
            self.model.train()
            self.model.on_epoch_start(epoch)
            total_samples, sums = 0, {}
            diagnostics = {}
            for batch in self.sampler.batches(epoch):
                self.optimizer.zero_grad(set_to_none=True)
                loss = self.model.compute_loss(batch.to(self.device))
                if loss.total.ndim != 0 or not torch.isfinite(loss.total):
                    raise FloatingPointError("Model returned a non-finite or nonscalar training loss")
                loss.total.backward()
                if total_samples == 0:
                    diagnostics = {name: value.detach().item()
                                   for name, value in self.model.training_diagnostics().items()}
                self.optimizer.step()
                total_samples += loss.batch_size
                for name, value in {"total": loss.total, **loss.components}.items():
                    sums[name] = sums.get(name, 0.0) + value.detach().double() * loss.batch_size
                state["global_step"] += 1
            self.model.on_epoch_end(epoch)
            self.scheduler.step()
            state["epoch"] = epoch
            if torch.device(self.device).type == "cuda":
                torch.cuda.synchronize(self.device)
            train_seconds = time.monotonic() - started
            record = {"epoch": epoch, "global_step": state["global_step"],
                      "loss": {name: value.item() / total_samples for name, value in sums.items()},
                      "lr": self.optimizer.param_groups[0]["lr"]}
            if diagnostics:
                record["diagnostics"] = diagnostics
            eval_started = time.monotonic()
            improved = False
            # Validate epoch one and a fixed cadence, independent of the epoch budget.
            if epoch == 1 or epoch % self.config["eval_every"] == 0:
                metrics = self.evaluator.evaluate(self.model, valid)
                record["valid"] = metrics
                metric = metrics[self.evaluator.config["monitor"]]
                best = state["best_metric"]
                improved = best is None or (metric > best if self.evaluator.config["mode"] == "max" else metric < best)
                if improved:
                    state.update(best_metric=metric, best_epoch=epoch, bad_evaluations=0)
                else:
                    state["bad_evaluations"] += 1
            eval_seconds = time.monotonic() - eval_started
            checkpoint_started = time.monotonic()
            if improved:
                self._save("best.pt", state)
            self._save("last.pt", state)
            record["train_seconds"] = train_seconds
            record["eval_seconds"] = eval_seconds
            record["checkpoint_seconds"] = time.monotonic() - checkpoint_started
            record["seconds"] = time.monotonic() - started
            append_metrics(self.run_dir / "metrics.jsonl", record)
            print(f"epoch={epoch} loss={record['loss']['total']:.6f} "
                  f"valid={record.get('valid', {})} seconds={record['seconds']:.1f} "
                  f"train={train_seconds:.2f} eval={eval_seconds:.2f} "
                  f"checkpoint={record['checkpoint_seconds']:.2f}", flush=True)
        best_path = self.run_dir / "best.pt"
        if not best_path.exists():
            raise ValueError("No best checkpoint exists; resume requires the original run's best.pt")
        load_checkpoint(best_path, self.model, self.checkpoint_config, self.fingerprint)
        test_metrics = self.evaluator.evaluate(self.model, test) if evaluate_test else None
        result = {"best_epoch": state["best_epoch"], "best_validation_metric": state["best_metric"],
                  "monitor": self.evaluator.config["monitor"], "test": test_metrics,
                  "evaluation_protocol": {**self.evaluator.config, "tie_break": "item_id_ascending",
                                          "padding": "excluded", "aggregation": "macro_user_mean"}}
        if not evaluate_test:
            result.pop("test")
        filename = "result.json" if evaluate_test else "validation_result.json"
        write_json(self.run_dir / filename, result)
        return result

    def _save(self, filename, state):
        save_checkpoint(self.run_dir / filename, self.model, self.optimizer, self.scheduler,
                        self.sampler, state, self.checkpoint_config, self.fingerprint)
