# BM3 + DAMPS: five datasets, three seeds

This extension adds trainable DAMPS to the verified [original BM3 runtime](../bm3/README.md). It uses the group's spectral module from `src/mmrecsys/nn/damps.py` at commit `13691bf6016939f68bf487ca547febca789f67e7`. The active experiment is **15 full-DAMPS runs**, using seeds **999, 2024, 2025** on five datasets. The matching 15 runs from the original BM3 archive provide the paired control. The user cancelled all component-removal training and seeds 2026/42 on 12 November 2026; existing extra-seed outputs remain archived and are excluded from this report.

## Integration and source audit

`model.py` inherits the original BM3 model. Image and text embeddings are projected to 64 dimensions by BM3's existing projections, then calibrated by DAMPS at every loss calculation, before creating the stop-gradient dropout targets. BM3's graph propagation, predictor, loss weights, optimizer, and recommendation scoring stay the same. DAMPS receives training gradients through the online modality branches. Its fixed phase prior and initial variance weights are estimated once from all initial projected item features. No extra randomly initialized layers are added; backbone initialization matches the same-seed baseline.

This is the **projected, trainable integration**, identified as `projected_trainable_v1`. The author's v1.0.0 BM3 adapter (`af90958f5fb43214caeda9729fefd5c9044e26ac`) invokes its feature filter only in the constructor and wraps the output in new embedding Parameters. That disconnects DAMPS parameters from the later training loss. It also reconstructs to the raw feature widths. The reference adapter imports `Damps`, whereas the supplied module defines `DAMPS`. The two author files are preserved as `author_bm3_reference.py` and `author_damps_reference.py` for inspection; they are not the training entry point. Our results must be described using the integration above, rather than as execution of the unchanged author adapter.

The copied group module preserves its version-3 phase signs (image minus, text plus), unconstrained trainable AVRF weights, and source IMCF expression. It is byte-identical to the group source at the stated commit. At dimension 64, the full module adds 101 trainable scalars.

| Variant | APC | AVRF | IMCF | Extra trainable scalars |
| --- | --- | --- | --- | ---: |
| `full` | yes | yes | yes | 101 |
| `no_apc` | no | yes | yes | 68 |
| `no_avrf` | yes | no | yes | 33 |
| `no_imcf` | yes | yes | no | 99 |

The component switches remain available for implementation tests. Only `full` is scheduled for the requested experiment; component-removal runs are cancelled.

## Experimental protocol

The suite runs Baby, Sports, Clothing, Electronics (`elec`), and MicroLens with seeds 999, 2024, and 2025: **15 full-DAMPS runs**, paired with 15 archived baseline runs. Every training/evaluation setting is copied from the corresponding baseline's `docs/results/bm3/runs.json`; only extension identifiers are added. Data fingerprints must also match.

Each run has at most 1000 epochs and selects the first maximum validation Recall@20; training stops after 21 consecutive non-improving validations. Test ranking masks training history and retains validation interactions, matching the original baseline. Metrics use the upstream rounding before computing three-seed means and sample standard deviations. Each completed checkpoint is separately reconstructed and re-evaluated; all four validation/test metrics and the selected epoch must match before a run becomes verified.

## Run on Linux

Use the [existing BM3 environment](../bm3/README.md#环境与数据): Python 3.8.20, PyTorch 2.4.1+cu121, NumPy 1.22.4 and the pinned runtime requirements. Run these commands from the repository root:

```bash
python -m pytest -q baselines/bm3_damps/tests
python baselines/bm3_damps/smoke.py \
  --data /path/to/datasets --output /path/to/new-smoke --variants full
nohup python -u baselines/bm3_damps/run_selected.py \
  --data /path/to/datasets --output /path/to/new-suite \
  --gpus GPU-uuid-1 GPU-uuid-2 GPU-uuid-3 GPU-uuid-4 \
  > /path/to/launcher.log 2>&1 < /dev/null &
```

Use idle GPUs identified by UUID. The launcher runs one process per GPU, freezes the source digest and configuration, and writes `status.json`. Each job keeps `best.pt`, `last.pt`, complete epoch history, metadata, test rankings, independent evaluation and a `verified.json` marker. A failed job is retried once; completed jobs are skipped when relaunching with `--resume`. The selected launcher adopts already-running desired workers by their exact suite path and GPU UUID, avoiding duplicate training. A file lock prevents two launchers for the same suite. Checkpoints stay with their original run directory. Smoke runs use two batches per epoch and 32 evaluation users and are excluded from formal results.

`run_suite.py` remains byte-identical to the initial executed worker because its digest is stored in checkpoints. Use `run_selected.py` as the current entry point. The scope-change utility archived the original 100-job plan, cancelled its launcher and maintenance process, stopped the unwanted seed-42 worker, and retained the three running Clothing workers. The model, optimizer settings, and checkpoint configuration were unchanged.

The integration tests check active-parameter gradients, same-seed backbone initialization, unchanged baseline behavior, output dimensions, fixed priors, checkpoint reload, and exact restart equivalence. Training imports the extension only inside its worker process; it does not modify the original BM3 runtime on disk. See the original runtime's GPL-3.0 license and retained upstream notices.

## Result delivery

Export and upload only after all 15 requested runs are verified. The main table receives four test metrics per dataset. Recompute the original BM3 column using the same seeds 999/2024/2025, while preserving its original five-seed archive. The comparison contains 15 baseline and 15 full-DAMPS observations. No ablation result is delivered.

```bash
# Once all 15 full-DAMPS runs are verified:
python baselines/bm3_damps/results.py --suite /path/to/new-suite \
  --stage main --output docs/results/bm3-damps
python baselines/bm3_damps/plot_results.py --output docs/results/bm3-damps
```

The export checks configurations, source/data identity, complete seed coverage, checkpoint reloads, earliest validation selection, and early stopping before producing portable evidence and spreadsheet-ready means. Extra seeds and ablation variants fail the publication checks. `paired-main-sheet.tsv` contains the matching BM3 and BM3+DAMPS columns. Existing group formatting is the template for the upload.

## Thirty-minute maintenance

`maintain.py` checks the restricted 15-job suite every 1800 seconds and writes `maintenance/status.json` plus append-only `history.jsonl` next to the suite. It records verified counts, dataset coverage, current worker epochs, stale logs, free disk space, source identity, and recovery actions. It exits after all 15 results have been verified, exported, and plotted.

If the launcher and all its workers have exited before completion, it can resume the same suite from saved checkpoints. Recovery requires unchanged training source, at least 20 GiB free, and no foreign compute processes on the specified GPU UUIDs. A lock prevents duplicate maintainers; automatic queue recoveries are capped at three. It never kills a training process. Stalled workers, source mismatches and repeated failures are recorded for investigation. The main launcher also retries each failed job once.

```bash
nohup python -u baselines/bm3_damps/maintain.py \
  --suite /path/to/new-suite --data /path/to/datasets \
  --gpus GPU-uuid-1 GPU-uuid-2 GPU-uuid-3 GPU-uuid-4 \
  --interval 1800 > /path/to/maintenance.log 2>&1 < /dev/null &
```

Use the same GPU UUIDs as the training launcher. The complete three-seed main experiment receives a `ready-main.json` marker with `uploaded=false`; local export is not evidence of a Google Sheets upload. Recovery invokes only `run_selected.py` and cannot restore cancelled jobs.
