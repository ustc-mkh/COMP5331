# Running Experiments

English · [简体中文](zh/experiments.md) · [Documentation](README.md)

Run these commands from the repository root after following the [Linux quickstart](../README.md).

## Select a model and dataset

For BM3, SMORE, FREEDOM, MGCN, and LIRDRec, use `configs/experiments/<model>_<dataset>.yaml` for the baseline or prefix the filename with `damps_` to enable DAMPS. Presets exist for `baby`, `sports`, `clothing`, `elec`, and `microlens`. Prefer the experiment YAML when you need its dataset-specific overrides.

```bash
python -m mmrecsys.cli train --config configs/experiments/bm3_baby.yaml
python -m mmrecsys.cli train --config configs/experiments/damps_smore_sports.yaml
python -m mmrecsys.cli train --config configs/experiments/damps_freedom_baby.yaml --set model.damps_graph_source=raw
```

### PGL and LGMRec

Use model and dataset defaults directly. DAMPS is off by default. When enabled, `concat` keeps the original image/text channels and appends two calibrated channels. In PGL the fixed item graph still comes from raw features. In LGMRec the global hypergraph still uses raw features, and the collaborative and hypergraph output widths grow to match the four local channels.

```bash
python -m mmrecsys.cli train --model pgl --dataset baby
python -m mmrecsys.cli train --model pgl --dataset baby --set model.damps_enabled=true
python -m mmrecsys.cli train --model lgmrec --dataset baby
python -m mmrecsys.cli train --model lgmrec --dataset baby --set model.damps_enabled=true --set model.damps_mode=official
```

`official` follows the reference integration pattern, not a separate implementation of the DAMPS operator. PGL calibrates at initialization for graph construction and calls DAMPS during forward passes; its external modality projections do not receive a useful gradient. LGMRec calibrates only at initialization and freezes the resulting features. At embedding width 64, baseline / concat / official scoring widths are 128 / 256 / 128 for PGL and 64 / 256 / 64 for LGMRec. Do not mix checkpoints across modes or attribute all concat gains to calibration: width and parameter count also change.

## Paired ablations and search

```bash
python -m mmrecsys.experiment.ablation --config configs/experiments/damps_mgcn_baby.yaml --device cpu --seeds 999
python -m mmrecsys.experiment.search --dry-run
```

Ablation runs the baseline, full DAMPS, and three single-component removals. It writes `runs.json` and `summary.json` but does not resume the entire queue. Use [the LIRDRec suite](lirdrec.md#full-experiment-protocol-and-queue) if you need the resumable five-dataset LIRDRec protocol. [Hyperparameter search](search.md) ranks candidates by paired validation results, not test metrics.

## Multi-GPU MGCN comparison

```bash
python -m mmrecsys.experiment.compare --dry-run
python -m mmrecsys.experiment.compare --datasets baby sports --gpus cpu --set train.epochs=2
python -m mmrecsys.experiment.compare --mode full --gpus 0 1 --seeds 999 2024 2025
```

The default comparison covers Baby, Sports, Clothing, Elec, and MicroLens, with two jobs per dataset and seed (10 jobs per seed). Quick mode uses at most 20 epochs and patience 5; full mode uses at most 1000 epochs and patience 20. It schedules independent runs, one per GPU at a time, not distributed training. Specify available GPUs explicitly on shared machines. Indices refer to visible GPUs; e.g. if `CUDA_VISIBLE_DEVICES=4,5`, pass `--gpus 0 1`. The dry run does not read data or initialize GPUs.

Use `--resume YOUR_COMPARE_DIRECTORY --gpus 0 1` to skip completed jobs and restart incomplete ones. `runs.json` records status and paths; `summary.md`, `summary.csv`, and `summary.json` report Recall/NDCG@10/20 from complete seed pairs. Absolute gain is DAMPS minus baseline, and relative gain divides the difference in means by the baseline mean. The reports also distinguish mean per-seed relative gain, omitting seeds whose baseline is zero from that statistic. A two-epoch check is not a convergence result.

## Spectral analysis

Provide a trusted baseline and DAMPS MGCN checkpoint with matching data fingerprints, matching representation shapes, and trainable features. The analysis reads them on CPU without constructing graphs or training.

```bash
python -m mmrecsys.experiment.spectral_analysis --baseline "$BASELINE_RUN/best.pt" \
  --damps "$DAMPS_RUN/best.pt" --output runs/spectral-example --no-plots
```

Set `BASELINE_RUN` and `DAMPS_RUN` to your run directories before executing this command. The output includes per-item statistics in `per_item.npz` and aggregate statistics and protocol metadata in `summary.json`. For PNG/PDF plots, install `matplotlib` with pip and omit `--no-plots`. Plot titles currently refer to Clothing; use Clothing checkpoints for those figures.
