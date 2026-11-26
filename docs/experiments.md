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

Use `--model` and `--dataset` to select the defaults. DAMPS is off by default; enabling it selects `concat` unless you choose another mode.

```bash
python -m mmrecsys.cli train --model pgl --dataset baby
python -m mmrecsys.cli train --model lgmrec --dataset baby --set model.damps_enabled=true
```

See the [PGL](pgl.md) and [LGMRec](lgmrec.md) guides for the differences between `concat` and `official` and the full commands.

## Reproduce the seven baseline/DAMPS pairs

After preparing Baby, run these two Bash blocks in the same shell. They run 14 training jobs on one dataset, not smoke tests. For the five models with experiment presets, start each pair from the same dataset-specific YAML. The ablation command changes the DAMPS switches and writes the paired test metrics to `summary.json` in its output directory.

```bash
dataset=baby
seed=999
for model in bm3 smore freedom mgcn lirdrec; do
  python -m mmrecsys.experiment.ablation \
    --config "configs/experiments/damps_${model}_${dataset}.yaml" \
    --variants baseline full --seeds "$seed" || break
done
```

For PGL and LGMRec, select the integration explicitly. This example uses `official` for the reference integration; use `concat` only if you intend to compare the wider representation. Each training run writes its own `result.json`.

```bash
for model in pgl lgmrec; do
  python -m mmrecsys.cli train --model "$model" --dataset "$dataset" --seed "$seed" || break
  python -m mmrecsys.cli train --model "$model" --dataset "$dataset" --seed "$seed" \
    --set model.damps_enabled=true --set model.damps_mode=official || break
done
```

Compare test Recall and NDCG at 10 and 20 with the same dataset, seed, and training budget. The validation split selects the checkpoint; do not select a seed or mode by its test score.

## Paired ablations and search

```bash
python -m mmrecsys.experiment.ablation --config configs/experiments/damps_mgcn_baby.yaml --device cpu --seeds 999
python -m mmrecsys.experiment.search --dry-run
```

Ablation runs the baseline, full DAMPS, and three single-component removals. It writes `runs.json` and `summary.json` but does not resume the entire queue. [Hyperparameter search](search.md) ranks candidates by paired validation results, not test metrics.

## Spectral analysis

Provide a trusted baseline and DAMPS MGCN checkpoint with matching data fingerprints, matching representation shapes, and trainable features. The analysis reads them on CPU without constructing graphs or training.

```bash
python -m mmrecsys.experiment.spectral_analysis --baseline "$BASELINE_RUN/best.pt" \
  --damps "$DAMPS_RUN/best.pt" --output runs/spectral-example --no-plots
```

Set `BASELINE_RUN` and `DAMPS_RUN` to your run directories before executing this command. The output includes per-item statistics in `per_item.npz` and aggregate statistics and protocol metadata in `summary.json`. For PNG/PDF plots, omit `--no-plots`. Plot titles currently refer to Clothing; use Clothing checkpoints for those figures.
