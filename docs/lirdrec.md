# LIRDRec Integration and Full Experiments

English · [简体中文](zh/lirdrec.md) · [Model index](models.md)

LIRDRec is integrated into the shared data, trainer, evaluator, and checkpoint stack. A historical 75-run study covered five datasets, five variants, and three seeds (999, 2024, 2025). Full DAMPS fell below the baseline on all 20 dataset-by-metric cross-seed means. This did not reproduce the reported gains and does not establish what the upstream integration would achieve. Historical result tables are not included in this checkout.

## Implementation and fixed configuration

The baseline follows [enoche/LIRDRec](https://github.com/enoche/LIRDRec). It retains frozen image/text features, fourfold hidden width, orthonormal DCT-II feature sharing, binary kNN, summed UI representations, PWC fusion, and full-representation mean-square regularization. The DAMPS variant calibrates after the two-layer modality projections and preserves the raw-feature graph and DCT path. The [reference DAMPS integration](https://github.com/ustc-mkh/KDD2026_DAMPS) also changes feature layers, widths, and some graph/DCT inputs; this controlled integration is not an exact port.

PWC updates at each epoch start and stores its epoch, fusion weights, and attention history in the checkpoint. Evaluation recomputes representations without changing PWC history. kNN ties favor smaller item IDs, and training samples negatives from all released items subject to training-history exclusion, rather than only from items observed in training.

`configs/models/lirdrec.yaml` uses 64-dimensional projections, two UI layers, one item layer, k=10, image graph weight 0.1, regularizer 0.0001, and PWC base and initial weights of 0.9. Adam uses a constant 0.0001 learning rate, no weight decay, and no graph dropout. This is a fixed configuration, not a per-dataset optimum.

```bash
python -m mmrecsys.cli train --config configs/experiments/lirdrec_baby.yaml --device cpu --set train.epochs=2
python -m mmrecsys.cli train --config configs/experiments/damps_lirdrec_baby.yaml --device cpu --set train.epochs=2
```

These checks use full Baby data but do not count as full training. `tests/test_lirdrec.py` compares loss and representations against an independent NumPy reference; `tests/test_lirdrec_experiment.py` checks training, evaluation, and resume.

## Full experiment protocol and queue

The full protocol trains up to 1000 epochs, validates every epoch, selects by validation Recall@20, stops after 20 checks without improvement, and then reports test Recall/NDCG@10/20. Each dataset and seed runs `baseline`, `full`, `no_apc`, `no_avrf`, and `no_imcf` with a shared backbone configuration and budget.

The queue uses one explicitly chosen NVIDIA GPU by UUID. Choose a GPU you are authorized to use (see `nvidia-smi -L`), replace the placeholder, and start from the repository root:

```bash
LIRDREC_GPU_UUID="YOUR_GPU_UUID"
python -m mmrecsys.experiment.lirdrec_suite \
  --output runs/lirdrec-five-datasets --gpu-uuid "$LIRDREC_GPU_UUID" \
  --datasets baby sports clothing elec microlens --seeds 999 2024 2025 --wait-for-gpu

python -m mmrecsys.experiment.lirdrec_suite --resume runs/lirdrec-five-datasets
```

Add `--dry-run` to the first command to preview without launching workers. Default concurrency is one; `--max-parallel` accepts 1–10 workers on the same GPU if resources permit. Different datasets may overlap, but the next seed starts only after the previous seed's jobs finish. Each worker is limited to two CPU compute threads. An outside compute process blocks dispatch; `--wait-for-gpu` waits on the designated GPU, not on alternate devices.

Resume validates configuration, source, and data identity and restarts unfinished runs from their original `last.pt` when available. After stopping a queue, you may resume with a different authorized GPU UUID or concurrency limit, subject to the queue and GPU locks. Reuse of existing runs requires `--reuse SLOT_ID=RUN_DIR` and a matching `--reuse-source snapshot.tar.gz`. A worker failure is recorded and stops the queue without silently changing the batch size or model.

`suite.json` holds the plan and state, `results.json` holds per-run metrics, and `summary.csv` / `summary.json` aggregate seed results. Compare paired seeds and report their count and variability. The paper's Table 3 removes individual components from full LIRDRec+DAMPS; Figure 3 instead adds individual components to MGCN and is not a LIRDRec comparison target.
