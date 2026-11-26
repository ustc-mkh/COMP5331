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

## Full experiment protocol

The full protocol trains up to 1000 epochs, validates every epoch, selects by validation Recall@20, stops after 20 checks without improvement, and then reports test Recall/NDCG@10/20. Each dataset and seed runs `baseline`, `full`, `no_apc`, `no_avrf`, and `no_imcf` with a shared backbone configuration and budget.

Run the five variants for Baby with the shared ablation command:

```bash
python -m mmrecsys.experiment.ablation --config configs/experiments/damps_lirdrec_baby.yaml --seeds 999 2024 2025
```

Compare paired seeds and report their count and variability. The paper's Table 3 removes individual components from full LIRDRec+DAMPS; Figure 3 instead adds individual components to MGCN and is not a LIRDRec comparison target.
