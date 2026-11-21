# BM3, SMORE, and FREEDOM

English · [简体中文](zh/models.md) · [Documentation](README.md)

These native models run through the common registry, data loader, optimizer, trainer, all-item evaluator, and checkpoint format. They do not import old standalone runners or the MMRec package. Use the [experiment guide](experiments.md) for commands. Each model has baseline and DAMPS presets for five datasets; these are fixed configurations, not the outputs of a fresh per-dataset search.

| Model | Training batch and objective | DAMPS integration |
| --- | --- | --- |
| BM3 | Positive-only; dropout bootstrap targets with stopped gradients, multimodal alignment, and global representation regularization. | Optional project-specific extension: calibrates after projection and trains with the modality loss. |
| SMORE | One negative; BPR, squared regularization of final representations, and side-information/behavior InfoNCE. | Calibrates projected features before SMORE's native spectral fusion and receives gradients. |
| FREEDOM | One negative; recommendation BPR and weighted image/text auxiliary BPR. | Calibrates once at initialization; copies the outputs into independent trainable feature tables and freezes the calibrator. |

BM3 averages the UI graph layers including layer zero, includes an item ID residual, and scores with the predictor. Its regularizer is `(||U||₂ + ||I||₂) / n_items`, not a sum of squares or a batch BPR regularizer. Presets retain the original standalone per-dataset model settings.

SMORE uses separate weighted cosine kNN graphs, an edgewise-max fusion graph, three complex spectral weights, feature-wise attention, preference dropout, and side-information fusion. Its DAMPS initialization statistics are estimated once across all items. Presets retain fixed Task 4 settings.

FREEDOM uses binary kNN, weighted image/text graph fusion, degree-sensitive UI edge pruning each epoch, and the full UI graph at evaluation time. For DAMPS, the operator calibrates in embedding width D, then applies the inverse FFT at each modality's original feature width. Those features are copied to trainable tables and projected back to D by the model. Other backbones still use the standard D-width DAMPS output. Older FREEDOM+DAMPS checkpoints that used D-width tables are not directly resumable. The calibrated graph cache uses the actual initialization feature content; the resulting graph is also saved in the checkpoint.

## FREEDOM raw-graph control

The default `model.damps_graph_source=calibrated` constructs the item graph from calibrated features. To keep the baseline's raw-feature kNN graph and reuse its cache, set `model.damps_graph_source=raw`. DAMPS still initializes the auxiliary feature tables; their widths and auxiliary BPR stay the same. This is a control for testing graph degradation, not part of the reference integration or a guarantee of improvement.

```bash
python -m mmrecsys.cli train --config configs/experiments/damps_freedom_baby.yaml \
  --set model.damps_graph_source=raw --set model.embedding_dim=64 --seed 999
```

## Sources and validation

- BM3: enoche/BM3 commit `ef82a333823ae01ac5964039720f697532b40ea3` from the original `baselines/bm3` tree; original code license GPL-3.0.
- SMORE: the former `task4/runtime/src/models/smore.py`, based on kennethorq/SMORE (WSDM 2025); retains the original spectral fusion and projection-level DAMPS insertion.
- FREEDOM: the former `task6/models/freedom.py`, based on enoche/FREEDOM; the native implementation removes host-specific paths, private SciPy APIs, and duplicate training infrastructure.

`tests/test_migrated_models.py` covers batches and gradients, the original BM3 loss, SMORE's spectral convolution and fusion graph, FREEDOM's pruning and initialization, ablations, checkpoint reload, and epoch-boundary resume. Engineering tests are not a substitute for full dataset experiments.
