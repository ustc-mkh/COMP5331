# BM3

English · [简体中文](zh/bm3.md) · [Model index](models.md)

BM3 trains from positive user–item interactions without negative sampling. It averages the normalized user–item graph representations from layer zero through the last layer, then adds an item ID residual. A shared predictor maps the online embeddings; the targets are dropout views of the same embeddings with gradients stopped, not outputs of a separate moving-average model.

## Training and scoring

The bootstrap loss aligns users with their interacted items in both directions using cosine distance. The modality loss aligns each image/text representation with the item target and its own dropout target. The regularizer is `(‖U‖₂ + ‖I‖₂) / n_items` over the full user and item representations, not squared L2 or batch BPR regularization. Evaluation uses the dot product of the predicted user and item embeddings; modality features influence scores through training, not through a separate scoring branch.

## DAMPS integration

The optional [DAMPS](damps.md) extension calibrates the projected image and text features before creating the modality targets and online predictions. It estimates initialization statistics once over all items and receives gradients through the online modality branch. The collaborative graph and scoring formula stay unchanged.

## Run it

The model YAML defaults to width 64, one graph layer, dropout 0.5, and modality loss weight 2.0. Use the experiment YAMLs for dataset-specific overrides; the presets retain the original standalone model settings.

```bash
python -m mmrecsys.cli train --config configs/experiments/bm3_baby.yaml
python -m mmrecsys.cli train --config configs/experiments/damps_bm3_baby.yaml
```

Adapted from enoche/BM3 commit `ef82a333823ae01ac5964039720f697532b40ea3` (GPL-3.0). The model tests cover the bootstrap equations, scoring, gradients, and checkpoint recovery.

[Implementation](../src/mmrecsys/models/bm3.py) · [Default config](../configs/models/bm3.yaml) · [Tests](../tests/test_migrated_models.py)
