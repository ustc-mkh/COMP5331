# FREEDOM

English · [简体中文](zh/freedom.md) · [Model index](models.md)

FREEDOM fixes the multimodal item graph and prunes the training user–item graph. It selects neighbors by cosine similarity, gives selected edges binary weights before normalization, and mixes the image and text graphs. Each training epoch resamples user–item edges with inverse-square-root degree weights; evaluation uses the full training interaction graph.

## Training and scoring

The model adds item-graph propagation of item IDs to the collaborative item representation. Its main BPR loss compares one positive and one negative per user. Two auxiliary BPR terms score projected image and text features against the user embedding. `model.reg_weight` weights the sum of these auxiliary losses; it is not an L2 penalty in this implementation. Evaluation uses the final collaborative and item-graph representations, not the auxiliary projections directly.

## DAMPS integration and raw-graph control

[DAMPS](damps.md) runs only at initialization. After calibrating at embedding width D, the inverse FFT reconstructs each modality at its original input width. The model copies the calibrated outputs into independent trainable tables, then projects them back to D for the auxiliary losses. The calibrator is frozen. Older FREEDOM+DAMPS checkpoints with D-wide feature tables are not directly resumable.

With DAMPS enabled, `model.damps_graph_source=calibrated` (the default) builds the item graph from the initial calibrated features, hashes their contents for the cache, and saves the graph in the checkpoint. Set `raw` to use the baseline's original feature graph and cache while keeping the same calibrated auxiliary feature initialization. This isolates graph construction from feature initialization; it is not a claim of improved metrics.

## Run it

The model YAML defaults to width 64, two user–item layers, one item layer, k=10, image graph weight 0.1, and an interaction edge drop rate of 0.8.

```bash
python -m mmrecsys.cli train --config configs/experiments/freedom_baby.yaml
python -m mmrecsys.cli train --config configs/experiments/damps_freedom_baby.yaml
python -m mmrecsys.cli train --config configs/experiments/damps_freedom_baby.yaml --set model.damps_graph_source=raw
```

Adapted from the former Task 6 implementation based on enoche/FREEDOM. The model tests cover pruning, one-time calibration, the raw-graph control, and checkpoint recovery.

[Implementation](../src/mmrecsys/models/freedom.py) · [Default config](../configs/models/freedom.yaml) · [Tests](../tests/test_migrated_models.py)
