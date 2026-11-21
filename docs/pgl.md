# PGL

English · [简体中文](zh/pgl.md) · [Model index](models.md)

PGL trains on a subgraph of the user–item interactions and evaluates on the full training interaction graph. It builds a separate binary kNN graph for each modality and mixes the two normalized item graphs.

## Graph modes and training

- `model.mode=local` (default) samples a new subgraph each epoch using inverse-square-root endpoint-degree weights. The default sampled edge fraction is 0.3.
- `model.mode=global` uses a cached sparse reconstruction of the normalized interaction graph. The reconstruction combines leading and trailing singular values and keeps entries whose absolute weight reaches `model.global_threshold`.

The encoder averages user–item propagation layers including layer zero, then adds item-graph context to the item representation. Training uses BPR with one negative per positive and optional InfoNCE between pairs of dropout views. The default `model.ssl_weight=0.0` makes the contrastive term contribute zero. Evaluation scores user and item embeddings by dot product.

## DAMPS integration

| Setting | Modality channels and item graph | Scoring width |
| --- | --- | ---: |
| Baseline | Original image/text channels; kNN from raw features. | 2D |
| `concat` | Appends two calibrated channels and two corresponding user embedding tables. Normalizes each item channel before concatenation. kNN still uses raw features. | 4D |
| `official` | Builds kNN from the initial calibrated features, then uses the shared DAMPS operator during each forward pass. The operator uses its own feature tables and projections; the outer modality projections do not receive gradients. | 2D |

D is `model.embedding_dim`. `official` follows the reference integration pattern using this project's [DAMPS](damps.md), not a separate implementation. Do not mix checkpoints across modes. The wider `concat` representation also changes parameter count, so it does not isolate calibration.

## Run it

The model YAML uses D=64, two user–item layers, one item layer, and k=10. Run the baseline, default `concat`, or reference-style integration:

```bash
python -m mmrecsys.cli train --model pgl --dataset baby
python -m mmrecsys.cli train --model pgl --dataset baby --set model.damps_enabled=true
python -m mmrecsys.cli train --model pgl --dataset baby --set model.damps_enabled=true --set model.damps_mode=official
```

The integration tests cover local/global modes, graph behavior, gradient paths, and checkpoint reload.

[Implementation](../src/mmrecsys/models/pgl.py) · [Default config](../configs/models/pgl.yaml) · [Tests](../tests/test_backbone_integrations_pgl_lgmrec.py)
