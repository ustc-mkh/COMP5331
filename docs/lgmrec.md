# LGMRec

English · [简体中文](zh/lgmrec.md) · [Model index](models.md)

LGMRec combines collaborative ID embeddings, local modality propagation, and global modality hypergraphs. It averages the collaborative user–item layers including layer zero. Each local branch projects item features, initializes users by averaging their interacted items, and propagates on the user–item graph.

## Training and scoring

Each global branch learns modality-specific hyperedge assignment logits. It aggregates item logits into user logits and uses Gumbel-Softmax to form assignments. The global branches exchange messages through the hyperedges and contribute a normalized sum weighted by `model.alpha`. Assignment dropout is training-only, but Gumbel sampling also occurs at evaluation.

Training uses BPR with one negative per positive, a cross-modal hypergraph contrastive loss for users and positive items, and batch embedding norm regularization. The contrastive denominator uses all corresponding text-side users or items, computed in chunks. Evaluation scores the fused representations by dot product.

## DAMPS integration

| Setting | Local and global features | Scoring width |
| --- | --- | ---: |
| Baseline | Projects fixed original features locally and uses those original features for global assignments. | D |
| `concat` | Concatenates four normalized local branches: original image/text and calibrated image/text. Global assignments still use original features; the collaborative and hypergraph outputs expand to match the local width. Calibration projections receive gradients. | 4D |
| `official` | Runs shared DAMPS once at initialization. The resulting features are frozen and feed both local projections and global assignments. The calibrator is not called during training. | D |

D is `model.embedding_dim`. The `official` path reuses this project's [DAMPS](damps.md) implementation. Checkpoints across the three settings are not interchangeable. Compare `concat` with care because it changes width and parameter count as well as calibration.

## Run it

The model YAML defaults to D=64, two collaborative layers, two local modality layers, one hypergraph layer, four hyperedges per modality, and a global weight of 0.3.

```bash
python -m mmrecsys.cli train --model lgmrec --dataset baby
python -m mmrecsys.cli train --model lgmrec --dataset baby --set model.damps_enabled=true
python -m mmrecsys.cli train --model lgmrec --dataset baby --set model.damps_enabled=true --set model.damps_mode=official
```

The integration tests cover hypergraph inputs, all four `concat` channels, gradient paths, and checkpoint reload.

[Implementation](../src/mmrecsys/models/lgmrec.py) · [Default config](../configs/models/lgmrec.yaml) · [Tests](../tests/test_backbone_integrations_pgl_lgmrec.py)
