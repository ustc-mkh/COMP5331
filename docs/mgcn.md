# MGCN Backbone and Baseline

English · [简体中文](zh/mgcn.md) · [Documentation](README.md)

The MGCN implementation follows *Multi-View Graph Convolutional Network for Multimedia Recommendation* (ACM MM 2023) and the [MMRec implementation](https://github.com/enoche/MMRec/blob/master/src/models/mgcn.py). MMRec is not required at runtime. Use `mgcn_<dataset>.yaml` for the baseline and `damps_mgcn_<dataset>.yaml` for the paired [DAMPS integration](damps.md).

## Current configuration

`configs/models/mgcn.yaml` sets embedding width 256, two user–item graph layers, one item graph layer, k=10, InfoNCE temperature 0.2, contrastive weight 0.01, and regularization weight 0.0001. The same backbone defaults apply to Baby, Sports, Clothing, Elec, and MicroLens. These are repository defaults, not paper-reported or dataset-optimized settings.

Shared training defaults are seed 999, Adam at learning rate 0.001 with no weight decay, decay factor `0.96^(epoch/50)`, batch size 2048, up to 1000 epochs, and validation each epoch with patience 20 on Recall@20. The project keeps the published split and reports Recall and NDCG at 10 and 20. The [search](search.md) varies contrastive weight on validation data.

## Equation mapping

| MGCN equations | Implementation |
| --- | --- |
| (1)–(2), behavior-guided purifier | Project each modality and use a sigmoid gate to modulate the item ID embedding. |
| (3)–(5), user–item view | Build a symmetrically normalized, self-loop-free graph from training edges and average ID representations from layer 0 through layer L. |
| (6)–(10), item view and user modality features | Propagate purified representations on cosine kNN graphs and aggregate items using the normalized user–item block. |
| (11)–(15), behavior-aware fusion | Extract a common representation with attention and gate modality residuals with behavior features. The author variant divides common plus summed residuals by three. |
| (16)–(19), objective and scoring | Use directional in-batch InfoNCE on users and positive items, BPR with dot-product scores, and regularization of the final batch representations. |

The default regularizer divides the summed squared user, positive-item, and negative-item representations by twice the actual batch size. InfoNCE uses stable cross-entropy over normalized vectors; repeated samples keep their positions. Features are trainable by default, but the fixed kNN graph uses the initial released features.

## Graphs and reproducibility

MGCN uses directed, cosine-weighted kNN; self-neighbors count toward k and symmetric normalization does not symmetrize edge directions. Same-score neighbors favor lower IDs. Zero vectors produce zero cosine scores. The graph builder does not clip negative cosine weights silently; it rejects negative weighted degrees when symmetric normalization would be undefined.

Exact kNN is quadratic in item count, but similarities are computed in chunks rather than stored as a full matrix. For a lower peak memory requirement, set `model.knn_chunk_size=128`. The default build runs on CPU and the model transfers its tensors to the requested device. The [architecture guide](architecture.md) covers cache identity and resume behavior.

```bash
python -m mmrecsys.cli train --config configs/experiments/mgcn_baby.yaml --device cpu --set train.epochs=2
CUDA_VISIBLE_DEVICES=2 python -m mmrecsys.cli train --config configs/experiments/damps_mgcn_baby.yaml --device cuda:0
```

The second command requires a system with GPU 2 available and maps it to `cuda:0` inside the process. Short runs and unit tests check implementation and workflow; they do not establish reproduction of paper metrics.
