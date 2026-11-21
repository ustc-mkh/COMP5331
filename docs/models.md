# Model Guides

English · [简体中文](zh/models.md) · [Documentation](README.md)

Each supported backbone has its own guide with implementation details, [DAMPS](damps.md) integration, and training examples.

| Model | Guide focus |
| --- | --- |
| [BM3](bm3.md) | Positive-only bootstrap training; optional calibration of projected modality features. |
| [SMORE](smore.md) | Spectral fusion with fixed modality graphs; DAMPS before native spectral fusion. |
| [FREEDOM](freedom.md) | Fixed item graph and interaction pruning; one-time DAMPS and a raw-graph control. |
| [MGCN](mgcn.md) | Multiview graph convolution and behavior-guided purification. |
| [LIRDRec](lirdrec.md) | DCT feature sharing and PWC fusion; includes the full experiment queue. |
| [PGL](pgl.md) | Local/global training subgraphs; two DAMPS integration modes. |
| [LGMRec](lgmrec.md) | Local modality propagation and global hypergraphs; two DAMPS integration modes. |

## Shared protocol

All models use the same released splits, train-history filtering, validation-based checkpoint selection, and all-item evaluation. Ranking breaks ties by item ID and reports user-macro-averaged metrics without rounding before aggregation. The evaluator handles held-out labels; model factories only receive training views. The trainer uses the shared sampler and early-stopping rule, and reloads the best validation checkpoint before test evaluation.

[Running experiments](experiments.md) explains how to select the preset and run ablations. See the [architecture guide](architecture.md) for data and checkpoint contracts. The unified implementations do not depend on the old standalone runners, and their checkpoints are not interchangeable with archived standalone runs.
