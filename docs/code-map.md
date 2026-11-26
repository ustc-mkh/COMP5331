# Code Map

English · [简体中文](zh/code-map.md) · [Documentation](README.md)

This index covers every Python file under `src/mmrecsys/` and `tests/`. It excludes the virtual environment, data, caches, and generated runs. Update both versions when adding a code file.

## Package entry points

| File | Purpose |
| --- | --- |
| [`src/mmrecsys/__init__.py`](../src/mmrecsys/__init__.py) | Marks the top-level package. |
| [`src/mmrecsys/cli.py`](../src/mmrecsys/cli.py) | Parses train and evaluate commands and dispatches runs. |
| [`src/mmrecsys/config.py`](../src/mmrecsys/config.py) | Loads and merges YAML, parses CLI overrides, and validates the final config. |
| [`src/mmrecsys/registry.py`](../src/mmrecsys/registry.py) | Maps model names to factories, config types, required modalities, and batch specifications. |

## Data

| File | Purpose |
| --- | --- |
| [`src/mmrecsys/data/__init__.py`](../src/mmrecsys/data/__init__.py) | Marks the data package. |
| [`src/mmrecsys/data/dataset.py`](../src/mmrecsys/data/dataset.py) | Validates published interactions and features and builds train and evaluation views. |
| [`src/mmrecsys/data/sampling.py`](../src/mmrecsys/data/sampling.py) | Shuffles epoch batches and draws training-only negatives on CPU or the training device. |
| [`src/mmrecsys/data/views.py`](../src/mmrecsys/data/views.py) | Defines train data, evaluation data, and device-transferable training batches. |

## Models and neural components

| File | Purpose |
| --- | --- |
| [`src/mmrecsys/models/__init__.py`](../src/mmrecsys/models/__init__.py) | Marks the model package. |
| [`src/mmrecsys/models/base.py`](../src/mmrecsys/models/base.py) | Defines the recommender, loss output, and scoring contracts and the dot-product scorer. |
| [`src/mmrecsys/models/_shared.py`](../src/mmrecsys/models/_shared.py) | Provides shared model config checks and cached UI graph normalization. |
| [`src/mmrecsys/models/bm3.py`](../src/mmrecsys/models/bm3.py) | Implements bootstrap self-supervision and optional DAMPS on projected modalities. |
| [`src/mmrecsys/models/freedom.py`](../src/mmrecsys/models/freedom.py) | Implements a frozen item graph, degree-sensitive UI edge pruning, and optional one-time DAMPS calibration. |
| [`src/mmrecsys/models/lgmrec.py`](../src/mmrecsys/models/lgmrec.py) | Combines local propagation with a global hypergraph and implements both DAMPS integration modes. |
| [`src/mmrecsys/models/lirdrec.py`](../src/mmrecsys/models/lirdrec.py) | Implements DCT feature sharing, PWC fusion, and projection-level DAMPS. |
| [`src/mmrecsys/models/mgcn.py`](../src/mmrecsys/models/mgcn.py) | Implements behavior-guided purification, multiview graph propagation, contrastive learning, and DAMPS. |
| [`src/mmrecsys/models/pgl.py`](../src/mmrecsys/models/pgl.py) | Implements PGL local/global graph modes and both DAMPS integration modes. |
| [`src/mmrecsys/models/smore.py`](../src/mmrecsys/models/smore.py) | Implements spectral fusion, modality graphs, and optional DAMPS before native fusion. |
| [`src/mmrecsys/nn/__init__.py`](../src/mmrecsys/nn/__init__.py) | Marks the neural component package. |
| [`src/mmrecsys/nn/damps.py`](../src/mmrecsys/nn/damps.py) | Implements APC, AVRF, and IMCF calibration, fusion, and component ablations. |
| [`src/mmrecsys/nn/graph.py`](../src/mmrecsys/nn/graph.py) | Builds sparse interaction graphs and exact chunked kNN graphs, and normalizes them. |
| [`src/mmrecsys/nn/losses.py`](../src/mmrecsys/nn/losses.py) | Implements shared BPR and InfoNCE losses. |

## Training and evaluation

| File | Purpose |
| --- | --- |
| [`src/mmrecsys/engine/__init__.py`](../src/mmrecsys/engine/__init__.py) | Marks the training engine package. |
| [`src/mmrecsys/engine/checkpoint.py`](../src/mmrecsys/engine/checkpoint.py) | Saves and restores run state and checks config and dataset compatibility. |
| [`src/mmrecsys/engine/evaluator.py`](../src/mmrecsys/engine/evaluator.py) | Ranks all items in blocks, masks history, and resolves ties by item ID. |
| [`src/mmrecsys/engine/metrics.py`](../src/mmrecsys/engine/metrics.py) | Computes scalar and batched per-user Recall and NDCG. |
| [`src/mmrecsys/engine/trainer.py`](../src/mmrecsys/engine/trainer.py) | Runs epochs, optimizer updates, validation, early stopping, and checkpointing. |

## Experiment workflows

| File | Purpose |
| --- | --- |
| [`src/mmrecsys/experiment/__init__.py`](../src/mmrecsys/experiment/__init__.py) | Marks the experiment package. |
| [`src/mmrecsys/experiment/ablation.py`](../src/mmrecsys/experiment/ablation.py) | Runs paired baseline, full DAMPS, and component ablations across seeds. |
| [`src/mmrecsys/experiment/artifacts.py`](../src/mmrecsys/experiment/artifacts.py) | Hashes content, writes artifacts atomically, and caches graphs. |
| [`src/mmrecsys/experiment/logging.py`](../src/mmrecsys/experiment/logging.py) | Writes run metadata, environment manifests, and JSON Lines metrics. |
| [`src/mmrecsys/experiment/runner.py`](../src/mmrecsys/experiment/runner.py) | Selects the device and assembles single-run training, resume, or evaluation. |
| [`src/mmrecsys/experiment/search.py`](../src/mmrecsys/experiment/search.py) | Runs resumable paired MGCN grid search on validation metrics. |
| [`src/mmrecsys/experiment/seed.py`](../src/mmrecsys/experiment/seed.py) | Initializes and saves or restores Python, NumPy, PyTorch, and selected-device RNG state. |
| [`src/mmrecsys/experiment/spectral_analysis.py`](../src/mmrecsys/experiment/spectral_analysis.py) | Reads trusted MGCN checkpoints and compares spectral entropy, energy concentration, and phase variance. |

## Tests

| File | Purpose |
| --- | --- |
| [`tests/conftest.py`](../tests/conftest.py) | Defines shared tiny data/config fixtures and limits compute threads. |
| [`tests/test_backbone_integrations_pgl_lgmrec.py`](../tests/test_backbone_integrations_pgl_lgmrec.py) | Tests PGL/LGMRec graph behavior, both DAMPS modes, gradients, and checkpoints. |
| [`tests/test_checkpoint_copy.py`](../tests/test_checkpoint_copy.py) | Checks atomic copy, failure handling, and serialization count. |
| [`tests/test_damps.py`](../tests/test_damps.py) | Compares DAMPS with reference equations and tests gradients, initialization, and state. |
| [`tests/test_data.py`](../tests/test_data.py) | Checks split isolation, invalid data, and CPU/device sampler behavior. |
| [`tests/test_device.py`](../tests/test_device.py) | Checks device selection, RNG isolation, and CUDA resume when available. |
| [`tests/test_evaluator.py`](../tests/test_evaluator.py) | Checks history masking, ties, padding, and batched metrics. |
| [`tests/test_experiment.py`](../tests/test_experiment.py) | Checks config precedence, reproducibility, and end-to-end training, ablation, and resume. |
| [`tests/test_graph.py`](../tests/test_graph.py) | Checks graph normalization, exact kNN, and cache identity. |
| [`tests/test_lirdrec.py`](../tests/test_lirdrec.py) | Checks LIRDRec against a NumPy reference and tests PWC and DAMPS gradients. |
| [`tests/test_lirdrec_experiment.py`](../tests/test_lirdrec_experiment.py) | Exercises LIRDRec training, evaluation, and exact epoch-boundary resume on tiny data. |
| [`tests/test_mgcn.py`](../tests/test_mgcn.py) | Compares MGCN equations and gradients with a dense reference and checks InfoNCE. |
| [`tests/test_migrated_models.py`](../tests/test_migrated_models.py) | Verifies BM3, SMORE, and FREEDOM losses, graphs, DAMPS paths, and resume. |
| [`tests/test_search.py`](../tests/test_search.py) | Checks grid expansion, validation-only ranking, and interrupted search recovery. |
| [`tests/test_spectral_analysis.py`](../tests/test_spectral_analysis.py) | Checks spectral entropy, energy ratios, phase variance, and zero-energy rejection. |

## Configuration

| File | Purpose |
| --- | --- |
| [`pyproject.toml`](../pyproject.toml) | Declares Python requirements, dependencies, the CLI entry point, and pytest settings. |
| [`configs/default.yaml`](../configs/default.yaml) | Sets shared data, training, evaluation, and runtime defaults. |
| [`configs/datasets/`](../configs/datasets/) | One YAML per dataset; selects the dataset name and relative directory. |
| [`configs/models/`](../configs/models/) | One YAML per registered backbone; defines model and optimizer defaults. |
| [`configs/experiments/`](../configs/experiments/) | Baseline and DAMPS presets for each BM3, SMORE, FREEDOM, MGCN, and LIRDRec dataset pair. |
| [`configs/search/damps_mgcn.yaml`](../configs/search/damps_mgcn.yaml) | Defines the default three-setting MGCN search. |
| [`configs/search/damps_mgcn_extended.yaml`](../configs/search/damps_mgcn_extended.yaml) | Defines the optional wider MGCN search. |
