# SMORE

English · [简体中文](zh/smore.md) · [Model index](models.md)

SMORE combines spectral modality fusion with a collaborative user–item graph. It builds fixed cosine-weighted kNN graphs from the original image and text features. The fusion graph is their edge union, with the maximum weight kept for edges present in both graphs.

## Training and scoring

SMORE applies an orthonormal real FFT to both projected modalities. Three learned complex filters act on the image spectrum, text spectrum, and their elementwise product. After inverse FFT, gated item ID embeddings propagate over the corresponding item graphs; the user–item graph aggregates the resulting item features into user views. Feature-wise attention and preference dropout combine the side views, which are added to the collaborative representation.

Training uses BPR with one negative per positive, squared regularization of the final batch representations divided by twice the actual batch size, and side-information/behavior InfoNCE on users and positive items. Evaluation scores the final embeddings by dot product.

## DAMPS integration

[DAMPS](damps.md) calibrates projected features before SMORE's own spectral fusion and trains with the recommendation objective. Initialization statistics use all item projections. The three fixed graphs still come from the original features, not the calibrated ones. The baseline retains SMORE's own spectral filters.

## Run it

The model YAML uses width 64, four user–item layers, one item layer, and image/text neighborhood sizes of 40/15. Presets retain the fixed Task 4 settings.

```bash
python -m mmrecsys.cli train --config configs/experiments/smore_baby.yaml
python -m mmrecsys.cli train --config configs/experiments/damps_smore_baby.yaml
```

Adapted from the former `task4/runtime/src/models/smore.py`, based on kennethorq/SMORE (WSDM 2025). The model tests cover spectral convolution, graph fusion, and DAMPS gradients.

[Implementation](../src/mmrecsys/models/smore.py) · [Default config](../configs/models/smore.yaml) · [Tests](../tests/test_migrated_models.py)
