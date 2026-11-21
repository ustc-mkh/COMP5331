# COMP5331 · DAMPS Reproduction

English · [简体中文](README.zh-CN.md)

This COMP5331 group project aims to reproduce the paper *Enhancing Multimodal Recommendation via Multimodal Representation Calibration in Spectral Domain*. We implement DAMPS, the paper's method for calibrating image and text representations in the spectral domain, and compare multimodal recommenders with and without it under a common training and evaluation protocol.

The project supports BM3, SMORE, FREEDOM, MGCN, LIRDRec, PGL, and LGMRec. The implementation and experiments are part of the reproduction effort; they do not establish that the paper's reported results have been reproduced.

## 1. Set up the environment

The setup below assumes Linux, Bash, and a local checkout. Use Python **3.10 or newer** with `venv` and `pip`. If `python3 -m venv` is unavailable on Debian or Ubuntu, install the `python3-venv` package for your Python version.

From the repository root:

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

- **CPU only:** Install the CPU build with `python -m pip install 'torch>=2.1' --index-url https://download.pytorch.org/whl/cpu`.
- **NVIDIA GPU:** Run `nvidia-smi`, then use the [PyTorch installation selector](https://pytorch.org/get-started/locally/) for Linux and a CUDA build compatible with your driver and Python version. The project requires PyTorch 2.1 or newer.

Then install the project in editable mode. This installs the remaining runtime dependencies (NumPy, SciPy, and PyYAML) and the optional test dependency (pytest):

```bash
python -m pip install -e '.[test]'
python -c 'import torch; print(torch.__version__, "CUDA available:", torch.cuda.is_available())'
python -m pytest -q
```

The tests use small fixtures and do not require the published datasets. For training only, use `python -m pip install -e .` instead. Activate `.venv` again when opening a new shell.

## 2. Prepare a dataset

Download and extract the preprocessed data from [MMRec's data page](https://github.com/enoche/MMRec/tree/master/data). To run the examples below, download **Baby** and arrange it as follows:

```text
data/
└── baby/
    ├── baby.inter
    ├── image_feat.npy
    └── text_feat.npy
```

Other dataset keys are `sports`, `clothing`, `elec` (Electronics), and `microlens`. Do not renumber IDs, reorder feature rows, or regenerate the splits. The CLI does not download data. If the dataset folders live elsewhere, add `--set data.root=/absolute/path/to/data` to the training command. See the [data guide](docs/data.md) for the file format.

## 3. Run training

From the repository root with the virtual environment active:

```bash
# CPU smoke run: one epoch on the full Baby dataset
python -m mmrecsys.cli train --config configs/experiments/mgcn_baby.yaml --device cpu --set train.epochs=1

# Paired baseline and DAMPS runs using repository presets
python -m mmrecsys.cli train --config configs/experiments/mgcn_baby.yaml
python -m mmrecsys.cli train --config configs/experiments/damps_mgcn_baby.yaml

# Example: PGL + DAMPS via model and dataset defaults
python -m mmrecsys.cli train --model pgl --dataset baby --set model.damps_enabled=true
```

Training selects CUDA automatically when available and otherwise uses CPU. Use `--device cpu` or `--device cuda:0` to override it; GPU indices refer to devices visible to the process. The first run may spend time building item graphs; later runs can reuse `cache/`. One epoch checks the workflow, not convergence.

Use `--set section.key=value` to override fields. The same CLI is available as `mmrecsys` after installation. See the [experiment guide](docs/experiments.md) for other models, ablations, multi-GPU comparisons, search, and spectral analysis.

## 4. Inspect, resume, or evaluate a run

Training prints the run path and saves `config.yaml`, `manifest.json`, `metrics.jsonl`, `best.pt`, `last.pt`, and `result.json` under `runs/`. Validation selects `best.pt`; test evaluation uses that checkpoint.

Replace `YOUR_RUN_DIRECTORY` with the directory printed by training (for example, `runs/mgcn-baby-...`):

```bash
RUN_DIR="YOUR_RUN_DIRECTORY"
python -m mmrecsys.cli train --resume "$RUN_DIR/last.pt" --set train.epochs=1000
python -m mmrecsys.cli evaluate --run "$RUN_DIR" --split test
```

Resume from the original run directory with both checkpoints intact. Set the epoch limit above the number already completed if you want to extend training; resume does not reset early stopping. Use only trusted checkpoints.

## Documentation

- [Documentation index](docs/README.md) · [中文文档](docs/zh/README.md)
- [Model guides](docs/models.md) · [Source and test file reference](docs/code-map.md)
- [Architecture](docs/architecture.md) · [DAMPS implementation](docs/damps.md)
