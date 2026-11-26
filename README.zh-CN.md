# COMP5331 · DAMPS 复现项目

[English](README.md) · 简体中文

这是 COMP5331 的小组项目，目标是复现论文《Enhancing Multimodal Recommendation via Multimodal Representation Calibration in Spectral Domain》。我们实现了论文提出的 DAMPS 方法，在频域校准图像和文本表示，并在统一的训练与评估协议下比较启用与不启用 DAMPS 的多模态推荐模型。

项目支持 BM3、SMORE、FREEDOM、MGCN、LIRDRec、PGL 和 LGMRec。当前实现和实验用于验证复现效果，并不代表已达到论文报告的结果。

## 1. 配置环境

使用 Python 3.10 或更高版本，在仓库根目录执行以下命令，安装训练、评估和绘图所需的依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## 2. 准备数据集

请分别从 [Google Drive 文件夹 1](https://drive.google.com/drive/folders/13cBy1EA_saTUuXxVllKgtfci2A09jyaG) 和 [Google Drive 文件夹 2](https://drive.google.com/drive/folders/14UyTAh_YyDV8vzXteBJiy9jv8TBDK43w) 下载预处理数据。两个文件夹合起来包含全部五个数据集：Baby、Sports、Clothing、Electronics（`elec`）和 MicroLens。解压后放在 `data/` 下，以下以 **Baby** 为例：

```text
data/
└── baby/
    ├── baby.inter
    ├── image_feat.npy
    └── text_feat.npy
```

其他数据集的配置名是 `sports`、`clothing`、`elec`（Electronics）和 `microlens`。不要重排编号、特征行或重新划分数据。训练入口不会自动下载数据。如果数据集目录存放在其他位置，可在训练命令中添加 `--set data.root=/absolute/path/to/data`。文件格式见[数据指南](docs/zh/data.md)。

## 3. 运行训练

激活虚拟环境后，在仓库根目录执行：

```bash
# CPU 流程检查：在完整 Baby 数据上训练一轮
python -m mmrecsys.cli train --config configs/experiments/mgcn_baby.yaml --device cpu --set train.epochs=1

# 使用仓库提供的实验配置，分别训练基线和 DAMPS 版本
python -m mmrecsys.cli train --config configs/experiments/mgcn_baby.yaml
python -m mmrecsys.cli train --config configs/experiments/damps_mgcn_baby.yaml

# 示例：通过模型与数据集的默认配置运行 PGL + DAMPS
python -m mmrecsys.cli train --model pgl --dataset baby --set model.damps_enabled=true
```

程序默认优先使用 CUDA，不可用时使用 CPU。可通过 `--device cpu` 或 `--device cuda:0` 指定设备；GPU 编号相对于当前进程可见的设备。首次运行可能需要花时间构建物品图，后续运行可复用 `cache/`。一轮训练只用于检查流程，不代表收敛。

使用 `--set section.key=value` 覆盖配置字段。安装后也可用 `mmrecsys` 命令调用同一入口。完整命令见[复现七组基线与 DAMPS 对照](docs/zh/experiments.md#复现七组基线与-damps-对照)；[实验指南](docs/zh/experiments.md)也介绍消融、搜索和频谱分析。

## 4. 查看产物、恢复或评估

训练会打印运行路径，并在 `runs/` 下保存 `config.yaml`、`manifest.json`、`metrics.jsonl`、`best.pt`、`last.pt` 和 `result.json`。程序根据验证集选出 `best.pt`，再用该 checkpoint 评估测试集。

将 `YOUR_RUN_DIRECTORY` 替换为训练打印的目录（例如 `runs/mgcn-baby-...`）：

```bash
RUN_DIR="YOUR_RUN_DIRECTORY"
python -m mmrecsys.cli train --resume "$RUN_DIR/last.pt" --set train.epochs=1000
python -m mmrecsys.cli evaluate --run "$RUN_DIR" --split test
```

恢复时保留原运行目录及其中的两个 checkpoint。若要延长训练，轮数上限需大于已完成轮数；恢复不会重置早停状态。只加载可信来源的 checkpoint。

## 文档

- [中文文档索引](docs/zh/README.md) · [English docs](docs/README.md)
- [模型文档](docs/zh/models.md) · [源码与测试文件索引](docs/zh/code-map.md)
- [项目架构](docs/zh/architecture.md) · [DAMPS 实现](docs/zh/damps.md)
