# 运行实验

[English](../experiments.md) · 中文 · [文档首页](README.md)

先按 [Linux 快速入门](../../README.zh-CN.md)配置环境，再在仓库根目录执行以下命令。

## 选择模型和数据集

BM3、SMORE、FREEDOM、MGCN 和 LIRDRec 使用 `configs/experiments/<model>_<dataset>.yaml` 运行基线，文件名加 `damps_` 前缀则启用 DAMPS。五个数据集预设为 `baby`、`sports`、`clothing`、`elec` 和 `microlens`。需要逐数据集覆盖项时，优先使用实验 YAML。

```bash
python -m mmrecsys.cli train --config configs/experiments/bm3_baby.yaml
python -m mmrecsys.cli train --config configs/experiments/damps_smore_sports.yaml
python -m mmrecsys.cli train --config configs/experiments/damps_freedom_baby.yaml --set model.damps_graph_source=raw
```

### PGL 和 LGMRec

使用 `--model` 和 `--dataset` 选择默认配置。DAMPS 默认关闭，启用后默认使用 `concat`。

```bash
python -m mmrecsys.cli train --model pgl --dataset baby
python -m mmrecsys.cli train --model lgmrec --dataset baby --set model.damps_enabled=true
```

参见 [PGL](pgl.md) 和 [LGMRec](lgmrec.md) 的独立指南，了解 `concat` 与 `official` 模式的区别和完整命令。

## 配对消融与超参数搜索

```bash
python -m mmrecsys.experiment.ablation --config configs/experiments/damps_mgcn_baby.yaml --device cpu --seeds 999
python -m mmrecsys.experiment.search --dry-run
```

消融入口运行基线、完整 DAMPS 和三组单组件移除，产出 `runs.json` 和 `summary.json`，不自动恢复整个队列。需要可恢复的五数据集 LIRDRec 协议时，使用 [LIRDRec 套件](lirdrec.md#完整实验协议与队列)。[超参数搜索](search.md)按配对验证指标排名，不使用测试指标选参。

## MGCN 多 GPU 对比

```bash
python -m mmrecsys.experiment.compare --dry-run
python -m mmrecsys.experiment.compare --datasets baby sports --gpus cpu --set train.epochs=2
python -m mmrecsys.experiment.compare --mode full --gpus 0 1 --seeds 999 2024 2025
```

默认对比 Baby、Sports、Clothing、Elec 和 MicroLens，每个数据集及种子运行两组，即每个种子共 10 组。快速模式最多 20 轮、早停耐心值为 5；完整模式最多 1000 轮、早停耐心值为 20。每张 GPU 同时运行一个独立任务，不是分布式训练。在共享机器上请显式选择空闲 GPU。编号相对于可见设备；例如 `CUDA_VISIBLE_DEVICES=4,5` 时传入 `--gpus 0 1`。预览不读取数据或初始化 GPU。

使用 `--resume YOUR_COMPARE_DIRECTORY --gpus 0 1` 跳过已完成任务，并从头重跑未完成任务。`runs.json` 记录状态及路径；`summary.md`、`summary.csv` 和 `summary.json` 报告完整种子配对的 Recall/NDCG@10/20。绝对提升为 DAMPS 减去基线，相对提升用均值差除以基线均值。另有逐种子相对提升的均值，该项排除基线为零的种子。两轮运行只是流程检查，不代表收敛。

## 频谱分析

提供可信来源的 MGCN 基线与 DAMPS checkpoint，两者的数据指纹和表示形状需一致，且均须保存可训练特征。此入口在 CPU 上读取 checkpoint，不构图、不训练。

```bash
python -m mmrecsys.experiment.spectral_analysis --baseline "$BASELINE_RUN/best.pt" \
  --damps "$DAMPS_RUN/best.pt" --output runs/spectral-example --no-plots
```

执行前，先将 `BASELINE_RUN` 和 `DAMPS_RUN` 设置为相应的运行目录。`per_item.npz` 保存逐物品统计，`summary.json` 保存汇总及统计协议。如需 PNG/PDF，使用 pip 安装 `matplotlib` 并移除 `--no-plots`。当前图表标题写明 Clothing，生成对应图表时请使用 Clothing 的 checkpoint。
