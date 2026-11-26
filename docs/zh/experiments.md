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

## 复现七组基线与 DAMPS 对照

准备好 Baby 数据后，在同一个 Bash 终端依次执行下列两段命令。它们在一个数据集上启动 14 次正式训练，不是单轮流程检查。对有实验预设的五个模型，每组从同一份数据集专用 YAML 出发，只切换 DAMPS 开关。消融入口在输出目录的 `summary.json` 中汇总两组测试指标。

```bash
dataset=baby
seed=999
for model in bm3 smore freedom mgcn lirdrec; do
  python -m mmrecsys.experiment.ablation \
    --config "configs/experiments/damps_${model}_${dataset}.yaml" \
    --variants baseline full --seeds "$seed" || break
done
```

PGL 与 LGMRec 需要明确选择接入方式。下面用 `official` 对照参考实现的接入方式；只有在比较加宽的表示时才改用 `concat`。每次训练的指标保存在各自运行目录的 `result.json` 中。

```bash
for model in pgl lgmrec; do
  python -m mmrecsys.cli train --model "$model" --dataset "$dataset" --seed "$seed" || break
  python -m mmrecsys.cli train --model "$model" --dataset "$dataset" --seed "$seed" \
    --set model.damps_enabled=true --set model.damps_mode=official || break
done
```

在相同数据集、种子和训练预算下比较测试集的 Recall@10/20 和 NDCG@10/20。用验证集选择 checkpoint，不按测试成绩挑选种子或接入方式。

## 配对消融与超参数搜索

```bash
python -m mmrecsys.experiment.ablation --config configs/experiments/damps_mgcn_baby.yaml --device cpu --seeds 999
python -m mmrecsys.experiment.search --dry-run
```

消融入口运行基线、完整 DAMPS 和三组单组件移除，产出 `runs.json` 和 `summary.json`，不自动恢复整个队列。[超参数搜索](search.md)按配对验证指标排名，不使用测试指标选参。

## 频谱分析

提供可信来源的 MGCN 基线与 DAMPS checkpoint，两者的数据指纹和表示形状需一致，且均须保存可训练特征。此入口在 CPU 上读取 checkpoint，不构图、不训练。

```bash
python -m mmrecsys.experiment.spectral_analysis --baseline "$BASELINE_RUN/best.pt" \
  --damps "$DAMPS_RUN/best.pt" --output runs/spectral-example --no-plots
```

执行前，先将 `BASELINE_RUN` 和 `DAMPS_RUN` 设置为相应的运行目录。`per_item.npz` 保存逐物品统计，`summary.json` 保存汇总及统计协议。如需 PNG/PDF，移除 `--no-plots`。当前图表标题写明 Clothing，生成对应图表时请使用 Clothing 的 checkpoint。
