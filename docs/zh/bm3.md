# BM3

[English](../bm3.md) · 中文 · [模型索引](models.md)

BM3 只用用户–物品正交互训练，不需要负采样。模型对归一化用户–物品图上从第 0 层到最后一层的表示取平均，并加入物品 ID 残差。在线分支使用共享预测器；目标分支对同一表示使用 Dropout 并停止梯度，不使用独立的滑动平均模型。

## 训练与打分

自举损失用余弦距离双向对齐用户和交互物品。模态损失将图像和文本表示分别与物品目标及各自的 Dropout 目标对齐。正则项在全部用户和物品表示上计算 `(‖U‖₂ + ‖I‖₂) / n_items`，不是平方 L2，也不是批次 BPR 正则。评估使用预测器输出的用户与物品表示的内积；模态特征影响训练，但没有单独的打分分支。

## DAMPS 接入

可选的 [DAMPS](damps.md) 在生成模态目标和在线预测之前校准投影后的图像与文本特征。初始化统计只在全体物品上估计一次，梯度经由在线模态分支传回。协同图与打分公式不变。

## 运行

模型 YAML 默认使用 64 维嵌入、1 层图传播、0.5 的 Dropout 和 2.0 的模态损失权重。逐数据集覆盖项请使用实验 YAML；这些预设沿用原独立实现的模型设置。

```bash
python -m mmrecsys.cli train --config configs/experiments/bm3_baby.yaml
python -m mmrecsys.cli train --config configs/experiments/damps_bm3_baby.yaml
```

本实现移植自 enoche/BM3 的 `ef82a333823ae01ac5964039720f697532b40ea3` 提交（GPL-3.0）。模型测试覆盖自举公式、打分、梯度和断点恢复。

[实现](../../src/mmrecsys/models/bm3.py) · [默认配置](../../configs/models/bm3.yaml) · [测试](../../tests/test_migrated_models.py)
