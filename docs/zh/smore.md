# SMORE

[English](../smore.md) · 中文 · [模型索引](models.md)

SMORE 将频域模态融合与用户–物品协同图结合。模型从原始图像和文本特征分别构造固定的余弦加权 kNN 图。融合图取两图的边并集，重合边保留较大的权重。

## 训练与打分

SMORE 对投影后的两种模态执行正交归一化实数 FFT，分别用三个可训练的复数滤波器处理图像频谱、文本频谱及两者的逐元素乘积。逆 FFT 后，经门控的物品 ID 表示沿对应物品图传播，再通过用户–物品图聚合成用户侧视图。逐特征注意力和偏好 Dropout 融合侧信息，所得表示与协同表示相加。

训练使用每个正例一个负例的 BPR、最终批次表示的平方正则（分母为实际批次大小的两倍），以及用户和正物品上的侧信息–行为 InfoNCE。评估取最终表示的内积。

## DAMPS 接入

[DAMPS](damps.md) 在 SMORE 原生频域融合前校准投影特征，并随推荐目标训练。初始化统计使用全体物品投影。三个固定图仍基于原始特征，而非校准特征。关闭 DAMPS 的基线仍保留 SMORE 自身的频域滤波。

## 运行

模型 YAML 默认使用 64 维嵌入、4 层用户–物品图、1 层物品图，图像和文本邻居数分别为 40 和 15。实验预设沿用 Task 4 的固定设置。

```bash
python -m mmrecsys.cli train --config configs/experiments/smore_baby.yaml
python -m mmrecsys.cli train --config configs/experiments/damps_smore_baby.yaml
```

本实现移植自原 `task4/runtime/src/models/smore.py`，参考 kennethorq/SMORE（WSDM 2025）。模型测试覆盖频域卷积、融合图和 DAMPS 梯度。

[实现](../../src/mmrecsys/models/smore.py) · [默认配置](../../configs/models/smore.yaml) · [测试](../../tests/test_migrated_models.py)
