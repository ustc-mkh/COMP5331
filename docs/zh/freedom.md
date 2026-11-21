# FREEDOM

[English](../freedom.md) · 中文 · [模型索引](models.md)

FREEDOM 使用固定的多模态物品图，并对训练用的用户–物品图剪枝。模型按余弦相似度选择邻居，对选中的边赋二值权重后归一化，再加权混合图像图和文本图。每轮训练使用端点度数的逆平方根乘积作为采样权重，重新采样交互边；评估使用完整训练交互图。

## 训练与打分

模型将物品 ID 在物品图上的传播结果加到协同物品表示上。主 BPR 损失为每个用户比较一个正例和一个负例。两项辅助 BPR 分别用用户表示对投影后的图像、文本特征打分。`model.reg_weight` 对辅助损失之和加权，在此实现中不是 L2 惩罚。评估使用协同表示与物品图表示，不直接使用辅助投影。

## DAMPS 接入与原始图对照

[DAMPS](damps.md) 仅在初始化运行。先在嵌入维度 D 上校准，再通过逆 FFT 分别还原至各模态的原始输入宽度。模型将输出复制到独立的可训练特征表，随后投影回 D 维用于辅助损失；校准器冻结。特征表为 D 维的旧 FREEDOM+DAMPS checkpoint 不能直接恢复。

启用 DAMPS 时，默认 `model.damps_graph_source=calibrated` 使用初始化校准特征构建物品图，以实际特征内容生成缓存键，并将图保存到 checkpoint。设为 `raw` 则复用基线的原始特征构图及缓存，同时保留相同的辅助特征校准初始化。这可分离构图与特征初始化的影响，不代表指标一定提高。

## 运行

模型 YAML 默认使用 64 维嵌入、2 层用户–物品图、1 层物品图、k=10、图像图权重 0.1 和交互边丢弃率 0.8。

```bash
python -m mmrecsys.cli train --config configs/experiments/freedom_baby.yaml
python -m mmrecsys.cli train --config configs/experiments/damps_freedom_baby.yaml
python -m mmrecsys.cli train --config configs/experiments/damps_freedom_baby.yaml --set model.damps_graph_source=raw
```

本实现移植自原 Task 6 中参考 enoche/FREEDOM 的版本。模型测试覆盖剪枝、一次性校准、原始图对照与断点恢复。

[实现](../../src/mmrecsys/models/freedom.py) · [默认配置](../../configs/models/freedom.yaml) · [测试](../../tests/test_migrated_models.py)
