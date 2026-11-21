# PGL

[English](../pgl.md) · 中文 · [模型索引](models.md)

PGL 在用户–物品交互子图上训练，在完整训练交互图上评估。模型为两种模态分别构建二值 kNN 图，归一化后混合成物品图。

## 图模式与训练

- `model.mode=local`（默认）：每轮用端点度数的逆平方根乘积作为采样权重，重新采样子图；默认保留 0.3 的交互边。
- `model.mode=global`：使用归一化交互图的缓存稀疏重建。重建时将较大与较小的奇异值配对相乘，并保留绝对权重达到 `model.global_threshold` 的元素。

编码器对包含第 0 层的用户–物品传播结果取平均，再将物品图传播结果加到物品表示上。训练采用每个正例一个负例的 BPR，并可加入两份 Dropout 视图之间的 InfoNCE。默认 `model.ssl_weight=0.0`，对比项不计入总损失。评估使用用户与物品表示的内积。

## DAMPS 接入

| 设置 | 模态通道及物品图 | 打分维度 |
| --- | --- | ---: |
| 基线 | 原始图文两路；用原始特征构建 kNN。 | 2D |
| `concat` | 追加两路校准特征及对应的两张用户嵌入表。各物品通道分别归一化后拼接；kNN 仍使用原始特征。 | 4D |
| `official` | 用初始化校准特征构建 kNN，每次前向调用共享 DAMPS。算子使用内部特征表和投影，外部模态投影不接收梯度。 | 2D |

D 为 `model.embedding_dim`。`official` 参考上游接入流程，使用本项目的 [DAMPS](damps.md) 算子，并非另一份实现。不同模式不要混用 checkpoint。`concat` 也增加了表示宽度和参数量，不能单独衡量校准的作用。

## 运行

模型 YAML 默认 D=64、2 层用户–物品图、1 层物品图、k=10。以下分别运行基线、默认 `concat` 和参考接入模式：

```bash
python -m mmrecsys.cli train --model pgl --dataset baby
python -m mmrecsys.cli train --model pgl --dataset baby --set model.damps_enabled=true
python -m mmrecsys.cli train --model pgl --dataset baby --set model.damps_enabled=true --set model.damps_mode=official
```

集成测试覆盖局部与全局模式、图结构、梯度路径和 checkpoint 重载。

[实现](../../src/mmrecsys/models/pgl.py) · [默认配置](../../configs/models/pgl.yaml) · [测试](../../tests/test_backbone_integrations_pgl_lgmrec.py)
