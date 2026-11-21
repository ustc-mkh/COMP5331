# LGMRec

[English](../lgmrec.md) · 中文 · [模型索引](models.md)

LGMRec 结合协同 ID 表示、局部模态传播与全局模态超图。协同分支对包含第 0 层的用户–物品表示取平均。局部分支先投影物品特征，再对用户交互过的物品取平均得到初始用户表示，随后在用户–物品图上传播。

## 训练与打分

各全局分支学习对应模态的物品超边分配 logits，聚合成用户 logits，并通过 Gumbel-Softmax 得到分配。全局分支经超边传递消息，其输出相加、归一化后由 `model.alpha` 加权。仅训练时对分配执行 Dropout，但评估时仍进行 Gumbel 采样。

训练使用每个正例一个负例的 BPR、用户和正物品上的跨模态超图对比损失，以及批次嵌入范数正则。对比损失的分母覆盖文本侧的全部用户或全部物品，分块计算。评估使用融合表示的内积。

## DAMPS 接入

| 设置 | 局部与全局特征 | 打分维度 |
| --- | --- | ---: |
| 基线 | 局部分支投影固定的原始特征；全局分配也使用原始特征。 | D |
| `concat` | 拼接四路归一化后的局部表示：原始图文两路及校准图文两路。全局分配仍采用原始特征；协同和超图输出扩展到与局部表示相同的宽度。校准投影可以接收梯度。 | 4D |
| `official` | 初始化时运行一次共享 DAMPS。所得特征冻结后用于局部投影和全局分配；训练过程中不调用校准器。 | D |

D 为 `model.embedding_dim`。`official` 复用本项目的 [DAMPS](damps.md) 算子。三种设置的 checkpoint 不可互换。`concat` 同时改变表示宽度与参数量，对比时不能把效果仅归因于校准。

## 运行

模型 YAML 默认 D=64、2 层协同传播、2 层局部模态传播、1 层超图传播、每个模态 4 条超边、全局权重为 0.3。

```bash
python -m mmrecsys.cli train --model lgmrec --dataset baby
python -m mmrecsys.cli train --model lgmrec --dataset baby --set model.damps_enabled=true
python -m mmrecsys.cli train --model lgmrec --dataset baby --set model.damps_enabled=true --set model.damps_mode=official
```

集成测试覆盖超图输入、`concat` 的四个通道、梯度路径和 checkpoint 重载。

[实现](../../src/mmrecsys/models/lgmrec.py) · [默认配置](../../configs/models/lgmrec.yaml) · [测试](../../tests/test_backbone_integrations_pgl_lgmrec.py)
