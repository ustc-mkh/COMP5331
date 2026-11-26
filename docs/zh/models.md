# 模型文档

[English](../models.md) · 中文 · [文档首页](README.md)

每个推荐模型都有独立页面，介绍具体实现、[DAMPS](damps.md) 接入方式及运行示例。

| 模型 | 文档重点 |
| --- | --- |
| [BM3](bm3.md) | 仅用正例的自举训练；可选投影后模态校准。 |
| [SMORE](smore.md) | 固定模态图与原生频域融合；DAMPS 接在融合之前。 |
| [FREEDOM](freedom.md) | 固定物品图与交互剪枝；一次性 DAMPS 校准及原始图对照。 |
| [MGCN](mgcn.md) | 多视图图卷积与行为引导净化。 |
| [LIRDRec](lirdrec.md) | DCT 特征共享与 PWC 融合。 |
| [PGL](pgl.md) | 局部或全局训练子图；支持两种 DAMPS 接入方式。 |
| [LGMRec](lgmrec.md) | 局部模态传播与全局超图；支持两种 DAMPS 接入方式。 |

## 统一协议

所有模型共用发布划分、训练历史过滤、验证集选模和全物品评估。排序同分时按物品 ID 处理，指标按用户宏平均，汇总前不提前舍入。评估器管理验证和测试标签，模型工厂只接收训练视图。训练器使用统一的采样及早停规则，并在测试前重新加载验证集最佳 checkpoint。

预设选择与消融命令见[实验指南](experiments.md)，数据与 checkpoint 约定见[架构](architecture.md)。统一实现不依赖旧的独立运行器，checkpoint 不与归档的独立实验互换。
