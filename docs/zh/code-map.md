# 代码文件索引

[English](../code-map.md) · 中文 · [文档首页](README.md)

下面列出仓库中 `src/mmrecsys/` 和 `tests/` 下的所有 Python 文件。不包含虚拟环境、数据、缓存或生成的运行产物；新增代码文件时请同步更新两种语言。

## 包与入口

| 文件 | 用途 |
| --- | --- |
| [`src/mmrecsys/__init__.py`](../../src/mmrecsys/__init__.py) | 声明顶层包。 |
| [`src/mmrecsys/cli.py`](../../src/mmrecsys/cli.py) | 解析训练和评估命令，并调用实验入口。 |
| [`src/mmrecsys/config.py`](../../src/mmrecsys/config.py) | 加载并合并 YAML，解析 CLI 覆盖项并校验最终配置。 |
| [`src/mmrecsys/registry.py`](../../src/mmrecsys/registry.py) | 登记模型工厂、配置类型、所需模态和批次规格。 |

## 数据

| 文件 | 用途 |
| --- | --- |
| [`src/mmrecsys/data/__init__.py`](../../src/mmrecsys/data/__init__.py) | 声明数据子包。 |
| [`src/mmrecsys/data/dataset.py`](../../src/mmrecsys/data/dataset.py) | 校验发布的交互与特征，建立训练和评估视图。 |
| [`src/mmrecsys/data/sampling.py`](../../src/mmrecsys/data/sampling.py) | 打乱每轮批次，在 CPU 或训练设备上仅依据训练数据采样负例。 |
| [`src/mmrecsys/data/views.py`](../../src/mmrecsys/data/views.py) | 定义训练数据、评估数据和可迁移设备的训练批次。 |

## 模型与神经网络组件

| 文件 | 用途 |
| --- | --- |
| [`src/mmrecsys/models/__init__.py`](../../src/mmrecsys/models/__init__.py) | 声明模型子包。 |
| [`src/mmrecsys/models/base.py`](../../src/mmrecsys/models/base.py) | 定义推荐模型、损失输出及打分接口，并实现内积打分器。 |
| [`src/mmrecsys/models/_shared.py`](../../src/mmrecsys/models/_shared.py) | 提供公共模型配置校验和带缓存的用户–物品图归一化。 |
| [`src/mmrecsys/models/bm3.py`](../../src/mmrecsys/models/bm3.py) | 实现自举式自监督，以及投影后可选的 DAMPS 校准。 |
| [`src/mmrecsys/models/freedom.py`](../../src/mmrecsys/models/freedom.py) | 实现固定物品图、度敏感的用户–物品边剪枝和可选的一次性 DAMPS 校准。 |
| [`src/mmrecsys/models/lgmrec.py`](../../src/mmrecsys/models/lgmrec.py) | 结合局部传播与全局超图，实现两种 DAMPS 接入方式。 |
| [`src/mmrecsys/models/lirdrec.py`](../../src/mmrecsys/models/lirdrec.py) | 实现 DCT 特征共享、PWC 融合和投影级 DAMPS。 |
| [`src/mmrecsys/models/mgcn.py`](../../src/mmrecsys/models/mgcn.py) | 实现行为引导净化、多视图图传播、对比学习和 DAMPS。 |
| [`src/mmrecsys/models/pgl.py`](../../src/mmrecsys/models/pgl.py) | 实现 PGL 局部及全局图模式，以及两种 DAMPS 接入方式。 |
| [`src/mmrecsys/models/smore.py`](../../src/mmrecsys/models/smore.py) | 实现频域融合、模态图及融合前可选的 DAMPS 校准。 |
| [`src/mmrecsys/nn/__init__.py`](../../src/mmrecsys/nn/__init__.py) | 声明神经网络组件子包。 |
| [`src/mmrecsys/nn/damps.py`](../../src/mmrecsys/nn/damps.py) | 实现 APC、AVRF 和 IMCF 频域校准、融合与组件消融。 |
| [`src/mmrecsys/nn/graph.py`](../../src/mmrecsys/nn/graph.py) | 构建稀疏交互图和分块精确 kNN 图，并进行归一化。 |
| [`src/mmrecsys/nn/losses.py`](../../src/mmrecsys/nn/losses.py) | 实现共用的 BPR 与 InfoNCE 损失。 |

## 训练与评估

| 文件 | 用途 |
| --- | --- |
| [`src/mmrecsys/engine/__init__.py`](../../src/mmrecsys/engine/__init__.py) | 声明训练引擎子包。 |
| [`src/mmrecsys/engine/checkpoint.py`](../../src/mmrecsys/engine/checkpoint.py) | 保存和恢复运行状态，并验证配置与数据兼容性。 |
| [`src/mmrecsys/engine/evaluator.py`](../../src/mmrecsys/engine/evaluator.py) | 分块执行全物品排序，屏蔽历史并按物品 ID 处理同分。 |
| [`src/mmrecsys/engine/metrics.py`](../../src/mmrecsys/engine/metrics.py) | 逐用户或批量计算 Recall 与 NDCG。 |
| [`src/mmrecsys/engine/trainer.py`](../../src/mmrecsys/engine/trainer.py) | 执行训练轮次、优化器更新、验证、早停和 checkpoint 保存。 |

## 实验流程

| 文件 | 用途 |
| --- | --- |
| [`src/mmrecsys/experiment/__init__.py`](../../src/mmrecsys/experiment/__init__.py) | 声明实验子包。 |
| [`src/mmrecsys/experiment/ablation.py`](../../src/mmrecsys/experiment/ablation.py) | 按种子配对运行基线、完整 DAMPS 和组件消融。 |
| [`src/mmrecsys/experiment/artifacts.py`](../../src/mmrecsys/experiment/artifacts.py) | 计算内容摘要，原子写入产物并缓存图。 |
| [`src/mmrecsys/experiment/compare.py`](../../src/mmrecsys/experiment/compare.py) | 调度独立的 MGCN 对照任务并生成配对汇总。 |
| [`src/mmrecsys/experiment/lirdrec_suite.py`](../../src/mmrecsys/experiment/lirdrec_suite.py) | 调度、验证、恢复和汇总 LIRDRec 单卡套件。 |
| [`src/mmrecsys/experiment/logging.py`](../../src/mmrecsys/experiment/logging.py) | 写入运行元数据、环境清单与 JSON Lines 指标。 |
| [`src/mmrecsys/experiment/runner.py`](../../src/mmrecsys/experiment/runner.py) | 选择设备并组装单次训练、恢复或评估流程。 |
| [`src/mmrecsys/experiment/search.py`](../../src/mmrecsys/experiment/search.py) | 使用验证指标执行可恢复的 MGCN 配对网格搜索。 |
| [`src/mmrecsys/experiment/seed.py`](../../src/mmrecsys/experiment/seed.py) | 初始化并保存或恢复 Python、NumPy、PyTorch 和所选设备的随机状态。 |
| [`src/mmrecsys/experiment/spectral_analysis.py`](../../src/mmrecsys/experiment/spectral_analysis.py) | 读取可信 MGCN checkpoint，对比频谱熵、能量集中度和相位方差。 |

## 测试

| 文件 | 用途 |
| --- | --- |
| [`tests/conftest.py`](../../tests/conftest.py) | 定义公共小数据及配置样例，并限制计算线程。 |
| [`tests/test_backbone_integrations_pgl_lgmrec.py`](../../tests/test_backbone_integrations_pgl_lgmrec.py) | 测试 PGL/LGMRec 图行为、两种 DAMPS 模式、梯度与 checkpoint。 |
| [`tests/test_checkpoint_copy.py`](../../tests/test_checkpoint_copy.py) | 检查原子复制、失败处理及序列化次数。 |
| [`tests/test_compare.py`](../../tests/test_compare.py) | 检查配对任务、设备选择、汇总和失败后恢复。 |
| [`tests/test_damps.py`](../../tests/test_damps.py) | 对照 DAMPS 参考公式，并测试梯度、初始化与状态。 |
| [`tests/test_data.py`](../../tests/test_data.py) | 检查划分隔离、非法数据与 CPU/设备采样器行为。 |
| [`tests/test_device.py`](../../tests/test_device.py) | 检查设备选择、随机状态隔离和有 CUDA 时的恢复。 |
| [`tests/test_evaluator.py`](../../tests/test_evaluator.py) | 检查历史屏蔽、同分、填充与批量指标。 |
| [`tests/test_experiment.py`](../../tests/test_experiment.py) | 检查配置优先级、可复现性及端到端训练、消融与恢复。 |
| [`tests/test_graph.py`](../../tests/test_graph.py) | 检查图归一化、精确 kNN 和缓存标识。 |
| [`tests/test_lirdrec.py`](../../tests/test_lirdrec.py) | 使用 NumPy 实现对照 LIRDRec，并测试 PWC 和 DAMPS 梯度。 |
| [`tests/test_lirdrec_experiment.py`](../../tests/test_lirdrec_experiment.py) | 在小数据上检查 LIRDRec 训练、评估和 epoch 边界精确恢复。 |
| [`tests/test_lirdrec_suite.py`](../../tests/test_lirdrec_suite.py) | 测试 LIRDRec 队列、GPU 占用、故障处理与恢复记录。 |
| [`tests/test_mgcn.py`](../../tests/test_mgcn.py) | 对照 MGCN 稠密公式和梯度，并检查 InfoNCE。 |
| [`tests/test_migrated_models.py`](../../tests/test_migrated_models.py) | 验证 BM3、SMORE 和 FREEDOM 的损失、图、DAMPS 路径和恢复。 |
| [`tests/test_search.py`](../../tests/test_search.py) | 检查网格展开、仅用验证集的排名及中断后恢复。 |
| [`tests/test_spectral_analysis.py`](../../tests/test_spectral_analysis.py) | 检查频谱熵、能量占比、相位方差和零能量输入拒绝。 |

## 配置

| 文件 | 用途 |
| --- | --- |
| [`pyproject.toml`](../../pyproject.toml) | 声明 Python 要求、依赖、CLI 入口和 pytest 设置。 |
| [`configs/default.yaml`](../../configs/default.yaml) | 设置公共数据、训练、评估与运行时默认值。 |
| [`configs/datasets/`](../../configs/datasets/) | 每个数据集一个 YAML，指定数据集名称与相对目录。 |
| [`configs/models/`](../../configs/models/) | 每个已注册骨干一个 YAML，定义模型和优化器默认值。 |
| [`configs/experiments/`](../../configs/experiments/) | 为 BM3、SMORE、FREEDOM、MGCN 和 LIRDRec 的各数据集组合提供基线与 DAMPS 预设。 |
| [`configs/search/damps_mgcn.yaml`](../../configs/search/damps_mgcn.yaml) | 定义默认的三组 MGCN 搜索配置。 |
| [`configs/search/damps_mgcn_extended.yaml`](../../configs/search/damps_mgcn_extended.yaml) | 定义可选的更宽 MGCN 搜索网格。 |
