# BM3、SMORE 与 FREEDOM

三个模型均原生接入 `src/mmrecsys`，复用模型注册表、发布数据、配置校验、采样器、Adam、训练器、全物品评估及 checkpoint。模型不依赖旧的独立运行目录或 MMRec Python 包。旧 BM3/SMORE 实验报告、归档和独立运行代码已移除；统一实现需要重新训练，不继承旧成绩或 checkpoint。

## 运行入口

在仓库根目录安装项目后运行：

```bash
python -m pip install -e '.[test]'
python -m mmrecsys.cli train --config configs/experiments/bm3_baby.yaml
python -m mmrecsys.cli train --config configs/experiments/smore_baby.yaml
python -m mmrecsys.cli train --config configs/experiments/freedom_baby.yaml
```

每个模型都提供 baby、sports、clothing、elec、microlens 五数据集配置，以及对应的 `damps_<model>_<dataset>.yaml`。这些是固定配置，不表示逐数据集搜索最优参数。BM3 沿用原独立代码的逐数据集模型参数；SMORE 沿用原 Task 4 固定参数；FREEDOM 使用模型 YAML 中声明的公共参数，未迁移原入口的外部服务器配置或超参数搜索逻辑。

```bash
python -m mmrecsys.cli train --config configs/experiments/damps_smore_sports.yaml
python -m mmrecsys.cli train --model freedom --dataset baby --set model.damps_enabled=true
python -m mmrecsys.experiment.ablation --config configs/experiments/smore_baby.yaml --seeds 999 2024 2025
python -m mmrecsys.cli train --resume runs/<run_id>/last.pt --set train.epochs=1000
python -m mmrecsys.cli evaluate --run runs/<run_id>
```

`--model/--dataset` 使用模型公共默认值；需要逐数据集固定参数时使用相应实验 YAML。三个模型的 `damps_enabled` 默认均为 false，三个 DAMPS 组件通过 `damps_apc`、`damps_avrf`、`damps_imcf` 控制。

## 模型与 DAMPS 接入

| 模型 | 训练批次 | 主要损失 | DAMPS 行为 |
| --- | --- | --- | --- |
| BM3 | 正样本，无负采样 | 停止梯度的 Dropout 目标、自监督模态对齐、全表示范数正则 | 可选扩展：投影后校准，随模态损失训练 |
| SMORE | 一个负样本 | BPR、最终表示平方正则、侧信息/行为 InfoNCE | 投影后校准，再执行 SMORE 原有频域融合；随损失训练 |
| FREEDOM | 一个负样本 | 主推荐 BPR、加权图文辅助 BPR | 初始化校准一次，输出复制为独立可训练特征表；校准模块冻结 |

BM3 保留零层参与平均的 UI 图传播、物品 ID 残差及 predictor 打分。正则为 `(||U||₂ + ||I||₂) / n_items`，不是平方和，也不是批内 BPR 正则。其新增 DAMPS 开关是统一框架中的扩展，默认基线不启用。

SMORE 保留图像/文本各自的 cosine 加权 kNN 图、取边权最大值的融合图、三路复数频谱权重、按特征维度的注意力、偏好 Dropout 及侧信息融合。DAMPS 直接使用共享模块的投影输入接口，初始统计在全部物品上估计一次。

FREEDOM 保留二值 kNN、图文图加权、度敏感的 UI 边剪枝，以及评估时使用完整 UI 图的行为。`on_epoch_start()` 每轮重新剪枝；固定图和采样权重作为 buffer 随模型迁移设备。DAMPS 版本使用共享算子固定的 D 维输出，替代旧 Task 6 按原始特征宽度执行逆 FFT 的行为；随后投影输入维度相应调整。校准模块不参与推荐损失更新。校准后的图缓存以实际初始特征内容为指纹，该物品图也进入 checkpoint，保证重载时恢复实际构图状态。

## 统一协议

全部模型采用 `mmrecsys` 的发布划分、训练历史过滤、分块精确 kNN、物品 ID 同分排序、未提前舍入的用户宏平均指标、验证集选模和最佳权重重载。随机顺序由公共采样器决定；早停为连续 `train.patience` 次验证未提升。SMORE 正则分母使用当前批次的真实大小。旧运行器的随机轨迹、四位小数舍入与早停边界不再作为兼容目标。

模型只读取 `TrainData`；验证和测试标签由公共评估器管理。超参数选择使用验证集；测试集在加载最佳 checkpoint 后评估。新增模型没有单独的训练器、优化器循环或评估器。

## 来源与许可

- BM3：原 `baselines/bm3` 固定的 enoche/BM3 commit `ef82a333823ae01ac5964039720f697532b40ea3`，原作者 enoche（enoche.chow@gmail.com），原始代码许可为 GPL-3.0。
- SMORE：原 `task4/runtime/src/models/smore.py`，参考 kennethorq/SMORE（WSDM 2025，源文件署名 rongqing001@e.ntu.edu.sg），保留该实现的频域融合及投影级 DAMPS 接入位置。
- FREEDOM：原 `task6/models/freedom.py`，源文件参考 FREEDOM、署名 enoche.chow@gmail.com。移植模型计算，移除外部绝对路径、私有 SciPy API、训练器及重复工具代码。

验证见 `tests/test_migrated_models.py`，覆盖模型批次与梯度、BM3 原损失及梯度方程、SMORE 频域卷积与融合图、FREEDOM 剪枝和初始化校准状态、公共组件消融、最佳权重重载以及精确 epoch 边界恢复。测试通过不代表已经取得正式数据集实验指标。
