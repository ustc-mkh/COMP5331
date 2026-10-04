# DAMPS 实现与复现约定

依据项目根目录 [DAMPS.pdf](../DAMPS.pdf)：*Enhancing Multimodal Recommendation via Multimodal Representation Calibration in Spectral Domain*（KDD 2026）。本项目已将统一 DAMPS 模块接入 MGCN 与 LIRDRec，AVRF、FFT、IMCF 及融合初始化依据作者源码 `KDD2026_DAMPS-v1.0.0/Wmhwxl-KDD2026_DAMPS-af90958/src/models/damps.py` 实现；相位旋转方向已恢复为与论文及作者源码一致的 image−/text+。不依赖 MMRec 运行，也不声称全部行为或论文指标已对齐。

## 接入位置与模块

实现位于 `src/mmrecsys/nn/damps.py`，接口为 `image_out, text_out = damps(image, text)`；输入与输出均为两个 `[N, d]` 实数张量。MGCN 接入时传入原始特征，使 DAMPS 创建自己的 `image_embedding/text_embedding` 和 `image_trs/text_trs`；前向使用内部投影，外部投影的数值不参与频域运算。独立的纯频域接口仍支持直接传入投影。N 为全体物品数量，不是交互 batch 大小。

MGCN 在门控净化之前调用 DAMPS，频域统计初始化和每次前向均使用 DAMPS 内部的特征 embedding 与投影层。外部 MGCN 特征层仍注册在模型中，但在默认 `batch_final` 正则下不接收推荐损失梯度。与作者一样，内部 embedding 是独立 Parameter，构造时与外部 embedding 共享原始特征存储；投影层参数独立。`trainable_features=false` 时两套特征均冻结，投影仍可训练。校准后的两个表示继续进入原有门控、物品图传播、用户侧聚合和多视图融合。ID 协同分支、由原始特征构建的固定 kNN 图及损失结构沿用 backbone。`regularization=parameters` 时新增可训练参数也进入既有参数正则项；默认 `batch_final` 仍只正则化最终批次表示。

LIRDRec 使用纯频域接口，在原 backbone 的图文投影后接入 DAMPS，保持固定原始特征、4 倍隐藏宽度、原始特征构图与 DCT 共享分支。这是本项目统一模块的受控接入，与参考 DAMPS 版 LIRDRec 的内部特征层、投影和接入位置存在差异，详见 [LIRDRec 说明](lirdrec.md)。

## 公式对应

| 论文 | 实现 |
| --- | --- |
| 式（1） | MGCN 使用 DAMPS 内部独立 embedding 和投影；LIRDRec 复用 backbone 投影 |
| 式（2）—（4） | 沿最后一维 `rfft`，F = floor(d/2)+1；振幅 `abs`，相位 `angle` |
| 式（5）—（8），APC | text − image 相位差，初始化时跨物品求 sin/cos 均值并使用 atan2，保存为固定先验；image 乘 exp(−j(θ/2+ψ))，text 乘 exp(+j(θ/2+ψ)) |
| 式（9）—（18），AVRF | 按频率跨物品计算 median、MAD、总体方差；noise=(1.4826 MAD)²，signal=max(var−noise,0)，以 signal/(signal+noise+ε) 为初始统计，再标准化、sigmoid、logit，作为可学习权重 |
| 式（19）—（22），IMCF | 逐元素交叉功率 image × conj(text)，用其模平方除以两模态功率之积加 1e−8，再分别过滤两个模态 |
| 式（23） | 共享的两个可学习 logits 经 softmax，融合 AVRF 与 IMCF 两个并行分支 |
| 式（24） | `irfft(..., n=d)` 还原实数表示 |

两个过滤分支均作用于相位校准后的频谱，不能将它们串联。模型构造时在 no_grad 下用初始全物品投影调用 initialize，一次性估计 AVRF 初值和相位先验；在优化器创建前完成。每次前向仅使用当前投影、可学习 AVRF 权重、固定相位先验和可学习残差，不重新估计统计量，也不缓存带梯度的全图表示。独立使用 DAMPS 时必须先 initialize，重复初始化会报错。

APC 使用共享频率残差 ψ，形状 `[F]`；融合 logits 形状 `[2]`。ψ 初始化为零，融合 logits 初始化为 [0.6, 0.4]，softmax 后权重约为 [0.549834, 0.450166]。另有 image/text 两组 `[F]` 可学习 AVRF 权重，频域部分增加 3F+2 个参数（不含内部 embedding 和投影层），d=64 时为 101 个。相位先验以持久 buffer 保存，不参与优化。

## 按文章保留的歧义与数值约定

1. **相位符号。** 式（5）定义 R=text−image，式（7）—（8）的 image−/text+ 旋转会使相位差变成 R+θ+2ψ，而非减去 θ。当前保留作者符号，不自动修正；曾尝试的反向旋转变体已按用户要求撤回。测试覆盖正负偏移及接近 ±π 的圆周角行为。
2. **Coherence 的实际含义。** 作者源码没有统计平均，权重在精确算术下为 Q/(Q+1e−8)，其中 Q 是图文功率乘积。大功率位置接近直通，极小功率位置受到抑制；这不等同于经过统计平均的相干估计。实现不添加跨物品平均或频率平滑。
3. **零值处理。** AVRF 初始化方差比使用 ε=1e−6（model.damps_eps，可配置），跨频率标准化分母加 1e−6，logit 分母加 1e−8。IMCF 分母统一加 1e−8，不额外截断，零功率时权重自然为零。圆均值的 sin/cos 和同时为零时采用零偏移。`angle(0)` 沿用 PyTorch 的零相位约定。
4. **FFT 与端点。** 初始化统计及前向正逆 FFT 均按作者源码使用 norm="ortho"。论文没有单独描述实数频谱端点；实现对所有频点执行作者的旋转，再由 irfft 忽略 DC 及偶数 d 的 Nyquist 频点虚部，没有额外禁止端点旋转。奇数 d 显式传入 n，防止输出长度变化。
5. **统计约定。** AVRF 振幅转 double 后计算总体方差（correction=0）和 median，偶数样本取中间较小项。方差比转 float32 后跨频率标准化，标准差使用 correction=1；仅一个频点时约定标准差为零以避免 NaN。sin/cos 具有周期性，无需先显式 wrap 相位差。支持 float32/float64；没有引入混合精度训练。

AVRF 前向直接乘可学习参数，不再 sigmoid 或截断，因此权重可以为负或大于 1，与作者源码一致。相位先验固定、残差可学习，旋转符号恢复为 image−/text+。

MGCN 已使用作者的两套特征层结构，DAMPS 内部特征层的初始化顺序与作者一致；完整 MGCN 的层初始化顺序及训练采样仍与作者不完全相同。因此数值算子对齐不代表训练轨迹或论文指标完全对齐。模块保存 implementation_version=3 标记；加载时显式检查版本，原作者方向的版本 3 checkpoint 可按原配置恢复，旋转修正实验的版本 4 checkpoint 会被拒绝。已有实验结果保留用于对照。

## 配置和命令

```bash
python -m mmrecsys.cli train --config configs/experiments/damps_mgcn_baby.yaml
python -m mmrecsys.cli train --config configs/experiments/damps_mgcn_sports.yaml
python -m mmrecsys.cli train --config configs/experiments/damps_mgcn_clothing.yaml
python -m mmrecsys.cli train --config configs/experiments/damps_mgcn_elec.yaml
```

上述配置的 `model.name=mgcn`；LIRDRec 配置使用 `model.name=lirdrec`。两者的主开关均为 `model.damps_enabled`（默认 false），DAMPS 实验配置将其设为 true。`model.damps_apc`、`model.damps_avrf`、`model.damps_imcf` 默认 true，可用 CLI 覆盖：

```bash
python -m mmrecsys.cli train --config configs/experiments/damps_mgcn_baby.yaml --set model.damps_apc=false
```

消融时，去掉 APC 即跳过相位旋转；去掉任一过滤分支后，剩余分支权重为 1；两个过滤分支都去掉时，直接还原相位校准频谱；三个组件都去掉时返回内部投影（纯频域接口则原样返回输入），因此 MGCN 中该消融不再等价于关闭 DAMPS。未使用的 ψ、AVRF 权重和 logits 不注册为参数。论文没有详细指定消融权重处理，这里是显式工程约定。

对照实验应在相同数据、种子、backbone 参数与评估协议下切换 `damps_enabled`。遵循论文第 4.1 节，沿用 backbone 参数，不针对 DAMPS 单独优化 backbone。MGCN 的 cl_weight=0.01 是启动值，不代表已核验的逐数据集最优值；LIRDRec 的固定参数见其模型配置和说明。数据沿用发布划分，使用全物品排序及 Recall/NDCG@10、@20；验证选模，测试只用于最终指标。

## 验证与范围

测试覆盖独立 NumPy 公式参考、相位符号与小功率 coherence、独立 AVRF 初始化统计参考、权重更新与先验固定、状态保存恢复、奇偶 embedding 维度、八种组件组合、零/常量/单物品输入、参数与输入梯度、关闭组件后的内部投影直通、作者源码前向和梯度对照、特征层更新及冻结、MGCN 损失与 scorer，以及启用 DAMPS 的训练—评估—epoch 边界恢复。

运行 `python -m pytest -q`（若使用项目虚拟环境则 `.venv/bin/python -m pytest -q`）。这些测试验证公式和工程行为，不代表论文指标已复现。当前提供四个 Amazon 数据集的 MGCN 实验配置，以及五个数据集（含 MicroLens）的 LIRDRec 配置。LIRDRec 使用纯频域接口并保留原基线结构，接入差异与完整实验约定见 [LIRDRec 说明](lirdrec.md)。

LIRDRec 的五数据集 × 五组 × 三种子（999、2024、2025），共 75 组正式实验已完成。完整 DAMPS 的 20 项跨种子指标均值均低于本项目基线，尚未复现论文所报提升；数值与论文对照见 [实验结果](results/lirdrec/)。

FFT 计算量约 O(N d log d)；全物品 median 和方差仅在初始化时计算，不再每个训练 batch 重新统计。没有新增 N×N 稠密矩阵。优先保持公式与梯度语义，后续应先测量真实开销，再考虑等价优化。


## 训练诊断与配对消融

`metrics.jsonl` 每轮的 `diagnostics` 记录该轮**第一个 batch、backward 后且 optimizer.step 前**的快照，不是全轮平均。门控指标使用本次全物品表示；梯度来自该训练 batch。评估不覆盖该快照。指标不参与损失，记录过程不保留计算图、不消耗随机数。

- `gate/image|text/saturated_fraction`：sigmoid 输出 <0.01 或 >0.99 的比例。
- `gate/image|text/exact_zero_one_fraction`：输出恰好为 0 或 1 的比例。
- `gate/image|text/mean_derivative`：平均 sigmoid 导数 g(1−g)。
- `gradient/projection/...`、`gradient/gate/...`、`gradient/damps/...`：参数梯度 L2 范数及零元素比例。
- `damps/avrf_mix_weight`、`damps/imcf_mix_weight`：当前两个分支的 softmax 权重（仅双分支启用时）。

没有添加门控归一化、幅度限制或梯度裁剪。用日志判断饱和与梯度随训练的变化，而不是自动改变方法。

配对运行入口：

```bash
# 单个种子的五组对照：baseline/full/no_apc/no_avrf/no_imcf
python -m mmrecsys.experiment.ablation --config configs/experiments/damps_mgcn_baby.yaml --device cuda:3 --seeds 999
# 三个种子；每个种子运行全部五组，共 15 次完整训练
python -m mmrecsys.experiment.ablation --config configs/experiments/damps_mgcn_baby.yaml --device cuda:3 --seeds 999 2024 2025
# 只比较基线和完整模型，参数覆盖对所有组共同生效
python -m mmrecsys.experiment.ablation --seeds 999 2024 2025 --variants baseline full --set model.cl_weight=0.02
```

各组共享 backbone 配置；同种子下 backbone 初始参数相同，DAMPS 独立投影层初始化会消耗额外随机数，但创建于 backbone 初始化之后，采样由 seed/epoch 独立确定。组件消融仅改变对应开关。程序验证集选取每次运行的 checkpoint，不按测试指标筛选种子或组合。

该入口的结果保存在 `runs/ablation-<dataset>-<time>-<id>/<variant>/`，每次运行使用对应 backbone 的目录前缀。`runs.json` 逐次记录已完成实验；`summary.json` 汇总各组跨种子的指标均值与样本标准差，单种子标准差为 null。此 `ablation` 入口不自动续跑；中断时可按 runs.json 找到已完成或对应子目录中的未完成实验，并使用常规 --resume 入口恢复单次运行。LIRDRec 的 `lirdrec_suite` 另行支持队列恢复，详见 [完整实验协议与队列](lirdrec.md#完整实验协议与队列)。


## 当前推荐的参数范围

重新核对论文第 4.1 节与 MMRec 后，默认搜索固定 `n_ui_layers=2`、`n_item_layers=1`、`knn_k=10`，仅比较 `cl_weight=[0.001, 0.01, 0.1]`。单种子 baseline/full 共 6 次训练，详见 [超参数搜索说明](search.md)。不将 DAMPS 发布包的 192 组宽网格视为复现论文所必需的设置；论文没有公布逐数据集最终 CL 权重。
