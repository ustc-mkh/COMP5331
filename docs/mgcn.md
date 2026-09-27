# MGCN 实现与复现实验

模型公式依据：项目根目录的 *Multi-View Graph Convolutional Network for Multimedia Recommendation*（ACM MM 2023）。默认训练配置依据 Wang 等的 *Enhancing Multimodal Recommendation via Multimodal Representation Calibration in Spectral Domain*（KDD 2026）第 4.1 节，并核对本地 MMRec 的配置与实现。本项目独立实现，不导入 MMRec。

## Wang 等（2026）的实验设置

论文第 4.1 节（印刷页 1464–1465）说明：基线使用 MMRec，沿用各 backbone 的默认超参数；训练/验证/测试划分为 80%/10%/10%，使用全物品排序，报告 Recall 和 NDCG 的 @10、@20。当前配置对应表 2 的 MGCN **raw 基线**；`+Ours` 需要额外实现 DAMPS，不能仅靠训练配置得到。

具体数值来自本地 [MMRec 公共配置](../MMRec/src/configs/overall.yaml)、[MGCN 配置](../MMRec/src/configs/model/MGCN.yaml)和 [MGCN 实现](../MMRec/src/models/mgcn.py)，不是论文逐项列出的参数：

| 设置 | 本项目配置 | MMRec 来源 |
| --- | --- | --- |
| 随机种子 | `seed=999` | `seed: [999]` |
| 优化器、初始学习率 | Adam，`0.001`，weight decay `0` | 公共配置与 MGCN 配置 |
| 学习率衰减 | `0.96^(epoch/50)` | `learning_rate_scheduler: [0.96, 50]` |
| 训练轮数、batch | 最多 1000 轮，2048 | `epochs`、`train_batch_size` |
| 验证、早停 | 每轮验证，`patience=20`，监控 `Recall@20` | `eval_step=1`、`stopping_step=20`、`valid_metric` |
| embedding、图层数 | 64，UI 图 2 层，物品图 1 层 | `embedding_size`、`n_ui_layers`、`n_layers` |
| kNN、正则系数 | 10，`0.0001` | `knn_k`、`reg_weight` |
| InfoNCE 温度 | `0.2` | `calculate_loss` 的实际调用值；不是未使用的 `self.tau=0.5` |
| 对比损失权重 | 单次默认 `0.01`；候选 `0.001 / 0.01 / 0.1` | `cl_loss` 列表，由 MMRec 枚举 |
| 融合、正则、特征微调 | `author`、`batch_final`、`true` | MGCN 实现 |

Baby、Sports、Clothing、Elec 共用这些默认值。论文没有报告各数据集最终选中的 `cl_loss`，因此不能将 `0.01` 当作已验证的最优值。配置加载器只接受单个 `model.cl_weight`；下面提供枚举候选值的命令。

沿用发布数据中的 `x_label`，不重新随机划分。评估 K 采用论文的 `[10, 20]`；MMRec 公共配置还计算 @5、@50 及 Precision、MAP。评估用户分块保留 256（MMRec 为 4096），物品分块为 4096，以控制内存，不改变全物品排序协议。论文报告的 Python 3.8.20 / PyTorch 2.4.1 是其运行环境，本项目仍遵循 `pyproject.toml` 的依赖要求。

## 公式对应

| 论文 | 本项目实现 |
| --- | --- |
| (1)–(2) Behavior-Guided Purifier | 每个模态线性投影，再通过 sigmoid 门控与物品 ID embedding 逐元素相乘 |
| (3)–(5) User-Item View | 仅训练边构建无自环二部图；`D^-1/2 A D^-1/2`；平均第 0 至 L 层 ID 表示 |
| (6)–(9) Item-Item View | 原始特征余弦 kNN，保留余弦边权，归一化后传播净化表示；默认一层 |
| (10) 用户模态表示 | 使用归一化二部图的用户—物品块聚合物品模态表示 |
| (11)–(14) Behavior-Aware Fuser | 共享的 attention 网络提取公共表示；各模态残差乘以行为表示生成的偏好门 |
| (15) 模态融合 | `author` 使用 MMRec 的 `(common + residual_sum) / 3`（双模态） |
| (16) 对比目标 | 用户和正物品各计算一项方向性 in-batch InfoNCE，温度 0.2 |
| (17)–(18) 打分 | 行为与多模态表示相加，用户/物品内积 |
| (19) 总损失 | BPR + `cl_weight * InfoNCE` + `reg_weight * regularizer` |

InfoNCE 使用归一化向量、batch 内对侧向量作为候选，以及 cross-entropy/log-sum-exp 的稳定实现。这与作者的 batch 近似一致；不会照搬论文式 (16) 中分母索引的排版问题。重复用户/物品在 batch 中保留独立位置，正对为对角线；这是作者代码的约定。

## MMRec 实现约定

实验配置位于 `configs/experiments/mgcn_{baby,sports,clothing,elec}.yaml`，分别对应 Baby、Sports、Clothing、Electronics；也可使用 `--dataset` 切换数据集。各数据集共享 `configs/models/mgcn.yaml` 中的 MGCN 设置。

融合使用 `(common + residual_sum) / 3`（双模态），正则化使用 batch 最终用户、正负物品表示的平方和除以 `2B`。模态特征按 MMRec 的 `freeze=False` 开启微调，可训练特征表保存到 checkpoint；kNN 图仍由最初发布特征构建，训练中不更新。

配置对齐不等于逐位复刻 MMRec：其正则除数是配置 batch size，本实现使用当前实际 B；其早停在无提升次数 `>20` 时触发，本实现在 `>=20` 时触发；初始化顺序、采样和同分排序等也可能影响训练结果。本地 MMRec 的 `quick_start.py` 按测试指标汇总候选组合，本项目始终按验证集选择超参数，测试标签不参与选模。尚未验证达到论文表格指标。

## 图构建与缓存

默认 k=10、余弦加权、有向 kNN、不额外对称化，自身允许参与 top-k（计入 k）。对称归一化指 `D^-1/2 S D^-1/2`，并不意味着将有向 kNN 转为无向图。所有选项均保存在最终配置中。

相似度按行分块计算，逐块选取 k 个邻居，不生成完整 N×N 矩阵。相似度相同时优先较小物品 ID；零向量使用零余弦。为保持式 (7)，不会默默截断负余弦；如果选中边的加权度为负，则明确报错，因为式 (8) 的实数逆平方根无定义。可按实验目的显式选择二值边权，但这会改变方法。

缓存保存 CPU COO indices/values/shape，并以临时文件加原子重命名发布。键包含图算法版本、原始特征内容及排列摘要、节点编号规则、k、相似度、边权、自环、对称化、归一化和同分规则。训练图包含训练划分摘要。图和固定特征不写入 checkpoint；重新运行时根据数据指纹重建或命中缓存。

构图为精确 kNN，计算量仍为 O(N²d)，内存为 O(chunk×N + N×k)。Elec 等大数据集首次建图可能较慢；调小 `model.knn_chunk_size` 可降低构图峰值内存。当前在 CPU 构图，再统一移动模型需要的 Tensor 到训练设备。

## 数据、训练和评估

`TrainData` 仅含全局规模、训练交互、训练 CSR、特征和指纹；验证/测试标签由 `EvalData` 持有。读取时检查列名、整数 ID、全局编号连续性、划分标签、重复交互、特征行数及非有限值。所有发布训练交互作为正反馈。

负采样只排除训练正例，在补集中均匀采样；不读取验证/测试标签。当前采用单进程批次生成器，不提供 worker 配置。每个 epoch 的独立种子由实验 seed 和 epoch 派生，稠密用户也不会进入无限拒绝采样循环。

评估按用户和物品双重分块进行全物品排序。先屏蔽历史，再合并各块 top-k；同分按物品 ID 升序；不足 K 的有效候选不会用被屏蔽物品补齐。Recall/NDCG 按有目标且符合条件的用户取宏平均，并记录用户数。NDCG 的理想排序长度为 `min(K, 目标数)`。

默认验证/测试均只屏蔽训练历史，且要求评估用户有训练交互，与架构文档协议一致。`eval.history=train_valid` 可让测试额外屏蔽验证历史，验证阶段仍只屏蔽训练历史。测试标签不参与选模。

第 1 轮进行基线验证，此后按 `train.eval_every` 验证；早停 patience 按验证次数计算，同分不更新最佳模型。结束后加载验证最佳模型执行测试。损失日志中的 `contrastive`、`regularization` 已乘相应权重，总和等于 `total`。

## 命令

在项目根目录安装：

```bash
python -m pip install -e '.[test]'
python -m pytest -q
python -m mmrecsys.cli train --model mgcn --dataset baby --seed 999
python -m mmrecsys.cli train --config configs/experiments/mgcn_baby.yaml
```

指定训练 GPU（编号为当前进程可见的 CUDA 设备编号）：

```bash
python -m mmrecsys.cli train --dataset baby --device cuda:2
```

程序在初始化前设置当前 CUDA 设备，随机种子与 checkpoint 中的 CUDA 随机状态仅处理训练所在的 GPU；CPU 训练不读取 CUDA 随机状态。旧版保存的多 GPU 随机状态仍可读取，恢复时只使用目标 GPU 的状态。修复前已启动的进程需要重启，已有 CUDA 上下文不会因源码更新而释放。

如需限制进程只看到指定 GPU，可以使用：

```bash
CUDA_VISIBLE_DEVICES=2 python -m mmrecsys.cli train --dataset baby --device cuda:0
```

此时可见 GPU 会重新编号，进程内的 `cuda:0` 对应被选中的 GPU。恢复旧训练时应保持原有可见设备映射和 `--device`，例如原运行使用 `cuda:2`，恢复也使用 `--device cuda:2`。

完整数据短跑与其他数据集：

```bash
python -m mmrecsys.cli train --dataset baby --device cpu --set train.epochs=2
python -m mmrecsys.cli train --dataset sports --device cuda:0
python -m mmrecsys.cli train --dataset clothing --device cuda:0
python -m mmrecsys.cli train --dataset elec --device cuda:0 --set model.knn_chunk_size=128
```

按 MMRec 提供的候选值运行四个数据集（共 12 次完整训练）：

```bash
for dataset in baby sports clothing elec; do
  for cl_weight in 0.001 0.01 0.1; do
    python -m mmrecsys.cli train --dataset "$dataset" --seed 999 \
      --set "model.cl_weight=$cl_weight"
  done
done
```

每个数据集比较各 run 的 `result.json` 中的 `best_validation_metric`（Recall@20），选择最大值对应的配置，再报告该 run 的测试指标。每次训练只运行一个候选值；上述循环不是已完成的调参结果。

恢复与独立评估（将 `<run_id>` 替换为训练输出的目录名）：

```bash
python -m mmrecsys.cli train --resume runs/<run_id>/last.pt --set train.epochs=1000
python -m mmrecsys.cli evaluate --run runs/<run_id> --split test
python -m mmrecsys.cli evaluate --run runs/<run_id> --split valid --device cpu
```

配置优先级为公共默认 < 数据集 < 模型 < 实验 < CLI。`--set` 接受 YAML 值，未知字段报错。所有配置内相对路径按项目根解析；安装为 editable 后，从其他工作目录调用也成立。当前配置文件位于仓库中，推荐以 editable 方式使用。

每次训练生成 `config.yaml`、`manifest.json`、`metrics.jsonl`、`best.pt`、`last.pt`、`result.json`。manifest 包括数据摘要、Git 版本/脏状态、依赖版本、设备和评估规则。

恢复只接受原 run 的 `last.pt`，并要求保留该 run 的 `best.pt`。保存模型、优化器、调度器、epoch、global step、早停状态和 Python/NumPy/Torch/训练 GPU CUDA/采样器随机状态。允许延长 epoch 上限和变更输出/缓存位置；其他训练配置及数据指纹必须匹配。不承诺批次中途恢复或跨设备/依赖版本逐位相同；已早停的 run 恢复时仍保留其早停状态。

## 验证范围

测试覆盖数据隔离、错误数据拒绝、负采样、手算图归一化、分块 kNN、缓存失效、稠密公式对照、梯度、稳定 InfoNCE、屏蔽/同分/不足 K 的排序和指标、配置优先级、随机状态以及 epoch 边界恢复的一致性。短跑用于验证工程流程；复现论文表格还需完整训练和仅使用验证集的超参数选择。
