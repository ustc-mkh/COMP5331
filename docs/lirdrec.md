# LIRDRec 接入与完整实验

本项目已接入 LIRDRec backbone，复用已有数据加载、训练器、评估、checkpoint 和 DAMPS 组件开关。Baby、Sports、Clothing、Electronics（配置名 `elec`）和 MicroLens 的五组对照/消融均已完成种子 999、2024、2025 的正式训练，共 75 组。

完整 DAMPS 在五数据集四项指标的 20 项跨种子均值上均低于本项目 LIRDRec 基线，尚未复现论文所报提升。这些结果对应本项目统一 DAMPS 接入，不是作者参考实现的精确复现。逐种子指标、均值与样本标准差、配对变化及论文对照见 [精选结果](results/lirdrec/)。

## 参考与实现约定

- 基线参考 [enoche/LIRDRec](https://github.com/enoche/LIRDRec)，模型代码位于 `src/models/lirdrec.py`，参数参考其 `src/configs/model/LIRDRec.yaml` 和 `src/configs/overall.yaml`。
- DAMPS 参考组长的 [修复版仓库](https://github.com/ustc-mkh/KDD2026_DAMPS)，复用本项目 `src/mmrecsys/nn/damps.py`，数值约定见 [DAMPS 文档](damps.md)。
- 保留基线的固定图文特征、4 倍隐藏宽度、正交 DCT-II 共享分支、二值 kNN 图、UI 层求和、PWC 融合和全体最终表示的均方正则。DCT 使用已有 SciPy 依赖计算。
- DAMPS 在图文两层 MLP 投影后各前向校准一次，图结构和共享 DCT 分支保持相同。参考 DAMPS 版 LIRDRec 在构造和前向阶段均调用模块；其模块忽略传入特征的数值，实际使用内部可训练原始特征及独立线性层，因此不能把两次调用理解成同一向量连续过滤两遍。参考版还改变隐藏宽度及构造图和 DCT 分支的输入。本实现保留原基线结构，是组内 DAMPS 算子的受控接入，不能视为该参考文件的精确复现。
- PWC 按作者递推规则在每轮开始更新权重，保存 epoch、融合权重和历史 attention，支持恢复。评估从当前参数重新计算表示，不更新 PWC 历史。作者原版读取上次训练前向缓存的表示，这一点已修正。
- kNN 同分按物品编号稳定排序；作者的 `torch.topk` 没有固定同分顺序。删除了作者基线中前向未使用的可训练特征副本。
- 复用本项目的全局物品负采样；作者从训练中出现的物品采样。Baby 的全局物品数为 7050，训练出现 7047 个，因此采样分布存在细微差异。不声称与作者训练轨迹逐步相同。

## 固定实验配置

`configs/models/lirdrec.yaml` 采用 64 维投影、2 层 UI 图、1 层物品图、k=10、图像图权重 0.1、正则 0.0001、PWC base/初始权重均为 0.9。Adam 学习率为 0.0001，沿用作者恒定学习率调度，权重衰减和图 dropout 均为 0。

其中部分参数原本属于作者的搜索候选。本次 75 组实验固定使用同一个组合，没有进行完整超参数搜索，也没有使用测试集挑选参数，因此不能将该组合称为各数据集的论文最优参数。

## 单卡流程检查

需要验证新环境时，可先运行两轮流程检查。先根据服务器规则选择一张卡，并将下面的占位符换成该卡的完整 GPU UUID。限制可见设备后，程序内使用 `cuda:0`。一次执行一条命令：

```bash
LIRDREC_GPU_UUID="<GPU UUID>"
CUDA_VISIBLE_DEVICES="$LIRDREC_GPU_UUID" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python -m mmrecsys.cli train \
  --config configs/experiments/lirdrec_baby.yaml --device cuda:0 \
  --set runtime.num_threads=2 --set train.epochs=2

CUDA_VISIBLE_DEVICES="$LIRDREC_GPU_UUID" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python -m mmrecsys.cli train \
  --config configs/experiments/damps_lirdrec_baby.yaml --device cuda:0 \
  --set runtime.num_threads=2 --set train.epochs=2
```

上述检查仍使用完整 Baby 数据、发布划分和全物品评估，只缩短训练轮数，不纳入 75 组正式结果。数据应位于 `data/baby/`；原始数据、依赖、运行目录、缓存与 checkpoint 不提交到 Git，精选结果单独保存在 `docs/results/lirdrec/`。各次运行保存配置、数据指纹、逐轮日志、最佳和最后 checkpoint 及测试结果。

三个消融开关为 `model.damps_apc`、`model.damps_avrf` 和 `model.damps_imcf`。`mmrecsys.experiment.ablation` 可读取 `damps_lirdrec_baby.yaml` 顺序运行五组；五数据集完整队列使用下文的 `lirdrec_suite`。

## 验证

`tests/test_lirdrec.py` 用独立的 NumPy 小图公式检查基线表示与损失，并检查 PWC 递推、DAMPS 及消融反向传播。`tests/test_lirdrec_experiment.py` 验证小数据训练、重新加载评估和连续/恢复训练的一致性。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python -m pytest -q
```

## 完整实验协议与队列

已完成的 75 组实验固定使用 `configs/models/lirdrec.yaml` 的参数，不使用测试集调参。每次最多训练 1000 轮，每轮验证，以验证集 Recall@20 选择最佳 checkpoint，连续 20 次未提升则停止，之后评估测试集。论文没有披露实际训练轮数或种子数量；1000/20 和默认种子 999 来自参考代码，三种子方案来自项目实验约定。

每个数据集、每个种子运行五组：`baseline`、`full`、`no_apc`、`no_avrf`、`no_imcf`。APC 消融跳过相位旋转；移除一个过滤分支后，另一个分支权重为 1。组间数据划分、骨干参数与训练预算相同。

套件默认串行执行 Baby 的种子 999 五组，然后 Sports、Clothing、Elec、MicroLens 各自的种子 999 五组，最后补齐全部数据集的 2024、2025。合计 75 组，符合配置、源码和数据身份的已完成实验可复用。队列只使用一张明确指定的 GPU，每个训练进程限制两个 CPU 计算线程；不自动占用其他显卡。

队列入口为 `mmrecsys.experiment.lirdrec_suite`，默认并发数为 1；如需同卡并行，可根据实测资源占用显式设置 `--max-parallel`（范围 1–10）。并行时允许同一种子的不同数据集重叠，仍等待种子 999 全部完成后才开始 2024，2024 全部完成后才开始 2025。并发数只改变调度，训练配置不变，并随队列保存以供恢复使用。

先检查空闲情况，将下面占位符换成获准使用的卡的完整 UUID；入口会检查其他计算进程是否占用该卡。

```bash
LIRDREC_GPU_UUID="<GPU UUID>"
python -m mmrecsys.experiment.lirdrec_suite \
  --output runs/lirdrec-five-datasets --gpu-uuid "$LIRDREC_GPU_UUID" \
  --datasets baby sports clothing elec microlens --seeds 999 2024 2025 --wait-for-gpu

# 恢复同一队列，保留实验顺序、配置和已完成结果
python -m mmrecsys.experiment.lirdrec_suite --resume runs/lirdrec-five-datasets

# 原卡长期被占用时，可在队列停止后指定另一张空闲卡恢复
python -m mmrecsys.experiment.lirdrec_suite --resume runs/lirdrec-five-datasets \
  --gpu-uuid "$LIRDREC_GPU_UUID" --wait-for-gpu

# 确认资源足够后，可停止队列并在同一卡上改为最多 2 个并行任务
python -m mmrecsys.experiment.lirdrec_suite --resume runs/lirdrec-five-datasets \
  --max-parallel 2 --wait-for-gpu
```

加 `--dry-run` 仅查看计划，不初始化 CUDA 或创建实验目录。每组使用独立子进程，结束后释放显卡；缺少数据或实验失败（包括显存不足）时保存状态并停止本队列的其他任务，不跳到后面的阶段，也不自动缩减 batch 或修改模型。停止 supervisor 会一并清理其训练子进程，保留已有 checkpoint。并行冷启动可能重复构图，正式启动前可在 CPU 上用每个数据集的一份原始 baseline 配置构建共享图缓存；这不进行训练，也不改变正式实验配置。

显卡占用检查只放行当前 supervisor 启动且仍存活的 worker PID。出现其他计算进程时暂停新增任务，并继续记录现有任务的完成状态；默认待现有任务结束后退出，启用 `--wait-for-gpu` 则每 30 秒检查指定卡，外部进程退出后自动继续。等待策略随队列保存。恢复会验证训练源码、配置和数据，未完成训练从原目录的 `last.pt` 继续。迁移显卡需取得队列及新卡的独占锁，拒绝与仍在运行的本队列或残留训练进程重叠，并记录迁移历史。`--reuse SLOT_ID=RUN_DIR` 配合 `--reuse-source snapshot.tar.gz` 可导入已有实验，但必须通过源码快照、完整训练预算、数据指纹和结果检查。

队列目录的 `suite.json` 记录状态及全部配置，`results.json` 保留逐次结果，`summary.csv` / `summary.json` 在每次完成后更新统计；部分完成时明确保留实际种子数。每个子目录保存日志和常规训练产物。训练源码或数据发生变化时，需要建立新队列，不把不同实现的结果混在一起。

结果按数据集和模型组汇总每个种子的四项测试指标及跨种子均值、样本标准差，完整 DAMPS 的提升使用相同种子的基线配对计算。保留所有成功或失败结果，不按测试成绩筛选种子。Table 2 提供原版和完整 DAMPS 的 LIRDRec 数值；Table 3 还提供 LIRDRec 在五个数据集上的 `-APC`、`-AVRF`、`-IMCF` 和完整 DAMPS 四项指标，可用于消融结果对照。Figure 3 展示的是向 MGCN 单独加入各组件，与 Table 3 的移除组件实验不同，不将该图的数值作为 LIRDRec 目标。与论文比较时仍须注明本实现的已知差异，单种子结果不代表三种子均值或统计显著性。
