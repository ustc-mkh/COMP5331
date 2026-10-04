# COMP5331 · DAMPS 复现项目

基于 [DAMPS.pdf](DAMPS.pdf)（KDD 2026）复现频域多模态表示校准框架 DAMPS，当前接入 MGCN 与 LIRDRec backbone，支持 APC、AVRF、IMCF 及组件消融。LIRDRec 提供 Baby、Sports、Clothing、Electronics 和 MicroLens 的配置。直接读取 MMRec 发布的数据，不依赖 MMRec 运行。

LIRDRec 已完成五个数据集 × 五组对照/消融 × 三个种子（999、2024、2025），共 75 组正式实验。完整 DAMPS 在五数据集四项指标的 20 项跨种子均值上均低于本项目基线，尚未复现论文所报提升；精选结果与论文对照见 [LIRDRec 实验结果](docs/results/lirdrec/)。基线参数、统一 DAMPS 接入方式、作者代码差异及运行命令见 [LIRDRec 接入说明](docs/lirdrec.md)。

当前已恢复作者源码的 image−/text+ 相位旋转方向。可训练 AVRF、固定相位先验、正交 FFT、带 epsilon 的 IMCF 及融合初始化沿用已核对的源码行为。提供门控/梯度诊断与配对消融入口；初始化顺序及训练协议仍有差异，尚未复现论文指标。公式对应、数值约定与局限见 [DAMPS 实现说明](docs/damps.md)，工程结构见 [架构设计](docs/architecture.md)，基线细节见 [MGCN backbone](docs/mgcn.md)。

## 安装与训练

需要 Python 3.10+、PyTorch、NumPy、SciPy、PyYAML；GPU 训练需要匹配的 PyTorch/CUDA 环境。

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

默认自动选择 CUDA 或 CPU，可使用 `--device cpu` 或 `--device cuda:0` 指定设备。

LIRDRec 完整套件要求显式传入一张卡的 UUID，默认串行运行，详见 [完整实验入口](docs/lirdrec.md#完整实验协议与队列)。

新 GPU 实验默认将训练交互及采样索引常驻显存，每轮在 GPU 上生成负样本；可用 `--set train.preload_to_device=false` 关闭。旧实验恢复保留原采样方式。性能与计时说明见 [性能说明](docs/performance.md)。

MGCN 常规训练固定使用 `n_ui_layers=4`、`n_item_layers=2`（原版 `n_layers`）、`cl_weight=0.01`（原版 `cl_loss`）、`knn_k=10`、`seed=999`，DAMPS 与基线共享这些设置。LIRDRec 使用独立的 `configs/models/lirdrec.yaml`。超参数搜索有独立的覆盖配置。

训练命令：

```bash
python -m mmrecsys.cli train --config configs/experiments/damps_mgcn_baby.yaml
```

其他数据集的完整训练配置（Elec 对应 Electronics）：

```bash
python -m mmrecsys.cli train --config configs/experiments/damps_mgcn_sports.yaml
python -m mmrecsys.cli train --config configs/experiments/damps_mgcn_clothing.yaml
python -m mmrecsys.cli train --config configs/experiments/damps_mgcn_elec.yaml
```

基线对照与消融：

```bash
# MGCN 基线
python -m mmrecsys.cli train --config configs/experiments/mgcn_baby.yaml
# 去掉 APC；另两个开关为 model.damps_avrf / model.damps_imcf
python -m mmrecsys.cli train --config configs/experiments/damps_mgcn_baby.yaml --set model.damps_apc=false
```

配置中的 `model.name=mgcn` 指 backbone；DAMPS 由 `model.damps_enabled=true` 启用。旧 `mgcn_*.yaml` 保留为基线配置。启用 DAMPS 时，新运行目录以 `damps-mgcn-` 开头；基线目录以 `mgcn-` 开头。具体组件开关保存在 `config.yaml`，断点恢复沿用原目录。

## 配对对照与诊断

```bash
python -m mmrecsys.experiment.ablation --config configs/experiments/damps_mgcn_baby.yaml --device cuda:3 --seeds 999
```

依次运行基线、完整 DAMPS 和三组组件消融；同种子共享 backbone 初始化及采样。用 `--seeds 999 2024 2025` 运行多种子实验；结果目录包含 `runs.json` 和跨种子 `summary.json`。每轮 `metrics.jsonl` 的 `diagnostics` 记录首批次门控饱和率、梯度和融合权重，口径详见 [DAMPS 文档](docs/damps.md)。

本版更改了频域计算，旧 DAMPS checkpoint 不直接兼容，请新建训练运行。

## 四数据集多 GPU 对比

新增入口默认比较 Baby、Sports、Clothing、Elec 的 MGCN 与完整 MGCN+DAMPS，每个种子共 8 次训练。采用独立实验并行：每张 GPU 同时运行一个训练进程，空闲后领取下一个任务，单次训练仍使用一张 GPU。

```bash
# 查看全部配置，不读取数据、不占用 GPU、不创建输出目录
python -m mmrecsys.experiment.compare --dry-run

# 快速检查：默认 seed=999，最多 20 轮，验证指标连续 5 次未提升则早停
python -m mmrecsys.experiment.compare --gpus 1 2 3 4

# 更短的流程检查：仍使用完整数据和模型
python -m mmrecsys.experiment.compare --gpus 1 2 --set train.epochs=2

# 完整训练：最多 1000 轮、patience=20，三个配对种子（共 24 次训练）
python -m mmrecsys.experiment.compare --mode full --gpus 1 2 3 4 --seeds 999 2024 2025

# 选择部分数据集，或使用 CPU
python -m mmrecsys.experiment.compare --datasets baby sports --gpus cpu

# 中断后跳过已完成的任务；失败或未完成的任务从头重跑
python -m mmrecsys.experiment.compare --resume runs/compare-<时间>-<ID> --gpus 1 2 3 4
```

不传 `--gpus` 时使用所有可见 GPU，无 CUDA 时使用 CPU。GPU 编号相对于 `CUDA_VISIBLE_DEVICES`；例如 `CUDA_VISIBLE_DEVICES=4,5` 时应传 `--gpus 0 1`。按机器空闲情况显式指定 GPU；脚本不会自动判断其他进程是否占用显存。`--set` 同时覆盖两种模型的公共配置；种子由 `--seeds`、设备由 `--gpus`、DAMPS 开关由对照组定义决定。`--output` 可指定一个尚不存在的结果目录。

每个任务使用同一种子、相同 backbone、采样及训练预算，仅切换 DAMPS；在各自验证集最佳 checkpoint 上评估测试集。结果写入 `runs/compare-*/`：

- `summary.md`：可直接阅读的四数据集对照表。
- `summary.csv`、`summary.json`：Recall@10/20、NDCG@10/20 的均值、样本标准差、配对绝对差和相对提升；单种子标准差为空。
- `runs.json`：完整配置、任务状态、最佳轮次、验证指标、测试指标、耗时、日志及 checkpoint 目录。
- 各任务目录下的 `train.log`：独立训练日志；启动时打印路径，可用 `tail -f` 查看。

绝对差为 `DAMPS − MGCN`；相对提升为 `100 × (DAMPS均值 − MGCN均值) / MGCN均值`。另外记录逐种子相对提升的均值 `paired_gain_pct_mean`，零基线种子从此统计中排除并报告有效数量；总体基线为零时相对提升为空。汇总仅纳入两种模型均成功完成的种子，不混合未配对结果。某项失败时保留其余结果并返回非零退出码，修改问题后可恢复；恢复使用原配置，可重新指定 GPU。

快速模式适合检查训练、梯度诊断和评估流程，不代表已经收敛。提升为正不能单独证明实现正确，短轮次无提升也不能判定实现错误；复现结论应结合完整训练、多种子波动和现有单元测试。首次运行仍需建立完整物品图，可能耗时较长，后续运行复用缓存。

## 超参数搜索

```bash
# 预览搜索规模，不训练
python -m mmrecsys.experiment.search --dry-run
# MMRec 默认结构，仅搜索 3 个 CL 权重：单种子共 6 次训练
python -m mmrecsys.experiment.search --device cuda:3 --seeds 999 --min-baseline 0.09
# 中断后继续（使用输出的 search_dir）
python -m mmrecsys.experiment.search --resume runs/search-baby-<时间>-<ID>
```

按验证 Recall@20 的配对相对提升排名，候选阶段不评估测试集。`--min-baseline 0.09` 是 Baby 的可选基线门槛，可防止较弱基线导致相对提升虚高；其他数据集需自行设置。默认固定 UI 层数 2、物品图层数 1、k=10，仅搜索 CL 权重 `[0.001, 0.01, 0.1]`。搜索空间见 `configs/search/damps_mgcn.yaml`，输出包含 `leaderboard.json`、`best.json` 和最佳配置。完整用法见 [搜索说明](docs/search.md)。

## 数据

将已发布数据放在项目根目录：

```text
data/<baby|sports|clothing|elec|microlens>/
  <dataset>.inter
  image_feat.npy
  text_feat.npy
```

沿用发布的全局编号、划分和特征行顺序；MicroLens 使用图像和文本特征。训练图仅使用 `x_label=0` 交互；验证标签选模，测试标签仅用于最终评估。数据需按 [数据使用指南](docs/data.md) 和 [MMRec 数据说明](https://github.com/enoche/MMRec/tree/master/data) 自行下载，训练入口不会自动下载或重新预处理。

## 恢复与评估

每次训练输出独立的 `runs/<run_id>/`，包括最终配置、环境及数据指纹、逐轮指标、最佳/最后 checkpoint 和测试结果。将下面的 `<run_id>` 替换为实际目录名：

```bash
python -m mmrecsys.cli train --resume runs/<run_id>/last.pt --set train.epochs=1000
python -m mmrecsys.cli evaluate --run runs/<run_id> --split test
```

配置支持 `--set dotted.key=value`，未知字段报错。相对配置和数据路径按项目根解析。原始 `runs/`、`data/`、`cache/` 及 checkpoint 不提交 Git；可审阅的 LIRDRec 精选结果和说明保存在 `docs/results/lirdrec/`。

代码位于 `src/mmrecsys/`，模型注册表负责声明所需模态和批次类型，训练器不包含模型名称分支。关键行为测试位于 `tests/`。
