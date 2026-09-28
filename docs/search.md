# MGCN / DAMPS 配对超参数搜索

入口：`python -m mmrecsys.experiment.search`。每个候选设置对 MGCN 基线和完整 DAMPS 使用相同 backbone 超参数、种子、数据及评估协议。每次运行按验证指标独立选择最佳 epoch，搜索以两者验证指标的提升排名。

## 默认空间与目标

`configs/search/damps_mgcn.yaml` 依据 DAMPS 论文第 4.1 节及本地 `MMRec/src/configs/model/MGCN.yaml`：论文说明沿用 backbone 默认参数，不对集成 DAMPS 的 backbone 做专门调参。MMRec 固定图结构，只将 cl_loss 列入 hyper_parameters。

| 参数 | 默认设置 | 来源 |
| --- | --- | --- |
| model.n_ui_layers | 固定 2 | MMRec MGCN n_ui_layers |
| model.n_item_layers | 固定 1 | MMRec MGCN n_layers |
| model.knn_k | 固定 10 | MMRec MGCN knn_k |
| model.cl_weight | 搜索 0.001、0.01、0.1 | MMRec MGCN cl_loss |

共 **3 组设置**。默认 seed=999，baseline/full 配对共 **6 次训练**；三个种子共 18 次。此前默认的 192 组来自 DAMPS 发布包的宽网格，不是论文要求必须执行的搜索。现保存在 `configs/search/damps_mgcn_extended.yaml`，仅显式指定 --grid 时使用。

其他参数沿用已有配置：embedding_dim=64、Adam lr=0.001、weight_decay=0、reg_weight=1e−4、InfoNCE temperature=0.2、batch_size=2048、最多 1000 轮、patience=20、每轮验证、学习率衰减 0.96^(epoch/50)。模型配置的衰减覆盖 MMRec overall 中的 [1.0,50]；temperature=0.2 来自实际 InfoNCE 调用，而非代码中未使用的 self.tau=0.5。这里的 1000 是训练上限，不是必须训练到 1000 轮。

论文没有给出 Baby/Sports/Clothing/Elec 各自的最终 cl_loss，因此不能把某个值断言为论文最优。现有单次默认 0.01 仍作为启动值。若目标是严格沿用 backbone 选参，先按基线验证指标从三个权重中选出一个，再固定给 DAMPS；下面保留的“最大配对提升”目标是用户要求的探索性比较，不等同于该选参协议。

默认目标 relative：对每个种子计算 `100 × (DAMPS验证值 − 基线验证值) / 基线验证值`，再对种子取平均。使用实验配置的 eval.monitor，默认 Recall@20；要求 eval.mode=max。不是测试指标提升，也不是先对指标平均再计算比例。

`--objective absolute` 改为平均绝对差 `DAMPS验证值 − 基线验证值`。相对目标遇到任一种子的基线验证值为零时，该组标为不适合排名，而非人为添加分母 epsilon。分数相同时先比较 DAMPS 验证均值，再按固定网格顺序决定。

`--min-baseline` 按基线验证均值过滤，默认 0。最大相对提升不一定对应最好的推荐效果，尤其不能通过削弱基线获得虚高收益；建议同时检查 baseline_mean、damps_mean 和正收益种子数。Baby 可预先设定 0.09 门槛，但这不是通用数据集阈值。如果全部收益为负，脚本会明确报告没有找到正提升。

## 使用方法

```bash
# 只检查配置和组合数量，不训练、不创建运行目录
python -m mmrecsys.experiment.search --dry-run

# Baby，单种子验证筛选，候选阶段不运行测试评估
python -m mmrecsys.experiment.search --device cuda:3 --seeds 999 --min-baseline 0.09

# 多种子配对搜索
python -m mmrecsys.experiment.search --device cuda:3 --seeds 999 2024 2025 --min-baseline 0.09

# 换数据集
python -m mmrecsys.experiment.search --config configs/experiments/damps_mgcn_sports.yaml --device cuda:3

# 只运行网格中的前两组，适合确认流程；不是随机搜索或完整搜索
python -m mmrecsys.experiment.search --device cuda:3 --limit 2

# 可共用 CLI 参数覆盖，如修改训练轮数（短跑不能用于论文指标结论）
python -m mmrecsys.experiment.search --device cpu --limit 1 --set train.epochs=2
```

自定义 YAML 网格通过 `--grid path/to/grid.yaml` 传入，键为 dotted configuration key，值为非空候选列表。允许搜索 `model.n_ui_layers`、`model.n_item_layers`、`model.knn_k`、`model.cl_weight`、`model.reg_weight`、`model.embedding_dim`、`model.temperature`、`optimizer.lr`。未知字段、重复候选和非法参数会报错；组件开关由配对实验固定管理。网格值覆盖基础配置/--set 中同名值，其他配置对所有组共同生效。

例如先固定图结构，只搜索对比权重和学习率：

```yaml
model.cl_weight: [0.001, 0.01, 0.1]
optimizer.lr: [0.0005, 0.001]
```

原宽网格仍可按需执行：

```bash
python -m mmrecsys.experiment.search --grid configs/search/damps_mgcn_extended.yaml --dry-run
```

**旧搜索不会自动缩小。** `--resume` 读取已有 plan.json 中冻结的网格，不读取新的默认 YAML；旧 192 组计划仍会继续原计划。要采用三组小网格，请新建搜索。此次配置调整不修改已有运行、结果或计划，也不会自动终止正在运行的进程。旧结果中同一参数配置可作为参考，但新搜索不会自动导入其他目录的 checkpoint。

## 产物与恢复

新建 `runs/search-<dataset>-<time>-<id>/`：

- `plan.json`：基础配置、网格、种子、目标、数据指纹、当前源代码摘要。
- `trial-XXXX/seed-N/baseline|full/<run_id>/`：每次训练的配置、日志与 checkpoint；候选训练结束只生成 `validation_result.json`，不生成含测试指标的 result.json。
- `leaderboard.json`：所有已完成配对候选的验证排名，包括逐种子路径、验证值、均值提升、相对提升标准差与正提升种子数。单种子标准差为 null。
- `best.json`：当前满足门槛的最佳候选，搜索过程中随完成的候选更新。
- `best_baseline.yaml` / `best_full.yaml`：可直接传给常规 train 的最佳设置，种子默认使用本次搜索的第一个种子。

```bash
python -m mmrecsys.experiment.search --resume runs/search-baby-<时间>-<ID>
```

恢复使用原计划，不接受重新指定 device、grid、seeds、--set 等参数。已完成的候选从结果文件读取；中断的运行从 last.pt 继续；若首轮 checkpoint 尚未写出，则重新启动该次训练。源代码或数据指纹变化会拒绝恢复，避免将不同实现混入同一排名。不要同时启动两个进程恢复同一个搜索目录。

## 最终测试与结论边界

完成验证选参后，才可评估选出的同一组参数：

```bash
python -m mmrecsys.experiment.search --resume runs/search-baby-<时间>-<ID> --evaluate-best
```

也可在新搜索命令中加 `--evaluate-best`，它会等全部候选完成后再评估胜出组合的基线和 DAMPS checkpoints。结果写入 `best_test.json`，不会改变验证排名；已存在该文件时跳过重复评估。不将测试结果反馈给本轮选参。

脚本找到的是所列网格、种子和训练预算下的最大验证提升，不保证全局最优、测试提升或统计显著性。建议单种子筛选后用多个独立种子验证候选，并同时报告绝对性能。搜索开始前确定目标和门槛，避免观察测试结果后反复调整。
