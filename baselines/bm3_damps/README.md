# BM3 + DAMPS：五数据集三种子实验归档

本目录保存已经完成的 15 次 BM3+DAMPS 训练代码及其原始 BM3 依赖。实验使用 Baby、Sports、Clothing、Electronics、MicroLens，种子为 **999、2024、2025**，与既有 BM3 的相同 15 次运行配对。结果、逐轮历史、均值、样本标准差和图表见[实验结果](../../docs/results/bm3-damps/README.md)。

主分支已将 BM3 合入 `src/mmrecsys`。本次运行早于该迁移，因此将实际执行的目录结构保存在 `snapshot/`，并保持训练文件字节与 checkpoint 中的摘要一致。统一框架中的模型、配置、入口及组员文件均不修改。复跑这组结果应使用下面的归档入口；新框架的训练结果需要另外记录。

## 接入方式

在 BM3 自身的图文投影后，对 64 维特征执行共享 DAMPS，再构建 stop-gradient/dropout 目标。每次计算训练损失都会调用校准模块，使其参数接收梯度。保留 BM3 的图传播、预测器、损失权重与推荐打分；完整 DAMPS 增加 101 个可训练标量。实现标识为 `projected_trainable_v1`。

共享模块取自项目提交 `13691bf6016939f68bf487ca547febca789f67e7`，原始 BM3 取自 `enoche/BM3` 的 `ef82a333823ae01ac5964039720f697532b40ea3`。作者 v1.0.0 的 BM3 适配器只在初始化时执行校准，再将输出包装为新的 embedding 参数，重建宽度也不同。因此本实验衡量的是投影后可训练接入的效果。作者参考文件及详细审计保留在 [snapshot/baselines/bm3_damps](snapshot/baselines/bm3_damps/)。该目录内 README 是运行时文档，其相对路径按归档根目录解释；当前安装与入口以下文为准。

## 环境与复跑

在仓库根目录使用 Linux + Conda：

```bash
SNAPSHOT=baselines/bm3_damps/snapshot
conda env create -f "$SNAPSHOT/baselines/bm3/runtime/environment.yml"
conda activate bm3-baby
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r "$SNAPSHOT/baselines/bm3/runtime/requirements.txt"
python -m pytest -q "$SNAPSHOT/baselines/bm3_damps/tests"
python "$SNAPSHOT/baselines/bm3_damps/smoke.py" \
  --data /path/to/datasets --output /path/to/new-smoke --variants full
nohup python -u "$SNAPSHOT/baselines/bm3_damps/run_selected.py" \
  --data /path/to/datasets --output /path/to/new-suite \
  --gpus GPU-uuid-1 GPU-uuid-2 \
  > /path/to/launcher.log 2>&1 < /dev/null &
```

数据根目录下须有 `baby/`、`sports/`、`clothing/`、`elec/`、`microlens/`，每个目录包含相应 `.inter`、`image_feat.npy`、`text_feat.npy`；文件名与哈希见归档 runtime 的 `configs/*_files.json`。使用空闲 GPU 的真实 UUID。入口只调度 15 次完整 DAMPS 训练，每张卡同时一个进程；已完成运行可用相同参数加 `--resume` 继续。旧 `run_suite.py` 为保持训练摘要而保留，**不要直接用它启动队列**。

每次最多 1000 轮，按首次验证 Recall@20 最大值选模，连续 21 轮不提升时停止。全物品测试排序屏蔽训练历史并保留验证交互。所有配置与对应种子的 BM3 相同，仅增加 DAMPS 标识。所有完成的 checkpoint 均独立重载，核对四项验证指标、四项测试指标与最佳轮次。

## 核验与重新导出

无需 GPU、数据或第三方依赖，即可核验本 PR 的完整结果：

```bash
python baselines/bm3_damps/verify.py
```

复跑完成后，从归档根目录导出新结果，避免与已归档结果混淆：

```bash
cd baselines/bm3_damps/snapshot
python baselines/bm3_damps/results.py --suite /path/to/new-suite \
  --stage main --output docs/results/bm3-damps
python baselines/bm3_damps/plot_results.py --output docs/results/bm3-damps
```

`snapshot/docs/results/bm3/runs.json` 保留原 25 次 BM3 的配置和来源，作为冻结队列的输入；当前导出器只选择 999/2024/2025。其他种子及组件消融不属于本次结果。原始数据、checkpoint、环境目录不提交；源码及证据哈希见结果中的 `provenance.json` 与 `packaging.json`。GPL-3.0 许可与上游声明随归档保留。
