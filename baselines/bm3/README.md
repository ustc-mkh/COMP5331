# BM3 baseline 五数据集复现

本目录提供原始 BM3 的运行代码、环境与五数据集配置。Baby、Sports、Clothing、Electronics、MicroLens 各完成 5 个种子，共 25 次正式实验；结果均通过最佳 checkpoint 的独立重载核验。

- [实验结果、样本标准差与论文对照](../../docs/results/bm3/README.md)
- [实际运行配置和评估协议](../../docs/results/bm3/training-protocol.json)
- [源码迁移清单](migration_manifest.json)

## 代码组织

沿用组内 [SMORE 独立运行代码的提交方式](../../task4/README.md)，`runtime/` 保存完成实验所用的 BM3 运行实现，结果入口位于 `docs/results/bm3/`。本目录使用独立的 Python 3.8 环境；项目根目录的 `mmrecsys` 训练入口保持原样。

`runtime/upstream/BM3/` 固定 enoche/BM3 commit `ef82a333823ae01ac5964039720f697532b40ea3`。`bm3baseline/model.py` 继承原始 forward、损失、预测器和可训练图文特征，仅用 SciPy 公共接口替换邻接矩阵构建时的私有 DOK API。数据加载沿用发布编号与划分，训练封装增加验证集选模、断点恢复和独立重载核验。

`migration_manifest.json` 记录实际执行源码的 SHA256。`runtime/bm3baseline/` 与四个后续数据集的训练归档逐文件一致。Baby 在多数据集扩展前完成，对应适配层保存于 `source-history/baby-20261107/`；其模型核心与后续版本相同。上游 commit 标识官方 BM3 源码，适配层身份由迁移清单记录。

## 环境与数据

从仓库根目录进入独立运行目录，创建自己的 Conda 环境：

```bash
cd baselines/bm3/runtime
conda create -n bm3-baseline python=3.8.20 pip=24.3.1
conda activate bm3-baseline
python -m pip install -r requirements.txt
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip check
python scripts/fetch_upstream.py --verify
python -m pytest -q
```

CPU 验证可将 PyTorch 安装索引改为 `https://download.pytorch.org/whl/cpu`。正式结果的环境为 Python 3.8.20、PyTorch 2.4.1+cu121、NumPy 1.22.4、SciPy 1.10.1，设备为 RTX 3090。

使用 [DAMPS 官方仓库](https://github.com/Wmhwxl/KDD2026_DAMPS) 提供的 [Amazon 数据](https://drive.google.com/drive/folders/13cBy1EA_saTUuXxVllKgtfci2A09jyaG) 和 [MicroLens 数据](https://drive.google.com/drive/folders/14UyTAh_YyDV8vzXteBJiy9jv8TBDK43w)。每个数据集目录需包含 `<dataset>.inter`、`image_feat.npy`、`text_feat.npy`，其中 Electronics 的目录名为 `elec`。下载后与 `configs/*_files.json` 中的 SHA256 校验。MicroLens 使用图文特征。

## 训练、重载和汇总

下面脚本在用户指定的一张空闲 GPU 上依次运行 25 个配置；数据与输出目录设置在仓库外。每个种子最多 1000 轮，按验证集 Recall@20 早停。执行前替换两处路径并选择空闲 GPU。

```bash
export DATA_ROOT=/path/to/released-data
export OUTPUT_ROOT=/path/to/new-bm3-runs
export CUDA_VISIBLE_DEVICES=0
set -euo pipefail
for dataset in baby sports clothing elec microlens; do
  python run.py check-data --data-dir "$DATA_ROOT/$dataset" \
    --output "$OUTPUT_ROOT/validation/$dataset.json"
  for seed in 999 2024 2025 2026 42; do
    run_dir="$OUTPUT_ROOT/$dataset/seed-$seed"
    PYTHONHASHSEED="$seed" python run.py train \
      --config "configs/$dataset.yaml" --data-dir "$DATA_ROOT/$dataset" \
      --run-dir "$run_dir" --device cuda --seed "$seed"
    python run.py evaluate --checkpoint "$run_dir/best.pt" \
      --data-dir "$DATA_ROOT/$dataset" --device cuda \
      --output "$run_dir/reloaded.json"
    python run.py plot --run-dir "$run_dir"
  done
  python run.py aggregate --runs "$OUTPUT_ROOT/$dataset" \
    --output "$OUTPUT_ROOT/summary/$dataset"
done
```

恢复中断训练时，在原命令中添加 `--resume "$run_dir/last.pt"` 并保留同目录的 `best.pt`。新训练目录必须不存在，避免覆盖已完成结果。运行封装中的 `aggregate` 仅内置 Baby 论文参考值；本次五数据集论文对照由结果目录的 `paper-table2.json` 与验证脚本完成。

## 验证与许可

从仓库根目录运行轻量结果验证，无需 GPU、数据集或 PyTorch：

```bash
python docs/results/bm3/verify.py
```

完整运行测试在上述独立环境的 `runtime/` 内执行，覆盖原始模型方法一致性、图归一化、评估过滤、断点恢复及多数据集加载。数据、权重、环境目录与完整机器日志保留在实验存储中；版本控制包含结果字段、逐轮曲线、指纹和来源。BM3 代码保留 [GPL-3.0 许可](runtime/LICENSE) 和上游版权信息。
