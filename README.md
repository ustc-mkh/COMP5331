# COMP5331 · 多模态推荐复现

已按照 [架构设计](docs/architecture.md) 独立实现 MGCN 及统一的数据、采样、训练和评估框架，直接读取 MMRec 发布的数据，不依赖 MMRec 运行。BM3、SMORE 等其他方法尚未实现。

## 安装与训练

需要 Python 3.10+、PyTorch、NumPy、SciPy、PyYAML；GPU 训练需要匹配的 PyTorch/CUDA 环境。

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

默认自动选择 CUDA 或 CPU，可使用 `--device cpu` 或 `--device cuda:0` 指定设备。

训练命令：

```bash
python -m mmrecsys.cli train --config configs/experiments/mgcn_baby.yaml  --device cuda:2
```

其他数据集的完整训练配置（Elec 对应 Electronics）：

```bash
python -m mmrecsys.cli train --config configs/experiments/mgcn_sports.yaml
python -m mmrecsys.cli train --config configs/experiments/mgcn_clothing.yaml
python -m mmrecsys.cli train --config configs/experiments/mgcn_elec.yaml
```

## 数据

将已发布数据放在项目根目录：

```text
data/<baby|sports|clothing|elec>/
  <dataset>.inter
  image_feat.npy
  text_feat.npy
```

沿用发布的全局编号、划分和特征行顺序。训练图仅使用 `x_label=0` 交互；验证标签选模，测试标签仅用于最终评估。数据格式与来源见 [数据使用指南](docs/data.md) 和 [MMRec 数据说明](https://github.com/enoche/MMRec/tree/master/data)。不重新预处理或下载数据。

## 恢复与评估

每次训练输出独立的 `runs/<run_id>/`，包括最终配置、环境及数据指纹、逐轮指标、最佳/最后 checkpoint 和测试结果。将下面的 `<run_id>` 替换为实际目录名：

```bash
python -m mmrecsys.cli train --resume runs/<run_id>/last.pt --set train.epochs=1000
python -m mmrecsys.cli evaluate --run runs/<run_id> --split test
```

配置支持 `--set dotted.key=value`，未知字段报错。相对配置和数据路径按项目根解析。图缓存位于 `cache/`，实验产物与缓存均不提交 Git。

代码位于 `src/mmrecsys/`，模型注册表负责声明所需模态和批次类型，训练器不包含模型名称分支。关键行为测试位于 `tests/`。
