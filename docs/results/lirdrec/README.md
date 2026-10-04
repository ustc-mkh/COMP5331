# LIRDRec 五数据集实验结果

本目录交付完整的 **75 次实验**：Baby、Sports、Clothing、Electronics、MicroLens，各运行 baseline、完整 DAMPS（full）、去除 APC、去除 AVRF、去除 IMCF 五组，以及 999、2024、2025 三个种子。最后一组于 2026-10-02 21:48 UTC 完成。结果来自固定配置的组内实现，**没有复现论文报告的 DAMPS 提升**。

## 结果

下面均为三个种子的测试集 Recall@20 算术平均。相对变化为 `(full / baseline - 1) × 100%`。

| 数据集 | 本地 baseline | 本地 full | 论文 full | 本地 full 相对 baseline |
| --- | ---: | ---: | ---: | ---: |
| Baby | 0.10552 | 0.10325 | 0.10610 | −2.15% |
| Sports | 0.11356 | 0.11163 | 0.11600 | −1.71% |
| Clothing | 0.09371 | 0.09049 | 0.09940 | −3.44% |
| Electronics | 0.06741 | 0.06639 | 0.06840 | −1.52% |
| MicroLens | 0.11238 | 0.11216 | 0.12390 | −0.20% |

五个数据集的 Recall@10/20、NDCG@10/20 合计 20 项均值中，full **全部低于本地 baseline，也全部低于论文 full**。相较本地去除 APC、AVRF、IMCF 的结果，full 分别在 12/20、0/20、5/20 项上更高；论文 Table 3 中对应比较均为 20/20。去除 AVRF 后每项均值都优于 full，是后续排查线索，不能据此认定实现存在错误或否定方法。

## 实验协议与解释边界

- 使用 MMRec 发布划分和图文特征，不重新划分数据；MicroLens 不使用视频特征。五组共享相同骨干、数据与训练预算，每组保留全部三个种子。
- 固定 64 维、2 层用户物品图、1 层物品图、k=10、学习率 0.0001；完整配置见 [training-protocol.json](training-protocol.json)。这些是参考代码候选中的固定起始组合，没有完成逐数据集超参数搜索，也没有使用测试集挑选参数。
- 每次最多训练 1000 轮，每轮验证；验证集 Recall@20 连续 20 次未提升即停止，选验证最佳 checkpoint 后评估测试集。使用全物品排名、训练历史过滤、按用户宏平均；相同分数按物品编号排序。
- `sample_std` 是三个种子的样本标准差（分母为 n−1）。与本地 baseline 的差值按相同种子配对。指标值用小数表示，百分比字段是相对差异，不是百分点；这些描述统计不构成显著性检验。
- 论文 Table 2/3 的 LIRDRec 数值作为外部参照，论文没有明确披露这些值对应的种子数量。Table 3 是去除组件的消融；Figure 3 的 MGCN 单组件添加实验不作为本目录的 LIRDRec 目标。
- 本实现保留原 LIRDRec 骨干，在两层 MLP 后接入组内共享 DAMPS；参考 DAMPS 文件的内部特征学习、隐藏宽度及图/DCT 输入行为不同。评估缓存、负采样、kNN 同分规则和 DAMPS 数值约定也存在已记录的差异。详见 [LIRDRec 接入说明](../../lirdrec.md) 和 [DAMPS 实现说明](../../damps.md)。因此不能将这里的结果称为参考 DAMPS 文件的精确复现，更不能称为论文性能复现成功。

## 文件说明

| 文件 | 用途 |
| --- | --- |
| [runs.json](runs.json) | 75 次结果：数据集、变体、种子、最佳轮次、验证指标及测试四指标；不含原始运行目录。 |
| [summary.csv](summary.csv)、[summary.json](summary.json) | 25 组 × 4 指标的均值、样本标准差和相同种子的 baseline 配对差值。 |
| [three-seeds-vs-paper.csv](three-seeds-vs-paper.csv) | 100 项论文对照，保留每个种子的原始指标、均值、标准差及绝对/相对差距。 |
| [three-seeds-damps-gains.csv](three-seeds-damps-gains.csv) | 20 项 full 对 baseline 的本地与论文增益，以及本地配对差值。 |
| [paper-table2.json](paper-table2.json)、[paper-table3.json](paper-table3.json) | 论文 LIRDRec 基线、full 和消融数值；包含仓库内 PDF 的相对路径、物理页码及 SHA256。 |
| [data-fingerprints.json](data-fingerprints.json) | 训练实际使用的 15 个数据文件的 SHA256、大小、公开下载来源，以及加载后的数据统计与指纹。 |
| [training-source.json](training-source.json) | 训练时捕获的源码/配置逐文件 SHA256、按数据集的组合指纹及历史工作树说明。 |
| [training-environment.json](training-environment.json) | 75 份训练 manifest 一致记录的 Python、依赖及 GPU 型号；未记录的环境项目明确列出。 |
| [training-protocol.json](training-protocol.json) | 可组合还原 75 次有效配置的共同设置、数据集/变体覆盖和种子；省略部署路径。 |
| [export-provenance.json](export-provenance.json) | 原实验批次标识、时间和输入产物 SHA256，说明导出核验范围。 |
| [verify.py](verify.py) | 仅用 Python 标准库，从逐次结果重新核验汇总、论文对照、增益和来源指纹；不训练、不加载 checkpoint。 |

训练时所有 manifest 都记录了有未提交改动的工作树：其中 Git 提交号只是历史基点，不能单独还原新增的 LIRDRec 实现。本目录以训练时保存的逐文件哈希标识实际源码，并在导出时与归档源码核对。PR 整理时创建的实现提交 `fa30173a1dbd7e70e37e01ba022c401df820589a` 与这 31 份训练文件逐字节一致，可用于定位对应实现；它是在训练结束后创建的提交，不是当时 manifest 记录的历史基点。后续文档提交只改变了模型配置的一行说明注释，其解析后配置不变；训练时哈希和导出时差异均保留在 `training-source.json`。

数据哈希是本地下载后记录的内容身份，不是发布者独立签发的校验值。本目录不包含原始数据、checkpoint、逐轮日志、图缓存、依赖环境、运行进程信息或硬件身份。实际训练版本来自运行时 manifest，不用导出当天的环境替代；未记录 NVIDIA 驱动和完整依赖锁，因此不能声称环境逐字节可复现。

## 核验

在仓库根目录运行：

```bash
python3 docs/results/lirdrec/verify.py
```

核验不依赖训练数据、PyTorch 或 GPU。它检查 75 次组合齐全、100 项三种子均值/样本标准差、论文数值和 PDF 身份、配对增益以及导出中的机器路径。原始数据文件哈希来自冻结的实验记录，本脚本不下载或重新扫描原始数据，也不重新执行训练。
