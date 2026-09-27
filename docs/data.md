# 多模态推荐数据使用指南

本项目直接使用 MMRec 已预处理的数据，供独立复现 BM3、MGCN、SMORE 使用。所有模型共享发布的用户/物品编号、数据划分和特征排列，无需重新做 k-core、编号映射、划分或特征提取。

项目已实现独立的数据加载器与 MGCN 训练入口，见 [MGCN 实现说明](mgcn.md)。本文保留发布数据格式与校验约定；同级外部目录 `../MMRec/` 仅供参考，不是运行依赖。

## 1. 数据获取与目录

下载入口见 [MMRec 数据说明](https://github.com/enoche/MMRec/tree/master/data)。该说明提供 Baby/Sports/Elec 的下载入口；Clothing 的来源和具体发布版本需由数据提供者另行记录，不将本地文件存在视为版本已经核验。

下载并解压后，目录应为：

```text
data/
├── README.md
├── baby/
│   ├── baby.inter
│   ├── image_feat.npy
│   └── text_feat.npy
├── clothing/
│   ├── clothing.inter
│   ├── image_feat.npy
│   └── text_feat.npy
├── elec/
│   ├── elec.inter
│   ├── image_feat.npy
│   └── text_feat.npy
└── sports/
    ├── sports.inter
    ├── image_feat.npy
    └── text_feat.npy
```

数据文件不提交到 Git，本使用指南保留在版本控制中。克隆项目后需要自行准备数据；若另一个发布版本的文件名不同，应根据其说明确认对应关系。

## 2. 文件含义

| 文件 | 格式和用途 |
| --- | --- |
| `<dataset>.inter` | 带表头的 TSV 交互文件，包含 `userID`、`itemID`、`rating`、`timestamp`、`x_label` |
| `image_feat.npy` | `[n_items, image_dim]` 图像特征，第 i 行对应物品 ID i |
| `text_feat.npy` | `[n_items, text_dim]` 文本特征，第 i 行对应物品 ID i |
| `i_id_mapping.csv`、`u_id_mapping.csv`（可选） | 原始 ID 与发布编号的对应关系，追溯来源时使用；实际分隔符以文件内容为准 |
| `user_graph_dict.npy`（可选） | 用户图辅助文件，本文三个目标方法不需要读取 |

`x_label` 的取值为：`0` 训练、`1` 验证、`2` 测试。对这三个隐式反馈推荐方法，每条训练交互作为正反馈，不再按 `rating` 筛选。`timestamp` 可用于分析，不用于重新划分。

用户与物品各自从 0 编号，属于两个不同的编号空间。构建联合二部图时，若用户占据前 `n_users` 个节点，物品节点索引应为 `n_users + itemID`；特征矩阵仍用原始的 `itemID` 索引。

以下是当前本地文件的实测统计，属于发布划分的原始记录数，尚未施加训练器的评估用户过滤：

| 数据集 | 用户数 | 物品数 | 训练交互 | 验证交互 | 测试交互 | 图像维度 | 文本维度 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baby | 19,445 | 7,050 | 118,551 | 20,559 | 21,682 | 4,096 | 384 |
| clothing | 39,387 | 23,033 | 197,338 | 40,150 | 41,189 | 4,096 | 384 |
| elec | 192,403 | 63,001 | 1,254,441 | 211,296 | 223,451 | 4,096 | 384 |
| sports | 35,598 | 18,357 | 218,409 | 37,899 | 40,029 | 4,096 | 384 |

不要在模型代码中写死这些维度；从加载后的数组读取。更换数据版本后，应重新核对统计与特征对应关系。

## 3. 最小读取示例

以下代码仅依赖 NumPy 和 Python 标准库。请在**项目根目录**运行，将 `dataset` 改为目标数据集：

```python
from pathlib import Path
import csv
import numpy as np

dataset = "baby"
root = Path("data") / dataset
with (root / f"{dataset}.inter").open(encoding="utf-8", newline="") as stream:
    rows = csv.DictReader(stream, delimiter="\t")
    interactions = np.asarray(
        [(int(r["userID"]), int(r["itemID"]), int(r["x_label"])) for r in rows],
        dtype=np.int64,
    )

assert interactions.ndim == 2 and interactions.shape[1] == 3
assert np.isin(interactions[:, 2], [0, 1, 2]).all()
n_users = int(interactions[:, 0].max()) + 1
n_items = int(interactions[:, 1].max()) + 1
assert np.array_equal(np.unique(interactions[:, 0]), np.arange(n_users))
assert np.array_equal(np.unique(interactions[:, 1]), np.arange(n_items))

train = interactions[interactions[:, 2] == 0, :2]
valid = interactions[interactions[:, 2] == 1, :2]
test = interactions[interactions[:, 2] == 2, :2]
image = np.load(root / "image_feat.npy", mmap_mode="r", allow_pickle=False)
text = np.load(root / "text_feat.npy", mmap_mode="r", allow_pickle=False)
assert image.ndim == text.ndim == 2
assert image.shape[0] == text.shape[0] == n_items

print(f"{dataset}: users={n_users}, items={n_items}")
print(f"train={len(train)}, valid={len(valid)}, test={len(test)}")
print(f"image={image.shape}, text={text.shape}")
```