# Dataset Guide

English · [简体中文](zh/data.md) · [Documentation](README.md)

Download the preprocessed datasets from the [MMRec data page](https://github.com/enoche/MMRec/tree/master/data). MMRec is a reference and data source, not a runtime dependency. The project reuses the released split, IDs, and feature order; there is no download, reindexing, split generation, k-core filtering, or feature extraction step.

## File layout

```text
data/
├── baby/
│   ├── baby.inter
│   ├── image_feat.npy
│   └── text_feat.npy
└── ...
```

Use the same layout for `sports`, `clothing`, `elec` (Electronics), and `microlens`, replacing `baby.inter` with the corresponding dataset filename. MicroLens uses the released image and text features, not video features. To use an external data location, set `data.root` to the directory containing the dataset subdirectories. Data files are not committed.

| File | Format and use |
| --- | --- |
| `<dataset>.inter` | TSV with a header; the loader uses `userID`, `itemID`, and `x_label`. Published files also include `rating` and `timestamp`. |
| `image_feat.npy` | A two-dimensional NumPy array of image features, with row i for item ID i. |
| `text_feat.npy` | A two-dimensional NumPy array of text features, with row i for item ID i. |
| ID mapping CSVs and `user_graph_dict.npy` | Optional upstream artifacts; not required for the standard MGCN or LIRDRec commands. |

`x_label` is 0 for training, 1 for validation, and 2 for test. Every training interaction counts as positive feedback regardless of rating. Users and items have separate contiguous, zero-based ID spaces. In a joint bipartite graph, item nodes use an offset of `n_users`; feature arrays still use the original item ID. Keep the feature arrays aligned and finite.

## Reference counts

These counts were observed in the released files used in this project, before filtering evaluation users:

| Dataset | Users | Items | Train | Validation | Test | Image width | Text width |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baby | 19,445 | 7,050 | 118,551 | 20,559 | 21,682 | 4,096 | 384 |
| clothing | 39,387 | 23,033 | 197,338 | 40,150 | 41,189 | 4,096 | 384 |
| elec | 192,403 | 63,001 | 1,254,441 | 211,296 | 223,451 | 4,096 | 384 |
| sports | 35,598 | 18,357 | 218,409 | 37,899 | 40,029 | 4,096 | 384 |
| microlens | 98,129 | 17,228 | 500,064 | 101,121 | 103,989 | 1,024 | 1,024 |

The paper describes an 80/10/10 split, but the released files can differ (Baby is about 73.73/12.79/13.48). Keep the released split rather than changing it to match the stated ratios. Verify IDs, feature rows, and counts again if you switch releases; equal totals do not establish identical splits.

## Minimal inspection example

Run from the repository root with NumPy installed:

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
