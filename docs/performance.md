# 训练性能与计时

训练日志和 `metrics.jsonl` 分别记录 `train_seconds`、`eval_seconds`、
`checkpoint_seconds`，原有 `seconds` 仍表示整轮耗时。训练计时在 CUDA
边界同步；与作者日志中的 `training time` 比较时，应使用 `train_seconds`。

性能优化包括：

- 按用户训练历史长度批量生成均匀的补集排名，用批量二分查找映射到负样本。
  不引入验证/测试交互，不采用可能在稠密用户上无限重试的拒绝采样。
- DAMPS 自有特征时直接调用内部投影，跳过输出会被覆盖的主干投影。
  保留主干参数注册以兼容 checkpoint。
- 在设备上累计损失，轮末才转为 Python 数值；每步非有限损失检查保留。
- 验证分块的 item ID 已升序，省去重复 ID 排序；保留稳定分数排序。
  用 CSR 批量查询命中项并计算 Recall/NDCG，保持逐用户累加顺序。
- `best.pt` / `last.pt` 保存策略保持不变。

## 随机序列与恢复

CPU 批量 sampler 使用 `algorithm_version=2`；显存常驻 sampler 使用版本 3。
负采样分布仍是从训练正例
补集中均匀采样，但与旧算法的具体随机序列不同；新实验的最终指标需要重新验证。

恢复没有版本标记的旧 checkpoint 时，自动启用旧算法（版本 1），保持其采样
序列。因此旧实验恢复后不会获得批量负采样的加速。新 checkpoint 保存版本号，
恢复后继续使用相同算法。无需改变现有训练命令。

## 短时性能测量

2026-09-28，Baby，A100 GPU 1，batch size 2048，当前
`configs/experiments/damps_mgcn_baby.yaml`，58 batches/epoch。使用 CUDA
同步分段计时，checkpoint 写入 `/tmp`，不包含数据加载和构图。

| 环节 | 优化前（秒） | 优化后（秒） |
| --- | ---: | ---: |
| 训练循环内负采样 | 1.50 | 0.21 |
| 模型训练等计算 | 1.64 | 1.61 |
| 验证 | 1.48–1.50 | 0.22–0.33 |
| 一次完整 checkpoint 保存 | 0.72–0.77 | 0.81–0.84 |

这是短时诊断测量，不是完整收敛实验，也不是与作者完整训练的端到端对比。
计时脚本的训练计算段保留逐步损失 `.item()`，用于前后同口径比较；正式 Trainer
另有设备端损失累计优化。实际保存耗时受实验目录存储性能影响，指标提升时会保存
两次 checkpoint。

前一阶段验证：88 项测试通过，覆盖稠密用户、负采样均匀性、旧采样序列、批量指标与
标量实现完全一致，以及连续训练/断点恢复的模型参数一致性。


## 显存常驻训练数据

新实验默认 `train.preload_to_device=true`。使用 CUDA 时，训练交互、负采样所需
的历史索引在开始训练前一次上传；每轮在同一 GPU 上完成打乱、整轮均匀补集采样，
再切片生成 batch。原本模型的多模态特征和图结构已常驻 GPU，无需再次复制。
验证目标和 CPU 元数据仍保留原存储方式。不会预存所有 epoch 的负样本。

```bash
python -m mmrecsys.cli train --config configs/experiments/damps_mgcn_baby.yaml --device cuda:1
# 对照 CPU 采样路径
python -m mmrecsys.cli train --config configs/experiments/damps_mgcn_baby.yaml --device cuda:1 --set train.preload_to_device=false
```

CPU 训练自动使用 CPU sampler。旧运行的保存配置缺少该字段时，按 false 处理，
恢复训练沿用旧算法。不要在恢复旧实验时切换该选项；如需 GPU 采样，请启动新实验。
GPU 和 CPU 随机序列不同，同一模式下可以重现并精确恢复。

2026-09-28，Baby / A100 GPU 2 / batch size 2048，同一模型交替运行 CPU 与 GPU
采样路径，排除首轮预热，后三轮训练耗时中位数：

| 项目 | CPU 批量采样 | GPU 常驻采样 |
| --- | ---: | ---: |
| 每轮采样并传输（独立测量，5 次中位数） | 103 ms | 8 ms |
| 每轮训练（不含验证与保存） | 2.01 s | 1.45 s |

此测试训练循环包含逐步有限性检查、反向传播和 Adam 更新，不包含诊断和损失日志。
训练耗时缩短约 28%，吞吐约为 1.39 倍；不是完整收敛运行的速度保证。
额外常驻张量约 5.7 MB，不含每轮打乱、负样本及查找临时张量。
最终 92 项测试通过，新增 GPU 采样边界/分布测试、真实 CUDA 连续训练与恢复
参数逐位一致测试，以及旧配置兼容测试。完整收敛指标尚未重跑。
