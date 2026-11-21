# Training Performance and Timing

English · [简体中文](zh/performance.md) · [Documentation](README.md)

`metrics.jsonl` records `train_seconds`, `eval_seconds`, and `checkpoint_seconds`; `seconds` is the full epoch duration. Training timers synchronize CUDA at measurement boundaries. Use `train_seconds` when comparing training-only timings.

## Runtime behavior

- Negative sampling draws uniformly from the complement of training positives. Batched binary searches map ranks in that complement to items without rejection loops for dense users.
- When DAMPS owns the modality projections, the model skips backbone projections whose outputs would be unused. Training accumulates losses on the device, but retains per-step non-finite checks.
- Evaluation reuses sorted item IDs and computes metric hits in batches, retaining score ordering, ID tie-breaking, and the per-user aggregation order.
- When validation improves, the trainer serializes `best.pt` once and atomically copies it to `last.pt`. Otherwise it saves only `last.pt`. The files are independent, not hard links; the pair is not one atomic transaction. A copy failure preserves the old `last.pt` and raises an error.

## Device-resident sampling and resume

New CUDA runs default to `train.preload_to_device=true`. Training interactions and sampling indices move to the training GPU once; each epoch shuffles and samples on the GPU. Evaluation targets remain on CPU. Disable this with `--set train.preload_to_device=false` for a new run to compare the CPU sampler or reduce GPU memory use. CPU training always uses the CPU sampler.

CPU batched sampling uses algorithm version 2 and device-resident sampling uses version 3. Legacy checkpoints without a sampler version fall back to version 1. Keep the original sampling configuration on resume: the distribution is unchanged, but different algorithms and devices generate different random sequences.

## Historical measurements

Short Baby runs on an A100 with batch size 2048 measured 1.50 → 0.21 s for CPU negative sampling and 1.48–1.50 → 0.22–0.33 s for validation after batching improvements. A later CPU-versus-GPU sampler comparison on Baby measured 103 → 8 ms for sampling plus transfer and 2.01 → 1.45 s for training per epoch after warmup. The latter timings exclude validation, checkpointing, and diagnostic logging. These are historical measurements, not speed guarantees for the current configurations or full runs.
