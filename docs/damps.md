# DAMPS Implementation and Reproduction Conventions

English · [简体中文](zh/damps.md) · [Documentation](README.md)

The shared operator implements *Enhancing Multimodal Recommendation via Multimodal Representation Calibration in Spectral Domain* (KDD 2026). It follows the reference implementation for the phase rotation direction, orthonormal FFT, AVRF initialization, IMCF epsilon, and branch fusion initialization. Matching those numerical conventions does not guarantee matching training trajectories or paper metrics.

## Integration and equations

MGCN's DAMPS module owns feature tables and independent projections and runs before the behavior-guided purifier. The backbone projections do not feed the DAMPS path and receive no gradient under the default final-representation regularization. LIRDRec uses the projected-input interface after its original modality MLPs and leaves its raw-feature graph and DCT branch unchanged. See [BM3, SMORE, and FREEDOM](models.md) and the [experiment guide](experiments.md) for other integrations.

| Paper | Implementation |
| --- | --- |
| (1)–(4), projection and FFT | Apply `rfft` to the last dimension of all item projections, producing F = floor(D/2) + 1 frequencies. |
| (5)–(8), APC | Estimate a fixed circular mean θ of text-minus-image phase across items. Rotate image by −(θ/2+ψ) and text by +(θ/2+ψ); ψ is trainable and starts at zero. |
| (9)–(18), AVRF | Estimate power reliability from amplitude median, MAD and population variance across items. Noise is `(1.4826 × MAD)²`, signal is `max(variance − noise, 0)`; standardize the signal-to-total ratio, then apply sigmoid and logit to initialize trainable weights. These weights are used directly in forward passes, without clamping. |
| (19)–(22), IMCF | Use the squared magnitude of `image × conj(text)` divided by the product of the two powers plus 1e−8. |
| (23)–(24), fusion and inverse | Apply AVRF and IMCF in parallel after APC; combine with a shared softmax over logits initialized to [0.6, 0.4]. Use `irfft` with an explicit output length. |

Both FFT directions use orthonormal normalization. The default output width is D, including when D is odd; FREEDOM explicitly requests original modality widths for its one-time calibration. The operator rotates all bins and lets `irfft` ignore imaginary terms at DC and, for even output lengths, Nyquist. When both circular-mean components vanish, use zero offset. AVRF uses `model.damps_eps` (default 1e−6) for the ratio denominator; standardization and logit have their own numerical guards. Standalone projected-input usage requires one initialization across all item projections before the first forward pass; repeating initialization is an error.

The author rotation adds θ + 2ψ to the text-minus-image phase difference rather than subtracting θ. The code retains that convention. IMCF is an elementwise power ratio, mathematically Q/(Q+1e−8) with Q the product of modality powers; it is not averaged coherence. Version-4 checkpoints from the reverse-rotation experiment are rejected by the version-3 operator. Checkpoint compatibility also depends on the integration and the model configuration.

## Ablation and diagnostics

```bash
python -m mmrecsys.cli train --config configs/experiments/damps_mgcn_baby.yaml --set model.damps_apc=false
python -m mmrecsys.experiment.ablation --config configs/experiments/damps_mgcn_baby.yaml --seeds 999 2024 2025
```

The other component switches are `model.damps_avrf` and `model.damps_imcf`. Without APC there is no phase rotation. With one filter enabled it has unit weight; with neither filter, invert the APC spectrum directly. When all three switches are off, the default-width operator returns its projected input. In the owned-feature interface used by MGCN, this is the DAMPS module's own projection, so disabling all three components is not equivalent to disabling DAMPS.

Where available, `metrics.jsonl` records diagnostic snapshots after backward and before the optimizer step on the first batch of each epoch. These include gate saturation (below 0.01 or above 0.99), exact zeros/ones, mean sigmoid derivative, gradient L2 norms and zero fractions, and AVRF/IMCF fusion weights when both filters are active. They are not epoch averages and do not modify the loss.

Numerical and integration tests check independent formula references, phase signs, gradients, degenerate inputs, component switches, and checkpoint behavior. See [LIRDRec](lirdrec.md) for the limits of the historical reproduction results.
