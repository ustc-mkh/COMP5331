"""Figure-1-style diagnostics from trusted local MGCN checkpoints.

No graph construction, training, or checkpoint mutation. Appendix A metrics:
log2 power entropy; top-ceil(q*F) power ratio (q is unspecified in paper);
population variance across frequencies of unwrapped principal-angle differences.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import linear

from mmrecsys.nn.damps import DAMPS


def spectral_metrics(image, text):
    """Inputs are complex [items, frequencies] spectra, not amplitudes."""
    if image.ndim != 2 or image.shape != text.shape or not image.is_complex() or not text.is_complex():
        raise ValueError("Expected matching complex [items, frequencies] spectra")
    result = {}
    for name, z in (("image", image), ("text", text)):
        power = z.abs().double().square()
        total = power.sum(-1, keepdim=True)
        if not torch.isfinite(power).all() or (total <= 0).any():
            raise ValueError("Nonfinite or zero-energy spectrum has undefined metrics")
        p = power / total
        result[name + "_entropy"] = -(p * p.clamp_min(torch.finfo(p.dtype).tiny).log2()).sum(-1)
        sorted_p = p.sort(dim=-1, descending=True).values
        for q in (0.1, 0.2, 0.25, 0.3, 0.5):
            k = math.ceil(q * p.shape[-1])
            result[f"{name}_energy_top{q:g}"] = sorted_p[:, :k].sum(-1)
    delta = text.angle().double() - image.angle().double()
    result["phase_variance"] = delta.var(-1, correction=0)
    wrapped = torch.atan2(delta.sin(), delta.cos())
    result["wrapped_phase_variance"] = wrapped.var(-1, correction=0)
    result["circular_phase_variance"] = 1 - torch.complex(delta.cos().mean(-1), delta.sin().mean(-1)).abs()
    return {k: v.cpu().numpy() for k, v in result.items()}


def summarize(metrics):
    return {key: {"mean": float(x.mean()), "std": float(x.std()),
                  "q05": float(np.quantile(x, .05)), "q50": float(np.median(x)),
                  "q95": float(np.quantile(x, .95))} for key, x in metrics.items()}


def fft(pair):
    return tuple(torch.fft.rfft(x, dim=-1, norm="ortho") for x in pair)


@torch.no_grad()
def extract(path):
    # Repository checkpoints include trusted Python/NumPy RNG state.
    record = torch.load(path, map_location="cpu", weights_only=False)
    cfg, state = record["config"]["model"], record["model"]
    if cfg["name"] != "mgcn" or not cfg["trainable_features"]:
        raise ValueError("This extractor supports MGCN checkpoints with stored trainable features")
    meta = {"path": str(path), "fingerprint": record["fingerprint"],
            "config": record["config"], "state": record["state"]}
    if not cfg["damps_enabled"]:
        pair = tuple(linear(state[f"feature_embeddings.{m}.weight"],
                            state[f"projections.{m}.weight"], state[f"projections.{m}.bias"])
                     for m in ("image", "text"))
        return {"baseline": fft(pair)}, meta
    substate = {k.removeprefix("damps."): v for k, v in state.items() if k.startswith("damps.")}
    model = DAMPS(cfg["embedding_dim"], apc=cfg["damps_apc"], avrf=cfg["damps_avrf"],
                  imcf=cfg["damps_imcf"], eps=cfg["damps_eps"],
                  raw_image=substate["image_embedding.weight"], raw_text=substate["text_embedding.weight"])
    model.load_state_dict(substate, strict=True)
    model.eval()
    before = fft(model.project_features())
    apc = model.calibrate_phase(*before) if model.apc else before
    # Capture the actual spectra passed to irfft, without duplicating forward logic.
    from unittest.mock import patch
    original, captured = torch.fft.irfft, []
    def capture(z, *args, **kwargs):
        captured.append(z.detach().clone())
        return original(z, *args, **kwargs)
    with patch("torch.fft.irfft", side_effect=capture):
        after = model()
    if len(captured) != 2:
        raise ValueError("Expected two inverse FFT calls")
    meta["implementation_version"] = int(model.implementation_version)
    return {"before": before, "apc_only": apc, "filtered_spectrum": tuple(captured),
            "after": fft(after)}, meta


def plot_panel(before, after, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    fig = plt.figure(figsize=(15, 5.5), layout="constrained")
    grid = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.2])
    def density(ax, x, color, label):
        if x.std() < 1e-12:
            ax.axvline(x.mean(), color=color, label=label)
            return
        xs = np.linspace(x.min() - .1*x.std(), x.max() + .1*x.std(), 350)
        ys = gaussian_kde(x)(xs)
        ax.plot(xs, ys, color=color)
        ax.fill_between(xs, ys, alpha=.25, color=color, label=label)
        ax.axvline(x.mean(), color=color, ls="--", alpha=.7)
    for row, data in enumerate((before, after)):
        for col, (metric, xlabel) in enumerate((("entropy", "Spectral entropy (bits)"),
                                               ("energy_top0.25", "Energy ratio (top 25%, ceil)"))):
            ax = fig.add_subplot(grid[row, col])
            for modality, color in (("image", "#f4a582"), ("text", "#cf6290")):
                x = data[f"{modality}_{metric}"]
                density(ax, x, color, f"{modality}: μ={x.mean():.3f}")
            ax.set(xlabel=xlabel, ylabel="PD", title=f"({chr(97 + 2*row + col)}) {'Before' if row == 0 else 'After'}")
            ax.legend(fontsize=9)
    ax = fig.add_subplot(grid[:, 2])
    for data, color, label in ((before, "#68a6ad", "Before"), (after, "#c65252", "After")):
        x = data["phase_variance"]
        density(ax, x, color, f"{label}: μ={x.mean():.3f}, σ={x.std():.3f}")
    ax.set(xlabel="Variance of phase offset (rad²)", ylabel="PD", title="(e)")
    ax.legend(fontsize=9)
    fig.suptitle(title)
    fig.savefig(path.with_suffix(".png"), dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--damps", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    baseline, bmeta = extract(args.baseline)
    calibrated, dmeta = extract(args.damps)
    if bmeta["fingerprint"] != dmeta["fingerprint"]:
        raise ValueError("Dataset fingerprints differ")
    if "baseline" not in baseline or "after" not in calibrated:
        raise ValueError("Expected baseline then DAMPS checkpoints")
    if baseline["baseline"][0].shape != calibrated["after"][0].shape:
        raise ValueError("Representations have different shapes")
    spectra = {**baseline, **calibrated}
    metrics = {k: spectral_metrics(*v) for k, v in spectra.items()}
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "per_item.npz", **{f"{stage}/{key}": value for stage, values in metrics.items() for key, value in values.items()})
    summary = {"baseline": bmeta, "damps": dmeta, "n_items": spectra["before"][0].shape[0],
               "n_frequencies": spectra["before"][0].shape[1],
               "protocol": {"entropy": "log2 normalized rfft power; no endpoint doubling",
                            "energy": "top ceil(q*F) per item; paper does not specify q",
                            "phase": "var(angle(text)-angle(image)), correction=0, across frequencies",
                            "after": "rfft of actual real DAMPS output; filtered_spectrum also reported"},
               "metrics": {k: summarize(v) for k, v in metrics.items()}}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not args.no_plots:
        plot_panel(metrics["before"], metrics["after"], args.output / "within_model", "MGCN Clothing: same DAMPS checkpoint, before vs after calibration")
        plot_panel(metrics["baseline"], metrics["after"], args.output / "between_models", "MGCN Clothing: independently trained baseline vs DAMPS")
    print(json.dumps({stage: {key: value["mean"] for key, value in values.items()} for stage, values in summary["metrics"].items()}, indent=2))


if __name__ == "__main__":
    main()
