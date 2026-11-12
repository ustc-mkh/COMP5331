"""Rebuild mean/sample-SD plots from verified portable evidence."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from results import DATASETS, SEEDS, VARIANTS, read, verify

NAMES = dict(baby="Baby", sports="Sports", clothing="Clothing", elec="Electronics", microlens="MicroLens")
LABELS = dict(baseline="BM3", full="BM3 + DAMPS", no_apc="w/o APC", no_avrf="w/o AVRF", no_imcf="w/o IMCF")
COLORS = {"baseline":"#697386", "full":"#2166ac", "no_apc":"#e08214", "no_avrf":"#7b3294", "no_imcf":"#00876c"}


def save(fig, directory, name):
    fig.savefig(directory/(name+".png"), dpi=180, bbox_inches="tight")
    fig.savefig(directory/(name+".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot(output):
    verify(output)
    directory = output/"figures"
    directory.mkdir(exist_ok=True)
    summary = read(output/"summary.json")["metrics"]
    lookup = {(r["dataset"],r["variant"],r["metric"]):r for r in summary}
    plt.rcParams.update({"font.size":10, "axes.spines.top":False, "axes.spines.right":False})
    fig, axes = plt.subplots(1,2,figsize=(13,4.3),layout="constrained")
    x = np.arange(len(DATASETS))
    for ax, metric, label in zip(axes,("recall@20","ndcg@20"),("Recall@20","NDCG@20")):
        for offset, variant in ((-.18,"baseline"),(.18,"full")):
            rows = [lookup[d,variant,metric] for d in DATASETS]
            ax.bar(x+offset,[r["mean"] for r in rows],.36, yerr=[r["sample_std"] for r in rows],
                   capsize=3,color=COLORS[variant],label=LABELS[variant])
        ax.set_xticks(x,[NAMES[d] for d in DATASETS]); ax.set_ylabel(label)
        ax.grid(axis="y",alpha=.2); ax.set_axisbelow(True); ax.set_ylim(bottom=0)
    axes[0].legend(frameon=False)
    fig.suptitle("BM3 and trainable projected DAMPS — three seeds, mean ± sample SD")
    save(fig,directory,"bm3-damps-comparison")
    if "no_apc" in {r["variant"] for r in summary}:
        fig, axes = plt.subplots(2,5,figsize=(17,7.5),layout="constrained")
        order = ("baseline","no_apc","no_avrf","no_imcf","full")
        for i,metric in enumerate(("recall@20","ndcg@20")):
            for j,dataset in enumerate(DATASETS):
                ax=axes[i,j]
                rows=[lookup[dataset,v,metric] for v in order]
                ax.bar(range(5),[r["mean"] for r in rows],yerr=[r["sample_std"] for r in rows],
                       color=[COLORS[v] for v in order],capsize=2)
                ax.set_xticks(range(5),[LABELS[v] for v in order],rotation=45,ha="right")
                ax.set_title(NAMES[dataset]); ax.set_ylim(bottom=0); ax.grid(axis="y",alpha=.2); ax.set_axisbelow(True)
                if j==0: ax.set_ylabel(("Recall@20","NDCG@20")[i])
        fig.suptitle("BM3 component-removal ablations — five seeds, mean ± sample SD")
        save(fig,directory,"component-ablations")
    # Individual paired observations keep seed variation visible alongside the aggregate.
    runs=read(output/"runs.json")["runs"]
    index={(r["dataset"],r["variant"],r["seed"]):r for r in runs}
    seeds=SEEDS
    fig,axes=plt.subplots(1,5,figsize=(16,3.8),layout="constrained")
    for ax,dataset in zip(axes,DATASETS):
        for seed in seeds:
            values=[index[dataset,v,seed]["test"]["recall@20"] for v in ("baseline","full")]
            ax.plot([0,1],values,marker="o",alpha=.7,lw=1,label=str(seed))
        ax.set_xticks([0,1],["BM3","+ DAMPS"]); ax.set_title(NAMES[dataset]); ax.grid(axis="y",alpha=.2)
    axes[0].set_ylabel("Recall@20"); axes[-1].legend(title="Seed",fontsize=8,frameon=False)
    fig.suptitle("Same-seed comparisons; each checkpoint selected by validation Recall@20")
    save(fig,directory,"paired-seeds")
    print("PLOTS",directory)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    plot(parser.parse_args().output.resolve())
