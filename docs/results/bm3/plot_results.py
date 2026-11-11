"""Generate five-dataset paper comparison and validation convergence figures."""
import json
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATASETS = ['baby', 'sports', 'clothing', 'elec', 'microlens']
LABELS = ['Baby', 'Sports', 'Clothing', 'Electronics', 'MicroLens']
METRICS = ['recall@10', 'recall@20', 'ndcg@10', 'ndcg@20']

def main():
    rows = json.loads((ROOT/'summary.json').read_text())['metrics']
    paper = json.loads((ROOT/'paper-table2.json').read_text())['datasets']
    output = ROOT/'figures'
    output.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for ax, metric in zip(axes.flat,METRICS):
        selected = [next(r for r in rows if r['dataset']==d and r['metric']==metric) for d in DATASETS]
        ax.bar([i-.18 for i in range(5)], [paper[d][metric] for d in DATASETS], width=.36,label='Paper BM3 raw',color='#a6adb5')
        ax.bar([i+.18 for i in range(5)], [r['mean'] for r in selected], width=.36,
               yerr=[r['sample_std'] for r in selected],capsize=3,label='Reproduction (5 seeds)',color='#3977a8')
        ax.set_xticks(range(5),LABELS)
        ax.set(title=metric,ylabel='Test metric',ylim=(0,None))
        ax.grid(axis='y',alpha=.2)
        ax.set_axisbelow(True)
    axes[0,0].legend(fontsize=9)
    fig.suptitle('Original BM3: paper vs. reproduction (mean ± sample SD)')
    for ext in ['png','pdf']: fig.savefig(output/('paper-comparison.'+ext),dpi=180)
    plt.close(fig)
    runs = json.loads((ROOT/'runs.json').read_text())['runs']
    fig, axes = plt.subplots(1,5,figsize=(16,3.3),constrained_layout=True)
    for ax,d,label in zip(axes,DATASETS,LABELS):
        for run in [r for r in runs if r['dataset']==d]:
            with (ROOT/run['history']).open(newline='') as stream: history=list(csv.DictReader(stream))
            line,=ax.plot([int(r['epoch']) for r in history], [float(r['valid_recall@20']) for r in history],label=str(run['seed']),lw=1)
            ax.plot(run['best_epoch'],run['best_validation']['recall@20'],'.',color=line.get_color())
        ax.set(title=label,xlabel='Epoch',ylabel='Validation Recall@20')
        ax.grid(alpha=.2)
    axes[-1].legend(title='Seed',fontsize=7)
    for ext in ['png','pdf']: fig.savefig(output/('validation-curves.'+ext),dpi=180)
    plt.close(fig)

if __name__ == '__main__': main()
