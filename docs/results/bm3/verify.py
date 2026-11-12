"""Verify the portable BM3 evidence using only the Python standard library."""
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
BASE = REPO/'baselines/bm3'
DATASETS = ('baby', 'sports', 'clothing', 'elec', 'microlens')
SEEDS = (999, 2024, 2025, 2026, 42)
METRICS = ('recall@10', 'recall@20', 'ndcg@10', 'ndcg@20')

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def close(actual, expected):
    assert math.isclose(float(actual), expected, rel_tol=1e-12, abs_tol=1e-15), (actual, expected)

def main():
    runs = read(ROOT/'runs.json')
    assert runs['complete'] and runs['n_runs'] == len(runs['runs']) == 25
    by_run = {(r['dataset'], r['seed']):r for r in runs['runs']}
    assert set(by_run) == {(d,s) for d in DATASETS for s in SEEDS}
    protocol = read(ROOT/'training-protocol.json')
    data = read(ROOT/'data-fingerprints.json')['datasets']
    provenance = read(ROOT/'evidence-provenance.json')['original_artifact_sha256']
    for (dataset, seed), run in by_run.items():
        assert run['id'] == dataset+'-'+str(seed) and run['variant'] == 'baseline'
        assert {k:v for k,v in run['config'].items() if k != 'seed'} == protocol['configs'][dataset]
        assert run['config']['seed'] == seed
        assert run['test'] == run['reloaded']['test']
        assert run['best_validation'] == run['reloaded']['validation']
        assert run['best_epoch'] == run['reloaded']['epoch']
        assert run['data_fingerprint'] == run['reloaded']['data_fingerprint'] == data[dataset]['fingerprint']
        assert set(run['test']) == set(METRICS)
        assert all(0 <= x <= 1 for x in run['test'].values())
        history = ROOT/run['history']
        assert sha(history) == provenance[run['id']]['history.csv']
        with history.open(newline='') as stream:
            epochs = list(csv.DictReader(stream))
        assert len(epochs) == run['epochs_completed']
        best = max(epochs, key=lambda e:float(e['valid_recall@20']))
        assert int(best['epoch']) == run['best_epoch']
        assert run['epochs_completed'] == min(run['best_epoch']+21, 1000)
        for m in METRICS:
            close(best['valid_'+m], run['best_validation'][m])
    summary = read(ROOT/'summary.json')
    assert summary['complete'] and summary['seeds'] == list(SEEDS)
    rows = summary['metrics']
    assert len(rows) == 20
    assert {(r['dataset'],r['metric']) for r in rows} == {(d,m) for d in DATASETS for m in METRICS}
    for row in rows:
        values = [by_run[row['dataset'],s]['test'][row['metric']] for s in SEEDS]
        assert row['n_runs'] == 5 and row['variant'] == 'baseline'
        close(row['mean'],mean(values))
        close(row['sample_std'],stdev(values))
    with (ROOT/'summary.csv').open(newline='') as stream:
        exported = list(csv.DictReader(stream))
    assert len(exported) == len(rows)
    for actual, expected in zip(exported,rows):
        assert actual == {k:str(v) for k,v in expected.items()}
    paper = read(ROOT/'paper-table2.json')
    assert sha(ROOT/paper['source']['pdf_path']) == paper['source']['pdf_sha256']
    assert paper['source']['column'] == 'BM3 / raw'
    with (ROOT/'five-seeds-vs-paper.csv').open(newline='') as stream:
        compared = list(csv.DictReader(stream))
    assert len(compared) == 20
    for actual, expected in zip(compared,rows):
        d,m = expected['dataset'],expected['metric']
        assert actual['dataset'] == d and actual['metric'] == m
        reference = paper['datasets'][d][m]
        close(actual['paper_value'],reference)
        close(actual['our_mean'],expected['mean'])
        close(actual['our_sample_std'],expected['sample_std'])
        close(actual['absolute_difference'],expected['mean']-reference)
        close(actual['relative_difference_pct'],(expected['mean']/reference-1)*100)
    for record in read(BASE/'migration_manifest.json')['files']:
        assert sha(REPO/record['path']) == record['sha256'], record['path']
    upstream = read(BASE/'runtime/upstream/manifest.json')
    for name, record in upstream['files'].items():
        assert sha(BASE/'runtime/upstream/BM3'/name) == record['sha256']
    for dataset in DATASETS:
        files = read(BASE/'runtime/configs'/(dataset+'_files.json'))
        assert all(files[name]['sha256'] == value for name,value in data[dataset]['sha256'].items())
    assert sha(BASE/'runtime/bm3baseline/model.py') == sha(BASE/'source-history/baby-20261107/bm3baseline/model.py')
    print('PASS: 25 runs, independent reloads, 20 mean/SD/paper comparisons, histories, data and source hashes')

if __name__ == '__main__':
    main()
