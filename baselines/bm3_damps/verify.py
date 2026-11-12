"""Verify the portable three-seed result package using only Python's standard library."""
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'docs/results/bm3-damps'
SNAPSHOT = Path(__file__).resolve().parent / 'snapshot'


def main():
    manifest = json.loads((OUTPUT / 'packaging.json').read_text(encoding='utf-8'))
    for name, expected in manifest['snapshot_sha256'].items():
        assert hashlib.sha256((SNAPSHOT / name).read_bytes()).hexdigest() == expected, name
    for name, expected in manifest['baseline_history_sha256'].items():
        assert hashlib.sha256((OUTPUT / name).read_bytes()).hexdigest() == expected, name
    spec = importlib.util.spec_from_file_location('frozen_results', SNAPSHOT / 'baselines/bm3_damps/results.py')
    results = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(results)
    results.verify(OUTPUT)
    print('PASS: relocated snapshot and all paired baseline histories match packaging hashes')


if __name__ == '__main__':
    main()
