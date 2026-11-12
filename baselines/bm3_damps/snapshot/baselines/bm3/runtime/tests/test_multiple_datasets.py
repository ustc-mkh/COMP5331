import json
import numpy as np
import pandas as pd
import pytest

from bm3baseline.config import load_config, write_json
from bm3baseline.data import inspect_data
from bm3baseline.engine import train, evaluate_checkpoint
from bm3baseline.report import aggregate, TARGET


@pytest.mark.parametrize('dataset', ['sports', 'clothing', 'elec', 'microlens'])
def test_released_dataset_names_use_matching_files(dataset, tmp_path):
    data = tmp_path / dataset
    data.mkdir()
    rows = [(u, (u + k) % 32, 0 if k < 5 else (1 if k == 5 else 2))
            for u in range(32) for k in range(7)]
    pd.DataFrame(rows, columns=['userID', 'itemID', 'x_label']).to_csv(data/(dataset+'.inter'), sep='\t', index=False)
    rng = np.random.RandomState(17)
    for name in ['image_feat.npy', 'text_feat.npy']:
        np.save(data/name, rng.randn(32, 8).astype('float32'))
    config = load_config(overrides={'dataset':dataset, 'inter_file_name':dataset+'.inter',
                                    'embedding_size':4, 'feat_embed_dim':4,
                                    'epochs':1, 'threads':1, 'train_batch_size':32, 'eval_batch_size':16})
    report, splits = inspect_data(data, dataset=dataset, require_baby=False)
    assert dataset+'.inter' in report['sha256'] and len(splits) == 3
    output = tmp_path/'run'
    result = train(config, data, output, require_baby=False)
    assert result['kind'] == 'smoke' and result['config']['dataset'] == dataset
    assert evaluate_checkpoint(output/'best.pt', data, require_baby=False)['test'] == result['test']


def test_nonbaby_aggregation_does_not_apply_baby_paper_target(tmp_path):
    config=load_config(overrides={'dataset':'sports', 'inter_file_name':'sports.inter'})
    record={'kind':'formal','config':config,'seed':999,'test':TARGET,
            'data_fingerprint':'fixture','source_commit':'fixture'}
    write_json(tmp_path/'runs'/'seed-999'/'result.json',record)
    rows=aggregate(tmp_path/'runs',tmp_path/'summary')
    assert all(row['paper_target'] is None and row['relative_gap_percent'] is None for row in rows)


def test_mismatched_dataset_filename_rejected():
    with pytest.raises(ValueError, match='filename'):
        load_config(overrides={'dataset':'sports'})
