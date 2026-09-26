import json
import os

import pandas as pd
import pytest

from vetnutrilm import cli
from vetnutrilm.prompts import CATEGORIES


def test_demo(tmp_path):
    cli.main(['demo', '--output', str(tmp_path / 'demo')])
    summary = json.loads((tmp_path / 'demo' / 'demo.json').read_text())
    assert summary['notes'] == 12 and summary['linked_groups'] == 10  # p1 pair and identical nail-trim notes
    assert 'Not an accuracy estimate' in summary['note']


def test_demo_refuses_non_empty_output(tmp_path):
    (tmp_path / 'x').mkdir()
    (tmp_path / 'x' / 'keep.txt').write_text('')
    with pytest.raises(SystemExit):
        cli.main(['demo', '--output', str(tmp_path / 'x')])


def _category_file(path, n=40):
    rows = []
    for i in range(n):
        row = dict(id=f'r{i}', patient=f'p{i // 2}', text=f'Invented note {i}')
        row.update({c: int((i + k) % 3 == 0) for k, c in enumerate(CATEGORIES)})
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_prepare_and_evaluate_categories(tmp_path):
    records = tmp_path / 'records.csv'
    _category_file(records)
    cli.main(['prepare', '--task', 'categories', '--input', str(records), '--output', str(tmp_path / 'folds.csv')])
    allocation = pd.read_csv(tmp_path / 'folds.csv')
    assert allocation.groupby('group').fold.nunique().max() == 1
    frame = pd.read_csv(records)
    preds = frame[['id'] + CATEGORIES].copy()
    preds['categories_invalid'] = 0
    preds.to_csv(tmp_path / 'preds.csv', index=False)
    cli.main(['evaluate', '--task', 'categories', '--records', str(records),
              '--predictions', str(tmp_path / 'preds.csv'), '--output', str(tmp_path / 'eval.json')])
    result = json.loads((tmp_path / 'eval.json').read_text())
    assert result['point']['macro_f1'] == 1.0 and result['invalid_outputs'] == 0


def test_evaluate_refuses_invalid_binary(tmp_path):
    pd.DataFrame(dict(id=['a', 'b'], text=['x', 'y'], label=[1, 0])).to_csv(tmp_path / 'r.csv', index=False)
    pd.DataFrame(dict(id=['a', 'b'], binary=[1, -1])).to_csv(tmp_path / 'p.csv', index=False)
    with pytest.raises(SystemExit):
        cli.main(['evaluate', '--task', 'binary', '--records', str(tmp_path / 'r.csv'),
                  '--predictions', str(tmp_path / 'p.csv'), '--output', str(tmp_path / 'e.json')])


PRIVATE = {k: os.environ.get(f'VETNUTRILM_{k.upper()}') for k in
           ('binary_dataset', 'binary_results', 'category_dataset', 'category_results')}


@pytest.mark.skipif(not all(PRIVATE.values()), reason='Authorised run outputs not supplied')
def test_reproduce_published_figures():
    from vetnutrilm import reproduce
    rows = reproduce.compare(reproduce.binary_figures(PRIVATE['binary_dataset'], PRIVATE['binary_results']),
                             reproduce.category_figures(PRIVATE['category_dataset'], PRIVATE['category_results']))
    assert rows and all(r[3] for r in rows), [r for r in rows if not r[3]]
