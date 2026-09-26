"""Recompute the published VetNutriLM figures from the saved run outputs.

Needs authorised access to the private run packages (the dataset file and the returned
results folders). Every prediction file is checked against its completion hash, then
the point estimates and grouped bootstrap intervals are recomputed and compared with
``src/vetnutrilm/published_results.json``. Any difference fails loudly.
"""
import hashlib
import json
from importlib import resources
from pathlib import Path

import numpy as np

from . import metrics


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_run(dataset, results):
    """Return (labels, groups, {'baseline': predictions, 'tuned': predictions}) in record order."""
    rows = json.loads(Path(dataset).read_text(encoding='utf-8'))['records']
    for r in rows:
        if hashlib.sha256(r['text'].encode()).hexdigest() != r['text_sha256']:
            raise ValueError(f"Text hash mismatch for record {r['Ref']}")
    by_ref = {r['Ref']: r for r in rows}
    families = {}
    for family, folders in [('baseline', ['baseline']), ('tuned', [f'fold_{f}' for f in range(1, 6)])]:
        combined = {}
        for name in folders:
            folder = Path(results) / name
            complete = json.loads((folder / 'complete.json').read_text(encoding='utf-8'))
            if _digest(folder / 'predictions.json') != complete['predictions_sha256']:
                raise ValueError(f'Prediction file changed: {folder}')
            for p in json.loads((folder / 'predictions.json').read_text(encoding='utf-8')):
                if p['Ref'] in combined or p['truth'] != by_ref[p['Ref']]['labels']:
                    raise ValueError(f"Duplicate or mismatched record {p['Ref']}")
                if family == 'tuned' and (p['invalid_or_truncated'] or p['fold'] != by_ref[p['Ref']]['fold']):
                    raise ValueError(f"Unexpected tuned prediction for record {p['Ref']}")
                combined[p['Ref']] = p['prediction']
        if set(combined) != set(by_ref):
            raise ValueError(f'{family} predictions do not cover every record')
        families[family] = np.asarray([combined[r['Ref']] for r in rows], dtype=int)
    return (np.asarray([r['labels'] for r in rows], dtype=int),
            np.asarray([r['group'] for r in rows]), families)


def binary_figures(dataset, results):
    y, groups, fam = load_run(dataset, results)
    y, tuned, base = y[:, 0], fam['tuned'][:, 0], fam['baseline'][:, 0]
    point = metrics.binary_scores(y, tuned)
    intervals = {k: metrics.grouped_interval(lambda a, b, k=k: metrics.binary_scores(a, b)[k], groups, y, tuned)
                 for k in point}
    f1 = lambda a, b: metrics.binary_counts(a, b)['f1']
    return dict(records=len(y), positives=int(y.sum()), groups=len(set(groups.tolist())), point=point,
                intervals=intervals, pretrained_f1=f1(y, base),
                f1_difference=metrics.paired_difference(f1, groups, y, tuned, base))


def category_figures(dataset, results):
    y, groups, fam = load_run(dataset, results)
    tuned = fam['tuned']
    point = metrics.category_scores(y, tuned)
    return dict(records=len(y), groups=len(set(groups.tolist())), point=point,
                macro_f1_interval=metrics.grouped_interval(lambda a, b: metrics.category_scores(a, b)['macro_f1'], groups, y, tuned),
                micro_f1_interval=metrics.grouped_interval(lambda a, b: metrics.category_scores(a, b)['micro_f1'], groups, y, tuned))


def published():
    return json.loads(resources.files('vetnutrilm').joinpath('published_results.json').read_text(encoding='utf-8'))


def compare(binary=None, category=None, places=4):
    """Return a list of (figure, published, recomputed, matches) rows."""
    expected, rows = published(), []

    def add(name, want, got):
        rows.append((name, want, got, got is not None and round(got, places) == round(want, places)))

    if binary:
        e = expected['binary']
        for k in ('f1', 'sensitivity', 'precision', 'specificity', 'accuracy'):
            add(f'binary {k}', e['point'][k], binary['point'][k])
            add(f'binary {k} lower', e['intervals'][k]['lower'], binary['intervals'][k]['lower'])
            add(f'binary {k} upper', e['intervals'][k]['upper'], binary['intervals'][k]['upper'])
        add('binary pretrained F1', e['pretrained_f1'], binary['pretrained_f1'])
        add('binary F1 difference lower', e['f1_difference']['lower'], binary['f1_difference']['lower'])
        add('binary F1 difference upper', e['f1_difference']['upper'], binary['f1_difference']['upper'])
    if category:
        e = expected['category']
        add('category macro F1', e['macro_f1'], category['point']['macro_f1'])
        add('category micro F1', e['micro_f1'], category['point']['micro_f1'])
        add('category macro F1 lower', e['macro_f1_interval']['lower'], category['macro_f1_interval']['lower'])
        add('category macro F1 upper', e['macro_f1_interval']['upper'], category['macro_f1_interval']['upper'])
        add('category micro F1 lower', e['micro_f1_interval']['lower'], category['micro_f1_interval']['lower'])
        add('category micro F1 upper', e['micro_f1_interval']['upper'], category['micro_f1_interval']['upper'])
        for name, want, got in zip(e['categories'], e['per_category_f1'], category['point']['per_category_f1']):
            add(f'category {name} F1', want, got)
    return rows
