"""Read and validate record files.

A record file is a CSV with columns:
- ``id``: unique record identifier;
- ``text``: the full clinical note (never truncated or cleaned);
- ``patient`` (optional): a patient key, used only to keep a dog's records together;
- binary labels: ``label`` with values 0 or 1;
- category labels: one 0/1 column per category, named as in ``CATEGORIES``.
"""
import pandas as pd

from .prompts import CATEGORIES


def read_records(path, task=None):
    frame = pd.read_csv(path, dtype={'id': str, 'patient': str}, keep_default_na=False)
    for column in ('id', 'text'):
        if column not in frame:
            raise ValueError(f'Missing column: {column}')
    if not frame.id.is_unique:
        raise ValueError('Record ids must be unique')
    if 'patient' not in frame:
        frame['patient'] = ''
    if task == 'binary':
        frame['label'] = _binary(frame, 'label')
    elif task == 'categories':
        for c in CATEGORIES:
            frame[c] = _binary(frame, c)
    return frame


def labels(frame, task):
    return frame[['label']].to_numpy() if task == 'binary' else frame[CATEGORIES].to_numpy()


def _binary(frame, column):
    if column not in frame:
        raise ValueError(f'Missing label column: {column}')
    values = pd.to_numeric(frame[column], errors='coerce')
    if not values.isin([0, 1]).all():
        raise ValueError(f'Column {column} must contain only 0 and 1')
    return values.astype(int)
