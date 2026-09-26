import hashlib

import numpy as np
import pytest

from vetnutrilm import folds, grouping, metrics, prompts
from vetnutrilm.parsing import parse_binary, parse_categories


def test_prompts_unchanged():
    assert hashlib.sha256(prompts.BINARY_PROMPT.encode()).hexdigest() == prompts.BINARY_PROMPT_SHA256
    assert hashlib.sha256(prompts.CATEGORY_PROMPT.encode()).hexdigest() == prompts.CATEGORY_PROMPT_SHA256
    assert prompts.BINARY_PROMPT_SHA256 == '4c4c5e093fcab84331d882d0aaa65c0806644ddab467bd01d657f8897925e9cb'
    assert prompts.CATEGORY_PROMPT_SHA256 == '33e22864de0b98b15ac763c705075a61edb551e70b64f18205bff2b18b55bc46'
    assert prompts.TRAINING == {'rank': 8, 'alpha': 16, 'dropout': 0.05, 'learning_rate': 0.0001,
                                'weight_decay': 0.01, 'epochs': 1, 'accumulation': 8, 'seed': 42,
                                'targets': r'.*language_model.*\.(q_proj|k_proj|v_proj|o_proj)$'}
    assert len(prompts.CATEGORIES) == 6


@pytest.mark.parametrize('raw,value,invalid', [
    ('[1]', [1], False), (' [0] ', [0], False), ('```json\n[1]\n```', [-1], True),
    ('Yes', [-1], True), ('[2]', [-1], True), ('[1,0]', [-1], True), ('[true]', [-1], True)])
def test_parse_binary(raw, value, invalid):
    assert parse_binary(raw) == (value, invalid)


def test_parse_categories():
    assert parse_categories('[1,0,0,0,0,1]') == ([1, 0, 0, 0, 0, 1], False)
    assert parse_categories('[1,0]') == ([0] * 6, True)
    assert parse_categories('```\n[1,0,0,0,0,1]\n```') == ([0] * 6, True)


def test_grouping_links_patients_and_identical_text():
    ids = ['a', 'b', 'c', 'd', 'e']
    patients = ['p1', 'p1', 'p2', 'p3', '']
    texts = ['one', 'two', 'Nail  trim', 'nail trim', 'nail trim.']
    g = grouping.connected_groups(ids, patients, texts)
    assert g['a'] == g['b'] == 'a'
    assert g['c'] == g['d'] == 'c'       # case and whitespace ignored
    assert g['e'] == 'e'                 # punctuation kept
    with pytest.raises(ValueError):
        grouping.connected_groups(['a', 'a'], ['', ''], ['x', 'y'])


def test_binary_folds_keep_groups_together():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(60), 2)
    labels = rng.integers(0, 2, len(groups))
    f = folds.binary_folds(labels, groups)
    assert set(f) == {1, 2, 3, 4, 5}
    for g in set(groups):
        assert len(set(f[groups == g])) == 1


def test_category_folds_balance_and_groups():
    rng = np.random.default_rng(1)
    groups = np.repeat(np.arange(80), 2)
    y = (rng.random((len(groups), 6)) < 0.3).astype(int)
    f = folds.category_folds(y, groups)
    assert set(f) == {1, 2, 3, 4, 5}
    for g in set(groups):
        assert len(set(f[groups == g])) == 1
    for k in range(1, 6):
        assert (y[f == k].sum(axis=0) > 0).all()


def test_metrics_known_values():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 1])
    p = np.array([1, 1, 0, 0, 0, 1, 0, 1])
    s = metrics.binary_scores(y, p)
    assert s['sensitivity'] == 0.75 and s['precision'] == 0.75 and s['specificity'] == 0.75
    assert s['f1'] == 0.75 and s['accuracy'] == 0.75
    c = metrics.category_scores(np.tile(y, (6, 1)).T, np.tile(p, (6, 1)).T)
    assert c['macro_f1'] == 0.75 and c['micro_f1'] == 0.75


def test_bootstrap_is_seeded_and_grouped():
    y = np.array([1, 0, 1, 0, 1, 0, 1, 1, 0, 0])
    p = np.array([1, 0, 0, 0, 1, 1, 1, 1, 0, 0])
    groups = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    f1 = lambda a, b: metrics.binary_counts(a, b)['f1']
    first = metrics.grouped_interval(f1, groups, y, p, replicates=200)
    assert first == metrics.grouped_interval(f1, groups, y, p, replicates=200)
    assert first['lower'] <= f1(y, p) <= first['upper']
    for sample in metrics.group_samples(groups, replicates=20):
        assert len(sample) % 2 == 0      # whole groups of two


def test_published_results_packaged():
    from vetnutrilm.reproduce import published
    r = published()
    assert round(r['binary']['point']['f1'], 4) == 0.9406
    assert round(r['category']['macro_f1'], 4) == 0.9267
    assert r['category']['categories'] == prompts.CATEGORIES
