"""Grouped fold allocation for the two tasks.

Binary: stratified group k-fold (scikit-learn), seed 42, as used for the published
binary folds.

Categories: a greedy group allocation that balances record counts, the positives and
negatives of each of the six categories, and any sampling-source counts. Rarer groups
are placed first; 32 seeded restarts keep the allocation with the fewest empty
positive/negative cells, then the smallest normalised squared deviation. No model
outputs are used.
"""
import numpy as np


def binary_folds(labels, groups, k=5, seed=42):
    """Return a fold number (1..k) per record."""
    from sklearn.model_selection import StratifiedGroupKFold
    labels, groups = np.asarray(labels), np.asarray(groups)
    folds = np.zeros(len(labels), dtype=int)
    for fold, (_, test) in enumerate(StratifiedGroupKFold(k, shuffle=True, random_state=seed)
                                     .split(np.zeros(len(labels)), labels, groups), 1):
        folds[test] = fold
    _check(folds, groups, k)
    return folds


def category_folds(labels, groups, sources=None, k=5, seed=42, restarts=32):
    """Return a fold number (1..k) per record for six-column 0/1 ``labels``."""
    y = np.asarray(labels, dtype=float)
    groups = np.asarray(groups)
    if y.ndim != 2 or y.shape[1] != 6:
        raise ValueError('Expected six label columns')
    parts = [np.ones((len(y), 1)), y, 1 - y]
    if sources is not None:
        sources = np.asarray(sources)
        parts.append(np.stack([(sources == s).astype(float) for s in sorted(set(sources.tolist()))], axis=1))
    features = np.concatenate(parts, axis=1)
    keys = sorted(set(groups.tolist()))
    index = {g: n for n, g in enumerate(keys)}
    values = np.zeros((len(keys), features.shape[1]))
    np.add.at(values, [index[g] for g in groups], features)
    target = values.sum(axis=0) / k
    scale = np.maximum(target, 1)
    rarity = (values[:, 1:7] / np.maximum(values[:, 1:7].sum(axis=0), 1)).sum(axis=1)
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(restarts):
        order = sorted(range(len(values)), key=lambda i: (-rarity[i], -values[i, 0], rng.random()))
        totals = np.zeros((k, values.shape[1]))
        assigned = np.zeros(len(values), dtype=int)
        for i in order:
            choices = []
            for f in rng.permutation(k):
                trial = totals.copy()
                trial[f] += values[i]
                choices.append((float(np.square((trial - target) / scale).sum()), int(f)))
            f = min(choices, key=lambda item: item[0])[1]
            totals[f] += values[i]
            assigned[i] = f + 1
        score = (int((totals[:, 1:13] == 0).sum()), float(np.square((totals - target) / scale).sum()))
        if best is None or score < best[0]:
            best = (score, assigned.copy())
    folds = np.array([best[1][index[g]] for g in groups])
    _check(folds, groups, k)
    return folds


def _check(folds, groups, k):
    if set(folds.tolist()) != set(range(1, k + 1)):
        raise ValueError('Every fold must receive records')
    for f in range(1, k + 1):
        if set(groups[folds == f].tolist()) & set(groups[folds != f].tolist()):
            raise ValueError('A linked group was split across folds')
