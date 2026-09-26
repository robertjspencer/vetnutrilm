"""Performance measures and grouped bootstrap intervals.

Measures are calculated from predictions pooled across held-out folds. Intervals
resample whole linked groups (2,000 replicates, seed 42), are conditional on the
fitted models and folds, and are not independent validation.
"""
import numpy as np

REPLICATES, SEED = 2000, 42


def binary_counts(y, predicted):
    y, predicted = np.asarray(y, bool), np.asarray(predicted, bool)
    tp = int((y & predicted).sum()); fp = int((~y & predicted).sum())
    fn = int((y & ~predicted).sum()); tn = int((~y & ~predicted).sum())
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                precision=tp / (tp + fp) if tp + fp else None,
                recall=tp / (tp + fn) if tp + fn else None,
                f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None)


def binary_scores(y, predicted):
    """F1, sensitivity, precision, specificity and accuracy for one 0/1 output."""
    c = binary_counts(y, predicted)
    return dict(f1=c['f1'], sensitivity=c['recall'], precision=c['precision'],
                specificity=c['tn'] / (c['tn'] + c['fp']) if c['tn'] + c['fp'] else None,
                accuracy=(c['tp'] + c['tn']) / len(np.asarray(y)))


def category_scores(y, predicted):
    """Per-category F1, macro F1 (None if any category is undefined) and micro F1."""
    y, predicted = np.asarray(y, int), np.asarray(predicted, int)
    per = [binary_counts(y[:, c], predicted[:, c])['f1'] for c in range(y.shape[1])]
    return dict(per_category_f1=per,
                macro_f1=float(np.mean(per)) if all(v is not None for v in per) else None,
                micro_f1=binary_counts(y.ravel(), predicted.ravel())['f1'])


def group_samples(groups, replicates=REPLICATES, seed=SEED):
    """Yield row indices for each bootstrap replicate, drawing whole groups with replacement."""
    groups = np.asarray(groups)
    keys = sorted(set(groups.tolist()))
    members = [np.flatnonzero(groups == k) for k in keys]
    rng = np.random.default_rng(seed)
    for _ in range(replicates):
        yield np.concatenate([members[i] for i in rng.integers(0, len(keys), len(keys))])


def percentile(values, level=0.95):
    """Percentile interval over defined replicates; withheld if under 95% are defined."""
    defined = [v for v in values if v is not None]
    tail = 100 * (1 - level) / 2
    publish = len(defined) >= 0.95 * len(values)
    return dict(defined=len(defined),
                lower=float(np.percentile(defined, tail)) if publish else None,
                upper=float(np.percentile(defined, 100 - tail)) if publish else None)


def grouped_interval(statistic, groups, *arrays, replicates=REPLICATES, seed=SEED):
    """Interval for ``statistic(*arrays[sample])`` over grouped bootstrap samples."""
    arrays = [np.asarray(a) for a in arrays]
    return percentile([statistic(*(a[s] for a in arrays)) for s in group_samples(groups, replicates, seed)])


def paired_difference(statistic, groups, y, first, second, replicates=REPLICATES, seed=SEED):
    """Interval for statistic(y, first) minus statistic(y, second) on the same samples."""
    y, first, second = np.asarray(y), np.asarray(first), np.asarray(second)
    return percentile([statistic(y[s], first[s]) - statistic(y[s], second[s])
                       for s in group_samples(groups, replicates, seed)])
