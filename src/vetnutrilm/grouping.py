"""Link records that must stay together when splitting data.

Two records are linked if they share a patient, or if their note text is identical
after ignoring case and repeated whitespace (punctuation is kept). Links are
transitive, so the result is a set of connected groups. Keeping each group within
one fold stops a model being tested on a dog, or a copied note, it trained on.
"""
import re


def normalise_text(text):
    return re.sub(r'\s+', ' ', str(text or '').strip()).casefold()


def connected_groups(ids, patients, texts):
    """Return {record id: group id}, where the group id is the first record id in the group.

    ``ids``, ``patients`` and ``texts`` are parallel sequences. Empty patients or
    empty texts create no link.
    """
    ids = list(ids)
    if len(set(ids)) != len(ids):
        raise ValueError('Record ids must be unique')
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for values in (list(patients), [normalise_text(t) for t in texts]):
        if len(values) != len(ids):
            raise ValueError('Parallel inputs differ in length')
        seen = {}
        for i, value in zip(ids, values):
            if value in (None, ''):
                continue
            if value in seen:
                parent[find(i)] = find(seen[value])
            else:
                seen[value] = i
    first = {}
    for i in ids:
        first.setdefault(find(i), i)
    return {i: first[find(i)] for i in ids}
