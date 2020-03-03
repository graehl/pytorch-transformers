def uniqued(xs):
    s = set()
    u = []
    for x in xs:
        if x not in s:
            u.append(x)
        else:
            s.add(x)
    return u

import collections

def group_by_lc(xs):
    s = collections.defaultdict(set)
    for x in xs:
        k = x.lower()
        s[k].add(x)
    return s

def uniqued_by_first(xs):
    s = set()
    u = []
    for x in xs:
        key = x[0]
        if key not in s:
            u.append(x)
        else:
            s.add(key)
    return u


def arg_max(logits):
    y = logits[0]
    besti = 0
    for i, x in enumerate(logits):
        if x >= y:
            y = x
            besti = i
    return besti


def confidence_in(logits, chose):
    assert isinstance(logits, list)
    assert isinstance(chose, int)
    y = logits[chose]
    logits[chose] = -999999
    conf = y - max(logits)
    logits[chose] = y
    return conf
