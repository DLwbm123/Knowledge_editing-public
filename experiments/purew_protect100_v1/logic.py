"""Fixed source/protection clocks and exact shared-missing transition bounds."""
from collections import Counter
from itertools import product


def clocks(owner, step):
    assert 1 <= owner <= 100 and 1 <= step <= 160
    return dict(history=(step - 1) % (owner - 1) if owner > 1 else None,
                protection=((owner - 1) * 160 + step - 1) % 61)


def resume_check(state, units):
    assert sorted(units) == list(range(1, len(units) + 1)) and all(v == 160 for v in units.values())
    if state is None:
        assert not units
        return
    i, s = state['owner'], state['step']
    assert 1 <= i <= 100 and 1 <= s <= 160 and state['committed'] == 160 * (i - 1) + s
    assert i in (len(units), len(units) + 1)
    if i == len(units): assert s == 160
    assert state['clocks'] == clocks(i, s)


def transitions(pairs, scores):
    # Payload keys from different query IDs cannot share a component; enumerate components, not all missing globally.
    unknown = {k for pair in pairs for k in pair if scores[k] is None}
    adjacency = {k: set() for k in unknown}
    for a, b in pairs:
        if a in unknown and b in unknown:
            adjacency[a].add(b); adjacency[b].add(a)
    components, remaining = [], set(unknown)
    while remaining:
        component, todo = set(), [remaining.pop()]
        while todo:
            k = todo.pop()
            if k in component: continue
            component.add(k); todo.extend(adjacency[k] - component)
        remaining -= component; components.append(component)
    known = Counter(str(scores[a]) + str(scores[b]) for a, b in pairs if scores[a] is not None and scores[b] is not None)
    bounds = {k: [known[k], known[k]] for k in ('11', '10', '01', '00')}
    for component in components:
        assert len(component) <= 20, 'Unexpected cross-query binding component; audit instead of approximate bounds'
        rows = [pair for pair in pairs if component.intersection(pair)]
        keys = sorted(component); values = []
        for bits in product((0, 1), repeat=len(keys)):
            assigned = dict(zip(keys, bits))
            values.append(Counter(str(scores[a] if scores[a] is not None else assigned[a]) +
                                  str(scores[b] if scores[b] is not None else assigned[b]) for a, b in rows))
        for key in bounds:
            bounds[key][0] += min(x[key] for x in values)
            bounds[key][1] += max(x[key] for x in values)
    return dict(pairs=len(pairs), shared_unknown_payloads=len(unknown), known_transitions=dict(known),
                unresolved_pairs=sum(scores[a] is None or scores[b] is None for a, b in pairs), count_bounds=bounds)


def selfcheck():
    assert clocks(100, 160) == dict(history=60, protection=17)
    resume_check(dict(owner=2, step=4, committed=164, clocks=clocks(2, 4)), {1: 160})
    try: resume_check(dict(owner=2, step=4, committed=165, clocks=clocks(2, 4)), {1: 160})
    except AssertionError: pass
    else: raise AssertionError('Broken source clock accepted')
    out = transitions([('m', 'm'), ('m', 'z'), ('o', 'm')], dict(m=None, z=0, o=1))
    assert out['count_bounds'] == {'11': [0, 2], '10': [1, 1], '01': [0, 0], '00': [0, 2]}
    scores = {str(i): None for i in range(200)}
    assert transitions([(str(i), str(i)) for i in range(200)], scores)['count_bounds']['10'] == [0, 0]
    return dict(status='PASS', exact_component_missing=True, corrupt_clock_rejected=True)
