"""Aggregate only the previously locked routed events, including unknowns."""
import statistics

ROLES = ('T0', 'FIT', 'T1G', 'T2G', 'T1L', 'T2L')
FIELDS = ('core_correct', 'full_supported')


def summarize(rows):
    grouped = {}
    for owner, value in rows:
        if value is not None and type(value) is not bool:
            raise ValueError('invalid score')
        grouped.setdefault(owner, []).append(value)
    if not grouped:
        return {'events': 0, 'edits': 0, 'unknown': 0, 'bounds': None}
    return {'events': len(rows), 'edits': len(grouped), 'unknown': sum(v is None for _, v in rows),
            'bounds': [100 * statistics.mean(sum(v is True for v in vs) / len(vs) for vs in grouped.values()),
                       100 * statistics.mean(sum(v is not False for v in vs) / len(vs) for vs in grouped.values())]}


def compare(consumers, decisions, roles):
    if roles.get('status') != 'EVALUATION_ROLES_FROZEN' or roles.get('owners') != 150:
        raise ValueError('role lock missing')
    pairs = {}
    for row in consumers:
        key = (row['arm'], row['query_id'])
        if key in pairs:
            raise ValueError('duplicate arm/query')
        pairs[key] = row['key']
    arms = ('MED', 'D', 'DU', 'CE', 'UNIFORM')
    query_sets = [{q for arm, q in pairs if arm == a} for a in arms]
    if any(len(q) != 392 for q in query_sets) or any(q != query_sets[0] for q in query_sets[1:]):
        raise ValueError('arms do not share 392 identities')
    results = {}
    for role in ROLES:
        entries = [(owner, q) for owner in range(1, 151)
                   for q in roles['roles'][str(owner)][role] if q in query_sets[0]]
        results[role] = {}
        for field in FIELDS:
            values = {a: [(o, decisions.get(pairs[a, q], {}).get(field)) for o, q in entries] for a in arms}
            comparisons = {}
            for left, right in (('UNIFORM', 'DU'), ('CE', 'UNIFORM'), ('CE', 'DU'), ('MED', 'CE'), ('MED', 'UNIFORM')):
                gain = loss = unknown = 0
                for (_, old), (_, new) in zip(values[left], values[right]):
                    if old is None or new is None:
                        unknown += 1
                    else:
                        gain += old is False and new is True
                        loss += old is True and new is False
                comparisons[left + '_to_' + right] = {'gain': gain, 'loss': loss, 'unknown': unknown}
            results[role][field] = {'arms': {a: summarize(rows) for a, rows in values.items()},
                                    'pairs': comparisons}
    gates = {}
    for arm in ('CE', 'UNIFORM'):
        gates[arm] = {}
        for role in ROLES:
            panel = results[role]['core_correct']['arms']
            baseline, candidate = panel['MED']['bounds'], panel[arm]['bounds']
            if baseline is None or candidate is None:
                verdict = None
            elif role == 'T2G':
                verdict = True if candidate[0] > baseline[1] else (False if candidate[1] <= baseline[0] else None)
            else:
                allowed = 5 if role == 'T2L' else 0
                verdict = True if candidate[0] >= baseline[1] - allowed else (False if candidate[1] < baseline[0] - allowed else None)
            gates[arm][role] = verdict
    return {'status': 'PAIRED_DEV_SUMMARY', 'events': 392, 'panels': results, 'gates': gates}
