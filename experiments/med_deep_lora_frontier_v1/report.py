"""Aggregate frozen candidates on original support, without publishing identities."""
import json
from pathlib import Path
import statistics

ROLES = ('T0', 'FIT', 'T1G', 'T2G', 'T1L', 'T2L')
FIELDS = ('core_correct', 'full_supported')
GAMMAS = (.25, .5, .75, 1.)


def aggregate(rows):
    by = {}
    for owner, value in rows:
        if value not in (True, False, None):
            raise ValueError('Invalid judgment')
        by.setdefault(owner, []).append(value)
    unknown = sum(v is None for _, v in rows)
    def macro(replacement):
        return 100 * statistics.mean(statistics.mean(replacement if v is None else v for v in values)
                                     for values in by.values()) if by else None
    return dict(owners=len(by), events=len(rows), correct=sum(v is True for _, v in rows), unknown=unknown,
                macro_percent=macro(False) if not unknown else None,
                lower_percent=macro(False), upper_percent=macro(True))


def compare(rows):
    complete = [(a, b) for a, b in rows if a is not None and b is not None]
    return dict(events=len(rows), unknown=len(rows)-len(complete),
                gains=sum(a is False and b is True for a, b in complete),
                losses=sum(a is True and b is False for a, b in complete))


def evaluate(packet, decisions, baseline, roles, fixed_packet, fixed_decisions):
    if len(packet['consumers']) != 2032 or len(baseline) != 508 or len(roles) != 16:
        raise ValueError('Panel coverage mismatch')
    values, metadata = {}, {}
    for row in baseline:
        for arm in ('BASE', 'MED'):
            values[(row['owner'], row['query_id'], arm)] = row[arm]
    for row in fixed_packet['consumers']:
        decision = fixed_decisions.get(row['key'])
        values[(row['owner'], row['query_id'], row['arm'])] = decision
    for row in packet['consumers']:
        arm = f"LORA_{row['gamma']:.2f}"
        identity = (row['owner'], row['query_id'], arm)
        if identity in values:
            raise ValueError('Duplicate candidate')
        values[identity] = decisions.get(row['key'])
        metadata[identity] = row
    arms = ('BASE', 'MED', 'C21', 'CDEEP') + tuple(f'LORA_{g:.2f}' for g in GAMMAS)
    events = {(int(owner), q) for owner, groups in roles.items() for role in ROLES for q in groups[role]}
    if any((o, q, arm) not in values for o, q in events for arm in arms):
        raise ValueError('Original role support incomplete')
    def score(owner, qid, arm, field):
        d = values[(owner, qid, arm)]
        return None if d is None else d[field]
    panels = {}
    for role in ROLES:
        selected = [(int(owner), q) for owner, groups in roles.items() for q in groups[role]]
        panels[role] = {}
        for field in FIELDS:
            panels[role][field] = {arm: dict(**aggregate([(o, score(o, q, arm, field)) for o, q in selected]),
                vs_med=compare([(score(o, q, 'MED', field), score(o, q, arm, field)) for o, q in selected]))
                for arm in arms}
    gates = {}
    for gamma in GAMMAS:
        arm = f'LORA_{gamma:.2f}'
        g = {}
        for role in ('T0','FIT','T1G','T2G','T2L'):
            p = panels[role]['core_correct']
            candidate, med = p[arm]['macro_percent'], p['MED']['macro_percent']
            threshold = 90. if role == 'T2G' else (med-5 if role == 'T2L' and med is not None else med)
            g[role] = None if candidate is None or threshold is None else candidate >= threshold - 1e-10
        gates[arm] = g
    if any(v is None for x in gates.values() for v in x.values()):
        status, selected = 'INCONCLUSIVE', None
    else:
        qualifying = [arm for arm in gates if all(gates[arm].values())]
        if not qualifying:
            status, selected = 'DEV_FRONTIER_NOT_FOUND', None
        elif any(panels['T2G']['full_supported'][a]['macro_percent'] is None for a in qualifying):
            status, selected = 'INCONCLUSIVE', None
        else:
            selected = max(qualifying, key=lambda a: (panels['T2G']['core_correct'][a]['macro_percent'],
                panels['T2G']['full_supported'][a]['macro_percent'], panels['T2L']['core_correct'][a]['macro_percent'],
                -float(a.split('_')[1])))
            status = 'DEV_FRONTIER_SELECTED'
    generation = {a: dict(predictions=sum(x[2]==a for x in metadata),
                    eos=sum(v['eos'] for k, v in metadata.items() if k[2]==a),
                    at_cap=sum(v['at_cap'] for k, v in metadata.items() if k[2]==a),
                    empty=sum(v['empty'] for k, v in metadata.items() if k[2]==a)) for a in arms[4:]}
    return dict(status=status, selected=selected, gates=gates, panels=panels, generation=generation,
                t1l='NA', scope='exposed development panel', scorer_snapshot='unknown; user accepted comparison')
