"""Aggregate bound decisions for the preregistered exposed development panel.

This module does not load decisions or establish their provenance. The caller
must bind all four arms and the panel before passing private records here.
"""
import statistics

ARMS = ('BASE', 'MED', 'CDEEP', 'REPLAY')
FIELDS = ('core_correct', 'full_supported', 'joint_supported')
ROLES = ('T0', 'FIT', 'T1G', 'T2G', 'T1L', 'T2L')


def score(row, arm, field):
    result = row['scores'][arm]
    core, full = result['core_correct'], result['full_supported']
    if any(v is not None and type(v) is not bool for v in (core, full)):
        raise ValueError('Decisions must be booleans or unknown')
    if field == 'joint_supported':
        return None if core is None or full is None else core and full
    return result[field]


def aggregate(rows, arm, field):
    values = [(r['owner'], score(r, arm, field)) for r in rows]
    groups = {}
    for owner, value in values:
        groups.setdefault(owner, []).append(value)
    unknown = sum(v is None for _, v in values)
    return dict(events=len(values), owners=len(groups), unknown=unknown,
                correct=sum(v is True for _, v in values),
                micro_percent=100*sum(v is True for _, v in values)/len(values) if values and not unknown else None,
                macro_percent=100*statistics.mean(statistics.mean(x) for x in groups.values()) if groups and not unknown else None)


def contrast(rows, reference, candidate, field):
    pairs = [(r['owner'], score(r, reference, field), score(r, candidate, field)) for r in rows]
    valid = [(o, a, b) for o, a, b in pairs if a is not None and b is not None]
    owners = {}
    for owner, a, b in valid:
        owners.setdefault(owner, []).append(int(b)-int(a))
    return dict(events=len(pairs), complete=len(valid), unknown=len(pairs)-len(valid),
                gains=sum(not a and b for _, a, b in valid) if valid else None,
                losses=sum(a and not b for _, a, b in valid) if valid else None,
                macro_delta_pp=100*statistics.mean(statistics.mean(x) for x in owners.values())
                if valid and len(valid)==len(pairs) else None)


def evaluate(rows, expected_events=508, expected_owners=16):
    if len(rows) != expected_events or len({r['owner'] for r in rows}) != expected_owners:
        raise ValueError('Panel coverage mismatch')
    if len({(r['owner'], r['query_id']) for r in rows}) != len(rows):
        raise ValueError('Duplicate event')
    # Unqualified events remain in the all-event FULL protection check but
    # must not enter any role-specific denominator.
    if any(r['role'] not in (*ROLES, None) or set(r['scores']) != set(ARMS) for r in rows):
        raise ValueError('Role or arm coverage mismatch')
    panels = {}
    for role in ROLES:
        group = [r for r in rows if r['role'] == role]
        panels[role] = {}
        for field in FIELDS:
            panels[role][field] = dict(arms={a:aggregate(group,a,field) for a in ARMS},
                replay_vs_deep=contrast(group,'CDEEP','REPLAY',field),
                replay_vs_med=contrast(group,'MED','REPLAY',field))
    common = [r for r in rows if r['role']=='T2L' and score(r,'BASE','core_correct') is True
              and score(r,'MED','core_correct') is True]
    deep_damage = contrast(common,'MED','CDEEP','core_correct')
    replay_damage = contrast(common,'MED','REPLAY','core_correct')
    all_full = contrast(rows,'MED','REPLAY','full_supported')
    d = panels['T2G']['core_correct']
    gdeep, gmed = d['replay_vs_deep']['macro_delta_pp'], d['replay_vs_med']['macro_delta_pp']
    native = [panels[x]['core_correct']['replay_vs_med'] for x in ('T0','FIT')]
    unknown = any(score(r,arm,field) is None for r in rows for arm in ARMS for field in FIELDS)
    complete = not unknown and len(common)==29 and deep_damage['losses']==10 and all(x['events'] for x in native)
    passed = bool(complete and gdeep is not None and gdeep>=-2 and gmed is not None and gmed>=2
                  and replay_damage['losses'] is not None and replay_damage['losses']<=5
                  and all(x['losses']==0 for x in native) and all_full['losses']==0)
    return dict(status='DEV_SCREEN_PASS' if passed else 'DEV_SCREEN_NOT_PASSED' if complete else 'INCOMPLETE',
                panels=panels, t2l_common_support=len(common), deep_t2l_damage=deep_damage,
                replay_t2l_damage=replay_damage, all_full_vs_med=all_full,
                limitations=['exposed development panel, not independent clinical validation',
                             'patient independence and medical truth unknown'])
