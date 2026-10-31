"""Aggregate bound decisions for the preregistered exposed development panel.

This module does not load decisions or establish their provenance. The caller
must bind all four arms and the panel before passing private records here.
"""
import statistics

ARMS = ('BASE', 'MED', 'CDEEP', 'SOURCE_TEACHER')
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
                teacher_vs_deep=contrast(group,'CDEEP','SOURCE_TEACHER',field),
                teacher_vs_med=contrast(group,'MED','SOURCE_TEACHER',field))
    common = [r for r in rows if r['role']=='T2L' and score(r,'BASE','core_correct') is True
              and score(r,'MED','core_correct') is True]
    deep_damage = contrast(common,'MED','CDEEP','core_correct')
    teacher_damage = contrast(common,'MED','SOURCE_TEACHER','core_correct')
    all_full = contrast(rows,'MED','SOURCE_TEACHER','full_supported')
    comparisons = {role: panels[role]['core_correct']['teacher_vs_med'] for role in ROLES}
    def threshold(role, minimum):
        item = comparisons[role]
        value = item['macro_delta_pp']
        if not item['events'] or item['unknown'] or value is None:
            return None
        return value >= minimum
    gates = {'T2G': threshold('T2G', 2.0), 'T2L': threshold('T2L', -5.0)}
    gates.update({role: threshold(role, 0.0) for role in ('T0','FIT','T1G','T1L')})
    supported = [value for value in gates.values() if value is not None]
    # A missing support cannot be reported as a pass, even if other gates pass.
    local_status = 'FAIL' if False in supported else 'PASS' if supported else 'UNKNOWN'
    combined = 'FAIL' if False in supported else 'UNKNOWN' if None in gates.values() else 'PASS'
    return dict(status=combined, supported_gate_status=local_status, gates=gates,
                panels=panels, t2l_common_support=len(common), deep_t2l_damage=deep_damage,
                teacher_t2l_damage=teacher_damage, all_full_vs_med=all_full,
                limitations=['exposed development panel, not independent clinical validation',
                             'patient independence and medical truth unknown'])
