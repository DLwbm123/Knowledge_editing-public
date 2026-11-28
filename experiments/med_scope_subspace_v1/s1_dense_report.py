"""Apply the prospective S1′ endpoints to same-round paired outcomes."""
from collections import Counter
from pathlib import Path
import statistics
import sys

from .evaluate import routing_report
from .experiment import frozen_core, json_read, json_write


REQUIRED = dict(official_ind=29, hold_vis=15, hold_ind=15, native=30, fit=120,
                modal_record_holdout=30, context_record_holdout=30)


def judge(arm, report, invalid):
    groups = report['groups'][arm]
    paired = report['paired']

    def limit(role, field, *, lower=False):
        result = paired[f'{arm} vs P4 – {role}']['effects'][field]['ci95']
        if result is None:
            return -float('inf') if lower else float('inf')
        return result[0 if lower else 1]

    valid = lambda role: not (invalid[arm,role] or invalid['P4',role])
    return dict(
        official_noninferior=valid('official_ind') and
            paired[f'{arm} vs P4 – official_ind']['unknown']==0 and
            limit('official_ind','core',lower=True)>=-.10,
        visual_superior=valid('hold_vis') and limit('hold_vis','exact')<0 and limit('hold_vis','base_kl')<0,
        independent_noninferior=valid('hold_ind') and limit('hold_ind','exact')<=.05 and
            limit('hold_ind','base_kl')<=.001,
        source_preserved=all(valid(role) and groups[role]['core_correct']==REQUIRED[role] and
            groups[role]['medians'].get('answer_nll',float('inf'))<=.05 for role in ('native','fit')),
    )


def report(rows, decisions, arms):
    for row in rows:
        if row['role']=='hold_pos':
            row['role'] = row['family']
        row['core'] = decisions.get(row['score_id'])
    for arm in (*arms,'P4'):
        counts = Counter(r['role'] for r in rows if r['route']==arm)
        if counts!=REQUIRED:
            raise ValueError('S1 primary or parallel held family support changed')
    invalid = Counter()
    for r in rows:
        role = r['role']
        fields = ('base_kl',) if role in {'hold_ind','hold_vis'} else ('answer_nll',)
        if r['raw']['at_cap'] or any(k not in r['metric'] for k in fields):
            invalid[r['route'],role] += 1
        elif role in {'hold_ind','hold_vis'} and r['exact'] is None:
            invalid[r['route'],role] += 1
        elif role in {'official_ind','native','fit'} and r['core'] is None:
            invalid[r['route'],role] += 1
    controls = [('V-SGD-dense','V-U-ft'),('V-SGD-TT','V-SGD-dense'),('V-U-ft','V-U')]
    scored = routing_report(rows, [('P4',arm) for arm in arms]+controls)
    for arm in (*arms,'P4'):
        for role in ('native','fit'):
            values = [r['metric']['answer_nll'] for r in rows if r['route']==arm and r['role']==role
                      and 'answer_nll' in r['metric']]
            scored['groups'][arm][role]['p95_answer_nll'] = (
                statistics.quantiles(values,n=20,method='inclusive')[-1] if len(values)>1 else None)
    gates = {a:judge(a,scored,invalid) for a in ('V-U','V-U-ft') if a in arms}
    return dict(scored,gate=gates,s2_admitted='V-U' in gates and all(gates['V-U'].values()),
                invalid={f'{arm} {role}':n for (arm,role),n in invalid.items()})


def main():
    decisions_path, output = map(Path, sys.argv[1:])
    arms = json_read('selection.json')['training_arms']
    rows = [r for lane in (0,1) for r in json_read(f'behavior{lane}.json')['rows']]
    if len({(r['id'],r['route']) for r in rows})!=len(rows):
        raise ValueError('Evaluation shards overlap')
    json_write(output, report(rows,frozen_core(decisions_path),arms))


if __name__=='__main__':
    main()
