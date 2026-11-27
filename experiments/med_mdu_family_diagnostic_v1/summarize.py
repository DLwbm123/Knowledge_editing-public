"""Summarize frozen paired metrics without patient text or model outputs."""
import json
import math
from pathlib import Path
import random
import statistics
import sys
from collections import Counter, defaultdict


def quantile(values, fraction):
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    low = math.floor(index)
    high = math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def describe(values):
    return {'n': len(values), 'median': statistics.median(values),
            'q25': quantile(values, .25), 'q75': quantile(values, .75)} if values else {'n': 0}


def paired_ci(rows, evaluate, group='owner'):
    by_edit = defaultdict(list)
    for row in rows:
        by_edit[row[group]].append(evaluate(row))
    values = [statistics.median(numbers) for numbers in by_edit.values()]
    if not values:
        return {'n': 0}
    seed = random.Random(20261127)
    boots = [statistics.median(seed.choices(values, k=len(values))) for _ in range(2000)]
    return {**describe(values), 'ci95_lo': quantile(boots, .025), 'ci95_hi': quantile(boots, .975)}


def by_status(row):
    previous = row['status']['MED']
    terminal = row['status']['MDU150']
    if terminal['behavior'] == 'A':
        return 'MDU_refusal'
    if previous['behavior'] == 'A' and terminal['core']:
        return 'repaired_MED_refusal'
    if previous['core'] and terminal['core']:
        return 'both_correct'
    if terminal['core']:
        return 'other_repair'
    return 'other'


def main(root, previous=None):
    with (root / 'metrics.jsonl').open() as stream:
        rows = [json.loads(line) for line in stream]
    done = json.loads((root / 'done.json').read_text())
    assert done == {'events': 1375, 'forwards': 4125, 'training': 0, 'generation': 0, 'score': 0}
    assert len(rows) == done['events'] == len({(r['owner'], r['query']) for r in rows})
    assert Counter(r['role'] for r in rows) == {'FIT':600, 'NATIVE':150, 'T2G':571, 'T2L':54}
    fit = defaultdict(list)
    for r in rows:
        if r['role'] == 'FIT':
            fit[r['owner'], r['selected']].append(r)
    groups = defaultdict(list)
    for r in rows:
        groups[r['variant']].append(r)
        if r['variant'] == 'negation_uncertainty' and r['audit'] == 'V':
            groups['indirect_valid'].append(r)
            groups['indirect_valid_' + by_status(r)].append(r)
    metrics = {}
    for group, data in groups.items():
        metrics[group] = {
            'events': len(data),
            'edits': len({r['owner'] for r in data}),
            'same_owner_route': sum(r['routed_to_owner'] for r in data),
            'med_to_mdu_nll_improvement': paired_ci(data, lambda r:r['arms']['MED']['nll']-r['arms']['MDU']['nll']),
            'med_to_mdu_first_token_improvement': paired_ci(
                data, lambda r:r['arms']['MED']['first_nll']-r['arms']['MDU']['first_nll']),
        }
        for arm in ('MED','MDU'):
            m = metrics[group][arm] = {}
            for name in ('projection','residual','kl','first_nll','nll'):
                m[name] = describe([r['arms'][arm][name] for r in data])
            if group not in ('NATIVE','FIT','T2L'):
                subset = [r for r in data if (r['owner'],r['selected']) in fit]
                for name in ('projection','residual'):
                    m[f'{name}_over_fit'] = paired_ci(subset, lambda r:
                        r['arms'][arm][name] / statistics.median(
                            f['arms'][arm][name] for f in fit[r['owner'],r['selected']]))
    locality = groups['T2L']
    metrics['locality_kl_change'] = {
        'events':len(locality),
        'increased':sum(r['arms']['MDU']['kl'] > r['arms']['MED']['kl'] for r in locality),
        'mdu_minus_med_by_selected_expert':paired_ci(
            locality, lambda r:r['arms']['MDU']['kl'] - r['arms']['MED']['kl'], group='selected'),
    }
    if previous is not None:
        selected_owner = {r['selected']:r['owner'] for r in rows
                          if r['role']=='NATIVE' and r['routed_to_owner']}
        risk = {r['owner']:r['risk'] for r in json.loads(previous.read_text())['rows'] if r['arm']=='MDU'}
        per_expert = defaultdict(list)
        for event in groups['T2L']:
            owner = selected_owner.get(event['selected'])
            if owner in risk:
                per_expert[owner].append(event['arms']['MDU']['kl'] - event['arms']['MED']['kl'])
        x = [sum(v[1] for v in risk[owner].values())/sum(v[0] for v in risk[owner].values())
             for owner in per_expert]
        y = [statistics.mean(values) for values in per_expert.values()]
        def ranks(values):
            return [1+sum(t<v for t in values)+(sum(t==v for t in values)-1)/2 for v in values]
        metrics['locality_correlation'] = {
            'experts':len(x), 'events':sum(map(len,per_expert.values())),
            'spearman':statistics.correlation(ranks(x),ranks(y)),
        }
    print(json.dumps({'counts':done,'metrics':metrics},indent=2,allow_nan=False))


if __name__ == '__main__':
    main(Path(sys.argv[1]), Path(sys.argv[2]) if len(sys.argv)>2 else None)
