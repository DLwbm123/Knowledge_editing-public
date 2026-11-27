"""Aggregate the matched pilot with fixed FIT denominators and image-paired intervals."""
from collections import defaultdict
import json
from pathlib import Path
import random
from statistics import median
import sys

from summarize_reference import norm


ARMS = ('MED', 'MDU', 'P1', 'P2', 'P3', 'P4')
POSITIVES = ('holdout_modal', 'holdout_context')
NEGATIVE = 'holdout_independent'


def read(path):
    return json.loads(path.read_text())


def margin(groups, arm, owners):
    return median(value for o in owners for value in groups[arm][o]['positive']) - median(
        value for o in owners for value in groups[arm][o]['negative'])


def difference(groups, left, right, owners):
    rng = random.Random(20261127)
    deltas = []
    for _ in range(2000):
        sample = rng.choices(owners, k=len(owners))
        deltas.append(margin(groups, left, sample) - margin(groups, right, sample))
    deltas.sort()
    return dict(delta=margin(groups, left, owners) - margin(groups, right, owners),
                ci95=[deltas[49], deltas[1949]])


def summarize(run, scored, stage0, roster):
    rows = [json.loads(line) for i in range(2) for line in (run / f'metrics{i}.jsonl').open()]
    assert len(rows) == 258
    anchors = read(stage0 / 'fit_denominators.private.json')
    targets = {e['edit_id']: e['source_rows'][0]['reference'] for e in read(roster)['sources'].values()}
    decisions = read(scored / 'DECISIONS.json')['decisions']
    consumers = read(scored / 'consumers.private.json')
    answers = {(r['event'], r['arm']): decisions.get(r['key'], {}).get('core_correct') for r in consumers}
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    by_role = defaultdict(list)
    owners = sorted({r['owner'] for r in rows})
    assert len(owners) == 30 and len(answers) == len(rows) * len(ARMS)
    for n, row in enumerate(rows):
        role, owner = row['role'], row['owner']
        by_role[role].append((n, row))
        for arm in ARMS:
            ratio = row['reference_prefix'][arm]['projection'] / anchors[str(owner)]
            if role in POSITIVES:
                groups[arm][owner]['positive'].append(ratio)
            if role == NEGATIVE:
                groups[arm][owner]['negative'].append(ratio)
    result = {}
    for role, block in by_role.items():
        d = {}
        for arm in ARMS:
            known = [answers[n, arm] for n, _ in block]
            prefixes = ('reference_prefix', 'base_prefix') if role == NEGATIVE else ()
            ref = [r['reference_prefix'][arm] for _, r in block]
            d[arm] = dict(
                core_correct=sum(k is True for k in known), unknown=sum(k is None for k in known),
                projection_median=median(x['projection'] for x in ref),
                fixed_fit_readout_median=median(r['reference_prefix'][arm]['projection'] /
                                                anchors[str(r['owner'])] for _, r in block),
                residual_median=median(x['residual'] for x in ref),
                answer_nll_median=median(x['nll'] for x in ref),
                at_cap=sum(r['bank_generation'][arm]['at_cap'] for _, r in block))
            for name in prefixes:
                d[arm][name + '_kl_median'] = median(r[name].get('bank_' + arm, r[name][arm])['kl']
                                                       for _, r in block)
            if role == NEGATIVE:
                d[arm]['selected_target_exact'] = sum(
                    bool(r['route']['activated']) and
                    norm(r['bank_generation'][arm]['answer']) == norm(targets[r['route']['logical_edit_id']])
                    for _, r in block)
        result[role] = dict(events=len(block), arms=d)

    comparisons = {f'{a}_minus_{b}': difference(groups, a, b, owners)
                   for a, b in (('P4', 'P2'), ('P4', 'P3'), ('P2', 'P1'))}
    delta = {a: margin(groups, a, owners) for a in ARMS}
    neg = result[NEGATIVE]['arms']
    hist = result['historical_t2l']['arms']
    locality_fields = ('reference_prefix_kl_median', 'base_prefix_kl_median',
                       'projection_median', 'selected_target_exact')
    gates = {}
    for a in ARMS[2:]:
        gates[a] = dict(
            locality=(all(neg[a][k] <= neg['MDU'][k] for k in locality_fields)
                      and hist[a]['unknown'] == hist['MDU']['unknown'] == 0
                      and hist[a]['core_correct'] >= hist['MDU']['core_correct']),
            source=all(result[r]['arms'][a]['core_correct'] == result[r]['events']
                       and result[r]['arms'][a]['projection_median'] >= result[r]['arms']['MDU']['projection_median']
                       for r in ('native', 'fit')))
    return dict(groups=result, selection_margin=delta, paired_comparisons=comparisons,
                P4_vs_P2_selection_pass=(comparisons['P4_minus_P2']['ci95'][0] > 0
                                         and result[POSITIVES[0]]['arms']['P4']['fixed_fit_readout_median'] >= .60),
                gates=gates, judge_snapshot=None)


if __name__ == '__main__':
    run, scored, stage0, roster, output = (Path(a) for a in sys.argv[1:])
    result = summarize(run, scored, stage0, roster)
    with output.open('x') as f:
        json.dump(result, f, ensure_ascii=False, allow_nan=False, indent=2)
