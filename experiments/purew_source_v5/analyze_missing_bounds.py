"""Anonymous review of frozen consumer bindings; no generation or Judge calls."""
import itertools
import json
import os
from collections import defaultdict
from pathlib import Path


def trace(values):
    seen = False
    forgotten = 0
    for value in values:
        if seen and not value:
            forgotten += 1
        seen |= bool(value)
    return dict(all_prefix_correct=int(all(values)), insertion_correct=values[0],
                correct_observations=sum(values), forget_observations=forgotten,
                ever_forgot=int(forgotten > 0))


def assignments(group):
    unknown = sorted({x['key'] for xs in group.values() for x in xs if x['correct'] is None})
    assert len(unknown) <= 3, 'Reassess exhaustive review if the frozen bindings differ'
    for values in itertools.product((0, 1), repeat=len(unknown)):
        assignment = dict(zip(unknown, values))
        yield {arm: [x['correct'] if x['correct'] is not None else assignment[x['key']]
                     for x in sorted(xs, key=lambda x: x['prefix'])] for arm, xs in group.items()}


def accumulate(result, outcomes):
    for arm in ('U', 'H_SOURCE', 'H_minus_U'):
        for metric in outcomes[0]['U']:
            values = [(o['H_SOURCE'][metric] - o['U'][metric]) if arm == 'H_minus_U'
                      else o[arm][metric] for o in outcomes]
            interval = result.setdefault(arm, {}).setdefault(metric, [0, 0])
            interval[0] += min(values)
            interval[1] += max(values)


def analyze(rows):
    scores = {}
    identities = defaultdict(set)
    final = defaultdict(lambda: defaultdict(dict))
    histories = defaultdict(lambda: defaultdict(list))
    for x in rows:
        assert x['status'] in ('FORMAT_VALID', 'MISSING')
        assert (x['correct'] in (0, 1)) if x['status'] == 'FORMAT_VALID' else x['correct'] is None
        assert x['key'] not in scores or scores[x['key']] == x['correct']
        scores[x['key']] = x['correct']
        identity = (x['owner'], x['query_id'], x['role'])
        identities[x['key']].add(identity)
        if x['method'] in ('BASE', 'U', 'H_SOURCE'):
            final[identity][x['method']] = [x]
        for arm in ('U', 'H_SOURCE'):
            if x['method'].startswith(arm + '_PREFIX_'):
                histories[identity][arm].append(x)
    # Independent unknowns between queries make summed scenario bounds exact.
    assert all(len(v) == 1 for v in identities.values())
    continuous, per_edit = {}, {owner: {} for owner in range(1, 25)}
    for identity, group in histories.items():
        owner, _, role = identity
        assert set(group) == {'U', 'H_SOURCE'} and role in ('NATIVE', 'T2G')
        assert all(sorted(x['prefix'] for x in xs) == list(range(owner, 25)) for xs in group.values())
        outcomes = [{a: trace(vs) for a, vs in values.items()} for values in assignments(group)]
        accumulate(continuous.setdefault(role, {}), outcomes)
        accumulate(per_edit[owner].setdefault(role, {}), outcomes)
    paired = {}
    for identity, group in final.items():
        assert set(group) == {'BASE', 'U', 'H_SOURCE'}
        role = identity[-1]
        outcomes = []
        for values in assignments(group):
            b, u, h = (values[a][0] for a in ('BASE', 'U', 'H_SOURCE'))
            outcomes.append(dict(U=dict(correct=u, damage=int(b == 1 and u == 0)),
                                 H_SOURCE=dict(correct=h, damage=int(b == 1 and h == 0))))
        accumulate(paired.setdefault(role, {}), outcomes)
    assert len(rows) == 3783
    assert sum(len(g['U']) for k, g in histories.items() if k[-1] == 'NATIVE') == 300
    assert sum(len(g['U']) for k, g in histories.items() if k[-1] == 'T2G') == 1200
    assert trace([1, 0, 1])['all_prefix_correct'] == 0 and trace([1, 0, 1])['forget_observations'] == 1
    assert trace([0, 0])['forget_observations'] == 0
    assert all(o['U'] == o['H_SOURCE'] for o in assignments(dict(
        U=[dict(prefix=1, key='shared', correct=None)],
        H_SOURCE=[dict(prefix=1, key='shared', correct=None)])))
    return dict(status='FROZEN_RESULT_REVIEW_WITH_MISSING_BOUNDS',
                interpretation='These are missing-score bounds, not statistical confidence intervals',
                continuity_definition='All prefixes from insertion must be correct; forgetting means any earlier correct then a later incorrect prefix',
                full_denominators=dict(source=24, T2G=96, source_prefix_observations=300,
                                       T2G_prefix_observations=1200, old_protection=96, new_protection=45),
                paired_final=paired, continuous=continuous, per_edit=per_edit,
                old_Base_correct=62, old_protection_verified=False,
                original_old_protection_flag_caveat='Original flag counts only known damage; upper bound3 prevents verifying zero damage',
                scoring=dict(valid_payloads=252, missing_payloads=198, pending_payloads=0),
                original_results_unchanged=True, new_Judge_calls=0)


if __name__ == '__main__':
    result = analyze(json.loads(Path(os.environ['ANALYSIS_INPUT']).read_text())['rows'])
    original = json.loads(Path(os.environ['ORIGINAL_RESULTS']).read_text())
    for role in ('NATIVE', 'T2G', 'HELDOUT', 'NEW_HELDOUT'):
        assert result['paired_final'][role]['H_minus_U']['correct'] == original['H_minus_U'][role]['correct_count_difference_bounds']
        for arm in ('U', 'H_SOURCE'):
            assert result['paired_final'][role][arm]['correct'] == original['panels'][arm][role]['bounds']['candidate_correct']
            assert result['paired_final'][role][arm]['damage'] == original['panels'][arm][role]['bounds']['new_damage']
    Path(os.environ['ANALYSIS_OUTPUT']).write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(status='PASS', complete_denominators=True, original_paired_bounds_reproduced=True)))
