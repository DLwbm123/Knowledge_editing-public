"""Bound frozen four-arm results without new training, generation, or scoring."""
import itertools
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path

ARMS = ('U', 'H_SOURCE', 'TT88_SOURCE', 'BALANCEDIT_SOURCE')


def assignments(group):
    unknown = sorted({x['key'] for xs in group.values() for x in xs if x['correct'] is None})
    assert len(unknown) <= 3
    for values in itertools.product((0, 1), repeat=len(unknown)):
        assigned = dict(zip(unknown, values))
        yield {arm: [x['correct'] if x['correct'] is not None else assigned[x['key']]
                     for x in sorted(xs, key=lambda x: x['prefix'])] for arm, xs in group.items()}


def trace(values):
    seen = False
    lost = 0
    for value in values:
        if seen and not value:
            lost += 1
        seen |= bool(value)
    return dict(insertion_correct=values[0], all_prefix_correct=int(all(values)),
                correct_observations=sum(values), forget_observations=lost,
                ever_forgot=int(lost > 0))


def collect(target, outcomes):
    for arm in ARMS:
        for metric in outcomes[0][arm]:
            values = [o[arm][metric] for o in outcomes]
            interval = target.setdefault('arms', {}).setdefault(arm, {}).setdefault(metric, [0, 0])
            interval[0] += min(values)
            interval[1] += max(values)
    for left, right in itertools.combinations(ARMS, 2):
        for metric in outcomes[0][left]:
            values = [o[left][metric] - o[right][metric] for o in outcomes]
            interval = target.setdefault('paired', {}).setdefault(left + '_minus_' + right, {}).setdefault(metric, [0, 0])
            interval[0] += min(values)
            interval[1] += max(values)


def review(raw, original):
    rows = raw['rows']
    assert len(rows) == 8433
    scores, identities = {}, defaultdict(set)
    histories, finals = defaultdict(lambda: defaultdict(list)), defaultdict(dict)
    for x in rows:
        assert x['status'] in ('FORMAT_VALID', 'MISSING')
        assert (x['correct'] in (0, 1)) if x['status'] == 'FORMAT_VALID' else x['correct'] is None
        assert x['key'] not in scores or scores[x['key']] == x['correct']
        scores[x['key']] = x['correct']
        identity = (x['owner'], x['query_id'], x['role'])
        identities[x['key']].add(identity)
        if x['method'] in ('BASE',) + ARMS:
            finals[identity][x['method']] = [x]
        for arm in ARMS:
            if x['method'].startswith(arm + '_PREFIX_'):
                histories[identity][arm].append(x)
    # No missing variable crosses query identities; summed scenario bounds are exact.
    assert all(len(v) == 1 for v in identities.values())
    continuous, final, per_edit = {}, {}, {owner: {} for owner in range(1, 25)}
    for identity, group in histories.items():
        owner, _, role = identity
        assert set(group) == set(ARMS)
        assert all(sorted(x['prefix'] for x in xs) == list(range(owner, 25)) for xs in group.values())
        outcomes = [{arm: trace(values[arm]) for arm in ARMS} for values in assignments(group)]
        collect(continuous.setdefault(role, {}), outcomes)
        collect(per_edit[owner].setdefault(role, {}), outcomes)
    for identity, group in finals.items():
        assert set(group) == set(('BASE',) + ARMS)
        outcomes = []
        for values in assignments(group):
            base = values['BASE'][0]
            outcomes.append({a: dict(correct=values[a][0], damage=int(base == 1 and values[a][0] == 0)) for a in ARMS})
        collect(final.setdefault(identity[-1], {}), outcomes)
    for arm in ARMS:
        for role, denominator in (('NATIVE', 300), ('T2G', 1200)):
            assert sum(len(g[arm]) for k, g in histories.items() if k[-1] == role) == denominator
        assert continuous['NATIVE']['arms'][arm]['all_prefix_correct'] == original['continual'][arm]['T0_all_prefix_retained_owner_count_bounds']
        for role in ('NATIVE', 'T2G', 'HELDOUT', 'NEW_HELDOUT'):
            assert final[role]['arms'][arm]['correct'] == original['panels'][arm][role]['bounds']['candidate_correct']
            assert final[role]['arms'][arm]['damage'] == original['panels'][arm][role]['bounds']['new_damage']
    for role in final:
        for pair, metrics in final[role]['paired'].items():
            assert metrics['correct'] == original['paired_final'][pair][role]['left_minus_right_bounds']
    assert trace([1, 0, 1])['all_prefix_correct'] == 0
    assert trace([1, 0, 1])['forget_observations'] == 1
    assert trace([0, 0])['forget_observations'] == 0
    assert all(o['U'] == o['H_SOURCE'] for o in assignments(dict(
        U=[dict(prefix=1, key='same', correct=None)], H_SOURCE=[dict(prefix=1, key='same', correct=None)])))
    costs = {}
    for arm, timing in original['inference'].items():
        structure = timing['structure']
        events = [e for e in raw['performance'][arm]['events'] if not e['warmup']]
        assert len(events) == 36
        costs[arm] = dict(added_parameters=structure['added_parameters'],
                         editor_branch=structure['editor_branch'],
                         serialized_bank_bytes=structure['serialized_bank_bytes'],
                         replacement_W_file_bytes=raw['captures']['CAPTURE_' + arm + '.json']['bytes'] if arm in ('U', 'H_SOURCE') else 0,
                         peak_GPU_allocated_bytes=structure['gpu_peak_allocated_bytes'],
                         mean_of_query_medians_seconds=statistics.mean(timing['paired_query_medians_seconds']),
                         source8_mean_of_query_medians_seconds=statistics.mean(timing['paired_query_medians_seconds'][:8]),
                         source8_tokens_per_three_repeats=sum(e['tokens'] for e in events if e['query_slot'] < 8),
                         total_tokens=timing['total_tokens'], total_seconds=timing['total_seconds'],
                         median_route_seconds=timing['median_route_seconds'],
                         median_prefill_seconds=timing['median_prefill_seconds'],
                         per_query_tokens=timing['lengths'])
    assert costs['H_SOURCE']['per_query_tokens'][:8] == costs['TT88_SOURCE']['per_query_tokens'][:8] == costs['BALANCEDIT_SOURCE']['per_query_tokens'][:8]
    return dict(status='COMPLETE_REVIEW_WITH_INHERITED_MISSING',
                interval_kind='Missing-score identification bounds, not statistical confidence intervals',
                scoring=original['scoring_statuses'], missing_consumers=original['missing_consumers'],
                full_denominators=dict(source=24, T2G=96, source_prefix=300, T2G_prefix=1200, old_protection=96, new_protection=45),
                final=final, continuous=continuous, per_edit=per_edit, inference_and_storage=costs,
                protection_prefixes=original['protection_prefixes'], resource=original['resource'],
                engineering_failures_retained=raw['engineering_archives'],
                owned_matrix_files_remaining=raw['current_owned_matrix_files'],
                original_results_unchanged=True, new_Judge_calls=0,
                limitations=original['limitations'] + ['single fixed method order and three timing repeats; instrumented timing',
                    'replacement W payload and added expert bank are distinct deployment representations'])


if __name__ == '__main__':
    raw = json.loads(Path(os.environ['ANALYSIS_INPUT']).read_text())
    original = json.loads(Path(os.environ['ORIGINAL_RESULTS']).read_text())
    result = review(raw, original)
    Path(os.environ['ANALYSIS_OUTPUT']).write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(status='PASS', original_bounds_reproduced=True, four_complete_prefix_denominators=True)))
