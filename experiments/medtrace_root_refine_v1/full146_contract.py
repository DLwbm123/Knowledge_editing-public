"""Pure CPU contracts for the original cohort, sources and output coverage."""
import collections
import copy
import json
PREFIXES = (1, 50, 100, 146)
ARMS = ('C', 'M1', 'REFINE')
CAPS = dict(updates=70080, backwards=1331520, operator_backwards=0,
            non_generation_forwards=3200000, local_solves=373760,
            generation_attempts=17373, Judge_attempts=17373)


def task_binding(t):
    return {k: t[k] for k in ('edit_id', 'order', 'native', 'fit_questions', 'seed')}


def source_binding(rows):
    return [{k: r.get(k) for k in ('query_id', 'dataset', 'image_path', 'original_image_path',
                                  'question', 'reference', 'source_group')} for r in rows]


def source_review(original, tasks, anchors, reviews, forbidden):
    errors = []
    if len(tasks) != 146 or [task_binding(t) for t in tasks] != [task_binding(t) for t in original]:
        errors.append('ORIGINAL_TASK_VERSION_OR_ORDER_CHANGED')
    by_owner = collections.defaultdict(list)
    for row in anchors:
        by_owner[row['owner']].append(row)
    if set(by_owner) != set(range(1, 147)):
        errors.append('MISSING_OR_EXTRA_OWNER')
    for owner in range(1, 147):
        rows = by_owner[owner]
        if (len(rows) != 2 or len({r['source_group'] for r in rows}) != 2
                or len({r.get('image_path') for r in rows}) != 2):
            errors.append(f'ANCHOR_COUNT_OR_DUPLICATE:{owner}')
        examined = reviews.get(str(owner), [])
        if not 2 <= len(examined) <= 6 or len({r['candidate_id'] for r in examined}) != len(examined):
            errors.append(f'REVIEW_COUNT:{owner}')
        for row in rows:
            evidence = next((v for v in examined if v['candidate_id'] == row.get('candidate_id')), {})
            required = dict(relation_status='PASS_SOURCE_EVIDENCE',
                image_review='PASS_READABLE_SOURCE_FIGURE',
                basis_conflict_review='PASS_NO_CONTRADICTORY_TARGET',
                role_check='PASS_KNOWN_IMAGE_AND_CASE_EXCLUSION')
            case = row['source_group'].split(':')[0] if row['source_group'].startswith('PMC') else row['source_group']
            if (row.get('role') != 'GFIT' or row['source_group'] in forbidden or case in forbidden
                    or not evidence.get('admitted') or not evidence.get('license')
                    or not evidence.get('source_evidence') or not evidence.get('image_binding')
                    or evidence.get('image_binding', {}).get('path') != row.get('image_path')
                    or evidence.get('task_binding') != task_binding(original[owner-1])
                    or any(evidence.get(k) != v for k, v in required.items())):
                errors.append(f'SOURCE_EVIDENCE:{owner}')
    return dict(status='PASS' if not errors else 'SOURCE_ADMISSION_PENDING', errors=errors,
                owners=146, required_anchors=292, training_ready=not errors)


def fit_rows(task):
    return [dict(task['native'], query_id=f"edit_{task['order']}_{i}", question=q,
                 audit_role='FIT') for i, q in enumerate(task['fit_questions'], 1)]


def ids(tasks):
    return {q for t in tasks for e in t['events'] for q in e['all_probe_query_ids']}


def evaluation_plan(tasks, queries, anchors, base_mask):
    assert len(tasks) == 146 and [t['order'] for t in tasks] == list(range(1, 147))
    assert len(anchors) == 292 and all(len(fit_rows(t)) == 4 for t in tasks)
    rows = []
    def add(method, mode, prefix, owner, query):
        rows.append(dict(method=method, mode=mode, prefix=prefix, owner=owner, query=query))
    for method in ('M1', 'REFINE'):
        for task in tasks:
            for q in sorted(ids([task])):
                add(method, 'single', 1, task['order'], queries[q])
            for row in fit_rows(task):
                add(method, 'single', 1, task['order'], row)
            add(method, 'insertion', task['order'], task['order'], task['native'])
        for n in PREFIXES:
            for q in sorted(ids(tasks[:n])):
                add(method, 'bank', n, 0, queries[q])
            for task in tasks[:n]:
                for row in fit_rows(task):
                    add(method, 'bank', n, 0, row)
    for q in sorted(ids(tasks)):
        add('BASE', 'base', 0, 0, queries[q])
    for task in tasks:
        for row in fit_rows(task):
            add('BASE', 'base', 0, 0, row)
    for row in anchors:
        add('BASE', 'base', 0, 0, dict(row, audit_role='GFIT'))
    keys = [(x['method'], x['mode'], x['prefix'], x['owner'], x['query']['query_id']) for x in rows]
    assert len(keys) == len(set(keys)) == 17373
    assert collections.Counter(x['method'] for x in rows) == dict(M1=7494, REFINE=7494, BASE=2385)
    supported = {t['order']: {q for e in t['events'] if e['task'] == 'T2L'
                              for q in e['all_probe_query_ids'] if base_mask[q]} for t in tasks}
    assert sum(map(len, supported.values())) == 54 and sum(bool(s) for s in supported.values()) == 31
    return rows


def consumer_id(row):
    return json.dumps([row[k] for k in ('method', 'mode', 'prefix', 'owner')]
                      + [row['query']['query_id']], separators=(',', ':'))


def success(gates):
    required = {'complete', 'T2L_better_than_M1', 'T2L_edit_CI_positive',
                'T2L_source_CI_positive', 'T2L_historical_floor', 'FIT_per_owner',
                'T2G_per_owner', 'prefix_FIT_T2G', 'Base_correct_no_new_damage'}
    return set(gates) == required and all(gates[k] is True for k in required)


def selfcheck():
    good = {k: True for k in ('complete', 'T2L_better_than_M1', 'T2L_edit_CI_positive',
        'T2L_source_CI_positive', 'T2L_historical_floor', 'FIT_per_owner', 'T2G_per_owner',
        'prefix_FIT_T2G', 'Base_correct_no_new_damage')}
    assert success(good) and not success({})
    for key in good:
        assert not success(dict(good, **{key: False}))
    assert sum(146 * 160 * k for k in (3, 3, 51)) == CAPS['backwards']
    assert 146 * 160 * 16 == CAPS['local_solves']
    tasks = [dict(edit_id=str(i), order=i, native=dict(query_id=str(i)), fit_questions=[], seed=i)
             for i in range(1, 147)]
    assert source_review(tasks, tasks, [], {}, set())['status'] != 'PASS'
    changed = copy.deepcopy(tasks); changed[-1]['seed'] += 1
    assert 'ORIGINAL_TASK_VERSION_OR_ORDER_CHANGED' in source_review(tasks, changed, [], {}, set())['errors']
    return dict(status='PASS', missing_unknown_changed_seed_rejected=True,
                each_failed_performance_gate_rejected=True, finite_budget_arithmetic=True)
