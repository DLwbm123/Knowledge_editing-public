"""Build a method-blinded, source-bound full-answer packet without loading a model."""
import json
from pathlib import Path
import random
import itertools
import statistics


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def build(private):
    private = Path(private)
    root = private / 'matched_completed'
    tasks = read(root / 'TASKS.json')['tasks']
    assert read(root / 'COMPLETE.json')['counts']['generations'] == 462
    assert read(root / 'WORKER_EXIT.json')['code'] == 0
    assert read(root / 'GUARD_EXIT.json')['code'] == 0
    rows, unique = [], {}
    all_new = []
    for task in tasks:
        for arm in ('C', 'M13B', 'M13P', 'S'):
            assert len(read(root / task['task_id'] / arm / 'TRAIN_COMPLETE.json')['curve']) == 160
            all_new.append(read(root / task['task_id'] / arm / 'ZERO_SMOKE.json'))
        for arm in ('BASE', 'M13B', 'M13P', 'S'):
            blocks = [('native', [task['target']]), ('heldout', task['heldout']),
                      ('protect', task['protect'])]
            if arm != 'BASE':
                blocks.append(('FIT', task['train'][1:]))
            for kind, questions in blocks:
                for row in questions:
                    if arm == 'BASE' and kind == 'native':
                        answer = task['Base_target']
                    elif arm == 'BASE' and kind == 'protect':
                        answer = task['Base_protect'][row['query_id']]
                    else:
                        folder = root / task['task_id'] / arm
                        path = folder / row['query_id'] if arm == 'BASE' else folder / 'outputs' / row['query_id']
                        answer = read(path.with_suffix('.json'))
                        all_new.append(answer)
                    assert answer['binding']['question'] == row['question']
                    assert answer['binding']['reference'] == row['reference']
                    assert answer['binding']['image_path'] == row['image_path']
                    key = (task['source_group'], row['question'], row['reference'], answer['raw_answer'])
                    if key not in unique:
                        unique[key] = dict(source_group=key[0], question=key[1], reference=key[2],
                                           full_answer=key[3])
                    rows.append(dict(task_id=task['task_id'], role=task['role'], arm=arm,
                                     kind=kind, query_id=row['query_id'], key=key))
    assert len(rows) == 448 and len(all_new) == 462
    shuffled = list(unique)
    random.Random(10720261024).shuffle(shuffled)
    mapping = {key: f'R{index:04d}' for index, key in enumerate(shuffled, 1)}
    blind = [dict(review_id=mapping[key], **unique[key]) for key in shuffled]
    for row in rows:
        row['review_id'] = mapping[row.pop('key')]
    write(private / 'V107_BLIND_PACKET.json', blind)
    write(private / 'V107_MAPPING_SEALED.json', rows)
    source = read(private / 'V95_SOURCE_PACKET.json')
    write(private / 'V107_BLIND_SOURCE.json', {g['metadata']['source_group']:
          [dict(question=q['question'], answer=q['answer']) for q in g['source_English']]
          for g in source['groups']})
    summary = dict(bindings=len(rows), unique_full_answers=len(blind), new_generations=len(all_new),
                   EOS=sum(v['EOS'] for v in all_new), at_cap=sum(v['at_cap'] for v in all_new),
                   method_labels_hidden=True, model_calls=0)
    write(private / 'V107_REVIEW_PACKET_RECEIPT.json', summary)
    return summary


def paired(a, b):
    differences = [x - y for x, y in zip(a, b)]
    n = len(differences)
    observed = sum(differences) / n
    extreme = sum(abs(sum(s * v for s, v in zip(signs, differences)) / n) >= abs(observed) - 1e-15
                  for signs in itertools.product((-1, 1), repeat=n))
    rng = random.Random(10720261024)
    bootstrap = sorted(sum(rng.choice(differences) for _ in range(n)) / n for _ in range(10000))
    def quantile(q):
        index = q * (len(bootstrap) - 1)
        lo = int(index)
        return bootstrap[lo] + (bootstrap[min(lo + 1, len(bootstrap) - 1)] - bootstrap[lo]) * (index - lo)
    return dict(n_groups=n, paired_differences=differences, mean_difference=observed,
                exact_two_sided_p=extreme / 2**n, bootstrap95=[quantile(.025), quantile(.975)])


def summarize(private):
    private = Path(private)
    frozen = read(private / 'V107_BLIND_VERDICTS_FROZEN.json')
    assert frozen['method_mapping_opened'] is False and frozen['all_183_complete_answers_read']
    labels = {v['review_id']: v for v in frozen['labels']}
    bindings = read(private / 'V107_MAPPING_SEALED.json')
    assert len(labels) == 183 and len(bindings) == 448
    joined = [dict(v, **{k: value for k, value in labels[v['review_id']].items() if k != 'review_id'})
              for v in bindings]
    summaries, groups, comparisons = {}, {}, {}
    for role in ('DEV_CANDIDATE', 'CONFIRM_CANDIDATE'):
        summaries[role], groups[role] = {}, {}
        for arm in ('BASE', 'M13B', 'M13P', 'S'):
            rows = [v for v in joined if v['role'] == role and v['arm'] == arm]
            by_kind = {}
            for kind in ('native', 'FIT', 'heldout', 'protect'):
                selected = [v for v in rows if v['kind'] == kind]
                by_kind[kind] = dict(n=len(selected), core=sum(v['core_correct'] for v in selected),
                    full=sum(v['full_supported'] for v in selected),
                    extra_contradiction=sum(v['extra_explicit_contradiction'] for v in selected),
                    unsupported_expansion=sum(v['unsupported_expansion'] for v in selected),
                    unverified_diagnostic_core=sum(v['unverified_diagnostic_core'] for v in selected))
            per_group = []
            for task in sorted({v['task_id'] for v in rows}):
                selected = [v for v in rows if v['task_id'] == task]
                target = [v for v in selected if v['kind'] in ('native', 'heldout')]
                protect = [v for v in selected if v['kind'] == 'protect']
                assert len(target) == 3 and protect
                robust = all(v['core_correct'] for v in target)
                keep = all(v['full_supported'] for v in protect)
                per_group.append(dict(task_id=task, target_robust=robust, protect_all_full=keep,
                                      joint=robust and keep))
            by_kind.update(joint_success=sum(v['joint'] for v in per_group),
                           target_robust_groups=sum(v['target_robust'] for v in per_group),
                           protection_all_groups=sum(v['protect_all_full'] for v in per_group),
                           n_groups=len(per_group))
            summaries[role][arm] = by_kind
            groups[role][arm] = per_group
        comparisons[role] = {}
        for a, b in (('M13P', 'M13B'), ('M13P', 'S'), ('S', 'M13B')):
            comparisons[role][a + '_minus_' + b] = paired(
                [int(v['joint']) for v in groups[role][a]], [int(v['joint']) for v in groups[role][b]])
    mechanism = {}
    for arm in ('M13B', 'M13P', 'S'):
        ratios, relative_errors, active = [], [], []
        for path in (private / 'matched_completed').glob('T*/' + arm + '/TRAIN_COMPLETE.json'):
            for row in read(path)['curve']:
                target = row['matching']['target']
                relative_errors.append(abs(row['operator_step_norm'] - target) / target if target else 0.)
                active.append(row['balance']['active'])
                for group in row.get('anti_parallel', {}).get('groups', []):
                    if group['penalty_norm_before']:
                        ratios.append(group['penalty_norm_after'] / group['penalty_norm_before'])
        mechanism[arm] = dict(steps=len(active), balance_cap_active=sum(active),
            max_relative_step_matching_error=max(relative_errors), nonzero_penalty_groups=len(ratios),
            rho_min=min(ratios) if ratios else None, rho_median=statistics.median(ratios) if ratios else None,
            rho_below_half=sum(v < .5 for v in ratios))
    result = dict(summaries=summaries, group_outcomes=groups, comparisons=comparisons, mechanism=mechanism,
                  cost=read(private / 'matched_completed/GUARD_EXIT.json'),
                  counts=read(private / 'matched_completed/COMPLETE.json')['counts'])
    write(private / 'V107_UNBLINDED_SUMMARY.json', result)
    write(private / 'V107_SCORED_BINDINGS_PRIVATE.json', joined)
    return result
