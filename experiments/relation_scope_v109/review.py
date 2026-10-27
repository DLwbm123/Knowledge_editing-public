"""Exact frozen semantic inheritance and source-group outcomes for scoped generation."""
import importlib.util
import json
from pathlib import Path
import random


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def identity(row):
    return tuple(row[k] for k in ('source_group', 'question', 'reference', 'full_answer'))


def build(private, old107, old108, admission_name='SCOPE_ADMISSION_FROZEN.json'):
    private, old107, old108 = Path(private), Path(old107), Path(old108)
    labels107 = {r['review_id']: r for r in read(old107 / 'V107_BLIND_VERDICTS_FROZEN.json')['labels']}
    labels108 = {r['review_id']: r for r in read(old108 / 'BLIND_VERDICTS_FROZEN.json')['labels']}
    library = {identity(r): labels107[r['review_id']] for r in read(old107 / 'V107_BLIND_PACKET.json')}
    library.update({identity(r): labels108[r['review_id']] for r in read(old108 / 'BLIND_PACKET.json')})
    root = private / 'completed'
    assert read(root / 'COMPLETE.json')['status'] == 'BLIND_REVIEW_PENDING'
    assert read(root / 'WORKER_EXIT.json')['code'] == read(root / 'GUARD_EXIT.json')['code'] == 0
    assert read(root / admission_name)['admitted']
    bindings, inherited, novel = [], {}, {}
    for task in read(root / 'TASKS.json')['tasks']:
        assert read(root / task['task_id'] / 'GENERATION_COMPLETE.json')['OFF_Base_exact']
        for kind, rows in (('native', [task['target']]), ('FIT', task['train'][1:]),
                           ('heldout', task['heldout']), ('protect', task['protect'])):
            for row in rows:
                answer = read(root / task['task_id'] / 'outputs' / (row['query_id'] + '.json'))
                assert all(answer['binding'][name] == row[name] for name in ('question', 'reference', 'image_path'))
                value = dict(source_group=task['source_group'], question=row['question'],
                             reference=row['reference'], full_answer=answer['raw_answer'])
                key = identity(value)
                (inherited if key in library else novel)[key] = library[key] if key in library else value
                bindings.append(dict(task_id=task['task_id'], role=task['role'], query_id=row['query_id'],
                    kind=kind, ON=answer['edit_enabled'], EOS=answer['EOS'], at_cap=answer['at_cap'], key=key))
    assert len(bindings) == 126
    shuffled = list(novel)
    random.Random(10920261025).shuffle(shuffled)
    new_ids = {key: f'V{n:04d}' for n, key in enumerate(shuffled, 1)}
    known_ids = {key: f'I{n:04d}' for n, key in enumerate(inherited, 1)}
    for row in bindings:
        key = row.pop('key')
        row['review_id'] = known_ids[key] if key in known_ids else new_ids[key]
    write(private / 'BLIND_PACKET.json', [dict(novel[k], review_id=new_ids[k]) for k in shuffled])
    write(private / 'INHERITED_VERDICTS.json', [dict(inherited[k], review_id=known_ids[k]) for k in inherited])
    write(private / 'MAPPING_SEALED.json', bindings)
    return dict(bindings=126, unique_new=len(novel), inherited_bindings=sum(r['review_id'].startswith('I') for r in bindings))


def summarize(private, old107, old108, seed=10920261025, admission_name='SCOPE_ADMISSION_FROZEN.json'):
    private, old107, old108 = Path(private), Path(old107), Path(old108)
    frozen = read(private / 'BLIND_VERDICTS_FROZEN.json')
    assert frozen['method_mapping_opened'] is False
    labels = {r['review_id']: r for r in read(private / 'INHERITED_VERDICTS.json') + frozen['labels']}
    new = [dict(r, **{k: v for k, v in labels[r['review_id']].items() if k != 'review_id'})
           for r in read(private / 'MAPPING_SEALED.json')]
    sets = dict(SCOPE=new, P_SOURCE_STOP=read(old108 / 'SCORED_BINDINGS_PRIVATE.json'))
    old = read(old107 / 'V107_SCORED_BINDINGS_PRIVATE.json')
    sets.update({a: [r for r in old if r['arm'] == a] for a in ('M13B', 'M13P', 'S')})
    summaries, groups = {}, {}
    for arm, values in sets.items():
        summaries[arm], groups[arm] = {}, {}
        for role in ('DEV_CANDIDATE', 'CONFIRM_CANDIDATE', 'ALL'):
            rows = [r for r in values if role == 'ALL' or r['role'] == role]
            group = []
            for task in sorted({r['task_id'] for r in rows}):
                selected = [r for r in rows if r['task_id'] == task]
                robust = all(r['core_correct'] for r in selected if r['kind'] in ('native', 'heldout'))
                kept = all(r['full_supported'] for r in selected if r['kind'] == 'protect')
                group.append(dict(task_id=task, target_robust=robust, protect_all_full=kept, joint=robust and kept))
            groups[arm][role] = group
            summaries[arm][role] = {kind: dict(n=len([r for r in rows if r['kind'] == kind]),
                core=sum(r['core_correct'] for r in rows if r['kind'] == kind),
                full=sum(r['full_supported'] for r in rows if r['kind'] == kind)) for kind in ('native', 'FIT', 'heldout', 'protect')}
            summaries[arm][role].update(joint=sum(r['joint'] for r in group), n_groups=len(group))
    routing = {role: {kind: dict(n=len([r for r in new if r['kind'] == kind and (role == 'ALL' or r['role'] == role)]),
        ON=sum(r['ON'] for r in new if r['kind'] == kind and (role == 'ALL' or r['role'] == role)))
        for kind in ('native', 'FIT', 'heldout', 'protect')} for role in ('DEV_CANDIDATE', 'CONFIRM_CANDIDATE', 'ALL')}
    helper = Path(__file__).parents[1] / 'm13p_confirmation_prep_v107/review.py'
    spec = importlib.util.spec_from_file_location('paired', helper)
    paired = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(paired)
    comparisons = {role: {a: paired.paired([int(v['joint']) for v in groups['SCOPE'][role]],
        [int(v['joint']) for v in groups[a][role]], seed=seed)
        for a in ('P_SOURCE_STOP', 'M13B', 'M13P', 'S')} for role in ('CONFIRM_CANDIDATE', 'ALL')}
    primary = summaries['SCOPE']['CONFIRM_CANDIDATE']
    all_rows = summaries['SCOPE']['ALL']
    gate = dict(held_fit_source_30=primary['native']['core'] + primary['FIT']['core'] == 30,
        held_fit_expression_12=primary['heldout']['core'] == 12,
        held_fit_protect_10_full=primary['protect']['full'] == 10,
        held_fit_joint_6=primary['joint'] == 6,
        all98_source_expression=all_rows['native']['core'] + all_rows['FIT']['core'] + all_rows['heldout']['core'] == 98,
        all28_protect_full=all_rows['protect']['full'] == 28)
    result = dict(summaries=summaries, group_outcomes=groups, routing=routing, comparisons=comparisons,
        development_gate=gate, development_gate_pass=all(gate.values()),
        evaluation_EOS=sum(r['EOS'] for r in new), evaluation_at_cap=sum(r['at_cap'] for r in new),
        admission=read(private / 'completed' / admission_name),
        counts=read(private / 'completed/COMPLETE.json')['counts'], cost=read(private / 'completed/GUARD_EXIT.json'))
    write(private / 'RESULTS.json', result)
    write(private / 'SCORED_BINDINGS_PRIVATE.json', new)
    return result
