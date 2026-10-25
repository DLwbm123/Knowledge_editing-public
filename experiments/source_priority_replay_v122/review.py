"""Exact-label inheritance and full-answer review for a fixed-state path extension."""
import importlib.util
import json
from pathlib import Path
import random


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def key(source_group, answer):
    return (source_group, answer['binding']['question'], answer['binding']['reference'], answer['raw_answer'])


def build(private, previous):
    private, previous = Path(private), Path(previous)
    root, old_root = private / 'completed', previous / 'completed'
    assert read(root / 'WORKER_EXIT.json')['code'] == read(root / 'GUARD_EXIT.json')['code'] == 0
    old_tasks = {t['task_id']: t for t in read(old_root / 'TASKS.json')['tasks']}
    library = {}
    for row in read(previous / 'SCORED_BINDINGS_PRIVATE.json'):
        folder = old_root / row['task_id'] / ('BASE_NEW' if row['arm'] == 'BASE' else row['arm'])
        answer = read(folder / ('' if row['arm'] == 'BASE' else 'outputs') / (row['query_id'] + '.json'))
        library[key(old_tasks[row['task_id']]['source_group'], answer)] = row
    mappings, novel, inherited, answers = [], {}, {}, []
    for task in read(root / 'TASKS.json')['tasks']:
        complete = read(root / task['task_id'] / 'SOURCE_PRIORITY_PATH/GENERATION_COMPLETE.json')
        assert read(root / task['task_id'] / 'SOURCE_PRIORITY_PATH/TRAIN_COMPLETE.json')['save_load_exact']
        assert complete['rows'] == 7 + len(task['protect']) + len(task['protect_extra']) + len(task['protect_new'])
        for kind, rows in (('native', [task['target']]), ('FIT', task['train'][1:]),
                          ('heldout', task['heldout']), ('protect', task['protect']),
                          ('protect_replay', task['protect_extra']), ('protect_new', task['protect_new'])):
            for row in rows:
                answer = read(root / task['task_id'] / 'SOURCE_PRIORITY_PATH/outputs' / (row['query_id'] + '.json'))
                for name in ('question', 'reference', 'image_path'):
                    assert answer['binding'][name] == row[name]
                identity = key(task['source_group'], answer)
                if identity in library:
                    inherited[identity] = library[identity]
                else:
                    novel[identity] = dict(source_group=task['source_group'], question=row['question'],
                        reference=row['reference'], full_answer=answer['raw_answer'])
                mappings.append(dict(task_id=task['task_id'], role=task['role'], kind=kind,
                                     query_id=row['query_id'], identity=identity))
                answers.append(answer)
    assert len(mappings) == len(answers) == 238
    assert len({(r['task_id'], r['query_id']) for r in mappings}) == 238
    for task in read(root / 'TASKS.json')['tasks']:
        replay_ids = {r['query_id'] for r in mappings if r['task_id'] == task['task_id'] and r['kind'] == 'protect_replay'}
        new_ids = {r['query_id'] for r in mappings if r['task_id'] == task['task_id'] and r['kind'] == 'protect_new'}
        assert replay_ids == {r['query_id'] for r in task['protect_extra']}
        assert new_ids == {r['query_id'] for r in task['protect_new']} and replay_ids.isdisjoint(new_ids)
    shuffled = list(novel)
    random.Random(12220261025).shuffle(shuffled)
    new_ids = {value: f'V{index:04d}' for index, value in enumerate(shuffled, 1)}
    old_ids = {value: f'I{index:04d}' for index, value in enumerate(inherited, 1)}
    for row in mappings:
        identity = row.pop('identity')
        row['review_id'] = new_ids.get(identity, old_ids.get(identity))
        assert row['review_id']
    fields = ('source_reference_supported', 'core_correct', 'core_status', 'extra_explicit_contradiction',
              'unsupported_expansion', 'unverified_diagnostic_core', 'full_supported', 'reason')
    write(private / 'BLIND_PACKET.json', [dict(review_id=new_ids[k], **novel[k]) for k in shuffled])
    write(private / 'INHERITED_VERDICTS.json', [dict(review_id=old_ids[k],
        **{name: inherited[k][name] for name in fields}, inheritance='Exact group, question, reference, complete answer') for k in inherited])
    write(private / 'MAPPING_SEALED.json', mappings)
    receipt = dict(bindings=238, novel_unique_full_answers=len(novel), inherited_unique_full_answers=len(inherited),
        inherited_bindings=sum(r['review_id'].startswith('I') for r in mappings),
        EOS=sum(a['EOS'] for a in answers), at_cap=sum(a['at_cap'] for a in answers))
    write(private / 'REVIEW_PACKET_RECEIPT.json', receipt)
    return receipt


def summarize(private, previous):
    private, previous = Path(private), Path(previous)
    root = private / 'completed'
    frozen = read(private / 'BLIND_VERDICTS_FROZEN.json')
    assert frozen['method_mapping_opened'] is False
    labels = {v['review_id']: v for v in read(private / 'INHERITED_VERDICTS.json') + frozen['labels']}
    rows = [dict(row, **{k: v for k, v in labels[row['review_id']].items() if k != 'review_id'})
            for row in read(private / 'MAPPING_SEALED.json')]
    old = read(previous / 'SCORED_BINDINGS_PRIVATE.json')
    base = [r for r in old if r['arm'] == 'BASE']
    parent = [r for r in old if r['arm'] == 'PATH_REPLAY']
    assert len(rows) == len(parent) == 238 and len(base) == 56
    supported = {(r['task_id'], r['query_id']) for r in base if r['full_supported']}
    assert len(supported) == 47
    sets = dict(SOURCE_PRIORITY_PATH=rows, PARENT_PATH_REPLAY=parent)
    outcomes, damage, summary = {}, {}, {}
    for arm, values in sets.items():
        damage[arm] = {(r['task_id'], r['query_id']) for r in values
            if r['kind'] == 'protect_new' and (r['task_id'], r['query_id']) in supported and not r['full_supported']}
        outcomes[arm] = []
        for task in sorted({r['task_id'] for r in values}):
            selected = [r for r in values if r['task_id'] == task]
            target = [r for r in selected if r['kind'] in ('native', 'heldout')]
            protected = [r for r in selected if r['kind'] == 'protect']
            base_ids = {identity for identity in supported if identity[0] == task}
            assert len(target) == 3 and protected
            joint = all(r['core_correct'] for r in target) and all(r['full_supported'] for r in protected)
            joint = joint and bool(base_ids) and not bool(base_ids & damage[arm])
            outcomes[arm].append(dict(task_id=task, joint=joint, Base_support=len(base_ids),
                                      Base_damage=len(base_ids & damage[arm])))
        summary[arm] = {kind: dict(n=len([r for r in values if r['kind'] == kind]),
            core=sum(r['core_correct'] for r in values if r['kind'] == kind),
            full=sum(r['full_supported'] for r in values if r['kind'] == kind),
            extra_contradiction=sum(r['extra_explicit_contradiction'] for r in values if r['kind'] == kind),
            unsupported_expansion=sum(r['unsupported_expansion'] for r in values if r['kind'] == kind))
            for kind in ('native', 'FIT', 'heldout', 'protect', 'protect_replay', 'protect_new')}
        summary[arm].update(joint=sum(r['joint'] for r in outcomes[arm]), n_groups=14, Base_damage=len(damage[arm]))
    mechanism = read(root / 'PATH_MECHANISM_AUDIT.json')
    assert len(mechanism) == 28
    restored = damage['PARENT_PATH_REPLAY'] - damage['SOURCE_PRIORITY_PATH']
    conditions = dict(target98_core=all(r['core_correct'] for r in rows if r['kind'] in ('native', 'FIT', 'heldout')),
        native28_full=summary['SOURCE_PRIORITY_PATH']['protect']['full'] == 28,
        native28_tokens_Base_exact=all(r['Base_tokens_exact'] for r in mechanism),
        no_Base47_damage=not damage['SOURCE_PRIORITY_PATH'],
        challenge_full_at_least_Base=summary['SOURCE_PRIORITY_PATH']['protect_new']['full'] >= 47,
        joint_at_least_parent=summary['SOURCE_PRIORITY_PATH']['joint'] >= 12,
        known84_core=all(r['core_correct'] for r in rows if r['kind'] in ('protect', 'protect_replay')))
    helper = Path(__file__).parents[1] / 'm13p_confirmation_prep_v107/review.py'
    spec = importlib.util.spec_from_file_location('paired_statistics', helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    comparisons = dict(joint=module.paired([int(r['joint']) for r in outcomes['SOURCE_PRIORITY_PATH']],
        [int(r['joint']) for r in outcomes['PARENT_PATH_REPLAY']], seed=12220261025),
        Base_damage=module.paired([r['Base_damage'] for r in outcomes['SOURCE_PRIORITY_PATH']],
        [r['Base_damage'] for r in outcomes['PARENT_PATH_REPLAY']], seed=12220261025))
    ranks = [read(root / t['task_id'] / 'SOURCE_PRIORITY_PATH/TRAIN_COMPLETE.json')['fixed_basis_rank']
             for t in read(root / 'TASKS.json')['tasks']]
    result = dict(status='COMPLETE_DEVELOPMENT_GATE_PASS' if all(conditions.values()) else 'COMPLETE_DEVELOPMENT_GATE_FAILED',
        summary=summary, group_outcomes=outcomes, advancement_conditions=conditions,
        advancement_pass=all(conditions.values()), restored_parent_damage=len(restored),
        new_damage_vs_parent=len(damage['SOURCE_PRIORITY_PATH'] - damage['PARENT_PATH_REPLAY']), comparisons=comparisons,
        counts=read(root / 'COMPLETE.json')['counts'], cost=read(root / 'GUARD_EXIT.json'),
        basis=dict(total_rank=sum(ranks), total_bytes=sum(ranks) * 14336 * 8, reused_original_V118=True),
        native_Base_tokens_exact=sum(r['Base_tokens_exact'] for r in mechanism),
        maximum_active_residual_norm=max(r['maximum_active_residual_norm'] for r in mechanism),
        scope='All14 exposed development groups; fresh2240updates and already exposed56 challenges; no new Base calls')
    write(private / 'RESULTS.json', result)
    write(private / 'SCORED_BINDINGS_PRIVATE.json', rows)
    return result
