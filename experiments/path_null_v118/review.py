"""Blind full-answer review and frozen development gates for the two-arm path test."""
import importlib.util
import json
from pathlib import Path
import random


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def key(row):
    return tuple(row[k] for k in ('source_group', 'question', 'reference', 'full_answer'))


def build(private, old107, old108):
    private, old107, old108 = map(Path, (private, old107, old108))
    root = private / 'completed'
    tasks = read(root / 'TASKS.json')['tasks']
    assert read(root / 'WORKER_EXIT.json')['code'] == read(root / 'GUARD_EXIT.json')['code'] == 0
    assert len(read(root / 'ALL_TRAINING_FROZEN.json')['selected_steps']) == 28
    library = {}
    for packet, verdicts in ((old107 / 'V107_BLIND_PACKET.json', old107 / 'V107_BLIND_VERDICTS_FROZEN.json'),
                             (old108 / 'BLIND_PACKET.json', old108 / 'BLIND_VERDICTS_FROZEN.json')):
        labels = {v['review_id']: v for v in read(verdicts)['labels']}
        for row in read(packet):
            library[key(row)] = labels[row['review_id']]
    mappings, answers, novel, inherited = [], [], {}, {}
    for task in tasks:
        for arm in ('BASE', 'DOUBLE_CONTROL', 'PATH_NULL'):
            if arm == 'BASE':
                kinds = [('protect_extra', task['protect_extra'])]
            else:
                dest = root / task['task_id'] / arm
                train = read(dest / 'TRAIN_COMPLETE.json')
                assert train['save_load_exact'] and train['source_only_selection']
                assert read(dest / 'GENERATION_COMPLETE.json')['rows'] == 7 + len(task['protect']) + len(task['protect_extra'])
                kinds = [('native', [task['target']]), ('FIT', task['train'][1:]),
                         ('heldout', task['heldout']), ('protect', task['protect']),
                         ('protect_extra', task['protect_extra'])]
            for kind, rows in kinds:
                for row in rows:
                    folder = root / task['task_id'] / arm
                    answer = read(folder / ('' if arm == 'BASE' else 'outputs') / (row['query_id'] + '.json'))
                    for name in ('question', 'reference', 'image_path'):
                        assert answer['binding'][name] == row[name]
                    value = dict(source_group=task['source_group'], question=row['question'],
                                 reference=row['reference'], full_answer=answer['raw_answer'])
                    identity = key(value)
                    if identity in library:
                        inherited[identity] = library[identity]
                    else:
                        novel[identity] = value
                    mappings.append(dict(task_id=task['task_id'], role=task['role'], arm=arm,
                                         kind=kind, query_id=row['query_id'], identity=identity))
                    answers.append(answer)
    assert len(mappings) == len(answers) == 420
    shuffled = list(novel)
    random.Random(11820261025).shuffle(shuffled)
    new_ids = {value: f'V{index:04d}' for index, value in enumerate(shuffled, 1)}
    old_ids = {value: f'I{index:04d}' for index, value in enumerate(inherited, 1)}
    for row in mappings:
        identity = row.pop('identity')
        row['review_id'] = new_ids.get(identity, old_ids.get(identity))
        assert row['review_id']
    write(private / 'BLIND_PACKET.json', [dict(review_id=new_ids[k], **novel[k]) for k in shuffled])
    write(private / 'INHERITED_VERDICTS.json', [dict(inherited[k], review_id=old_ids[k],
        inheritance='Exact source group, question, reference and complete answer') for k in inherited])
    write(private / 'MAPPING_SEALED.json', mappings)
    receipt = dict(bindings=420, novel_unique_full_answers=len(novel), inherited_unique_full_answers=len(inherited),
                   inherited_bindings=sum(r['review_id'].startswith('I') for r in mappings),
                   final_EOS=sum(a['EOS'] for a in answers), final_cap=sum(a['at_cap'] for a in answers))
    write(private / 'REVIEW_PACKET_RECEIPT.json', receipt)
    return receipt


def summarize(private):
    private = Path(private)
    root = private / 'completed'
    frozen = read(private / 'BLIND_VERDICTS_FROZEN.json')
    assert frozen['method_mapping_opened'] is False
    labels = {v['review_id']: v for v in read(private / 'INHERITED_VERDICTS.json') + frozen['labels']}
    rows = [dict(row, **{k: v for k, v in labels[row['review_id']].items() if k != 'review_id'})
            for row in read(private / 'MAPPING_SEALED.json')]
    assert len(rows) == 420
    arms = ('DOUBLE_CONTROL', 'PATH_NULL')
    groups = sorted({r['task_id'] for r in rows})
    base = [r for r in rows if r['arm'] == 'BASE']
    assert len(base) == 56 and all(r['kind'] == 'protect_extra' for r in base)
    supported = {(r['task_id'], r['query_id']) for r in base if r['full_supported']}
    damaged, outcomes, summary = {}, {}, {}
    for arm in arms:
        values = [r for r in rows if r['arm'] == arm]
        assert len(values) == 182
        damaged[arm] = {(r['task_id'], r['query_id']) for r in values
                       if r['kind'] == 'protect_extra' and (r['task_id'], r['query_id']) in supported
                       and not r['full_supported']}
        outcomes[arm] = []
        for task in groups:
            selected = [r for r in values if r['task_id'] == task]
            target = [r for r in selected if r['kind'] in ('native', 'heldout')]
            protected = [r for r in selected if r['kind'] == 'protect']
            base_ids = {identity for identity in supported if identity[0] == task}
            assert len(target) == 3 and protected
            joint = all(r['core_correct'] for r in target) and all(r['full_supported'] for r in protected)
            joint = joint and bool(base_ids) and not bool(base_ids & damaged[arm])
            outcomes[arm].append(dict(task_id=task, joint=joint, new_protect_Base_support=len(base_ids),
                                     new_protect_Base_damage=len(base_ids & damaged[arm])))
        summary[arm] = {kind: dict(n=len([r for r in values if r['kind'] == kind]),
            core=sum(r['core_correct'] for r in values if r['kind'] == kind),
            full=sum(r['full_supported'] for r in values if r['kind'] == kind),
            extra_contradiction=sum(r['extra_explicit_contradiction'] for r in values if r['kind'] == kind),
            unsupported_expansion=sum(r['unsupported_expansion'] for r in values if r['kind'] == kind))
            for kind in ('native', 'FIT', 'heldout', 'protect', 'protect_extra')}
        summary[arm].update(joint=sum(r['joint'] for r in outcomes[arm]), n_groups=14,
                            new_protect_Base_damage=len(damaged[arm]))
    null = [r for r in rows if r['arm'] == 'PATH_NULL']
    mechanism = read(root / 'PATH_MECHANISM_AUDIT.json')
    assert len(mechanism) == 28
    restored = damaged['DOUBLE_CONTROL'] - damaged['PATH_NULL']
    support_groups = len({identity[0] for identity in supported})
    conditions = dict(target_98_core=all(r['core_correct'] for r in null if r['kind'] in ('native', 'FIT', 'heldout')),
        original_protect_28_full=summary['PATH_NULL']['protect']['full'] == 28,
        original_protect_Base_tokens_exact=all(r['Base_tokens_exact'] for r in mechanism),
        sufficient_new_protect_Base_support=len(supported) >= 28 and support_groups >= 10,
        no_new_protect_Base_damage=not damaged['PATH_NULL'],
        new_protect_full_not_lower=summary['PATH_NULL']['protect_extra']['full'] >= summary['DOUBLE_CONTROL']['protect_extra']['full'],
        joint_not_lower=summary['PATH_NULL']['joint'] >= summary['DOUBLE_CONTROL']['joint'],
        restores_at_least_one_control_damage=bool(restored))
    helper = Path(__file__).parents[1] / 'm13p_confirmation_prep_v107/review.py'
    spec = importlib.util.spec_from_file_location('paired_statistics', helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    comparisons = dict(joint=module.paired([int(r['joint']) for r in outcomes['PATH_NULL']],
        [int(r['joint']) for r in outcomes['DOUBLE_CONTROL']], seed=11820261025),
        new_protect_Base_damage=module.paired([r['new_protect_Base_damage'] for r in outcomes['PATH_NULL']],
        [r['new_protect_Base_damage'] for r in outcomes['DOUBLE_CONTROL']], seed=11820261025))
    result = dict(status='COMPLETE_DEVELOPMENT_GATE_PASS' if all(conditions.values()) else 'COMPLETE_DEVELOPMENT_GATE_FAILED',
        summary=summary, group_outcomes=outcomes, comparisons=comparisons,
        new_protect_Base=dict(n=56, core=sum(r['core_correct'] for r in base), full=len(supported), source_groups=support_groups),
        advancement_conditions=conditions, advancement_pass=all(conditions.values()),
        restored_control_damage=len(restored), new_damage_vs_control=len(damaged['PATH_NULL'] - damaged['DOUBLE_CONTROL']),
        selected_steps=read(root / 'ALL_TRAINING_FROZEN.json')['selected_steps'],
        counts=read(root / 'COMPLETE.json')['counts'], cost=read(root / 'GUARD_EXIT.json'),
        path_mechanism=dict(n=28, Base_tokens_exact=sum(r['Base_tokens_exact'] for r in mechanism),
            maximum_active_residual_norm=max(r['maximum_active_residual_norm'] for r in mechanism)),
        basis=dict(total_bytes=sum(r['basis_bytes'] for r in read(root / 'ALL_BASES_FROZEN.json')['audits'].values()),
                   total_rank=sum(r['rank'] for r in read(root / 'ALL_BASES_FROZEN.json')['audits'].values())),
        scope='All14 exposed development groups; original protection is construction supervision')
    write(private / 'RESULTS.json', result)
    write(private / 'SCORED_BINDINGS_PRIVATE.json', rows)
    return result
