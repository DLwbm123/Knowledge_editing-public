"""Blind full-answer review and preregistered candidate selection for fixed replay."""
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


def build(private, sources):
    private = Path(private)
    root = private / 'completed'
    assert read(root / 'WORKER_EXIT.json')['code'] == read(root / 'GUARD_EXIT.json')['code'] == 0
    assert len(read(root / 'ALL_TRAINING_FROZEN.json')['completed']) == 28
    library = {}
    for packet, verdicts in sources:
        labels = {v['review_id']: v for v in read(verdicts)['labels']}
        for row in read(packet):
            library[key(row)] = labels[row['review_id']]
    mappings, answers, novel, inherited = [], [], {}, {}
    for task in read(root / 'TASKS.json')['tasks']:
        for arm in ('BASE', 'REPLAY_DOUBLE', 'PATH_REPLAY'):
            if arm == 'BASE':
                kinds = [('protect_new', task['protect_new'])]
            else:
                dest = root / task['task_id'] / arm
                train = read(dest / 'TRAIN_COMPLETE.json')
                assert train['fixed_steps'] and train['step'] == 160
                assert train['no_evaluation_selection'] and train['save_load_exact']
                assert read(dest / 'GENERATION_COMPLETE.json')['rows'] == 7 + 5 * len(task['protect'])
                kinds = [('native', [task['target']]), ('FIT', task['train'][1:]),
                         ('heldout', task['heldout']), ('protect', task['protect']),
                         ('protect_replay', task['protect_extra']), ('protect_new', task['protect_new'])]
            for kind, rows in kinds:
                for row in rows:
                    folder = root / task['task_id'] / ('BASE_NEW' if arm == 'BASE' else arm)
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
    assert len(mappings) == len(answers) == 532
    shuffled = list(novel)
    random.Random(12020261025).shuffle(shuffled)
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
        **{name: inherited[k][name] for name in fields}, inheritance='Exact group, question, reference, full answer') for k in inherited])
    write(private / 'MAPPING_SEALED.json', mappings)
    receipt = dict(bindings=532, novel_unique_full_answers=len(novel), inherited_unique_full_answers=len(inherited),
        inherited_bindings=sum(r['review_id'].startswith('I') for r in mappings),
        EOS=sum(a['EOS'] for a in answers), at_cap=sum(a['at_cap'] for a in answers))
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
    assert len(rows) == 532
    base = [r for r in rows if r['arm'] == 'BASE']
    assert len(base) == 56
    support = {(r['task_id'], r['query_id']) for r in base if r['full_supported']}
    support_groups = len({identity[0] for identity in support})
    mechanism = read(root / 'PATH_MECHANISM_AUDIT.json')
    assert len(mechanism) == 28
    summary, outcomes, damage, gates = {}, {}, {}, {}
    for arm in ('REPLAY_DOUBLE', 'PATH_REPLAY'):
        values = [r for r in rows if r['arm'] == arm]
        assert len(values) == 238
        damage[arm] = {(r['task_id'], r['query_id']) for r in values
            if r['kind'] == 'protect_new' and (r['task_id'], r['query_id']) in support and not r['full_supported']}
        outcomes[arm] = []
        for task in sorted({r['task_id'] for r in values}):
            selected = [r for r in values if r['task_id'] == task]
            target = [r for r in selected if r['kind'] in ('native', 'heldout')]
            protected = [r for r in selected if r['kind'] == 'protect']
            base_ids = {identity for identity in support if identity[0] == task}
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
        gates[arm] = dict(target98_core=all(r['core_correct'] for r in values if r['kind'] in ('native', 'FIT', 'heldout')),
            known84_core=all(r['core_correct'] for r in values if r['kind'] in ('protect', 'protect_replay')),
            native28_full=summary[arm]['protect']['full'] == 28,
            sufficient_Base_support=len(support) >= 28 and support_groups >= 10,
            no_new_Base_damage=not damage[arm],
            new_full_at_least_Base=summary[arm]['protect_new']['full'] >= len(support))
        if arm == 'PATH_REPLAY':
            gates[arm]['native28_Base_tokens_exact'] = all(r['Base_tokens_exact'] for r in mechanism)
    passed = {a: all(g.values()) for a, g in gates.items()}
    selected = 'REPLAY_DOUBLE' if passed['REPLAY_DOUBLE'] else 'PATH_REPLAY' if passed['PATH_REPLAY'] else None
    helper = Path(__file__).parents[1] / 'm13p_confirmation_prep_v107/review.py'
    spec = importlib.util.spec_from_file_location('paired_statistics', helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    comparisons = dict(joint=module.paired([int(r['joint']) for r in outcomes['PATH_REPLAY']],
        [int(r['joint']) for r in outcomes['REPLAY_DOUBLE']], seed=12020261025),
        Base_damage=module.paired([r['Base_damage'] for r in outcomes['PATH_REPLAY']],
        [r['Base_damage'] for r in outcomes['REPLAY_DOUBLE']], seed=12020261025))
    result = dict(status='COMPLETE_DEVELOPMENT_CANDIDATE_SELECTED' if selected else 'COMPLETE_BOTH_DEVELOPMENT_GATES_FAILED',
        selected_candidate=selected, candidate_pass=passed, candidate_conditions=gates,
        summary=summary, group_outcomes=outcomes, comparisons=comparisons,
        Base_new=dict(n=56, core=sum(r['core_correct'] for r in base), full=len(support), source_groups=support_groups),
        counts=read(root / 'COMPLETE.json')['counts'], cost=read(root / 'GUARD_EXIT.json'),
        path_mechanism=dict(n=28, Base_tokens_exact=sum(r['Base_tokens_exact'] for r in mechanism),
            max_active_residual_norm=max(r['maximum_active_residual_norm'] for r in mechanism)),
        scope='All14 exposed development groups; 84 known protection training rows and56 new expression tests')
    write(private / 'RESULTS.json', result)
    write(private / 'SCORED_BINDINGS_PRIVATE.json', rows)
    return result
