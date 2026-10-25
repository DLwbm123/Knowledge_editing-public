"""V130 full-answer packet; fixed V126 rubric, exact inherited labels, no model calls."""
import json
from pathlib import Path
import random


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def build(private, sources):
    private = Path(private)
    root = private / 'completed'
    assert read(root / 'WORKER_EXIT.json')['code'] == read(root / 'GUARD_EXIT.json')['code'] == 0
    assert len(read(root / 'ALL_TRAINING_FROZEN_B_BASE_SEQUENCE.json')['completed']) == 14
    control = read(root / 'A_CONTROL_REPRODUCTION.json')
    assert control['pass_all238'] and len(control['comparisons']) == 238
    assert all(all(r[k] for k in ('binding_exact', 'complete_text_exact', 'complete_tokens_exact'))
               for r in control['comparisons'])
    counts = read(root / 'COMPLETE.json')['counts']
    expected = dict(updates=4480, backwards=22400, target_backwards=17920,
                    replay_backwards=4480, operator_backwards=4480, generations=526,
                    replay_supervised_tokens=68300)
    assert all(counts[k] == v for k, v in expected.items())
    assert read(root / 'GENERATION_COMPLETE_B_BASE_SEQUENCE.json')['formal_rows'] == 238
    assert read(root / 'ALL_TRAINING_FROZEN_B_BASE_SEQUENCE.json')['time'] < read(
        root / 'GENERATION_COMPLETE_B_BASE_SEQUENCE.json')['time']
    library = {}
    for packet, verdicts in sources:
        labels = {v['review_id']: v for v in read(verdicts)['labels']}
        for row in read(packet):
            identity = tuple(row[k] for k in ('source_group', 'question', 'reference', 'full_answer'))
            label = labels[row['review_id']]
            if identity in library:
                for name in ('core_correct', 'full_supported', 'extra_explicit_contradiction', 'unsupported_expansion'):
                    assert label[name] == library[identity][name], (identity[:3], name)
            library[identity] = label
    mappings, novel, inherited, answers = [], {}, {}, []
    for task in read(root / 'TASKS.json')['tasks']:
        complete = read(root / task['task_id'] / 'B_BASE_SEQUENCE/GENERATION_COMPLETE.json')
        assert read(root / task['task_id'] / 'B_BASE_SEQUENCE/TRAIN_COMPLETE.json')['save_load_exact']
        assert complete['rows'] == 7 + len(task['protect']) + len(task['protect_extra']) + len(task['protect_new'])
        for kind, rows in (('native', [task['target']]), ('FIT', task['train'][1:]),
                          ('heldout', task['heldout']), ('protect', task['protect']),
                          ('protect_replay', task['protect_extra']), ('protect_new', task['protect_new'])):
            for row in rows:
                answer = read(root / task['task_id'] / 'B_BASE_SEQUENCE/outputs' / (row['query_id'] + '.json'))
                control_answer = read(root / task['task_id'] / 'A_SHORT_REFERENCE/outputs' / (row['query_id'] + '.json'))
                assert answer['binding'] == control_answer['binding']
                for name in ('question', 'reference', 'image_path'):
                    assert answer['binding'][name] == row[name]
                identity = (task['source_group'], answer['binding']['question'],
                            answer['binding']['reference'], answer['raw_answer'])
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
    random.Random(13020261025).shuffle(shuffled)
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


def summarize(private, previous, base_previous):
    import importlib.util

    private = Path(private)
    root = private / 'completed'
    frozen = read(private / 'BLIND_VERDICTS_FROZEN.json')
    assert frozen['method_mapping_opened'] is False
    labels = {v['review_id']: v for v in read(private / 'INHERITED_VERDICTS.json') + frozen['labels']}
    rows = [dict(row, **{k: v for k, v in labels[row['review_id']].items() if k != 'review_id'})
            for row in read(private / 'MAPPING_SEALED.json')]
    parent = read(Path(previous) / 'SCORED_BINDINGS_PRIVATE.json')
    key = lambda r: (r['task_id'], r['query_id'])
    indexed = {arm: {key(r): r for r in values} for arm, values in [('A', parent), ('B', rows)]}
    assert len(rows) == len(parent) == len(indexed['B']) == 238
    assert indexed['A'].keys() == indexed['B'].keys()
    assert all(indexed['A'][key(r)]['kind'] == r['kind'] for r in rows)
    tasks = read(root / 'TASKS.json')['tasks']
    groups = [t['task_id'] for t in tasks]
    assert len(groups) == len(set(groups)) == 14
    base_known = {key(r): dict(core_correct=r['base_core'], full_supported=r['base_full'])
                  for r in read(private / 'IDENTITY_TRANSITIONS_PRIVATE.json')}
    base_challenge = {key(r): r for r in read(Path(base_previous) / 'SCORED_BINDINGS_PRIVATE.json')
                      if r['arm'] == 'BASE' and r['kind'] == 'protect_new'}
    assert len(base_known) == 84 and len(base_challenge) == 56
    supported = {k for k, r in base_challenge.items() if r['full_supported']}
    assert len(supported) == 47
    kinds = ('native', 'FIT', 'heldout', 'protect', 'protect_replay', 'protect_new')
    sets = dict(target98=('native', 'FIT', 'heldout'), known84=('protect', 'protect_replay'),
                native28=('protect',), challenge56=('protect_new',))
    summary, outcomes = {}, {}
    for arm, values in [('A', parent), ('B', rows)]:
        summary[arm] = {}
        for name, selected_kinds in list(sets.items()) + [(k, (k,)) for k in kinds]:
            selected = [r for r in values if r['kind'] in selected_kinds]
            summary[arm][name] = dict(n=len(selected), core=sum(r['core_correct'] for r in selected),
                full=sum(r['full_supported'] for r in selected),
                unsupported_expansion=sum(r['unsupported_expansion'] for r in selected),
                extra_contradiction=sum(r['extra_explicit_contradiction'] for r in selected),
                unknown_core=sum('UNKNOWN' in r['core_status'] or 'UNVERIFIED' in r['core_status']
                                 for r in selected))
        outcomes[arm] = []
        for task in groups:
            selected = [r for r in values if r['task_id'] == task]
            target = [r for r in selected if r['kind'] in ('native', 'heldout')]
            protected = [r for r in selected if r['kind'] == 'protect']
            base_ids = {k for k in supported if k[0] == task}
            damage = sum(not indexed[arm][k]['full_supported'] for k in base_ids)
            assert len(target) == 3 and len(protected) == len(next(t for t in tasks if t['task_id'] == task)['protect'])
            common = all(r['full_supported'] for r in protected) and bool(base_ids) and damage == 0
            group_values = dict(task_id=task, Base_support=len(base_ids), Base_damage=damage,
                joint=common and all(r['core_correct'] for r in target),
                complete_joint=common and all(r['full_supported'] for r in target))
            for name, selected_kinds in sets.items():
                subset = [r for r in selected if r['kind'] in selected_kinds]
                group_values[name + '_core_fraction'] = sum(r['core_correct'] for r in subset) / len(subset)
                group_values[name + '_full_fraction'] = sum(r['full_supported'] for r in subset) / len(subset)
            outcomes[arm].append(group_values)
        summary[arm].update(joint=sum(r['joint'] for r in outcomes[arm]),
            complete_joint=sum(r['complete_joint'] for r in outcomes[arm]),
            Base47_damage=sum(r['Base_damage'] for r in outcomes[arm]), n_groups=14)

    def transitions(before, after, keys, field):
        counts = dict(kept_correct=0, new_damage=0, repaired=0, remains_incorrect=0)
        identities = []
        for k in sorted(keys):
            a, b = bool(before[k][field]), bool(after[k][field])
            state = 'kept_correct' if a and b else 'new_damage' if a else 'repaired' if b else 'remains_incorrect'
            counts[state] += 1
            identities.append(dict(task_id=k[0], query_id=k[1], before=a, after=b, transition=state))
        assert sum(counts.values()) == len(keys)
        return dict(counts=counts, identities=identities)

    identity_transitions = {}
    for name, base in [('known84', base_known), ('challenge56', base_challenge)]:
        identity_transitions[name] = {}
        for pair, before, after in [('Base_to_A', base, indexed['A']), ('Base_to_B', base, indexed['B']),
                                     ('A_to_B', indexed['A'], indexed['B'])]:
            identity_transitions[name][pair] = {field: transitions(before, after, base.keys(), field)
                for field in ('core_correct', 'full_supported')}
    write(private / 'IDENTITY_TRANSITIONS_FINAL_PRIVATE.json', identity_transitions)
    transition_counts = {name: {pair: {field: value['counts'] for field, value in metrics.items()}
                         for pair, metrics in pairs.items()} for name, pairs in identity_transitions.items()}
    mechanism = read(root / 'PATH_MECHANISM_AUDIT_B_BASE_SEQUENCE.json')
    assert len(mechanism) == 28
    conditions = dict(target98_core=summary['B']['target98']['core'] == 98,
        target98_full=summary['B']['target98']['full'] == 98,
        known84_core=summary['B']['known84']['core'] == 84,
        native28_full=summary['B']['native28']['full'] == 28,
        native28_tokens_Base_exact=all(r['Base_tokens_exact'] for r in mechanism),
        no_Base47_damage=summary['B']['Base47_damage'] == 0,
        challenge_full_at_least49=summary['B']['challenge56']['full'] >= 49,
        core_joint_at_least13=summary['B']['joint'] >= 13,
        no_A_correct_core_damage=all(v['A_to_B']['core_correct']['new_damage'] == 0
                                     for v in transition_counts.values()),
        no_A_correct_full_damage=all(v['A_to_B']['full_supported']['new_damage'] == 0
                                     for v in transition_counts.values()))
    helper = Path(__file__).parents[1] / 'm13p_confirmation_prep_v107/review.py'
    spec = importlib.util.spec_from_file_location('v130_statistics', helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    comparisons = {}
    for name in ('target98_full_fraction', 'known84_core_fraction', 'known84_full_fraction',
                 'challenge56_full_fraction', 'joint', 'complete_joint', 'Base_damage'):
        a = [float(r[name]) for r in outcomes['A']]
        b = [float(r[name]) for r in outcomes['B']]
        result = module.paired(b, a, seed=13020261025)
        differences = [y - x for x, y in zip(a, b)]
        result['leave_one_group_out'] = [dict(excluded=task,
            mean_difference=(sum(differences) - differences[i]) / 13) for i, task in enumerate(groups)]
        result['positive_groups'] = sum(d > 0 for d in differences)
        result['negative_groups'] = sum(d < 0 for d in differences)
        result['nonzero_group_differences'] = [dict(task_id=t, difference=d)
                                               for t, d in zip(groups, differences) if d != 0]
        comparisons[name] = result
    generated = [read(root / task['task_id'] / arm / 'ZERO_SMOKE.json') for task in tasks
                 for arm in ('A_SHORT_REFERENCE', 'B_BASE_SEQUENCE')]
    generated += [read(path) for task in tasks for arm in ('A_SHORT_REFERENCE', 'B_BASE_SEQUENCE')
                  for path in (root / task['task_id'] / arm / 'outputs').glob('*.json')]
    assert len(generated) == 504
    base_generated = [read(path) for path in (root / 'Q').glob('*/Base_paths/*.json')]
    assert len(base_generated) == 22
    count = read(root / 'COMPLETE.json')['counts']
    assert count['generations'] == len(generated) + 22
    result = dict(status='COMPLETE_DEVELOPMENT_GATE_PASS' if all(conditions.values()) else 'COMPLETE_DEVELOPMENT_GATE_FAILED',
        advancement_pass=all(conditions.values()), advancement_conditions=conditions, summary=summary,
        group_outcomes=outcomes, transitions=transition_counts, comparisons=comparisons,
        counts=count, cost=read(root / 'GUARD_EXIT.json'),
        A_control_exact238=read(root / 'A_CONTROL_REPRODUCTION.json')['pass_all238'],
        review_packet=read(private / 'REVIEW_PACKET_RECEIPT.json'),
        native28_Base_token_exact=sum(r['Base_tokens_exact'] for r in mechanism),
        maximum_active_residual_norm=max(r['maximum_active_residual_norm'] for r in mechanism),
        generated_A_B=dict(n=504, EOS=sum(r['EOS'] for r in generated), at_cap=sum(r['at_cap'] for r in generated)),
        generated_new_total=dict(n=526, EOS=sum(r['EOS'] for r in generated + base_generated),
                                 at_cap=sum(r['at_cap'] for r in generated + base_generated)),
        scope='All14 exposed DEV source groups; same-assistant source-contract review, not independent clinical adjudication',
        decision='FREEZE_FULL146_PREPARATION' if all(conditions.values()) else 'CLOSE_FIXED_RECIPE_NO_LOCAL_VARIANTS_NO146_LAUNCH')
    write(private / 'SCORED_BINDINGS_PRIVATE.json', rows)
    write(private / 'RESULTS.json', result)
    return result

