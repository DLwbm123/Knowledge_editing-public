"""Inherit only exact frozen semantic bindings; blind-review every new full answer."""
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


def build(private, previous):
    private, previous = Path(private), Path(previous)
    root = private / 'completed'
    tasks = read(root / 'TASKS.json')['tasks']
    assert read(root / 'WORKER_EXIT.json')['code'] == read(root / 'GUARD_EXIT.json')['code'] == 0
    assert len(read(root / 'ALL_TRAINING_FROZEN.json')['selected_steps']) == 14
    verdicts = {r['review_id']: r for r in read(previous / 'V107_BLIND_VERDICTS_FROZEN.json')['labels']}
    library = {key(r): verdicts[r['review_id']] for r in read(previous / 'V107_BLIND_PACKET.json')}
    bindings, novel, inherited = [], {}, {}
    outputs = []
    for task in tasks:
        dest = root / task['task_id']
        train = read(dest / 'TRAIN_COMPLETE.json')
        assert train['save_load_exact'] and train['P_prefix_source_NLL_exact'] and train['source_only_selection']
        assert read(dest / 'GENERATION_COMPLETE.json')['rows'] == len(task['train'] + task['heldout'] + task['protect'])
        for kind, rows in (('native', [task['target']]), ('FIT', task['train'][1:]),
                           ('heldout', task['heldout']), ('protect', task['protect'])):
            for row in rows:
                answer = read(dest / 'outputs' / (row['query_id'] + '.json'))
                for name in ('question', 'reference', 'image_path'):
                    assert answer['binding'][name] == row[name]
                value = dict(source_group=task['source_group'], question=row['question'],
                             reference=row['reference'], full_answer=answer['raw_answer'])
                identity = key(value)
                if identity in library:
                    inherited[identity] = library[identity]
                else:
                    novel[identity] = value
                bindings.append(dict(task_id=task['task_id'], role=task['role'], kind=kind,
                                     query_id=row['query_id'], key=identity))
                outputs.append(answer)
    assert len(bindings) == len(outputs) == 126
    shuffled = list(novel)
    random.Random(10820261025).shuffle(shuffled)
    ids = {identity: f'V{index:04d}' for index, identity in enumerate(shuffled, 1)}
    known = {identity: f'I{index:04d}' for index, identity in enumerate(inherited, 1)}
    for row in bindings:
        identity = row.pop('key')
        row['review_id'] = ids[identity] if identity in ids else known[identity]
    write(private / 'BLIND_PACKET.json', [dict(review_id=ids[k], **novel[k]) for k in shuffled])
    write(private / 'INHERITED_VERDICTS.json', [dict(inherited[k], review_id=known[k],
          inheritance='exact source group, question, reference and full answer') for k in inherited])
    write(private / 'MAPPING_SEALED.json', bindings)
    counts = dict(bindings=126, unique_new_answers=len(novel), unique_inherited_answers=len(inherited),
                  inherited_bindings=sum(r['review_id'].startswith('I') for r in bindings),
                  EOS=sum(r['EOS'] for r in outputs), at_cap=sum(r['at_cap'] for r in outputs))
    write(private / 'REVIEW_PACKET_RECEIPT.json', counts)
    return counts


def summarize(private, previous):
    private, previous = Path(private), Path(previous)
    frozen = read(private / 'BLIND_VERDICTS_FROZEN.json')
    assert frozen['method_mapping_opened'] is False
    labels = {v['review_id']: v for v in read(private / 'INHERITED_VERDICTS.json') + frozen['labels']}
    rows = [dict(r, **{k: v for k, v in labels[r['review_id']].items() if k != 'review_id'})
            for r in read(private / 'MAPPING_SEALED.json')]
    old = read(previous / 'V107_SCORED_BINDINGS_PRIVATE.json')
    assert len(rows) == 126 and len(old) == 448
    sets = dict(P_SOURCE_STOP=rows)
    sets.update({a: [r for r in old if r['arm'] == a] for a in ('M13B', 'M13P', 'S')})
    groups, summary = {}, {}
    for arm, values in sets.items():
        group = []
        for task in sorted({v['task_id'] for v in values}):
            selected = [v for v in values if v['task_id'] == task]
            targets = [v for v in selected if v['kind'] in ('native', 'heldout')]
            protect = [v for v in selected if v['kind'] == 'protect']
            assert len(targets) == 3 and protect
            robust = all(v['core_correct'] for v in targets)
            retained = all(v['full_supported'] for v in protect)
            group.append(dict(task_id=task, target_robust=robust, protect_all_full=retained,
                              joint=robust and retained))
        groups[arm] = group
        summary[arm] = {kind: dict(n=len([r for r in values if r['kind'] == kind]),
            core=sum(r['core_correct'] for r in values if r['kind'] == kind),
            full=sum(r['full_supported'] for r in values if r['kind'] == kind),
            extra_contradiction=sum(r['extra_explicit_contradiction'] for r in values if r['kind'] == kind),
            unsupported_expansion=sum(r['unsupported_expansion'] for r in values if r['kind'] == kind))
            for kind in ('native', 'FIT', 'heldout', 'protect')}
        summary[arm].update(joint=sum(r['joint'] for r in group), n_groups=14)
    helper = Path(__file__).parents[1] / 'm13p_confirmation_prep_v107/review.py'
    spec = importlib.util.spec_from_file_location('paired_statistics', helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    comparisons = {arm: module.paired([int(v['joint']) for v in groups['P_SOURCE_STOP']],
        [int(v['joint']) for v in groups[arm]], seed=10820261025) for arm in ('M13P', 'M13B', 'S')}
    damage = {arm: {(r['task_id'], r['query_id']) for r in values
              if r['kind'] == 'protect' and not r['full_supported']} for arm, values in sets.items()}
    train = read(private / 'completed/ALL_TRAINING_FROZEN.json')
    gate = dict(all_98_source_held_core=all(r['core_correct'] for r in rows if r['kind'] != 'protect'),
        protect_at_least_B=summary['P_SOURCE_STOP']['protect']['full'] >= 27,
        no_new_B_damage=not (damage['P_SOURCE_STOP'] - damage['M13B']),
        joint_at_least_B=summary['P_SOURCE_STOP']['joint'] >= 13,
        fewer_updates=train['counts']['updates'] < 2240)
    result = dict(summary=summary, group_outcomes=groups, comparisons=comparisons,
        advancement_conditions=gate, advancement_pass=all(gate.values()),
        damage_counts={k: len(v) for k, v in damage.items()},
        P_new_damage=len(damage['P_SOURCE_STOP'] - damage['M13P']),
        P_restored=len(damage['M13P'] - damage['P_SOURCE_STOP']),
        B_new_damage=len(damage['P_SOURCE_STOP'] - damage['M13B']),
        B_restored=len(damage['M13B'] - damage['P_SOURCE_STOP']),
        selected_steps=train['selected_steps'], counts=read(private / 'completed/COMPLETE.json')['counts'],
        cost=read(private / 'completed/GUARD_EXIT.json'), scope='All14 exposed development groups')
    write(private / 'RESULTS.json', result)
    write(private / 'SCORED_BINDINGS_PRIVATE.json', rows)
    return result
