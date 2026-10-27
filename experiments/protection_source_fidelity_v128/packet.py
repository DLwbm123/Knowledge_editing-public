"""Source-question equivalence packet, independent of model outputs."""
import json
from pathlib import Path
import random
import time


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def build(task_path, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    started = time.time()
    triplets = []
    for task in read(task_path)['tasks']:
        for original in task['protect']:
            paraphrases = [p for p in task['protect_extra']
                if p['query_id'].startswith(original['query_id'] + '_PARA')]
            assert len(paraphrases) == 2
            assert len({p['query_id'] for p in paraphrases}) == 2
            for row in paraphrases:
                for name in ('reference', 'image_path'):
                    assert row[name] == original[name]
            triplets.append(dict(task_id=task['task_id'], original=original, paraphrases=paraphrases))
    assert len(triplets) == 28
    assert len({(t['task_id'], t['original']['query_id']) for t in triplets}) == 28
    random.Random(12820261025).shuffle(triplets)
    packet, mappings = [], []
    for index, triplet in enumerate(triplets, 1):
        review_id = f'F{index:03d}'
        packet.append(dict(review_id=review_id, original_question=triplet['original']['question'],
            reference=triplet['original']['reference'], paraphrases=[dict(expression=j,
                question=p['question']) for j, p in enumerate(triplet['paraphrases'], 1)]))
        mappings.append(dict(review_id=review_id, task_id=triplet['task_id'],
            original_query_id=triplet['original']['query_id'], paraphrases=[dict(expression=j,
                query_id=p['query_id']) for j, p in enumerate(triplet['paraphrases'], 1)]))
    write(destination / 'BLIND_PACKET.json', packet)
    write(destination / 'MAPPING_SEALED.json', mappings)
    receipt = dict(time=time.time(), triplets=28, original_identity_support=28,
        expression_comparisons=56, new_model_forwards=0, updates=0, generations=0,
        external_judge_calls=0, seconds=time.time() - started)
    write(destination / 'PACKET_RECEIPT.json', receipt)
    return receipt


def summarize(destination, previous_training_labels):
    destination = Path(destination)
    labels = read(destination / 'VERDICTS_FROZEN.json')
    assert labels['mapping_opened'] is False and len(labels['labels']) == 56
    lookup = {(r['review_id'], r['expression']): r for r in labels['labels']}
    assert len(lookup) == 56
    old = {(r['task_id'], r['query_id']): r['core_correct'] for r in read(previous_training_labels)
           if r['kind'] in ('protect', 'protect_replay')}
    assert len(old) == 84
    rows = [dict(task_id=t['task_id'], query_id=p['query_id'],
        old_training_core_correct=old[(t['task_id'], p['query_id'])],
        **lookup[(t['review_id'], p['expression'])])
        for t in read(destination / 'MAPPING_SEALED.json') for p in t['paraphrases']]
    result = dict(comparisons=56, original_identity_support=28, old_results_changed=False,
                  old_training_denominator=84, new_model_forwards=0, updates=0, generations=0,
                  external_judge_calls=0, same_assistant_not_independent=True)
    result['summary'] = {name: {label: sum(r['status'] == label for r in selected)
        for label in ('EQUIVALENT', 'SCOPE_CHANGED', 'UNDETERMINED')}
        for name, selected in [('all56', rows), ('old_core_correct', [r for r in rows if r['old_training_core_correct']]),
                                ('old_core_incorrect', [r for r in rows if not r['old_training_core_correct']])]}
    write(destination / 'SCORED_PRIVATE.json', rows)
    write(destination / 'RESULTS.json', result)
    return result
