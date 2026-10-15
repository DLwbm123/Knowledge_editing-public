"""Preparation only: fixed training-source selection and isolated input-collision counts."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import unicodedata

QUOTAS = {'Organ': 4, 'Modality': 2, 'Plane': 1, 'Quantity': 1}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode()).hexdigest()


def normalized(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def select_source_only(pool, annotations):
    """No evaluation inputs, answers, correctness or damaged-query identity arguments."""
    selected = []
    groups = set()
    for kind, count in QUOTAS.items():
        candidates = [(i, row, annotations[(row['source_file'], str(row['source_qid']))])
                      for i, row in enumerate(pool)]
        chosen = 0
        for i, row, raw in candidates:
            if raw.get('content_type') != kind or row['source_group'] in groups:
                continue
            assert row['dataset'] == 'SLAKE' and row['source_role'] == 'train'
            assert Path(row['source_file']).name == 'train.json' and raw['q_lang'] == 'en'
            assert row['question'] == raw['question'] and row['reference'] == raw['answer']
            assert row['source_group'] == 'SLAKE:' + raw['img_name'].split('/')[0]
            selected.append((i, row, raw))
            groups.add(row['source_group'])
            chosen += 1
            if chosen == count:
                break
        assert chosen == count, 'Fixed quota unavailable; no replacement or evaluation filtering'
    return sorted(selected, key=lambda x: x[0])


def input_collisions(selected_inputs, comparison_inputs):
    """Independent checker accepts input identity only, never evaluation answers/scores."""
    allowed = {'question', 'image_sha256', 'source_group'}
    assert all(set(x) == allowed for x in selected_inputs + comparison_inputs)
    inputs = {(x['image_sha256'], normalized(x['question'])) for x in comparison_inputs}
    groups = {x['source_group'] for x in comparison_inputs}
    texts = {normalized(x['question']) for x in comparison_inputs}
    return dict(full_input=sum((x['image_sha256'], normalized(x['question'])) in inputs
                               for x in selected_inputs),
                source_groups=len({x['source_group'] for x in selected_inputs} & groups),
                text_only=sum(normalized(x['question']) in texts for x in selected_inputs))


def audit_source_availability(pool, annotations):
    selected = select_source_only(pool, annotations)
    assert len(pool) == 61 and len(selected) == 8
    assert all(Path(row['image_path']).is_file() for _, row, _ in selected)
    for _, row, _ in selected:
        with Path(row['image_path']).open('rb') as image:
            assert image.read(1), 'Unreadable or empty source image'
    bound = [{k: row[k] for k in ('dataset', 'source_qid', 'source_group', 'question',
                                 'reference', 'image_sha256')} for _, row, _ in selected]
    summary = dict(status='SOURCE_AVAILABLE_INPUT_COLLISION_CHECK_PENDING',
                   pool_questions=61, pool_source_groups=len({x['source_group'] for x in pool}),
                   selected_questions=8, selected_source_groups=8, selection_quotas=QUOTAS,
                   selection_rule='quota order, earliest existing BASIS order, distinct source groups; then original pool order',
                   selected_content_types=dict(Counter(raw['content_type'] for _, _, raw in selected)),
                   selected_modalities=dict(Counter(raw['modality'] for _, _, raw in selected)),
                   selected_locations=dict(Counter(raw['location'] for _, _, raw in selected)),
                   source_annotations_exactly_matched=8, image_files_readable=8,
                   source_role='SLAKE train annotations', KG_questions=0,
                   selected_binding=digest(bound),
                   historical_training_Base_correct_pool_selection_disclosed=True,
                   patient_independence='UNKNOWN', independent_clinician_confirmation=False,
                   native_Base_statistics='NOT_COMPUTED', new_model_calls=0, new_Judge=0,
                   new_GPU_updates=0, new_installations=0, clinical_capability_coverage='NOT_ESTABLISHED',
                   image_bytes_rehashed=False)
    locators = [dict(pool_index=i, source_file=row['source_file'], source_qid=row['source_qid'])
                for i, row, _ in selected]
    return selected, summary, locators


def selfcheck():
    a = dict(question=' Which organ? ', image_sha256='a', source_group='g1')
    b = dict(question='which ORGAN?', image_sha256='b', source_group='g2')
    assert input_collisions([a], [b]) == dict(full_input=0, source_groups=0, text_only=1)
    assert input_collisions([a], [dict(b, image_sha256='a')])['full_input'] == 1
    try:
        input_collisions([a], [dict(b, answer='never accepted')])
    except AssertionError:
        pass
    else:
        raise AssertionError('Evaluation answer field accepted')
    pool, annotations = [], {}
    for kind, count in QUOTAS.items():
        for _ in range(count):
            n = len(pool)
            row = dict(dataset='SLAKE', source_role='train', source_file='train.json',
                       source_qid=n, source_group=str(n), question=str(n), reference='source')
            pool.append(row)
            annotations['train.json', str(n)] = dict(content_type=kind, q_lang='en',
                                                     question=str(n), answer='source',
                                                     img_name=str(n)+'/source.jpg')
            row['source_group'] = 'SLAKE:' + str(n)
    assert len(select_source_only(pool, annotations)) == 8
    return dict(status='PASS', fixed_source_selection=True, answer_field_rejected=True,
                normalized_input_check=True, model_calls=0, GPU_calls=0)


if __name__ == '__main__':
    print(json.dumps(selfcheck()))
