"""Build the single-epoch packet after a separate role admission."""
import hashlib
import json
import os
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def gather(stage):
    assert stage == 'A'
    root, parent, original = (Path(os.environ[k]) for k in
                               ('EXPERIMENT_ROOT', 'PARENT_RUN', 'REFERENCE_MED_RUN'))
    assert read(root / 'ROLE_ADMISSION.json')['status'] == 'PASS'
    assert (root / 'STAGE_A_GENERATION_COMPLETE.json').is_file()
    f = read(root / 'INPUTS_FROZEN.json')
    lock = read(root / 'PROTOCOL_LOCK.json')
    owners = (47, 117, 93, 87)
    plan = [v for v in f['plan'] if v['mode'] == 'single' and v['owner'] in owners]
    assert len(plan) == lock['single_consumers']
    tasks = [v for v in f['tasks'] if v['order'] in owners]
    assert {v['order'] for v in tasks} == set(owners)
    consumers = []
    fields = ('question', 'reference', 'image_sha256', 'prompt_ids', 'attention_mask', 'runtime', 'generation')
    for row in plan:
        cid = json.dumps([row[k] for k in ('method', 'mode', 'prefix', 'owner')] +
                         [row['query']['query_id']], separators=(',', ':'))
        candidate = read(root / 'outputs' / (digest(cid) + '.json'))
        original_med = read(original / 'outputs' / (digest(cid) + '.json'))
        qid = row['query']['query_id']
        base = read(parent / 'private/outputs/BASE_base_p0/0' / (digest(qid) + '.json'))
        for arm, value in (('MED', candidate), ('LEGACY', original_med), ('BASE', base)):
            binding, reference = value['binding']['judge_input'], candidate['binding']['judge_input']
            assert all(binding[k] == reference[k] for k in fields)
            assert binding['question'] == row['query']['question'] and binding['reference'] == row['query']['reference']
            answer = value['R0']
            assert answer['raw_token_ids']
            payload = dict(query_id=qid, question=binding['question'], reference=binding['reference'],
                           image_sha256=binding['image_sha256'], prompt_ids=binding['prompt_ids'], answer=answer)
            consumers.append(dict(arm=arm, mode=row['mode'], prefix=row['prefix'], owner=row['owner'],
                query_id=qid, payload_key=digest(payload), question=binding['question'], reference=binding['reference'],
                image_sha256=binding['image_sha256'], answer=answer, EOS=answer['raw_token_ids'][-1] == 2,
                at_cap=len(answer['raw_token_ids']) >= 1024, active_target=candidate['active_target'],
                effective_expert=value.get('effective_expert'), prediction_reused=value.get('prediction_reused', False)))
    assert len(consumers) == 3 * len(plan)
    return dict(tasks=tasks, base_mask=f['base_mask'], consumers=consumers, clock=read(root/'CLOCK.json'),
                stage=stage, input_binding=lock['input_binding'])
