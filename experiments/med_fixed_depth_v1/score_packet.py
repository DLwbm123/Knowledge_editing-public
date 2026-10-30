"""Freeze all generated inputs before reading permitted DEV references."""
import hashlib
import json
from pathlib import Path
import time
import extract_reference


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def write_new(path, value):
    path = Path(path)
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False)
    path.chmod(0o600)


def build(root, source):
    root = Path(root)
    if time.time() >= 1793447820 - 620:
        raise TimeoutError('Frozen deadline')
    for name in ('FIRST_PAIR_EVAL_COMPLETE.json', 'FIRST_EVAL_EXIT.json', 'FIRST_EVAL_GUARD_EXIT.json',
                 'BATCH_EVAL_GUARD_EXIT.json'):
        if not (root / name).is_file():
            raise ValueError('Prediction dependency incomplete')
    if json.loads((root / 'FIRST_PAIR_EVAL_COMPLETE.json').read_text())['engineering_status'] != 'PASS':
        raise ValueError('First-pair evaluation failed')
    if any(json.loads((root / name).read_text())['code'] != 0 for name in ('FIRST_EVAL_EXIT.json', 'FIRST_EVAL_GUARD_EXIT.json')):
        raise ValueError('First-pair evaluation exit failed')
    if json.loads((root / 'BATCH_EVAL_GUARD_EXIT.json').read_text())['codes'] != [0, 0, 0, 0]:
        raise ValueError('Batch evaluation guard failed')
    for i in range(4):
        if not (root / f'BATCH_EVAL_COMPLETE_{i}.json').is_file() or json.loads((root / f'BATCH_EVAL_EXIT_{i}.json').read_text())['code'] != 0:
            raise ValueError('Batch evaluation shard failed')
    lock = json.loads((root / 'PANEL_IDENTITY_LOCK.json').read_text())
    admission = json.loads((root / 'ROLE_ADMISSION.json').read_text())
    inputs = json.loads((root / 'EVAL_INPUTS_PRIVATE.json').read_text())
    expected = {(r['owner'], r['query']['query_id']): r['query'] for r in inputs}
    if len(expected) != 508:
        raise ValueError('Fixed input coverage mismatch')
    prior = {(r['owner'], r['query_id']): r for r in json.loads((root / 'PRIOR_INPUT_BINDINGS_PRIVATE.json').read_text())}
    output_files = list((root / 'first_eval_outputs').glob('*.json')) + list((root / 'batch_eval_outputs').glob('*.json'))
    if len(output_files) != 1016:
        raise ValueError('Prediction file coverage mismatch')
    predictions = {}
    for path in output_files:
        row = json.loads(path.read_text())
        key = (row['owner'], row['query_id'], row['arm'])
        if key in predictions or key[:2] not in expected or row['arm'] not in ('C21', 'CDEEP'):
            raise ValueError('Duplicate or unplanned prediction')
        q, old = expected[key[:2]], prior[key[:2]]
        if row['question'] != q['question'] or row['image_sha256'] != q['image_sha256']:
            raise ValueError('Question or image mismatch')
        if row['original_consumer_id'] != old['consumer_id'] or row['original_route'] != old['route']:
            raise ValueError('Original route binding mismatch')
        if any(row[k] != old['input'][k] for k in ('question', 'image_sha256', 'prompt_ids', 'attention_mask', 'generation')):
            raise ValueError('Original prompt or generation binding mismatch')
        if row['original_execution_binding'] != old['prediction_binding'] or not row['raw_token_ids']:
            raise ValueError('Original execution binding or answer invalid')
        predictions[key] = row
    if set(predictions) != {(o, q, a) for o, q in expected for a in ('C21', 'CDEEP')}:
        raise ValueError('Full prediction coverage incomplete')
    completion = dict(status='ALL_PREDICTIONS_FROZEN', events=len(expected), predictions=len(predictions), time=time.time())
    write_new(root / 'PREDICTIONS_FROZEN.json', completion)
    refs = extract_reference.extract(source, lock, admission, completion)
    references = {(r['owner'], r['query_id']): r['reference'] for r in refs}
    payloads, consumers = {}, []
    for (owner, qid, arm), row in sorted(predictions.items()):
        answer = dict(raw_answer=row['raw_answer'], raw_token_ids=row['raw_token_ids'])
        payload = dict(query_id=qid, question=row['question'], reference=references[(owner, qid)],
                       image_sha256=row['image_sha256'], prompt_ids=row['prompt_ids'], answer=answer)
        key = digest(payload)
        record = dict(opaque_query_id=key, question=row['question'], gold_answer=references[(owner, qid)],
                      raw_base_answer=row['raw_answer'])
        if key in payloads and payloads[key] != record:
            raise ValueError('Payload collision')
        payloads[key] = record
        consumers.append(dict(owner=owner, query_id=qid, arm=arm, key=key, eos=row['eos'], at_cap=row['at_cap'],
                              route=row['route'], prediction_reused_from=row['prediction_reused_from']))
    packet = dict(status='SCORE_PACKET_FROZEN', consumers=consumers, payloads=payloads,
                  source_events=len(expected), frozen_time=completion['time'], method_blind=True)
    write_new(root / 'SCORE_PACKET_PRIVATE.json', packet)
    return dict(status=packet['status'], consumers=len(consumers), unique_payloads=len(payloads),
                original_reference_decoded_only_after_prediction_freeze=True)
