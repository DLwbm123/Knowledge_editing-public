"""Freeze the four-point panel before opening the admitted reference slice."""
import hashlib
import json
from pathlib import Path
import time

GAMMAS = (.25, .5, .75, 1.)
CONTRACT = 'MEDPROP_SMALL4_CORE_FULL_SOURCE_V1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def build(root, fixed, source, extract_reference):
    root, fixed = Path(root), Path(fixed)
    if json.loads((root/'WORKER_EXIT.json').read_text())['code'] != 0:
        raise ValueError('Worker failed')
    if json.loads((root/'GUARD_EXIT.json').read_text())['code'] != 0:
        raise ValueError('Guard failed')
    completion = json.loads((root/'EVAL_COMPLETE.json').read_text())
    if completion.get('status') != 'GENERATION_COMPLETE' or completion.get('predictions') != 2032:
        raise ValueError('Prediction completion missing')
    inputs = json.loads((fixed/'EVAL_INPUTS_PRIVATE.json').read_text())
    prior_rows = json.loads((fixed/'PRIOR_INPUT_BINDINGS_PRIVATE.json').read_text())
    expected = {(x['owner'], x['query']['query_id']): x['query'] for x in inputs}
    prior = {(x['owner'], x['query_id']): x for x in prior_rows}
    if len(expected) != 508 or len(inputs) != 508 or set(prior) != set(expected) or len(prior_rows) != 508:
        raise ValueError('Input coverage mismatch')
    files = list((root/'predictions').glob('*.json'))
    if len(files) != 2032:
        raise ValueError('Prediction file coverage mismatch')
    predictions = {}
    for path in files:
        row = json.loads(path.read_text())
        key = (row['owner'], row['query_id'], row['gamma'])
        if key in predictions or key[:2] not in expected or row['gamma'] not in GAMMAS or row['arm'] != 'DEEP_LORA_R4':
            raise ValueError('Unexpected prediction')
        q, old = expected[key[:2]], prior[key[:2]]
        route = old['route']
        normalized_route = dict(activated=route['activated'], logical_edit_id=route['logical_edit_id'] if route['activated'] else None)
        if (row['route'] != normalized_route or row['original_route'] != route or row['original_consumer_id'] != old['consumer_id']
            or row['original_execution_binding'] != old['prediction_binding'] or not row['raw_token_ids']
            or row['question'] != q['question'] or row['image_sha256'] != q['image_sha256']
            or any(row[k] != old['input'][k] for k in ('question','image_sha256','prompt_ids','attention_mask','generation'))):
            raise ValueError('Original input or route binding differs')
        predictions[key] = row
    if set(predictions) != {(o, q, g) for o, q in expected for g in GAMMAS}:
        raise ValueError('Full panel incomplete')
    frozen = dict(status='ALL_PREDICTIONS_FROZEN', events=508, predictions=2032, time=time.time())
    with (root/'PREDICTIONS_FROZEN.json').open('x') as f:
        json.dump(frozen, f)
    lock = json.loads((fixed/'PANEL_IDENTITY_LOCK.json').read_text())
    admission = json.loads((fixed/'ROLE_ADMISSION.json').read_text())
    refs = extract_reference.extract(source, lock, admission, frozen)
    references = {(x['owner'], x['query_id']): x['reference'] for x in refs}
    if len(refs) != 508 or set(references) != set(expected):
        raise ValueError('Reference coverage mismatch')
    payloads, consumers = {}, []
    for (owner, qid, gamma), row in sorted(predictions.items()):
        reference = references[(owner, qid)]
        # The scorer sees these three semantic fields; method and token metadata are not a score identity.
        semantic = dict(protocol=CONTRACT, model='gpt-6-astra', reasoning='medium',
                        question=row['question'], gold_answer=reference, raw_base_answer=row['raw_answer'])
        key = digest(semantic)
        legacy = dict(query_id=qid, question=row['question'], reference=reference, image_sha256=row['image_sha256'],
                      prompt_ids=row['prompt_ids'], answer=dict(raw_answer=row['raw_answer'], raw_token_ids=row['raw_token_ids']))
        record = dict(opaque_query_id=key, question=row['question'], gold_answer=reference, raw_base_answer=row['raw_answer'])
        if key not in payloads:
            payloads[key] = dict(record=record, legacy_keys=[])
        if payloads[key]['record'] != record:
            raise ValueError('Scoring identity collision')
        payloads[key]['legacy_keys'].append(digest(legacy))
        consumers.append(dict(owner=owner, query_id=qid, gamma=gamma, key=key, eos=row['eos'],
                              at_cap=row['at_cap'], empty=not row['raw_answer'].strip(), route=row['route']))
    result = dict(status='SCORE_PACKET_FROZEN', consumers=consumers, payloads=payloads,
                  source_events=508, predictions=2032, method_blind=True)
    with (root/'SCORE_PACKET_PRIVATE.json').open('x') as f:
        json.dump(result, f, ensure_ascii=False)
    (root/'SCORE_PACKET_PRIVATE.json').chmod(0o600)
    return dict(status=result['status'], consumers=len(consumers), unique_payloads=len(payloads))
