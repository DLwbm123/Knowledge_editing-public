"""Freeze replay predictions before opening approved development references.

The orchestration layer must independently validate execution receipts and
provide the original fixed-panel inputs, routes and prior execution bindings.
"""
import hashlib
import json
from pathlib import Path
import time


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def write_new(path, value):
    path = Path(path)
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False)
    path.chmod(0o600)


def freeze(root, inputs, prior_bindings, prediction_files, completion, expected_events=508):
    root = Path(root)
    if completion.get('status') != 'GENERATION_COMPLETE' or completion.get('events') != expected_events:
        raise ValueError('Generation dependency is incomplete')
    expected = {(r['owner'], r['query']['query_id']): r['query'] for r in inputs}
    prior = {(r['owner'], r['query_id']): r for r in prior_bindings}
    if len(inputs) != expected_events or len(expected) != expected_events or set(prior) != set(expected):
        raise ValueError('Original input binding coverage mismatch')
    if len(prior_bindings) != expected_events or len(prediction_files) != expected_events:
        raise ValueError('Prior or prediction count mismatch')
    predictions = {}
    for path in prediction_files:
        row = json.loads(Path(path).read_text())
        key = (row['owner'], row['query_id'])
        if key in predictions or key not in expected or row['arm'] != 'REPLAY':
            raise ValueError('Duplicate or unplanned prediction')
        query, old = expected[key], prior[key]
        if row['question'] != query['question'] or row['image_sha256'] != query['image_sha256']:
            raise ValueError('Question or image mismatch')
        if row['original_consumer_id'] != old['consumer_id'] or row['original_route'] != old['route']:
            raise ValueError('Original route or consumer mismatch')
        original_route = old['route']
        route = dict(activated=original_route['activated'],
                     logical_edit_id=original_route['logical_edit_id'] if original_route['activated'] else None)
        if row['route'] != route:
            raise ValueError('Candidate changed original route')
        if any(row[k] != old['input'][k] for k in ('question','image_sha256','prompt_ids','attention_mask','generation')):
            raise ValueError('Prompt or generation mismatch')
        if row['original_execution_binding'] != old['prediction_binding'] or not row['raw_token_ids']:
            raise ValueError('Execution binding or answer missing')
        if not row['route']['activated'] and not row['prediction_reused_from']:
            raise ValueError('OFF event must reuse original Base prediction')
        predictions[key] = row
    if set(predictions) != set(expected):
        raise ValueError('Prediction coverage incomplete')
    frozen = dict(status='ALL_PREDICTIONS_FROZEN', events=expected_events,
                  predictions=len(predictions), time=time.time())
    write_new(root/'PREDICTIONS_FROZEN.json', frozen)
    return predictions, frozen


def build(root, predictions, frozen, source, lock, admission, reference_extractor):
    root = Path(root)
    receipt = json.loads((root/'PREDICTIONS_FROZEN.json').read_text())
    if receipt != frozen or frozen.get('status') != 'ALL_PREDICTIONS_FROZEN':
        raise ValueError('Missing persisted prediction freeze')
    if len(predictions) != frozen['events']:
        raise ValueError('Frozen prediction count mismatch')
    # This is the first point at which the approved reference source is opened.
    refs = reference_extractor.extract(source, lock, admission, frozen)
    references = {(r['owner'], r['query_id']): r['reference'] for r in refs}
    if len(refs) != frozen['events'] or set(references) != set(predictions):
        raise ValueError('Reference coverage mismatch')
    payloads, consumers = {}, []
    for (owner, qid), row in sorted(predictions.items()):
        answer = dict(raw_answer=row['raw_answer'], raw_token_ids=row['raw_token_ids'])
        payload = dict(query_id=qid, question=row['question'], reference=references[(owner,qid)],
                       image_sha256=row['image_sha256'], prompt_ids=row['prompt_ids'], answer=answer)
        key = digest(payload)
        record = dict(opaque_query_id=key, question=row['question'], gold_answer=references[(owner,qid)],
                      raw_base_answer=row['raw_answer'])
        if key in payloads and payloads[key] != record:
            raise ValueError('Payload collision')
        payloads[key] = record
        consumers.append(dict(owner=owner,query_id=qid,arm='REPLAY',key=key,eos=row['eos'],at_cap=row['at_cap'],
                              empty=not row['raw_answer'].strip(),route=row['route'],
                              prediction_reused_from=row['prediction_reused_from']))
    packet = dict(status='SCORE_PACKET_FROZEN',consumers=consumers,payloads=payloads,
                  source_events=len(predictions),frozen_time=frozen['time'],method_blind=True)
    write_new(root/'SCORE_PACKET_PRIVATE.json',packet)
    return dict(status=packet['status'],consumers=len(consumers),unique_payloads=len(payloads))
