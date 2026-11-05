"""Build the score packet only after all selected predictions are frozen."""
import json
from pathlib import Path
import prior_binding
import stream_extract as stream
import selected_records
import hashlib

CONTRACT = 'MEDPROP_SMALL4_CORE_FULL_SOURCE_V1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def build(root, old150, selected_ids):
    root, old150 = Path(root), Path(old150)
    wanted = set(selected_ids)
    if len(wanted) != 635 or len(selected_ids) != 635:
        raise ValueError('selected event lock differs')
    frozen = json.loads((root / 'PREDICTIONS_FROZEN.json').read_text())
    if frozen.get('status') != 'ALL_PREDICTIONS_FROZEN' or frozen.get('events') != 2148:
        raise ValueError('all bank predictions must freeze first')
    if json.loads((root / 'EVAL_GUARD_EXIT.json').read_text()).get('code') != 0:
        raise ValueError('evaluation guard failed')
    inputs = selected_records.read_selected(root / 'BANK_INPUTS_PRIVATE.json', 'records',
                                             {'consumer_id': True}, lambda r: r['consumer_id'], wanted)
    fields = {'consumer_id': True, 'binding': {'input': {'query_id': True, 'question': True,
              'image_sha256': True, 'reference': True}}, 'R0': {'raw_answer': True, 'raw_token_ids': True}}
    records, consumers = {}, []
    for cid, item in inputs.items():
        key = hashlib.sha256(cid.encode()).hexdigest()
        pred = json.loads((root / 'predictions' / (key + '.json')).read_text())
        if pred.get('consumer_id') != cid or pred.get('query_id') != item['query']['query_id']:
            raise ValueError('prediction identity differs')
        if not pred.get('candidate') or pred.get('original_route') != item['route']:
            raise ValueError('candidate routing differs')
        if any(pred.get(k) != item['judge_input'][k] for k in
               ('question', 'image_sha256', 'prompt_ids', 'attention_mask', 'generation')):
            raise ValueError('candidate input differs')
        filename = item['original_filename']
        if Path(filename).name != filename:
            raise ValueError('unsafe original filename')
        with (old150 / 'eval_outputs' / filename).open('rb') as f:
            old = prior_binding.select(f, stream.nonspace(f), fields)
        q, oq = item['query'], old.get('binding', {}).get('input', {})
        if old.get('consumer_id') != cid or any(oq.get(k) != q[k] for k in ('query_id', 'question', 'image_sha256')):
            raise ValueError('original input differs')
        if not isinstance(oq.get('reference'), str):
            raise ValueError('missing authorized reference')
        for arm, answer in (('MED', old.get('R0', {})), ('CANDIDATE', pred)):
            if not isinstance(answer.get('raw_answer'), str) or not answer.get('raw_token_ids'):
                raise ValueError('invalid answer')
            semantic = {'protocol': CONTRACT, 'model': 'gpt-6-astra', 'reasoning': 'medium',
                        'question': q['question'], 'gold_answer': oq['reference'],
                        'raw_base_answer': answer['raw_answer']}
            semantic_key = digest(semantic)
            legacy = {'query_id': q['query_id'], 'question': q['question'], 'reference': oq['reference'],
                      'image_sha256': q['image_sha256'], 'prompt_ids': item['judge_input']['prompt_ids'],
                      'answer': {'raw_answer': answer['raw_answer'], 'raw_token_ids': answer['raw_token_ids']}}
            record = {'opaque_query_id': semantic_key, 'question': q['question'],
                      'gold_answer': oq['reference'], 'raw_base_answer': answer['raw_answer']}
            entry = records.setdefault(semantic_key, {'record': record, 'legacy_keys': []})
            if entry['record'] != record:
                raise ValueError('semantic collision')
            entry['legacy_keys'].append(digest(legacy))
            consumers.append({'consumer_id': cid, 'query_id': q['query_id'], 'arm': arm,
                              'key': semantic_key, 'owner': item['owner'], 'role': q['audit_role'],
                              'route': item['route'], 'candidate_expert': True})
    if len(consumers) != 1270:
        raise ValueError('selected consumer count differs')
    result = {'status': 'SCORE_PACKET_FROZEN', 'events': 635, 'consumers': consumers,
              'payloads': records, 'new_judge_authorized': False}
    path = root / 'SCORE_PACKET_PRIVATE.json'
    with path.open('x') as f:
        json.dump(result, f, ensure_ascii=False)
    path.chmod(0o600)
    return {'status': result['status'], 'events': 635, 'unique_payloads': len(records)}
