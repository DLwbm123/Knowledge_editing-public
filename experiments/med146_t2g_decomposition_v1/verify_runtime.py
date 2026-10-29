"""Check matching saved input and runtime bindings; return aggregate counts only."""
import hashlib
import json
import os
from pathlib import Path
import signal

signal.alarm(120)
root = Path(os.environ['AUDIT_SOURCE'])
f = json.loads((root / 'INPUTS_FROZEN.json').read_text())
by = {}
for row in f['plan']:
    if row['mode'] == 'single' or (row['mode'] == 'bank' and row['prefix'] == 146):
        key = (row['mode'], row['owner'] if row['mode'] == 'single' else 0, row['query']['query_id'])
        assert key not in by
        by[key] = row

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()

def load(row):
    cid = json.dumps([row[k] for k in ('method', 'mode', 'prefix', 'owner')] + [row['query']['query_id']], separators=(',', ':'))
    return json.loads((root / 'outputs' / (digest(cid) + '.json')).read_text())

count = mismatch = invalid = 0
for task in f['tasks']:
    ids = dict.fromkeys(q for event in task['events'] if event['task'] == 'T2G' for q in event['all_probe_query_ids'])
    for q in ids:
        if f['base_mask'][q] is not False:
            continue
        try:
            one = load(by[('single', task['order'], q)])
            bank = load(by[('bank', 0, q)])
            x, y = one['binding']['judge_input'], bank['binding']['judge_input']
            fields = ('question', 'reference', 'image_sha256', 'prompt_ids', 'attention_mask', 'runtime', 'generation')
            a, b = one['prediction_binding'], bank['prediction_binding']
            prediction = ('input', 'model_source', 'execution_source', 'deterministic_greedy', 'generation_rng_binding')
            mismatch += not (all(x[k] == y[k] for k in fields) and all(a[k] == b[k] for k in prediction))
            count += 1
        except (OSError, ValueError, KeyError, TypeError):
            invalid += 1
print(json.dumps({'paired_events': count, 'complete_binding_mismatch': mismatch, 'missing_or_invalid': invalid,
                  'expected_events': 557, 'verdict': 'PASS' if count == 557 and mismatch == invalid == 0 else 'FAIL'}))
