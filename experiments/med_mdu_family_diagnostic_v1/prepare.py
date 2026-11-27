"""Build the read-only teacher-forcing panel from frozen local scoring artifacts."""
import json
from pathlib import Path

root = Path('.private')
summary = root / 'med_mdu150_t2g_supplement_v1_20261106/scoring'
audit = root / 't2g_frozen_semantic_audit_20261126'

def read(path):
    return json.loads(path.read_text())

baseline = read(summary / 'BASELINE.private.json')
roles = baseline['roles']
assert roles['status'] == 'EVALUATION_ROLES_FROZEN' and roles['owners'] == 150
scores = {**baseline['decisions'], **read(summary / 'DECISIONS.json')['decisions']}
keys = {(x['query_id'], x['arm']): x['key'] for x in baseline['consumers']}
packet = read(root / 'med150_balancedit_v1_20261028/scoring/PACKET_PRIVATE.json')
references = {
    x['query_id']: packet['payloads'][x['key']]
    for x in packet['consumers'] if x['mode'] == 'bank' and x['prefix'] == 150
}
validity = {r['q']: r for r in read(audit / 'labels.private.json')['labels']}
behavior = {(r['q'], r['arm']): r['class'] for r in read(audit / 'response_behavior.private.json')['labels']}
rows = []
for owner, parts in roles['roles'].items():
    for role in ('T2G', 'T2L'):
        for query in parts[role]:
            item = references[query]
            assert item['gold_answer'].strip()
            status = {}
            for arm in ('MED', 'MDU150'):
                key = keys[query, arm]
                status[arm] = {
                    'core': scores[key]['core_correct'],
                    'behavior': behavior.get((query, arm)),
                }
                assert type(status[arm]['core']) is bool
            rows.append({
                'owner': int(owner), 'query': query, 'role': role,
                'question': item['question'], 'reference': item['gold_answer'],
                'audit': validity[query]['judge'] if query in validity else None,
                'status': status,
            })
assert len(rows) == 625 and sum(r['role'] == 'T2G' for r in rows) == 571
print(json.dumps(rows, ensure_ascii=False))
