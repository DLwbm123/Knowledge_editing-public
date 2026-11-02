import json,os
from pathlib import Path
old=Path('/private/tmp/.r.20261102.07');root=Path('/private/tmp/.r.20261102.08')
m=json.loads((old/'BATCH_MANIFEST.json').read_text());lock=json.loads((old/'SUPPLEMENT_INPUT.private.json').read_text())
assert json.loads((old/'WORKER_EXIT.json').read_text())['code']==0
assert json.loads((old/'GUARD_EXIT.json').read_text())['code']==0
ids=[]
for b in m['batches']:
 op=old/'batches'/b['batch_id']/'operator'
 if not (op/'ATTEMPT.json').exists():
  assert (op/'FAILURE_NO_RETRY.json').is_file() and m['batch_categories'][b['batch_id']]=='history_unverified'
  ids.extend(x['opaque_query_id'] for x in b['records'])
assert len(ids)==14 and len(set(ids))==14 and set(ids)<=set(lock['categories']['history_unverified'])
root.mkdir(mode=0o700,exist_ok=False)
f=root/'SUPPLEMENT_INPUT.private.json';f.write_text(json.dumps({'status':'SUPPLEMENT_IDENTITIES_FROZEN','categories':{'history_unverified':sorted(ids),'conflict_unknown':[]},'payloads':{k:lock['payloads'][k] for k in ids}},ensure_ascii=False));f.chmod(0o600)
print('unattempted',len(ids))
