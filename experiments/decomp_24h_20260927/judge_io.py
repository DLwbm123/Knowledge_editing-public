"""Atomic, idempotent publication of already-scored responses; no Judge calls."""
import json,hashlib,fcntl,time
from pathlib import Path
from storage import Store

def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def publish(root,bid,scores,evidence,response):
 r=Path(root);store=Store(r)
 with (r/'JUDGE_PUBLISH.lock').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX);ledger=json.loads((r/'RESOURCE_LEDGER.json').read_text());attempt=next(a for a in ledger['judge_attempts'] if a['id']==bid)
  assert attempt['status'] in ['RESERVED','FORMAT_VALID'] and set(attempt['keys'])==set(scores)
  assert evidence['status']=='FORMAT_VALID' and not evidence.get('tool_event_types') and not evidence.get('errors')
  assert response['batch_id']==bid and len(response['decisions'])==len(scores)
  decisions={x['opaque_query_id']:x['is_correct'] for x in response['decisions']};assert len(decisions)==len(scores)
  for key,s in scores.items():
   pending=json.loads((r/'private/judge/pending'/f'{key}.json').read_text());record=pending['record']
   assert s['key']==key and type(s['is_correct']) is bool and s['payload_binding']==digest(record) and decisions[record['opaque_query_id']]==s['is_correct']
  intent=dict(batch_id=bid,scores=scores,evidence=evidence,response=response);rel=f'private/judge/success_intents/{bid}.json';path=r/rel
  if path.exists():assert json.loads(path.read_text())==intent
  else:store.write(rel,json.dumps(intent).encode(),exclusive=True,pin=True)
  for key,s in scores.items():
   rel=f'private/judge/scores/{key}.json';p=r/rel
   if p.exists():assert json.loads(p.read_text())==s,'Refuse conflicting score overwrite'
   else:store.write(rel,json.dumps(s).encode(),exclusive=True,pin=True)
  store.write(f'private/judge/evidence/{bid}.json',json.dumps(evidence).encode(),pin=True)
  with (r/'RESOURCE_LEDGER.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX);d=json.loads((r/'RESOURCE_LEDGER.json').read_text());a=next(x for x in d['judge_attempts'] if x['id']==bid);a.update(status='FORMAT_VALID',usage=evidence.get('usage'),published_epoch=time.time());store.write('RESOURCE_LEDGER.json',json.dumps(d).encode())
 return dict(published=len(scores),rejudge=False)
def recover(root):
 r=Path(root);done=[]
 for p in (r/'private/judge/success_intents').glob('*.json'):
  x=json.loads(p.read_text());d=json.loads((r/'RESOURCE_LEDGER.json').read_text());a=next(a for a in d['judge_attempts'] if a['id']==x['batch_id'])
  if a['status']=='RESERVED' or any(not (r/'private/judge/scores'/f'{k}.json').exists() for k in x['scores']):done.append(publish(r,x['batch_id'],x['scores'],x['evidence'],x['response']))
 return done

def quarantine(root,bid,evidence):
 r=Path(root);store=Store(r)
 with (r/'JUDGE_PUBLISH.lock').open('a') as publish_lock:
  fcntl.flock(publish_lock,fcntl.LOCK_EX)
  assert not (r/f'private/judge/success_intents/{bid}.json').exists(),'Recover successful intent first'
  with (r/'RESOURCE_LEDGER.lock').open('a') as f:
   fcntl.flock(f,fcntl.LOCK_EX);d=json.loads((r/'RESOURCE_LEDGER.json').read_text());a=next(x for x in d['judge_attempts'] if x['id']==bid)
   assert a['status'] in ['RESERVED','FAILED_NO_RETRY']
   assert not any((r/f'private/judge/scores/{k}.json').exists() for k in a['keys']),'Partial old publication requires successful response recovery'
   store.write(f'private/judge/evidence/{bid}.json',json.dumps(evidence).encode(),pin=True)
   p=r/'private/judge/JUDGE_MISSING_LOCK.json';missing=json.loads(p.read_text());missing['keys']=sorted(set(missing['keys'])|set(a['keys']));store.write(str(p.relative_to(r)),json.dumps(missing).encode(),pin=True)
   a.update(status='FAILED_NO_RETRY',missing_items=len(a['keys']));store.write('RESOURCE_LEDGER.json',json.dumps(d).encode())
 return dict(status='FAILED_NO_RETRY',keys=a['keys'])
