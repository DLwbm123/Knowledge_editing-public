"""Quarantine failed transport requests; never resubmit their keys."""
import json


def transport_failure(evidence):
    errors=json.dumps(evidence.get('errors',[])).lower()
    return (evidence.get('status')=='FAILED_NO_RETRY'
            and not evidence.get('tool_event_types')
            and any(term in errors for term in ['connection failed','error sending request','sse idle timeout','stream disconnected']))


def quarantine(local_root, remote_root, remote, batch, rows, evidence):
    keys=[r['key'] for r in rows]
    bid=batch['batch_id']
    # Commit the shared missing ledger first; a partial local sync is fail-closed.
    remote('''from pathlib import Path
import json,fcntl,os
r=Path(ROOT);keys=KEYS;bid=BID;ev=EVIDENCE
with (r/'RESOURCE_LEDGER.lock').open('a') as f:
 fcntl.flock(f,fcntl.LOCK_EX)
 p=r/'RESOURCE_LEDGER.json';d=json.loads(p.read_text());a=next(x for x in d['judge_attempts'] if x['id']==bid)
 assert a['items']==len(keys) and a['status']=='RESERVED'
 a.update(status='FAILED_NO_RETRY',missing_items=len(keys))
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));os.replace(t,p)
 p=r/'private/judge/JUDGE_MISSING_LOCK.json';d=json.loads(p.read_text());d['keys']=sorted(set(d['keys'])|set(keys))
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));os.replace(t,p)
 (r/'private/judge/evidence'/f'{bid}.json').write_text(json.dumps(ev,indent=2))
'''.replace('ROOT',repr(remote_root)).replace('KEYS',repr(keys)).replace('BID',repr(bid)).replace('EVIDENCE',repr(evidence)))
    p=local_root/'blocked_keys.json';d=json.loads(p.read_text());d['keys']=sorted(set(d['keys'])|set(keys));p.write_text(json.dumps(d,indent=2))


if __name__=='__main__':
    e=dict(status='FAILED_NO_RETRY',errors=[{'message':'Connection failed: error sending request'}],tool_event_types=[])
    assert transport_failure(e)
    assert not transport_failure(dict(e,tool_event_types=['command_execution']))
    assert not transport_failure(dict(e,errors=[{'message':'invalid response format'}]))
    assert not transport_failure(dict(e,status='FORMAT_VALID'))
    print('PASS: only failed transport calls can be quarantined; tool/schema failures stop')
