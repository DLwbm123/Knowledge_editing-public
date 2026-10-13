"""Qwen amendment: retain full-key queue, consumer bindings and one-attempt ledger."""
import json
import os
from pathlib import Path
import sys
import time

RUN=Path(os.environ['RUN_ROOT']);sys.path.insert(0,str(RUN/'private/tools'))
import judge_queue as h
q=h.q
from audit import read,write,digest
from scripts.medtrace.astra_judge_bundle import validate
ROOT=q.ROOT
connect=q.connect
resources=q.resources
reconcile=q.reconcile
status=h.status

def lock():return read(ROOT/'EPOCH_MANIFEST.json')

def initialize(db):
    a=read(RUN/'QWEN_JUDGE_AMENDMENT.json');m=read(RUN/'RUN_MANIFEST.json')
    assert a['concurrency']==32 and a['records_per_context']==1
    identity={k:a[k] for k in ['model','snapshot','reasoning_effort','protocol','concurrency','records_per_context','temperature','seed','max_tokens','quantization','dtype','enable_thinking','enable_prefix_caching','enforce_eager','gpu_memory_utilization','max_num_batched_tokens','context_candidates','guided_decoding','packages']}
    identity['prompt']=q.PROMPT
    d=dict(epoch='MEDTRACE_CP_INIT_H8_QWEN_C32_20261005_V1',model=a['model'],snapshot=a['snapshot'],reasoning_effort=a['reasoning_effort'],protocol=a['protocol'],prompt=q.PROMPT,judge_identity=identity,batch_size=32,attempts_per_payload=1,interleave_seed=20260912,old_success_scores_reused="EXACT_PARENT_QWEN_FULL_KEY_ONLY",starting_epoch=m['starting_epoch'],deadline_epoch=m['deadline_epoch'],Judge_limit=m['Judge_limit'],amendment_binding=digest(a))
    d['config_binding']=digest(d)
    if (ROOT/'EPOCH_MANIFEST.json').exists():assert lock()==d
    else:write(ROOT/'EPOCH_MANIFEST.json',d)
    db.execute('CREATE TABLE IF NOT EXISTS inherited (key TEXT PRIMARY KEY, parent_row TEXT, batch_receipt TEXT)')
    write(ROOT/'PERMANENT_MISSING_INHERITANCE_AUDIT.json',dict(rule='Exact complete Qwen identity: inherit valid and permanent missing with original evidence; never create new attempt by changing epoch',Astra_SOL='Different identities read-only, not mixed'))

def payload(db,row,b,out):
    assert (row['question'],row['reference'],row['image_sha256'])==(b['question'],b['reference'],b['image_sha256'])
    assert isinstance(out['raw_answer'],str) and isinstance(out['raw_token_ids'],list) and all(type(x) is int for x in out['raw_token_ids'])
    full=dict(query_id=row['query_id'],question=row['question'],reference=row['reference'],image_sha256=row['image_sha256'],image_path=b['image_path'],prompt_ids=b['prompt_ids'],attention_mask=b['attention_mask'],runtime=b['runtime'],generation=b['generation'],output=out,judge=lock()['judge_identity'])
    key=digest(full);record=dict(opaque_query_id=digest([lock()['epoch'],key]),question=row['question'],gold_answer=row['reference'],raw_base_answer=out['raw_answer'])
    previous=db.execute('SELECT binding FROM payload WHERE key=?',(key,)).fetchone()
    if previous:assert json.loads(previous['binding'])==full
    else:
        inherited=parent_result(full)
        if inherited is None:db.execute('INSERT INTO payload VALUES (?,?,?,?,?,?)',(key,json.dumps(record),json.dumps(full),'PENDING',None,None))
        else:
            row,saved=inherited;bid='INHERITED_PARENT_'+row['batch'];dest=ROOT/'inherited'/(row['batch']+'.json')
            if dest.exists():assert read(dest)==saved
            else:write(dest,saved)
            db.execute('INSERT INTO payload VALUES (?,?,?,?,?,?)',(key,row['record'],json.dumps(full),row['status'],bid,row['correct']))
            db.execute('INSERT INTO inherited VALUES (?,?,?)',(key,json.dumps(row),str(dest)))
    return key

_PARENT_CACHE=None
_PARENT_BATCHES=None

def validate_inherited(full,row,saved,batch,identity):
    assert digest(full)==row['key'] and json.loads(row['binding'])==full
    assert full['judge']==identity and row['key'] in batch['keys']
    assert row['batch']==batch['id']
    assert row['status'] in ('FORMAT_VALID','MISSING')
    assert saved['batch']['batch_id']==batch['id']
    records=saved['batch']['records'];index=batch['keys'].index(row['key'])
    assert len(records)==len(batch['keys']) and records[index]==json.loads(row['record'])
    evidence=saved['evidence']
    if row['status']=='FORMAT_VALID':
        assert batch['status']=='FORMAT_VALID' and evidence['status']=='FORMAT_VALID'
        assert evidence['judge_identity']==identity and evidence['actual_model']==identity['model'] and evidence['snapshot']==identity['snapshot']
        assert evidence['input_binding']==digest(saved['batch']) and evidence['exit_code']==0 and not evidence.get('errors') and not evidence['tool_event_types'] and all(evidence['isolation_checks'].values())
        decisions=validate(saved['batch'],saved['response']);assert row['correct']==int(decisions[index]['is_correct'])
    else:
        assert batch['status']=='FAILED_NO_RETRY' and row['correct'] is None
        assert evidence['status']!='FORMAT_VALID'
    return True


def parent_result(full):
    global _PARENT_CACHE,_PARENT_BATCHES
    if _PARENT_CACHE is None:
        import sqlite3
        _PARENT_CACHE={}
        plan=read(RUN/'PLAN_CONFIG.json')
        for parent in map(Path,[plan['parent_run'],plan['historical_structured_run']]):
            assert read(parent/'private/judge_common/SCORER_DONE.json')['status']=='COMMON_SCORING_COMPLETE_WITH_MISSING'
            plock=read(parent/'private/judge_common/EPOCH_MANIFEST.json');assert plock['judge_identity']==lock()['judge_identity']
            batches={b['id']:b for b in read(parent/'RESOURCE_LEDGER.json')['Judge_batches']}
            db=sqlite3.connect('file:'+str(parent/'private/judge_common/queue.sqlite')+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
            has_inherited=db.execute("SELECT 1 FROM sqlite_master WHERE name='inherited'").fetchone()
            inherited={r['key']:dict(r) for r in db.execute('SELECT * FROM inherited')} if has_inherited else {}
            for rr in db.execute('SELECT * FROM payload'):
                row=dict(rr)
                if row['key'] in inherited:
                    saved=read(inherited[row['key']]['batch_receipt']);row=json.loads(inherited[row['key']]['parent_row']);batch=saved['parent_attempt']
                else:
                    batch=batches[row['batch']];saved=read(parent/'private/judge_common/evidence'/(row['batch']+'.json'))
                binding=json.loads(row['binding']);validate_inherited(binding,row,saved,batch,lock()['judge_identity'])
                item=(row,dict(saved,parent_attempt=batch,parent_epoch=plock))
                if row['key'] in _PARENT_CACHE:
                    prior=_PARENT_CACHE[row['key']][0];assert (prior['binding'],prior['status'],prior['correct'])==(row['binding'],row['status'],row['correct'])
                else:_PARENT_CACHE[row['key']]=item
            db.close()
    return _PARENT_CACHE.get(digest(full))


h.payload=payload
ingest=h.ingest

def request(db, operation):
    reconcile(db)
    if operation['action'] == 'next':
        ingest(db)
        remaining = read(RUN / 'RUN_MANIFEST.json')['Judge_limit'] - read(RUN / 'RESOURCE_LEDGER.json')['Judge_attempts']
        return dict(state=status(db), rows=[dict(key=r['key'], record=json.loads(r['record'])) for r in db.execute("SELECT key,record FROM payload WHERE status='PENDING' ORDER BY rowid LIMIT ?", (min(32, max(0, remaining)),))])
    if operation['action'] == 'reserve':
        keys = operation['keys']; bid = operation['batch_id']
        assert keys and len(keys) == len(set(keys)) <= 32
        for k in keys:
            assert db.execute('SELECT status FROM payload WHERE key=?', (k,)).fetchone()['status'] == 'PENDING'
        with resources() as ledger:
            m = read(RUN / 'RUN_MANIFEST.json')
            assert time.time() < m['deadline_epoch'] - 600 and not (RUN / 'STOP').exists()
            assert ledger['Judge_attempts'] + len(keys) <= m['Judge_limit']
            attempts = ledger.setdefault('Judge_batches', [])
            assert not any(a['id'] == bid or set(a['keys']) & set(keys) for a in attempts)
            previous = list(reversed(attempts))
            consecutive = 0
            for a in previous:
                if a['status'] == 'FAILED_NO_RETRY' and a.get('transport_failure'):
                    consecutive += 1
                else:
                    break
            assert consecutive < 3, 'Three transport failures: no new requests'
            ledger['Judge_attempts'] += len(keys)
            attempts.append(dict(id=bid, keys=keys, status='RESERVED', started_epoch=time.time(), model=lock()['model'], reasoning_effort=lock()['reasoning_effort'], snapshot=lock()['snapshot']))
        reconcile(db)
        return dict(status='RESERVED', items=len(keys))
    if operation['action'] == 'publish':
        bid = operation['batch_id']; evidence = operation['evidence']; response = operation.get('response')
        ledger = read(RUN / 'RESOURCE_LEDGER.json'); a = next(x for x in ledger['Judge_batches'] if x['id'] == bid)
        rows = [db.execute('SELECT * FROM payload WHERE key=?', (k,)).fetchone() for k in a['keys']]
        batch = dict(batch_id=bid, records=[json.loads(r['record']) for r in rows])
        valid = evidence.get('status') == 'FORMAT_VALID' and response is not None
        if valid:
            assert evidence['actual_model'] == lock()['model'] and evidence['reasoning_effort'] == lock()['reasoning_effort'] and evidence['snapshot'] == lock()['snapshot'] and evidence['judge_identity'] == lock()['judge_identity'] and evidence['input_binding'] == digest(batch)
            assert evidence['exit_code'] == 0 and not evidence.get('errors') and not evidence['tool_event_types'] and all(evidence['isolation_checks'].values())
            decisions = validate(batch, response)
        else:
            decisions = [None] * len(rows)
        destination = ROOT / 'evidence' / (bid + '.json')
        saved = dict(batch=batch, evidence=evidence, response=response)
        if destination.exists():
            assert read(destination) == saved
        else:
            write(destination, saved)
        for row, decision in zip(rows, decisions):
            new = 'FORMAT_VALID' if valid else 'MISSING'
            if row['status'] == 'FORMAT_VALID':
                assert valid and row['correct'] == int(decision['is_correct'])
            else:
                assert row['status'] in ('RESERVED', 'MISSING')
                assert row['status'] != 'MISSING' or not valid, 'Permanent missing cannot be rejudged'
                db.execute('UPDATE payload SET status=?,correct=?,batch=? WHERE key=?',
                    (new, int(decision['is_correct']) if valid else None, bid, row['key']))
        db.commit()
        with resources() as ledger:
            a = next(x for x in ledger['Judge_batches'] if x['id'] == bid)
            a.update(status='FORMAT_VALID' if valid else 'FAILED_NO_RETRY', ended_epoch=time.time(), transport_failure=operation.get('transport_failure', False))
            if not valid:
                ledger['Judge_failed_payloads'] = ledger.get('Judge_failed_payloads', 0) + len(rows)
        write(ROOT / 'STATUS.json', status(db))
        return dict(status='FORMAT_VALID' if valid else 'MISSING', items=len(rows))
    if operation['action'] == 'status':
        return status(db)
    if operation['action'] == 'reserved':
        return [a for a in read(RUN / 'RESOURCE_LEDGER.json').get('Judge_batches', []) if a['status'] == 'RESERVED']
    raise ValueError('Unknown action')

