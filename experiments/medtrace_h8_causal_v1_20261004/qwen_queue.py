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
    d=dict(epoch='MEDTRACE_H8_QWEN_C32_20261004_V1',model=a['model'],snapshot=a['snapshot'],reasoning_effort=a['reasoning_effort'],protocol=a['protocol'],prompt=q.PROMPT,judge_identity=identity,batch_size=32,attempts_per_payload=1,interleave_seed=20260912,old_success_scores_reused=0,starting_epoch=m['starting_epoch'],deadline_epoch=m['deadline_epoch'],Judge_limit=m['Judge_limit'],amendment_binding=digest(a))
    d['config_binding']=digest(d)
    if (ROOT/'EPOCH_MANIFEST.json').exists():assert lock()==d
    else:write(ROOT/'EPOCH_MANIFEST.json',d)
    write(ROOT/'PERMANENT_MISSING_INHERITANCE_AUDIT.json',dict(historical_Astra_missing_preserved=len(h.MISSING),inherited_same_Qwen_identity=0,reason=a['old_missing']))

def payload(db,row,b,out):
    assert (row['question'],row['reference'],row['image_sha256'])==(b['question'],b['reference'],b['image_sha256'])
    assert isinstance(out['raw_answer'],str) and isinstance(out['raw_token_ids'],list) and all(type(x) is int for x in out['raw_token_ids'])
    full=dict(query_id=row['query_id'],question=row['question'],reference=row['reference'],image_sha256=row['image_sha256'],image_path=b['image_path'],prompt_ids=b['prompt_ids'],attention_mask=b['attention_mask'],runtime=b['runtime'],generation=b['generation'],output=out,judge=lock()['judge_identity'])
    key=digest(full);record=dict(opaque_query_id=digest([lock()['epoch'],key]),question=row['question'],gold_answer=row['reference'],raw_base_answer=out['raw_answer'])
    previous=db.execute('SELECT binding FROM payload WHERE key=?',(key,)).fetchone()
    if previous:assert json.loads(previous['binding'])==full
    else:db.execute('INSERT INTO payload VALUES (?,?,?,?,?,?)',(key,json.dumps(record),json.dumps(full),'PENDING',None,None))
    return key

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

