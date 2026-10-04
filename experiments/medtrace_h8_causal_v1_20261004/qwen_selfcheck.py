"""CPU-only boundary checks in a temporary synthetic queue, zero formal attempts."""
import json
import os
from pathlib import Path
import tempfile

import qwen_queue as q

def main():
    real=Path(os.environ['RUN_ROOT']);original=q.read(real/'RESOURCE_LEDGER.json')
    frozen=q.lock();manifest=q.read(real/'RUN_MANIFEST.json')
    with tempfile.TemporaryDirectory(dir=real/'private/tmp',prefix='qwen_check_') as name:
        root=Path(name);judge=root/'private/judge_common';judge.mkdir(parents=True)
        for module in (q,q.q,q.h):module.RUN=root
        q.ROOT=q.q.ROOT=q.h.ROOT=judge
        q.write(judge/'EPOCH_MANIFEST.json',frozen)
        q.write(root/'RUN_MANIFEST.json',manifest)
        q.write(root/'RESOURCE_LEDGER.json',dict(Judge_attempts=0,Judge_failed_payloads=0,gpu_seconds_used=0,gpu_sessions=[]))
        db=q.connect()
        row=dict(query_id='synthetic',question='synthetic',reference='synthetic',image_sha256='image')
        base=dict(question='synthetic',reference='synthetic',image_sha256='image',image_path='synthetic',prompt_ids=[1],attention_mask=[1],runtime={'fixture':True},generation={'fixture':True})
        out=dict(raw_answer='synthetic',raw_token_ids=[1])
        key=q.payload(db,row,base,out)
        assert q.payload(db,row,base,out)==key
        other=q.payload(db,row,base,dict(out,raw_token_ids=[2]));assert other!=key
        altered=dict(frozen,judge_identity=dict(frozen['judge_identity'],snapshot='different'))
        q.write(judge/'EPOCH_MANIFEST.json',altered)
        assert q.payload(db,row,base,out)!=key
        q.write(judge/'EPOCH_MANIFEST.json',frozen);db.commit()
        q.request(db,dict(action='reserve',batch_id='test',keys=[key]))
        try:q.request(db,dict(action='reserve',batch_id='duplicate',keys=[key]));raise RuntimeError('duplicate accepted')
        except AssertionError:pass
        record=json.loads(db.execute('SELECT record FROM payload WHERE key=?',(key,)).fetchone()[0]);batch=dict(batch_id='test',records=[record])
        response=dict(batch_id='test',decisions=[dict(opaque_query_id=record['opaque_query_id'],is_correct=True)])
        ev=dict(status='FORMAT_VALID',actual_model=frozen['model'],reasoning_effort='disabled',snapshot=frozen['snapshot'],judge_identity=frozen['judge_identity'],input_binding=q.digest(batch),exit_code=0,errors=[],tool_event_types=[],isolation_checks={'fixture':True})
        try:q.request(db,dict(action='publish',batch_id='test',evidence=dict(ev,snapshot='wrong'),response=response));raise RuntimeError('wrong snapshot accepted')
        except AssertionError:pass
        q.request(db,dict(action='publish',batch_id='test',evidence=ev,response=response))
        assert db.execute('SELECT correct FROM payload WHERE key=?',(key,)).fetchone()[0]==1
        q.request(db,dict(action='reserve',batch_id='failed',keys=[other]))
        q.request(db,dict(action='publish',batch_id='failed',evidence={'status':'FAILED_NO_RETRY'},response=None))
        assert db.execute('SELECT status FROM payload WHERE key=?',(other,)).fetchone()[0]=='MISSING'
        try:q.request(db,dict(action='reserve',batch_id='forbidden-retry',keys=[other]));raise RuntimeError('missing retried')
        except AssertionError:pass
        db.close()
    assert q.read(real/'RESOURCE_LEDGER.json')==original, 'CPU test touched formal ledger'
    print(json.dumps(dict(status='PASS',checks=['full identity dedup','tokens and snapshot separate keys','one attempt','wrong snapshot rejected','valid response publication','missing terminal'],formal_Judge_attempts=original['Judge_attempts'],GPU_loads=0)))

if __name__=='__main__':main()
