"""CPU synthetic queue integration, using the existing sealed payload implementation."""
import copy
import json
import os
from pathlib import Path
import tempfile
import importlib
queue=importlib.import_module(os.environ.get('M13_CPU_QUEUE_MODULE','queue146'))
contract=queue.contract
import astra_queue as a
from fine_queue import full_identity,selfcheck


def rejected(fn):
    try:fn()
    except AssertionError:return
    raise AssertionError('Invalid queue operation accepted')


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    packet=json.loads(Path(os.environ['M13_CPU_PACKET']).read_text())
    reserved=contract.coverage_reservation(packet['original_tasks'],packet['base_mask'])
    plan=copy.deepcopy(reserved)
    for row in plan:
        qid=row['query']['query_id']
        row['query']=dict(query_id=qid,question='fixture Q',reference='fixture A',image_path='fixture image')
    q=a.q;judge=dict(model='gpt-6-astra',reasoning_effort='medium',protocol=q.PROTOCOL,prompt=q.PROMPT)
    def data(row):
        inp=dict(row['query'],image_sha256='synthetic fixture')
        binding=dict(inp,prompt_ids=[1],attention_mask=[1],runtime={'precision':'FP32'},generation={'max_new_tokens':1024})
        return dict(consumer_id=contract.original.consumer_id(row),
            **{k:row[k] for k in ('method','mode','prefix','owner')},role='FIXTURE',lock='fixture lock',
            binding=dict(input=inp,judge_input=binding),R0=dict(raw_answer='fixture A',raw_token_ids=[1,2]))
    def outputs(rows):return ((str(i),data(row)) for i,row in enumerate(rows))
    with tempfile.TemporaryDirectory(prefix='e.',dir=os.environ['M13_CPU_TMP']) as tmp:
        root=Path(tmp);q.ROOT=root/'queue';db=q.connect()
        q.write(q.ROOT/'EPOCH_MANIFEST.json',dict(epoch=queue.EPOCH,**judge))
        rejected(lambda:queue.ingest_generated(q,db,reserved,[],full_identity,judge,'fixture lock',{}))
        first=data(plan[0]);key=q.digest(full_identity(first,judge))
        second=data(next(row for row in plan if row['query']['query_id']!=plan[0]['query']['query_id']))
        missing=q.digest(full_identity(second,judge))
        known={key:('FORMAT_VALID',1),missing:('MISSING',None)}
        with db:result=queue.ingest_generated(q,db,plan,outputs(plan),full_identity,judge,'fixture lock',known)
        assert result['consumers']==contract.CAPS['evaluation_consumers'] and not result['all_scores_valid'] and not result['all_attempts_resolved']
        assert result['consumer_status']['FORMAT_VALID']>1 and result['consumer_status']['MISSING']>1
        db.commit()
        assert queue.known_scores(q,[q.ROOT/'queue.sqlite'],judge)[key]==('FORMAT_VALID',1)
        historical=q.ROOT/'queue.sqlite'
        q.ROOT=root/'conflict';conflict=q.connect()
        row=db.execute('SELECT * FROM payload WHERE key=?',(key,)).fetchone()
        conflict.execute('INSERT INTO payload VALUES (?,?,?,?,?,?)',tuple(row));conflict.execute('UPDATE payload SET correct=0')
        conflict.commit();conflict.close()
        rejected(lambda:queue.known_scores(q,[historical,q.ROOT/'queue.sqlite'],judge))
        q.ROOT=historical.parent
        rejected(lambda:queue.request(a,q,root,db,dict(action='reserve',keys=[key],worker=0)))
        rejected(lambda:queue.request(a,q,root,db,dict(action='reserve',keys=[missing],worker=0)))
        db.execute("UPDATE payload SET status='FORMAT_VALID',correct=1 WHERE status='PENDING'")
        assert queue.coverage(db)['all_attempts_resolved'] and not queue.coverage(db)['all_scores_valid']
        before=db.execute('SELECT count(*) FROM payload').fetchone()[0]
        changed=copy.deepcopy(first);changed['R0']['raw_token_ids']=[3,2]
        assert q.digest(full_identity(changed,judge))!=key
        db.rollback();db.close()
        for kind in ('missing','duplicate','changed_input','wrong_method'):
            q.ROOT=root/kind;db=q.connect();q.write(q.ROOT/'EPOCH_MANIFEST.json',dict(epoch=queue.EPOCH,**judge))
            def altered():
                for i,row in enumerate(plan):
                    if kind=='missing' and i==len(plan)-1:continue
                    d=data(row)
                    if i==1 and kind=='duplicate':d=data(plan[0])
                    if i==0 and kind=='changed_input':d['binding']['input']['question']='changed'
                    if i==0 and kind=='wrong_method':d['method']='REFINE'
                    yield str(i),d
            def attempt():
                with db:queue.ingest_generated(q,db,plan,altered(),full_identity,judge,'fixture lock',{})
            rejected(attempt)
            assert db.execute('SELECT count(*) FROM consumer').fetchone()[0]==0
            assert db.execute('SELECT count(*) FROM payload').fetchone()[0]==0
            db.close()
    result=dict(status='PASS',fixture_consumers=contract.CAPS['evaluation_consumers'],unique_fixture_payloads=before,
        exact_consumer_coverage=True,reserved_sources_rejected=True,
        missing_duplicate_changed_input_wrong_method_rollback=True,
        inherited_valid_and_permanent_missing_preserved=True,repeated_attempt_rejected=True,
        resolved_missing_is_not_valid_complete=True,token_change_not_reused=True,
        conflicting_history_rejected=True,full_identity=selfcheck(),
        new_GPU_leases=0,new_VLM_calls=0,new_Judge=0,training_ready=False,
        scorer_worker_launcher_implemented=False)
    Path(os.environ['M13_CPU_REPORT']).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
