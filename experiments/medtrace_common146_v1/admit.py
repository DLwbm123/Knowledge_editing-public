"""Validate frozen raw bindings and prepare a new, score-free common epoch."""
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

RUN = Path(os.environ['RUN_ROOT'])
sys.path.insert(0, str(RUN/'private/tools'))
import astra_queue as a
q = a.q
q.ROOT = RUN/'private/judge_common146_medium'
EPOCH = 'MEDTRACE_TT88_BASELINES_COMMON_MEDIUM_20261011_V1'
METHODS = ('TT88_W0_RETRO146', 'balancedit', 'belora')


def connect(path):
    db = sqlite3.connect('file:'+str(path)+'?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    return db


def slot(row):
    if row['mode'] in ('single_R0', 'single'):
        return ('single_R0', 1, row['edit_order'], row['query_id'])
    if row['mode']=='insertion_R0' or row['folder']=='native':
        return ('insertion_R0', row['prefix'], row['prefix'], row['query_id'])
    return ('bank_R0', row['prefix'], 0, row['query_id'])


def identity(row, base, output, judge):
    assert all(row[k]==base[k] for k in ('question','reference','image_sha256'))
    return dict(query_id=row['query_id'],question=row['question'],reference=row['reference'],
        image_sha256=row['image_sha256'],image_path=base['image_path'],prompt_ids=base['prompt_ids'],
        attention_mask=base['attention_mask'],runtime=base['runtime'],generation=base['generation'],
        output=output,judge=judge)


def selfcheck():
    base=dict(question='q',reference='a',image_sha256='i',image_path='p',prompt_ids=[1],
              attention_mask=[1],runtime={'lane':'a'},generation={'cap':1024})
    row=dict(base,query_id='q1');out=dict(raw_answer='a',raw_token_ids=[2]);judge={'effort':'medium'}
    first=identity(row,base,out,judge)
    assert first!=identity(row,base,dict(out,raw_token_ids=[3]),judge)
    assert first!=identity(row,dict(base,prompt_ids=[3]),out,judge)
    assert slot(dict(mode='single',prefix=1,edit_order=3,query_id='q',folder='native'))==slot(dict(mode='single_R0',prefix=1,edit_order=3,query_id='q',folder='PANEL'))
    assert slot(dict(mode='sequential',prefix=50,edit_order=50,query_id='q',folder='panel'))==slot(dict(mode='bank_R0',prefix=50,edit_order=0,query_id='q',folder='PANEL'))
    return dict(status='PASS',token_and_prompt_changes_separate=True,mode_projection=True)


def main():
    began=time.time();selfcheck()
    config=q.read(RUN/'PLAN_CONFIG.json');old=Path(config['baseline_parent']);tt=Path(config['TT_parent'])
    assert q.read(RUN/'RESOURCE_LEDGER.json')['Judge_attempts']==0
    # Reuse the full historical method/phase admission, writing only in this run.
    spec=importlib.util.spec_from_file_location('baseline_admit',RUN/'private/tools/baseline_admit.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.RUN=RUN;module.main()
    tasks=q.read(tt/'private/BENCHMARK146_QUEUE.json')['tasks']
    legacy=q.read(old/'private/legacy_stage17/COHORT_AND_SUPPORT_LEDGER.json')
    ledger=q.read(tt/'private/EVAL_LEDGER.json');bindings=q.read(tt/'private/EVAL_BINDINGS.json')
    by={t['edit_id']:t for t in legacy['tasks']}
    assert [t['edit_id'] for t in tasks]==legacy['main_T0'] and len(tasks)==146
    assert all(all(t[k]==by[t['edit_id']][k] for k in ('native','fit_questions','seed','order')) for t in tasks)
    sources={METHODS[0]:connect(tt/'private/judge_astra_medium/queue.sqlite'),
             'balancedit':connect(old/'private/judge_common/queue.sqlite')}
    sources['belora']=sources['balancedit']
    ttrows=list(sources[METHODS[0]].execute('SELECT * FROM consumer WHERE method=?',(METHODS[0],)))
    target={slot(r) for r in ttrows};assert len(ttrows)==len(target)==5722
    qids={r['query_id'] for r in ttrows};assert len(qids)==1509
    assert all(ledger['Base_correctness'][k]==legacy['Base_correctness'][k] for k in qids)
    judge=dict(model='gpt-6-astra',reasoning_effort='medium',protocol=q.PROTOCOL,prompt=q.PROMPT)
    db=q.connect();assert db.execute('SELECT count(*) FROM payload').fetchone()[0]==0
    manifest=dict(epoch=EPOCH,**judge,maximum_new_Judge=3146,workers=4,batch_size=50,
        attempts_per_payload=1,old_scores_reused=0,new_training=0,new_generation=0,
        original_Base_masks=True,Base_rejudging=False)
    q.write(q.ROOT/'EPOCH_MANIFEST.json',manifest)
    checks={};proof=[]
    for method in METHODS:
        source=sources[method];rows=list(source.execute('SELECT * FROM consumer WHERE method=?',(method,)))
        selected={};raw_checked=0
        for c in rows:
            s=slot(c)
            if s not in target:continue
            d=q.read(c['path']);assert q.digest(d)==c['output_binding'], 'Changed raw consumer binding'
            row=ledger['queries'][c['query_id']];base=bindings[row['opaque_Base_id']]
            assert d['binding']['input']==row
            output=d['R0'] if method==METHODS[0] else d.get('modes',d)['R0' if method=='balancedit' else 'NATIVE']
            if method==METHODS[0]:
                assert d['binding']['judge_input']==base
                assert d['binding']['mode']==c['mode'] and d['binding']['prefix']==c['prefix']
            prior=json.loads(source.execute('SELECT binding FROM payload WHERE key=?',(c['payload_key'],)).fetchone()[0])
            full=identity(row,base,output,judge)
            prior['judge']=judge
            assert full==prior, 'Raw tokens/input/runtime differ from admitted payload'
            key=q.payload(db,row,base,output);assert key==q.digest(full)
            raw_checked+=1
            if s in selected:
                assert selected[s]==key, 'Duplicate evaluation slot has conflicting outputs'
                continue
            selected[s]=key;mode,prefix,owner,qid=s
            cid=q.digest([method,s])
            db.execute('INSERT INTO consumer VALUES (?,?,?,?,?,?,?,?,?,?)',
                (cid,method,mode,prefix,owner,'PANEL' if mode!='insertion_R0' else 'INSERTION',qid,c['path'],c['output_binding'],key))
            proof.append(dict(consumer=cid,parent_consumer=c['id'],parent_payload=c['payload_key']))
        assert set(selected)==target
        checks[method]=dict(evaluation_slots=len(selected),raw_records_checked=raw_checked,
                            unique_payloads=len(set(selected.values())))
    assert db.execute('SELECT count(*) FROM payload').fetchone()[0]==3146
    assert db.execute('SELECT count(*) FROM consumer').fetchone()[0]==17166
    assert not db.execute('SELECT 1 FROM payload WHERE status!=\'PENDING\' OR correct IS NOT NULL').fetchone()
    db.execute("INSERT INTO done VALUES ('generation')");db.commit()
    q.write(RUN/'private/CONSUMER_PROVENANCE.json',proof)
    q.write(RUN/'private/BENCHMARK146_QUEUE.json',dict(tasks=tasks))
    q.write(RUN/'private/EVAL_LEDGER.json',dict(queries={k:ledger['queries'][k] for k in qids},Base_correctness={k:ledger['Base_correctness'][k] for k in qids}))
    result=dict(status='PASS',methods=checks,consumers=17166,payloads=3146,queries=1509,
        original_order_and_training_inputs=True,all_output_bindings=True,all_raw_text_and_tokens=True,
        full_input_runtime_generation_identity=True,original_Base_masks=True,old_scores_reused=0,
        additional_training=0,additional_generation=0,GPU_hours=0,seconds=time.time()-began,selfcheck=selfcheck())
    q.write(RUN/'public/ADMISSION.json',result);q.write(q.ROOT/'READY.json',result)
    print(json.dumps(result))


if __name__=='__main__':
    main()
