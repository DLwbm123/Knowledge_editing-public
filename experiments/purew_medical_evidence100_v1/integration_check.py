"""Finite CPU admission of actual solver/update/recovery and three-arm queue."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
payload=json.loads(sys.stdin.readline());P=Path(payload['protocol']['reference_run'])
os.environ.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
with tempfile.TemporaryDirectory(prefix='e.',dir='/data/bmw/tmp') as folder:
    R=Path(folder);(R/'private/tools').mkdir(parents=True);(R/'public').mkdir()
    for p in (P/'private/tools').glob('*.py'):shutil.copy2(p,R/'private/tools'/p.name)
    for name,s in payload['files'].items():compile(s,name,'exec');(R/'private/tools'/name).write_text(s)
    (R/'private/source').symlink_to((P/'private/source').resolve())
    for name in ('BENCHMARK146_QUEUE.json','EVAL_LEDGER.json'):(R/'private'/name).symlink_to(P/'private'/name)
    os.environ.update(RUN_ROOT=str(R),ACTION='cpu_fixture',ARM='M');sys.path.insert(0,str(R/'private/tools'))
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    from curvature import selfcheck,Risk
    result=selfcheck();print('CPU math PASS',flush=True)
    check=subprocess.run([sys.executable,str(R/'private/tools/access.py')],capture_output=True,text=True)
    assert check.returncode==0,check.stderr
    result.update(json.loads(check.stdout));print('CPU access PASS',flush=True)
    import common as c
    c.write(R/'private/FROZEN_PROTOCOL.json',payload['protocol']);c.write(R/'private/GPU_SOURCE_VERSION.json',dict(commit='CPU_FIXTURE'))
    c.write(R/'RUN_MANIFEST.json',dict(deadline_epoch=time.time()+3600,GPU_seconds_limit=86400,Judge_limit=6543,min_free_bytes=1,owned_weight_limit_bytes=6*1024**3))
    c.write(R/'STAGE_CAP.json',dict(GPU_seconds_limit=86400));c.write(R/'RESOURCE_LEDGER.json',dict(Judge_attempts=0,Astra_purew146_batches=[],gpu_sessions=[],gpu_seconds_used=0,weights_observed_peak_bytes=0))
    import learner
    from types import SimpleNamespace
    class Toy(torch.nn.Module):
        def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.randn(7,3).half())
        def forward(self,inputs_embeds,**kwargs):return SimpleNamespace(logits=torch.nn.functional.linear(inputs_embeds,self.weight))
    torch.manual_seed(33);model=Toy();runtime=SimpleNamespace(model=model,device=torch.device('cpu'))
    batch=(dict(inputs_embeds=torch.randn(1,3,3).half()),torch.tensor([[-100,2,4]]));full=[batch]*5
    weight=model.weight;initial=weight.detach().clone();master=initial.float().clone();risk=Risk(weight)
    import random,numpy as np
    random.seed(3);np.random.seed(3);torch.manual_seed(3)
    rng=c.rng();npr=np.random.get_state()
    def advance(m,rr,start,end):
        for slot in range(start,end+1):
            random.random();np.random.random();torch.rand(2)
            m,event=learner.optimize(runtime,weight,m,rr,full,[],None,slot)
        return m,event
    whole,e=advance(master.clone(),risk,1,8);native=weight.detach().clone();whole_rng=c.rng()
    with torch.no_grad():weight.copy_(initial)
    c.restore_rng(rng);np.random.set_state(npr);rr=Risk(weight)
    partial,event=advance(master.clone(),rr,1,4)
    saved=learner.save(partial,weight,rr,1,4,4,event,initial.float())
    loaded=torch.load(R/'private/active/M/state.pt',map_location='cpu',weights_only=True)
    assert set(loaded['risk'])=={'B','v','rho','q','ev','erho','eq','compressions'}
    assert not any(k in loaded for k in ('question','image','labels','teacher','features','old_cases'))
    assert loaded['completed_slots']==(loaded['owner']-1)*160+loaded['slot']
    bad=dict(loaded,completed_slots=5)
    try:learner.validate_state(bad)
    except AssertionError:pass
    else:raise AssertionError('Corrupt clock accepted')
    c.restore_rng(loaded['rng']);n=loaded['numpy_rng'];np.random.set_state((n[0],n[1].numpy().astype('uint32'),n[2],n[3],n[4]))
    rr.restore(loaded['risk'],'cpu');resumed,event=advance(loaded['master'],rr,5,8)
    assert torch.equal(whole,resumed) and torch.equal(native,weight)
    assert torch.equal(whole_rng['torch_rng'],c.rng()['torch_rng'])
    # Medical loss is before-edit current scalar drift; no absence/opposite target.
    anchors=learner.current_anchors(runtime,full,full)
    assert len(anchors['removed'])==5
    probes=learner.probe_rows(runtime,weight,full,[],20260913);assert probes.shape==(4,weight.numel())
    before=c.rng()['torch_rng'];learner.probe_rows(runtime,weight,full,full,20360913)
    assert torch.equal(before,c.rng()['torch_rng'])
    # Check an actual medical-slot resume retains the before-edit current anchors.
    with torch.no_grad():weight.copy_(initial)
    mr=Risk(weight);mm=initial.float().clone();aa=learner.current_anchors(runtime,full,full)
    mm,ee=learner.optimize(runtime,weight,mm,mr,full,full,aa,1)
    learner.save(mm,weight,mr,1,1,1,ee,initial.float(),anchors=aa)
    ms=torch.load(R/'private/active/M/state.pt',map_location='cpu',weights_only=True);learner.validate_state(ms)
    assert ms['current_anchors']==aa
    result.update(actual_executor_update=True,exact_master_native_rng_resume=True,corrupt_source_clock_rejected=True,
        old_sample_free_state=True,separate_probe_RNG=True,current_anchor_scalar_loss=True)
    print('CPU executor PASS',flush=True)
    import queue_medical as queue
    q=queue.q;q.ROOT.mkdir(parents=True,exist_ok=True)
    q.write(q.ROOT/'EPOCH_MANIFEST.json',dict(epoch='CPU_FIXTURE',model='gpt-6-astra',reasoning_effort='medium',protocol=q.PROTOCOL,prompt=q.PROMPT))
    db=q.connect();queue.ingest(db);old=queue.ro(Path(payload['protocol']['K_reference_run'])/'private/judge_purew146/queue.sqlite')
    before=db.execute('SELECT count(*) FROM payload').fetchone()[0]
    tasks=c.read(R/'private/BENCHMARK146_QUEUE.json')['tasks'];ledger=c.read(R/'private/EVAL_LEDGER.json')
    lookup={(x['mode'],x['prefix'],x['edit_order'],x['query_id']):x['path'] for x in old.execute("SELECT * FROM consumer WHERE method='K'")}
    for arm in ('M','M-S','M-E'):
        for mode,owner,prefix in [('bank_R0',0,n) for n in (1,50,100)]+[('insertion_R0',n,n) for n in range(1,101)]:
            phase=dict(arm=arm,mode=mode,owner=owner,prefix=prefix,protocol=c.digest(payload['protocol']),code='CPU_FIXTURE')
            ids=queue.expected_ids(tasks,ledger,phase);dest=R/'private/outputs'/arm/mode/str(owner)/str(prefix)
            for qid in ids:
                d=c.read(lookup[mode,prefix,owner,qid]);d['binding']['phase']=phase;c.write(dest/(qid+'.json'),d)
            c.write(dest/'COMPLETE.json',dict(phase=phase,outputs=len(ids)))
    queue.ingest(db);assert not any(db.execute('SELECT count(*) FROM consumer WHERE method=?',(a,)).fetchone()[0] for a in ('M','M-S','M-E'))
    assert not queue.terminal_allowed(db) and queue.status(db)['status']=='GENERATING'
    c.write(R/'private/FINAL_CONSUMED.json',dict(test_only=True));c.write(R/'private/GENERATION_COMPLETE.json',dict(test_only=True))
    queue.ingest(db);queue.ingest(db)
    assert db.execute('SELECT count(*) FROM payload').fetchone()[0]==before
    assert all(db.execute('SELECT count(*) FROM consumer WHERE method=?',(a,)).fetchone()[0]==2181 for a in ('M','M-S','M-E'))
    assert queue.terminal_allowed(db)
    # Exercise report's terminal binding, shared missing, and true insertion transitions.
    for arm in ('M','M-S','M-E'):
        c.write(R/'private'/('FINAL_CONSUMED_'+arm+'.json'),dict(test_only=True))
        for owner in range(1,101):
            for slot in range(1,161):
                c.write(R/'private/steps'/arm/str(owner)/(str(slot)+'.json'),dict(owner=owner,slot=slot,accepted=True,
                    proposal_norm=.001,actual_native_norm=.001,risk_after=.001,source_CE=.2,medical_scope_loss=0.))
    learner.b.COUNTS['attempt_slots']=48000;learner.b.counts()
    c.write(q.ROOT/'ALL_WORKERS_COMPLETE.json',dict(test_only=True))
    import medical_report
    medical_report.main();final=c.read(R/'public/RESULTS.json')
    assert final['status']=='COMPLETE_WITH_MISSING' and not final['scientific_success']
    primary=next(x for x in final['contrasts'] if x['comparison']=='M-E-minus-M-S-prefix100' and x['task']=='T2G')
    assert primary['delta_macro_bounds_pp']==[0.,0.]
    assert next(x for x in final['panels'] if x['arm']=='M-E' and x['prefix']==100 and x['task']=='T2G')['observations']==386
    assert not final['gates']['M-E']['zero_Base_damage']
    result.update(status='PASS',three_arm_consumers=6543,idempotent_output_binding=True,scoring_waits_all_trajectories=True,
        shared_missing_exact_cancellation=True,full_denominator_and_harm_report=True,actual_insertion_not_single=True,
        real_VLM=False,GPU_updates=0,Judge_requests=0,full_execution_recovery_checked=True)
    print(json.dumps(result))
