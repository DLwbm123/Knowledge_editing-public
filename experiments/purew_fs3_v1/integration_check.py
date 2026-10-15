"""Finite CPU checks of Fisher, actual native gates/recovery, and complete scoring contract."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
payload=json.loads(sys.stdin.readline());P=Path(payload['protocol']['M_reference_run'])
os.environ.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
with tempfile.TemporaryDirectory(prefix='e.',dir='/data/bmw/tmp') as folder:
    R=Path(folder);(R/'private/tools').mkdir(parents=True);(R/'public').mkdir()
    for p in (P/'private/tools').glob('*.py'):shutil.copy2(p,R/'private/tools'/p.name)
    for name,s in payload['files'].items():compile(s,name,'exec');(R/'private/tools'/name).write_text(s)
    (R/'private/source').symlink_to((P/'private/source').resolve())
    for name in ('BENCHMARK146_QUEUE.json','EVAL_LEDGER.json'):(R/'private'/name).symlink_to(P/'private'/name)
    os.environ.update(RUN_ROOT=str(R),ACTION='cpu_fixture',ARM='F');sys.path.insert(0,str(R/'private/tools'))
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    import common as c
    c.write(R/'private/FROZEN_PROTOCOL.json',payload['protocol']);c.write(R/'private/GPU_SOURCE_VERSION.json',dict(commit='CPU_FIXTURE'))
    c.write(R/'RUN_MANIFEST.json',dict(deadline_epoch=time.time()+3600,GPU_seconds_limit=86400,Judge_limit=483,min_free_bytes=1,owned_weight_limit_bytes=20*1024**3))
    c.write(R/'STAGE_CAP.json',dict(GPU_seconds_limit=86400));c.write(R/'RESOURCE_LEDGER.json',dict(Judge_attempts=0,Astra_purew146_batches=[],gpu_sessions=[],gpu_seconds_used=0,weights_observed_peak_bytes=0))
    from curvature import selfcheck,Risk
    result=selfcheck()
    from geometry import SupportRisk,fisher_noise,native_candidate
    p=torch.tensor([.1,.2,.7],dtype=torch.float64);C=torch.diag(p.sqrt())-p[:,None]*p.sqrt()[None,:]
    assert torch.allclose(C@C.T,torch.diag(p)-p[:,None]*p[None,:],atol=1e-14)
    torch.manual_seed(77);w=torch.randn(19).half();sr=SupportRisk(w);births=[];point=w.double()
    for i in range(3):
        rows=torch.randn(4,19,dtype=torch.float64)*.03;births.append((rows,point.clone()))
        before_q,before_v=sr.q,sr.v.clone();sr.consolidate(rows)
        assert sr.q==before_q and torch.equal(sr.v,before_v)
        exact=sum((a.T@a for a,_ in births),torch.zeros(19,19,dtype=torch.float64))
        assert torch.allclose(sr.B.T@sr.B,exact,atol=1e-12)
        delta=torch.randn(19,dtype=torch.float64)*.01;sr.advance(delta);point+=delta
        oracle=sum(.5*float((a@(point-b)).square().sum()) for a,b in births)
        assert abs(sr.q-oracle)<1e-12
    assert len(sr.B)==12 and sr.rho==0
    restored=SupportRisk(w);restored.restore(sr.state(),'cpu')
    assert torch.equal(restored.H(delta),sr.H(delta)) and restored.q==sr.q
    # Actual rounding causes risk rejection despite a feasible real-valued proposal.
    native=torch.ones(1,dtype=torch.float16);master=torch.tensor([1.00045]);rr=Risk(native);rr.B=torch.ones(1,1,dtype=torch.float64)
    _,_,ev=native_candidate(master,native,torch.tensor([-5.]),rr,epsilon=.00125)
    assert 'RISK' in ev['candidate_diagnostics'][0]['rejection_reasons']
    rr=Risk(native);m,n,ev=native_candidate(master,native,torch.tensor([-.01]),rr)
    assert not ev['accepted'] and torch.equal(m,master) and torch.equal(n,native)
    assert any('PROXIMAL' in x['rejection_reasons'] for x in ev['candidate_diagnostics'])
    assert any('ZERO_NATIVE' in x['rejection_reasons'] for x in ev['candidate_diagnostics'])
    result.update(Fisher_covariance_identity=True,S_full_sampled_covariance=True,cumulative_birth_energy_exact=True,
        actual_quantized_risk_rejection=True,actual_proximal_rejection=True,actual_zero_rejection=True,source_metric_epsilon_units='token-mean sampled KL nats')
    check=subprocess.run([sys.executable,str(R/'private/tools/access.py')],capture_output=True,text=True);assert check.returncode==0,check.stderr
    import learner,checkpoint,random,numpy as np
    from types import SimpleNamespace
    class Toy(torch.nn.Module):
        def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.randn(7,3).half())
        def forward(self,inputs_embeds,**kwargs):return SimpleNamespace(logits=torch.nn.functional.linear(inputs_embeds,self.weight))
    torch.manual_seed(33);model=Toy();runtime=SimpleNamespace(model=model,device=torch.device('cpu'))
    batch=(dict(inputs_embeds=torch.randn(1,3,3).half()),torch.tensor([[-100,2,4]]));full=[batch]*5
    weight=model.weight;initial=weight.detach().clone();master=initial.float().clone();rr=Risk(weight)
    random.seed(3);np.random.seed(3);torch.manual_seed(3);rng=c.rng();npr=np.random.get_state()
    def advance(m,risk,start,end):
        for slot in range(start,end+1):
            random.random();np.random.random();torch.rand(2)
            m,event=learner.optimize(runtime,weight,m,risk,full,[],None,slot)
        return m,event
    whole,event=advance(master.clone(),rr,1,8);end_native=weight.detach().clone();end_rng=c.rng()
    with torch.no_grad():weight.copy_(initial)
    c.restore_rng(rng);np.random.set_state(npr);rr=Risk(weight)
    partial,event=advance(master.clone(),rr,1,4);learner.old.save(partial,weight,rr,1,4,4,event,initial.float())
    state=checkpoint.read(R/'private/active/F/state.pt');learner.validate_state(state)
    bad=dict(state,completed_slots=5)
    try:learner.validate_state(bad)
    except AssertionError:pass
    else:raise AssertionError('Corrupted clock accepted')
    c.restore_rng(state['rng']);n=state['numpy_rng'];np.random.set_state((n[0],n[1].numpy().astype('uint32'),n[2],n[3],n[4]));rr.restore(state['risk'],'cpu')
    resumed,event=advance(state['master'],rr,5,8)
    assert torch.equal(whole,resumed) and torch.equal(end_native,weight) and torch.equal(end_rng['torch_rng'],c.rng()['torch_rng'])
    before=c.rng()['torch_rng'];probes=learner.probe_rows(runtime,weight,full,[],20260913)
    assert probes.shape==(4,weight.numel()) and torch.equal(before,c.rng()['torch_rng'])
    checkpoint.write(R/'private/active/F/copy.pt',state);assert checkpoint.exact(state,checkpoint.read(R/'private/active/F/copy.pt'))
    support=SupportRisk(state['native'])
    for _ in range(2):support.consolidate(torch.randn(4,state['native'].numel(),dtype=torch.float64)*.03)
    full_state=dict(state,arm='S',risk=support.state());learner.validate_state(full_state)
    checkpoint.write(R/'private/active/S/copy.pt',full_state)
    support2=SupportRisk(state['native']);support2.restore(checkpoint.read(R/'private/active/S/copy.pt')['risk'],'cpu')
    g=torch.randn_like(state['master'])*.01
    ma,na,ea=native_candidate(state['master'],state['native'],g,support)
    mb,nb,eb=native_candidate(state['master'],state['native'],g,support2)
    assert torch.equal(ma,mb) and torch.equal(na,nb) and ea==eb
    assert not any(k in state for k in ('question','image','features','old_cases','teacher'))
    result.update(actual_executor_update=True,exact_master_native_rng_resume=True,corrupt_clock_rejected=True,separate_probe_RNG=True,
        current_only_IO_guard=True,compressed_state_roundtrip=True)
    print('CPU math/native/recovery PASS',flush=True)
    import queue_fs as queue
    q=queue.q;q.ROOT.mkdir(parents=True,exist_ok=True);q.write(q.ROOT/'EPOCH_MANIFEST.json',dict(epoch='CPU_FIXTURE',model='gpt-6-astra',reasoning_effort='medium',protocol=q.PROTOCOL,prompt=q.PROMPT))
    c.write(R/'private/M_PARENT_PROTOCOL_BINDING.json',dict(protocol=c.digest(payload['protocol'])))
    db=q.connect();queue.ingest(db)
    env=dict(os.environ,ENTRY=str(R/'private/tools/queue_fs.py'),PYTHONPATH='');env.pop('ACTION',None)
    runner="import os,runpy,sys;runpy.run_path(os.environ['ENTRY'],run_name='__main__');assert 'torch' not in sys.modules"
    fresh=subprocess.run([sys.executable,'-'],input=runner,env=env,capture_output=True,text=True,timeout=30)
    assert fresh.returncode==0,fresh.stderr
    assert json.loads(fresh.stdout)==[]
    env['ENTRY']=str(R/'private/tools/report_fs.py')
    fresh=subprocess.run([sys.executable,'-'],input="import os,runpy;runpy.run_path(os.environ['ENTRY'],run_name='entry_import_selfcheck')",env=env,capture_output=True,text=True,timeout=30)
    assert fresh.returncode==0,fresh.stderr
    result.update(standalone_queue_entry=True,standalone_report_import=True,queue_no_torch=True)
    old=queue.parent.ro(Path(payload['protocol']['K_reference_run'])/'private/judge_purew146/queue.sqlite')
    lookup={x['query_id']:x['path'] for x in old.execute("SELECT * FROM consumer WHERE method='K' AND mode='bank_R0' AND prefix=100")}
    tasks=c.read(R/'private/BENCHMARK146_QUEUE.json')['tasks'];ledger=c.read(R/'private/EVAL_LEDGER.json')
    for arm in ('M','F','S'):
        phases=[('bank_R0',0,1),('bank_R0',0,3),('protect100_R0',0,100)]+[('insertion_R0',n,n) for n in (1,2,3)]
        for mode,owner,prefix in phases:
            phase=dict(arm=arm,mode=mode,owner=owner,prefix=prefix,protocol=c.digest(payload['protocol']),code='CPU_FIXTURE')
            dest=R/'private/outputs'/arm/mode/str(owner)/str(prefix);ids=queue.expected_ids(tasks,ledger,mode,owner,prefix)
            for qid in ids:
                d=c.read(lookup[qid]);d['binding']['phase']=phase;c.write(dest/(qid+'.json'),d)
            c.write(dest/'COMPLETE.json',dict(phase=phase,outputs=len(ids)))
    queue.ingest(db);assert db.execute('SELECT count(*) FROM consumer').fetchone()[0]==0 and not queue.parent.terminal_allowed(db)
    c.write(R/'private/FINAL_CONSUMED.json',dict(test_only=True));c.write(R/'private/GENERATION_COMPLETE.json',dict(test_only=True))
    queue.ingest(db);queue.ingest(db);assert db.execute('SELECT count(*) FROM consumer').fetchone()[0]==483
    from evaluator import protection_ids
    target=protection_ids(tasks,ledger,'T2L')[0];key=db.execute("SELECT payload_key FROM consumer WHERE method='F' AND mode='protect100_R0' AND query_id=?",(target,)).fetchone()[0]
    db.execute("UPDATE payload SET status='MISSING',correct=NULL WHERE key=?",(key,));db.commit()
    for arm in ('F','S'):
        for owner in (1,2,3):
            for slot in range(1,161):c.write(R/'private/steps'/arm/str(owner)/(str(slot)+'.json'),dict(owner=owner,slot=slot,accepted=True,
                actual_native_norm=.001,source_CE=.2,risk_after=.001,candidate_diagnostics=[]))
    learner.old.b.COUNTS['attempt_slots']=960;learner.old.b.counts();c.write(q.ROOT/'ALL_WORKERS_COMPLETE.json',dict(test_only=True))
    import report_fs
    report_fs.main();final=c.read(R/'public/RESULTS.json')
    assert final['status']=='COMPLETE_WITH_MISSING' and not final['gates']['F']['zero_individual_Base_damage']
    assert next(x for x in final['contrasts'] if x['comparison']=='S-minus-F')['delta_macro_bounds_pp']==[0.,0.]
    assert {x['task']:x['observations'] for x in final['protection'] if x['arm']=='F'}=={'T1L':2,'T2L':34}
    coverage=report_fs.partial_coverage({'M':161,'F':161,'S':161},db)
    assert coverage['terminal_consumers']=={'M':161,'F':161,'S':161}
    db.execute("UPDATE payload SET status='PENDING' WHERE key=?",(key,));db.commit()
    coverage=report_fs.partial_coverage({'M':161,'F':161,'S':161},db)
    assert sum(coverage['generated_not_terminal'].values())>0 and all(n==0 for n in coverage['generated_not_ingested'].values())
    assert report_fs.partial_coverage({'M':61,'F':0,'S':0},None)['generated_not_ingested']['M']==61
    result.update(partial_scoring_coverage=True)
    result.update(status='PASS',expected_consumers=483,full_protection_probes=36,idempotent_queue=True,full_generation_barrier=True,
        shared_missing_exact_cancellation=True,unknown_protection_cannot_pass=True,terminal_report_verified=True,GPU_updates=0,Judge_requests=0)
    print(json.dumps(result))
