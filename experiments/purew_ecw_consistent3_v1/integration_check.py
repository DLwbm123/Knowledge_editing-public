"""Disposable production operators, caller rollback and complete/scientific-stop cohort fixtures."""
import contextlib,io,json,os,sys,tempfile,shutil,time
from pathlib import Path
x=json.loads(sys.stdin.readline());root=Path(tempfile.mkdtemp(prefix='e.',dir='/data/bmw/tmp'));tools=root/'private/tools';tools.mkdir(parents=True)
parent=Path('/data/bmw/Knowledge_editing/outputs/purew-ecw-bcd3-v1-20261015/run/private')
for path in (parent/'tools').glob('*.py'):shutil.copy2(path,tools/path.name)
shutil.copy2(parent/'tools/partial_report.py',tools/'historical_binding.py')
for name,text in x['files'].items():(tools/name).write_text(text)
(root/'private/FROZEN_PROTOCOL.json').write_text(json.dumps(x['effective']))
(root/'private/GPU_SOURCE_VERSION.json').write_text(json.dumps(dict(commit='CPU_FIXTURE')))
os.environ.update(RUN_ROOT=str(root),ACTION='CPU_INTEGRATION',ARM='J',TMPDIR=str(root));sys.path.insert(0,str(parent/'source'));sys.path.insert(0,str(tools))
import controller,queue_revision,report_revision,worker,calibrator
from cpu_check import main,exact
from core import Memory
from panels import expected_ids
import torch
result=main(x['protocol']);q=queue_revision.q
pool=json.loads(Path('/data/bmw/Knowledge_editing/outputs/purew-protect100-v1-20261013/run/private/BASIS_BUFFER.json').read_text())
bound=[{k:pool[y['pool_index']][k] for k in ('dataset','source_qid','source_group','question','reference','image_sha256')} for y in x['locators']]
assert q.digest(bound)==x['protocol']['calibration']['binding'];result['frozen_calibration_binding']=True
for name in ('BENCHMARK146_QUEUE.json','EVAL_LEDGER.json'):shutil.copy2(parent/name,root/'private'/name)
tasks=json.loads((parent/'BENCHMARK146_QUEUE.json').read_text())['tasks'];ledger=json.loads((parent/'EVAL_LEDGER.json').read_text())
counts=[len(expected_ids(tasks,ledger,'bank_R0',0,z)) for z in [1,3]]+[sum(len(expected_ids(tasks,ledger,'insertion_R0',z,z)) for z in [1,2,3]),len(expected_ids(tasks,ledger,'protect100_R0',0,100))];assert counts==[46,64,15,36]
q.ROOT.mkdir(parents=True,exist_ok=True);q.write(q.ROOT/'EPOCH_MANIFEST.json',dict(epoch=x['protocol']['version'],model='gpt-6-astra',reasoning_effort='medium',protocol=q.PROTOCOL,prompt=q.PROMPT,attempts_per_payload=1,workers=4,batch_size=50))
db=q.connect();queue_revision.ingest(db);assert not db.execute('SELECT 1 FROM consumer').fetchone();assert not queue_revision.parent.terminal_allowed(db)
weight=torch.ones((4,5),dtype=torch.float16,requires_grad=True);master=weight.detach().float();risk=Memory(weight,'J');risk.consolidate(torch.ones((1,20),dtype=torch.float64)*.01,'Base')
worker.loss=lambda model,full,removed,anchors,slot:(weight.float().square().mean(),1.,0.)
worker.backward=lambda value,w:torch.autograd.grad(value,w)[0].float();worker.b.call=lambda kind:None;worker.b.counts=lambda:None
master,event=worker.attempt(None,weight,master,risk,[],[],[],1);assert event['accepted'] and event['actual_native_norm']>0
before=weight.detach().clone();old=master.clone();oldstate=risk.state()
risk.solve=lambda gradient:(torch.zeros(weight.numel(),dtype=torch.float64),dict(solver_reason='FINITE_DUAL_REJECT'))
after,event=worker.attempt(None,weight,master,risk,[],[],[],2);assert not event['accepted'] and torch.equal(after,old) and torch.equal(weight,before) and exact(oldstate,risk.state());del risk.solve
# Real hard-gate caller with differentiable current KL, but unacceptable actual-native KL.
import medical
original_logits=worker.b.logits;worker.b.logits=lambda model,batch:(torch.stack([weight.float().flatten(),-weight.float().flatten()],dim=1),torch.zeros(20,dtype=torch.long))
original_direct=medical.direct_KLs;medical.direct_KLs=lambda *args:[.02]*5
anchor=worker.b.logits(None,None)[0].detach().double().softmax(-1)
after,rejected=worker.attempt(None,weight,master,risk,[],[None]*5,[anchor]*5,3,hard_removed=True)
assert not rejected['accepted'] and torch.equal(after,master) and torch.equal(weight,before) and exact(oldstate,risk.state())
assert any('CURRENT_REMOVED_HARD_KL' in z['reasons'] for z in rejected['candidates']);assert 'current_linear_predicted_KLs' in rejected
medical.direct_KLs=original_direct;worker.b.logits=original_logits
for name,data in [('RUN_MANIFEST.json',dict(deadline_epoch=time.time()+600,GPU_seconds_limit=3600,min_free_bytes=0,owned_weight_limit_bytes=1024**3)),('STAGE_CAP.json',dict(GPU_seconds_limit=3600)),('RESOURCE_LEDGER.json',dict(gpu_seconds_used=0,gpu_sessions=[],weights_observed_peak_bytes=0))]:q.write(root/name,data)
worker.c.rng=lambda:dict(torch_rng=torch.get_rng_state(),python_rng=__import__('random').getstate(),cuda_rng=None)
worker.save(weight,master,risk,1,2,False,event,[]);saved=worker.validate_state(torch.load(root/'private/active/J/state.pt',weights_only=True));assert torch.equal(saved['native'],weight) and exact(saved['memory'],risk.state())
bad=dict(saved,clocks=dict(saved['clocks'],FIT=3))
try:worker.validate_state(bad)
except AssertionError:pass
else:raise AssertionError('Corrupt clock accepted')
prior=queue_revision.parent.ro(parent/'judge_purew146/queue.sqlite');oldrows=list(prior.execute("SELECT * FROM consumer WHERE method='A'"));assert len(oldrows)==161
terminals={a:dict(prefix=1 if a=='J' else 3,status='SCIENTIFIC_STOP' if a=='J' else 'COMPLETE',consumers=87 if a=='J' else 161) for a in ('N','J','NJ')}
for arm,t in terminals.items():
    panels={}
    for row in oldrows:
        if t['prefix']==1 and (row['mode']=='bank_R0' and row['prefix']!=1 or row['mode']=='insertion_R0' and row['edit_order']!=1):continue
        data=q.read(row['path']);phase=dict(data['binding']['phase'],arm=arm,protocol=q.digest(x['effective']),code='CPU_FIXTURE');data['binding']['phase']=phase
        dest=root/'private/outputs'/arm/row['mode']/str(phase['owner'])/str(row['prefix'])/(row['query_id']+'.json');q.write(dest,data);panels[str(dest.parent)]=phase
    for folder,phase in panels.items():q.write(Path(folder)/'COMPLETE.json',dict(phase=phase,outputs=len(list(Path(folder).glob('*.json')))))
    for owner in range(1,t['prefix']+1):
        for slot in range(1,21):q.write(root/'private/steps'/arm/str(owner)/(str(slot)+'.json'),dict(slot=slot,accepted=True,zero_native=False,candidates=[],current_risk={}))
        q.write(root/'public'/('SOURCE_'+arm+'_'+str(owner)+'.json'),dict(final=dict(weighted_all_FIT_CE=.05)));q.write(root/'public'/('PATH_'+arm+'_'+str(owner)+'.json'),{})
q.write(root/'private/ARM_TERMINALS.json',terminals);q.write(root/'private/FINAL_CONSUMED.json',{})
q.write(root/'private/GENERATION_COMPLETE.json',dict(new_consumers=409,full_protocol_complete=False))
queue_revision.ingest(db);assert db.execute('SELECT count(*) FROM consumer').fetchone()[0]==409
assert not db.execute("SELECT 1 FROM payload WHERE status IN ('PENDING','RESERVED')").fetchone();assert queue_revision.parent.terminal_allowed(db)
q.write(q.ROOT/'ALL_WORKERS_COMPLETE.json',{});q.write(root/'RESOURCE_LEDGER.json',dict(gpu_seconds_used=0,gpu_sessions=[],Judge_attempts=0,Astra_purew146_batches=[]))
with contextlib.redirect_stdout(io.StringIO()):report_revision.main()
report=q.read(root/'public/RESULTS.json');assert report['status']=='PARTIAL_SCIENTIFIC_STOP' and not report['scientific_success']
assert report['all_consumers']==570 and len(report['protection'])==8
assert all(sum(p['Base_correct_to_wrong_bounds'][0] for p in report['protection'] if p['arm']==arm)==4 for arm in ('N','J','NJ'))
assert report['contrasts'][1]['status']=='NA_MATCHED_PREFIX3_NOT_GENERATED'
# Report the all-complete route with the same production ingest, not a second model trajectory.
arm='J';panels={}
for row in oldrows:
    if row['mode']=='protect100_R0' or row['mode']=='bank_R0' and row['prefix']==1 or row['mode']=='insertion_R0' and row['edit_order']==1:continue
    data=q.read(row['path']);phase=dict(data['binding']['phase'],arm=arm,protocol=q.digest(x['effective']),code='CPU_FIXTURE');data['binding']['phase']=phase
    dest=root/'private/outputs'/arm/row['mode']/str(phase['owner'])/str(row['prefix'])/(row['query_id']+'.json');q.write(dest,data);panels[str(dest.parent)]=phase
for folder,phase in panels.items():q.write(Path(folder)/'COMPLETE.json',dict(phase=phase,outputs=len([p for p in Path(folder).glob('*.json') if p.name!='COMPLETE.json'])))
for owner in (2,3):
    for slot in range(1,21):q.write(root/'private/steps'/arm/str(owner)/(str(slot)+'.json'),dict(slot=slot,accepted=True,zero_native=False,candidates=[],current_risk={}))
    q.write(root/'public'/('SOURCE_'+arm+'_'+str(owner)+'.json'),dict(final=dict(weighted_all_FIT_CE=.05)));q.write(root/'public'/('PATH_'+arm+'_'+str(owner)+'.json'),{})
terminals['J']=dict(prefix=3,status='COMPLETE',consumers=161);q.write(root/'private/ARM_TERMINALS.json',terminals);q.write(root/'private/GENERATION_COMPLETE.json',dict(new_consumers=483,full_protocol_complete=True));queue_revision.ingest(db)
(root/'private/REPORT_COMPLETE.json').unlink()
with contextlib.redirect_stdout(io.StringIO()):report_revision.main()
report=q.read(root/'public/RESULTS.json');assert report['all_consumers']==644 and len(report['contrasts'])==4 and not report['scientific_success'];assert all(a['status']=='ESTIMATED' for a in report['contrasts'])
result.update(production_imports=True,calibration_dual_factor_formula_CPU_only=True,real_attempt_native_accept_reject_rollback=True,current_KL_direction_and_true_gate_reject_rollback=True,atomic_save_fsync_quota=True,clock_corruption_rejected=True,generation_barrier_no_early_terminal=True,partial409_and_complete483_new_consumers=True,historical_B_external161_no_new_Judge=True,protection_damage_prevents_success=True,matched_prefix3_NA_when_unmeasured=True,scoring_counts=counts,disposable_root=str(root),synthetic_Judge_calls=0)
print(json.dumps(result))
