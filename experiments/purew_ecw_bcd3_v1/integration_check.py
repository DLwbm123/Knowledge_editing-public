"""Disposable production imports/queue and mocked source-attempt exercise; no model/CUDA."""
import json,os,sys,tempfile,shutil
from pathlib import Path
x=json.loads(sys.stdin.readline());root=Path(tempfile.mkdtemp(prefix='e.',dir='/data/bmw/tmp'));tools=root/'private/tools';tools.mkdir(parents=True)
parent=Path('/data/bmw/Knowledge_editing/outputs/purew-ecw-revision2-v1-20261014/run/private')
for path in (parent/'tools').glob('*.py'):shutil.copy2(path,tools/path.name)
for name,text in x['files'].items():(tools/name).write_text(text)
(root/'private/FROZEN_PROTOCOL.json').write_text(json.dumps(x['effective']))
(root/'private/GPU_SOURCE_VERSION.json').write_text(json.dumps(dict(commit='CPU_FIXTURE')))
os.environ.update(RUN_ROOT=str(root),ACTION='CPU_INTEGRATION',ARM='C',TMPDIR=str(root))
sys.path.insert(0,str(parent/'source'));sys.path.insert(0,str(tools))
import controller,queue_revision,report_revision,worker,calibrator
from cpu_check import main,exact
from core import Memory
from panels import expected_ids
import torch
result=main(x['protocol'])
pool=json.loads(Path('/data/bmw/Knowledge_editing/outputs/purew-protect100-v1-20261013/run/private/BASIS_BUFFER.json').read_text())
bound=[{k:pool[x['pool_index']][k] for k in ('dataset','source_qid','source_group','question','reference','image_sha256')} for x in x['locators']]
assert queue_revision.q.digest(bound)==x['protocol']['calibration']['binding']
result['frozen_calibration_parameter_input_binding']=True
tasks=json.loads((parent/'BENCHMARK146_QUEUE.json').read_text())['tasks'];ledger=json.loads((parent/'EVAL_LEDGER.json').read_text())
counts=[len(expected_ids(tasks,ledger,'bank_R0',0,z)) for z in [1,3]]+[sum(len(expected_ids(tasks,ledger,'insertion_R0',z,z)) for z in [1,2,3]),len(expected_ids(tasks,ledger,'protect100_R0',0,100))]
assert counts==[46,64,15,36]
q=queue_revision.q;q.ROOT.mkdir(parents=True,exist_ok=True)
(q.ROOT/'EPOCH_MANIFEST.json').write_text(json.dumps(dict(epoch=x['protocol']['version'],model='gpt-6-astra',reasoning_effort='medium',protocol=q.PROTOCOL,prompt=q.PROMPT,attempts_per_payload=1,workers=4,batch_size=50)))
db=q.connect();queue_revision.ingest(db)
assert db.execute("SELECT count(*) FROM consumer WHERE method='A'").fetchone()[0]==161
assert not db.execute("SELECT 1 FROM done WHERE name='generation'").fetchone()
assert not queue_revision.parent.terminal_allowed(db)
# Exercise the real attempt caller: accepted update followed by finite solver rejection,
# preserving native/master and every independent risk block. Mock only model-dependent loss.
weight=torch.ones((4,5),dtype=torch.float16,requires_grad=True);master=weight.detach().float();risk=Memory(weight,'C')
risk.consolidate(torch.ones((1,20),dtype=torch.float64)*.01,'Base')
for name in ('loss','backward'):
    assert hasattr(worker,name)
worker.loss=lambda model,full,removed,anchors,slot:(weight.float().square().mean(),1.,0.)
worker.backward=lambda value,w:torch.autograd.grad(value,w)[0].float()
worker.b.call=lambda kind:None;worker.b.counts=lambda:None
master,event=worker.attempt(None,weight,master,risk,[],[],[],1)
assert event['accepted'] and event['actual_native_norm']>0
before=weight.detach().clone();old=master.clone();oldstate=io_state=risk.state()
risk.solve=lambda gradient:(torch.zeros(weight.numel(),dtype=torch.float64),dict(solver_reason='FINITE_DUAL_REJECT'))
after,event=worker.attempt(None,weight,master,risk,[],[],[],2)
assert not event['accepted'] and torch.equal(after,old) and torch.equal(weight,before) and exact(oldstate,risk.state())
del risk.solve
import medical
original_direct=medical.direct_KLs
medical.direct_KLs=lambda *args:[.02]*5
after,rejected=worker.attempt(None,weight,master,risk,[],[None]*5,[],3,hard_removed=True)
assert not rejected['accepted'] and torch.equal(after,master) and torch.equal(weight,before) and exact(oldstate,risk.state())
assert any('CURRENT_REMOVED_HARD_KL' in z['reasons'] for z in rejected['candidates'])
assert all('CURRENT_REMOVED_HARD_KL' in z['reasons'] or 'ZERO_NATIVE' in z['reasons'] for z in rejected['candidates'])
medical.direct_KLs=original_direct
result['each_view_hard_KL_reject_and_rollback']=True
for name,data in [('RUN_MANIFEST.json',dict(deadline_epoch=__import__('time').time()+60,GPU_seconds_limit=3600,min_free_bytes=0,owned_weight_limit_bytes=1024**3)),('STAGE_CAP.json',dict(GPU_seconds_limit=3600)),('RESOURCE_LEDGER.json',dict(gpu_seconds_used=0,gpu_sessions=[],weights_observed_peak_bytes=0))]:
    (root/name).write_text(json.dumps(data))
worker.c.rng=lambda:dict(torch_rng=torch.get_rng_state(),python_rng=__import__('random').getstate(),cuda_rng=None)
worker.save(weight,master,risk,1,2,False,event,[])
saved=worker.validate_state(torch.load(root/'private/active/C/state.pt',weights_only=True))
assert torch.equal(saved['native'],weight) and exact(saved['memory'],risk.state())
bad=dict(saved,clocks=dict(saved['clocks'],FIT=3))
try:worker.validate_state(bad)
except AssertionError:pass
else:raise AssertionError('Damaged source clock accepted')
# Round-trip the production queue and anonymous report using immutable A outputs as
# evaluation-only synthetic B/C/D fixtures. They never enter a learner or a Judge.
import contextlib,io
prior=queue_revision.parent.ro(parent.parent/'private/judge_purew146/queue.sqlite')
oldrows=list(prior.execute("SELECT * FROM consumer WHERE method='Y'"))
for arm in ('B','C','D'):
    panels={}
    for row in oldrows:
        data=json.loads(Path(row['path']).read_text());phase=dict(data['binding']['phase'],arm=arm,protocol=q.digest(x['effective']),code='CPU_FIXTURE')
        data['binding']['phase']=phase
        dest=root/'private/outputs'/arm/row['mode']/str(phase['owner'])/str(row['prefix'])/(row['query_id']+'.json')
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(data));panels[str(dest.parent)]=phase
    for folder,phase in panels.items():
        count=len(list(Path(folder).glob('*.json')))
        (Path(folder)/'COMPLETE.json').write_text(json.dumps(dict(phase=phase,outputs=count)))
    for owner in (1,2,3):
        folder=root/'private/steps'/arm/str(owner);folder.mkdir(parents=True,exist_ok=True)
        for slot in range(1,21):
            (folder/(str(slot)+'.json')).write_text(json.dumps(dict(slot=slot,accepted=True,zero_native=False,candidates=[],current_risk={})))
        (root/'public').mkdir(exist_ok=True)
        (root/'public'/('SOURCE_'+arm+'_'+str(owner)+'.json')).write_text(json.dumps(dict(final=dict(weighted_all_FIT_CE=.05))))
        (root/'public'/('PATH_'+arm+'_'+str(owner)+'.json')).write_text('{}')
for name in ('BENCHMARK146_QUEUE.json','EVAL_LEDGER.json'):shutil.copy2(parent/name,root/'private'/name)
(root/'private/FINAL_CONSUMED.json').write_text('{}');(root/'private/GENERATION_COMPLETE.json').write_text('{}')
queue_revision.ingest(db)
assert db.execute('SELECT count(*) FROM consumer').fetchone()[0]==644
assert not db.execute("SELECT 1 FROM payload WHERE status IN ('PENDING','RESERVED')").fetchone()
assert queue_revision.parent.terminal_allowed(db)
(q.ROOT/'ALL_WORKERS_COMPLETE.json').write_text('{}')
(root/'RESOURCE_LEDGER.json').write_text(json.dumps(dict(gpu_seconds_used=0,gpu_sessions=[],Judge_attempts=0,Astra_purew146_batches=[])))
with contextlib.redirect_stdout(io.StringIO()):report_revision.main()
report=json.loads((root/'public/RESULTS.json').read_text())
assert report['all_consumers']==644 and not report['scientific_success']
assert len(report['contrasts'])==5 and len(report['protection'])==8
assert all(sum(p['Base_correct_to_wrong_bounds'][0] for p in report['protection'] if p['arm']==arm)==4 for arm in ('A','B','C','D'))
result.update(production_queue_report_644_fixture=True,protection_damage_prevents_success=True,synthetic_fixture_Judge_calls=0,production_atomic_save_fsync_quota=True,source_clock_corruption_rejected=True,production_imports=True,scoring_metadata_counts=counts,A_reused_consumers=161,generation_barrier_no_terminal=True,real_attempt_native_accept_and_reject_rollback=True,
    one_attempt_scoring_inherited=True,disposable_root=str(root))
print(json.dumps(result))
