"""Production singleton-J controller, frozen math, queue/report, and zero-gap counterexample."""
import contextlib,io,json,os,sys,tempfile,shutil,time,subprocess
from pathlib import Path
x=json.loads(sys.stdin.readline());root=Path(tempfile.mkdtemp(prefix='e.',dir='/data/bmw/tmp'));tools=root/'private/tools';tools.mkdir(parents=True)
parent=Path('/data/bmw/Knowledge_editing/outputs/purew-ecw-bcd3-v1-20261015/run/private')
for p in (parent/'tools').glob('*.py'):shutil.copy2(p,tools/p.name)
shutil.copy2(parent/'tools/partial_report.py',tools/'historical_binding.py')
for name,text in x['files'].items():(tools/name).write_text(text)
os.environ.update(RUN_ROOT=str(root),ACTION='CPU_J_ONLY',ARM='J',TMPDIR=str(root));sys.path[:0]=[str(tools),str(parent/'source')]
(root/'private/FROZEN_PROTOCOL.json').write_text(json.dumps(x['protocol']))
(root/'private/source').symlink_to((parent/'source').resolve(),target_is_directory=True)
subprocess.run([sys.executable,'-c',"import runpy;runpy.run_path("+repr(str(tools/'report_revision.py'))+",run_name='production_entry_import_check')"],env=dict(os.environ,PYTHONPATH=''),check=True,capture_output=True,text=True)
from cpu_check import main as math_check
result=math_check(x['protocol'])
import controller,queue_revision,report_revision,worker
c=controller.c;q=queue_revision.q
for n in ('BENCHMARK146_QUEUE.json','EVAL_LEDGER.json'):shutil.copy2(parent/n,root/'private'/n)
(root/'private/operator').mkdir();shutil.copy2(parent/'operator/source.json',root/'private/operator/source.json');shutil.copy2(parent/'operator/review.json',root/'private/operator/review.json')
for n in ('CPU_VALIDATION','J_CONTINUATION_CPU','J_PLAIN_BASE_REUSE'):c.write(root/'public'/(n+'.json'),dict(status='PASS'))
(root/'private/calibration').mkdir();(root/'private/calibration/Base_plain.pt').write_bytes(b'CPU_FIXTURE_NOT_A_MODEL')
c.write(root/'private/GPU_SOURCE_VERSION.json',dict(commit='CPU_FIXTURE'))
c.write(root/'RESOURCE_LEDGER.json',dict(gpu_seconds_used=17.175475358963013,gpu_sessions=[],Judge_attempts=0,Astra_purew146_batches=[]))
c.write(root/'private/attempts/calibration_fixture.json',dict(attempt_slots=0,backwards=4,generation_attempts=2))
# Any attempt to enter normalized/shared calibration or a non-J stage is a test failure.
controller.calibration=lambda:(_ for _ in ()).throw(AssertionError('J invoked normalized calibration'))
controller.provide=lambda *args,**kw:None;controller.retire=lambda *args:None
stage_calls=[];stop=True

def stage(action,arm,owner,phase=None):
    assert arm=='J';stage_calls.append((action,owner,phase))
    if action.startswith('learn_'):
        c.write(root/'private/attempts'/('learn_fixture_'+str(owner)+'.json'),dict(attempt_slots=20))
        p=root/'private/active/J/state.pt';p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'CPU_ONLY')
        if stop:c.write(root/'private/control/J/SCIENTIFIC_STOP.json',dict(status='SCIENTIFIC_STOP'))
    if phase=='reload':c.write(root/'private/FINAL_CONSUMED_J.json',dict(status='PASS'))
controller.stage=stage
state={}
assert not controller.validation_ready(state,[],72830,2)
assert not controller.validation_ready(state,[123],50000,22000)
assert not controller.validation_ready(state,[],72830,2)
assert not controller.validation_ready(state,[],72830,2)
assert controller.validation_ready(state,[],72830,2)
controller.wait_user_validation=lambda:None
controller.main();gen=c.read(root/'private/GENERATION_COMPLETE.json')
assert gen['new_consumers']==87 and not gen['full_protocol_complete'];assert not (root/'private/TRAINING_GENERATION_COMPLETE.json').exists()
assert not any('N' in action or 'calibration' in action for action,_,_ in stage_calls)
assert c.read(root/'RESOURCE_LEDGER.json')['gpu_seconds_used']==17.175475358963013
assert c.read(root/'private/attempts/calibration_fixture.json')['backwards']==4
# Production ingest/report of the actual-terminal singleton cohort; evaluation fixtures never enter a learner.
prior=queue_revision.parent.ro(parent/'judge_purew146/queue.sqlite');oldrows=list(prior.execute("SELECT * FROM consumer WHERE method='A'"));assert len(oldrows)==161
q.ROOT.mkdir(parents=True,exist_ok=True);c.write(q.ROOT/'EPOCH_MANIFEST.json',dict(epoch=x['protocol']['version'],model='gpt-6-astra',reasoning_effort='medium',protocol=q.PROTOCOL,prompt=q.PROMPT,attempts_per_payload=1,workers=4,batch_size=50))
db=q.connect();(root/'private/GENERATION_COMPLETE.json').unlink();queue_revision.ingest(db);assert not queue_revision.parent.terminal_allowed(db)

def fixtures(prefix):
    panels={}
    for row in oldrows:
        if prefix==1 and (row['mode']=='bank_R0' and row['prefix']!=1 or row['mode']=='insertion_R0' and row['edit_order']!=1):continue
        data=c.read(row['path']);phase=dict(data['binding']['phase'],arm='J',protocol=q.digest(x['protocol']),code='CPU_FIXTURE');data['binding']['phase']=phase
        dest=root/'private/outputs/J'/row['mode']/str(phase['owner'])/str(row['prefix'])/(row['query_id']+'.json');c.write(dest,data);panels[str(dest.parent)]=phase
    for folder,phase in panels.items():c.write(Path(folder)/'COMPLETE.json',dict(phase=phase,outputs=len([p for p in Path(folder).glob('*.json') if p.name!='COMPLETE.json'])))
    for owner in range(1,prefix+1):
        for slot in range(1,21):c.write(root/'private/steps/J'/str(owner)/(str(slot)+'.json'),dict(slot=slot,accepted=True,zero_native=False,candidates=[],current_risk={}))
        c.write(root/'public'/('SOURCE_J_'+str(owner)+'.json'),dict(final=dict(weighted_all_FIT_CE=.05)));c.write(root/'public'/('PATH_J_'+str(owner)+'.json'),{})
fixtures(1);c.write(root/'private/GENERATION_COMPLETE.json',gen);queue_revision.ingest(db)
assert db.execute('SELECT count(*) FROM consumer').fetchone()[0]==87 and queue_revision.parent.terminal_allowed(db)
c.write(q.ROOT/'ALL_WORKERS_COMPLETE.json',{})
with contextlib.redirect_stdout(io.StringIO()):report_revision.main()
r=c.read(root/'public/RESULTS.json');assert r['status']=='PARTIAL_SCIENTIFIC_STOP' and r['all_consumers']==248 and not r['scientific_success']
assert r['contrasts'][0]['status']=='NA_MATCHED_PREFIX3_NOT_GENERATED' and len(r['prefix1_secondary'])==1
assert r['deferred_arms']==dict(N='NA_NOT_STARTED_ZERO_GAP_THEORY_PENDING',NJ='NA_NOT_STARTED_ZERO_GAP_THEORY_PENDING')
# Exercise the complete three-source controller path and the corresponding 161-output queue.
stop=False;(root/'private/control/J/SCIENTIFIC_STOP.json').unlink();controller.main()
assert c.read(root/'private/GENERATION_COMPLETE.json')['new_consumers']==161
assert c.read(root/'private/TRAINING_GENERATION_COMPLETE.json')['formal_slots']==60
fixtures(3);queue_revision.ingest(db);assert db.execute('SELECT count(*) FROM consumer').fetchone()[0]==161
(root/'private/REPORT_COMPLETE.json').unlink()
with contextlib.redirect_stdout(io.StringIO()):report_revision.main()
r=c.read(root/'public/RESULTS.json');assert r['all_consumers']==322 and len(r['contrasts'])==1 and r['contrasts'][0]['status']=='ESTIMATED' and not r['scientific_success']
assert sum(z['Base_correct_to_wrong_bounds'][0] for z in r['protection'] if z['arm']=='J')==4
# A finite vanishing KL cannot upper-bound singleton tie-break flips; the regret bound remains defined.
import torch
p=torch.tensor([.5,.5],dtype=torch.float64)
for e in (.1,.01,.001):
    qq=torch.tensor([.5-e,.5+e],dtype=torch.float64);kl=float((p*(p.log()-qq.log())).sum())
    assert qq.argmax()!=p.argmax() and abs(kl+.5*__import__('math').log(1-4*e*e))<1e-14
    regret=float(p.max()-p[qq.argmax()]);assert regret==0 and regret<=(2*kl)**.5
result.update(standalone_report_entry_empty_PYTHONPATH=True,J_singleton_calibration_decoupled=True,user_validation_empty_before_busy_cannot_launch=True,user_validation_busy_then60_seconds_idle=True,normalized_calibration_never_called=True,other_arms_never_launched=True,old_cost_counters_preserved=True,production_source1_stop87_and_full161=True,report_B_plus_J248_and322=True,unknown_and_damage_fail=True,prefix1_secondary_not_prefix3=True,zero_gap_no_continuous_singleton_flip_bound=True,regret_bound_no_floor=True,disposable_root=str(root))
print(json.dumps(result))
