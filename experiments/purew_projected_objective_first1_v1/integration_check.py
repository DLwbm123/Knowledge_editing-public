# Executed after cpu_check in same disposable CPU process.
import contextlib,subprocess,controller,queue_revision,report_revision
q=queue_revision.q
for name in ('BENCHMARK146_QUEUE.json','EVAL_LEDGER.json'):shutil.copy2(parent/name,root/'private'/name)
q.ROOT.mkdir(parents=True,exist_ok=True)
q.write(q.ROOT/'EPOCH_MANIFEST.json',dict(epoch=payload['protocol']['version'],model='gpt-6-astra',reasoning_effort='medium',protocol=q.PROTOCOL,prompt=q.PROMPT,attempts_per_payload=1,workers=4,batch_size=50))
db=q.connect();queue_revision.ingest(db)
assert not db.execute('SELECT 1 FROM consumer').fetchone() and not queue_revision.parent.terminal_allowed(db)
q.write(root/'public/CPU_VALIDATION.json',dict(status='PASS'));q.write(root/'public/BASE_PARAMETER_REUSE.json',dict(status='PASS'))
q.write(root/'private/operator/source.json',dict(rows=[{}]));q.write(root/'private/operator/review.json',dict(rows=[dict(region_status='ELIGIBLE_COARSE_BOX')]))
q.write(root/'RESOURCE_LEDGER.json',dict(gpu_sessions=[],gpu_seconds_used=0,Judge_attempts=0,Astra_purew146_batches=[]))
q.write(root/'RUN_MANIFEST.json',dict(deadline_epoch=time.time()+600,GPU_seconds_limit=3600,min_free_bytes=0,owned_weight_limit_bytes=1024**3))
q.write(root/'STAGE_CAP.json',dict(GPU_seconds_limit=3600))
controller.provide=lambda *args:None;controller.retire=lambda *args:None
calls=[]
def stage(action,arm,owner,phase=None):
    calls.append((action,arm))
    if action.startswith('admit_'):q.write(root/'public'/('REAL_VLM_ADMISSION_'+arm+'.json'),dict(status='PASS'))
    if action.startswith('learn_'):
        assert all((root/'public'/('REAL_VLM_ADMISSION_'+a+'.json')).exists() for a in ('PM','PU'))
        q.write(root/'private/active'/arm/'state.pt',{})
        for slot in range(1,21):q.write(root/'private/steps'/arm/'1'/(str(slot)+'.json'),dict(slot=slot,accepted=True,zero_native=False,candidates=[],current_risk={},complete_source_after_slot=dict(weighted_all_FIT_CE=.05)))
    if phase=='reload':q.write(root/'private'/('FINAL_CONSUMED_'+arm+'.json'),dict(status='PASS'))
for stopped in (False,True):
    calls.clear()
    if stopped:q.write(root/'private/control/PM/SCIENTIFIC_STOP.json',dict(reason='EDIT_NOT_FIT'))
    with patch.object(controller,'stage',stage):controller.main()
    assert len(calls)==10 and all(a.startswith(('eval_base','admit_')) for a,_ in calls[:4])
    assert q.read(root/'private/GENERATION_COMPLETE.json')['full_protocol_complete']==(not stopped)
    assert q.read(root/'private/GENERATION_COMPLETE.json')['new_consumers']==174
# The real R state has an explicit external R identity and immutable internal J coordinates.
worker.c.rng=lambda:dict(torch_rng=torch.get_rng_state(),python_rng=__import__('random').getstate(),cuda_rng=None)
q.write(root/'RUN_MANIFEST.json',dict(deadline_epoch=time.time()+600,GPU_seconds_limit=3600,min_free_bytes=0,owned_weight_limit_bytes=1024**3))
q.write(root/'STAGE_CAP.json',dict(GPU_seconds_limit=3600))
master=weight.detach().float().clone()
worker.save(weight,master,risk,1,2,False,accepted,[])
saved=worker.validate_state(torch.load(root/'private/active/PM/state.pt',weights_only=True))
assert saved['arm']=='PM' and saved['memory']['arm']=='J'
bad=dict(saved,owner=2)
try:worker.validate_state(bad)
except AssertionError:pass
else:raise AssertionError('R accepted source2 state')

prior=queue_revision.parent.ro(parent/'judge_purew146/queue.sqlite')
oldrows=list(prior.execute("SELECT * FROM consumer WHERE method='SH00'"));assert len(oldrows)==87
for arm in ('PM','PU'):
    panels={}
    for row in oldrows:
        data=q.read(row['path']);phase=dict(data['binding']['phase'],arm=arm,protocol=q.digest(payload['protocol']),code='CPU_FIXTURE');data['binding']['phase']=phase
        dest=root/'private/outputs'/arm/row['mode']/str(phase['owner'])/str(row['prefix'])/(row['query_id']+'.json')
        q.write(dest,data);panels[str(dest.parent)]=phase
    for folder,phase in panels.items():q.write(Path(folder)/'COMPLETE.json',dict(phase=phase,outputs=len(list(Path(folder).glob('*.json')))))
    q.write(root/'public'/('SOURCE_'+arm+'_1.json'),dict(final=dict(weighted_all_FIT_CE=.05)))
    q.write(root/'public'/('PATH_'+arm+'_1.json'),{})
    for phase in ('INITIAL','FINAL'):q.write(root/'public'/('GRADIENT_'+phase+'_'+arm+'.json'),dict(norms=[1,0,0]))
    q.write(root/'public'/('MARGIN_DIAGNOSTIC_'+arm+'.json'),dict(finite=10,positive=10))
q.write(root/'private/GPU_SOURCE_VERSION.json',dict(commit='CPU_FIXTURE'))
queue_revision.ingest(db)
assert db.execute('SELECT count(*) FROM consumer').fetchone()[0]==174
assert not db.execute("SELECT 1 FROM payload WHERE status IN ('PENDING','RESERVED')").fetchone()
assert queue_revision.parent.terminal_allowed(db)
q.write(q.ROOT/'ALL_WORKERS_COMPLETE.json',{})
with contextlib.redirect_stdout(io.StringIO()):report_revision.main()
r=q.read(root/'public/RESULTS.json')
assert r['all_consumers']==261 and all(x==dict(FORMAT_VALID=87) for x in r['coverage'].values())
assert r['status']=='PARTIAL_SCIENTIFIC_STOP' and not r['scientific_success']
assert all(x['delta_macro_bounds_pp']==[0.,0.] for x in r['contrasts'])
assert all(x['damage_count_difference_bounds']==[0.,0.] for x in r['protection_contrasts'])
assert all(not r['gates'][a]['zero_individual_Base_damage'] for a in ('PM','PU'))
(root/'private/REPORT_COMPLETE.json').unlink()
g=q.read(root/'private/GENERATION_COMPLETE.json');g['full_protocol_complete']=True;q.write(root/'private/GENERATION_COMPLETE.json',g)
(root/'private/source').symlink_to(parent/'source',target_is_directory=True)
env=dict(os.environ,PYTHONPATH='')
rp=subprocess.run([sys.executable,'-c',"import os,runpy;runpy.run_path(os.environ['RUN_ROOT']+'/private/tools/report_revision.py',run_name='__main__')"],env=env,text=True,capture_output=True)
assert rp.returncode==0,rp.stderr
assert q.read(root/'public/RESULTS.json')['status']=='COMPLETE'
# Exact shared unknown-variable cancellation and non-cancelled worst cases.
agg=report_revision.r
assert list(agg.score_bounds(agg.combine([{'u':1.},{'u':-1.}]),{'u':None}))==[0.,0.]
assert list(agg.score_bounds({'u':1.,'v':-1.},{'u':None,'v':None}))==[-1.,1.]
# Exercise the actual per-arm/global counter guards and residency cap.
import runtime
for x in (root/'private/attempts').glob('*.json'):x.unlink()
os.environ['ACTION']='learn_PM_1'
for key,limit in [('attempt_slots',20),('committed_updates',21),('backwards',300),('forwards',15000),('generation_attempts',92),('candidate_checks',160),('active_set_checks',0)]:
    worker.b.COUNTS=dict(forwards=0,backwards=0,committed_updates=0,generation_attempts=0)
    worker.b.COUNTS[key]=limit
    runtime.counts()
    worker.b.COUNTS[key]=limit+1
    try:runtime.counts()
    except AssertionError:pass
    else:raise AssertionError(('Missing per-arm cap',key))
q.write(root/'RUN_MANIFEST.json',dict(deadline_epoch=time.time()+600,GPU_seconds_limit=21600,min_free_bytes=0,owned_weight_limit_bytes=1024**3))
q.write(root/'STAGE_CAP.json',dict(GPU_seconds_limit=21600))
q.write(root/'RESOURCE_LEDGER.json',dict(gpu_sessions=[dict(arm='PM',started_epoch=time.time()-7201,ended_epoch=time.time(),resident_seconds=7201)],gpu_seconds_used=7201,Judge_attempts=0,Astra_purew146_batches=[]))
try:runtime.budget()
except AssertionError:pass
else:raise AssertionError('Per-arm residency budget transferable')
print(json.dumps(dict(status='PASS',CPU_seconds=time.monotonic()-started,production_projection_gradient_oracle=True,extreme_logits_finite_no_floor=True,expanded_image_mask_and_corruption_rejection=True,PM_no_prefix_gradient=True,PU_prefix_gradient=True,SH00_candidate_path_bitwise=True,rejected_native_master_risk_rollback=True,state_roundtrip_next_step_exact=True,clock_corruption_rejected=True,two_arms_admitted_before_slots=True,queue_generation_barrier=True,report_261_consumers=True,shared_payload_difference_exact=True,damage_fails_success=True,per_arm_and_total_budget_limits=True,diagnostic_RNG_and_modes_preserved=True,CUDA_calls=0,real_model_calls=0,Judge_calls=0)))
