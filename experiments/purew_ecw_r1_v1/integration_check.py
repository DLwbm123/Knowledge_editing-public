# Executed after cpu_check.py in the same disposable CPU process; never a run install.
import contextlib,subprocess
import controller,queue_revision,report_revision
q=queue_revision.q
for name in ('BENCHMARK146_QUEUE.json','EVAL_LEDGER.json'):
    shutil.copy2(parent/name,root/'private'/name)
q.ROOT.mkdir(parents=True,exist_ok=True)
q.write(q.ROOT/'EPOCH_MANIFEST.json',dict(epoch=payload['protocol']['version'],model='gpt-6-astra',reasoning_effort='medium',protocol=q.PROTOCOL,prompt=q.PROMPT,attempts_per_payload=1,workers=4,batch_size=50))
db=q.connect();queue_revision.ingest(db)
assert not db.execute('SELECT 1 FROM consumer').fetchone() and not queue_revision.parent.terminal_allowed(db)
# Exercise both controller branches without CUDA or source-case reads.
q.write(root/'public/CPU_VALIDATION.json',dict(status='PASS'))
q.write(root/'public/BASE_PARAMETER_REUSE.json',dict(status='PASS'))
q.write(root/'private/operator/source.json',dict(rows=[{}]))
q.write(root/'private/operator/review.json',dict(rows=[dict(region_status='ELIGIBLE_COARSE_BOX')]))
q.write(root/'RESOURCE_LEDGER.json',dict(gpu_sessions=[],gpu_seconds_used=0,Judge_attempts=0,Astra_purew146_batches=[]))
q.write(root/'private/attempts/fixture.json',dict(attempt_slots=20))
controller.provide=lambda *args:None;controller.retire=lambda *args:None
calls=[]
def stage(action,arm,owner,phase=None):
    assert arm=='R' and owner in (0,1)
    calls.append((action,owner))
    if action=='admit_R':q.write(root/'public/REAL_VLM_ADMISSION_R.json',dict(status='PASS'))
    if action=='eval_reload_R':q.write(root/'private/FINAL_CONSUMED_R.json',dict(status='PASS'))
for stopped in (False,True):
    calls.clear();q.write(root/'private/active/R/state.pt',{})
    if stopped:q.write(root/'private/control/R/SCIENTIFIC_STOP.json',dict(reason='EDIT_NOT_FIT'))
    with patch.object(controller,'stage',stage):controller.main()
    assert [a for a,o in calls]==['eval_base_R','admit_R','learn_R_1','eval_R_1','eval_reload_R']
    assert q.read(root/'private/GENERATION_COMPLETE.json')['full_protocol_complete']==(not stopped)
    assert q.read(root/'private/GENERATION_COMPLETE.json')['new_consumers']==87
# The real R state has an explicit external R identity and immutable internal J coordinates.
worker.c.rng=lambda:dict(torch_rng=torch.get_rng_state(),python_rng=__import__('random').getstate(),cuda_rng=None)
q.write(root/'RUN_MANIFEST.json',dict(deadline_epoch=time.time()+600,GPU_seconds_limit=3600,min_free_bytes=0,owned_weight_limit_bytes=1024**3))
q.write(root/'STAGE_CAP.json',dict(GPU_seconds_limit=3600))
worker.save(weight,master,risk,1,2,False,accepted,[])
saved=worker.validate_state(torch.load(root/'private/active/R/state.pt',weights_only=True))
assert saved['arm']=='R' and saved['memory']['arm']=='J'
bad=dict(saved,owner=2)
try:worker.validate_state(bad)
except AssertionError:pass
else:raise AssertionError('R accepted source2 state')
# Existing J outputs are fixtures only; no new model/Judge calls, same binding must share missing.
prior=queue_revision.parent.ro(parent/'judge_purew146/queue.sqlite')
oldrows=list(prior.execute("SELECT * FROM consumer WHERE method='J'"));assert len(oldrows)==87
panels={}
for row in oldrows:
    data=q.read(row['path']);phase=dict(data['binding']['phase'],arm='R',protocol=q.digest(payload['protocol']),code='CPU_FIXTURE');data['binding']['phase']=phase
    dest=root/'private/outputs/R'/row['mode']/str(phase['owner'])/str(row['prefix'])/(row['query_id']+'.json')
    q.write(dest,data);panels[str(dest.parent)]=phase
for folder,phase in panels.items():q.write(Path(folder)/'COMPLETE.json',dict(phase=phase,outputs=len(list(Path(folder).glob('*.json')))))
q.write(root/'private/GPU_SOURCE_VERSION.json',dict(commit='CPU_FIXTURE'))
for slot in range(1,21):q.write(root/'private/steps/R/1'/(str(slot)+'.json'),dict(slot=slot,accepted=True,zero_native=False,candidates=[],current_risk={}))
q.write(root/'public/SOURCE_R_1.json',dict(final=dict(weighted_all_FIT_CE=.05)))
q.write(root/'public/PATH_R_1.json',{})
q.write(root/'public/MARGIN_DIAGNOSTIC_R.json',dict(finite=10,positive=10))
queue_revision.ingest(db)
assert db.execute('SELECT count(*) FROM consumer').fetchone()[0]==87
assert not db.execute("SELECT 1 FROM payload WHERE status IN ('PENDING','RESERVED')").fetchone()
assert queue_revision.parent.terminal_allowed(db)
q.write(q.ROOT/'ALL_WORKERS_COMPLETE.json',{})
with contextlib.redirect_stdout(io.StringIO()):report_revision.main()
r=q.read(root/'public/RESULTS.json')
assert r['all_consumers']==174 and r['coverage']['J']==r['coverage']['R']==dict(FORMAT_VALID=69,MISSING=18)
assert r['status']=='PARTIAL_SCIENTIFIC_STOP' and not r['scientific_success']
assert r['contrasts'][0]['delta_macro_bounds_pp']==[0.,0.] and len(r['protection'])==4
assert not r['gates']['R']['zero_individual_Base_damage']
(root/'private/REPORT_COMPLETE.json').unlink()
g=q.read(root/'private/GENERATION_COMPLETE.json');g['full_protocol_complete']=True;q.write(root/'private/GENERATION_COMPLETE.json',g)
(root/'private/source').symlink_to(parent/'source',target_is_directory=True)
with contextlib.redirect_stdout(io.StringIO()):report_revision.main()
assert q.read(root/'public/RESULTS.json')['status']=='COMPLETE_WITH_UNKNOWN'
(root/'private/REPORT_COMPLETE.json').unlink()
env=dict(os.environ,PYTHONPATH='')
rp=subprocess.run([sys.executable,'-c',"import os,runpy;runpy.run_path(os.environ['RUN_ROOT']+'/private/tools/report_revision.py',run_name='__main__')"],env=env,text=True,capture_output=True)
assert rp.returncode==0,rp.stderr
print(json.dumps(dict(status='PASS',CPU_seconds=time.monotonic()-started,R_independent_identity=True,internal_J_coordinates=True,source1_only_both_controller_paths=True,actual_FP16_R_caller_accept_and_KL_rollback=True,corrupt_source2_rejected=True,no_early_Judge=True,shared_18_missing_no_rejudge=True,exact_shared_missing_difference_zero=True,report_174_consumers=True,partial_and_complete_unknown_report=True,standalone_report_empty_pythonpath=True,zero_damage_gate_fails_with_damage_or_unknown=True,CUDA_calls=0,real_model_calls=0,Judge_calls=0,full_dimension_peak_not_measured=True,disposable_root=str(root))))
