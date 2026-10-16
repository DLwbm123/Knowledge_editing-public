# Executed after cpu_check in same disposable CPU process.
import contextlib,subprocess,controller,queue_revision,report_revision,configuration
q=queue_revision.q
historical_compatibility=configuration.check_history(payload['protocol'])
assert historical_compatibility['old_arms']==['SH00','SH10','PM','PU','HM','HU']
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
def provide(*args):
    q.write(root/'private/window/HR/current.json',{})
controller.provide=provide
# The real launch guard rejects preparation before inspecting/installing a run.
from unittest.mock import mock_open
try:configuration.require_execution(payload['protocol'],dict(protocol_version=configuration.VERSION,active_arms=['HR'],execution_authorized=False),{})
except AssertionError:pass
else:raise AssertionError('Preparation authorized execution')
with patch('sys.stdin',io.StringIO(json.dumps(dict(protocol=payload['protocol'],authorization=dict(protocol_version=configuration.VERSION,active_arms=['HR'],execution_authorized=False),hardware={},files=payload['files']))+'\n')),patch.object(Path,'exists',side_effect=AssertionError('Premature filesystem access')):
    try:exec(compile(payload['files']['bootstrap.py'],'bootstrap.py','exec'),{'__name__':'cpu_bootstrap_guard'})
    except AssertionError as exc:assert 'explicit execution authorization' in str(exc)
    else:raise AssertionError('Bootstrap admitted preparation')
# Local deployment and consumer entry points also refuse before launching child processes.
import deploy
q.write(root/'preparation/PROTOCOL.json',payload['protocol'])
q.write(root/'preparation/EXECUTION_AUTHORIZATION.json',dict(protocol_version=configuration.VERSION,active_arms=['HR'],execution_authorized=False))
q.write(root/'preparation/HARDWARE_AUTHORIZATION.json',{})
with patch.object(deploy,'OUT',root/'preparation'),patch.object(subprocess,'check_output',side_effect=AssertionError('Unexpected process')):
    try:deploy.main()
    except AssertionError as exc:assert 'explicit execution authorization' in str(exc)
    else:raise AssertionError('Local deployment admitted preparation')
# Only disposable controller fixtures receive synthetic execution permission; no stage is launched.
q.write(root/'public/PROTOCOL.json',payload['protocol'])
q.write(root/'public/EXECUTION_AUTHORIZATION.json',dict(protocol_version=configuration.VERSION,active_arms=['HR'],execution_authorized=True,install=True,model_calls=True,Judge=True))
q.write(root/'public/HARDWARE_AUTHORIZATION.json',dict(allocation_authorized=True,physical_GPU=0,GPU_UUID='GPU-CPU-FIXTURE'))
calls=[]
def stage(action,arm,owner,phase=None):
    calls.append((action,arm))
    if phase=='base':q.write(root/'public'/('BASE_IDENTITY_'+arm+'.json'),dict(status='PASS'))
    if action.startswith('admit_'):q.write(root/'public'/('REAL_VLM_ADMISSION_'+arm+'.json'),dict(status='PASS'))
    if action.startswith('learn_'):
        assert all(q.read(root/'public'/(kind+'_'+a+'.json'))['status']=='PASS' for a in ('HR',) for kind in ('BASE_IDENTITY','REAL_VLM_ADMISSION'))
        q.write(root/'private/active'/arm/'state.pt',{})
        q.write(root/'private/window'/arm/'teacher.json',{})
        for slot in range(1,21):q.write(root/'private/steps'/arm/'1'/(str(slot)+'.json'),dict(slot=slot,accepted=True,zero_native=False,candidates=[],current_risk={},complete_source_after_slot=dict(weighted_all_FIT_CE=.05)))
    if phase in ('owner','reload'):assert (root/'private/window'/arm/'teacher.json').exists()
    if phase=='reload':q.write(root/'private'/('FINAL_CONSUMED_'+arm+'.json'),dict(status='PASS'))
for stopped in (False,True):
    calls.clear()
    if stopped:q.write(root/'private/control/HR/SCIENTIFIC_STOP.json',dict(reason='EDIT_NOT_FIT'))
    with patch.object(controller,'stage',stage):controller.main()
    assert len(calls)==5 and all(a.startswith(('eval_base','admit_')) for a,_ in calls[:2])
    assert q.read(root/'private/GENERATION_COMPLETE.json')['full_protocol_complete']==(not stopped)
    assert q.read(root/'private/GENERATION_COMPLETE.json')['new_consumers']==87
    assert not (root/'private/window/HR/teacher.json').exists()
# Recovery executes only the formal/evaluation stages and keeps the original clock.
import restart_controller
calls.clear()
q.write(root/'public/RESTART1_RECEIPT.json',dict(restart_number=1,formal_slots_before=0,original_manifest=q.read(root/'RUN_MANIFEST.json')))
original_controller_stage=controller.stage
with patch.object(restart_controller,'original_stage',stage):restart_controller.main()
controller.stage=original_controller_stage
assert [a for a,_ in calls]==['learn_HR_1_restart1','eval_HR_1','eval_reload_HR']
receipt=q.read(root/'public/RESTART1_RECEIPT.json');receipt['original_manifest']['deadline_epoch']+=1
q.write(root/'public/RESTART1_RECEIPT.json',receipt)
with patch.object(restart_controller,'original_stage',side_effect=AssertionError('Must reject before launch')):
    try:restart_controller.main()
    except AssertionError:pass
    else:raise AssertionError('Restart allowed clock replacement')
# The real R state has an explicit external R identity and immutable internal J coordinates.
worker.c.rng=lambda:dict(torch_rng=torch.get_rng_state(),python_rng=__import__('random').getstate(),cuda_rng=None)
q.write(root/'RUN_MANIFEST.json',dict(deadline_epoch=time.time()+600,GPU_seconds_limit=3600,min_free_bytes=0,owned_weight_limit_bytes=1024**3))
q.write(root/'STAGE_CAP.json',dict(GPU_seconds_limit=3600))
master=weight.detach().float().clone()
worker.save(weight,master,risk,1,2,False,accepted,anchors)
saved=worker.validate_state(torch.load(root/'private/active/HR/state.pt',weights_only=True))
assert saved['arm']=='HR' and saved['memory']['arm']=='J'
bad=dict(saved,owner=2)
try:worker.validate_state(bad)
except AssertionError:pass
else:raise AssertionError('HR accepted source2 state')
for bad in (dict(saved,objective_recipe='ODDS_HINGE_ALPHA_095_V1'),dict(saved,arm='HM'),dict(saved,clocks=dict(saved['clocks'],FIT=99))):
    try:worker.validate_state(bad)
    except AssertionError:pass
    else:raise AssertionError('Old recipe, arm or clock admitted')

prior=queue_revision.parent.ro(parent/'judge_purew146/queue.sqlite')
oldrows=list(prior.execute("SELECT * FROM consumer WHERE method='SH00'"));assert len(oldrows)==87
for arm in ('HR',):
    panels={}
    for row in oldrows:
        data=q.read(row['path']);phase=dict(data['binding']['phase'],arm=arm,protocol=q.digest(payload['protocol']),code='CPU_FIXTURE');data['binding']['phase']=phase
        dest=root/'private/outputs'/arm/row['mode']/str(phase['owner'])/str(row['prefix'])/(row['query_id']+'.json')
        q.write(dest,data);panels[str(dest.parent)]=phase
    for folder,phase in panels.items():q.write(Path(folder)/'COMPLETE.json',dict(phase=phase,outputs=len(list(Path(folder).glob('*.json')))))
    q.write(root/'public'/('SOURCE_'+arm+'_1.json'),dict(final=dict(weighted_all_FIT_CE=.05,scope_KL=.2)))
    q.write(root/'public'/('PATH_'+arm+'_1.json'),{})
    for phase in ('INITIAL','FINAL'):q.write(root/'public'/('GRADIENT_'+phase+'_'+arm+'.json'),dict(norms=[1,0,0]))
    q.write(root/'public'/('MARGIN_DIAGNOSTIC_'+arm+'.json'),dict(finite=10,positive=10))
    for phase in ('INITIAL','FINAL'):q.write(root/'public'/('EVIDENCE_'+phase+'_'+arm+'.json'),details)
q.write(root/'private/GPU_SOURCE_VERSION.json',dict(commit='CPU_FIXTURE'))
queue_revision.ingest(db)
assert db.execute('SELECT count(*) FROM consumer').fetchone()[0]==87
assert not db.execute("SELECT 1 FROM payload WHERE status IN ('PENDING','RESERVED')").fetchone()
assert queue_revision.parent.terminal_allowed(db)
q.write(q.ROOT/'ALL_WORKERS_COMPLETE.json',{})
with contextlib.redirect_stdout(io.StringIO()):report_revision.main()
r=q.read(root/'public/RESULTS.json')
assert r['all_consumers']==609 and all(x==dict(FORMAT_VALID=87) for x in r['coverage'].values())
assert r['status']=='PARTIAL_SCIENTIFIC_STOP' and not r['scientific_success']
assert next(x for x in r['contrasts'] if x['comparison']=='HR-HM')['delta_macro_bounds_pp']==[0.,0.]
assert all(x['edit_ci'] is None and x['source_ci'] is None for x in r['contrasts'])
assert next(x for x in r['cross_arm_T2G'] if x['comparison']=='HM-to-HR')['known_transitions']=={'11':3,'00':1}
assert r['target_shape_protection_interaction']['comparison']=='(HR-HM)-(SH10-SH00)'
assert all(not r['gates'][a]['zero_individual_Base_damage'] for a in ('HR',))
assert r['actual_new_consumer_count']==87 and r['historical_external_consumers']==522
assert r['diagnostics']['HR_evidence']['INITIAL']==details
(root/'private/REPORT_COMPLETE.json').unlink()
g=q.read(root/'private/GENERATION_COMPLETE.json');g['full_protocol_complete']=True;q.write(root/'private/GENERATION_COMPLETE.json',g)
(root/'private/source').symlink_to(parent/'source',target_is_directory=True)
env=dict(os.environ,PYTHONPATH='')
rp=subprocess.run([sys.executable,'-c',"import os,runpy;runpy.run_path(os.environ['RUN_ROOT']+'/private/tools/report_revision.py',run_name='__main__')"],env=env,text=True,capture_output=True)
assert rp.returncode==0,rp.stderr
assert q.read(root/'public/RESULTS.json')['status']=='COMPLETE'
# Make only the disposable HR fixture perfect, then independently fail each science gate.
# Reuse existing payloads/outputs: this performs no new model call or Judge request.
hr_rows=list(db.execute("SELECT * FROM consumer WHERE method='HR'"))
keys={row['payload_key'] for row in hr_rows}
original_scores={key:db.execute('SELECT status,correct FROM payload WHERE key=?',(key,)).fetchone() for key in keys}
for key in keys:db.execute("UPDATE payload SET status='FORMAT_VALID',correct=1 WHERE key=?",(key,))
db.commit()
def rerun_report():
    (root/'private/REPORT_COMPLETE.json').unlink(missing_ok=True)
    with contextlib.redirect_stdout(io.StringIO()):report_revision.main()
    return q.read(root/'public/RESULTS.json')
perfect=rerun_report();assert perfect['scientific_success']
assert perfect['analysis']['decision']=='JOINT_GATES_PASS_NO_AUTOMATIC_EXPANSION'
source_file=root/'public/SOURCE_HR_1.json';source=q.read(source_file)
q.write(source_file,dict(final=dict(weighted_all_FIT_CE=.1,scope_KL=.2)))
assert not rerun_report()['scientific_success'];q.write(source_file,source)
margin_file=root/'public/MARGIN_DIAGNOSTIC_HR.json';margin=q.read(margin_file)
q.write(margin_file,dict(finite=10,positive=9))
assert not rerun_report()['scientific_success'];q.write(margin_file,margin)
# A permanent unknown payload must fail completeness without becoming pending/retried.
key=next(iter(keys));db.execute("UPDATE payload SET status='MISSING',correct=NULL WHERE key=?",(key,));db.commit()
assert not rerun_report()['scientific_success']
for key,(status,correct) in original_scores.items():db.execute('UPDATE payload SET status=?,correct=? WHERE key=?',(status,correct,key))
db.commit()
damaged=rerun_report();assert not damaged['scientific_success']
assert damaged['analysis']['decision']=='SOURCE_GATES_PASS_EXTERNAL_PROTECTION_FAIL'
assert damaged['analysis']['next_action']=='STOP_AFTER_DELIVERY_NO_RETRY_NO_TUNING_NO_NEXT_ARM'
# Exact shared unknown-variable cancellation and non-cancelled worst cases.
agg=report_revision.r
assert list(agg.score_bounds(agg.combine([{'u':1.},{'u':-1.}]),{'u':None}))==[0.,0.]
assert list(agg.score_bounds({'u':1.,'v':-1.},{'u':None,'v':None}))==[-1.,1.]
# Exercise the actual per-arm/global counter guards and residency cap.
import runtime
for x in (root/'private/attempts').glob('*.json'):x.unlink()
os.environ['ACTION']='learn_HR_1'
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
q.write(root/'RESOURCE_LEDGER.json',dict(gpu_sessions=[dict(arm='HR',started_epoch=time.time()-7201,ended_epoch=time.time(),resident_seconds=7201)],gpu_seconds_used=7201,Judge_attempts=0,Astra_purew146_batches=[]))
try:runtime.budget()
except AssertionError:pass
else:raise AssertionError('Per-arm residency budget transferable')
db.close();prior.close()
shutil.rmtree(root)
print(json.dumps(dict(status='PASS',successful_CPU_sandbox_retired=True,protocol_version=configuration.VERSION,CPU_seconds=time.monotonic()-started,
    restart_skips_admission_and_preserves_clock=True,formal_learner_20_slots_both_terminal_branches=True,production_HR_loss_and_gradient_oracle=True,removed_full_vocabulary_forward_KL=True,
    no_conditional_or_prefix_gradient=True,HR_candidate_uses_combined_Armijo=True,
    candidate_exception_native_rollback=True,SH00_candidate_path_bitwise=True,
    extreme_logits_finite_no_floor=True,expanded_image_mask_and_corruption_rejection=True,
    rejected_native_master_risk_rollback=True,state_roundtrip_next_step_exact=True,
    old_recipe_arm_and_clock_rejected=True,historical_compatibility_readonly=True,HR_Base_admission_before_slots=True,
    teacher_retired_only_after_reload=True,queue_generation_barrier=True,report_609_consumers=True,
    hinge_gradient_boundary_high_teacher=True,corrected_gap_aggregation=True,
    scalar_evidence_matches_original_margin=True,diagnostic_backwards_counted=True,
    single_source_no_population_CI=True,cross_arm_T2G_transitions=True,
    shared_payload_difference_exact=True,damage_fails_success=True,CE_boundary_margin_and_missing_fail_success=True,positive_joint_gate_fixture=True,
    single_arm_and_total_budget_limits=True,diagnostic_RNG_and_modes_preserved=True,
    preparation_launch_denied_before_mutation=True,CUDA_calls=0,real_model_calls=0,Judge_calls=0)))
