"""Operational full-lane admission, after the finite pilot has actually ended."""
import os
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile
import time
import torch
import runtime146 as worker
import comparison146
import contract
import trajectory

c,r,RUN=worker.c,worker.r,worker.RUN


def historical_view(old):
    import astra_queue as a
    index=c.read(RUN/'private/HISTORICAL_146_SCORE_INDEX1.json')
    payloads={}
    for folder in ('judge_astra_medium','judge_lora146_astra_medium','judge_astra_medium_recovery1'):
        with sqlite3.connect('file:'+str(old/'private'/folder/'queue.sqlite')+'?mode=ro',uri=True) as db:
            for key,binding,status,score in db.execute("SELECT key,binding,status,correct FROM payload WHERE status IN ('FORMAT_VALID','MISSING')"):
                full=json.loads(binding);assert a.q.digest(full)==key
                payloads.setdefault(key,[]).append((full,status,score))
    resolved={}
    for row in index:
        values=payloads[row['payload_key']]
        choices=[v for v in values if v[1:]==(row['status'],row['correct'])]
        assert choices and all(v[0]==choices[0][0] for v in values)
        if row['payload_key'] in resolved:assert resolved[row['payload_key']]==choices[0]
        resolved[row['payload_key']]=choices[0]
    path=RUN/'private/HISTORICAL_REUSE_VIEW1.sqlite';assert not path.exists()
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE payload(key TEXT PRIMARY KEY,binding TEXT,status TEXT,correct INTEGER)')
        db.executemany('INSERT INTO payload VALUES (?,?,?,?)',[(key,json.dumps(v[0],ensure_ascii=False),v[1],v[2]) for key,v in resolved.items()])
    c.write(RUN/'private/M13_146_REUSE_QUEUES.json',[str(path)])
    published=c.read(old/'public/LORA146_RESULTS.json')
    assert published['TT_existing_25_panels_reproduced'] and published['TT_no_retraining_or_rejudging']
    c.write(RUN/'private/HISTORICAL_PERFORMANCE_PANELS1.json',dict(
        comparison_type='HISTORICAL_DIFFERENT_TRAINING_RECIPE',recipe='native140/A2_80/W0_320',
        matched_native_FIT160_control=False,FIT_not_inferred_from_T1G=True,
        TT_parameters=published['TT_parameters'],LoRA_parameters=published['LoRA_parameters'],
        not_standard_unmodified_LoRA=published['not_standard_unmodified_LoRA'],
        panels=[v for v in published['panels'] if not (v['mode']=='bank_R0' and v['prefix']==1)],
        original_missing_bounds_preserved=True))


def evaluation_queries(old):
    queries={}
    with sqlite3.connect('file:'+str(old/'private/judge_astra_medium/queue.sqlite')+'?mode=ro',uri=True) as db:
        for binding, in db.execute("SELECT DISTINCT p.binding FROM consumer c JOIN payload p ON p.key=c.payload_key WHERE c.method='TT88_W0_RETRO146'"):
            full=json.loads(binding)
            row={key:full[key] for key in ('query_id','question','reference','image_sha256','image_path')}
            row['audit_role']='EVAL'
            if row['query_id'] in queries:assert queries[row['query_id']]==row
            queries[row['query_id']]=row
    assert contract.original.ids(worker.tasks())<=queries.keys()
    return queries


def admit():
    packet=worker.packet();assert contract.source_admission(packet)['status']=='PASS'
    assert c.read(RUN/'private/NATIVE_FMIN_EXIT1.json')['exit_code']==0
    ledger=c.read(RUN/'RESOURCE_LEDGER.json');manifest=c.read(RUN/'RUN_MANIFEST.json')
    assert all(v.get('ended_epoch') for v in ledger['gpu_sessions'])
    assert ledger['gpu_seconds_used']<manifest['GPU_seconds_limit'] and time.time()<manifest['deadline_epoch']
    assert not list((RUN/'private').glob('FAILURE*'))
    for file in ('SELECTED_TRAINING_CPU_CHECKS1.json','NATIVE_FMIN_REFACTOR_CPU_CHECKS1.json',
                 'SELECTED_QUEUE_CPU_CHECKS1.json','SELECTED_REPORT_CPU_CHECKS2.json'):
        assert c.read(RUN/'private'/file)['status']=='PASS'
    source=c.read(RUN/'private/SELECTED_SOURCE1.json');assert source['commit']
    for owner in (1,2):
        task=worker.tasks()[owner-1];initial=r.p.tr.TT4(task['seed'],8,8).state_dict()
        canonical=r.mm.canonicalize(initial)[0]
        for arm in ('C',)+comparison146.METHODS:
            saved=r.p.load_state(worker.point(task,arm));expected=initial if arm=='C' else canonical
            assert saved['arm']==arm and saved['order']==owner
            assert all(torch.equal(saved['states']['BASE'][k],expected[k]) for k in expected)
            result=c.read(RUN/'private/results'/arm/f'{owner}.json')
            assert result['save_load_exact'] and len(result['curve'])==160
        for prefix in ('CALIBRATION','NATIVE_M13','NATIVE_FMIN'):
            assert c.read(RUN/'private'/f'{prefix}_COMPLETE_{owner}.json')['steps']==160
        assert c.read(RUN/'private'/f'NATIVE_FMIN_FIRST_STEP_REAL_{owner}.json')['status']=='PASS'
    old=Path(os.environ['PAPER_PARENT'])
    audit=c.read(RUN/'private/HISTORICAL_146_REUSE_AUDIT1.json')
    assert audit['status']=='PASS_READONLY_HISTORICAL_REUSE' and audit['original146_tasks_seed_events_and_Base_mask_exact']
    routers=c.read(RUN/'private/HISTORICAL_R0_BINDINGS146.json')
    assert set(routers)=={str(i) for i in range(1,147)}
    for task in worker.tasks():
        saved=r.p.load_state(Path(routers[str(task['order'])]))
        assert len(saved['entries'])==1 and saved['entries'][0]['logical_edit_id']==task['edit_id']
        entry=saved['entries'][0]
        assert torch.isfinite(torch.as_tensor(entry['key'])).all() and entry['radius']>0
        for row in trajectory.source_rows(packet,task['order']):assert Path(row['image_path']).is_file()
    queries=evaluation_queries(old)
    plan=comparison146.evaluation_plan(packet,queries)
    for row in plan:assert Path(c.local_path(row['query']['image_path'])).is_file()
    assert set(manifest['physical_GPUs'])=={0,1}
    assert manifest['owned_weight_limit_bytes']==2*1024**3 and manifest['min_free_bytes']==8*1024**3
    assert shutil.disk_usage(RUN).free>=300*1024**3,'Full payload/storage reservation plus existing reserve'
    with tempfile.NamedTemporaryFile(dir=RUN) as probe:
        probe.write(b'native146');probe.flush();assert Path(probe.name).read_bytes()==b'native146'
    scope=dict(primary_methods=list(comparison146.METHODS),source_recipe=contract.SOURCE_RECIPE,
        source_weights=list(contract.SOURCE_WEIGHTS),steps=160,parameters=7168,M13_flags=[True,False,True],
        FISHER_MIN_gain_fraction=.9,FISHER_MIN_budget_fraction=.1,alignment_rule=False,
        prefixes=list(comparison146.PREFIXES),pilot_owners_reused=[1,2],
        original_task_bindings=[contract.original.task_binding(t) for t in worker.tasks()])
    new_plan=c.read(RUN/'PLAN_CONFIG.json')
    new_plan.update(stage='SELECTED_NATIVE_FIT146_FULL',maximum_counts=comparison146.CAPS,
        lifecycle=dict(C='same-owner M13 C-step and Fmin budget/save-load',
            resume='complete trajectory and final save/load',primary_weights='all actual single/insertion/prefix generation',
            Fisher='last training and source diagnostics'),GPU_time_allocation='Two models within the original remaining 48h window')
    # New full-stage allocation follows a closed pilot; its old cap and costs remain auditable.
    ceiling=ledger['gpu_seconds_used']+2*max(0.,manifest['deadline_epoch']-time.time())
    with (RUN/'private/SELECTED_ADMISSION_ONCE1').open('x'):pass
    c.write(RUN/'private/ORIGINAL_PILOT_MANIFEST1.json',manifest)
    c.write(RUN/'private/ORIGINAL_PILOT_PLAN1.json',c.read(RUN/'PLAN_CONFIG.json'))
    historical_view(old)
    manifest.update(stage='SELECTED_NATIVE_FIT146_FULL',GPU_seconds_limit=ceiling,
        Judge_limit=c.read(RUN/'private/INHERITED_COST.json')['Judge_attempts']+comparison146.CONSUMERS)
    c.write(RUN/'RUN_MANIFEST.json',manifest);c.write(RUN/'STAGE_CAP.json',dict(GPU_seconds_limit=ceiling));c.write(RUN/'PLAN_CONFIG.json',new_plan)
    for name,value in [('SELECTED_SCOPE1.json',scope),('EVAL_PLAN146.json',plan),('ROUTER_BINDINGS146.json',routers),
                       ('ORIGINAL146_TASKS.json',worker.tasks()),('ORIGINAL_BASE_MASK146.json',packet['base_mask']),
                       ('SOURCE_ADMISSION146.json',dict(status='PASS',packet_binding=c.digest(packet))),
                       ('FINAL_COMPARISON_ADMISSION.json',dict(status='PASS',primary_methods=list(comparison146.METHODS),
                            source_recipe=contract.SOURCE_RECIPE,finite_budget_frozen=True,all_declared_consumers_implemented=True))]:
        c.write(RUN/'private'/name,value)
    c.write(RUN/'private/SELECTED_EXECUTION_BUDGET1.json',dict(pilot_complete_within_original_limit=True,
        prior_GPU_seconds_limit=c.read(RUN/'private/ORIGINAL_PILOT_MANIFEST1.json')['GPU_seconds_limit'],
        GPU_seconds_used=ledger['gpu_seconds_used'],new_GPU_seconds_limit=ceiling,
        original_deadline_epoch=manifest['deadline_epoch'],remaining_full_stage_GPU_seconds=ceiling-ledger['gpu_seconds_used'],
        cumulative_costs_not_reset=True,maximum_counts=comparison146.CAPS))
    c.write(RUN/'private/SELECTED_RUNTIME_ADMISSION1.json',dict(status='PASS',execution_ready=True,launch_authorized=True,
        owners=list(range(3,147)),physical_GPUs=[0,1],packet_binding=c.digest(packet),scope_binding=c.digest(scope),
        source_commit=source['commit'],plan_binding=c.digest(new_plan),
        cpu_checks_binding=c.digest(c.read(RUN/'private/SELECTED_TRAINING_CPU_CHECKS1.json')),
        source_and_router_paths_checked=True,live_network_gate='Every owner/arm verifies actual zero Base and source gradient/RNG before updates',
        live_full146_performance_confirmed=False,epoch=time.time()))


if __name__=='__main__':admit()
