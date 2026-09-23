"""Single-GPU persistent campaign DAG, absolute deadlines and independent accounting."""
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from freshstart.runtime import ROOT,read,write,epoch,gpu_session,load_runtime,check_budget,ledger_lock


def child():
    import copy
    import torch
    from freshstart.pipeline import initialize,input_key
    from freshstart.plan import compile_plan
    from freshstart.worker import run_arm,event,state_hash
    cfg=read(ROOT/'ACTIVE_TASK.json');mode=cfg['mode'];seed=cfg.get('seed',20260923)
    tasks=read(ROOT/'private/FRESH_TASKS.json')['tasks']
    if mode=='order':
        import random
        random.Random(20261001).shuffle(tasks)
    for task in tasks:
        task['seed']=int(hashlib.sha256(json.dumps([seed,task['canonical_edit_id']]).encode()).hexdigest()[:8],16)
    p0=mode=='canary';run=ROOT/('canary' if p0 else 'formal-'+mode)
    if p0:tasks=tasks[:7]
    result=dict(mode=mode,status='STARTING',arms={})
    write(ROOT/'TASK_RESULT.json',result)
    try:
        with gpu_session('P0_RUNTIME' if p0 else mode):
            runtime=load_runtime(run,seed)
            def boundary(_module,_args):check_budget(training=True,p0=p0)
            handle=runtime.model.register_forward_pre_hook(boundary)
            if mode=='baseline':
                from freshstart.baseline import run as run_baseline
                run_baseline(runtime,tasks);result['status']='COMPLETE';return
            # The compiler rebuilds order, arrivals, masks, owners and pending slots from Base-only features.
            plan=compile_plan(runtime,tasks,seed);write(run/'PLAN.json',plan)
            arms=['F0','FH','FR','FA'] if p0 else cfg['arms']
            replay_n=len(tasks)
            for arm in arms:
                if p0 and arm=='FA' and 'FR' in result['arms'] and result['arms']['FR'].get('status')=='FAILED':
                    result['arms'][arm]=dict(status='DISABLED_PAIRED_CANARY_FAILURE');continue
                try:
                    if not p0 and arm=='FR':
                        cost=read(run/'COST_SAMPLE_F0.json')['mean_seconds']
                        available=cfg['phase_deadline']-time.time()
                        replay_n=next((n for n in [len(tasks),5] if n<=len(tasks) and 2*n*cost*1.3<=available),0)
                        write(run/'REPLAY_BUDGET_DECISION.json',dict(common_prefix=replay_n,remaining_seconds=available,mean_measured_prefix_seconds=cost,reserve_factor=1.3,selection_used=False))
                    if arm in ['FR','FA'] and not replay_n:
                        result['arms'][arm]=dict(status='NOT_STARTED_BUDGET');continue
                    arm_tasks=tasks[:replay_n] if arm in ['FR','FA'] else tasks
                    outcome=run_arm(runtime,run,arm_tasks,plan,arm,seed,p0=p0)
                    result['arms'][arm]=dict(status='PASS',**outcome)
                except Exception as error:
                    result['arms'][arm]=dict(status='FAILED',error_type=type(error).__name__)
                    prior=read(run/f'{arm}_STATUS.json') if (run/f'{arm}_STATUS.json').exists() else {}
                    write(run/f'{arm}_STATUS.json',dict(prior,status='FAILED_KEEP_RECOVERY',error_type=type(error).__name__))
                    write(run/f'{arm}_FAILURE_PRIVATE.json',dict(error=str(error),traceback=traceback.format_exc()))
                    if arm in ['F0','FH']:raise
                    if not p0:raise
                write(ROOT/'TASK_RESULT.json',result)
            if p0:
                # Real-model mask and missing-teacher contract probes are isolated from scientific data eligibility.
                task=tasks[0];probe=ROOT/'canary-boundary-tests';probe_expert=initialize(runtime,task,probe,seed,p0=True)
                before=state_hash(probe_expert)
                missing=dict(task['U_fit'][0],image_sha256='CANARY_MISSING_INPUT_FINGERPRINT')
                try:event(runtime,probe,task,probe_expert,'FH',1,'missing-teacher',task['H_fit'],[missing],20,plan['plan_hash'],p0=True)
                except KeyError:missing_rejected=True
                else:raise RuntimeError('Missing exact teacher was not rejected')
                if state_hash(probe_expert)!=before:raise RuntimeError('Missing teacher wrote parameters')
                masked=event(runtime,probe,task,probe_expert,'FH',1,'explicit-mask-contract-probe',task['H_fit'],[],20,plan['plan_hash'],p0=True,masked_u=1)
                if masked['teacher_keys'] or any('U' in point['terms'] for point in masked['curve']):raise RuntimeError('Masked U reached teacher/gradient')
                if not all('native' in point['terms'] and 'H' in point['terms'] for point in masked['curve']):raise RuntimeError('Mask suppressed native/H')
                from collections import Counter
                double=sum(v==2 for v in Counter(s['prefix'] for s in plan['slots']).values())
                if not double:
                    for arm in ['FR','FA']:result['arms'][arm]['status']='DISABLED_NO_DOUBLE_SLOT_COVERAGE'
                accepted=[a for a,s in result['arms'].items() if s['status']=='PASS']
                files={str(p.relative_to(ROOT/'source')):hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (ROOT/'source/freshstart').glob('*.py')}
                receipt=dict(status='PASS',accepted_arms=accepted,canary_N=len(tasks),actual_double_slot_prefixes=double,
                    missing_exact_teacher_rejected=missing_rejected,mask_probe='REAL_MODEL_EXPLICIT_CONTRACT_MASK_NOT_SCIENTIFIC_ELIGIBILITY',
                    mask_probe_steps=masked['actual_steps'],fault_recovery=result['arms']['F0']['fault_injections'],
                    implementation_files=files,code_commit=subprocess.check_output(['git','-C',str(ROOT/'source'),'rev-parse','HEAD'],text=True).strip(),
                    historical_checkpoints_loaded=False,formal_canary_state_sharing=False,base_unchanged=runtime.base_guard.verify()['unchanged'])
                if not all(a in accepted for a in ['F0','FH']):raise RuntimeError('Reference arms not qualified')
                write(ROOT/'P0_ACCEPTANCE.json',receipt)
            handle.remove();result['status']='COMPLETE'
    except Exception as error:
        result.update(status='FAILED',error_type=type(error).__name__)
        write(ROOT/('TASK_FAILURE_'+mode+'.json'),dict(error=str(error),traceback=traceback.format_exc()))
        raise
    finally:write(ROOT/'TASK_RESULT.json',result)


def supervisor():
    from freshstart.report import generate
    manifest=read(ROOT/'RUN_MANIFEST.json');start=epoch(manifest['first_started_at']);deadline=epoch(manifest['deadline_at'])
    lock=(ROOT/'SUPERVISOR.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    write(ROOT/'SUPERVISOR_PID.json',dict(pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat(),deadline=manifest['deadline_at']))
    dag=[dict(mode='canary',latest_finish=manifest['p0_deadline_at']),dict(mode='primary',arms=['F0','FH','FR','FA'],seed=20260923),
         dict(mode='secondary',arms=['FH','F0'],seed=20260924,reason='full candidate protection panels unavailable; reference pair'),
         dict(mode='baseline'),dict(mode='METHOD_LOCK',wall_second=43200),
         dict(mode='confirmation',status='NO_INDEPENDENT_CONFIRMATION_AVAILABLE'),dict(mode='order',seed=20260923,shuffle_seed=20261001,arms=['FH','F0']),
         dict(mode='closeout')]
    write(ROOT/'TASK_DAG.json',dict(tasks=dag,N=16,one_physical_gpu=True,absolute_deadline=manifest['deadline_at'],STOP=str(ROOT/'STOP')))
    (ROOT/'logs').mkdir(exist_ok=True)
    entry=Path('/root/rivermind-data/job-622/child.py')
    entry.write_text("import sys\nsys.path.insert(0,'/root/rivermind-data/job-622/state/source')\nfrom freshstart.runner import child\nchild()\n")
    last_status=0;midpoint=False
    def status():
        nonlocal last_status,midpoint
        if time.time()-last_status>=1800:
            generate();last_status=time.time()
        if time.time()>=start+43200 and not midpoint:
            generate('MIDPOINT')
            write(ROOT/'public/METHOD_LOCK.json',dict(status='NO_VALIDATED_CANDIDATE',reference='FH',
                reason='missing independent rephrase/positive-image/strict-unrelated qualification panels',
                code_commit=read(ROOT/'P0_ACCEPTANCE.json')['code_commit'],locked_at=datetime.now(timezone.utc).isoformat(),confirmation='NO_INDEPENDENT_CONFIRMATION_AVAILABLE'))
            midpoint=True
    def execute(mode,limit,**extra):
        check_budget(training=True)
        write(ROOT/'ACTIVE_TASK.json',dict(mode=mode,phase_deadline=limit,**extra))
        with (ROOT/'logs'/f'{mode}.log').open('ab') as log:
            process=subprocess.Popen([sys.executable,str(entry)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=entry.parent)
        write(ROOT/'CURRENT_JOB.json',dict(mode=mode,pid=process.pid,deadline_epoch=limit,log=f'logs/{mode}.log'))
        while process.poll() is None:
            status()
            if time.time()>=min(limit,deadline) or (ROOT/'STOP').exists():
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=20)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                # Reconcile the owned killed process only; never reset prior accounting.
                with ledger_lock() as ledger:
                    for session in ledger['gpu_sessions']:
                        if session['pid']==process.pid and session.get('ended_epoch') is None:
                            end=time.time();session.update(ended_epoch=end,resident_seconds=end-session['started_epoch'],closure='SUPERVISOR_DEADLINE')
                            ledger['gpu_seconds_used']+=end-session['started_epoch']
                break
            time.sleep(5)
        write(ROOT/('EXIT_'+mode+'.json'),dict(exit_code=process.returncode,at=datetime.now(timezone.utc).isoformat()))
        generate('STAGE_'+mode.upper())
        return process.returncode==0
    try:
        if not execute('canary',epoch(manifest['p0_deadline_at'])):raise RuntimeError('P0 failed; no formal launch')
        accepted=read(ROOT/'P0_ACCEPTANCE.json')['accepted_arms']
        if not execute('primary',min(start+8*3600,time.time()+21600),arms=accepted,seed=20260923):
            write(ROOT/'PRIMARY_PARTIAL.json',dict(status='PARTIAL_STOPPED_PRESERVE_EVIDENCE'))
            result=read(ROOT/'TASK_RESULT.json')
            if any(result.get('arms',{}).get(a,{}).get('status')=='FAILED' for a in ['F0','FH']):
                raise RuntimeError('Reference-arm core failure; preserve evidence and stop dependent tasks')
        phase2_cap=min(start+12*3600,time.time()+14400)
        if time.time()<phase2_cap-1800:
            execute('secondary',phase2_cap-1800,arms=['FH','F0'],seed=20260924)
        if time.time()<phase2_cap:execute('baseline',min(phase2_cap,time.time()+1800))
        # Release CUDA between tasks; the wall deadline continues during method-lock waiting.
        while time.time()<start+43200 and not (ROOT/'STOP').exists():status();time.sleep(30)
        status()
        write(ROOT/'CONFIRMATION_STATUS.json',dict(status='NO_INDEPENDENT_CONFIRMATION_AVAILABLE'))
        if time.time()<epoch(manifest['no_new_training_after']):
            execute('order',min(epoch(manifest['no_new_training_after']),time.time()+21600),arms=['FH','F0'],seed=20260923)
        # No duplicate/fabricated edits are used to inflate this pool to 90 or 146.
        write(ROOT/'STRESS_STATUS.json',dict(status='NOT_EXECUTED_NO_LEGAL_NEW_EDIT_POOL',available=16))
        drain=min(deadline,time.time()+7200)
        while time.time()<drain:
            current=generate('FINAL')
            if current['scoring_complete']:break
            time.sleep(30)
        final=generate('FINAL')
        write(ROOT/'BACKUP_RECEIPT.json',dict(status='OFFHOST_BACKUP_NOT_AVAILABLE',local_event_versions_preserved=True,server_loss_protection=False))
        write(ROOT/'FINAL.json',dict(status='CLOSED_WITH_DOCUMENTED_LIMITATIONS',public_report='public/FINAL.md',**{k:final[k] for k in ['scoring_complete','fixed_prefix_consumers','fixed_prefix_unscored']}))
    except Exception as error:
        write(ROOT/'SUPERVISOR_FAILURE.json',dict(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc()))
        generate('BLOCKED')
        raise


if __name__=='__main__':supervisor()
