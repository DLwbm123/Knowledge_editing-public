"""Single new dependency chain; failed workers are never automatically restarted."""
import os
import subprocess
import time
import contract
from runtime146 import c,r,RUN,tasks,point,require_ready


def controller():
    import pipeline
    require_ready()
    with (RUN/'private/M13_146_CONTROLLER_ONCE').open('x'):pass
    def stage(action,parts=(0,1,2)):
        jobs={};pending=list(parts)
        while pending or jobs:
            c.budget()
            for process in list(jobs):
                if process.poll() is not None:
                    assert process.returncode==0,'Actual worker failed; retain state, no retry'
                    del jobs[process]
            for gpu in c.read(RUN/'RUN_MANIFEST.json')['physical_GPUs']:
                if not pending or len(jobs)>=3:break
                uuid=c.read(RUN/'PLAN_CONFIG.json')['hardware']['UUIDs'][str(gpu)]
                leases=c.read(RUN/'RESOURCE_LEDGER.json')['gpu_sessions']
                if gpu in jobs.values() or any(v['gpu_uuid']==uuid and not v.get('ended_epoch') for v in leases):continue
                observed,free=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(', ')
                assert observed==uuid
                if int(free)>=45*1024:jobs[pipeline.launch('runtime146.py',action,gpu,pending.pop(0))]=gpu
            if pending or jobs:time.sleep(5)
    stage('m13_146_reference')
    for task in tasks():
        data=c.read(RUN/'private/results/C'/f"{task['order']}.json")
        assert data['save_load_exact'] and len(data['curve'])==160
    c.write(RUN/'private/REFERENCE_EXACT.json',dict(status='PASS',new_matched_C=146,not_inherited_W0=True))
    stage('m13_146_fisher',(0,))
    stage('m13_146_train')
    for task in tasks():
        for arm in contract.METHODS:
            data=c.read(RUN/'private/results'/arm/f"{task['order']}.json")
            assert data['save_load_exact'] and len(data['curve'])==160
    c.write(RUN/'private/M13_146_CANDIDATES_FROZEN.json',dict(candidates=584,selection=False))
    stage('m13_146_single');stage('m13_146_bank')
    plan=c.read(RUN/'private/EVAL_PLAN146.json');outputs=[c.read(p) for p in (RUN/'private/outputs').glob('*/*/*.json')]
    assert len(outputs)==len(plan)==24867 and {d['consumer_id'] for d in outputs}=={contract.consumer_id(d) for d in plan}
    counts=[c.read(p) for p in (RUN/'private').glob('COUNTS_m13_146_*.json')]
    for k,cap in contract.CAPS.items():
        if k not in ('evaluation_consumers','Judge_attempts'):assert sum(v.get(k,0) for v in counts)<=cap
    c.write(RUN/'private/M13_146_GENERATION_COMPLETE.json',dict(outputs=24867,counts=counts,epoch=time.time()))
    pipeline.wait([pipeline.launch('queue146.py','m13_146_ingest')])
    while (not (RUN/'private/judge_m13_146_astra_medium/ALL_WORKERS_COMPLETE.json').exists()
           or not all((RUN/'private/judge_m13_146_astra_medium/workers'/str(worker)/'SCORER_DONE.json').exists() for worker in range(4))):
        c.budget();assert not list((RUN/'private/judge_m13_146_astra_medium/workers').glob('*/SCORER_FAILURE.json'));time.sleep(20)
    for worker in range(4):assert (RUN/'private/judge_m13_146_astra_medium/workers'/str(worker)/'SCORER_DONE.json').exists()
    pipeline.wait([pipeline.launch('report146.py','m13_146_report')])
    paths=[point(task,arm) for task in tasks() for arm in contract.METHODS]+[RUN/'private/OPERATOR_FISHER.pt']
    assert not list((RUN/'private/resume146').glob('*.pt'))
    assert all(p.is_file() and not p.is_symlink() and RUN.resolve() in p.resolve().parents for p in paths)
    c.write(RUN/'private/M13_146_CONSUMED.json',dict(paths=list(map(str,paths)),last_consumers='source diagnostics, single, all real banks, report'))
    for path in paths:path.unlink()
    c.write(RUN/'public/M13_146_EXECUTION_AUDIT.json',dict(status='PASS',counts=counts,deleted_files=len(paths),new_neural_retries=0))
    c.write(RUN/'private/M13_146_COMPLETE.json',dict(status='COMPLETE',epoch=time.time(),publication_pending=True))
