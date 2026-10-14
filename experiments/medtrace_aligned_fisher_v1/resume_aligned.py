"""One audited user-authorized continuation; completed editors are immutable."""
import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import traceback
import numpy as np
import torch
import full_aligned as a

p,r,c,RUN=a.p,a.r,a.c,a.RUN


def validate(saved,task,probe):
    assert saved['seed']==task['seed'] and saved['source_order']==probe['source_order']
    assert saved['budget']==probe['budget']
    assert probe['step']<=saved['step']<=160 and len(saved['curve'])==saved['step']
    assert [v['step'] for v in saved['curve']]==list(range(1,saved['step']+1))
    assert saved['scalar']['step']==saved['step'] and len(saved['scalar']['second_moments'])==2
    assert all(np.isfinite(x) and x>=0 for x in saved['scalar']['second_moments'])
    assert set(saved['state'])==set(r.mm.KEYS) and all(torch.isfinite(v).all() for v in saved['state'].values())
    assert all(saved['state'][k].shape==probe['before'][k].shape for k in r.mm.KEYS)
    assert saved['rng']['cuda_rng'] is not None
    assert all(v>=0 for v in saved['counts'].values())
    assert probe['scope_lock']==c.digest(c.read(RUN/'private/SCOPE_LOCK.json'))
    assert probe['target_lock']==c.digest(c.read(RUN/'private/TARGETS.json'))
    # Scientific checkpoint identity is bound by source order, original locks and seed.
    now=c.rng();numpy_now=np.random.get_state()
    try:
        c.restore_rng(dict(saved['rng'],cuda_rng=None));p.f.numpy_restore(saved['numpy_rng'])
        assert torch.equal(torch.get_rng_state(),saved['rng']['torch_rng'])
    finally:c.restore_rng(now);np.random.set_state(numpy_now)


def admission():
    stop=c.read(RUN/'private/USER_STOP_RESULT1.json');assert stop['status']=='STOPPED_VERIFIED'
    assert not (RUN/'private/COMPLETE.json').exists() and not (RUN/'private/FULL_EXIT1.json').exists()
    assert not list((RUN/'private').glob('FAILURE_full_*'))
    for row in c.read(RUN/'private/USER_STOP_REQUEST1.json')['verified']:
        path=Path('/proc')/str(row['pid'])/'stat'
        if path.exists():assert path.read_text().split()[21]!=row['ticks'] or path.read_text().split()[2]=='Z'
    tasks=r.s.tasks();complete=[];pending=[]
    for task in tasks:
        owner=task['order'];probe=r.p.load_state(RUN/'private/probes'/f'{owner}.pt')
        assert probe['seed']==task['seed'] and probe['source_order']==[v['query_id'] for v in r.s.source(task)]
        assert probe['scope_lock']==c.digest(c.read(RUN/'private/SCOPE_LOCK.json'))
        assert probe['target_lock']==c.digest(c.read(RUN/'private/TARGETS.json'))
        assert probe['budget']==.1*c.read(Path(os.environ['MODULE_PARENT'])/'private/calibration'/f'{owner}.json')['C_operator_penalty']
        point=r.s.point(task,a.NEW);result=RUN/'private/results'/a.NEW/f'{owner}.json'
        checkpoint=RUN/'private/full_resume'/f'{a.NEW}_{owner}.pt'
        if checkpoint.exists():validate(r.p.load_state(checkpoint),task,probe)
        if point.exists() or result.exists():
            assert point.exists() and result.exists() and checkpoint.exists()
            final=r.p.load_state(point);saved=r.p.load_state(checkpoint);out=c.read(result)
            assert out['save_load_exact'] and len(out['curve'])==saved['step']==160
            assert out['scalar_optimizer_state']==saved['scalar']==final['scalar_optimizer_state']
            assert all(torch.equal(saved['state'][k],final['states']['FINAL'][k]) for k in r.mm.KEYS)
            complete.append(owner)
        else:pending.append(dict(owner=owner,step=r.p.load_state(checkpoint)['step'] if checkpoint.exists() else probe['step'],checkpoint=checkpoint.exists(),partition=(owner-1)%3))
    assert complete==[1,2,3,5,6] and [v['owner'] for v in pending]==[4,7,8]
    old_counts={str(i):c.read(RUN/'private'/f'COUNTS_paired_train_{i}.json') for i in (0,1,2)}
    saved=r.p.load_state(RUN/'private/full_resume/ALIGNED_MIN_4.pt')
    assert saved['step']==80 and saved['counts']==old_counts['0']
    assert old_counts['1']['updates']==sum(160-r.p.load_state(RUN/'private/probes'/f'{i}.pt')['step'] for i in (2,5))
    # Each interrupted worker can lose at most one 20-step interval plus bounded setup.
    upper=dict(updates=40,backwards=7800,operator_backwards=1920,local_solves=2560,
               non_generation_forwards=25500,forwards=25500,generation_attempts=0)
    counts=[c.read(f) for f in (RUN/'private').glob('COUNTS_*.json')]
    totals={k:sum(v[k] for v in counts) for k in counts[0]}
    remaining=sum(160-v['step'] for v in pending)
    assert totals['updates']+remaining==1280 and totals['generation_attempts']==0
    limits=dict(backwards=200000,operator_backwards=46000,local_solves=65000,non_generation_forwards=1500000)
    per_step=dict(backwards=195,operator_backwards=48,local_solves=64,non_generation_forwards=600)
    forecast={k:totals[k]+upper[k]+remaining*per_step[k]+2000 for k in limits}
    assert all(forecast[k]<=limits[k] for k in limits)
    report=dict(status='PASS',epoch=time.time(),complete=complete,pending=pending,old_counts=old_counts,
        remaining_scientific_proposals=remaining,discarded_compute_upper_bound=upper,finite_forecast=forecast,
        original_locks_source_seed_budget=True,completed_owners_retrained=False,
        public_accounting=dict(completed_editors_reused=5,remaining_editors=3,checkpoint_step=80,
            exact_unsaved_compute_unknown=True,discarded_compute_upper_bound=upper,extra_training_leases=2))
    c.write(RUN/'private/USER_RESUME_ADMISSION1.json',report)
    return report


def restore(task,arm,state,scalar,curve,rng,numpy_rng):
    path=RUN/'private/full_resume'/f'{arm}_{task["order"]}.pt'
    if not path.exists():return state,scalar,curve,rng,numpy_rng
    saved=r.p.load_state(path);validate(saved,task,r.p.load_state(RUN/'private/probes'/f'{task["order"]}.pt'))
    assert saved['step']<160
    return ({k:v.to(os.environ['RESUME_DEVICE']) for k,v in saved['state'].items()},
            copy.deepcopy(saved['scalar']),copy.deepcopy(saved['curve']),saved['rng'],saved['numpy_rng'])


def train():
    report=c.read(RUN/'private/USER_RESUME_ADMISSION1.json');part=os.environ['PARTITION']
    r.COUNTS.update(report['old_counts'][part]);original_tasks=r.s.tasks
    pending={v['owner'] for v in report['pending']}
    # Keep original partition slices, filtering only after their assignment.
    all_tasks=original_tasks()
    class Assigned(list):
        def __getitem__(self,key):
            value=super().__getitem__(key)
            return [t for t in value if t['order'] in pending] if isinstance(key,slice) else value
    r.s.tasks=lambda:Assigned(all_tasks)
    os.environ['RESUME_DEVICE']='cuda:0'
    original=r.source_gradients
    memory=p.g.memory_record
    p.g.memory_record=lambda *args,**kwargs:memory(*args,**dict(kwargs,prefix='paired_full_user_resume1'))
    def gradients(runtime,hook,expert,sources,step,individual):
        if not getattr(expert,'_user_resume_verified',False):
            expert._user_resume_verified=True;state=r.old.snapshot(expert);rng=c.rng();n=np.random.get_state()
            _,first=original(runtime,hook,expert,sources,step,individual)
            grad={k:v.grad.detach().clone() for k,v in expert.named_parameters()};post=c.rng()
            expert.load_state_dict(state);c.restore_rng(rng);np.random.set_state(n)
            _,second=original(runtime,hook,expert,sources,step,individual)
            assert first==second and all(torch.equal(grad[k],v.grad) for k,v in expert.named_parameters())
            assert all(torch.equal(post[k],c.rng()[k]) for k in ('torch_rng','cuda_rng'))
            assert post['python_rng']==c.rng()['python_rng']
            c.write(RUN/'private/USER_RESUME_REAL1'/f'{part}_{step}.json',dict(status='PASS',step=step,
                restored_source_gradients_exact=True,GPU_RNG_exact=True,source_RMS_advanced_by_training_only=True))
            return None,second
        return original(runtime,hook,expert,sources,step,individual)
    r.source_gradients=gradients
    p.train(resume=restore)


def launch(part,gpu):
    import pipeline
    env=dict(os.environ,ACTION='paired_train',PARTITION=str(part),GPU=str(gpu),CUDA_VISIBLE_DEVICES=str(gpu))
    fd,entry=tempfile.mkstemp(prefix='e.',suffix='.py',dir=os.environ['TMPDIR']);os.close(fd)
    Path(entry).write_text("import os,sys,runpy\nsys.path.insert(0,os.environ['RUN_ROOT']+'/private/tools')\nrunpy.run_path(os.environ['RUN_ROOT']+'/private/tools/resume_aligned.py',run_name='__main__')\n")
    log=RUN/'logs'/f'user_resume1_train_{part}.log'
    job=subprocess.Popen([env['TRAIN_PYTHON'],'-u',entry],env=env,stdout=log.open('ab'),stderr=subprocess.STDOUT,start_new_session=True)
    c.write(RUN/'private'/f'START_CHAIN_USER_RESUME1_{part}.json',dict(pid=job.pid,start_ticks=Path(f'/proc/{job.pid}/stat').read_text().split()[21],entry=entry,gpu=gpu,epoch=time.time()))
    return job


def controller():
    assert c.read(RUN/'private/USER_RESUME_CPU_EXIT1.json')['exit_code']==0
    assert c.read(RUN/'public/USER_RESUME_CPU_CHECKS.json')['status']=='PASS'
    with (RUN/'private/USER_RESUME_ONCE1').open('x'):pass
    report=admission()
    with c.resources() as ledger:
        stopped=c.read(RUN/'private/USER_STOP_RESULT1.json');ended=stopped['epoch']
        for row in ledger['gpu_sessions']:
            if not row.get('ended_epoch'):
                assert row['pid'] in stopped['signaled_pids']
                seconds=ended-row['started_epoch'];row.update(ended_epoch=ended,resident_seconds=seconds,end_reason='USER_STOP_VERIFIED_UPPER_ENDPOINT')
                ledger['gpu_seconds_used']+=seconds
    expected=c.read(RUN/'PLAN_CONFIG.json')['hardware']['UUIDs']
    for gpu in (0,1):
        uuid,free=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(', ')
        assert uuid==expected[str(gpu)] and int(free)>=45*1024
    assert time.time()<c.read(RUN/'RUN_MANIFEST.json')['deadline_epoch']
    c.write(RUN/'public/PROGRESS.json',dict(status='ALIGNED_USER_RESUME1_TRAINING',complete=False))
    import pipeline
    pipeline.wait([launch(0,0),launch(1,1)])
    a.finish()


if __name__=='__main__':
    try:
        if os.environ['ACTION']=='user_resume_admit':admission()
        elif os.environ['ACTION']=='user_resume_controller':controller()
        elif os.environ['ACTION']=='paired_train':train()
        else:raise ValueError(os.environ['ACTION'])
        if os.environ['ACTION']=='user_resume_controller':
            c.write(RUN/'private/USER_RESUME_EXIT1.json',dict(exit_code=0,epoch=time.time()))
            c.write(RUN/'private/FULL_EXIT1.json',dict(exit_code=0,epoch=time.time(),user_resume=True))
    except BaseException as error:
        r.save_counts();c.write(RUN/'private'/f'FAILURE_user_resume1_{os.environ["ACTION"]}_{os.environ.get("PARTITION")}.json',dict(error=repr(error),traceback=traceback.format_exc(),counts=r.COUNTS,epoch=time.time(),retry=False))
        if os.environ['ACTION']=='user_resume_controller':c.write(RUN/'private/USER_RESUME_EXIT1.json',dict(exit_code=1,epoch=time.time()))
        raise
