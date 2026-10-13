"""Finite experiment controller, never an hourly monitor or an automatic next round."""
import fcntl
import os
from pathlib import Path
import subprocess
import tempfile
import time
import traceback
import controller_parent as parent
from audit import read,write

RUN=Path(os.environ['RUN_ROOT'])


def available(gpu):
    ok,reason=original_available(gpu)
    if not ok:return ok,reason
    uuid=read(RUN/'PLAN_CONFIG.json')['hardware']['UUIDs'][str(gpu)]
    rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True).splitlines()
    if any(row.split(',')[0].strip()==uuid for row in rows):return False,'must be idle: existing compute process'
    usage=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True).strip().split(', ')
    return (True,'PASS_IDLE') if int(usage[0])==0 and int(usage[1])<100 else (False,'must be idle: utilization or memory')


original_available=parent.available;parent.available=available


def start(action,gpu,part=None):
    fd,name=tempfile.mkstemp(prefix='e.',suffix='.py');os.close(fd)
    Path(name).write_text("import os,sys,runpy\nsys.path.insert(0,os.environ['RUN_ROOT']+'/private/tools')\nrunpy.run_path(os.environ['RUN_ROOT']+'/private/tools/stage_worker.py',run_name='__main__')\n")
    uuid=read(RUN/'PLAN_CONFIG.json')['hardware']['UUIDs'][str(gpu)]
    env=dict(os.environ,ACTION=action,GPU=str(gpu),CUDA_VISIBLE_DEVICES=str(gpu),CUBLAS_WORKSPACE_CONFIG=':4096:8',M3BENCH_FORMAL_AUTHORIZED_CUDA_VISIBLE_DEVICES=str(gpu),M3BENCH_FORMAL_ALLOWED_CUDA_VISIBLE_DEVICES=str(gpu),M3BENCH_FORMAL_EXPECTED_GPU_UUID=uuid)
    if part is not None:env['PARTITION']=str(part)
    p=subprocess.Popen(['/data/bmw/envs/v0/bin/python','-u',name],env=env,stdout=(RUN/'logs'/f'{action}_{gpu}.log').open('ab'),stderr=subprocess.STDOUT,start_new_session=True)
    state=dict(pid=p.pid,start_ticks=Path(f'/proc/{p.pid}/stat').read_text().split()[21],epoch=time.time(),action=action,gpu=gpu,partition=part,entry=name)
    write(RUN/'private'/f'START_{action}_{gpu}.json',state);print('START',action,gpu,p.pid,flush=True);return p,state


def phase(name):
    jobs=[]
    for gpu,part in ((6,0),(5,1),(4,2),(3,3)):
        if (RUN/'private'/f'{name}_PART_{part}_COMPLETE.json').exists():continue
        assert not (RUN/'private'/f'FAILURE_{name}_{gpu}.json').exists(),'Recorded failure: no automatic retry'
        parent.wait_capacity(gpu);jobs.append(start(name,gpu,part))
    parent.wait(jobs)
    assert all((RUN/'private'/f'{name}_PART_{p}_COMPLETE.json').exists() for p in range(4))
    write(RUN/'private/GENERATION_COMPLETE.json',dict(status='PHASE_GENERATED_NOT_SCORED',phase=name,epoch=time.time()))
    write(RUN/'private/SCORING_PHASE.json',dict(phase=name,training_complete=True,epoch=time.time()))
    parent.cpu_tool('qwen_scorer.py')
    write(RUN/'private/judge_common'/f'SCORER_DONE_{name}.json',read(RUN/'private/judge_common/SCORER_DONE.json'))
    parent.cpu_tool('stage_report.py')


def main():
    lock=(RUN/'private/CONTROLLER.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert read(RUN/'private/CPU_ADMISSION.json')['status']=='PASS'
    assert read(RUN/'private/SCORING_IMPLEMENTATION_ADMISSION.json')['status']=='PASS'
    if not (RUN/'private/GPU_MECHANICAL.json').exists():
        parent.wait_capacity(6);parent.wait([start('mechanical',6)])
    assert read(RUN/'private/GPU_MECHANICAL.json')['status']=='PASS'
    for name in ('P0','P1'):
        if not (RUN/'private'/f'REPORT_{name}_COMPLETE.json').exists():phase(name)
    gate=read(RUN/'private/P1_GATE.json');next_phase=gate['next_phase']
    # Commit to exactly one branch. Never use paired-control scores to retune.
    branch=RUN/'private/P2_BRANCH.json'
    decision=dict(phase=next_phase,gate=gate,algorithm_frozen=True)
    if branch.exists():assert read(branch)==decision
    else:write(branch,decision)
    if not (RUN/'private'/f'REPORT_{next_phase}_COMPLETE.json').exists():phase(next_phase)
    data=read(RUN/'private/P3_DATA_GATE.json')
    assert data['status']=='BLOCKED_DATA','Qualified new panel requires explicit frozen dataset evaluator admission'
    write(RUN/'public/P3_DATA_GATE.json',data)
    write(RUN/'private/CONTROLLER_COMPLETE.json',dict(status='RESULTS_COMPLETE_SCIENTIFIC_REVIEW_PENDING',P2=next_phase,P3='BLOCKED_DATA',epoch=time.time(),no_next_round=True,no_hourly_monitor=True))
    write(RUN/'public/PROGRESS.json',dict(status='RESULTS_COMPLETE_SCIENTIFIC_REVIEW_PUBLICATION_PENDING',P2=next_phase,P3='BLOCKED_DATA',GPU_hours=parent.used()/3600))


def partial(error):
    blocks=list((RUN/'private/blocks').glob('*.json'));finals=list((RUN/'private/edits').glob('e*/*/final.pt'));latest=list((RUN/'private/edits').glob('e*/*/latest.pt'))
    item=dict(status='INCOMPLETE',error=repr(error),epoch=time.time(),GPU_hours_including_live=parent.used()/3600,completed_paired_blocks=len(blocks),finals=len(finals),active_latest=len(latest),P1_expected=48,P2_expected='24 if PASS else32',missing_not_removed=True,automatic_retry=False)
    write(RUN/'public/PARTIAL_COMPLETION.json',item);write(RUN/'private/CONTROLLER_FAILURE.json',dict(item,traceback=traceback.format_exc()))


if __name__=='__main__':
    try:main()
    except Exception as error:partial(error);raise
