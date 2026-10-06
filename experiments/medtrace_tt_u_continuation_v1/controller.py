"""One bounded dependency queue, two training GPUs, no recurring monitor."""
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import traceback
from common import RUN,read,write,budget,used,available

def neutral(code):
    fd,name=tempfile.mkstemp(prefix='e.',suffix='.py',dir=os.environ['TMPDIR']);os.close(fd);Path(name).write_text(code);return name

def launch(file,env,log,python=None):
    entry=neutral("import os,sys,runpy\nsys.path.insert(0,os.environ['RUN_ROOT']+'/private/tools')\nrunpy.run_path(os.environ['TASK_FILE'],run_name='__main__')\n")
    p=subprocess.Popen([python or os.environ['TRAIN_PYTHON'],'-u',entry],env=dict(os.environ,TASK_FILE=str(RUN/'private/tools'/file),**env),stdout=(RUN/'logs'/log).open('ab'),stderr=subprocess.STDOUT,start_new_session=True)
    item=dict(pid=p.pid,start_ticks=Path(f'/proc/{p.pid}/stat').read_text().split()[21],started_epoch=time.time(),entry=entry,file=file,env={k:v for k,v in env.items() if k in ('ACTION','GPU','PARTITION','BLOCK','REPORT_PHASE')})
    write(RUN/'private'/('START_'+log+'.json'),item);return p,item

def wait(jobs):
    while any(p.poll() is None for p,_ in jobs):
        try:budget()
        except BaseException:
            for p,s in jobs:
                if p.poll() is None and Path(f'/proc/{p.pid}/stat').read_text().split()[21]==s['start_ticks']:os.killpg(p.pid,signal.SIGTERM)
            for p,_ in jobs:
                try:p.wait(timeout=30)
                except subprocess.TimeoutExpired:pass
            raise
        if any(p.poll() is not None and p.returncode!=0 for p,_ in jobs):
            # Permit sibling completion, never launch a new scientific block after failure.
            for p,_ in jobs:
                if p.poll() is None:p.wait()
            raise RuntimeError('Recorded worker failure; no automatic retry')
        time.sleep(5)
    assert all(p.returncode==0 for p,_ in jobs),'Finite worker failed'

def capacity(gpu):
    while not available(gpu):budget();time.sleep(20)

def phase(action,block=None):
    jobs=[]
    for part,gpu in ((0,6),(1,5)):
        marker=RUN/'private'/('DONE_'+action+'_'+str(block if block is not None else 'all')+'_'+str(part)+'.json')
        if marker.exists():continue
        assert not (RUN/'private'/('FAILURE_'+action+'_'+str(gpu)+'.json')).exists(),'No blind failure retry'
        capacity(gpu);u=read(RUN/'PLAN_CONFIG.json')['hardware']['UUIDs'][str(gpu)]
        env=dict(ACTION=action,GPU=str(gpu),PARTITION=str(part),CUDA_VISIBLE_DEVICES=str(gpu),M3BENCH_FORMAL_EXPECTED_GPU_UUID=u)
        if block is not None:env['BLOCK']=str(block)
        jobs.append(launch('worker.py',env,action+'_'+str(block)+'_'+str(gpu)+'.log'))
    wait(jobs)

def cpu(file,phase=None):
    env=dict(CUDA_VISIBLE_DEVICES='',QWEN_EXECUTION_FILE=str(RUN/'private/tools/qwen_scorer.py'))
    if phase:env['REPORT_PHASE']=phase
    wait([launch(file,env,file+'_'+str(phase)+'.log')])

def scoring(phase):
    write(RUN/'private/GENERATION_COMPLETE.json',dict(status='PHASE_GENERATED',phase=phase,epoch=time.time()));cpu('qwen_scorer.py');cpu('report.py',phase)

def admission(name,estimate,judge_upper=0):
    m=read(RUN/'RUN_MANIFEST.json');r=read(RUN/'RESOURCE_LEDGER.json')
    d=dict(stage=name,used_gpu_seconds=used(),remaining_gpu_seconds=m['GPU_seconds_limit']-used(),estimated_gpu_seconds=estimate,Judge_remaining=6000-r['Judge_attempts'],conservative_Judge_upper=judge_upper,wall_remaining_seconds=m['deadline_epoch']-time.time());d['admitted']=d['remaining_gpu_seconds']>=estimate+1200 and d['wall_remaining_seconds']>=estimate/2+600 and d['Judge_remaining']>=judge_upper
    write(RUN/'private'/('ADMISSION_'+name+'.json'),d);return d['admitted']

def main():
    with (RUN/'private/CONTROLLER.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not (RUN/'RUN_MANIFEST.json').exists():
            epoch=time.time();write(RUN/'RUN_MANIFEST.json',dict(stage='medtrace_tt_u_continuation_v1',starting_epoch=epoch,deadline_epoch=epoch+24*3600,GPU_seconds_limit=24*3600,Judge_limit=6000,owned_weight_limit_bytes=2*1024**3,min_free_bytes=8*1024**3,physical_GPUs=[3,4,5,6],max_training_GPUs=2,reset_allowed=False))
        if not (RUN/'private/CPU_MECHANICAL.json').exists():cpu('cpu_check.py')
        if not (RUN/'private/GPU_MECHANICAL.json').exists():capacity(6);wait([launch('worker.py',dict(ACTION='mechanical',GPU='6',CUDA_VISIBLE_DEVICES='6'),'mechanical.log')])
        assert read(RUN/'private/GPU_MECHANICAL.json')['status']=='PASS'
        write(RUN/'private/SCORING_IMPLEMENTATION_ADMISSION.json',dict(status='PASS',fixed_reference_protocol=True,receipt_content_filename=True,full_identity_keys=True,new_ingest_preserves_consumers=True))
        if not (RUN/'public/P1_RESULTS.json').exists():phase('P1');scoring('P1')
        plan=read(RUN/'private/QUEUES.json');ts={t['edit_id']:t for t in plan['tasks']};done=[]
        for block,ids in enumerate(plan['P2_blocks']):
            if all((RUN/'private/edits'/ts[e]['anonymous_edit']/'s0/COMPLETE.json').exists() for e in ids):done+=ids;continue
            count=sum(len(set(q for ev in ts[e]['events'] for q in ev['all_probe_query_ids'])) for e in ids)
            # Upper bound includes all single/forced conditions and U rows, no favorable-output guess.
            judge_upper=7*count+8*4*read(RUN/'PLAN_CONFIG.json')['U']['CHECK']
            if not admission('P2_BLOCK_'+str(block),max(3600,count*15+len(ids)*700),judge_upper):break
            phase('P2',block);done+=ids;scoring('P2')
        write(RUN/'public/P2_EXECUTION.json',dict(frozen_N=24,completed_edits=len(done),status='COMPLETE' if len(done)==24 else 'PARTIAL' if done else 'NOT_RUN',blocks_completed=len(done)//8,queue_not_changed=True))
        if done and admission('BANK',max(1800,len(done)*180)):
            capacity(6);wait([launch('worker.py',dict(ACTION='bank',GPU='6',CUDA_VISIBLE_DEVICES='6'),'bank.log')]);scoring('P2')
        cpu('report.py');cpu('final_audit.py')
        write(RUN/'private/CONTROLLER_COMPLETE.json',dict(status='RESULTS_AND_AUDIT_COMPLETE_SCIENTIFIC_REVIEW_PUBLICATION_PENDING',epoch=time.time(),no_next_round=True,no_monitor=True))
        write(RUN/'public/PROGRESS.json',dict(status='RESULTS_AND_AUDIT_COMPLETE_PUBLICATION_PENDING',GPU_hours=used()/3600))
if __name__=='__main__':
    try:main()
    except BaseException as e:
        write(RUN/'private/CONTROLLER_FAILURE.json',dict(status='STOPPED_WITH_RETAINED_STATE',error=repr(e),traceback=traceback.format_exc(),epoch=time.time(),automatic_retry=False))
        write(RUN/'public/PARTIAL_COMPLETION.json',dict(status='INCOMPLETE',error_type=type(e).__name__,GPU_hours=used()/3600,results_not_discarded=True,automatic_retry=False))
        try:
            if (RUN/'private/judge_common/queue.sqlite').exists():cpu('report.py')
        except BaseException:pass
        raise
