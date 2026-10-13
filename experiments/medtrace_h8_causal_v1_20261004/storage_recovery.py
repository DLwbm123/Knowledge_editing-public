"""One bounded storage-scan repair; leave the healthy original worker untouched."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import traceback

RUN=Path(os.environ['RUN_ROOT'])
REPAIR=Path(os.environ['REPAIR_ROOT'])
spec=importlib.util.spec_from_file_location('original_controller',RUN/'private/tools/controller.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)

def alive(identity):
    p=Path('/proc')/str(identity['pid'])/'stat'
    try:s=p.read_text().split()
    except FileNotFoundError:return False
    return s[21]==identity['start_ticks'] and s[2]!='Z'

def caps():
    m=c.read(RUN/'RUN_MANIFEST.json')
    assert not (RUN/'STOP').exists()
    assert time.time()<m['deadline_epoch'] and c.used()<m['GPU_seconds_limit']

def start(action,gpu,part=None):
    caps()
    fd,name=tempfile.mkstemp(prefix='e.',suffix='.py');os.close(fd)
    Path(name).write_text("import os,sys,runpy\nsys.path.insert(0,os.environ['RUN_ROOT']+'/private/tools')\nrunpy.run_path(os.environ['REPAIR_ROOT']+'/worker.py',run_name='__main__')\n")
    uuid={4:'GPU-fb8f2a01-3910-41f5-39f3-9ce03b7cb7dd',5:'GPU-4924bbd8-4082-3f4c-b4fb-78ca930122ca'}[gpu]
    env=dict(os.environ,ACTION=action,GPU=str(gpu),CUDA_VISIBLE_DEVICES=str(gpu),CUBLAS_WORKSPACE_CONFIG=':4096:8',M3BENCH_FORMAL_AUTHORIZED_CUDA_VISIBLE_DEVICES=str(gpu),M3BENCH_FORMAL_ALLOWED_CUDA_VISIBLE_DEVICES=str(gpu),M3BENCH_FORMAL_EXPECTED_GPU_UUID=uuid)
    if part is not None:env['PARTITION']=str(part)
    p=subprocess.Popen(['/data/bmw/envs/v0/bin/python','-u',name],env=env,stdout=(RUN/'logs'/f'repaired_{action}_{gpu}.log').open('ab'),stderr=subprocess.STDOUT,start_new_session=True)
    identity=dict(pid=p.pid,start_ticks=Path(f'/proc/{p.pid}/stat').read_text().split()[21],epoch=time.time(),action=action,gpu=gpu,partition=part,entry=name,execution_repair=c.read(REPAIR/'VERSION.json'))
    c.write(REPAIR/f'START_{action}_{gpu}.json',identity)
    print('START',json.dumps(identity),flush=True)
    return p,identity

def main():
    # The original controller must fail after its known failed GPU5 child exits.
    # It cannot launch a bank while this bounded replacement is running.
    failed=c.read(REPAIR/'START_shared_5.json')
    healthy=c.read(RUN/'private/START_shared_4.json')
    original=c.read(RUN/'private/CONTROLLER_RESUME_START.json')
    assert not alive(failed)
    assert c.read(RUN/'private/P1_COMPLETE.json')['branches']==32
    assert c.read(RUN/'private/PHASE_ADMISSION_P2.json')['admitted']
    jobs=[start('shared',5,1)]
    while alive(healthy):
        caps()
        assert jobs[0][0].poll() in (None,0), 'Repaired worker failed; no automatic retry'
        time.sleep(15)
    if not (RUN/'private/SHARED_PART_0_COMPLETE.json').exists():
        failure=c.read(RUN/'private/FAILURE_shared_4.json')
        assert failure['pid']==healthy['pid']
        assert 'FileNotFoundError' in failure['error'] and 'p.stat().st_size' in failure['traceback'] and '/private/edits/' in failure['traceback'], 'Unrelated failure; stop for root-cause review'
        for n in ['private/FAILURE_shared_4.json','RESOURCE_LEDGER.json','logs/shared_4.log']:
            shutil.copy2(RUN/n,REPAIR/('gpu4_'+Path(n).name))
        jobs.append(start('shared',4,0))
    c.wait(jobs)
    for i in (0,1):assert (RUN/f'private/SHARED_PART_{i}_COMPLETE.json').exists()
    until=time.time()+90
    while alive(original) and time.time()<until:time.sleep(5)
    assert not alive(original), 'Original controller did not settle; no duplicate phase'
    if (RUN/'private/CONTROLLER_FAILURE.json').exists():
        shutil.copy2(RUN/'private/CONTROLLER_FAILURE.json',REPAIR/'ORIGINAL_CONTROLLER_FAILURE.json')
    c.wait([start('bank',4)])
    if not c.admit('P3',2*3600):
        c.finish(['P0','P1','P2'],'P3_NOT_ADMITTED_FINITE_BUDGET');return
    c.wait([start('oracle',4)])
    c.finish(['P0','P1','P2','P3'],'FROZEN_GENERATION_COMPLETE')

if __name__=='__main__':
    try:main()
    except Exception as e:
        c.write(REPAIR/'RECOVERY_FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc(),epoch=time.time(),automatic_retry=False))
        raise
