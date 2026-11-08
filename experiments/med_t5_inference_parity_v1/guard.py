"""Independent wall-time/storage guard for two sequential, clean GPU processes."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


cfg=json.loads(Path(os.environ['DIAG_CONFIG']).read_text());root=Path(cfg['root']);active=None


def write(name,value):
    p=root/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2));t.replace(p)


def stop(*_):
    if active and active.poll() is None:
        os.killpg(active.pid,signal.SIGTERM)
        try:active.wait(timeout=15)
        except subprocess.TimeoutExpired:os.killpg(active.pid,signal.SIGKILL);active.wait(timeout=10)
    raise InterruptedError('Diagnostic guard stop')


signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
with (root/'GUARD_ONCE').open('x'):pass
results={}
try:
    for arm in ['A','B']:
        while True:
            free=int(subprocess.check_output(['nvidia-smi','-i',cfg['gpu'],'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
            write('HEALTH.json',dict(time=time.time(),hard_end=cfg['clock']['hard_end'],arm=arm,pending_GPU=free<60000))
            if time.time()>cfg['clock']['hard_end']-300:raise TimeoutError('GPU admission exceeded original clock')
            if free>=60000:break
            time.sleep(5)
        env=dict(os.environ,DIAG_ARM=arm,CURRENT_SOURCE=str(root/'tools/worker.py'))
        with (root/f'{arm}.log').open('x') as log:
            active=subprocess.Popen([cfg['python'],str(Path(cfg['neutral'])/'w.py')],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write('IDENTITY.private.json',dict(arm=arm,pid=active.pid))
        while active.poll() is None:
            size=0
            for owned in [root,Path(cfg['neutral'])]:
                for directory,_,files in os.walk(owned):
                    for name in files:
                        try:
                            p=Path(directory)/name
                            if not p.is_symlink():size+=p.stat().st_blocks*512
                        except FileNotFoundError:pass
            reason='TIME' if time.time()>cfg['clock']['hard_end']-90 else 'STORAGE' if size+cfg['storage_reserve']>=cfg['byte_limit'] else 'FREE_SPACE' if shutil.disk_usage(root).free<8*1024**3 else None
            write('HEALTH.json',dict(time=time.time(),bytes=size,hard_end=cfg['clock']['hard_end'],arm=arm))
            if reason:write('STOP.json',dict(reason=reason,time=time.time()));stop()
            time.sleep(2)
        results[arm]=active.returncode;write('EXIT.json',dict(results=results,time=time.time()))
        if active.returncode:raise RuntimeError(f'Arm {arm} exited {active.returncode}')
    assert all((root/arm/'COMPLETE.json').exists() for arm in ['A','B'])
    write('GENERATION_COMPLETE.json',dict(generations=64,results=results,updates=0,Judge_calls=0,time=time.time()))
    env=dict(os.environ,CURRENT_SOURCE=str(root/'tools/analyze.py'))
    with (root/'analysis.log').open('x') as log:
        finished=subprocess.run([cfg['python'],str(Path(cfg['neutral'])/'w.py')],env=env,stdout=log,stderr=subprocess.STDOUT,
            timeout=max(1,int(cfg['clock']['hard_end']-time.time()-30)))
    assert finished.returncode==0 and (root/'COMPLETE.json').exists(),'CPU comparison failed'
except BaseException as error:
    write('GUARD_FAILURE.json',dict(error=repr(error),results=results,time=time.time()));raise
