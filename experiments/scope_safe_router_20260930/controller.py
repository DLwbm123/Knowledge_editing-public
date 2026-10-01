"""Finite phase executor, not a recurring monitor or open-ended experiment search."""
import os,time,fcntl,subprocess,traceback,shutil
from pathlib import Path
from resources import ROOT,read,write
from phases import enqueue,score_closed,decide

def alive(pid):
    try:return (Path('/proc')/str(pid)/'stat').read_text().split()[2]!='Z'
    except FileNotFoundError:return False

def restore():
    q=read(ROOT/'QUEUE.json')
    if not q:enqueue('DEV');return 'DEV',[]
    phase=q[-1]['phase'];assert phase in ['DEV','REG']
    active=[]
    for a in read(ROOT/'ACTIVE_PROCESSES.json') if (ROOT/'ACTIVE_PROCESSES.json').exists() else []:
        if not alive(a['pid']):continue
        process=Path('/proc')/str(a['pid'])
        argv=process.joinpath('cmdline').read_bytes().split(b'\0')
        env=dict(x.split(b'=',1) for x in process.joinpath('environ').read_bytes().split(b'\0') if b'=' in x)
        assert b'/tmp/e14.py' in argv and env.get(b'RUN_ROOT')==str(ROOT).encode(),'Foreign live process: never adopt or terminate'
        assert env.get(b'PHYSICAL_GPU')==str(a['index']).encode()
        binding=next(g for g in read(ROOT/'GPU_BINDINGS.json')['devices'] if g['index']==a['index'])
        assert env.get(b'CUDA_VISIBLE_DEVICES')==binding['uuid'].encode()
        active.append(a)
    for j in q:
        if j['status']=='RUNNING':
            assert any(a['pid']==j['pid'] for a in active),'Dead job needs diagnosed repair and recorded recovery before requeue'
    return phase,active

def main():
    lock=(ROOT/'CONTROLLER.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    write(ROOT/'CONTROLLER_PID.json',dict(pid=os.getpid(),entry='/tmp/e15.py'))
    assert read(ROOT/'public/ROUTER_MECHANICAL_TESTS.json')['status']=='PASS'
    if read(ROOT/'RUN_STATUS.json').get('phase')=='CLOSED':return
    phase,active=restore()
    while True:
        q=read(ROOT/'QUEUE.json')
        for a in active[:]:
            if alive(a['pid']):continue
            active.remove(a)
            if any(j.get('pid')==a['pid'] and j['status']=='RUNNING' for j in q):raise RuntimeError('Worker failed; preserve recovery, no automatic retry')
        write(ROOT/'ACTIVE_PROCESSES.json',active)
        if (ROOT/'STOP').exists() or (ROOT/'JUDGE_FAILURE.json').exists():
            write(ROOT/'RUN_STATUS.json',dict(status='BLOCKED',phase=phase,reason='STOP_OR_JUDGE_FAILURE; no new jobs, preserve evidence'))
            return
        pending=[j for j in q if j['phase']==phase and j['status']!='COMPLETE']
        if not pending and not active and score_closed(phase):
            from reporting import report
            report(phase)
            if phase=='DEV':
                if not decide():finish('CLOSED_NOT_ADMITTED','No DEV router candidate clear joint PASS');return
                phase='REG'
            else:finish('COMPLETE','Finite preregistered REG complete');return
            enqueue(phase);continue
        if pending:
            db=read(ROOT/'STORAGE_LEDGER.json');managed=sum(a['bytes']+a.get('previous_bytes',0) for a in db['artifacts'].values() if a['status']!='DELETED')
            if managed>=10*1024**3:write(ROOT/'STORAGE_SOFT_AUDIT.json',dict(epoch=time.time(),managed_bytes=managed,action='No blind deletion; retained active and declared consumers',phase=phase))
            if managed>=18*1024**3:raise RuntimeError('Storage hard limit including 2 GiB control reserve')
            free=shutil.disk_usage(ROOT).free
            if free<8*1024**3:raise RuntimeError('Storage free-space guard')
            for g in read(ROOT/'GPU_BINDINGS.json')['devices']:
                if any(a['index']==g['index'] for a in active):continue
                if not any(j['status']=='PENDING' and all((ROOT/p).exists() for p in j.get('requires',[])) for j in pending):continue
                value=int(subprocess.check_output(['nvidia-smi','-i',str(g['index']),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
                if value<30000:continue
                uuid=subprocess.check_output(['nvidia-smi','-i',str(g['index']),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip();assert uuid==g['uuid']
                env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=uuid,PINNED_GPU_UUID=uuid,PHYSICAL_GPU=str(g['index']),PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',TMPDIR='/data/bmw/tmp',HF_HOME='/data/bmw/cache/huggingface',TORCH_HOME='/data/bmw/cache/torch',XDG_CACHE_HOME='/data/bmw/cache',CUDA_CACHE_PATH='/data/bmw/cache/cuda')
                with (ROOT/'logs'/f'worker{g["index"]}.log').open('ab') as f:p=subprocess.Popen(['/data/bmw/envs/v0/bin/python','/tmp/e14.py'],env=env,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                active.append(dict(pid=p.pid,index=g['index']));write(ROOT/'ACTIVE_PROCESSES.json',active)
                write(ROOT/'RUN_STATUS.json',dict(status='RUNNING',phase=phase,epoch=time.time(),workers=active,judge_model='gpt-6.1-sol'))
        time.sleep(15)

def finish(status,reason,reg=None):
    from reporting import close
    close(status,reason,reg);write(ROOT/'RUN_STATUS.json',dict(status=status,reason=reason,epoch=time.time(),phase='CLOSED',REG=reg or ('COMPLETE' if status=='COMPLETE' else 'NOT_ADMITTED')))
if __name__=='__main__':
    try:main()
    except Exception as e:
        write(ROOT/'CONTROLLER_FAILURE.json',dict(error=str(e),traceback=traceback.format_exc()));write(ROOT/'RUN_STATUS.json',dict(status='BLOCKED',reason='CONTROLLER_FAILURE'));raise
