"""Queue a preregistered final consumer within V86's original clock and ledger."""
import json,os,shutil,subprocess,time
from pathlib import Path
r=Path(os.environ['V81_ROOT']);dpc=Path(os.environ['DPC_ROOT']);read=lambda p:json.loads(p.read_text())
def write(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));t.replace(p)
def alive(t):
 try:return Path(f'/proc/{t["pid"]}/stat').read_text().rsplit(')',1)[1].split()[19]==str(t['start_ticks'])
 except FileNotFoundError:return False


def main():
    with (r/'SOURCE_CONSUMER_QUEUE_ONCE').open('x'):pass
    limits=read(r/'RESOURCE_LIMITS.json');deadline=limits['hard_end']-300
    write(r/'SOURCE_CONSUMER_QUEUED.json',dict(time=time.time(),deadline=deadline,physical_GPU=0,wait='DPC complete and exited; 45GiB free',original_budget=True))
    while time.time()<deadline:
        if (dpc/'FAILURE_worker.json').exists() or (r/'STOP').exists():
            write(r/'SOURCE_CONSUMER_BLOCKED.json',dict(reason='Parent failure/stop; retain 8 S0 states',time=time.time()));return
        if (dpc/'MAIN_COMPLETE.json').exists() and (dpc/'RESOURCE_GUARD_EXIT.json').exists():
            free=int(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
            if free>=45*1024:break
        time.sleep(10)
    else:
        write(r/'SOURCE_CONSUMER_BLOCKED.json',dict(reason='Original V86 deadline; keep pending dependency without resetting clock',time=time.time()));return
    targets=read(r/'RESOURCE_TARGETS.json');assert not any(alive(t) for t in targets['targets'])
    assert read(r/'SCORING_COMPLETE.json')['consumers']==775
    audit=r/'source_consumer/original_control';audit.mkdir(parents=True,exist_ok=False)
    for name in ['GUARD_ONCE','RESOURCE_SCOPE_CLOSED.json','RESOURCE_GUARD_EXIT.json','RESOURCE_GUARD_HEALTH.json']:
        p=r/name
        if p.exists():p.rename(audit/name)
    env=dict(os.environ,PHASE='source_compare',PARTITION='0',GPU='0',CUDA_VISIBLE_DEVICES='0')
    log=(r/'logs/source_compare_0.log').open('ab');worker=subprocess.Popen(['/data/bmw/envs/v0/bin/python','-u','/data/bmw/.q.20261023.86/q4.py'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close()
    target=dict(pid=worker.pid,start_ticks=Path(f'/proc/{worker.pid}/stat').read_text().rsplit(')',1)[1].split()[19],gpu=0,phase='source_compare',started_epoch=time.time())
    targets['targets'].append(target);write(r/'RESOURCE_TARGETS.json',targets);write(r/'START_source_compare_0.json',target)
    log=(r/'logs/source_compare_guard.log').open('ab');guard=subprocess.Popen(['/data/bmw/envs/v0/bin/python','-u','/data/bmw/.q.20261023.86/q2.py'],env=dict(env,CONTROL_ACTION='guard'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close()
    write(r/'START_source_compare_guard.json',dict(pid=guard.pid,start_ticks=Path(f'/proc/{guard.pid}/stat').read_text().rsplit(')',1)[1].split()[19],time=time.time()))
    code=worker.wait();targets=read(r/'RESOURCE_TARGETS.json');targets['targets'][-1]['ended_epoch']=time.time();write(r/'RESOURCE_TARGETS.json',targets)
    if code!=0:return
    assert read(r/'SOURCE_CONSUMER_COMPLETE.json')['status']=='PASS' and not alive(target)
    files=[r/'candidates/S0'/f'{i}.pt' for i in range(1,9)];assert set(r.rglob('*.pt'))==set(files)
    rows=[]
    for p in files:
        assert not p.is_symlink() and p.resolve().is_relative_to(r.resolve());rows.append(dict(path=str(p),bytes=p.stat().st_size))
    for p in files:p.unlink()
    write(r/'SOURCE_CONSUMER_FINAL_RETIREMENT.json',dict(status='RETIRED',time=time.time(),files=rows,count=8,bytes=sum(x['bytes'] for x in rows),no_backup=True,rebuild='Requires retraining',all_registered_weight_consumers_complete=True))


if __name__=='__main__':main()
