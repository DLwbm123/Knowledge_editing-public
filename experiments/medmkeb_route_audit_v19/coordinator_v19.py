"""Dependency-only background queue; no V18 process or output is changed."""
import json,os,time,subprocess,sys
from pathlib import Path
ROOT=Path(os.environ['V19_RUN']);V18=Path(os.environ['V18_RUN'])
def read(p):return json.loads(Path(p).read_text())
def write(name,value):
    p=ROOT/name;tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p)
def main():
    with (ROOT/'COORDINATOR2_ONCE').open('x'):pass
    limit=float(read(V18/'FORMAL_START.json')[0]['deadline']);write('QUEUE_STATUS.json',dict(status='WAITING_V18_DEPENDENCY',time=time.time(),deadline=limit))
    while time.time()<limit:
        for gpu in (0,1):
            f=V18/str(gpu)/'FORMAL_FAILURE.json'
            if f.exists():write('BLOCKED.json',dict(reason='V18 engineering failure; no additional model work launched',gpu=gpu,time=time.time()));return
        for part in (0,1):
            if (ROOT/str(part)/'PREFETCH_FAILURE.json').exists():write('BLOCKED.json',dict(reason='Prefetch engineering failure, do not duplicate keys',partition=part));return
        if all((V18/str(g)/'FORMAL_EXIT.json').exists() and (ROOT/str(g)/'PREFETCH_EXIT.json').exists() for g in (0,1)):break
        time.sleep(60)
    assert time.time()<limit,'Original global deadline reached before dependency admission'
    sys.path.insert(0,str(V18/'0/tools'));import consumers_v18
    records,expected=consumers_v18.collect(V18);assert consumers_v18.validate(records,expected)['outputs']==350
    sys.path.insert(0,str(ROOT/'tools'));from audit_v19 import inventory_be
    write('BE_SINGLE_REUSE_SUMMARY.json',inventory_be(V18,ROOT))
    for a in read(V18/'FORMAL_START.json'):
        assert read(V18/str(a['gpu'])/'FORMAL_EXIT.json')['code']==0
        p=Path('/proc')/str(a['pid']);assert not p.exists() or (p/'stat').read_text().split()[21]!=str(a['start_ticks'])
    while time.time()<limit:
        available=[]
        for g in (3,4):
            uuid,free=subprocess.check_output(['nvidia-smi','-i',str(g),'--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(', ');available.append(dict(gpu=g,uuid=uuid,free_MiB=int(free)))
        if min(x['free_MiB'] for x in available)>=37000:break
        write('QUEUE_STATUS.json',dict(status='WAITING_MEMORY',memory=available,time=time.time(),deadline=limit));time.sleep(60)
    assert time.time()<limit
    began=time.time();deadline=min(limit,began+7200-max(read(ROOT/str(g)/'PREFETCH_EXIT.json')['seconds'] for g in (0,1)));env=read(V18/'PARENT_ENV.json');env.update(V18_RUN=str(V18),V19_RUN=str(ROOT),DEADLINE_EPOCH=str(deadline));launch=[]
    for part,a in enumerate(available):
        assert read(ROOT/str(part)/'PREFETCH_EXIT.json')['code']==0
        g=a['gpu'];out=ROOT/str(part);assert (out/'tools').resolve()==ROOT/'tools'
        entry=Path('/data/bmw/tmp')/('q.20261019.19.'+str(g)+'.py');entry.write_text("import runpy,os\nrunpy.run_path(os.environ['WORKER_FILE'],run_name='__main__')\n")
        e=dict(os.environ);e.update(env);e.update(CUDA_VISIBLE_DEVICES=str(g),GPU=str(g),PARTITION=str(part),OUTPUT_RUN=str(out),WORKER_FILE=str(ROOT/'tools/worker_v19.py'),M3BENCH_FORMAL_AUTHORIZED_CUDA_VISIBLE_DEVICES=str(g),M3BENCH_FORMAL_ALLOWED_CUDA_VISIBLE_DEVICES='3,4,5',M3BENCH_FORMAL_EXPECTED_GPU_UUID=a['uuid']);log=(out/'worker.log').open('ab');p=subprocess.Popen(['/data/bmw/envs/v0/bin/python','-u',str(entry)],env=e,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close();ticks=(Path('/proc')/str(p.pid)/'stat').read_text().split()[21]
        item=dict(a,partition=part,pid=p.pid,start_ticks=ticks,start=began,deadline=deadline,entry=str(entry));write(str(part)+'/LAUNCH.json',item);launch.append(item)
    write('START.json',launch);write('QUEUE_STATUS.json',dict(status='MODEL_WORK_RUNNING',time=time.time(),deadline=deadline))
if __name__=='__main__':
    try:main()
    except BaseException:
        import traceback
        write('COORDINATOR_FAILURE.json',dict(error=traceback.format_exc(),time=time.time(),automatic_retry=False));raise
