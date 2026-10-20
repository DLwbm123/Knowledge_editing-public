"""One-shot dispatch after all main consumers exit; never a second monitor."""
import json,os,time,subprocess
from pathlib import Path
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
def allocation(plan,exits):
    forwards=sum(e['counts']['forwards'] for e in exits);generations=sum(e['counts']['generations'] for e in exits);seconds=sum(e['seconds'] for e in exits)
    per={str(i):sum(pair.split('|')[0] in {t['key'] for t in plan['tasks'][i::5]} for pair in plan['missing_pairs']) for i in range(5)}
    assert sum(per.values())==1584 and generations+1584<=4000
    f=min(350000,(3000000-forwards)//5);s=min(7200,(80*3600-seconds)/5)
    assert f>0 and s>0 and 5*s<=10*3600
    return dict(forwards_per_partition=f,seconds_per_partition=s,generations_per_partition=per,main_actual=dict(forwards=forwards,generations=generations,GPU_process_seconds=seconds),substage_new_Base=0,substage_new_query_forward=0)
def main(root):
    from consumers_v22 import validate
    from resource_guard_v22 import identity
    assert not (root/'STOP').exists() and not (root/'BANK_ONCE').exists()
    limits=read(root/'RESOURCE_LIMITS.json');assert time.time()<limits['hard_end']-1200
    guard=read(root/'RESOURCE_GUARD_LAUNCH.json');assert identity(guard)
    assert time.time()-read(root/'RESOURCE_GUARD_HEALTH.json')['time']<120
    assert read(root/'BANK_CPU_ADMISSION.json')['status']=='PASS'
    exits=[read(root/str(i)/'COMPACT_EXIT.json') for i in range(5)]
    assert all(e['code']==0 for e in exits)
    assert not any(identity(j) for j in read(root/'COMPACT_START.json')['jobs'])
    assert validate(root)['method_outputs']==1512
    plan=read(root/'ROUTE_PLAN_FROZEN.json');alloc=allocation(plan,exits)
    for task in plan['tasks']:
        folder=root/str((task['order']-1)%5)/task['key']
        assert read(folder/'MODEL_CONSUMERS_COMPLETE.json')['status']=='PASS'
        assert (folder/'SOURCE/final.pt').is_file() and (folder/'R0_ANCHORS.pt').is_file()
    gpus=[0,1,3,4,5];uuids=[]
    for gpu in gpus:
        fields=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
        assert int(fields[1])>=37000;uuids.append(fields[0].strip())
    with (root/'BANK_ONCE').open('x'):pass
    write(root/'BANK_ALLOCATION.json',alloc);jobs=[];entries=[]
    # Every child waits without importing torch until all five identities are guarded.
    for part,gpu in enumerate(gpus):
        out=root/'bank'/str(part);out.mkdir(parents=True,exist_ok=True)
        env=read(root/str(part)/'COMPACT_ENV.json');env.update(PARTITION=str(part),CUDA_VISIBLE_DEVICES=str(gpu),M3BENCH_FORMAL_EXPECTED_GPU_UUID=uuids[part])
        write(out/'ENV.json',env);entry=Path(f'/data/bmw/tmp/q.20261020.22.r{part}.py');assert not entry.exists()
        entry.write_text('import os,json,time,runpy\nfrom pathlib import Path\np=Path('+repr(str(out))+')\nos.environ.update({k:str(v) for k,v in json.loads((p/"ENV.json").read_text()).items()})\nwhile not (p/"GO.json").exists():\n if (Path(os.environ["V22_ROOT"])/"STOP").exists() or time.time()>'+repr(limits['hard_end']-300)+':raise SystemExit(2)\n time.sleep(.1)\nrunpy.run_path('+repr(str(root/'tools/bank_worker_v22.py'))+',run_name="__main__")\n')
        with (out/'worker.log').open('ab') as log:
            proc=subprocess.Popen(['/data/bmw/envs/v0/bin/python','-u',str(entry)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd='/data/bmw/tmp')
        ticks=Path(f'/proc/{proc.pid}/stat').read_text().rsplit(')',1)[1].split()[19]
        job=dict(pid=proc.pid,start_ticks=ticks,physical_gpu=gpu,partition=part,uuid=uuids[part],time=time.time())
        write(out/'LAUNCH.json',job);jobs.append(job);entries.append(str(entry))
    registry=read(root/'RESOURCE_TARGETS.json');registry['targets']+=jobs;write(root/'RESOURCE_TARGETS.json',registry)
    limits['external_owned_paths']+=entries;write(root/'RESOURCE_LIMITS.json',limits)
    assert all(identity(j) for j in jobs)
    write(root/'BANK_START.json',dict(time=time.time(),jobs=jobs,allocation=alloc,hard_end=limits['hard_end'],original_clock_not_reset=True))
    for part in range(5):write(root/'bank'/str(part)/'GO.json',dict(registered_with_original_guard=True))
    print(json.dumps(dict(status='LAUNCHED',jobs=jobs)))
if __name__=='__main__':
    import sys,traceback
    root=Path(sys.argv[1])
    try:main(root)
    except BaseException:
        write(root/'BANK_LAUNCH_FAILURE.json',dict(time=time.time(),error=traceback.format_exc(),automatic_retry=False));raise
