"""One-shot CPU/GPU dispatch; every neutral child is registered before GO."""
import json,os,subprocess,sys,time,traceback
from pathlib import Path
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
def main(root,phase):
    sys.path.insert(0,str(root/'tools'))
    from resource_guard import identity
    limits=read(root/'RESOURCE_LIMITS.json')
    assert not (root/'STOP').exists() and time.time()<limits['hard_end']-1200
    assert identity(read(root/'RESOURCE_GUARD_LAUNCH.json'))
    assert time.time()-read(root/'RESOURCE_GUARD_HEALTH.json')['time']<120
    if phase=='gpu':assert read(root/'CPU_ADMISSION.json')['status']=='PASS'
    assert phase in ('cpu','gpu')
    with (root/(phase.upper()+'_LAUNCH_ONCE')).open('x'):pass
    jobs=[];entries=[];parts=range(5) if phase=='gpu' else [-1]
    env_base=read(root/'PARENT_ENV.json')
    for part in parts:
        env=dict(env_base);env.update(V24_ROOT=str(root),V24_PRIOR_ROOT=read(root/'PROVENANCE.json')['prior_root'],
                                     PARTITION=str(part),PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(root/'tmp'))
        out=root/str(part) if part>=0 else root/'cpu';out.mkdir(exist_ok=True)
        if phase=='gpu':
            gpu=[0,1,3,4,5][part]
            fields=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
            assert int(fields[1])>=60000,'Wait for sufficient memory; do not disturb existing processes'
            env.update(CUDA_VISIBLE_DEVICES=str(gpu),M3BENCH_FORMAL_EXPECTED_GPU_UUID=fields[0].strip(),
                       M3BENCH_FORMAL_AUTHORIZED_CUDA_VISIBLE_DEVICES=str(gpu),
                       M3BENCH_FORMAL_ALLOWED_CUDA_VISIBLE_DEVICES='0,1,3,4,5',GPU=str(gpu))
        else:env['CUDA_VISIBLE_DEVICES']=''
        write(out/'ENV.json',env)
        tag=os.environ.get('V24_LAUNCH_TAG','')
        assert not tag or tag=='.r1','Only documented pre-model engineering repair admitted'
        entry=Path(os.environ['V24_NEUTRAL_TMP'])/('q.20261020.24.'+('w'+str(part) if part>=0 else 'c')+tag+'.py')
        assert not entry.exists()
        source='worker.py' if part>=0 else 'cpu_admission.py'
        entry.write_text('import json,os,time,runpy\nfrom pathlib import Path\np=Path('+repr(str(out))+')\nos.environ.update({k:str(v) for k,v in json.loads((p/"ENV.json").read_text()).items()})\nwhile not (p/"GO.json").exists():\n if (Path(os.environ["V24_ROOT"])/"STOP").exists() or time.time()>'+repr(limits['hard_end']-300)+':raise SystemExit(2)\n time.sleep(.1)\nrunpy.run_path('+repr(str(root/'tools'/source))+',run_name="__main__")\n')
        with (out/'worker.log').open('ab') as log:
            proc=subprocess.Popen([env['TRAIN_PYTHON'],'-u',str(entry)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=os.environ['V24_NEUTRAL_TMP'])
        ticks=Path(f'/proc/{proc.pid}/stat').read_text().rsplit(')',1)[1].split()[19]
        job=dict(pid=proc.pid,start_ticks=ticks,partition=part,time=time.time(),phase=phase,
                 deadline=time.time()+(4*3600 if part>=0 else 240))
        if part>=0 and (root/'GPU_DEADLINES.json').exists():job['deadline']=read(root/'GPU_DEADLINES.json')[str(part)]
        if part>=0:job.update(physical_gpu=gpu,uuid=fields[0].strip())
        write(out/'LAUNCH.json',job);jobs.append(job);entries.append(str(entry))
        # Register immediately: a later allocation/launch failure never leaves an unguarded child.
        registry=read(root/'RESOURCE_TARGETS.json');registry['targets'].append(job);write(root/'RESOURCE_TARGETS.json',registry)
        limits=read(root/'RESOURCE_LIMITS.json');limits['external_owned_paths'].append(str(entry));write(root/'RESOURCE_LIMITS.json',limits)
    assert all(identity(j) for j in jobs)
    write(root/(phase.upper()+'_START.json'),dict(time=time.time(),jobs=jobs,original_clock=limits['original_start']))
    for job in jobs:write((root/str(job['partition']) if job['partition']>=0 else root/'cpu')/'GO.json',dict(registered_before_GO=True))
    print(json.dumps(dict(status='LAUNCHED',phase=phase,jobs=jobs)))
if __name__=='__main__':
    root=Path(sys.argv[1])
    try:main(root,sys.argv[2])
    except BaseException:
        write(root/'LAUNCH_FAILURE.json',dict(time=time.time(),error=traceback.format_exc(),automatic_retry=False));raise
