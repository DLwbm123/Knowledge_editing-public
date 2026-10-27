"""One failed shard, frozen terminals reused; cumulative counts and original clock."""
import json,os,time,subprocess,shutil,sys
from pathlib import Path

def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)

def main(root,part):
    from resource_guard_v22 import identity
    from consumers_v22 import ARMS
    out=root/str(part);limits=read(root/'RESOURCE_LIMITS.json');guard=read(root/'RESOURCE_GUARD_LAUNCH.json')
    assert not (root/'STOP').exists() and time.time()<limits['hard_end']-1200
    assert identity(guard) and time.time()-read(root/'RESOURCE_GUARD_HEALTH.json')['time']<120
    assert not (out/'recovery1').exists()
    old=read(out/'COMPACT_LAUNCH.json');assert not identity(old)
    failure=read(out/'COMPACT_FAILURE.json');exit=read(out/'COMPACT_EXIT.json')
    assert exit['code']==1 and exit['counts']['generations']==0
    assert 'Expected all tensors to be on the same device' in failure['error']
    tasks=read(root/'DATA_ADMISSION.json')['tasks'][part::5];folder=out/tasks[0]['key']
    assert read(folder/'FROZEN.json')['all_terminals_frozen']
    assert not list(out.glob('E*/attempts/*/*.json')) and not list(out.glob('E*/outputs/*/*.json'))
    for arm in ARMS[:5]:
        assert read(folder/arm/'TRAIN_COMPLETE.json')['save_load_exact'] and (folder/arm/'final.pt').is_file()
    for arm in ARMS[5:]:
        assert read(folder/(arm+'_TRAIN.json'))['save_load_exact'] and (folder/(arm+'.pt')).is_file()
    assert (root/'TEACHER_READY.json').is_file()
    fields=subprocess.check_output(['nvidia-smi','-i',str(old['physical_gpu']),'--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
    assert fields[0].strip()==old['uuid'] and int(fields[1])>=37000
    assert shutil.disk_usage(root).free>limits['free_reserve_bytes']
    archive=out/'recovery1';archive.mkdir()
    for name in ['COMPACT_FAILURE.json','COMPACT_EXIT.json','COMPACT_MODEL_READY.json','COMPACT_REUSE_ADMISSION.json']:
        shutil.copyfile(out/name,archive/name)
    write(archive/'ADMISSION.json',dict(status='PASS',old_identity_gone=True,all_seven_frozen=True,no_generation_attempts=True,prior_counts=exit['counts'],prior_seconds=exit['seconds'],original_hard_end=limits['hard_end'],repair='Exact tensor equality on CPU; no numerical update or threshold change',time=time.time()))
    env=read(out/'COMPACT_ENV.json');entry=Path(os.environ['V22_NEUTRAL_TMP'])/f'q.20261020.22.u{part}.py';assert not entry.exists()
    # Store the tiny neutral entry within the measured root; guard need not reload external paths.
    measured=archive/'entry.py'
    measured.write_text('import os,json,time,runpy\nfrom pathlib import Path\np=Path('+repr(str(archive))+')\nos.environ.update({k:str(v) for k,v in json.loads((p.parent/"COMPACT_ENV.json").read_text()).items()})\nwhile not (p/"GO").exists():\n if (Path(os.environ["V22_ROOT"])/"STOP").exists() or time.time()>'+repr(limits['hard_end']-300)+':raise SystemExit(2)\n time.sleep(.1)\nrunpy.run_path('+repr(str(root/'tools/worker_recovery_v22.py'))+',run_name="__main__")\n')
    entry.symlink_to(measured)
    with (archive/'worker.log').open('ab') as log:
        proc=subprocess.Popen([sys.executable,'-u',str(entry)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=str(entry.parent))
    ticks=Path(f'/proc/{proc.pid}/stat').read_text().rsplit(')',1)[1].split()[19]
    job=dict(old,pid=proc.pid,start_ticks=ticks,time=time.time(),recovery='recovery1',original_job=old)
    write(archive/'LAUNCH.json',job)
    registry=read(root/'RESOURCE_TARGETS.json');registry['targets'].append(job);write(root/'RESOURCE_TARGETS.json',registry)
    assert identity(job)
    (out/'COMPACT_FAILURE.json').unlink();(out/'COMPACT_EXIT.json').unlink()
    write(out/'COMPACT_LAUNCH.json',job)
    start=read(root/'COMPACT_START.json');start['jobs'][part]=job;write(root/'COMPACT_START.json',start)
    write(archive/'GO',dict(registered_with_original_guard=True,time=time.time()))
    print(json.dumps(dict(status='LAUNCHED',job=job)))

if __name__=='__main__':main(Path(os.environ['V22_ROOT']),int(os.environ['PARTITION']))
