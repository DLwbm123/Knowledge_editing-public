"""One-shot guarded CPU evaluation after all hidden judgments are frozen."""
import os,json,time,subprocess
from pathlib import Path
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
def main(root):
    from resource_guard_v22 import identity
    assert not (root/'STOP').exists() and not (root/'ROUTE_DISPATCH_ONCE').exists()
    limits=read(root/'RESOURCE_LIMITS.json');assert time.time()<limits['hard_end']-1200
    assert identity(read(root/'RESOURCE_GUARD_LAUNCH.json')) and time.time()-read(root/'RESOURCE_GUARD_HEALTH.json')['time']<120
    assert read(root/'ROUTE_CPU_ADMISSION.json')['status']=='PASS'
    assert all(read(root/'bank'/str(i)/'EXIT.json')['code']==0 and not identity(read(root/'bank'/str(i)/'LAUNCH.json')) for i in range(5))
    packet=read(root/'BANK_BLIND_PACKET.json');verdict=dict(read(root/'BLIND_REUSE.json'));verdict.update(read(root/'BLIND_VERDICTS.json'));verdict.update(read(root/'BANK_BLIND_VERDICTS.json'))
    assert set(verdict)=={x['code'] for x in packet['items']}
    assert all(x.get('reason') and x['label'] in ('CORRECT','INCORRECT','UNCERTAIN','ABSTAIN') for x in verdict.values())
    entry=Path('/data/bmw/tmp/q.20261020.22.f.py');assert not entry.exists()
    with (root/'ROUTE_DISPATCH_ONCE').open('x'):pass
    entry.write_text('import time,sys,runpy\nfrom pathlib import Path\nr=Path('+repr(str(root))+')\nwhile not (r/"ROUTE_GO.json").exists():\n if (r/"STOP").exists() or time.time()>'+repr(limits['hard_end']-300)+':raise SystemExit(2)\n time.sleep(.1)\nsys.path.insert(0,str(r/"tools"))\nsys.argv=["q",str(r)]\nrunpy.run_path(str(r/"tools/route_evaluate_v22.py"),run_name="__main__")\n')
    with (root/'route_cpu.log').open('ab') as log:
        proc=subprocess.Popen(['/data/bmw/envs/v0/bin/python','-u',str(entry)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd='/data/bmw/tmp')
    job=dict(pid=proc.pid,start_ticks=Path(f'/proc/{proc.pid}/stat').read_text().rsplit(')',1)[1].split()[19],stage='SOURCE_ROUTE_CPU_FIT',time=time.time())
    registry=read(root/'RESOURCE_TARGETS.json');registry['targets'].append(job);write(root/'RESOURCE_TARGETS.json',registry)
    limits['external_owned_paths'].append(str(entry));write(root/'RESOURCE_LIMITS.json',limits);write(root/'ROUTE_LAUNCH.json',job)
    assert identity(job);write(root/'ROUTE_GO.json',dict(registered_with_original_guard=True));print(json.dumps(job))
if __name__=='__main__':
    import sys
    main(Path(sys.argv[1]))
