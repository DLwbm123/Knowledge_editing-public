"""One bounded repair of the missing time import; default is inspection only."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from launch import read,write,remote


def main():
    cfg=read(os.environ['DISPATCH_CONFIG']);local=Path(cfg['local'])
    prior=read(local/'DISPATCH_RECEIPT.json')
    for pid in (prior['local_guard_pid'],read(local/'LOCAL_SCORER_IDENTITY.json')['pid']):
        status=subprocess.run(['ps','-p',str(pid),'-o','pid='],capture_output=True,text=True)
        assert not status.stdout.strip(), 'Prior local process still exists; inspect its identity'
    assert read(local/'EXIT.json')['code']==1 and read(local/'LOCAL_GUARD_EXIT.json')['code']==1
    assert not list((local/'attempts').glob('*.json')), 'This repair is restricted to pre-grading failure'
    source=Path(__file__).parent
    replacements={name:(source/name).read_text() for name in ('run.py','ra_curvature_rebuild.py')}
    for name,value in replacements.items():compile(value,name,'exec')
    commit=subprocess.check_output(['git','-C',cfg['workspace'],'rev-parse','HEAD'],text=True).strip()
    prefix='import os,json,sys,subprocess,time,shutil\nfrom pathlib import Path\nq=Path('+repr(cfg['neutral'])+')\nos.environ.update(json.loads((q/\'ENV.json\').read_text()))\nr=Path(os.environ[\'EXPERIMENT_ROOT\'])\n'
    checks=prefix+"""
clock=json.loads((r/'CLOCK.json').read_text());lock=json.loads((r/'PROTOCOL_LOCK.json').read_text())
assert time.time()<clock['hard_end']-620
assert not (r/'STOP.json').exists() and not (r/'RECOVERY_REQUEST.json').exists()
for name in ('DISPATCH.json','WORKER_IDENTITY.json'):
 identity=json.loads((r/name).read_text());stat=Path('/proc',str(identity['pid']),'stat')
 assert not stat.exists() or stat.read_text().split(') ',1)[1].split()[19]!=str(identity['start_ticks']), 'Prior remote process is alive'
exit=json.loads((r/'WORKER_EXIT.json').read_text())
assert exit['code']==json.loads((r/'GUARD_EXIT.json').read_text())['code']==1
failure=json.loads((r/'FAILURE.json').read_text())
assert 'NameError' in failure['error'] and "'time' is not defined" in failure['error']
assert exit['counts']['updates']==exit['counts']['generations']==exit['counts']['alpha_selections']==0
assert (r/'curvature/anchors/0.pt').is_file()
gpu=subprocess.check_output(['nvidia-smi','-i',str(lock['selected_physical_GPU']),'--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
assert gpu[0].strip()==lock['expected_GPU_UUID'] and int(gpu[1])>=lock['minimum_free_MiB']
"""
    snapshot=remote(checks+"print(json.dumps(dict(status='READY',clock=clock,prior_counts=exit['counts'],saved_anchors=len(list((r/'curvature/anchors').glob('*.pt'))))))")
    if os.environ.get('APPLY_RECOVERY')!='1':print(json.dumps(snapshot));return
    assert not (local/'engineering_recovery1').exists()
    repair=checks+"""
archive=r/'engineering_recovery1';archive.mkdir()
for name in ('FAILURE.json','WORKER_EXIT.json','GUARD_EXIT.json','SCORE_FAILURE.json','GUARD_HEALTH.json'):
 p=r/name
 if p.exists():p.rename(archive/name)
for name in ('DISPATCH.json','WORKER_IDENTITY.json','SOURCE_VERSION.json'):
 shutil.copy2(r/name,archive/name)
for name,value in __REPLACEMENTS__.items():
 shutil.copy2(r/'tools'/name,archive/name)
 (r/'tools'/name).write_text(value)
request=dict(sequence=1,old_worker_exited=True,old_guard_exited=True,clock=clock,input_binding=lock['input_binding'],prior_counts=exit['counts'],engineering_commit=__COMMIT__,reason='Missing time import; saved anchor reused; no scientific change',time=time.time())
(r/'RECOVERY_REQUEST.json').write_text(json.dumps(request,indent=2))
with (r/'guard.log').open('a') as log:
 child=subprocess.Popen([sys.executable,str(q/'q.py')],env=dict(os.environ,ENTRY_MODE='guard'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
ticks=Path('/proc',str(child.pid),'stat').read_text().split(') ',1)[1].split()[19]
value=dict(pid=child.pid,start_ticks=ticks,time=time.time(),source_commit=__COMMIT__)
(r/'DISPATCH.json').write_text(json.dumps(value,indent=2));print(json.dumps(value))
""".replace('__REPLACEMENTS__',repr(replacements)).replace('__COMMIT__',repr(commit))
    started=remote(repair)
    time.sleep(2)
    health=remote(prefix+"print(json.dumps(dict(worker=json.loads((r/'WORKER_IDENTITY.json').read_text()),exit=(r/'GUARD_EXIT.json').exists(),failure=(r/'FAILURE.json').exists())))")
    assert not health['exit'] and not health['failure']
    archive=local/'engineering_recovery1';archive.mkdir()
    for name in ('FAILURE.json','EXIT.json','LOCAL_GUARD_EXIT.json','LOCAL_GUARD_HEALTH.json','DISPATCH_RECEIPT.json','LOCAL_SCORER_IDENTITY.json','IDENTITY.json'):
        p=local/name
        if p.exists():p.rename(archive/name)
    env=dict(os.environ,SCORING_CONFIG=cfg['scoring_config'],JUDGE_SHARED_ROOT=str(local),LOCAL_NEUTRAL_ENTRY=str(local/'s.py'),LOCAL_ENTRY_MODE='guard')
    with (local/'score.log').open('a') as log:
        child=subprocess.Popen([sys.executable,str(local/'s.py')],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=local)
    receipt=dict(time=time.time(),remote_guard=started,remote_worker=health['worker'],local_guard_pid=child.pid,source_commit=commit,original_clock=snapshot['clock'],prior_counts=snapshot['prior_counts'])
    write(local/'DISPATCH_RECEIPT.json',receipt);write(archive/'RECOVERY_RECEIPT.json',receipt)
    state=read(cfg['state']);state.update(status='STARTED_MEDRA_GPU7_RECOVERY1',branch=cfg['branch'],remote_guard=started,remote_worker=health['worker'],local_guard_pid=child.pid,engineering_source_commit=commit,last_update=time.time(),recovery1=receipt)
    write(cfg['state'],state)
    print(json.dumps(receipt))


if __name__=='__main__':main()
