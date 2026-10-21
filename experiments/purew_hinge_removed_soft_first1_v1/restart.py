"""Explicit once-only recovery of the existing HR run, never a new installation."""
import json,os,subprocess,sys,time
from pathlib import Path
W=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
OUT=W/'reports/purew_hinge_removed_soft_first1_v1_20261016'
STATE=W/'.local_run/hinge_removed_soft_first1_v1_20261016/restart1'
REMOTE=r'''
import json,os,sys,subprocess,shutil,tempfile,time,sqlite3
from pathlib import Path
x=json.loads(sys.stdin.readline());R=Path(x['authorization']['run_root']);a=x['authorization']
def read(p):return json.loads(Path(p).read_text())
def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2)+'\n')
assert a['restart_number']==a['maximum_restarts']==1 and a['authorized']
assert x['cpu']['status']=='PASS' and x['cpu']['formal_learner_20_slots_both_terminal_branches']
manifest=read(R/'RUN_MANIFEST.json');ledger=read(R/'RESOURCE_LEDGER.json')
assert manifest['starting_epoch']==a['original_activation_epoch'] and manifest['deadline_epoch']==a['original_deadline_epoch']
assert time.time()<manifest['deadline_epoch'] and ledger['gpu_seconds_used']<manifest['GPU_seconds_limit']
assert all(s.get('ended_epoch') for s in ledger['gpu_sessions'])
for p in (R/'private').glob('START_*.json'):
    old=read(p);stat=Path('/proc')/str(old['pid'])/'stat'
    assert not stat.exists() or stat.read_text().rsplit(')',1)[1].split()[19]!=old['start_ticks'],'Existing owned process'
assert not list((R/'private/steps/HR').glob('*/*.json'))
assert not (R/'private/active/HR/state.pt').exists() and not (R/'private/final/HR/native.pt').exists()
assert not (R/'private/GENERATION_COMPLETE.json').exists()
assert ledger['Judge_attempts']==0
rows=[read(p) for p in (R/'private/attempts').glob('*.json')]
assert sum(p.get('attempt_slots',0) for p in rows)==0
assert sum(p.get('committed_updates',0) for p in rows)==1
assert sum(p.get('backwards',0) for p in rows)==21
for name in ('BASE_IDENTITY_HR.json','REAL_VLM_ADMISSION_HR.json','REAL_VLM_ADMISSION_ALL.json'):assert read(R/'public'/name)['status']=='PASS'
failures=sorted((R/'private/failures').glob('*.json'))
assert {p.name for p in failures}=={'controller.json','learn_HR_1_3258301.json'}
assert 'UnboundLocalError' in read(R/'private/failures/learn_HR_1_3258301.json')['traceback']
db=sqlite3.connect('file:'+str(R/'private/judge_purew146/queue.sqlite')+'?mode=ro',uri=True)
assert db.execute('SELECT count(*) FROM consumer').fetchone()[0]==0
assert not db.execute("SELECT 1 FROM payload WHERE status IN ('PENDING','RESERVED')").fetchone();db.close()
env=read(R/'private/LAUNCH_ENV.json');gpu=int(env['CUDA_VISIBLE_DEVICES'])
u,free=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(', ')
assert u==read(R/'PLAN_CONFIG.json')['hardware']['UUIDs'][str(gpu)] and int(free)>=46080
mount=json.loads(subprocess.check_output(['findmnt','-J','-T',str(R)],text=True))['filesystems'][0]
assert mount['target']=='/data' and shutil.disk_usage(R).free>=manifest['owned_weight_limit_bytes']+manifest['min_free_bytes']
mem=int(next(s for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')).split()[1])*1024
assert mem>=96*1024**3
os.environ.update(env);sys.path[:0]=[str(R/'private/tools'),str(R/'private/source')]
import common as c
c.budget()
for name,text in x['files'].items():compile(text,name,'exec')
receipt=dict(restart_number=1,formal_slots_before=0,original_manifest=manifest,prior_GPU_seconds=ledger['gpu_seconds_used'],prior_Judge_attempts=0,prior_counts=rows,prior_source=read(R/'private/GPU_SOURCE_VERSION.json'),restart_source_commit=x['commit'],epoch=time.time(),admission_reused=True,free_MiB=int(free),mount=mount['target'])
archive=R/'private/restart1';archive.mkdir() # Exclusive claim: an uncertain launch never repeats.
write(archive/'CLAIM.json',receipt);write(archive/'RESOURCE_LEDGER_BEFORE.json',ledger)
probe=archive/'.probe';probe.write_text('restart1');assert probe.read_text()=='restart1';probe.unlink()
for name in x['files']:shutil.copy2(R/'private/tools'/name,archive/name) if (R/'private/tools'/name).exists() else None
for name in ('REMOVED_BIRTH_HR.json','PROGRESS.json'):
    if (R/'public'/name).exists():shutil.copy2(R/'public'/name,archive/name)
for p in failures:p.rename(archive/p.name)
for name,text in x['files'].items():(R/'private/tools'/name).write_text(text)
write(R/'public/RESTART1_AUTHORIZATION.json',a);write(R/'public/RESTART1_RECEIPT.json',receipt);write(R/'public/CPU_VALIDATION.json',x['cpu'])
write(R/'private/GPU_SOURCE_VERSION.json',dict(commit=x['commit'],original_commit=receipt['prior_source']['commit'],restart_number=1))
assert read(R/'RUN_MANIFEST.json')==manifest and read(R/'RESOURCE_LEDGER.json')==ledger
folder=Path(tempfile.mkdtemp(prefix='e.',dir='/data/bmw/tmp'));entry=folder/'e.py'
entry.write_text("import os,runpy\nrunpy.run_path(os.environ['TASK_FILE'],run_name='__main__')\n")
env.update(TMPDIR=str(folder),TASK_FILE=str(R/'private/tools/restart_controller.py'),ACTION='controller_restart1',ARM='CONTROLLER')
child=subprocess.Popen([env['TRAIN_PYTHON'],'-u',str(entry)],env=env,stdout=(R/'logs/controller_restart1.log').open('ab'),stderr=subprocess.STDOUT,start_new_session=True)
argv=subprocess.check_output(['ps','-ww','-p',str(child.pid),'-o','args='],text=True).strip()
assert not any(s in argv for s in ('wangbomin','Knowledge_editing','purew'))
start=dict(pid=child.pid,entry=str(entry),argv=argv,start_ticks=Path('/proc').joinpath(str(child.pid),'stat').read_text().rsplit(')',1)[1].split()[19],epoch=time.time(),activation_epoch=manifest['starting_epoch'],deadline_epoch=manifest['deadline_epoch'],physical_GPU=gpu,GPU_UUID=u,commit=x['commit'],new_installation=False,restart_number=1,run_root=str(R))
write(R/'private/START_controller_restart1.json',start);print(json.dumps(start))
'''

def main():
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=W,text=True).strip(),'Publish source first'
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip()
    x=dict(commit=commit,authorization=json.loads((OUT/'RESTART1_AUTHORIZATION.json').read_text()),cpu=json.loads((OUT/'PRODUCTION_CPU_VALIDATION.json').read_text()),files={name:(HERE/name).read_text() for name in ('worker.py','restart_controller.py')})
    STATE.mkdir(parents=True,exist_ok=True)
    with (STATE/'ATTEMPT.json').open('x') as f:json.dump(dict(epoch=time.time(),commit=commit),f)
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','pro5000',"/data/bmw/envs/v0/bin/python -c 'import sys,json;exec(json.loads(sys.stdin.readline()))'"],input=json.dumps(REMOTE)+'\n'+json.dumps(x)+'\n',text=True,capture_output=True,timeout=120)
    (STATE/'RESPONSE.json').write_text(json.dumps(dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)))
    assert result.returncode==0,result.stderr
    startup=json.loads(result.stdout)
    os.environ.update(WORKSPACE=str(W),CONTINUATION_STATE=str(STATE))
    from consumers import start_consumers
    start_consumers(startup)
if __name__=='__main__':main()
