"""One guarded recovery of the exact legacy-field error before any training gradient."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from deploy import W,HERE,R,start_consumers


def main():
    local=W/'.local_run'
    rows=json.loads((local/'SCORERS.json').read_text())['workers']+[json.loads((local/'FINITE_TAIL.json').read_text())]
    for row in rows:
        current=subprocess.run(['ps','-ww','-p',str(row['pid']),'-o','lstart=,args='],capture_output=True,text=True).stdout.strip()
        assert not (current.split()[:5]==row['identity'].split()[:5] and current.split()[-2:]==['-u',row['entry']]), 'Existing owned consumer still live'
    files={name:(HERE/name).read_text() for name in ('contract.py','run146.py','report146.py')}
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip()
    code='''import json,os,sys,time,shutil,tempfile,subprocess
from pathlib import Path
d=json.loads(sys.stdin.readline());r=Path(d['run']);a=r/'private/engineering_base_schema1';assert not a.exists()
read=lambda p:json.loads(p.read_text())
def live(s):
 p=Path('/proc',str(s['pid']),'stat')
 return p.exists() and p.read_text().split()[21]==s['start_ticks'] and p.read_text().split()[2]!='Z'
assert not live(read(r/'private/START_CONTROLLER.json')) and not live(read(r/'private/START_gpu.json'))
assert read(r/'public/BASELINE_REUSE_ADMISSION.json')['status']=='PASS'
assert not list((r/'private/steps').glob('*/*/*.json')) and not list((r/'private/active').glob('*.pt'))
failures=list((r/'private/failures').glob('*.json'));assert len(failures)==2
gpu=next(read(p) for p in failures if p.name.startswith('gpu_'))
assert 'Native Base identity differs' in gpu['traceback'] and gpu['counts']['backwards']==gpu['counts']['committed_updates']==0 and gpu['counts']['generation_attempts']==1
tasks=read(r/'private/BENCHMARK146_QUEUE.json')['tasks'];b=read(r/'private/EVAL_BINDINGS.json')[tasks[0]['native']['opaque_Base_id']]
probe=read(r/'private/mechanical/1.json');assert probe['R0']==dict(raw_answer=b['output']['model_answer_raw'],raw_token_ids=b['output']['raw_generated_token_ids'])
assert probe['binding']['judge_input']==b and b['output']['error'] is None
ledger=read(r/'RESOURCE_LEDGER.json');assert ledger['Judge_attempts']==0 and all(x.get('ended_epoch') for x in ledger['gpu_sessions'])
a.mkdir();(a/'failures').mkdir()
for p in failures:p.rename(a/'failures'/p.name)
for f in ['private/START_CONTROLLER.json','private/START_gpu.json','private/GPU_SOURCE_VERSION.json','logs/controller.log','logs/gpu.log']:
 p=r/f;shutil.copy2(p,a/(p.name+'.original'))
 if f.startswith('logs/'):p.rename(a/p.name)
for name,source in d['files'].items():
 p=r/'private/tools'/name;shutil.copy2(p,a/name);p.write_text(source)
sys.path.insert(0,str(r/'private/tools'));import contract
assert contract.base_completion(b['output'])==probe['R0']
for p in (r/'private/judge_purew146').glob('worker*/SCORER_STOP.json'):p.rename(a/(p.parent.name+'_SCORER_STOP.json'))
env=read(r/'private/LAUNCH_ENV.json');env.update(ACTION='controller',TASK_FILE=str(r/'private/tools/run146.py'));os.environ.update(env)
version=read(r/'private/GPU_SOURCE_VERSION.json');version['commit']=d['commit'];version['engineering_recovery']='base_completion_fields';(r/'private/GPU_SOURCE_VERSION.json').write_text(json.dumps(version))
fd,entry=tempfile.mkstemp(prefix='e.',suffix='.py',dir=env['TMPDIR']);os.close(fd);Path(entry).write_text("import os,runpy\\nrunpy.run_path(os.environ['TASK_FILE'],run_name='__main__')\\n")
child=subprocess.Popen([env['TRAIN_PYTHON'],'-u',entry],env=env,stdout=(r/'logs/controller.log').open('ab'),stderr=subprocess.STDOUT,start_new_session=True)
s=dict(pid=child.pid,start_ticks=Path(f'/proc/{child.pid}/stat').read_text().split()[21],entry=entry,epoch=time.time())
(r/'private/START_CONTROLLER.json').write_text(json.dumps(s));(r/'private/ENGINEERING_BASE_SCHEMA_RECOVERY.json').write_text(json.dumps(dict(status='CONTROLLER_RECOVERED',start=s,commit=d['commit'],frozen_protocol_unchanged=True,Base_probe_reused=True,formal_updates_before=0,Judge_before=0,old_failure_and_cost_preserved=True)))
print(json.dumps(dict(status='RECOVERED_FROM_UNSTARTED_TRAINING',start=s)))
'''
    payload=json.dumps(dict(run=R,files=files,commit=commit))+'\n'
    command="python3 -c 'import sys,json;exec(json.loads(sys.stdin.readline()))'"
    p=subprocess.run(['ssh','pro5000',command],input=json.dumps(code)+'\n'+payload,text=True,capture_output=True,check=True,timeout=90)
    startup=json.loads(p.stdout)
    archive=local/'engineering_base_schema1';archive.mkdir()
    for name in ('SCORERS.json','FINITE_TAIL.json','FINITE_TAIL_FAILURE.json','STARTUP.json'):
        path=local/name
        if path.exists():path.rename(archive/name)
    start_consumers(startup)


if __name__=='__main__':
    main()
