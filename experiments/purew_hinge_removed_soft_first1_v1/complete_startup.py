"""Complete restart1 after its proven pre-model controller import failure, once."""
import json,os,subprocess,time
from pathlib import Path
W=Path(__file__).resolve().parents[2];H=Path(__file__).resolve().parent
STATE=W/'.local_run/hinge_removed_soft_first1_v1_20261016/restart1'
code=r'''
import json,os,sys,subprocess,shutil,time
from pathlib import Path
x=json.loads(sys.stdin.readline());R=Path('/data/bmw/Knowledge_editing/outputs/purew-hinge-removed-soft-first1-v1-20261016/run')
def read(p):return json.loads(p.read_text())
def write(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
archive=R/'private/restart1';receipt=read(R/'public/RESTART1_RECEIPT.json');start=read(R/'private/START_controller_restart1.json')
assert read(R/'RUN_MANIFEST.json')==receipt['original_manifest']
assert read(R/'RESOURCE_LEDGER.json')==read(archive/'RESOURCE_LEDGER_BEFORE.json')
assert not (R/'private/START_learn_HR_1_restart1.json').exists()
assert not list((R/'private/steps/HR').glob('*/*.json')) and not list((R/'private/failures').glob('*.json'))
stat=Path('/proc')/str(start['pid'])/'stat'
assert not stat.exists() or stat.read_text().rsplit(')',1)[1].split()[19]!=start['start_ticks']
log=(R/'logs/controller_restart1.log').read_text()
assert "AttributeError: module 'controller' has no attribute 'c'" in log
assert x['cpu']['status']=='PASS' and x['cpu']['deployed_source_first_import_entry']
env=read(R/'private/LAUNCH_ENV.json');os.environ.update(env);sys.path[:0]=[str(R/'private/tools'),str(R/'private/source')]
import common as c
c.budget()
with (archive/'CONTROLLER_IMPORT_REPAIR.json').open('x') as f:json.dump(dict(epoch=time.time(),old_controller=start,old_commit=receipt['restart_source_commit'],new_commit=x['commit'],no_new_GPU_lease=True,no_formal_stage_launched=True,experiment_restart_number=1),f)
shutil.copy2(R/'private/tools/restart_controller.py',archive/'restart_controller_import_failure.py')
(R/'private/tools/restart_controller.py').write_text(x['source'])
write(R/'public/CPU_VALIDATION.json',x['cpu'])
receipt.update(controller_import_failure=dict(epoch=start['epoch'],old_commit=receipt['restart_source_commit'],new_GPU_seconds=0,new_formal_slots=0),restart_source_commit=x['commit'])
write(R/'public/RESTART1_RECEIPT.json',receipt)
write(R/'private/GPU_SOURCE_VERSION.json',dict(commit=x['commit'],original_commit=receipt['prior_source']['commit'],restart_number=1))
env.update(TMPDIR=str(Path(start['entry']).parent),TASK_FILE=str(R/'private/tools/restart_controller.py'),ACTION='controller_restart1',ARM='CONTROLLER')
child=subprocess.Popen([env['TRAIN_PYTHON'],'-u',start['entry']],env=env,stdout=(R/'logs/controller_restart1.log').open('ab'),stderr=subprocess.STDOUT,start_new_session=True)
argv=subprocess.check_output(['ps','-ww','-p',str(child.pid),'-o','args='],text=True).strip();assert not any(s in argv for s in ('wangbomin','Knowledge_editing','purew'))
start.update(pid=child.pid,argv=argv,start_ticks=Path('/proc').joinpath(str(child.pid),'stat').read_text().rsplit(')',1)[1].split()[19],epoch=time.time(),commit=x['commit'],controller_import_repaired=True)
write(R/'private/START_controller_restart1_importfix.json',start);print(json.dumps(start))
'''
assert not subprocess.check_output(['git','status','--porcelain'],cwd=W,text=True).strip()
x=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip(),source=(H/'restart_controller.py').read_text(),cpu=json.loads((W/'reports/purew_hinge_removed_soft_first1_v1_20261016/PRODUCTION_CPU_VALIDATION.json').read_text()))
with (STATE/'COMPLETE_STARTUP_ATTEMPT.json').open('x') as f:json.dump(dict(epoch=time.time(),commit=x['commit']),f)
p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','pro5000',"/data/bmw/envs/v0/bin/python -c 'import sys,json;exec(json.loads(sys.stdin.readline()))'"],input=json.dumps(code)+'\n'+json.dumps(x)+'\n',text=True,capture_output=True,timeout=120)
(STATE/'COMPLETE_STARTUP_RESPONSE.json').write_text(json.dumps(dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)))
assert p.returncode==0,p.stderr
print(p.stdout)
