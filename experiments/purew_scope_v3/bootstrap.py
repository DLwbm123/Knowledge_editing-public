"""Remote installer: fixed host storage, inherited costs, neutral background entry."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

payload=json.loads(sys.stdin.readline())
base_parent=Path('/data/bmw/Knowledge_editing/outputs/purew-scope-v1-20261009/run')
parent=Path('/data/bmw/Knowledge_editing/outputs/purew-scope-v2-20261010/run')
root=Path('/data/bmw/Knowledge_editing/outputs/purew-scope-v3-20261010/run')
sys.path.insert(0,str(parent/'private/tools'))
# Use the installed durable journal supplied in the authorized code payload.
namespace={};exec(payload['files']['journal.py'],namespace)
write=namespace['write'];alive=namespace['alive']
read=lambda p:json.loads(p.read_text())
assert read(parent/'private/REPORT_COMPLETE.json')['status'] in ('COMPLETE','INCOMPLETE_SCORING')
assert read(base_parent/'public/RESULTS.json')['status']=='COMPLETE'
assert all(s.get('ended_epoch') for s in read(parent/'RESOURCE_LEDGER.json')['gpu_sessions'])
mount=json.loads(subprocess.check_output(['findmnt','-J','-T',str(parent)],text=True))['filesystems'][0]
assert mount['target']!='/' and Path(str(parent)).is_relative_to('/data/bmw')
assert shutil.disk_usage(parent).free>=payload['protocol']['minimum_free_bytes']
root.mkdir(parents=True,exist_ok=True)
with tempfile.NamedTemporaryFile(dir=root) as probe:
    probe.write(b'admission');probe.flush();os.fsync(probe.fileno());probe.seek(0);assert probe.read()==b'admission'
for d in ('private/tools','logs','public'): (root/d).mkdir(parents=True,exist_ok=True)
with namespace['exclusive'](root/'private/locks/install.lock'):
    for oldfile in (base_parent/'private/tools').glob('*.py'):
        name='base_run.py' if oldfile.name=='run.py' else oldfile.name
        if name in payload['files']:continue
        target=root/'private/tools'/name
        if not target.exists():shutil.copy2(oldfile,target)
    for name,content in payload['files'].items():
        assert '/' not in name and name.endswith('.py')
        path=root/'private/tools'/name
        if path.exists():assert path.read_text()==content,'Installed code differs; inspect before engineering repair'
        else:path.write_text(content)
    for name in ('source','official_llava','cpu_gate','derived_inputs'):
        src=base_parent/'private'/name;dst=root/'private'/name
        assert src.exists()
        if not dst.exists():dst.symlink_to(src.resolve(),target_is_directory=True)
        assert dst.is_symlink() and dst.resolve()==src.resolve()
    for name in ('EVAL_BINDINGS.json','SOURCE_COMMIT.json'):
        dst=root/'private'/name
        if not dst.exists():shutil.copy2(base_parent/'private'/name,dst)
    frozen=root/'private/FROZEN_PROTOCOL.json'
    if frozen.exists():assert read(frozen)==payload['protocol']
    else:write(frozen,payload['protocol'])
    if not (root/'STAGE_CAP.json').exists():
        write(root/'STAGE_CAP.json',read(parent/'STAGE_CAP.json'))
    inherited=read(parent/'RESOURCE_LEDGER.json')
    if not (root/'RESOURCE_LEDGER.json').exists():
        write(root/'private/INHERITED_COST.json',inherited)
        write(root/'private/INHERITED_RUN.json',dict(root=str(parent),read_only=True))
        write(root/'RESOURCE_LEDGER.json',inherited)
        installed=time.time()
        manifest=dict(read(parent/'RUN_MANIFEST.json'),stage='purew_scope_v3',starting_epoch=installed,started_epoch=installed,deadline_epoch=installed+24*3600,
            physical_GPUs=[5],max_training_GPUs=1,max_training_workers=1,hourly_monitor_authorized=False,
            wall_limit_enabled=True,Judge_limit_enabled=True,
            GPU_seconds_limit=10**18,GPU_time_limit_enabled=False,Judge_limit=inherited['Judge_attempts']+1880,
            owned_weight_limit_bytes=2*1024**3,min_free_bytes=8*1024**3)
        write(root/'RUN_MANIFEST.json',manifest)
        config=read(parent/'PLAN_CONFIG.json');config['hardware']['allowed_GPUs']=[5]
        config['hardware']['UUIDs']={'5':payload['protocol']['GPU_UUID']}
        config['hardware']['physical_GPUs']=[5]
        write(root/'PLAN_CONFIG.json',config)
        write(root/'private/GPU_SOURCE_VERSION.json',dict(commit=payload['commit'],epoch=time.time(),source='published five-arm pure-W diagnostic code'))
    env=dict(os.environ,**read(parent/'private/LAUNCH_ENV.json'))
    env.update(RUN_ROOT=str(root),PUREW_PARENT=str(base_parent),PYTHONPATH=str(root/'private/tools'))
    assert Path(env['TRAIN_PYTHON']).is_file() and env['TMPDIR'].startswith('/data/bmw/')
    write(root/'private/LAUNCH_ENV.json',env)
    fd,entry=tempfile.mkstemp(prefix='e.',suffix='.py',dir=env['TMPDIR']);os.close(fd)
    Path(entry).write_text("import os,sys,runpy\nsys.path.insert(0,os.environ['RUN_ROOT']+'/private/tools')\nrunpy.run_path(os.environ['TASK_FILE'],run_name='__main__')\n")
    env.update(ACTION='scope_plan',TASK_FILE=str(root/'private/tools/run.py'))
    subprocess.run([env['TRAIN_PYTHON'],'-u',entry],env=env,check=True,stdout=(root/'logs/plan.log').open('ab'),stderr=subprocess.STDOUT)
    complete=root/'private/CONTROLLER_COMPLETE.json';start=root/'private/CONTROLLER_START.json'
    if complete.exists():result=dict(status='GPU_CHAIN_COMPLETE',receipt=read(complete))
    elif start.exists() and alive(read(start)):result=dict(status='ALREADY_RUNNING',start=read(start))
    else:
        assert time.time()<read(root/'RUN_MANIFEST.json')['deadline_epoch'],'Frozen deadline expired; do not reset'
        observations={}
        for gpu in (5,):
            row=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(', ')
            assert row[0]==read(root/'PLAN_CONFIG.json')['hardware']['UUIDs'][str(gpu)]
            owned=[s for s in read(root/'RESOURCE_LEDGER.json')['gpu_sessions']
                   if not s.get('ended_epoch') and s['gpu_uuid']==row[0]
                   and s['action'] in ('scope_admission','scope_train','scope_eval') and alive(s)]
            # A recovered controller adopts the live stage that already passed GPU admission.
            assert int(row[1])>=60000 or owned,'Wait for sufficient free memory; never stop other jobs'
            observations[str(gpu)]=dict(uuid=row[0],free_MiB=int(row[1]),adopted_PIDs=[s['pid'] for s in owned])
        env['ACTION']='scope_controller'
        child=subprocess.Popen([env['TRAIN_PYTHON'],'-u',entry],env=env,stdout=(root/'logs/controller.log').open('ab'),stderr=subprocess.STDOUT,start_new_session=True)
        ticks=Path(f'/proc/{child.pid}/stat').read_text().rsplit(')',1)[1].split()[19]
        receipt=dict(pid=child.pid,start_ticks=ticks,entry=entry,log=str(root/'logs/controller.log'),epoch=time.time(),GPUs=[5],preflight=observations,mount=mount)
        write(start,receipt);time.sleep(2)
        assert alive(receipt),'Controller stopped immediately; retain failure and logs'
        command=subprocess.check_output(['ps','-ww','-p',str(child.pid),'-o','args='],text=True).strip()
        assert not any(x.lower() in command.lower() for x in ('wangbomin','Knowledge_editing','scope','loki','dow-ke'))
        result=dict(status='STARTED',start=receipt,command=command)
    print(json.dumps(result))
