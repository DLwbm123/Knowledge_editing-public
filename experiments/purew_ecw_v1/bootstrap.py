"""Install the explicitly authorized quick validation once, independent of F/S."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
x=json.loads(sys.stdin.readline());p=x['protocol'];R=Path(p['run_root'])
parent=Path('/data/bmw/Knowledge_editing/outputs/purew-fs3-v1-20261014/run')
medical=Path('/data/bmw/Knowledge_editing/outputs/purew-medical-evidence100-v1-20261014/run')


def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')


assert not R.exists(),'One install; never retry an unknown outcome'
assert x['CPU']['status']=='PASS'
u,free=subprocess.check_output(['nvidia-smi','-i','4','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(', ')
assert u==p['GPU_UUID'] and int(free)>=45*1024
mount=json.loads(subprocess.check_output(['findmnt','-J','-T',str(parent)],text=True))['filesystems'][0]
assert mount['target']=='/data' and shutil.disk_usage(parent).free>=p['budgets']['minimum_free_bytes']+p['budgets']['owned_weight_limit_bytes']
R.mkdir(parents=True)
for name in ['private/tools','private/failures','private/operator','private/control/ECW','private/control/ECW_MEDICAL','private/window/ECW','private/window/ECW_MEDICAL','public','logs']:(R/name).mkdir(parents=True,exist_ok=True)
probe=R/'private/.probe';probe.write_text('probe');assert probe.read_text()=='probe';probe.unlink()
for path in (parent/'private/tools').glob('*.py'):shutil.copy2(path,R/'private/tools'/path.name)
for name,source in x['files'].items():compile(source,name,'exec');(R/'private/tools'/name).write_text(source)
for name in ['source','official_llava','cpu_gate']:(R/'private'/name).symlink_to((parent/'private'/name).resolve(),target_is_directory=True)
shutil.copytree(parent/'private/learner_gate',R/'private/learner_gate')
for name in ['BENCHMARK146_QUEUE.json','EVAL_LEDGER.json','METHOD_LOCK.json','EVAL_BINDINGS.json','SOURCE_COMMIT.json']:shutil.copy2(parent/'private'/name,R/'private'/name)
shutil.copy2(medical/'private/operator/source.json',R/'private/operator/source.json')
rows=json.loads((R/'private/operator/source.json').read_text())['rows'];tasks=json.loads((R/'private/BENCHMARK146_QUEUE.json').read_text())['tasks']
reviews=json.loads((medical/'private/operator/review.json').read_text())['rows']
index=next(i for i,d in enumerate(reviews) if d['region_status']=='ELIGIBLE_COARSE_BOX')
write(R/'private/operator/medical.json',dict(row=rows[index],task=tasks[index],review=reviews[index],selection='first source-qualified coarse box; no evaluation/model selection'))
write(R/'private/FROZEN_PROTOCOL.json',p);write(R/'public/QUICK_PROTOCOL.json',p);write(R/'public/CPU_VALIDATION.json',x['CPU'])
now=time.time();budget=p['budgets']
write(R/'RUN_MANIFEST.json',dict(starting_epoch=now,deadline_epoch=now+budget['wall_hours']*3600,GPU_seconds_limit=budget['GPU_process_hours']*3600,
    Judge_limit=0,min_free_bytes=budget['minimum_free_bytes'],owned_weight_limit_bytes=budget['owned_weight_limit_bytes']))
write(R/'STAGE_CAP.json',dict(GPU_seconds_limit=budget['GPU_process_hours']*3600))
write(R/'RESOURCE_LEDGER.json',dict(gpu_seconds_used=0,gpu_sessions=[],Judge_attempts=0,Astra_purew146_batches=[],weights_observed_peak_bytes=0))
write(R/'PLAN_CONFIG.json',dict(hardware=dict(allowed_GPUs=[4],UUIDs={'4':u})))
write(R/'private/CODE_BINDING.json',dict(commit=x['commit'],files=list(x['files']),parent_tools_readonly=True))
tmp=tempfile.mkdtemp(prefix='e.',dir='/data/bmw/tmp');env=dict(os.environ,**json.loads((parent/'private/LAUNCH_ENV.json').read_text()))
env.update(RUN_ROOT=str(R),TMPDIR=tmp,PYTHONPATH=str(R/'private/official_llava')+':'+str(R/'private/source')+':'+str(R/'private/tools'),
    CUDA_VISIBLE_DEVICES='4',M3BENCH_FORMAL_AUTHORIZED_CUDA_VISIBLE_DEVICES='4',M3BENCH_FORMAL_ALLOWED_CUDA_VISIBLE_DEVICES='4',M3BENCH_FORMAL_EXPECTED_GPU_UUID=u,
    ACTION='controller',ARM='ECW',TASK_FILE=str(R/'private/tools/controller.py'))
write(R/'private/LAUNCH_ENV.json',env)
fd,entry=tempfile.mkstemp(prefix='e.',suffix='.py',dir=tmp);os.close(fd);Path(entry).write_text("import os,runpy\nrunpy.run_path(os.environ['TASK_FILE'],run_name='__main__')\n")
child=subprocess.Popen([env['TRAIN_PYTHON'],'-u',entry],env=env,stdout=(R/'logs/controller.log').open('ab'),stderr=subprocess.STDOUT,start_new_session=True)
argv=subprocess.check_output(['ps','-ww','-p',str(child.pid),'-o','args='],text=True).strip()
assert not any(s in argv for s in ['wangbomin','Knowledge_editing','purew','ecw'])
start=dict(pid=child.pid,entry=entry,argv=argv,start_ticks=Path('/proc').joinpath(str(child.pid),'stat').read_text().rsplit(')',1)[1].split()[19],epoch=now,
    GPU_UUID=u,physical_GPU=4,run_root=str(R),commit=x['commit'],new_installation=True,full_ECW_authorized=False)
write(R/'private/START_CONTROLLER.json',start);print(json.dumps(start))
