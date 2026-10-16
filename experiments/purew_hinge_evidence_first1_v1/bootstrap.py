"""Once-only installation on the user-authorized pinned physical GPU."""
import json,os,shutil,subprocess,sys,tempfile,time
from pathlib import Path
x=json.loads(sys.stdin.readline());p=x['protocol']
# The guard runs before any filesystem mutation, GPU query or process launch.
config={};exec(compile(x['files']['configuration.py'],'configuration.py','exec'),config)
config['require_execution'](p,x['authorization'],x['hardware'])
R=Path(config['RUN_ROOT']);parent=Path(config['SCOPE_ROOT'])
def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
assert not R.exists(),'One installation; inspect unknown outcomes instead of retrying'
assert x['CPU']['status']=='PASS' and x['CPU']['protocol_version']==p['version']
compatibility=config['check_history'](p)
gpu=x['hardware'];index=gpu['physical_GPU']
u,free=subprocess.check_output(['nvidia-smi','-i',str(index),'--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(', ')
assert u==gpu['GPU_UUID'] and int(free)>=46080
mount=json.loads(subprocess.check_output(['findmnt','-J','-T',str(parent)],text=True))['filesystems'][0]
effective=config['effective_protocol'](p);budget=effective['budgets']
assert mount['target']=='/data' and shutil.disk_usage(parent).free>=budget['minimum_free_bytes']+budget['owned_matrix_limit_bytes']
memory={a.split(':')[0]:int(a.split(':')[1].split()[0])*1024 for a in Path('/proc/meminfo').read_text().splitlines() if a.startswith(('MemTotal:','MemAvailable:'))}
assert memory['MemAvailable']>=96*1024**3
now=time.time();R.mkdir(parents=True)
for arm in ('HE',):
    for kind in ('control','window','active','final','units'):(R/'private'/kind/arm).mkdir(parents=True,exist_ok=True)
for name in ['private/tools','private/failures','private/operator','public','logs']:(R/name).mkdir(parents=True,exist_ok=True)
probe=R/'private/.probe';probe.write_text('admission');assert probe.read_text()=='admission';probe.unlink()
for path in (parent/'private/tools').glob('*.py'):shutil.copy2(path,R/'private/tools'/path.name)
for name,text in x['files'].items():compile(text,name,'exec');(R/'private/tools'/name).write_text(text)
shutil.copy2(parent/'private/tools/historical_binding.py',R/'private/tools/historical_binding.py')
for name in ['source','official_llava','cpu_gate','derived_inputs','legacy_stage17']:(R/'private'/name).symlink_to((parent/'private'/name).resolve(),target_is_directory=True)
shutil.copytree(parent/'private/learner_gate',R/'private/learner_gate')
for name in ['BENCHMARK146_QUEUE.json','EVAL_LEDGER.json','METHOD_LOCK.json','EVAL_BINDINGS.json','SOURCE_COMMIT.json']:shutil.copy2(parent/'private'/name,R/'private'/name)
for name in ('source.json','review.json'):shutil.copy2(parent/'private/operator'/name,R/'private/operator'/name)
(R/'private/calibration').mkdir()
shutil.copy2(parent/'private/calibration/Base_plain.pt',R/'private/calibration/Base_plain.pt')
assert (R/'private/calibration/Base_plain.pt').stat().st_size==8455719655
write(R/'public/BASE_PARAMETER_REUSE.json',dict(status='PASS',source='SH00 immutable plain Base16',bytes=8455719655,binding=p['shared']['Base_binding'],new_calibration_generations=0,no_case_or_answer_replay=True))

write(R/'private/FROZEN_PROTOCOL.json',effective);write(R/'public/PROTOCOL.json',p);write(R/'public/EFFECTIVE_RUNTIME_PROTOCOL.json',effective)
for name,value in [('EXECUTION_AUTHORIZATION.json',x['authorization']),('HARDWARE_AUTHORIZATION.json',gpu),('CPU_VALIDATION.json',x['CPU'])]:write(R/'public'/name,value)
write(R/'public/HISTORICAL_COMPATIBILITY.json',compatibility)
write(R/'private/GPU_SOURCE_VERSION.json',dict(commit=x['commit']))
write(R/'RUN_MANIFEST.json',dict(starting_epoch=now,deadline_epoch=now+budget['wall_hours']*3600,GPU_seconds_limit=budget['GPU_process_hours']*3600,
    Judge_limit=budget['new_Judge'],min_free_bytes=budget['minimum_free_bytes'],owned_weight_limit_bytes=budget['owned_matrix_limit_bytes']))
write(R/'STAGE_CAP.json',dict(GPU_seconds_limit=budget['GPU_process_hours']*3600))
write(R/'RESOURCE_LEDGER.json',dict(gpu_seconds_used=0,gpu_sessions=[],Judge_attempts=0,Astra_purew146_batches=[],weights_observed_peak_bytes=0))
write(R/'PLAN_CONFIG.json',dict(hardware=dict(allowed_GPUs=[index],UUIDs={str(index):u})))
write(R/'public/STORAGE_ADMISSION.json',dict(status='PASS',mount=mount,disk_free_bytes=shutil.disk_usage(R).free,host=memory,read_write_probe=True,
    largest_factor_bytes=budget['largest_factor_FP64_bytes'],full_factor_live_measured=False,CPU_streamed_FP64=True))
tmp=tempfile.mkdtemp(prefix='e.',dir='/data/bmw/tmp');env=dict(os.environ,**json.loads((parent/'private/LAUNCH_ENV.json').read_text()))
env.update(RUN_ROOT=str(R),TMPDIR=tmp,PYTHONPATH=str(R/'private/official_llava')+':'+str(R/'private/source')+':'+str(R/'private/tools'),
    CUDA_VISIBLE_DEVICES=str(index),M3BENCH_FORMAL_AUTHORIZED_CUDA_VISIBLE_DEVICES=str(index),M3BENCH_FORMAL_ALLOWED_CUDA_VISIBLE_DEVICES=str(index),
    M3BENCH_FORMAL_EXPECTED_GPU_UUID=u,ACTION='controller',ARM='CONTROLLER',TASK_FILE=str(R/'private/tools/controller.py'),OMP_NUM_THEEADS='8',MKL_NUM_THEEADS='8')
write(R/'private/LAUNCH_ENV.json',env)
os.environ.update(env);sys.path.insert(0,str(R/'private/source'));sys.path.insert(0,str(R/'private/tools'))
import queue_revision as queue
queue.q.ROOT.mkdir(parents=True,exist_ok=True)
queue.q.write(queue.q.ROOT/'EPOCH_MANIFEST.json',dict(epoch=p['version'],model='gpt-6-astra',reasoning_effort='medium',protocol=queue.q.PROTOCOL,prompt=queue.q.PROMPT,attempts_per_payload=1,workers=4,batch_size=50))
db=queue.q.connect();queue.ingest(db);db.close()
fd,entry=tempfile.mkstemp(prefix='e.',suffix='.py',dir=tmp);os.close(fd);Path(entry).write_text("import os,runpy\nrunpy.run_path(os.environ['TASK_FILE'],run_name='__main__')\n")
child=subprocess.Popen([env['TRAIN_PYTHON'],'-u',entry],env=env,stdout=(R/'logs/controller.log').open('ab'),stderr=subprocess.STDOUT,start_new_session=True)
argv=subprocess.check_output(['ps','-ww','-p',str(child.pid),'-o','args='],text=True).strip();assert not any(s in argv for s in ['wangbomin','Knowledge_editing','purew','ecw'])
start=dict(pid=child.pid,entry=entry,argv=argv,start_ticks=Path('/proc').joinpath(str(child.pid),'stat').read_text().rsplit(')',1)[1].split()[19],epoch=now,
    GPU_UUID=u,physical_GPU=index,run_root=str(R),commit=x['commit'],activation_epoch=now,deadline_epoch=now+budget['wall_hours']*3600,
    current_free_MiB=int(free),initial_state='GPU_ADMISSION_PENDING',new_installation=True)
write(R/'private/START_CONTROLLER.json',start);print(json.dumps(start))
