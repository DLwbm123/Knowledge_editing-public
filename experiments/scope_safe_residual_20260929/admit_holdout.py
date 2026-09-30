import os,sys,json,time,subprocess,fcntl,shutil
from pathlib import Path
r=Path('/data/bmw/Knowledge_editing/outputs/scope-safe-residual-20260929/run');os.environ['RUN_ROOT']=str(r);sys.path.insert(0,str(r))
from resources import read,write
assert not (r/'STOP').exists() and not (r/'JUDGE_FAILURE.json').exists() and not (r/'HOLDOUT_QUEUE.json').exists()
assert all(j['status']=='COMPLETE' for j in read(r/'QUEUE.json'))
l=read(r/'RESOURCE_LEDGER.json');assert not any(not s.get('ended_epoch') for s in l['gpu_sessions']);assert all(a['status'] in ['FORMAT_VALID','FAILED_NO_RETRY'] for a in l['judge_attempts'])
assert shutil.disk_usage(r).free>=20*1024**3
probe=r/'auxiliary/write_probe';probe.parent.mkdir(exist_ok=True);probe.write_text('probe');assert probe.read_text()=='probe';probe.unlink()
rows=read(r/'private/LOCALITY_STRESS_HOLDOUT.json')['rows'];assert len(rows)==47 and len({x['source_group'] for x in rows})==24
assert all(Path(x['image_path']).exists() for x in rows)
fix=r/'fix/pre_holdout';fix.mkdir(exist_ok=True,parents=True)
for n in ['RESOURCE_LEDGER.json','RUN_STATUS.json','SCORER_DONE']:
 if (r/n).exists():shutil.copy2(r/n,fix/n)
queue=[dict(id=panel+'-'+method,method=method,orders=orders,status='PENDING') for panel,orders,methods in [('DEV24',list(range(1,25)),['E_orig','A0','AK','AU','AH','AHS_01']),('REG24',list(range(25,49)),['E_orig','A0','AH','AHS_01'])] for method in methods]
write(r/'HOLDOUT_PROTOCOL.json',dict(status='FROZEN_BEFORE_HOLDOUT_RESULTS',candidates=47,sources=24,all_candidates_evaluated=True,qualification='One common Base-correct mask; report unconditional accuracy too',banks=queue,prefixes=[4,8,12,24],new_training=0,retuning=False,attempts_baseline=l['judge_submission_attempt_items'],GPU_seconds_baseline=l['current_gpu_seconds'],judge_items_upper_bound=47*(1+4*len(queue)),P5_not_admitted_unchanged=True))
write(r/'HOLDOUT_QUEUE.json',queue);write(r/'RUN_STATUS.json',dict(status='RUNNING',phase='FROZEN_HOLDOUT_EVALUATION',training_complete=True))
(r/'SCORER_DONE').unlink(missing_ok=True)
Path('/tmp/s12.py').write_text('import os,sys,runpy\nsys.path.insert(0,os.environ["RUN_ROOT"])\nrunpy.run_module("holdout_worker",run_name="__main__")\n')
active=[]
for g in read(r/'GPU_BINDINGS.json')['devices']:
 free=int(subprocess.check_output(['nvidia-smi','-i',str(g['index']),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip());assert free>=30000
 env=os.environ.copy();env.update(RUN_ROOT=str(r),CUDA_VISIBLE_DEVICES=g['uuid'],PINNED_GPU_UUID=g['uuid'],PHYSICAL_GPU=str(g['index']),PYTHONUNBUFFERED='1',TMPDIR='/data/bmw/tmp',HF_HOME='/data/bmw/cache/huggingface',TORCH_HOME='/data/bmw/cache/torch',XDG_CACHE_HOME='/data/bmw/cache',CUDA_CACHE_PATH='/data/bmw/cache/cuda')
 with (r/'logs'/f'holdout{g["index"]}.log').open('ab') as f:p=subprocess.Popen(['/data/bmw/envs/v0/bin/python','/tmp/s12.py'],env=env,cwd=r,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 active.append(dict(pid=p.pid,index=g['index'],entry='/tmp/s12.py'))
write(r/'HOLDOUT_PROCESSES.json',active);print(json.dumps(active))
