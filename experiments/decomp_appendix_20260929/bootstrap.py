import subprocess
script=r'''
import os,sys,json,shutil,time,hashlib,subprocess
from pathlib import Path
from datetime import datetime,timezone,timedelta
p=Path('/data/bmw/Knowledge_editing/outputs/decomp-24h-20260927/run');r=Path('/data/bmw/Knowledge_editing/outputs/appendix-20260929/run')
assert not r.exists(),'Never reset an existing extension'
read=lambda x:json.loads(x.read_text())
assert not Path('/proc',str(read(p/'ORCHESTRATOR_PID.json')['pid'])).exists()
assert all(j['status']=='COMPLETE' for j in read(p/'QUEUE.json'))
assert (p/'RETAINED_CHECKPOINT_MANIFEST.json').exists()
h=read(p/'RESOURCE_LEDGER.json');assert all(s.get('ended_epoch') for s in h['gpu_sessions'])
assert read(p/'public/SCORING_COVERAGE.json')['IN_FLIGHT']==0
assert shutil.disk_usage(p).free>20*1024**3
os.environ['RUN_ROOT']=str(p);sys.path[:0]=[str(p),str(p/'source_patch'),str(p/'source')]
import torch
from bindings import expected,check_payload
from storage import digest_file
T=read(p/'private/TASKS_R2_LOCKED.json')['tasks'];imports={}
for t in T:
 if not 25<=t['order']<=48:continue
 rel=f'adapters/s20260927/M0/e{t["order"]:03d}.pt';x=torch.load(p/rel,map_location='cpu',weights_only=True);want=expected(t,20260927,'CP','M0','continuation',80);check_payload(x,rel,want)
 imports[rel]=dict(path=rel,sha256=digest_file(p/rel),binding=x['binding'],parent_expected=want)
r.mkdir(parents=True);now=datetime.now(timezone.utc)
def put(n,x):
 f=r/n;f.parent.mkdir(parents=True,exist_ok=True);f.write_text(json.dumps(x,ensure_ascii=False,indent=2))
for f in p.glob('*.py'):shutil.copy2(f,r/f.name)
for n in ['source','source_patch','models']:(r/n).symlink_to((p/n).resolve(),target_is_directory=True)
for n in ['private/base','private/keys','private/teachers','private/black','private/teacher_quality','private/judge/scores']:
 if (p/n).exists():shutil.copytree(p/n,r/n)
for n in ['private/judge/pending','private/judge/evidence','private/judge/success_intents','private/generations','private/curves','private/diagnostics','private/kd','public','logs','tmp','checkpoints','adapters','jobs','fix/pr5_v1']:(r/n).mkdir(parents=True,exist_ok=True)
for f in (p/'private').glob('*.json'):shutil.copy2(f,r/'private'/f.name)
shutil.copy2(p/'private/judge/JUDGE_MISSING_LOCK.json',r/'private/judge/JUDGE_MISSING_LOCK.json')
for n in ['SCIENCE_LOCK.json','CONFIG_LOCK.json','VERSION_LOCK.json','SUPPORT_REPAIR.json','STORAGE_POLICY.json','USER_LIMIT_WAIVER.json']:
 if (p/n).exists():shutil.copy2(p/n,r/n)
# Only verified immutable parent teachers are imported. No parent ledger is mutated.
for rel in imports:
 dest=r/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p/rel,dest)
put('IMPORTED_BINDINGS.json',imports)
ledger=dict(artifacts={},peak_bytes=0,deleted_bytes=0,deleted_count=0)
for rel,v in imports.items():ledger['artifacts'][rel]=dict(real_path=str(r/rel),owner_run=str(r),bytes=(r/rel).stat().st_size,kind='checkpoint',consumers=[],readers=0,pin=True,status='READY',hash=v['sha256'],verified_at=time.time())
put('STORAGE_LEDGER.json',ledger)
b=read(p/'EXPERIMENT_LOCK.json');b.update(first_started_at=now.isoformat(),no_new_training_after=(now+timedelta(hours=18)).isoformat(),no_large_generation_after=(now+timedelta(hours=21)).isoformat(),deadline_at=(now+timedelta(hours=24)).isoformat(),seeds=[20260927],methods=['M6','M7'],core=['M6','M7'],parent_public_commit='77641da01c853235c540066f91323396c507e9cd',stage_diagnostics_orders=[]);put('EXPERIMENT_LOCK.json',b)
m=read(p/'RUN_MANIFEST.json');m.update(first_started_at=now.isoformat(),deadline_at=b['deadline_at'],no_new_generation_after=b['no_large_generation_after']);put('RUN_MANIFEST.json',m)
rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True).splitlines();devices=[]
for row in rows:
 i,u,free=row.split(', ')
 if int(i) in [5,6,7]:devices.append(dict(index=int(i),uuid=u,free_MiB=int(free)))
assert len(devices)==3;put('GPU_BINDINGS.json',dict(devices=devices))
d=dict(historical_gpu_seconds=h['historical_gpu_seconds']+h['current_gpu_seconds'],parent_gpu_seconds=h['current_gpu_seconds'],current_gpu_seconds=0,lifetime_gpu_seconds=h['historical_gpu_seconds']+h['current_gpu_seconds'],gpu_sessions=[],reservations=[],historical_judge_attempts=h['judge_submission_attempt_items'],current_judge_attempts=0,judge_submission_attempt_items=h['judge_submission_attempt_items'],judge_submission_attempt_items_limit=6000,judge_attempts=h['judge_attempts'])
assert 6000-d['judge_submission_attempt_items']>=1440
# Sole extension coordinator; parent scorer/controller are finished. Reserve complete REG24 pair.
d['reservations']=[dict(block='E1_REG24',judge_items=1440,scope='2 methods x (285 single + 150 prefix12 + 285 prefix24); excludes any unknown-answer dedup discount',status='RESERVED',epoch=now.timestamp())];put('RESOURCE_LEDGER.json',d)
put('PARENT_HANDOFF_AUDIT.json',dict(parent=str(p),parent_public_commit=b['parent_public_commit'],parent_complete_jobs=len(read(p/'QUEUE.json')),parent_role_audit=read(p/'public/ROLE_AUDIT_STATUS.json'),parent_scoring=read(p/'public/SCORING_COVERAGE.json'),W0_remaining=0,retained_adapters=len(list((p/'adapters').rglob('*.pt'))),parent_controller_released=True,parent_gpu_sessions_closed=True,teacher_bindings_verified=len(imports),reuse='CP-M0 teacher and immutable Base/score caches only; M6/M7 rebuilt from common new Direct-W0',epoch=now.timestamp()))
put('INCREMENTAL_MATRIX_LOCK.json',dict(priority=['E1_REG24','E2_REG24','E1_E2_DEV24','E3_REG24','E4_REG24'],active='E1_REG24',seed=20260927,orders=list(range(25,49)),methods=['M6','M7'],canary_orders=[25,26],common_W0=True,rebuild_reason='Parent W0 deleted; retrain M6 paired with M7, do not claim old M6 has same W0',teacher='verified parent CP-M0',lambda_U=.01,lambda_D=.10,steps=80,judgement_upper_bound=1440,subsequent_blocks='NOT_ADMITTED until full paired scoring reservation',selection='Fixed protocol order; no outcome-based arm choice'))
put('QUEUE.json',[dict(id=f'E1-canary-LR-{o}',mode='train',kind='LR',methods=['M6','M7'],seed=20260927,order=o,requires=[],status='PENDING') for o in [25,26]])
put('ACTIVE_PROCESSES.json',[]);put('PILOT_PROCESSES.json',[]);put('RUN_STATUS.json',dict(status='PREFLIGHT',phase='E1_CANARY',epoch=now.timestamp()))
print(json.dumps(dict(root=str(r),teachers=len(imports),remaining_judge=6000-d['judge_submission_attempt_items'],reserved=1440,started_at=now.isoformat())))
'''
subprocess.run(['ssh','pro5000','/data/bmw/envs/v0/bin/python -'],input=script,text=True,check=True)
