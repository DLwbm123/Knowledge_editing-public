"""Create a fresh phase; historical ledgers and assets remain read-only."""
import json,os,subprocess,hashlib,shutil
from pathlib import Path
from datetime import datetime,timezone,timedelta
BASE=Path('/data/bmw/Knowledge_editing');ROOT=BASE/'outputs/decomp-24h-20260927/run';OLD=BASE/'outputs/textjoint-r4-20260926/run'
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
 legacy_prefix=os.environ['LEGACY_STORAGE_PREFIX'].rstrip('/')+'/'
 assert legacy_prefix.startswith('/') and legacy_prefix!='/'
 assert not ROOT.exists(),'Do not reset a phase'
 assert shutil.disk_usage(BASE).free>32*1024**3
 ROOT.mkdir(parents=True);now=datetime.now(timezone.utc)
 bindings=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.free,driver_version','--format=csv,noheader,nounits'],text=True)
 rows=[x.split(', ') for x in bindings.splitlines()];gpus=[dict(index=int(x[0]),uuid=x[1],name=x[2],total_MiB=int(x[3]),free_MiB=int(x[4]),driver=x[5]) for x in rows if int(x[0]) in [5,6,7]]
 assert len(gpus)==3 and all(x['free_MiB']>30000 for x in gpus)
 history=json.loads((OLD/'RESOURCE_LEDGER.json').read_text());assert all(s.get('ended_epoch') for s in history['gpu_sessions'])
 write(ROOT/'GPU_BINDINGS.json',dict(devices=gpus,topology=subprocess.check_output(['nvidia-smi','topo','-m'],text=True)))
 write(ROOT/'EXPERIMENT_LOCK.json',dict(review_commit='7dbf60f9cedb98d4ac52269567804f2d29b2aabc',first_started_at=now.isoformat(),no_new_training_after=(now+timedelta(hours=20)).isoformat(),no_large_generation_after=(now+timedelta(hours=22)).isoformat(),deadline_at=(now+timedelta(hours=24)).isoformat(),gpu_seconds_limit=72*3600,judge_lifetime_limit=6000,seeds=[20260927,20260928,20260929],physical_gpus=[5,6,7],methods=['M0','M1','M2','M3','M4','M5','M6'],core=['M1','M3','M4','M5'],layer='model.layers.30.mlp.down_proj',rank=4,dtype='float16',native_max_steps=200,native_check_every=20,native_diagnostic_generation_cap=128,A2_steps=80,W0_steps=320,continuation_steps=80,formal_generation_cap=1024,route='R0',panel='EXPOSED_REGRESSION',stage_diagnostics_orders=sorted(range(1,25),key=lambda i:hashlib.sha256(str(i).encode()).hexdigest())[:4],R5_evidence='No R5 execution artifacts found in migrated scope; not assumed successful'))
 write(ROOT/'RESOURCE_LEDGER.json',dict(historical_gpu_seconds=history['gpu_seconds_used'],current_gpu_seconds=0,gpu_sessions=[],reservations=[],historical_judge_attempts=history['judge_submission_attempt_items'],current_judge_attempts=0,judge_attempts=[]))
 write(ROOT/'STORAGE_POLICY.json',dict(hard_bytes=32*1024**3,soft_bytes=24*1024**3,checkpoint_bytes=8*1024**3,teacher_bytes=8*1024**3,final_model_bytes=2*1024**3,final_total_bytes=4*1024**3,min_free_bytes=20*1024**3,environment_root='/data/bmw/envs/decomp-20260927',environment_reserved_bytes=10*1024**3,protected_roots=[str(OLD),str(BASE/'outputs/textjoint-r2-20260925/run'),str(BASE/'outputs/textjoint-r3-20260926/run'),'/data/bmw/DataP','/data/bmw/hugging_cache']))
 for n in ['source','source_patch','models']:(ROOT/n).symlink_to(OLD/n,target_is_directory=True)
 for n in ['worker_v3.py','router_r3.py','metrics.py','metrics_r2.py','report_r3.py']:shutil.copyfile(OLD/n,ROOT/n)
 for n in ['private/base','private/keys','private/teachers','private/black','private/judge/pending','private/judge/scores','public','logs','tmp','checkpoints','adapters','jobs']:(ROOT/n).mkdir(parents=True,exist_ok=True)
 def mapped(x):
  if isinstance(x,dict):return {k:mapped(v) for k,v in x.items()}
  if isinstance(x,list):return [mapped(v) for v in x]
  return x.replace(legacy_prefix,'/data/bmw/') if isinstance(x,str) else x
 for n in ['TASKS_R2_LOCKED.json','BASE_MASKS.json','PRESSURE_VALIDATION_FROZEN.json','PRESSURE_TEST_FROZEN.json','AUXILIARY_POOL.json']:
  write(ROOT/'private'/n,mapped(json.loads((OLD/'private'/n).read_text())))
 write(ROOT/'CONFIG_LOCK.json',dict(gpu_uuid=gpus[0]['uuid']))
 write(ROOT/'RUN_STATUS.json',dict(status='PREFLIGHT',phase='ENVIRONMENT_AND_GUARDS',epoch=now.timestamp()))
 print(ROOT,now.isoformat())
if __name__=='__main__':main()
