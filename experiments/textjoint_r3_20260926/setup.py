"""Create isolated R3 state; R2 is read-only evidence, never a write target."""
from pathlib import Path
import os,json,shutil,time,subprocess
from datetime import datetime,timezone,timedelta
OLD=Path('/remote-home/wangbomin/Knowledge_editing/outputs/textjoint-r2-20260925/run')
ROOT=OLD.parents[1]/'textjoint-r3-20260926/run'
def read(p):return json.loads(p.read_text())
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def copytree(a,b):
 for parent,dirs,files in os.walk(a):
  q=b/Path(parent).relative_to(a);q.mkdir(parents=True,exist_ok=True)
  dirs[:]=[d for d in dirs if d not in ['__pycache__','.git']]
  for f in files:shutil.copyfile(Path(parent)/f,q/f)
def main():
 assert not ROOT.exists(), 'Do not reset an existing campaign'
 print(subprocess.check_output(['findmnt','-T',str(OLD)],text=True))
 assert shutil.disk_usage(OLD).free>20*1024**3
 probe=OLD.parent/'.r3-mount-probe';probe.write_text('r3');assert probe.read_text()=='r3';probe.unlink()
 ROOT.mkdir(parents=True);now=datetime.now(timezone.utc)
 write(ROOT/'RUN_MANIFEST.json',dict(run_id=ROOT.parent.name,review_commit='532b81be5b96848e5ccd9d46e8252d1155fe6aac',first_started_at=now.isoformat(),p0_deadline_at=(now+timedelta(hours=3)).isoformat(),no_new_training_after=(now+timedelta(hours=9)).isoformat(),no_new_generation_after=(now+timedelta(hours=11)).isoformat(),target_at=(now+timedelta(hours=12)).isoformat(),deadline_at=(now+timedelta(hours=14)).isoformat(),reserve_fraction=.25,independent_confirmation=False,verification_role='R2_VERIFY24_EXTENSION'))
 old=read(OLD/'RESOURCE_LEDGER.json');assert all(s.get('ended_epoch') for s in old['gpu_sessions'])
 write(ROOT/'private/HISTORICAL_LEDGER.json',old)
 write(ROOT/'RESOURCE_LEDGER.json',dict(gpu_seconds_limit=57600,gpu_seconds_used=old['gpu_seconds_used'],historical_gpu_seconds=old['gpu_seconds_used'],historical_judge_attempt_items=old['judge_submission_attempt_items'],gpu_sessions=[],reservations=[],judge_submission_attempt_items_limit=6000,judge_submission_attempt_items=old['judge_submission_attempt_items'],physical_requests_limit=6000,physical_requests=old['physical_requests'],judge_attempts=[],status='AUDIT'))
 for d in ['source','source_patch']:copytree(OLD/d,ROOT/d)
 for n in ['worker_v3.py','worker_r2.py','metrics.py','metrics_r2.py']:shutil.copyfile(OLD/n,ROOT/n)
 (ROOT/'models').symlink_to((OLD/'models').resolve(),target_is_directory=True)
 for n in ['TASKS_R2_LOCKED.json','TASKS_MATRIX.json','BASE_MASKS.json','PRESSURE_VALIDATION_FROZEN.json','PRESSURE_TEST_FROZEN.json','AUXILIARY_POOL.json']:
  shutil.copyfile(OLD/'private'/n,ROOT/'private'/n)
 for d in ['base','keys','black','judge/scores','judge/pending']:
  copytree(OLD/'private'/d,ROOT/'private'/d)
 for d in ['teachers','judge/evidence','runs','jobs','logs','cache','tmp']:(ROOT/('private/'+d if d in ['teachers','judge/evidence'] else d)).mkdir(parents=True,exist_ok=True)
 cfg=read(OLD/'CONFIG_LOCK.json');cfg.update(review_commit='532b81be5b96848e5ccd9d46e8252d1155fe6aac',arm_names=['P@80/R0','P+S@80/R0','P@80/R1','P+S@80/R1'],steps=80,seed=20260925,verification_role='R2_VERIFY24_EXTENSION')
 write(ROOT/'CONFIG_LOCK.json',cfg)
 write(ROOT/'RUN_STATUS.json',dict(status='AUDIT',phase='INVENTORY',epoch=time.time()))
 print(ROOT)
if __name__=='__main__':main()
