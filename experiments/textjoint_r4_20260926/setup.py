"""One-time R4 setup; carry exact historical usage and quarantine forward."""
import json,os,shutil,subprocess
from pathlib import Path
from datetime import datetime,timezone,timedelta
PREV=Path('/remote-home/wangbomin/Knowledge_editing/outputs/textjoint-r3-20260926/run')
ROOT=PREV.parents[1]/'textjoint-r4-20260926/run'
def read(p):return json.loads(p.read_text())
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def copytree(a,b):
 for parent,dirs,files in os.walk(a):
  dirs[:]=[d for d in dirs if d not in ['__pycache__','.git']];dest=b/Path(parent).relative_to(a);dest.mkdir(parents=True,exist_ok=True)
  for f in files:shutil.copyfile(Path(parent)/f,dest/f)
def main():
 assert not ROOT.exists(),'Never reset campaign state'
 print(subprocess.check_output(['findmnt','-T',str(PREV)],text=True));assert shutil.disk_usage(PREV).free>20*1024**3
 probe=PREV.parents[1]/'.r4-mount-probe';probe.write_text('r4');assert probe.read_text()=='r4';probe.unlink()
 ROOT.mkdir(parents=True);now=datetime.now(timezone.utc)
 manifest=dict(run_id=ROOT.parent.name,review_commit='ed375b4e879ce609bc74f9657f743a42c6538fd5',first_started_at=now.isoformat(),p0_deadline_at=(now+timedelta(hours=3)).isoformat(),no_new_training_after=(now+timedelta(hours=8)).isoformat(),no_new_generation_after=(now+timedelta(hours=9)).isoformat(),target_at=(now+timedelta(hours=10)).isoformat(),deadline_at=(now+timedelta(hours=12)).isoformat(),reserve_fraction=.25,independent_confirmation=False)
 write(ROOT/'RUN_MANIFEST.json',manifest)
 ledger=read(PREV/'RESOURCE_LEDGER.json');assert all(s.get('ended_epoch') for s in ledger['gpu_sessions'])
 write(ROOT/'private/HISTORICAL_LEDGER.json',ledger)
 write(ROOT/'RESOURCE_LEDGER.json',dict(gpu_seconds_limit=57600,gpu_seconds_used=ledger['gpu_seconds_used'],historical_gpu_seconds=ledger['gpu_seconds_used'],historical_judge_attempt_items=ledger['judge_submission_attempt_items'],gpu_sessions=[],reservations=[],judge_submission_attempt_items_limit=6000,judge_submission_attempt_items=ledger['judge_submission_attempt_items'],physical_requests_limit=6000,physical_requests=ledger['physical_requests'],judge_attempts=[],status='AUDIT'))
 for d in ['source','source_patch']:copytree(PREV/d,ROOT/d)
 for n in ['worker_v3.py','worker_r2.py','worker_r3.py','metrics.py','metrics_r2.py','router_r3.py','budget.py','report_r3.py']:shutil.copyfile(PREV/n,ROOT/n)
 shutil.copyfile(PREV/'worker_r2.py',ROOT/'baseline_r2.py')
 (ROOT/'models').symlink_to((PREV/'models').resolve(),target_is_directory=True)
 for n in ['TASKS_R2_LOCKED.json','TASKS_MATRIX.json','BASE_MASKS.json','PRESSURE_VALIDATION_FROZEN.json','PRESSURE_TEST_FROZEN.json','AUXILIARY_POOL.json']:
  shutil.copyfile(PREV/'private'/n,ROOT/'private'/n)
 for d in ['base','keys','black','teachers','judge/scores','judge/pending']:copytree(PREV/'private'/d,ROOT/'private'/d)
 shutil.copyfile(PREV/'private/judge/JUDGE_MISSING_LOCK.json',ROOT/'private/judge/JUDGE_MISSING_LOCK.json')
 for d in ['private/judge/evidence','jobs','runs','logs','tmp','cache','public']:(ROOT/d).mkdir(parents=True,exist_ok=True)
 cfg=read(PREV/'CONFIG_LOCK.json');cfg.update(review_commit=manifest['review_commit'],beta_candidates=[0,.125,.25,.5],new_beta_arms=[.125,.25],steps=80,seed=20260925,arm_names=['P','B125','B25','P+S'],route_status='P0_PENDING')
 write(ROOT/'CONFIG_LOCK.json',cfg)
 write(ROOT/'SELECTION_RULE.json',dict(frozen_before_outputs=True,reference='P@80/R0',cohort='DEV24 single complete common panel',constraints=dict(T0_no_new_errors=True,T1G_drop_max_pp=1,T2G_drop_max_pp=1),objective='maximum edit-macro T2L_PRESSURE; tie smaller beta',require_complete_scoring=True,statistical_noninferiority_claim=False,candidates=[.125,.25,.5],if_no_positive_beta='P80 reference; NO_FEASIBLE_POSITIVE_BETA'))
 write(ROOT/'RUN_STATUS.json',dict(status='AUDIT',phase='ASSETS_AND_DATA'))
 print(manifest)
if __name__=='__main__':main()
