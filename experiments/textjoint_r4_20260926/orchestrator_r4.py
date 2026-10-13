"""Canary-gated finite dose queue, then frozen validation; no Rscope without data."""
import os,sys,time,traceback
from pathlib import Path
ROOT=Path(os.environ['RUN_ROOT']);sys.path.insert(0,str(ROOT))
from budget import read,write,close
from execution import launch,wait,wait_scores,score_coverage,alive,stop_own
from report_r4 import report,select

def guard_reserve(seconds,development):
 d=read(ROOT/'RESOURCE_LEDGER.json');initial=d['gpu_seconds_limit']-d['historical_gpu_seconds'];reserve=.25*initial if development else 600
 if d['gpu_seconds_used']+seconds+reserve>d['gpu_seconds_limit']:raise TimeoutError('Preserve validation/closure residency reserve; do not expand this phase')

def main():
 write(ROOT/'ORCHESTRATOR_PID.json',dict(pid=os.getpid(),entry='/tmp/r4o.py',epoch=time.time()));active=[]
 try:
  assert read(ROOT/'AUDIT.json')['status']=='PASS_WITH_DATA_LIMITATIONS'
  assert read(ROOT/'ROUTER_LOCK.json')['status']=='R0_FALLBACK_INCOMPLETE_CAL_PLUS','Qualified routing needs its own implemented/calibrated path; never silently fall back'
  s=read(ROOT/'jobs/dev-canary/STATUS.json')
  if s['status']!='GPU_COMPLETE':wait([read(ROOT/'jobs/dev-canary/PROCESS.json')])
  canary=read(ROOT/'jobs/dev-canary/CANARY.json');assert canary['status']=='PASS'
  wait_scores()
  for f in (ROOT/'jobs/dev-canary').glob('*/p*/CONSUMERS.json'):
   assert all((ROOT/'private/judge/scores'/f"{r['judge_key']}.json").exists() for r in read(f)),'Canary Judge missing'
  write(ROOT/'CANARY_CLOSURE.json',dict(status='PASS',new_beta_trajectories=4,actual_steps=80,failed_historical_A48_excluded=True))
  trains=[read(p)['seconds'] for p in (ROOT/'runs/s20260925').glob('*/*/TRAINING_COMPLETE.json')];generations=[r['output']['seconds'] for p in (ROOT/'jobs/dev-canary').glob('*/p*/CONSUMERS.json') for r in read(p)]
  avg_train=sum(trains)/len(trains);avg_generation=sum(generations)/len(generations)
  core_estimate=44*avg_train+2*536*avg_generation+24*avg_train+(285+435)*avg_generation+1800
  write(ROOT/'public/BUDGET_FORECAST.json',dict(measured_train_seconds_per_edit80=avg_train,measured_generation_seconds_per_input=avg_generation,remaining_core_forecast_gpu_seconds=core_estimate,history_gpu_seconds=read(ROOT/'RESOURCE_LEDGER.json')['historical_gpu_seconds'],reserve_fraction=.25,parallel_hours_counted_per_GPU=True,new_arms=2,reused_reference_consumers=2776,optional_routes='not qualified',new_A48='not justified because route remains R0'))
  for chunk,orders in enumerate([list(range(3,13)),list(range(13,25))]):
   write(ROOT/'RUN_STATUS.json',dict(status='RUNNING',phase=f'DEV_DOSES_{chunk}',epoch=time.time()))
   guard_reserve(4400,True);active=[]
   for i,arm in enumerate(['B125','B25']):active.append(launch(dict(id=f'dev-{arm}-{chunk}'+('-resume1' if chunk==0 else ''),mode='single',orders=orders,writer=arm,prefixes=[]),2+(i+chunk)%2,2200))
   wait(active);active=[];report()
  guard_reserve(3000,True);write(ROOT/'RUN_STATUS.json',dict(status='RUNNING',phase='DEV_SEQ24',epoch=time.time()))
  for i,arm in enumerate(['B125','B25']):active.append(launch(dict(id=f'dev-seq-{arm}',mode='sequential',orders=list(range(1,25)),writer=arm,prefixes=[24]),2+i,1500))
  wait(active);active=[];write(ROOT/'RUN_STATUS.json',dict(status='SCORING',phase='DEV_SELECTION',epoch=time.time()));wait_scores();report();locked=select()
  # Attribution only describes fixed R3 failures; it cannot alter this selection.
  guard_reserve(600,False);active=[launch(dict(id='attribution',mode='attribution',orders=[35,36,42],entry='/tmp/r4d.py'),2,600)];wait(active);active=[]
  writer=locked['selected_writer']
  if writer in ['B125','B25']:
   write(ROOT/'RUN_STATUS.json',dict(status='RUNNING',phase='VERIFY_FROZEN_WRITER',epoch=time.time()));guard_reserve(4400,False)
   for i,orders in enumerate([list(range(25,37)),list(range(37,49))]):active.append(launch(dict(id=f'verify-{writer}-{i}',mode='single',orders=orders,writer=writer,prefixes=[]),2+i,2200))
   wait(active);active=[];guard_reserve(1800,False)
   active=[launch(dict(id=f'verify-seq-{writer}',mode='sequential',orders=list(range(25,49)),writer=writer,prefixes=[12,24]),3,1800)];wait(active);active=[]
  write(ROOT/'public/MATRIX_LOCK.json',dict(writer=writer,beta=locked['selected_beta'],route='R0',distinct_core_writers=list(dict.fromkeys(['P',writer])),tradeoff_reference='P+S beta=.5',route_arms_collapsed=True,interaction='degenerate zero because R*=R0; no route improvement identified',formal_panel='R2_VERIFY24_REGRESSION',A48='NOT_RUN: no new route capacity behavior to validate'))
  write(ROOT/'RUN_STATUS.json',dict(status='SCORING',phase='FINAL',epoch=time.time()));wait_scores();report(final=True)
  write(ROOT/'RUN_STATUS.json',dict(status='COMPLETE',phase='LOCAL_REPORT_READY_PUBLICATION_PENDING',epoch=time.time(),route_calibration_incomplete=True,independent_CONFIRM=False))
 except Exception as e:
  for r in active:
   if r and alive(r['pid']):stop_own(r['pid'])
  time.sleep(2)
  for r in active:
   if r and not alive(r['pid']):close(r['job'])
  write(ROOT/'RUN_STATUS.json',dict(status='BLOCKED',error=str(e),traceback=traceback.format_exc(),epoch=time.time()))
  try:report()
  except Exception:pass
  raise
if __name__=='__main__':main()
