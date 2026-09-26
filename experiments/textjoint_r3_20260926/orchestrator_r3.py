"""Finite dependency queue, no background research expansion or clock resets."""
import os,sys,subprocess,time,signal,traceback
from pathlib import Path
ROOT=Path(os.environ['RUN_ROOT']);sys.path.insert(0,str(ROOT))
from budget import read,write,reserve,close
from datetime import datetime

def epoch(s):return datetime.fromisoformat(s).timestamp()
def gpu(index):
 rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True)
 row=next(r.split(', ') for r in rows.splitlines() if r.split(', ')[0]==str(index))
 if int(row[2])>=100:raise RuntimeError(f'GPU {index} is not idle; do not disturb other tasks')
 return row[1]
def command(pid):
 try:return Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
 except FileNotFoundError:return ''
def alive(pid):
 try:return Path(f'/proc/{pid}/stat').read_text().split()[2]!='Z'
 except FileNotFoundError:return False

def stop_own(pid):
 if command(pid).strip()=='/root/anaconda3/bin/python /tmp/r3x.py':os.kill(pid,signal.SIGTERM)

def launch(job,index,seconds):
 jid=job['id'];folder=ROOT/'jobs'/jid
 if (folder/'STATUS.json').exists():
  state=read(folder/'STATUS.json');assert state['job']==job,'Job binding changed'
  if state['status']=='GPU_COMPLETE':return None
  if state['status']=='RUNNING' and (folder/'PROCESS.json').exists():
   receipt=read(folder/'PROCESS.json')
   assert alive(receipt['pid']) and command(receipt['pid']).strip()=='/root/anaconda3/bin/python /tmp/r3x.py'
   return receipt
  raise RuntimeError('Prior failure requires diagnosis, not automatic re-execution')
 uuid=gpu(index);write(folder/'JOB.json',job);reserve(jid,uuid,seconds)
 env=os.environ.copy();env.update(RUN_JOB_JSON=str(folder/'JOB.json'),JOB_ID=jid,PINNED_GPU_UUID=uuid,CUDA_VISIBLE_DEVICES=uuid,TMPDIR=str(ROOT/'tmp'),PYTHONUNBUFFERED='1')
 with (ROOT/'logs'/f'{jid}.log').open('ab') as out:p=subprocess.Popen(['/root/anaconda3/bin/python','/tmp/r3x.py'],env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
 receipt=dict(pid=p.pid,job=jid,gpu_uuid=uuid,started_epoch=time.time(),reserved_seconds=seconds)
 write(ROOT/f'ACTIVE_GPU{index}.json',receipt)
 time.sleep(.3);assert command(p.pid).strip()=='/root/anaconda3/bin/python /tmp/r3x.py'
 receipt['argv']=command(p.pid).strip();receipt['cwd']=str(Path(f'/proc/{p.pid}/cwd').resolve());write(folder/'PROCESS.json',receipt)
 return receipt

def wait(receipts):
 active=[r for r in receipts if r]
 while active:
  now=time.time();manifest=read(ROOT/'RUN_MANIFEST.json')
  stop=(ROOT/'STOP').exists() or now>=epoch(manifest['deadline_at']) or (ROOT/'JUDGE_FAILURE.json').exists()
  for r in list(active):
   if stop or now-r['started_epoch']>=r['reserved_seconds']-5:
    stop_own(r['pid']);time.sleep(2)
    if alive(r['pid']):raise RuntimeError('Worker did not exit after stop; do not mark ledger closed')
    close(r['job']);raise TimeoutError('Stop, Judge failure, or residency reservation reached')
   if not alive(r['pid']):
    close(r['job']);status=read(ROOT/'jobs'/r['job']/'STATUS.json')
    assert status['status']=='GPU_COMPLETE',status
    active.remove(r)
  if active:time.sleep(5)

def score_coverage():
 p=ROOT/'private/judge';pending={x.stem for x in (p/'pending').glob('*.json')};scored={x.stem for x in (p/'scores').glob('*.json')};failed=set(read(p/'JUDGE_MISSING_LOCK.json')['keys'])-scored
 ledger=read(ROOT/'RESOURCE_LEDGER.json');flight={k for a in ledger['judge_attempts'] if a['status']=='RESERVED' for k in a.get('keys',[])}-scored-failed
 c=dict(expected=len(pending),scored=len(pending&scored),FAILED_NO_RETRY=len(pending&failed),IN_FLIGHT=len(pending&flight),UNSUBMITTED=len(pending-scored-failed-flight),MISSING=len(pending-scored))
 write(ROOT/'public/JUDGE_COVERAGE.json',c);return c

def wait_scores(canary=False):
 deadline=epoch(read(ROOT/'RUN_MANIFEST.json')['deadline_at'])-600
 while time.time()<deadline:
  if (ROOT/'STOP').exists() or (ROOT/'JUDGE_FAILURE.json').exists():raise RuntimeError('Scorer blocked; preserve current results')
  c=score_coverage()
  if not c['UNSUBMITTED'] and not c['IN_FLIGHT']:
   if canary and c['MISSING']:raise RuntimeError('Canary Judge loop has missing decisions')
   return c
  time.sleep(10)
 raise TimeoutError('Scoring hard deadline reserve reached')

def main():
 write(ROOT/'ORCHESTRATOR_PID.json',dict(pid=os.getpid(),entry='/tmp/r3o.py',epoch=time.time()))
 active=[]
 try:
  # The first two-edit canary was explicitly launched after source/data audit.
  canary=read(ROOT/'ACTIVE_GPU2.json');canary.setdefault('reserved_seconds',600)
  wait([canary]);wait_scores(canary=True)
  write(ROOT/'CANARY_CLOSURE.json',dict(status='PASS',Judge='complete or exact-input reuse',GPU=read(ROOT/'jobs/canary/CANARY.json')))
  write(ROOT/'RUN_STATUS.json',dict(status='RUNNING',phase='CAL_ROUTE',epoch=time.time()))
  wait([launch(dict(id='calibration',mode='calibration',orders=list(range(1,25))),3,600)])
  # Full core upper bound is reserved by jobs; >25% of initial remainder remains
  # available for validation/score closure, not optional extensions.
  m=read(ROOT/'RUN_MANIFEST.json');d=read(ROOT/'RESOURCE_LEDGER.json')
  write(ROOT/'public/BUDGET_FORECAST.json',dict(core_forecast_gpu_seconds=15000,initial_remaining_gpu_seconds=57600-d['historical_gpu_seconds'],closure_reserved_gpu_seconds=.25*(57600-d['historical_gpu_seconds']),Judge_unique_upper_bound=4437,single_consumer_estimate=4*(268+285),sparse_sequential='each current prefix independently routed/generated',training_stop=m['no_new_training_after'],generation_stop=m['no_new_generation_after'],hard_stop=m['deadline_at']))
  phases=[]
  for panel,lo,hi in [('dev',1,24),('verify',25,48)]:
   for chunk,(a,b) in enumerate([(lo,lo+11),(lo+12,hi)]):
    jobs=[]
    for wi,writer in enumerate(['P','P+S']):
     jobs.append((dict(id=f'{panel}-single-{wi}-{chunk}',mode='single',orders=list(range(a,b+1)),writer=writer,routers=['R0','R1'],prefixes=[],diagnose=panel=='dev'),2+(wi+chunk)%2,2200 if writer=='P+S' else 1600))
    phases.append(jobs)
   phases.append([(dict(id=f'{panel}-seq-{wi}',mode='sequential',orders=list(range(lo,hi+1)),writer=writer,routers=['R0','R1'],prefixes=[12,24]),2+(wi+(panel=='verify'))%2,2000) for wi,writer in enumerate(['P','P+S'])])
  for jobs in phases:
   write(ROOT/'RUN_STATUS.json',dict(status='RUNNING',phase=jobs[0][0]['id'],epoch=time.time()))
   active=[]
   for spec in jobs:active.append(launch(*spec))
   wait(active);active=[]
   from report_r3 import report
   report(final=False)
  write(ROOT/'RUN_STATUS.json',dict(status='SCORING',phase='CORE',epoch=time.time()));wait_scores()
  from report_r3 import report
  report(final=False)
  # Extension A uses existing writers and the same VERIFY panel, never new cases.
  d=read(ROOT/'RESOURCE_LEDGER.json');remaining=d['gpu_seconds_limit']-d['gpu_seconds_used'];c=score_coverage()
  can_extend=(not c['MISSING'] and remaining>=7200 and d['judge_submission_attempt_items_limit']-d['judge_submission_attempt_items']>=1140 and time.time()+2400<epoch(m['no_new_generation_after']))
  extensions=dict(A48='AUTHORIZED_IF_CORE_CLOSED_AND_BUDGET_FITS' if can_extend else 'SKIPPED_CORE_OR_RESERVE_GATE',B_new24='NOT_QUALIFIED: no separately audited frozen new edit pool',C_second_seed='DEFERRED: new common P-W0 requires separate full initialization and measured budget after core; not substituted with old seed',D_baselines='LOWER_PRIORITY_THAN_MATCHED_CORE_AND_CLOSURE')
  write(ROOT/'public/EXTENSION_ASSESSMENT.json',extensions)
  if can_extend:
   ts=read(ROOT/'private/TASKS_R2_LOCKED.json')['tasks'];native_groups={t['native']['source_group'] for t in ts}
   pressure=[r for t in ts[24:] for r in t['evaluation'] if r['task']=='T2L_PRESSURE']
   assert all(r['source_group'] not in native_groups for r in pressure),'Whole-bank pressure role collision'
   jobs=[(dict(id=f'a48-{wi}',mode='sequential',orders=list(range(1,49)),eval_orders=list(range(25,49)),writer=w,routers=['R0','R1'],prefixes=[48]),2+wi,1800) for wi,w in enumerate(['P','P+S'])]
   write(ROOT/'RUN_STATUS.json',dict(status='RUNNING',phase='A48_SAME_VERIFY_INPUTS',epoch=time.time()))
   active=[]
   for spec in jobs:active.append(launch(*spec))
   wait(active);active=[];wait_scores()
  report(final=True)
  write(ROOT/'RUN_STATUS.json',dict(status='COMPLETE',phase='LOCAL_REPORT_READY_PUBLICATION_PENDING',epoch=time.time(),independent_confirmation=False))
 except Exception as e:
  for r in active:
   if r and alive(r['pid']):stop_own(r['pid'])
  time.sleep(2)
  for r in active:
   if r and not alive(r['pid']):close(r['job'])
  write(ROOT/'RUN_STATUS.json',dict(status='BLOCKED',phase='PRESERVED',error=str(e),traceback=traceback.format_exc(),epoch=time.time()))
  try:
   from report_r3 import report
   report(final=False)
  except Exception:pass
  raise
if __name__=='__main__':main()
