"""One campaign on two authorized GPUs; inherits the original clock and ledgers."""
import json,os,signal,subprocess,sys,time,traceback,hashlib
from datetime import datetime
from pathlib import Path
R=Path(os.environ['RUN_ROOT'])
os.environ['RUN_ROOT']=str(R)
sys.path[:0]=[str(R/'source_patch'),str(R/'source'),str(R)]
from metrics import job_report
from worker_v3 import request,input_id
GPU={2:os.environ['GPU2_UUID'],3:os.environ['GPU3_UUID']}
M=json.loads((R/'RUN_MANIFEST.json').read_text())
END=datetime.fromisoformat(M['deadline_at']).timestamp();TRAIN_END=datetime.fromisoformat(M['no_new_training_after']).timestamp();TARGET=datetime.fromisoformat(M['target_at']).timestamp()

def read(path):return json.loads(Path(path).read_text())
def write(path,value):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2));tmp.replace(p)
def used():
 d=read(R/'RESOURCE_LEDGER.json');return d['gpu_seconds_used']+sum(time.time()-s['started_epoch'] for s in d['gpu_sessions'] if s.get('ended_epoch') is None)
def check():
 if (R/'STOP').exists() or time.time()>=END or used()>=57600:raise TimeoutError('Original hard wall or 16 GPU-resident-hour limit')
 if (R/'JUDGE_FAILURE.json').exists():raise RuntimeError('Judge failure; preserve missing and stop expansion')

def alive(pid):
 try:os.kill(pid,0);return True
 except ProcessLookupError:return False

def wait_job(job_id,pid):
 status=R/'jobs'/job_id/'STATUS.json'
 while True:
  check();state=read(status) if status.exists() else {}
  if state.get('status')=='GPU_COMPLETE':return state
  if state.get('status')=='FAILED':raise RuntimeError(job_id+' failed: '+state.get('error',''))
  if not alive(pid):raise RuntimeError(job_id+' exited without success receipt')
  time.sleep(5)

def score_coverage():
 p={f.stem for f in (R/'private/judge/pending').glob('*.json')};s={f.stem for f in (R/'private/judge/scores').glob('*.json')}
 return dict(expected=len(p),scored=len(p&s),missing=len(p-s))
def wait_scores():
 began=time.time()
 while True:
  check();coverage=score_coverage();write(R/'SCORE_COVERAGE.json',coverage)
  if not coverage['missing']:return
  if time.time()-began>1800:raise TimeoutError('Judge backlog did not close in 30 minutes')
  time.sleep(10)

def freeze_base():
 if (R/'private/BASE_MASKS.json').exists():
  assert read(R/'private/BASE_MASKS.json')['status']=='FROZEN_A100_BEFORE_SCREEN';return
 wait_scores()
 scores={p.stem:read(p)['is_correct'] for p in (R/'private/judge/scores').glob('*.json')}
 masks=[]
 for t in read(R/'private/TASKS_MATRIX.json')['tasks']:
  for row in t['evaluation']:
   out=read(R/'private/base'/f'{input_id(row)}.json');key=request(row,out)
   assert key in scores
   masks.append(dict(edit=t['canonical_edit_id'],task=row['task'],query_id=row['query_id'],base_correct=scores[key],judge_key=key))
 write(R/'private/BASE_MASKS.json',dict(status='FROZEN_A100_BEFORE_SCREEN',rows=masks,epoch=time.time()))

def launch(job,gpu):
 check();status=R/'jobs'/job['id']/'STATUS.json'
 if status.exists():
  state=read(status)
  assert state['job']==job
  if state['status']=='GPU_COMPLETE':return read(R/f'ACTIVE_GPU{gpu}.json')['pid'] if (R/f'ACTIVE_GPU{gpu}.json').exists() else 0
  if state['status']=='FAILED':raise RuntimeError('Prior failure; no silent training retry: '+job['id'])
  active=read(R/f'ACTIVE_GPU{gpu}.json');assert active['job']==job and alive(active['pid']);return active['pid']
 path=R/f'JOB{gpu}.json';write(path,job)
 env=os.environ.copy();env.update(RUN_JOB_JSON=str(path),PINNED_GPU_UUID=GPU[gpu],CUDA_VISIBLE_DEVICES=GPU[gpu],TMPDIR=str(R/'tmp'),HF_HOME=str(R/'cache'),OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 log=R/'logs'/f"{job['id']}.log"
 with log.open('x') as out:p=subprocess.Popen([sys.executable,'-c','import worker_v3; worker_v3.main()'],cwd=R,stdin=subprocess.DEVNULL,stdout=out,stderr=out,start_new_session=True,env=env)
 write(R/f'ACTIVE_GPU{gpu}.json',dict(job=job,pid=p.pid,gpu_index=gpu,epoch=time.time()))
 print('START',job['id'],gpu,p.pid,flush=True);return p.pid

def group(phase,items):
 write(R/'RUN_STATUS.json',dict(status='RUNNING',phase=phase,jobs=[x[0]['id'] for x in items],epoch=time.time()))
 started=[(job,gpu,launch(job,gpu)) for job,gpu in items]
 try:
  for job,gpu,pid in started:wait_job(job['id'],pid)
 except BaseException:
  for job,gpu,pid in started:
   if pid and alive(pid):
    try:os.killpg(pid,signal.SIGINT)
    except ProcessLookupError:pass
  raise
 wait_scores();write(R/'private/JOB_RESULTS.json',job_report(R))
 print('CLOSED',phase,flush=True)

def job(name,mode,n,arms,**kwargs):return dict(id=name,mode=mode,N=n,arms=arms,seed=kwargs.pop('seed',20260924),**kwargs)
def screen_result(arm):
 v=job_report(R)[f'screen12-{arm.lower()}']
 assert v['status']['status']=='GPU_COMPLETE'
 return v

def forecast_arm(arm):
 import statistics
 points=[read(p)['seconds'] for p in (R/'runs/s20260924/B0').glob('e*/TRAINING_COMPLETE.json')]
 m=statistics.mean(points)
 return 12*m*dict(P=1,E=1.35,U=1.7,EU=2.5,PEU=2.5)[arm]*1.3

def close_report(status,best=None,error=None,n=12):
 report=job_report(R);write(R/'private/JOB_RESULTS.json',report)
 public=R/'public';public.mkdir(exist_ok=True)
 safe={k:dict(status=v['status']['status'],job=v['status']['job'],summary=v['summary'],comparisons=[{x:y for x,y in c.items() if x!='changes'} for c in v['comparisons']],consumers=v['consumers']) for k,v in report.items()}
 write(public/'RESULTS.json',safe)
 d=read(R/'RESOURCE_LEDGER.json');receipt=dict(status=status,error=error,best_exploratory_arm=best,independent_confirmation=False,expected_scores=score_coverage()['expected'],scored=score_coverage()['scored'],missing=score_coverage()['missing'],gpu_resident_seconds=d['gpu_seconds_used'],gpu_resident_limit=57600,judge_items_charged=d['judge_submission_attempt_items'],judge_limit=d['judge_submission_attempt_items_limit'],original_wall_deadline=M['deadline_at'],no_new_training_after=M['no_new_training_after'],prior_Blackwell_B0_not_pooled=True)
 if best:
  p=R/f'jobs/seq-best{n}/{best}/ACTIVE_BANK.pt'
  if p.exists():receipt['best_bank']=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
 write(R/'FINAL.json',receipt);write(public/'FINAL.json',{k:v for k,v in receipt.items() if k not in ['error','best_bank']})
 lines=['# MedTRACE 本轮迁移后实验报告','',f'状态：{status}。新主机 A100 GPU 2/3；原墙钟和 GPU/Judge 预算未重置。',f'最佳开发参考：{best or "无"}；独立确认集：无。',f'有效Judge评分：{receipt["scored"]}/{receipt["expected_scores"]}；missing={receipt["missing"]}。',f'累计GPU驻留：{receipt["gpu_resident_seconds"]/3600:.2f}/16小时；Judge保守计费：{receipt["judge_items_charged"]}/{receipt["judge_limit"]}项。','', '同轮Base掩码仅使用A100新生成，旧服务器B0成绩单列，不跨硬件直接配对。','结果表的edit-macro、micro、分子分母、missing与配对置信区间见RESULTS.json。','']
 if error:lines+=['阻塞原因：'+error,'']
 for name,v in safe.items():
  lines+=[f'## {name}',f'状态：{v["status"]}；消费者：{v["consumers"]}。','']
 (public/'FINAL_REPORT_ZH.md').write_text('\n'.join(lines))
 write(R/'RUN_STATUS.json',dict(status=status,best=best,error=error,epoch=time.time()))


def main():
 best=None;n=12
 try:
  active=read(R/'ACTIVE_GPU2.json');assert active['job']['id']=='base48-a100'
  wait_job('base48-a100',active['pid']);freeze_base()
  group('B0_P_SINGLE12',[(job('b0-12-a100','single',12,['B0']),2),(job('screen12-p','single',12,['P'],baseline_job='b0-12-a100'),3)])
  group('DIAGNOSTIC_E_SINGLE12',[(job('diagnostic12-a100','diagnostic',12,[]),2),(job('screen12-e','single',12,['E'],baseline_job='b0-12-a100'),3)])
  completed=['P','E'];deferred=[]
  for arms in [('U','EU'),('PEU',)]:
   batch=[]
   for arm,gpu in zip(arms,[2,3]):
    cost=forecast_arm(arm)
    if time.time()+cost+600>=TRAIN_END or used()+cost>=57600:
     deferred.append(dict(arm=arm,reason='original time or GPU budget',forecast_seconds=cost));continue
    batch.append((job(f'screen12-{arm.lower()}','single',12,[arm],baseline_job='b0-12-a100'),gpu))
   if batch:
    group('SCREEN_'+''.join(arms),batch);completed += [j['arms'][0] for j,g in batch]
  write(R/'SCREEN_MATRIX_STATUS.json',dict(completed=completed,deferred=deferred,all_six_complete=len(completed)==5))
  from controller_v2 import choose
  combined=dict(comparisons=[],summary=[])
  for arm in completed:
   v=screen_result(arm);combined['comparisons']+=v['comparisons'];combined['summary']+=v['summary']
  try:best=choose(combined)
  except RuntimeError:
   best=completed[0] if completed else None
   write(R/'CANDIDATE_SELECTION.json',dict(status='NO_COMPLETE_JOINT_DENOMINATOR',diagnostic_followup_arm=best,independent_confirmation=False))
  if not best:raise RuntimeError('No complete candidate to deploy')
  all_tasks=read(R/'private/TASKS_MATRIX.json')['tasks'];ledger=read(R/'RESOURCE_LEDGER.json');remaining_judge=ledger['judge_submission_attempt_items_limit']-ledger['judge_submission_attempt_items']
  plans=[]
  for target in [48,36]:
   times=[]
   for arm in ['B0',best]:
    a=[read(p)['seconds'] for p in (R/'runs/s20260924'/arm).glob('e*/TRAINING_COMPLETE.json')]
    times.append(sum(a)/len(a))
   wallcost=max(times)*(target-12)*1.3
   gpucost=sum(times)*(target-12)*1.3
   scorebound=2*sum((target-i)*len(t['evaluation']) for i,t in enumerate(all_tasks[:target]))
   feasible=time.time()+wallcost+900<TRAIN_END and used()+gpucost<57600 and scorebound<=remaining_judge
   plans.append(dict(N=target,wall_forecast_seconds=wallcost,gpu_forecast_seconds=gpucost,conservative_Judge_items=scorebound,remaining_Judge_items=remaining_judge,feasible=feasible))
   if feasible:
    group(f'EXPAND{target}',[(job(f'expand{target}-b0','single',target,['B0']),2),(job(f'expand{target}-best','single',target,[best],baseline_job=f'expand{target}-b0'),3)])
    n=target;break
  write(R/'EXPANSION_FORECAST.json',dict(plans=plans,chosen_N=n,epoch=time.time()))
  b0_reference=f'expand{n}-b0' if n>12 else 'b0-12-a100'
  write(R/'METHOD_LOCK.json',dict(best=best,N=n,status='EXPLORATORY_ONLY',epoch=time.time()))
  group('PAIRED_SEQUENTIAL',[(job(f'seq-b0-{n}','sequential',n,['B0']),2),(job(f'seq-best{n}','sequential',n,[best],baseline_job=f'seq-b0-{n}'),3)])
  if time.time()+1800<TRAIN_END and used()+3600<57600:
   group('BALANCEDIT',[(job(f'baseline-single{n}','single',n,['BalancEdit'],baseline_job=b0_reference),2),(job(f'baseline-sequential{n}','sequential',n,['BalancEdit'],baseline_job=f'seq-b0-{n}'),3)])
  first12=all_tasks[:12]
  seed_judge_upper=2*sum((12-i)*len(t['evaluation']) for i,t in enumerate(first12))+2*sum(len(t['evaluation']) for t in first12)
  for index,seed in enumerate([20260925,20260926],2):
   times=[]
   for arm in ['B0',best]:
    a=[read(p)['seconds'] for p in (R/'runs/s20260924'/arm).glob('e*/TRAINING_COMPLETE.json')]
    times.append(sum(a)/len(a))
   cost=12*max(times)*1.3
   remaining=read(R/'RESOURCE_LEDGER.json')['judge_submission_attempt_items_limit']-read(R/'RESOURCE_LEDGER.json')['judge_submission_attempt_items']
   if time.time()+cost+900>=TRAIN_END or used()+24*max(times)*1.3>=57600 or seed_judge_upper>remaining:break
   group(f'SEED{index}_SINGLE',[(job(f'seed{index}-b0-12','single',12,['B0'],seed=seed),2),(job(f'seed{index}-best12','single',12,[best],seed=seed,baseline_job=f'seed{index}-b0-12'),3)])
   group(f'SEED{index}_SEQUENTIAL',[(job(f'seed{index}-seq-b0-12','sequential',12,['B0'],seed=seed),2),(job(f'seed{index}-seq-best12','sequential',12,[best],seed=seed,baseline_job=f'seed{index}-seq-b0-12'),3)])
  remaining=read(R/'RESOURCE_LEDGER.json')['judge_submission_attempt_items_limit']-read(R/'RESOURCE_LEDGER.json')['judge_submission_attempt_items']
  reverse_bound=2*sum((12-i)*len(t['evaluation']) for i,t in enumerate(first12))
  if time.time()+1800<TARGET and used()+3600<57600 and reverse_bound<=remaining:
   group('REVERSE_ORDER_SEQUENTIAL',[(job('reverse-seq-b0-12','sequential',12,['B0'],reverse=True),2),(job('reverse-seq-best12','sequential',12,[best],reverse=True,baseline_job='reverse-seq-b0-12'),3)])
  close_report('COMPLETE_EXPLORATORY',best,n=n)
 except Exception as e:
  write(R/'ORCHESTRATOR_ERROR.json',dict(error=str(e),traceback=traceback.format_exc(),epoch=time.time()))
  close_report('BUDGET_STOP' if isinstance(e,TimeoutError) else 'BLOCKED',best,str(e),n=n)
if __name__=='__main__':main()
