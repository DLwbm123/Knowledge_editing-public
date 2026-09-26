"""Private paired evidence and public aggregates; never score a missing subset."""
import os,json,random,time,csv
from pathlib import Path
from collections import defaultdict,Counter
from statistics import mean
ROOT=Path(os.environ['RUN_ROOT'])
from budget import read,write
from metrics_r2 import summarize,retention

def paired(rows,scores,arms,weights):
 maps=[{(r['edit'],r['task'],r['query_id']):r for r in rows if r['arm']==a} for a in arms]
 if not maps or any(set(x)!=set(maps[0]) for x in maps):return dict(status='UNPAIRED_COHORT',arms=arms)
 per=defaultdict(list);changes=[]
 for key,r in maps[0].items():
  rr=[m[key] for m in maps];assert len({v['base_judge_key'] for v in rr})==1
  b=scores.get(r['base_judge_key'])
  if b is not retention(r['task']):continue
  vv=[scores.get(v['judge_key']) for v in rr]
  low=sum(w*(int(v) if v is not None else (0 if w>0 else 1)) for w,v in zip(weights,vv))
  high=sum(w*(int(v) if v is not None else (1 if w>0 else 0)) for w,v in zip(weights,vv))
  delta=sum(w*int(v) for w,v in zip(weights,vv)) if all(v is not None for v in vv) else None
  item=dict(edit=r['edit'],source=r['source_group'],task=r['task'],query_id=r['query_id'],values=vv,delta=delta,low=low,high=high)
  changes.append(item);per[r['task']].append(item)
 out={};rng=random.Random(20260926)
 for task,rs in per.items():
  edits=defaultdict(list);sources=defaultdict(list)
  for r in rs:edits[r['edit']].append(r);sources[r['source']].append(r)
  missing=sum(r['delta'] is None for r in rs);vals=[mean(r['delta'] for r in group) for group in edits.values()] if not missing else []
  boots=sorted(mean(rng.choices(vals,k=len(vals))) for _ in range(2000)) if vals else []
  source_boot=[]
  if not missing:
   keys=list(sources)
   for _ in range(2000):
    sampled=defaultdict(list)
    for k in rng.choices(keys,k=len(keys)):
     for r in sources[k]:sampled[r['edit']].append(r['delta'])
    source_boot.append(mean(mean(v) for v in sampled.values()))
   source_boot.sort()
  out[task]=dict(denominator=len(rs),edits=len(edits),source_groups=len(sources),missing=missing,delta_edit_macro=mean(vals) if vals else None,edit_bootstrap_CI95=[boots[49],boots[1949]] if boots else None,source_cluster_sensitivity_CI95=[source_boot[49],source_boot[1949]] if source_boot else None,missing_identification_bounds=[mean(mean(r['low'] for r in g) for g in edits.values()),mean(mean(r['high'] for r in g) for g in edits.values())],right_to_wrong=sum(len(r['values'])==2 and r['values']==[True,False] for r in rs),wrong_to_right=sum(len(r['values'])==2 and r['values']==[False,True] for r in rs))
 return dict(status='MISSING' if any(v['missing'] for v in out.values()) else 'COMPLETE',arms=arms,weights=weights,metrics=out,private_changes=changes)

def route_summary(rows,scores):
 out=[]
 for arm in sorted({r['arm'] for r in rows}):
  for role in ['positive','negative']:
   rs=[r for r in rows if r['arm']==arm and ('negative' if retention(r['task']) else 'positive')==role]
   if not rs:continue
   active=[r for r in rs if r['route']['activated']]
   item=dict(arm=arm,role=role,consumers=len(rs),reject=len(rs)-len(active),activate=len(active))
   if role=='positive':item.update(associated_hit=sum(r['route']['logical_edit_id']==r['edit'] for r in active),other_expert=sum(r['route']['logical_edit_id']!=r['edit'] for r in active))
   else:item.update(activation_damage=sum(scores.get(r['base_judge_key']) is True and scores.get(r['judge_key']) is False for r in active),missing_after_activation=sum(scores.get(r['judge_key']) is None for r in active),no_gold_expert=True)
   for key in ['d1','d2','radius','ratio','margin']:
    vals=[r['route_diagnostics'][key] for r in rs if r['route_diagnostics'][key] is not None];item[key+'_mean']=mean(vals) if vals else None
   out.append(item)
 return out

def report(final=False):
 p=ROOT/'private/judge';scores={f.stem:read(f)['is_correct'] for f in (p/'scores').glob('*.json')}
 cohorts=defaultdict(list);diagnostics=[]
 for job in (ROOT/'jobs').iterdir():
  for f in job.glob('*/p*/CONSUMERS.json'):
   rows=read(f)
   if not rows:continue
   if rows[0]['mode']=='canary':continue
   if rows[0]['mode']=='diagnostic':diagnostics+=rows;continue
   panel='A48_SAME_VERIFY24' if job.name.startswith('a48') else 'DEV24' if job.name.startswith('dev') else 'R2_VERIFY24_EXTENSION'
   cohorts[(panel,rows[0]['mode'],rows[0]['prefix'])]+=rows
 public={};allrows=[]
 for (panel,mode,prefix),rows in sorted(cohorts.items()):
  allrows+=rows;name=f'{panel}_{mode}_{prefix}';summary=summarize(rows,scores)
  for s in summary:
   rs=[r for r in rows if r['arm']==s['arm'] and r['task']==s['task'] and scores.get(r['base_judge_key']) is retention(s['task'])]
   s['unique_inputs']=len({r['input_id'] for r in rs})
   ed=defaultdict(list)
   for r in rs:ed[r['edit']].append(scores.get(r['judge_key']))
   s['missing_macro_bounds']=[mean(mean(int(v) if v is not None else fill for v in vs) for vs in ed.values()) for fill in [0,1]] if ed else None
  pairs={}
  for label,arms,weights in [('S_R0',['P@80_R0','P+S@80_R0'],[-1,1]),('routing_P',['P@80_R0','P@80_R1'],[-1,1]),('routing_PS',['P+S@80_R0','P+S@80_R1'],[-1,1]),('interaction',['P@80_R0','P+S@80_R0','P@80_R1','P+S@80_R1'],[1,-1,-1,1])]:
   pair=paired(rows,scores,arms,weights);changes=pair.pop('private_changes',[]);write(ROOT/'private/reports'/f'{name}_{label}_changes.json',changes);pairs[label]=pair
  public[name]=dict(summary=summary,paired=pairs,routes=route_summary(rows,scores))
 write(ROOT/'public/RESULTS_PUBLIC.json',public)
 pending={f.stem for f in (p/'pending').glob('*.json')};failed=set(read(p/'JUDGE_MISSING_LOCK.json')['keys'])-set(scores);ledger=read(ROOT/'RESOURCE_LEDGER.json')
 flight={k for a in ledger['judge_attempts'] if a['status']=='RESERVED' for k in a.get('keys',[])}-set(scores)-failed
 coverage=dict(expected_requests=len(pending),scored=len(pending&set(scores)),FAILED_NO_RETRY=len(pending&failed),IN_FLIGHT=len(pending&flight),UNSUBMITTED=len(pending-set(scores)-failed-flight),MISSING=len(pending-set(scores)),formal_consumers=len(allrows),formal_unique_execution_inputs=len({r['execution_id'] for r in allrows}),formal_unique_Judge_inputs=len({r['judge_key'] for r in allrows}))
 write(ROOT/'public/SCORING_COVERAGE.json',coverage)
 active_seconds=sum(time.time()-s['started_epoch'] for s in ledger['gpu_sessions'] if s.get('ended_epoch') is None)
 used=ledger['gpu_seconds_used']+active_seconds
 write(ROOT/'public/RESOURCE_LEDGER_AGGREGATE.json',dict(historical_gpu_seconds=ledger['historical_gpu_seconds'],new_gpu_seconds=used-ledger['historical_gpu_seconds'],cumulative_gpu_seconds=used,remaining_gpu_seconds=57600-used,historical_judge_attempt_items=ledger['historical_judge_attempt_items'],new_judge_attempt_items=ledger['judge_submission_attempt_items']-ledger['historical_judge_attempt_items'],cumulative_judge_attempt_items=ledger['judge_submission_attempt_items'],remaining_judge_attempt_items=6000-ledger['judge_submission_attempt_items'],active_sessions=sum(s.get('ended_epoch') is None for s in ledger['gpu_sessions'])))
 write(ROOT/'public/DIAGNOSTIC_AGGREGATE.json',dict(FORCED_ON_performance=summarize(diagnostics,scores),note='All diagnostic outputs force the associated expert; recorded R0 decisions are counterfactual routing metadata, not actual activation. Base OFF uses the frozen Base scores and outputs.'))
 # Compare bank transitions to single only on matching input/edit/arm identities.
 transitions=[]
 for (panel,mode,prefix),rows in cohorts.items():
  if mode!='sequential' or panel.startswith('A48'):continue
  single={(r['arm'],r['edit'],r['query_id']):r for r in cohorts.get((panel,'single',1),[])}
  matched=[(single[(r['arm'],r['edit'],r['query_id'])],r) for r in rows if (r['arm'],r['edit'],r['query_id']) in single]
  for arm in sorted({r['arm'] for r in rows}):
   for role in ['positive','negative']:
    mm=[(s,r) for s,r in matched if r['arm']==arm and ('negative' if retention(r['task']) else 'positive')==role]
    transitions.append(dict(panel=panel,prefix=prefix,arm=arm,role=role,paired_consumers=len(mm),single_OFF_bank_ON=sum(not s['route']['activated'] and r['route']['activated'] for s,r in mm),bank_changed_expert=sum(s['route']['activated'] and r['route']['activated'] and s['route']['logical_edit_id']!=r['route']['logical_edit_id'] for s,r in mm)))
 write(ROOT/'public/ROUTE_TRANSITIONS.json',transitions)
 write(ROOT/'public/CONFIG_PUBLIC.json',dict(layer='L30',rank=4,dtype='float16',seed=20260925,steps=80,review_commit=read(ROOT/'RUN_MANIFEST.json')['review_commit'],P_S_shared_W0=True,lambda_U=.01,old_U_weight=.5,new_U_weight=.5,uniform_new_sampler='frozen R2 stateless seed',Judge_model='gpt-6-astra',Judge_effort='high',Judge_protocol='MEDTRACE_STAGE17_SOURCE_AGREEMENT_V1',generation_max_new_tokens=1024,independent_confirmation=False,verification='R2_VERIFY24_EXTENSION',deadlines=read(ROOT/'RUN_MANIFEST.json')))
 if final:
  import torch
  retained=[];curve_rows=[]
  from worker_r3 import checkpoint,sha
  for t in read(ROOT/'private/TASKS_R2_LOCKED.json')['tasks']:
   for arm in ['P','P+S']:
    point=checkpoint(t,arm)
    if not point.exists():continue
    s=torch.load(point,map_location='cpu',weights_only=True);assert s['step']==80
    retained.append(dict(order=t['order'],writer=arm,steps=80,sha256=sha(point),bytes=point.stat().st_size))
    for c in s['curve']:curve_rows.append(dict(writer=arm,step=c['step'],native_ce=c['native_ce'],fit_ce=c['fit_ce'],U_kl=c['U_kl'],grad_norm=c['grad_norm'],clipped_grad_norm=c.get('clipped_grad_norm'),update_norm=c.get('update_norm'),CE_U_cosine=c.get('CE_U_cosine')))
  write(ROOT/'public/RETAINED_BANKS.json',dict(experts=retained,scope='matched P80/PS80 banks; P-W0/step0 and resume state retained; R2 untouched',deletion_condition='after all registered consumers and user review'))
  agg=[]
  for arm in ['P','P+S']:
   for step in [1,20,40,60,80]:
    rr=[r for r in curve_rows if r['writer']==arm and r['step']==step]
    if rr:agg.append(dict(writer=arm,step=step,edits=len(rr),**{k:mean(r[k] for r in rr if r[k] is not None) if any(r[k] is not None for r in rr) else None for k in rr[0] if k not in ['writer','step']}))
  write(ROOT/'public/CURVES_AGGREGATE.json',agg)
 lines=['# MedTRACE TextJoint-R3 中文报告','',('运行结果已归并；公开发布尚需验证。' if final else '阶段报告：实验未全部完成，当前数字不可视为完整矩阵。'),'','本轮固定 P@80 与 P+S@80，L30/rank4；R1 只增加拒绝条件。R2_VERIFY24_EXTENSION 已暴露，无独立 CONFIRM，所有结论均为探索性。','R2 历史更正：VERIFY T1L 最终 JSON 为 B0/P 2/2，旧文案 0/2 错误；2/2 不构成稳健保持证据。','',f"评分：{coverage['scored']}/{coverage['expected_requests']} 唯一请求；missing={coverage['MISSING']}。成功项不重判，失败项永久隔离。",'','|面板/模式/前缀|实验臂|任务|分子/分母|edit-macro|missing|','|---|---|---|---|---|---|']
 for name,obj in public.items():
  for s in obj['summary']:
   pct='null' if s['edit_macro'] is None else f"{s['edit_macro']*100:.2f}%"
   lines.append(f"|{name}|{s['arm']}|{s['task']}|{s['numerator']}/{s['denominator']}|{pct}|{s['missing']}|")
 lines+=['','主效应和交互见 RESULTS_PUBLIC.json：S_R0、routing_P、routing_PS、interaction。置信区间按编辑重采样；来源聚类仅为敏感性分析。缺失上下界不是置信区间。','标准 T2L 与 T2L_PRESSURE 分开。负样本没有默认正确专家；激活其他专家不自动定义为误路由。FORCED_ON 只用于诊断。','A48（若执行）复用同一 VERIFY24 输入，不增加独立病例数。CE/KL 曲线仅描述优化，不证明性能改善。','GPU、Judge 历史消耗已计入本轮累计账本；原始医疗材料、逐题变化、模型权重与执行绑定只保留在私有运行目录。']
 (ROOT/'public/FINAL_REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
 return public
if __name__=='__main__':report(final='--final' in __import__('sys').argv)
