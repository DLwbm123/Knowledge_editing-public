"""R4 complete-panel selection and paired dose reports; no routing claims without CAL."""
import os,json,time,random
from pathlib import Path
from collections import defaultdict
from statistics import mean
ROOT=Path(os.environ['RUN_ROOT'])
from budget import read,write
from metrics_r2 import summarize
from report_r3 import paired,route_summary
ARMS={'P':0.,'B125':.125,'B25':.25,'P+S':.5}
def collect():
 cohorts=defaultdict(list)
 for job in (ROOT/'jobs').iterdir():
  if not (job/'STATUS.json').exists():continue
  for f in job.glob('*/p*/CONSUMERS.json'):
   rows=read(f)
   if not rows:continue
   name=job.name.removeprefix('reuse-');panel='DEV24' if name.startswith('dev') else 'R2_VERIFY24_REGRESSION'
   cohorts[(panel,rows[0]['mode'],rows[0]['prefix'])]+=rows
 # No repeated consumer may artificially inflate N.
 for key,rows in cohorts.items():
  ids=[(r['arm'],r['edit'],r['task'],r['query_id']) for r in rows];assert len(ids)==len(set(ids)),key
 return cohorts

def select():
 rows=collect().get(('DEV24','single',1),[]);scores={p.stem:read(p)['is_correct'] for p in (ROOT/'private/judge/scores').glob('*.json')};summary=summarize(rows,scores)
 baseline={s['task']:s for s in summary if s['arm']=='P@80_R0'};decisions=[]
 for arm,beta in ARMS.items():
  pair=paired(rows,scores,['P@80_R0',arm+'@80_R0'],[-1,1]);ss={s['task']:s for s in summary if s['arm']==arm+'@80_R0'}
  complete=pair['status']=='COMPLETE' and all(ss.get(k,{}).get('status')=='COMPLETE' for k in ['T0','T1G','T2G','T2L_PRESSURE'])
  eligible=complete and pair['metrics']['T0']['right_to_wrong']==0 and all(pair['metrics'][k]['delta_edit_macro']>=-.01-1e-12 for k in ['T1G','T2G'])
  decisions.append(dict(writer=arm,beta=beta,eligible=eligible,complete=complete,pressure=ss.get('T2L_PRESSURE',{}).get('edit_macro'),constraint_deltas={k:pair.get('metrics',{}).get(k) for k in ['T0','T1G','T2G']}))
 if any(not d['complete'] for d in decisions):raise RuntimeError('Incomplete DEV dose comparison; cannot select using scored subset')
 best=sorted([d for d in decisions if d['eligible']],key=lambda d:(-d['pressure'],d['beta']))[0]
 lock=dict(status='LOCKED',selected_writer=best['writer'],selected_beta=best['beta'],selected_steps=80,route='R0',reason='highest DEV24 pressure macro among constraint-qualified arms; tie smaller beta',decisions=decisions,formal_role='R2_VERIFY24_REGRESSION',independent_CONFIRM=False,epoch=time.time())
 if (ROOT/'SELECTION_LOCK.json').exists():
  prior=read(ROOT/'SELECTION_LOCK.json');assert prior['selected_writer']==best['writer'];return prior
 write(ROOT/'SELECTION_LOCK.json',lock);write(ROOT/'public/SELECTION_LOCK.json',lock);return lock

def report(final=False):
 cohorts=collect();scores={p.stem:read(p)['is_correct'] for p in (ROOT/'private/judge/scores').glob('*.json')};results={};total=[]
 for (panel,mode,prefix),rows in sorted(cohorts.items()):
  total+=rows;name=f'{panel}_{mode}_{prefix}';ss=summarize(rows,scores)
  for s in ss:
   rr=[r for r in rows if r['arm']==s['arm'] and r['task']==s['task']];s['unique_inputs']=len({r['input_id'] for r in rr})
  pairs={}
  for arm in ['B125','B25','P+S']:
   if not any(r['arm']==arm+'@80_R0' for r in rows):continue
   p=paired(rows,scores,['P@80_R0',arm+'@80_R0'],[-1,1]);changes=p.pop('private_changes',[]);write(ROOT/'private/reports'/f'{name}_{arm}_changes.json',changes);pairs[arm]=p
  results[name]=dict(summary=ss,paired=pairs,routes=route_summary(rows,scores))
 write(ROOT/'public/RESULTS_PUBLIC.json',results)
 d=read(ROOT/'RESOURCE_LEDGER.json');active=sum(time.time()-s['started_epoch'] for s in d['gpu_sessions'] if not s.get('ended_epoch'));used=d['gpu_seconds_used']+active
 write(ROOT/'public/RESOURCE_LEDGER_AGGREGATE.json',dict(historical_gpu_seconds=d['historical_gpu_seconds'],new_gpu_seconds=used-d['historical_gpu_seconds'],cumulative_gpu_seconds=used,remaining_gpu_seconds=57600-used,historical_Judge_attempts=d['historical_judge_attempt_items'],new_Judge_attempts=d['judge_submission_attempt_items']-d['historical_judge_attempt_items'],cumulative_Judge_attempts=d['judge_submission_attempt_items'],remaining_Judge_attempts=6000-d['judge_submission_attempt_items'],active_sessions=sum(not s.get('ended_epoch') for s in d['gpu_sessions'])))
 pending={p.stem for p in (ROOT/'private/judge/pending').glob('*.json')};failed=set(read(ROOT/'private/judge/JUDGE_MISSING_LOCK.json')['keys'])-scores.keys();flight={k for a in d['judge_attempts'] if a['status']=='RESERVED' for k in a.get('keys',[])}-scores.keys()-failed
 write(ROOT/'public/SCORING_COVERAGE.json',dict(expected=len(pending),scored=len(pending&scores.keys()),FAILED_NO_RETRY=len(pending&failed),IN_FLIGHT=len(pending&flight),UNSUBMITTED=len(pending-scores.keys()-failed-flight),MISSING=len(pending-scores.keys()),formal_consumers=len(total),formal_missing_consumers=sum(r['judge_key'] not in scores for r in total),inherited_failed_A48_not_retried=True))
 if final:
  import torch
  gradients=[];retained=[]
  for p in (ROOT/'runs/s20260925').glob('*/*/TRAINING_COMPLETE.json'):
   receipt=read(p);arm=p.parent.parent.name;order=int(p.parent.name[1:]);point=p.parent/'private/edits'/p.parent.name/'C_NO_H/step-80.pt';s=torch.load(point,map_location='cpu',weights_only=True);assert s['step']==80
   curve=s['curve'];cs=[c['CE_U_cosine'] for c in curve if c.get('CE_U_cosine') is not None]
   gradients.append(dict(writer=arm,edit_order=order,CE_U_cosine_mean=mean(cs) if cs else None,negative_cosine_fraction=mean(v<0 for v in cs) if cs else None,preclip_norm_mean=mean(c['grad_norm'] for c in curve),postclip_norm_mean=mean(c['clipped_grad_norm'] for c in curve),actual_update_norm_mean=mean(c['update_norm'] for c in curve)))
   import hashlib
   retained.append(dict(writer=arm,edit_order=order,actual_steps=80,sha256=hashlib.file_digest(point.open('rb'),'sha256').hexdigest(),bytes=point.stat().st_size))
  write(ROOT/'public/GRADIENT_DISTRIBUTION.json',dict(per_edit=gradients,note='CE/U cosine uses raw gradients; Adam-preconditioned actual update magnitude is separate and not a causal explanation of evaluation errors'))
  write(ROOT/'public/RETAINED_BANKS.json',dict(new_experts=retained,historical_matching_P_PS_banks='R3 and R2 remain unchanged',P_W0_STEP0_and_optimizer_RNG_retained=True))
 lock=read(ROOT/'SELECTION_LOCK.json') if (ROOT/'SELECTION_LOCK.json').exists() else None
 write(ROOT/'public/CONFIG_PUBLIC.json',dict(review_commit='ed375b4e879ce609bc74f9657f743a42c6538fd5',beta_grid=[0,.125,.25,.5],new_arms=[.125,.25],steps=80,layer='L30',rank=4,dtype='float16',lambda_U=.01,seed=20260925,teacher='all experts off Base; full vocabulary token mean KL(Base||student); complete generated prefix max1024',sampler='unchanged R2 stateless uniform new U per step',selected=lock['selected_writer'] if lock else None,route='R0',route_gain='UNSUPPORTED_CAL_PLUS',matrix_collapse='R*=R0; do not manufacture duplicate routing arms',independent_CONFIRM=False,formal_role='R2_VERIFY24_REGRESSION',manifest=read(ROOT/'RUN_MANIFEST.json')))
 lines=['# MedTRACE TextJoint-R4 中文报告','',('训练/生成队列已完成，结果待公开交付核验。' if final else '阶段报告：队列尚未全部完成。'),'','本轮只新增 beta=0.125/0.25，复用完整绑定的 P80(beta=0)与P+S80(beta=0.5)。总KL权重0.01不变；共享P-W0、支持与采样器，L30/rank4/80步。','CAL_PLUS 合格跨图正例不足，因此 Rcal/Rscope 未获资格，保留R0；这不是标量阈值家族不可行的实验证明。A类只复用了R3 CAL，不能宣称完成了全新的四类校准。','新数据前置盘点发现授权core9目录11088行均存在历史Base暴露；全历史独立性未认证。使用旧VERIFY24作回归，不是新N24或CONFIRM。','',f"DEV冻结选择：{lock['selected_writer'] if lock else '尚未锁定'}。点估计门槛通过不等于统计非劣证明。",'','|面板/模式/前缀|臂|任务|分子/分母|edit-macro|missing|','|---|---|---|---|---|---|']
 for name,obj in results.items():
  for s in obj['summary']:
   pct='null' if s['edit_macro'] is None else f"{100*s['edit_macro']:.2f}%";lines.append(f"|{name}|{s['arm']}|{s['task']}|{s['numerator']}/{s['denominator']}|{pct}|{s['missing']}|")
 lines+=['','配对计数、edit bootstrap与来源聚类敏感性见RESULTS_PUBLIC.json；逐题变化只保留私有。标准T2L与压力保持分开，空分母不算PASS。','同路由比较writer；R*=R0时路由差值和交互退化为0，不重复生成或宣称路由创新。历史R3 A48失败请求继续missing，不自动补评。','逐编辑梯度余弦分布见GRADIENT_DISTRIBUTION.json。原始梯度与Adam预条件后的实际更新分开；loss下降不代表性能提升。','历史报告、选择锁、模型与账本均未改写。所有结果仍为探索性，seq60%不是安全阈值。']
 (ROOT/'public/FINAL_REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
 return results
if __name__=='__main__':report(final='--final' in __import__('sys').argv)
