"""Frozen-mask aggregates; partial cohorts and absent scores are never complete."""
import json,time,random
from collections import defaultdict
from statistics import mean
from resources import ROOT,read,write
from metrics_r2 import summarize,retention
from paired_stats import interaction,seed_aggregate,supports

def coverage():
 p=ROOT/'private/judge';pending={f.stem for f in (p/'pending').glob('*.json')};scored={f.stem for f in (p/'scores').glob('*.json')};d=read(ROOT/'RESOURCE_LEDGER.json');failed=set(read(p/'JUDGE_MISSING_LOCK.json')['keys'])-scored;flight={k for a in d['judge_attempts'] if a['status']=='RESERVED' for k in a.get('keys',[])}-scored-failed
 rows=[r for f in (ROOT/'jobs').rglob('CONSUMERS.json') for r in read(f)];needed={r[k] for r in rows for k in ['judge_key','base_judge_key']}
 c=dict(consumers=len(rows),unique_execution_inputs=len({r['execution_id'] for r in rows}),current_required_requests=len(needed),SCORED=len(needed&scored),FAILED_NO_RETRY=len(needed&failed),IN_FLIGHT=len(needed&flight),UNSUBMITTED=len(needed-scored-failed-flight),MISSING=len(needed-scored),historical_and_current_pending=len(pending),remaining_submission_attempt_items=6000-d['judge_submission_attempt_items'])
 write(ROOT/'public/SCORING_COVERAGE.json',c);return c

def report(final=False):
 scores={p.stem:read(p)['is_correct'] for p in (ROOT/'private/judge/scores').glob('*.json')};masks={(x['edit'],x['task'],x['query_id']):x['base_correct'] for x in read(ROOT/'private/BASE_MASKS.json')['rows']}
 pressure={r['query_id'] for name in ['PRESSURE_VALIDATION_FROZEN.json','PRESSURE_TEST_FROZEN.json'] for r in read(ROOT/'private'/name)['rows']};groups=defaultdict(list);allrows=[];parity=[];supportrows=[];seen={}
 for p in (ROOT/'jobs').rglob('CONSUMERS.json'):
  for r in read(p):
   identity=tuple(r[k] for k in ['seed','arm','mode','prefix','edit','task','query_id'])
   if identity in seen:
    assert (r['judge_key'],r['base_judge_key'])==seen[identity],'Conflicting repeated consumer; quarantine before aggregation'
    continue
   seen[identity]=(r['judge_key'],r['base_judge_key'])
   if r['task'].startswith(('G_','U_')):supportrows.append(r);continue
   k=(r['edit'],r['task'],r['query_id']);b=masks.get(k,True if r['task']=='T2L_PRESSURE' and r['query_id'] in pressure else None);oldkey=r['base_judge_key'];fixed='frozen:'+json.dumps(k)
   scores[fixed]=b;r=dict(r,base_judge_key=fixed);allrows.append(r);parity.append((b,scores.get(oldkey)));groups[(r['seed'],'DEV24' if r['order']<=24 else 'REG24',r['mode'],r['prefix'])].append(r)
 for (seed,panel,mode,prefix),rows in list(groups.items()):
  if seed==20260927 and panel=='DEV24':
   fixed=read(ROOT/'EXPERIMENT_LOCK.json')['stage_diagnostics_orders'];groups[(seed,'OPTIONAL_FIXED_DEV4',mode,prefix)]=[r for r in rows if r['order'] in fixed]
 result={}
 for (seed,panel,mode,prefix),rows in groups.items():
  key=f'{seed}_{panel}_{mode}_{prefix}';summary=summarize(rows,scores)
  for s in summary:s['unique_inputs']=len({r['input_id'] for r in rows if r['arm']==s['arm'] and r['task']==s['task'] and scores.get(r['base_judge_key']) is retention(s['task'])});s['expected_edits']=(4 if panel=='OPTIONAL_FIXED_DEV4' else 24) if mode=='single' else prefix
  comparisons={}
  for a,b in [('M0','M1'),('M1','M2'),('M2','M3'),('M1','M3'),('M1','M4'),('M3','M5'),('M4','M5'),('M1','M6'),('M6','M7'),('M1','M1_STRUCT'),('M4','M4_STRUCT'),('M1','M1_R8'),('M4','M4_R8')]:
   aa={(r['edit'],r['task'],r['query_id']):r for r in rows if r['arm']==a};bb={(r['edit'],r['task'],r['query_id']):r for r in rows if r['arm']==b}
   if not aa or set(aa)!=set(bb):comparisons[a+'_'+b]={'status':'UNPAIRED_COHORT'};continue
   bytask=defaultdict(list)
   for k,r in aa.items():
    base=scores.get(r['base_judge_key']);x=scores.get(r['judge_key']);y=scores.get(bb[k]['judge_key'])
    if base is retention(r['task']):bytask[r['task']].append(dict(edit=r['edit'],source=r['source_group'],delta=int(y)-int(x) if x is not None and y is not None else None))
   stats={}
   for task,vs in bytask.items():
    missing=sum(v['delta'] is None for v in vs);ed=defaultdict(list);src=defaultdict(list)
    for v in vs:ed[v['edit']].append(v['delta']);src[v['source']].append(v)
    rng=random.Random(20260927);means=[mean(v) for v in ed.values()] if not missing else [];boot=sorted(mean(rng.choices(means,k=len(means))) for _ in range(1000)) if means else [];cluster=[]
    if means:
     for _ in range(1000):
      e=defaultdict(list)
      for sg in rng.choices(list(src),k=len(src)):
       for v in src[sg]:e[v['edit']].append(v['delta'])
      cluster.append(mean(mean(v) for v in e.values()))
     cluster.sort()
    stats[task]=dict(missing=missing,denominator=len(vs),edits=len(ed),source_groups=len(src),delta_edit_macro=mean(means) if means else None,edit_CI95=[boot[24],boot[974]] if boot else None,source_cluster_CI95=[cluster[24],cluster[974]] if cluster else None)
   comparisons[a+'_'+b]=dict(status='NOT_ESTIMABLE' if not stats else 'MISSING' if any(s['missing'] for s in stats.values()) else 'COMPLETE_OBSERVED_COHORT',observed_inputs_paired=True,planned_panel_complete=len({r['edit'] for r in aa.values()})==((4 if panel=='OPTIONAL_FIXED_DEV4' else 24) if mode=='single' else prefix),scoring_complete=bool(stats) and not any(s['missing'] for s in stats.values()),metric_estimable=any(s['denominator'] and not s['missing'] for s in stats.values()),metrics=stats)
  result[key]=dict(summary=summary,paired=comparisons,interaction=interaction(rows,scores,retention),qualification_only=mode=='sequential' and prefix not in [12,24],exploratory=True)
 mask_path=ROOT/'private/SUPPORT_BASE_MASKS.json';support_masks=read(mask_path) if mask_path.exists() else {}
 for r in supportrows:
  k=r['base_judge_key'];v=scores.get(k)
  if k in support_masks:assert support_masks[k]==v,'Frozen current Base score changed'
  elif v is not None:support_masks[k]=v
 write(mask_path,support_masks)
 write(ROOT/'public/SUPPORT_METRICS.json',supports(supportrows,scores,read(ROOT/'EXPERIMENT_LOCK.json')['stage_diagnostics_orders']));write(ROOT/'public/THREE_SEED_AGGREGATE.json',seed_aggregate(allrows,scores,retention));
 write(ROOT/'public/CORE_MATRIX_RESULTS.json',result);c=coverage();d=read(ROOT/'RESOURCE_LEDGER.json');active=sum(time.time()-s['started_epoch'] for s in d['gpu_sessions'] if not s.get('ended_epoch'))
 write(ROOT/'public/RESOURCE_LEDGER_AGGREGATE.json',dict(historical_gpu_seconds=d['historical_gpu_seconds'],phase_gpu_seconds=d['current_gpu_seconds']+active,phase_limit=None if __import__('resources').limits_waived() else 72*3600,original_phase_limit=72*3600,wallclock_gpu_limits_waived=__import__('resources').limits_waived(),judge_attempts=d['judge_submission_attempt_items'],judge_limit=6000,active_gpu_sessions=sum(not s.get('ended_epoch') for s in d['gpu_sessions'])))
 kd=[read(p) for p in (ROOT/'private/kd').glob('*.json')]
 write(ROOT/'public/TEACHER_QUALITY_COVERAGE.json',dict(teacher_consumers=len(kd),G_fit_items=sum(len(x['quality']) for x in kd),qualified=sum(sum(x['quality']) for x in kd),normalization='Mean over qualified subset at lambda 0.10; all G_fit retain CE; zero KD if no qualified input'))
 costs=defaultdict(list);curves=defaultdict(list)
 for p in (ROOT/'private/curves').rglob('*.json'):
  v=read(p);i=v['identity'];costs[(i['kind'],i['method'],i['stage'])].append(v)
  for point in v['curve']:curves[(i['kind'],i['method'],i['stage'],point['step'])].append(point)
 write(ROOT/'public/TRAINING_AND_INFERENCE_COSTS.json',dict(stages=[dict(structure=k[0],method=k[1],stage=k[2],completed=len(vs),steps_sum=sum(v['steps'] for v in vs),seconds_sum=sum(v['seconds'] for v in vs),CE_forward_calls=sum(v['forwards'] for v in vs),CE_target_tokens=sum(v['target_tokens'] for v in vs)) for k,vs in sorted(costs.items())],note='CE call counters exclude KL calls. GPU residency includes teacher preparation, diagnostics and generation; no equal-compute claim.'))
 curve_rows=[]
 for k,vs in sorted(curves.items()):
  item=dict(structure=k[0],method=k[1],stage=k[2],step=k[3],observations=len(vs),CE_components=[mean(v['CE'][i] for v in vs) for i in range(len(vs[0]['CE']))])
  for metric in ['U_KL','D_KL','CE_gradient_norm','protection_gradient_norm','CE_U_cosine','preclip_norm','postclip_norm','update_norm','function_space_update_norm']:
   values=[v[metric] for v in vs if v.get(metric) is not None];item['CE_protection_cosine' if metric=='CE_U_cosine' else metric]=mean(values) if values else None
  curve_rows.append(item)
 write(ROOT/'public/CURVES_AGGREGATE.json',dict(training_diagnostics_not_performance=True,protection_cosine='CE versus total U plus optional D gradient; legacy raw CE_U_cosine is not U-only for distilled arms',points=curve_rows))
 counts=[read(p) for p in (ROOT/'jobs').rglob('COMPUTE_COUNTS.json')]
 write(ROOT/'public/MODEL_CALL_COUNTS.json',dict(instrumented_completed_job_groups=len(counts),model_calls=sum(x['model_calls'] for x in counts),input_positions=sum(x['input_positions'] for x in counts),coverage='Only instrumented attempts; earlier pilot/REG workers lack these total counters, not imputed'))
 singles={(r['seed'],r['arm'],r['edit'],r['query_id']):r for r in allrows if r['mode']=='single'};trans=defaultdict(list)
 for r in allrows:
  if r['mode']=='sequential':
   a=singles.get((r['seed'],r['arm'],r['edit'],r['query_id']))
   if a:trans[(r['seed'],r['arm'],'DEV24' if r['order']<=24 else 'REG24',r['prefix'])].append((a,r))
 write(ROOT/'public/ROUTING_TRANSITIONS.json',[dict(seed=k[0],method=k[1],panel=k[2],prefix=k[3],paired_consumers=len(rs),single_OFF_bank_ON=sum(not a['route']['activated'] and b['route']['activated'] for a,b in rs),changed_expert=sum(a['route']['logical_edit_id']!=b['route']['logical_edit_id'] for a,b in rs),negative_activation_damage=sum(retention(b['task']) and b['route']['activated'] and scores.get(b['base_judge_key']) is True and scores.get(b['judge_key']) is False for a,b in rs),positive_rejection=sum(not retention(b['task']) and not b['route']['activated'] for a,b in rs)) for k,rs in trans.items()])
 write(ROOT/'public/BASE_MASK_AUDIT.json',dict(primary='Original frozen Base masks; current backend Base only a parity audit',known_pairs=sum(a is not None and b is not None for a,b in parity),changed=sum(a!=b for a,b in parity if a is not None and b is not None),unknown_masks=sum(a is None for a,b in parity)))
 audit=read(ROOT/'fix/pr5_v1/ROLE_AUDIT_REPAIRED.json') if (ROOT/'fix/pr5_v1/ROLE_AUDIT_REPAIRED.json').exists() else read(ROOT/'fix/pr5_v1/ROLE_AUDIT_PRIVATE.json') if (ROOT/'fix/pr5_v1/ROLE_AUDIT_PRIVATE.json').exists() else {};write(ROOT/'public/ROLE_AUDIT_STATUS.json',{k:v for k,v in audit.items() if k not in ['full_input_collisions','official_missing']});
 q=read(ROOT/'QUEUE.json');complete=sum(j['status']=='COMPLETE' for j in q)
 text=f'# MedTRACE 追加阶段 E1 报告\n\n状态：{"阶段已收口" if final else "运行中，非最终报告"}。已完成 {complete}/{len(q)} 个已排入队列的任务组。\n\n本轮仅为 EXPOSED_REGRESSION 探索性比较，没有独立 CONFIRM；重复种子不增加独立病例数。完整目标矩阵与未执行项见 QUEUE_AMENDMENT.json。\n\n评分：当前消费者 {c["consumers"]}，缺评分请求 {c["MISSING"]}，累计提交 {d["judge_submission_attempt_items"]}/6000。缺失项与空分母不解释为改善。\n\n主指标使用历史冻结 Base masks，Blackwell 新 Base 输出仅用于数值一致性审计。有效编辑、分母、macro/micro、配对区间见 CORE_MATRIX_RESULTS.json；未配对/不完整队列不能跨样本量比较。\n\n机械检查和 loss 下降不是性能收益。本阶段教师兼容修复及验证见私有IMPORTED_TEACHER_REPAIR.json；未改写父阶段证据。\n'
 text+='\n角色修正继承PARENT_HANDOFF_AUDIT.json。辅助支持诊断只覆盖两个REG canary编辑。E1单种子，三种子汇总不作为本阶段结论；E2及后续块另行预算准入。\n'
 __import__('storage').Store(ROOT).write('public/EXTENSION_STAGE_REPORT_ZH.md',text.encode())
 return c
if __name__=='__main__':report()
