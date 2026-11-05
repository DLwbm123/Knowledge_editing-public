import json,os,sys
from pathlib import Path
w=Path(os.environ['WORKTREE']);root=Path(os.environ['LOCAL_SCORE_ROOT']);prep=Path(os.environ['PREP_ROOT'])
sys.path.insert(0,str(w/'experiments/med_mdu_boundary_average_v1'))
import score_driver as base
read=lambda p:json.loads(p.read_text())
for name in ('GUARD_EXIT.json','WORKER_EXIT.json'):
 if read(root/name)['code']!=0:raise ValueError('scoring process failed')
frozen=read(root/'SUPPLEMENT_INPUT.private.json');response=read(root/'DECISIONS.json');completion=read(root/'COMPLETE.json');b=read(prep/'BASELINE.private.json')
targets=set(frozen['payloads']);new=response['decisions'];unknown=set(response['unknown']);decisions=b['decisions']
if len(targets)!=8 or set(new)&unknown or set(new)|unknown!=targets or set(decisions)&targets or completion['new_attempt_items']>8:raise ValueError('partition mismatch')
fields=('core_correct','full_supported','unsupported_expansion','extra_contradiction')
for k,v in new.items():
 if v['opaque_query_id']!=k or any(type(v.get(f)) is not bool for f in fields) or v['full_supported'] and (not v['core_correct'] or v['unsupported_expansion'] or v['extra_contradiction']):raise ValueError('bad verdict')
decisions.update(new);panels=base.aggregate(b['rows'],decisions,b['roles']); conservative=base.aggregate(b['rows'],{k:v for k,v in decisions.items() if k not in set(b['conflicts'])},b['roles'])
old=read(w/'reports/med_mdu_boundary_average_v1_20261105/RESULTS.json')
for role in base.ROLES:
 for field in ('core_correct','full_supported'):
  for arm in ('MED','MCE','MDU'):
   if panels[role][field]['arms'][arm]!=old['panels'][role][field]['arms'][arm]:raise ValueError('old arms changed')
result={'status':'COMPLETE' if not unknown else 'INCOMPLETE','events':392,'panels':panels,'conflicts_unresolved':conservative,'scoring':{'frozen_unique':8,'attempt_items':completion['new_attempt_items'],'new_valid':len(new),'unknown':len(unknown),'automatic_retry':False,'history_coverage_complete':False}}
out=w/'reports/med_mdu_boundary_supplement_v1_20261105'
with (out/'RESULTS.json').open('x') as f:json.dump(result,f,indent=2,ensure_ascii=False)
lines=['# 边界平均 MDU 未知评分补全','','固定392共同路由开发事件，8个唯一未知载荷各最多一次补评，既有有效判断、预测与原报告保留。',f"实际尝试{completion['new_attempt_items']}、有效{len(new)}、未知{len(unknown)}；历史账本不完整，不排除更早未留存重复。",'', '| 指标 | MED | MCE | 原MDU | 边界平均MDU |','|---|---:|---:|---:|---:|']
for role in base.ROLES:
 for field,label in (('core_correct','CORE'),('full_supported','FULL')):
  def fmt(a):
   x=panels[role][field]['arms'][a];b=x['bounds']
   return 'NA' if b is None else (f'{b[0]:.4f}%' if b[0]==b[1] else f'{b[0]:.4f}–{b[1]:.4f}%')+f"（未知{x['unknown']}）"
  lines.append('| '+role+' '+label+' | '+' | '.join(fmt(a) for a in ('MED','MCE','MDU','BOUNDARY'))+' |')
for role in ('T2G','T2L'):
 z=panels[role]['core_correct'];pair=z['pairs']['MDU_to_BOUNDARY'];a=z['arms']['MDU']
 lines.append(f"\n{role}：{a['events']}事件/{a['edits']}编辑；相对原MDU，修复{pair['gain']}、损失{pair['loss']}、双对{pair['both_correct']}、双错{pair['both_incorrect']}、未知{pair['unknown']}。")
lines+=['','指标为编辑等权宏平均；未知赋值界限非置信区间。T1L无支持；该面板已曝光，边界切分不代表事实单元，不能作为独立测试。既有四项历史冲突保守敏感性见结果文件；历史记录不完整，可能有未留存重复。']
(out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
audit={'attempt_items':completion['new_attempt_items'],'valid':len(new),'unknown':len(unknown),'worker_exit':0,'guard_exit':0,'baseline_reproduced':True,'old_comparator_panels_unchanged':True,'new_training':0,'new_generation':0,'automatic_retry':False,'historical_attempt_coverage_complete':False}
(out/'EXECUTION_AUDIT.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
(root/'REPORT_COMPLETE.json').write_text(json.dumps({'status':result['status'],'unknown':len(unknown)}))
