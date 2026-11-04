"""Merge validated supplemental judgments without changing the original report."""
import json, os, sys
from pathlib import Path
w=Path(os.environ['WORKTREE']); r=Path(os.environ['LOCAL_SCORE_ROOT']); prep=Path(os.environ['PREP_ROOT'])
sys.path.insert(0,str(w/'experiments/med_medinit_101_v1'))
import summary

def load(p):return json.loads(p.read_text())
for name in ('WORKER_EXIT.json','GUARD_EXIT.json'):
    if load(r/name)['code']!=0: raise ValueError('score process failed: '+name)
frozen=load(r/'SUPPLEMENT_INPUT.private.json'); completed=load(r/'COMPLETE.json'); new=load(r/'DECISIONS.json')
keys=set(frozen['payloads']);valid=new['decisions'];unknown=set(new['unknown'])
if set(valid)&unknown or set(valid)|unknown!=keys or completed['target_payloads']!=97:raise ValueError('score coverage differs')
base=load(prep/'BASELINE.private.json');decisions=base['decisions'];
if set(decisions)&keys:raise ValueError('would overwrite previous decision')
fields=('core_correct','full_supported','unsupported_expansion','extra_contradiction')
for k,v in valid.items():
    if v.get('opaque_query_id')!=k or any(type(v.get(f)) is not bool for f in fields):raise ValueError('invalid response')
    if v['full_supported'] and (not v['core_correct'] or v['unsupported_expansion'] or v['extra_contradiction']):raise ValueError('invalid full decision')
roles=load(w/'.private/med_full_d150_v1_20261102/EVALUATION_ROLES_PRIVATE.json')
full=summary.compare(base['consumers'],{**decisions,**valid},roles,base['routing'])
conflicts=set(frozen['categories']['conflict_unknown'])
sensitive=summary.compare(base['consumers'],{**decisions,**{k:v for k,v in valid.items() if k not in conflicts}},roles,base['routing'])
full.update(supplement={'target':97,'new_valid':len(valid),'failed_or_missing':len(unknown),'held_unknown':len(frozen['excluded']),'attempt_items':completed['new_attempt_items'],'historical_attempt_coverage_complete':False,'conflicts_adjudicated':len(conflicts & set(valid))},sensitivity_conflicts_unresolved=sensitive['groups'])
out=w/'reports/med_medinit_101_supplement_v1_20261104';out.mkdir(exist_ok=True)
(out/'RESULTS.json').write_text(json.dumps(full,ensure_ascii=False,indent=2)+'\n')
lines=['# MED 初始化101+49：未知评分补评','','原2148事件、预测、原有效判定和路由不变。历史不可核实载荷至多一次，历史冲突独立裁决；历史尝试账本不完整，不排除此前未留存重复。',f"固定97载荷，新有效{len(valid)}、失败或缺失{len(unknown)}；另有8项按既有禁止重试及已知失败保留UNKNOWN。",'', '| 角色 CORE | MED | MCE | MDU |','|---|---:|---:|---:|']
for role in summary.ROLES:
    arms=full['groups']['all'][role]['core_correct']['arms']
    def fmt(a):
        x=arms[a];v=x['bounds'];return 'NA' if v is None else f"{v[0]:.4f}–{v[1]:.4f}%（未知{x['unknown']}）"
    lines.append('| '+role+' | '+' | '.join(fmt(a) for a in summary.ARMS)+' |')
lines+=['','编辑宏平均；区间为未知赋值边界，不是置信区间。CORE/FULL、路由和owner分层、配对增损及冲突保守敏感性见RESULTS.json。旧冲突原证据保留；已曝光开发集，T1L仅2事件，不自动替换原模型。']
(out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
(out/'EXECUTION_AUDIT.json').write_text(json.dumps({'status':completed['status'],'frozen_events':2148,'target_payloads':97,'attempt_items':completed['new_attempt_items'],'new_valid':len(valid),'failed_or_missing':len(unknown),'held_unknown':8,'automatic_retry':False,'model':'gpt-6-astra','reasoning':'medium','immutable_snapshot':'unknown','training':0,'generation':0,'worker_exit':0,'guard_exit':0,'history_coverage_complete':False},ensure_ascii=False,indent=2)+'\n')
(r/'REPORT_COMPLETE.json').write_text(json.dumps({'status':completed['status'],'valid':len(valid),'unknown':len(unknown),'held':8}))
