"""All nodes and fixed comparisons; select one common length using CAL only."""
from collections import defaultdict,Counter
import csv
import json
import os
from pathlib import Path
import random
import sqlite3
import time
from common import RUN,read,write,digest,used

def mean(x):return sum(x)/len(x) if x else None

def score_bounds(coeff,scores):
    known=sum(v*scores[k] for k,v in coeff.items() if scores.get(k) is not None)
    return [known+sum(min(0,v) for k,v in coeff.items() if scores.get(k) is None),known+sum(max(0,v) for k,v in coeff.items() if scores.get(k) is None)]
def combine(dicts):
    out=defaultdict(float)
    for d in dicts:
        for k,v in d.items():out[k]+=v
    return {k:v for k,v in out.items() if abs(v)>1e-15}
def bootstrap(coefficients,scores,groups):
    es=list(coefficients);clusters=defaultdict(list)
    for e in es:clusters[groups[e]].append(e)
    def sample(packages):
        rng=random.Random(20260912);low=[];high=[]
        for _ in range(10000):
            draw=rng.choices(packages,k=len(packages));selected=[e for pack in draw for e in pack];c=combine([{k:v/len(selected) for k,v in coefficients[e].items()} for e in selected]);a,b=score_bounds(c,scores);low.append(a);high.append(b)
        return [sorted(low)[249]*100,sorted(high)[9749]*100]
    if not es:return dict(edit_ci=None,source_ci=None)
    return dict(edit_ci=sample([[e] for e in es]),source_ci=sample(list(clusters.values())),source_groups=len(clusters),interpretation='Development uncertainty only; few source groups and shared missing keys retained')
def load():
    db=sqlite3.connect('file:'+str(RUN/'private/judge_common/queue.sqlite')+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    scores={r['key']:r['correct'] for r in db.execute('SELECT * FROM payload')};records=[]
    for c in db.execute('SELECT * FROM consumer'):
        d=read(c['path']);assert digest(d)==c['output_binding'];records.append((d,c['payload_key']))
    db.close();return records,scores

def panel(records,scores,cohort,arm,node,mode,task,ts,prefix=1):
    ledger=read(RUN/'private/EVAL_LEDGER.json');byqid=defaultdict(list)
    for d,key in records:
        p=d['binding']['phase'];q=d['binding']['input']
        if p['arm']==arm and p['node']==node and d['binding']['mode']==mode and p['prefix']==prefix:byqid[q['query_id']].append((d,key))
    rows=[];coeff={};peredit={};groups={}
    for t in ts:
        editrows=[]
        qids=list(dict.fromkeys(q for e in t['events'] if e['task']==task for q in e['all_probe_query_ids']))
        if task!='T0':qids=[q for q in qids if ledger['Base_correctness'][q]==task.endswith('L')]
        for qid in qids:
            candidates=[(d,k) for d,k in byqid[qid] if (mode=='bank_R0' or d['binding']['owner_order']==t['order'])]
            if mode.startswith('bank') and task.endswith('L'):candidates=[(d,k) for d,k in candidates if not d['active_target']]
            editrows.extend((d,k) for d,k in candidates)
        if not editrows:continue
        c=Counter(k for d,k in editrows);coeff[t['edit_id']]={k:n/len(editrows) for k,n in c.items()};bounds=score_bounds(coeff[t['edit_id']],scores)
        peredit[t['edit_id']]=dict(bounds=[x*100 for x in bounds],observations=len(editrows));groups[t['edit_id']]=t['native']['source_group'];rows+=editrows
    macro=score_bounds(combine([{k:v/len(coeff) for k,v in c.items()} for c in coeff.values()]),scores) if coeff else [None,None]
    missing=sum(scores[k] is None for d,k in rows);micro=mean([scores[k] for d,k in rows]) if rows and not missing else None
    return dict(cohort=cohort,arm=arm,node=node,mode=mode,prefix=prefix,task=task,macro=macro[0]*100 if coeff and not missing else None,macro_bounds=[x*100 if x is not None else None for x in macro],micro=micro*100 if micro is not None else None,known_correct=sum(scores[k]==1 for d,k in rows),observations=len(rows),missing=missing,edit_units=len(coeff),route_ON=sum(d['effective_expert'] is not None for d,k in rows),qualification='HISTORICAL_DEVELOPMENT_DIAGNOSTIC' if task.endswith('L') else 'ORIGINAL_FIXED_MASK'),coeff,groups

def summarize_u(records,cohort):
    tasks=read(RUN/'private/QUEUES.json')['tasks'];owners={t['order'] for t in tasks if t['cohort']==cohort};data=defaultdict(list)
    for d,_ in records:
        b=d['binding'];p=b['phase'];q=b['input']
        if b['owner_order'] in owners and q.get('role') in ('FIT','CAL','CHECK') and d['U_KL'] is not None:
            data[(p['arm'],p['node'],b['mode'],p['prefix'],q['role'])].append(d)
    result=[]
    for (arm,node,mode,prefix,role),ds in sorted(data.items()):
        units=defaultdict(list)
        for d in ds:units[(d['binding']['owner_order'],d['binding']['input']['source_group'])].append(d)
        sources={x['binding']['input']['source_group'] for x in ds};edits={x['binding']['owner_order'] for x in ds}
        result.append(dict(arm=arm,node=node,mode=mode,prefix=prefix,role=role,source_mean_KL=mean([mean([d['U_KL'] for d in xs]) for xs in units.values()]),Base_token_consistency=mean([mean([int(d['Base_token_consistency']) for d in xs]) for xs in units.values()]),source_groups=len(sources),images=len(sources),edit_source_relations=len(units),edits=len(edits),repeated_observations=len(ds),route_ON=sum(d['effective_expert'] is not None for d in ds),patient='UNKNOWN',accuracy_label='Behavior consistency, not medical accuracy',independence='Repeated experts/seeds do not add source groups'))
    return result

def select_length(metrics,u):
    if (RUN/'private/T_STAR.json').exists():return read(RUN/'private/T_STAR.json')
    err=read(RUN/'private/GPU_MECHANICAL.json')['numerical_repeat_error'];near=max(1e-8,10*err)
    def m(a,n,t):return next((x['macro'] for x in metrics if x['arm']==a and x['node']==n and x['task']==t and x['mode']=='single_R0'),None)
    def kl(a,n):return next((x['source_mean_KL'] for x in u if x['arm']==a and x['node']==n and x['role']=='CAL' and x['mode']=='single_R0'),None)
    w=kl('FROZEN_W0',0);rows=[]
    for n in (40,80,160,320):
        c=kl('CE_U_CLEAN',n);b=kl('CE_ONLY',n);checks={}
        for t,drop in [('T0',0),('T1G',2),('T2G',2)]:
            a,z=m('CE_U_CLEAN',n,t),m('FROZEN_W0',0,t);checks[t]=a is not None and z is not None and a>=z-drop-1e-10
        checks.update(CAL_groups=read(RUN/'PLAN_CONFIG.json')['U']['CAL']>=4,baseline_nonzero=w is not None and w>near,KL_20percent=w is not None and c is not None and w>near and c<=.8*w,better_than_CE_ONLY=b is not None and c is not None and b-c>err,above_repeat_error=w is not None and c is not None and w-c>err)
        rows.append(dict(length=n,checks=checks,qualified=all(checks.values()),W0_CAL_KL=w,CE_U_CAL_KL=c,CE_ONLY_CAL_KL=b))
    good=[x['length'] for x in rows if x['qualified']];t=min(good) if good else 80
    d=dict(t_star=t,qualified_lengths=good,status='DEVELOPMENT_RULE_PASSED' if good else 'PRESET_SHORT_ANCHOR_NO_NET_BENEFIT_ESTABLISHED',near_zero_threshold=near,numerical_repeat_error=err,relative_reduction='NOT_APPLICABLE_NEAR_ZERO' if w is not None and w<=near else 'APPLICABLE',candidates=rows,CHECK_read=False,P2_read=False,frozen_epoch=time.time())
    write(RUN/'private/T_STAR.json',d);write(RUN/'public/T_STAR.json',d);return d

def report(cohort):
    records,scores=load();ts=[t for t in read(RUN/'private/QUEUES.json')['tasks'] if t['cohort']==cohort]
    if cohort=='P2':ts=[t for t in ts if (RUN/'private/edits'/t['anonymous_edit']/'s0/COMPLETE.json').exists()]
    panels=[];coefficients={};groups={};keys=set()
    for d,k in records:
        p=d['binding']['phase']
        if d['binding']['owner_order'] in {t['order'] for t in ts} and d['binding']['input'].get('role') is None:keys.add((p['arm'],p['node'],d['binding']['mode'],p['prefix']))
    for a,n,mode,prefix in sorted(keys):
        if mode.startswith('insertion'):continue
        for task in ('T0','T1G','T2G','T1L','T2L'):
            m,c,g=panel(records,scores,cohort,a,n,mode,task,ts[:prefix] if mode.startswith('bank') else ts,prefix);panels.append(m);coefficients[(a,n,mode,prefix,task)]=c;groups.update(g)
    u=summarize_u(records,cohort);contrasts=[]
    pairs=[('CE_ONLY',320,'FROZEN_W0',0),('CE_U_CLEAN',320,'CE_ONLY',320),('CE_U_CLEAN',320,'FROZEN_W0',0)] if cohort=='P1' else [('CE_U_SINGLE',read(RUN/'private/T_STAR.json')['t_star'],'CE_ONLY',read(RUN/'private/T_STAR.json')['t_star']),('CE_U_MULTI',read(RUN/'private/T_STAR.json')['t_star'],'CE_U_SINGLE',read(RUN/'private/T_STAR.json')['t_star']),('CE_U_SINGLE',read(RUN/'private/T_STAR.json')['t_star'],'FROZEN_W0',0),('CE_U_MULTI',read(RUN/'private/T_STAR.json')['t_star'],'FROZEN_W0',0)]
    for a,n,b,nb in pairs:
        for task in ('T0','T1G','T2G'):
            aa=coefficients.get((a,n,'single_R0',1,task),{});bb=coefficients.get((b,nb,'single_R0',1,task),{});es=set(aa)&set(bb);cc={e:combine([aa[e],{k:-v for k,v in bb[e].items()}]) for e in sorted(es)}
            bounds=score_bounds(combine([{k:v/len(cc) for k,v in c.items()} for c in cc.values()]),scores) if cc else [None,None]
            contrasts.append(dict(comparison=a+'-'+b,task=task,node=n,paired_edits=len(cc),delta_bounds_pp=[x*100 if x is not None else None for x in bounds],**bootstrap(cc,scores,groups)))
    result=dict(cohort=cohort,status='COMPLETE' if len(ts)==(8 if cohort=='P1' else 24) else 'PARTIAL',expected_edits=8 if cohort=='P1' else 24,actual_edits=len(ts),seed_slots=3 if cohort=='P1' else 1,independent_units='Edits; seeds are repeated optimization',source_groups=len({t['native']['source_group'] for t in ts}),panels=panels,contrasts=contrasts,U=u,T1L='NA_INDEPENDENT_SCOPE_NOT_QUALIFIED',T2L='NA_INDEPENDENT_SCOPE_NOT_QUALIFIED',all_nodes_reported=True,missing_not_removed=True)
    write(RUN/'public'/(cohort+'_RESULTS.json'),result)
    if cohort=='P1':select_length(panels,u)
    return result

def final():
    p1=read(RUN/'public/P1_RESULTS.json') if (RUN/'public/P1_RESULTS.json').exists() else None;p2=report('P2') if (RUN/'private/judge_common/queue.sqlite').exists() and any((RUN/'private/edits').glob('P2_*/s0/COMPLETE.json')) else None
    ledger=read(RUN/'RESOURCE_LEDGER.json');sessions=ledger['gpu_sessions'];budget=dict(GPU_hours_including_live=used()/3600,caps=read(RUN/'PLAN_CONFIG.json')['caps'],Judge_payload_attempts=ledger['Judge_attempts'],resident_GPU_sessions=len(sessions),all_sessions_ended=all(s.get('ended_epoch') for s in sessions),weights_peak_bytes=ledger.get('weights_observed_peak_bytes',0))
    write(RUN/'public/BUDGET.json',budget)
    rows=[]
    for r in [p1,p2]:
        if r:rows+=r['panels']
    with (RUN/'public/RESULTS.csv').open('w',newline='') as f:
        columns=['cohort','arm','node','mode','prefix','task','macro','micro','known_correct','observations','missing','edit_units','route_ON','qualification'];w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore',lineterminator='\n');w.writeheader();w.writerows(rows)
    bankfiles=list((RUN/'private').glob('BANK_*_COMPLETE.json'));banks=[dict(arm=p.name.removeprefix('BANK_').removesuffix('_COMPLETE.json'),**read(p)) for p in bankfiles]
    write(RUN/'public/BANK_RESULTS.json',dict(status='COMPLETE' if banks and all(b['experts']==24 for b in banks) else 'PARTIAL' if banks else 'NOT_RUN',banks=banks,panels=[x for x in rows if x['mode'].startswith('bank')],real_generation=True,single_outputs_not_spliced=True))
    selection=read(RUN/'private/T_STAR.json') if (RUN/'private/T_STAR.json').exists() else None
    def mainrows(r):return '\n'.join('| '+ ' | '.join(str(x[k]) if x[k] is not None else 'NA' for k in ['arm','node','task','macro','known_correct','observations'])+' |' for x in r['panels'] if x['mode']=='single_R0' and x['task'] in ('T0','T1G','T2G')) if r else '尚未形成完整评分。'
    text='# TT-U 续训价值与保护收益：科学报告\n\n本轮为固定开发实验，不是独立确认。W0是已经完成编辑的TT专家；CE-only与带U条件是不同必要对照。所有节点及失败/缺失完整保留。\n\n| 条件 | 步数 | 指标 | 编辑宏平均% | 正确观测 | 重复观测分母 |\n|---|---:|---|---:|---:|---:|\n'+mainrows(p1)+'\n\nP2：\n\n| 条件 | 步数 | 指标 | 编辑宏平均% | 正确观测 | 重复观测分母 |\n|---|---:|---|---:|---:|---:|\n'+mainrows(p2)+'\n\n'
    text+='1. W0的优势与CE-only泛化退化：见全部节点表及P1三个预设配对差值，不能仅凭训练CE下降认定改善。\n2. 相同长度U净增量：以CE_U_320−CE_ONLY_320以及P2同t_star配对比较为准；具体差值、共同缺失界与来源敏感性见机器报告。\n3. 长短续训：公共长度为'+str(selection['t_star'] if selection else '未选择')+'；选择状态为'+str(selection['status'] if selection else '未到选择阶段')+'。80回退不叫最优checkpoint；40/80/160/320全部报告。\n4. 多来源U：仅在合法FIT组≥2时执行；CAL/CHECK分别报告，训练支持KL不等于未训练局部性。\n5. 专家能力与调用：R0与forced结果、ON计数分别记录。R0在保护输入全OFF时，KL零不能归因于专家保护成功。\n6. 正常小库：实际插入与前缀生成见BANK_RESULTS；NOT_RUN/PARTIAL不称完成。\n7. 下一轮候选：本轮不自动追加实验。未满足编辑/泛化容忍、CHECK保持与正常调用/小库收益三项时，候选仍为NOT_CONFIRMED；不能仅因研究方向固定而宣布U有效。\n8. 证据边界：P1历史8编辑已反复开发，3seed不是24独立病例；P2为TRANSFER_DEV，单seed；U重复专家不增加来源数。患者/检查/论文身份未知，不宣称患者独立。\n9. 独立局部性：T1L/T2L无独立作用域资格时为NA，原面板保持诊断记录；U行为一致性不是医学准确率，稳定错误不算成功。\n\n预算、全部节点、路由、缺失与配对统计以JSON/CSV为准。原始医学输入输出、评分键、来源标识、权重和执行路径保持私有。\n'
    write(RUN/'public/REPORT_STATUS.json',dict(P1=p1['status'] if p1 else 'NOT_COMPLETE',P2=p2['status'] if p2 else 'NOT_RUN',bank_status=read(RUN/'public/BANK_RESULTS.json')['status'],scientific_review='PENDING_HUMAN_READABLE_CONCRETE_INTERPRETATION',automatic_next_round=False))
    (RUN/'public/SCIENTIFIC_REVIEW_ZH.md').write_text(text)
if __name__=='__main__':
    if os.environ.get('REPORT_PHASE') in ('P1','P2'):report(os.environ['REPORT_PHASE'])
    else:final()
