"""Aggregate only matched cohorts; preserve missing and zero denominators."""
import json,os,random
from collections import defaultdict
from pathlib import Path
from statistics import mean
import metrics_r2 as metrics
ROOT=Path(os.environ['RUN_ROOT'])
def read(p):return json.loads(p.read_text())
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2))
def collect(ids):
    rows=[]
    for name in ids:
        for p in (ROOT/'jobs'/name).glob('*/p*/CONSUMERS.json'):rows+=read(p)
    identities=[(r['arm'],r['mode'],r['prefix'],r['edit'],r['task'],r['query_id']) for r in rows]
    if len(set(identities))!=len(identities):raise ValueError('Duplicate consumers across jobs')
    return rows

def report(ids,name):
    rows=collect(ids);scores={p.stem:read(p)['is_correct'] for p in (ROOT/'private/judge/scores').glob('*.json')}
    summary=metrics.summarize(rows,scores)
    arms=sorted({r['arm'] for r in rows});modes=sorted({(r['mode'],r['prefix']) for r in rows})
    pairs=[metrics.paired(rows,scores,a,m,p) for a in arms if a!='B0' for m,p in modes]
    route=[];clusters=[]
    for arm in arms:
        for task in ['T0','T1G','T1L','T2G','T2L','T2L_PRESSURE']:
            selected=[r for r in rows if r['arm']==arm and r['task']==task]
            if not selected:continue
            active=[r for r in selected if r['route']['logical_edit_id'] is not None]
            route.append(dict(arm=arm,task=task,consumers=len(selected),rejected=len(selected)-len(active),
                associated_hit=sum(r['route']['logical_edit_id']==r['edit'] for r in active),
                other_expert=sum(r['route']['logical_edit_id']!=r['edit'] for r in active),
                negative_role=metrics.retention(task),non_target_activation=len(active) if metrics.retention(task) else None,
                activated_base_correct_to_wrong=sum(scores.get(r['base_judge_key']) is True and scores.get(r['judge_key']) is False for r in active)))
    # Source-cluster sensitivity of paired micro differences, separate from edit bootstrap.
    for a in arms:
        if a=='B0':continue
        for mode,prefix in modes:
            def key(r):return r['edit'],r['task'],r['query_id']
            aa={key(r):r for r in rows if r['arm']=='B0' and (r['mode'],r['prefix'])==(mode,prefix)}
            bb={key(r):r for r in rows if r['arm']==a and (r['mode'],r['prefix'])==(mode,prefix)}
            for task in ['T0','T1G','T1L','T2G','T2L','T2L_PRESSURE']:
                groups=defaultdict(list);missing=0
                for k,r in aa.items():
                    if r['task']!=task:continue
                    t=bb.get(k);x=scores.get(r['judge_key']);y=scores.get(t['judge_key']) if t else None
                    if x is None or y is None:missing+=1;continue
                    groups[r['source_group']].append(int(y)-int(x))
                vals=list(groups.values());rng=random.Random(20260925)
                boot=sorted(mean(x for g in rng.choices(vals,k=len(vals)) for x in g) for _ in range(2000)) if vals and not missing else []
                clusters.append(dict(arm=a,mode=mode,prefix=prefix,task=task,sources=len(vals),missing=missing,
                    CI95=[boot[49],boot[1949]] if boot else None,estimand='source-cluster paired micro sensitivity'))
    keys={r['judge_key'] for r in rows};basekeys={r['base_judge_key'] for r in rows}
    result=dict(jobs=ids,summary=summary,paired=pairs,route_attribution=route,source_cluster_sensitivity=clusters,
                consumers=len(rows),unique_judge_requests=len(keys),missing_judge_keys=sum(k not in scores for k in keys),
                missing_base_keys=sum(k not in scores for k in basekeys),CONFIRM=None,
                interpretation='Exploratory; standard T2L and pressure separate; empty T1L is not PASS')
    write(ROOT/'private/reports'/f'{name}.json',result)
    public=dict(result,paired=[{k:v for k,v in p.items() if k!='changes'} for p in pairs])
    write(ROOT/'public'/f'{name}.json',public)
    return result

def choose(result):
    summary={(r['arm'],r['task']):r for r in result['summary'] if r['mode']=='single'}
    def value(arm,task):return summary.get((arm,task),{}).get('edit_macro')
    candidates=[]
    for arm in ['P@80','P@160','P@320','P+S@80','P+S@160','P+S@320']:
        vals=[value(arm,t) for t in ['T0','T1G','T2G','T2L_PRESSURE']]
        base=[value('B0',t) for t in ['T0','T1G','T2G','T2L_PRESSURE']]
        if any(x is None for x in vals+base):continue
        if vals[0]<.99 or vals[1]<base[1]-.01:continue
        pair=next(p for p in result['paired'] if p['arm']==arm)
        if pair['status']!='COMPLETE':continue
        if pair.get('metrics',{}).get('T1L',{}).get('right_to_wrong',0):continue
        # Prefer joint gains, then retention, then generalization. One global step.
        gains=(vals[2]-base[2],vals[3]-base[3])
        candidates.append(((gains[0]>=.02 and gains[1]>=.08,gains[1],gains[0],-int(arm.split('@')[1])),arm,gains))
    if candidates:
        _,label,gains=max(candidates)
        return dict(label=label,arm=label.split('@')[0],step=int(label.split('@')[1]),status='GLOBAL_DEV_LOCK',
                    gains_T2G_pressure=gains,T1L_gate='NOT_ASSESSABLE_EMPTY_DEV_DENOMINATOR',engineering_pass=False)
    return dict(label='P@320',arm='P',step=320,status='PREDECLARED_EXPLORATORY_FALLBACK_NO_QUALIFIED_WINNER',
                engineering_pass=False,T1L_gate='NOT_ASSESSABLE_EMPTY_DEV_DENOMINATOR')

if __name__=='__main__':
    assert choose({'summary':[],'paired':[]})['label']=='P@320'
    metrics.self_check();print('PASS')


def final_report():
    import csv
    state=read(ROOT/'RUN_STATUS.json');ledger=read(ROOT/'RESOURCE_LEDGER.json')
    write(ROOT/'public/RESOURCE_LEDGER_AGGREGATE.json',dict(gpu_hours=ledger['gpu_seconds_used']/3600,
        gpu_hours_limit=16,judge_attempt_items=ledger['judge_submission_attempt_items'],judge_limit=6000,
        inherited_judge_items=ledger['prior_run_judge_items'],failed_attempt_items=sum(x['items'] for x in ledger['judge_attempts'] if x['status']=='FAILED_NO_RETRY'),
        gpu_sessions=len(ledger['gpu_sessions']),unclosed_sessions=sum(s.get('ended_epoch') is None for s in ledger['gpu_sessions'])))
    curves=defaultdict(list)
    for path in (ROOT/'runs').glob('s20260925/*/e*/TRAINING_CURVES.json'):
        arm=path.parent.parent.name
        for stage,points in read(path).items():
            for point in points:
                for key,value in point.items():
                    if key!='step' and isinstance(value,(int,float)):curves[(arm,stage,point['step'],key)].append(value)
    # P+S exposes continuation and gradient traces; never infer utility from loss.
    for path in (ROOT/'runs/s20260925/P+S').glob('e*/U_TRACE.json'):
        for point in read(path):
            for key,value in point.items():
                if key!='step' and isinstance(value,(int,float)):curves[('P+S','continuation',point['step'],key)].append(value)
    with (ROOT/'public/CURVES_AGGREGATE.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['arm','stage','step','field','n','mean','min','max'])
        for key,vals in sorted(curves.items()):w.writerow([*key,len(vals),mean(vals),min(vals),max(vals)])
    lines=['# MedTRACE TextJoint-R2 实验报告','',f"状态：{state['status']}；阶段：{state['phase']}。",'',
        '审阅基线：97baf0f61351cc06b3bfcda8ae0d79d441668a3e。新分支独立执行，未合并 PR #1。',
        '本轮保持 L30、rank4、float16、原优化器与固定 Judge。P+S 只改变 continuation 保护，共享 P-W0；旧 U 保留，新 U 不足如实统计。',
        'DEV 使用前24编辑；VERIFY使用后24编辑。新增压力题源分离，但编辑历史已暴露，没有合法独立 CONFIRM，所有结论为探索性。',
        '标准 T2L 与 T2L_PRESSURE 分开。官方面板仅执行冻结 Base 掩码的 Fix/Retention 有效输入，不能解释为全题准确率。DEV T1L 分母为0，不通过工程门槛。',
        'Base候选的缺失评分未重判；只有已正确评分的输入进入冻结保持面板。消费者、执行输入、Judge请求因去重规则不同，不应混计。','']
    for split in ['VALIDATION','TEST']:
        a=read(ROOT/f'BASE_COVERAGE_{split}.json');write(ROOT/'public'/f'BASE_COVERAGE_{split}.json',a)
        lines.append(f"{split} Base候选：生成{a['generated']}，已评分{a['scored']}，missing={a['missing']}，Base正确{a['base_correct']}，来源组{a['base_correct_sources']}。")
    panel=read(ROOT/'PANEL_LOCK.json')
    write(ROOT/'public/PANEL_COVERAGE.json',{k:panel[k] for k in ['DEV24','VERIFY24']})
    support=read(ROOT/'U_SUPPORT_AUDIT.json')
    counts=support['counts'];write(ROOT/'public/U_SUPPORT_AGGREGATE.json',dict(edits=len(counts),hard2=sum(c['hard']==2 for c in counts),diverse2=sum(c['diverse']==2 for c in counts),
        old_U_preserved=True,old_U_in_previous_expanded=True,insufficient_hard=sum(c['hard']<2 for c in counts)))
    if (ROOT/'SELECTION_LOCK.json').exists():lines+=['','全局选择：'+json.dumps(read(ROOT/'SELECTION_LOCK.json'),ensure_ascii=False)]
    for name in ['CANARY','DEV24','VERIFY24_SINGLE','VERIFY24_SEQUENTIAL','VERIFY24_BASELINE']:
        path=ROOT/'public'/f'{name}.json'
        if not path.exists():continue
        r=read(path);lines+=['',f'## {name}',f"消费者{r['consumers']}；唯一Judge请求{r['unique_judge_requests']}；missing请求{r['missing_judge_keys']}。",'',
            '|臂|模式/前缀|任务|分子/分母|有效编辑/来源|macro|micro|missing|','|---|---|---|---|---|---|---|---|']
        for x in r['summary']:
            val=lambda v:'NA' if v is None else f'{v*100:.2f}%'
            lines.append(f"|{x['arm']}|{x['mode']}/{x['prefix']}|{x['task']}|{x['numerator']}/{x['denominator']}|{x['edits']}/{x['source_groups']}|{val(x['edit_macro'])}|{val(x['micro'])}|{x['missing']}|")
    lines+=['',f"累计 GPU 驻留 {ledger['gpu_seconds_used']/3600:.3f}/16 小时；Judge提交尝试 {ledger['judge_submission_attempt_items']}/6000（含前轮704项）。",
        '逐题变化与私有评分保留在 private/reports；公开仅聚合。配对区间以编辑为重采样单位，来源聚类敏感性单独提供；重复观测不增加病例数。',
        'loss/梯度曲线仅用于训练诊断，不等于自由生成性能提升。输出token一致性与参考答案正确性分别计算。',
        '保留本轮 B0/P/候选专家及关键初始化，用于配对复现与部署；基线临时权重在最后消费者完成后按清单清理。']
    if state.get('error'):lines+=['','运行遇到技术或预算阻塞，详细证据保留在私有状态回执。']
    (ROOT/'public/REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
