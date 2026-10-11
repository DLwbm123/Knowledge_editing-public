"""Common five-task metrics with fixed masks and shared-payload paired intervals."""
import csv
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
sys.path.insert(0,os.environ['RUN_ROOT']+'/private/tools')
import astra_report as ar
from admit_common146 import METHODS, connect, slot

RUN,r,c=ar.RUN,ar.r,ar.common
LABELS=dict(TT88_W0_RETRO146='TT88 W0 + R0',balancedit='BalancEdit locked adaptation',belora='BELoRA V2 effect-repaired')
MODES=[('single_R0',1)]+[('bank_R0',n) for n in (1,50,100,146)]
TASKS=('T0','T1G','T2G','T1L','T2L')


def records_for(consumers,tasks,ledger):
    records=[]
    for row in consumers:
        q=ledger['queries'][row['query_id']]
        active={(t['native']['image_sha256'],t['native']['question']) for t in tasks[:row['prefix']]} if row['mode']=='bank_R0' else set()
        records.append((dict(binding=dict(phase=dict(arm=row['method'],node=0,prefix=row['prefix']),
            input=q,mode=row['mode'],owner_order=row['edit_order']),
            active_target=(q['image_sha256'],q['question']) in active,effective_expert=None),row['payload_key']))
    return records


def summarize(records,scores,tasks,intervals=True):
    panels=[];coeffs={};groups={}
    for mode,prefix in MODES:
        for method in METHODS:
            for task in TASKS:
                m,coef,g=r.panel(records,scores,'RETRO146',method,0,mode,task,tasks[:prefix] if mode=='bank_R0' else tasks,prefix)
                m.pop('route_ON')
                if intervals:m.update(ar.bootstrap(coef,scores,g))
                panels.append(m);coeffs[method,mode,prefix,task]=coef;groups.update(g)
    return panels,coeffs,groups


def value(m):
    if m['macro'] is not None:return f"{m['macro']:.3f}"
    lo,hi=m['macro_bounds']
    return 'NA' if lo is None else f'[{lo:.3f}, {hi:.3f}]'


def main():
    started=time.time();ar.selfcheck()
    root=RUN/'private/judge_common146_medium';db=connect(root/'queue.sqlite')
    assert (root/'ALL_WORKERS_COMPLETE.json').exists()
    payloads={x['key']:x for x in db.execute('SELECT * FROM payload')}
    assert len(payloads)==3146 and all(x['status'] in ('FORMAT_VALID','MISSING') for x in payloads.values())
    scores={k:x['correct'] for k,x in payloads.items()};consumers=list(db.execute('SELECT * FROM consumer'))
    assert len(consumers)==17166
    resource=c.read(RUN/'RESOURCE_LEDGER.json');attempted=[k for b in resource['Astra_common146_batches'] for k in b['keys']]
    assert len(attempted)==len(set(attempted))==resource['Judge_attempts']==3146
    assert resource['gpu_seconds_used']==0 and resource['gpu_sessions']==[]
    # Saved evidence, input binding, isolation, model, and decisions must agree.
    from scripts.medtrace.astra_judge_bundle import validate
    for batch in resource['Astra_common146_batches']:
        evidence=c.read(root/'evidence'/(batch['id']+'.json'))
        assert evidence['batch']['records']==[json.loads(payloads[k]['record']) for k in batch['keys']]
        if batch['status']=='FORMAT_VALID':
            ev=evidence['evidence']
            assert ev['actual_model']=='gpt-6-astra' and ev['reasoning_effort']=='medium'
            assert ev['input_binding']==c.digest(evidence['batch']) and ev['exit_code']==0
            assert not ev.get('errors') and not ev['tool_event_types'] and all(ev['isolation_checks'].values())
            decisions=validate(evidence['batch'],evidence['response'])
            assert [int(x['is_correct']) for x in decisions]==[scores[k] for k in batch['keys']]
        else:
            assert batch['status']=='FAILED_NO_RETRY' and all(scores[k] is None for k in batch['keys'])
    tasks=c.read(RUN/'private/BENCHMARK146_QUEUE.json')['tasks'];ledger=c.read(RUN/'private/EVAL_LEDGER.json')
    records=records_for(consumers,tasks,ledger)
    panels,coeffs,groups=summarize(records,scores,tasks)
    contrasts=[]
    for mode,prefix in MODES:
        for baseline in METHODS[1:]:
            for task in TASKS:
                aa,bb=coeffs[METHODS[0],mode,prefix,task],coeffs[baseline,mode,prefix,task]
                assert set(aa)==set(bb)
                delta={e:r.combine([aa[e],{k:-v for k,v in bb[e].items()}]) for e in aa}
                bound=r.score_bounds(r.combine([{k:v/len(delta) for k,v in unit.items()} for unit in delta.values()]),scores) if delta else [None,None]
                contrasts.append(dict(baseline=baseline,mode=mode,prefix=prefix,task=task,
                    paired_edits=len(delta),delta_bounds_pp=[v*100 if v is not None else None for v in bound],**ar.bootstrap(delta,scores,groups)))
    assert len(panels)==75 and len(contrasts)==50
    # Recover old scores solely for a separate sensitivity comparison, never primary.
    config=c.read(RUN/'PLAN_CONFIG.json');tt=Path(config['TT_parent']);old=Path(config['baseline_parent'])
    old_dbs={METHODS[0]:connect(tt/'private/judge_astra_medium/queue.sqlite'),
             'balancedit':connect(old/'private/judge_common/queue.sqlite')}
    old_dbs['belora']=old_dbs['balancedit'];old_records=[];old_scores={}
    recovery=connect(tt/'private/judge_astra_medium_recovery1/queue.sqlite')
    recovered={x['key']:x['correct'] for x in recovery.execute('SELECT * FROM payload WHERE status=\'FORMAT_VALID\'')}
    target={slot(x) for x in consumers};selected={}
    for method,source in old_dbs.items():
        for row in source.execute('SELECT * FROM consumer WHERE method=?',(method,)):
            s=slot(row)
            if s not in target or (method,s) in selected:continue
            selected[method,s]=True;mode,prefix,owner,qid=s
            score=source.execute('SELECT correct FROM payload WHERE key=?',(row['payload_key'],)).fetchone()[0]
            if method==METHODS[0]:score=recovered.get(row['payload_key'],score)
            key=method+':'+row['payload_key'];old_scores[key]=score
            old_records.append(dict(method=method,mode=mode,prefix=prefix,edit_order=owner,query_id=qid,payload_key=key))
    assert len(old_records)==17166
    history,_,_=summarize(records_for(old_records,tasks,ledger),old_scores,tasks,intervals=False)
    prior=c.read(tt/'public/ASTRA_MEDIUM_RECOVERED_RESULTS.json')['panels']['benchmark146']
    for m in history:
        if m['arm']!=METHODS[0]:continue
        p=next(x for x in prior if (x['mode'],x['prefix'],x['task'])==(m['mode'],m['prefix'],m['task']))
        for field in ('known_correct','observations','missing','edit_units','macro'):
            assert m[field]==p[field] or (isinstance(m[field],float) and abs(m[field]-p[field])<1e-9)
    insertion=[]
    lookup={(x['method'],x['mode'],x['prefix'],x['query_id']):x['payload_key'] for x in consumers}
    for method in METHODS:
        cells={'both_correct':0,'lost':0,'gained':0,'both_wrong':0,'missing':0}
        for t in tasks:
            before=scores[lookup[method,'insertion_R0',t['order'],t['edit_id']]]
            after=scores[lookup[method,'bank_R0',146,t['edit_id']]]
            name='missing' if before is None or after is None else 'both_correct' if before and after else 'lost' if before else 'gained' if after else 'both_wrong'
            cells[name]+=1
        assert sum(cells.values())==146
        insertion.append(dict(method=method,**cells))
    missing=sum(v is None for v in scores.values())
    result=dict(status='COMPLETE_NO_MISSING' if not missing else 'TERMINAL_WITH_MISSING',
        model='gpt-6-astra',reasoning_effort='medium',payloads=3146,valid_payloads=3146-missing,missing_payloads=missing,
        consumers=17166,missing_consumers=sum(scores[x['payload_key']] is None for x in consumers),
        panels=panels,contrasts=contrasts,insertion=insertion,historical_scores_same_aggregation=history,
        prior_TT25_reproduced=True,Judge_attempts=3146,new_GPU_hours=0,new_training=0,new_generation=0,
        primary='bank146 T2G: TT88 minus each baseline',original_Base_masks=True,Base_rejudged=False,
        bootstrap_draws=10000,bootstrap_seed=20260912,independent_confirmation=False,
        locality='HISTORICAL_DIAGNOSTIC_NOT_QUALIFIED_PROTECTION',one_order=True,patient_independence='UNKNOWN',
        seconds=time.time()-started,Judge_batch_seconds=sum(b['ended_epoch']-b['started_epoch'] for b in resource['Astra_common146_batches']))
    c.write(RUN/'public/RESULTS.json',result)
    with (RUN/'public/PANELS.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['method','mode','prefix','task','macro','macro_lower','macro_upper','micro','correct','observations','missing','edits'])
        for p in panels:writer.writerow([p['arm'],p['mode'],p['prefix'],p['task'],p['macro'],*p['macro_bounds'],p['micro'],p['known_correct'],p['observations'],p['missing'],p['edit_units']])
    text='# 原146三方法共同Astra medium重评\n\n固定原顺序、原Base掩码与五任务。所有数字为编辑宏平均百分比；方括号是缺失识别界，不是置信区间。全方法复用原输出，共同新epoch，没有训练或生成。\n\n'
    text+='|模式|前缀|方法|T0|T1G|T2G|T1L诊断|T2L诊断|\n|---|---:|---|---:|---:|---:|---:|---:|\n'
    for mode,prefix in MODES:
        for method in METHODS:
            ms=[next(p for p in panels if (p['arm'],p['mode'],p['prefix'],p['task'])==(method,mode,prefix,t)) for t in TASKS]
            text+='|'+'|'.join([mode,str(prefix),LABELS[method]]+[value(m) for m in ms])+'|\n'
    text+='\n## 完整配对对比\n\n差值及区间单位为百分点。所有比较保留，不选择有利前缀。\n\n|模式|前缀|TT88减基线|任务|差值界|编辑95%区间|来源95%区间|\n|---|---:|---|---|---|---|---|\n'
    for p in contrasts:text+='|'+'|'.join(map(str,[p['mode'],p['prefix'],LABELS[p['baseline']],p['task'],p['delta_bounds_pp'],p['edit_ci'],p['source_ci']]))+'|\n'
    text+=f'\n新增Judge尝试3146，有效{3146-missing}、缺失{missing}；新增GPU/训练/生成全部为0。旧结果保留，旧分数仅用于独立敏感性表，不混入主表。完整微平均、分母、插入保持及历史同口径重算见RESULTS.json和PANELS.csv。\n'
    text+='\nBalancEdit为锁定适配版，BELoRA为effect-repaired版，不能称论文原版或等参数/等优化预算实验。TT为TT88 W0+R0，不是在线N/T/NT保护臂。原146是已触达开发队列，只有一个顺序，患者独立性未知；T1L/T2L仅历史局部性诊断，无独立合格医学保护证据。本轮无新Base评分和掩码敏感性，不宣称独立SOTA确认。\n'
    (RUN/'public/REPORT_ZH.md').write_text(text)
    c.write(root/'REPORT_COMPLETE.json',dict(status=result['status'],epoch=time.time(),panels=75,contrasts=50,publication_pending=True))
    print(json.dumps({k:v for k,v in result.items() if k not in ('panels','contrasts','historical_scores_same_aggregation')}))


if __name__=='__main__':main()
