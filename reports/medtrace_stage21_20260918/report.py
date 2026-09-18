"""New derived Stage21 tables only; never writes or rejudges Stage20."""
import sys,json,csv,math,hashlib,itertools
from pathlib import Path
from collections import defaultdict,Counter
from statistics import mean
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.medtrace.astra_judge_bundle import read
from scripts.medtrace.stage19_fasttrack_budget import write
from scripts.medtrace.stage18_score import query_id,score_key
from scripts.medtrace.stage17_prepare import digest
from scripts.medtrace.prepare_stage2_sources import normalized,reviewed_attribute
from scripts.medtrace.stage20_closeout import outputs,expected


def wilson(k,n):
    if not n:return None
    z=1.959963984540054;p=k/n;d=1+z*z/n;c=(p+z*z/2/n)/d;r=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [max(0,c-r),min(1,c+r)]


def report(root):
    root=Path(root);p=root/'private';pub=root/'public';stream=read(p/'STREAM.json');scores=read(p/'QUALIFIED_SCORE_CACHE.json')['scores']
    old=[json.loads(x) for x in (p/'STAGE20_OUTPUTS.jsonl').read_text().splitlines()]
    old=[dict(r,arm='Stage20_FACT_H' if r['arm']=='A' else 'Stage20_BE') for r in old]
    current=outputs(root);rows=old+current;metrics=[];routes=[];details=[];panels={}
    def correct(r,base=False):return scores.get(score_key(r['source'],r['Base'] if base else r['output']))
    def add(arm,n,panel,name,items,unit='QA'):
        vals=[v for v,g in items];known=[v for v in vals if v is not None];den=len(vals);complete=len(known)==den
        groups=defaultdict(list)
        for v,g in items:groups[g].append(v)
        macros=[mean(v) for v in groups.values() if all(x is not None for x in v)]
        macro=mean(macros) if macros and len(macros)==len(groups) else None
        loo=[mean([x for j,x in enumerate(macros) if j!=i]) for i in range(len(macros))] if len(macros)>1 and macro is not None else []
        metrics.append(dict(arm=arm,N=n,panel=panel,metric=name,numerator=sum(known),denominator=den,scored=len(known),value=sum(known)/den if den and complete else None,coverage=len(known)/den if den else None,unit=unit,Wilson95=wilson(sum(known),den) if den and complete and unit=='QA' else None,source_macro=macro,source_count=len(groups),leave_one_source_out=[min(loo),max(loo)] if loo else None))
    def items(rr,base=None):return [(correct(r),r['source'].get('source_group',r['source']['image_sha256'])) for r in rr if base is None or correct(r,True) is base]
    for arm in sorted({r['arm'] for r in rows}):
        prev={}
        for n in (11,19,32,45):
            dd=[r for r in rows if r['arm']==arm and r['prefix']==n and r['mode']=='endpoint'];d={r['query_id']:r for r in dd}
            if len(dd)!=len(d) or set(d)!=set(expected(stream,n,final=n==45)):continue
            panels[(arm,n)]=d
            native=[d[query_id(t['native'])] for t in stream['tasks'][:n]]
            add(arm,n,'native','accuracy',items(native));add(arm,n,'native','Fix',items(native,False))
            ins=[r for r in rows if r['arm']==arm and r['mode']=='insertion' and r['prefix']<=n]
            assert len(ins)==n
            add(arm,n,'insertion','accuracy',items(ins))
            for label,source in [('old',stream['core_rows']),('new',stream['new_rows']),('positive',stream['positive_rows'])]:
                for role in sorted({r['role'] for r in source}):
                    subset=[d[query_id(r)] for r in source if r['role']==role and query_id(r) in d];panel=label+'_'+role
                    add(arm,n,panel,'accuracy',items(subset))
                    if role in ('H_eval','U_eval'):
                        if role=='U_eval':
                            add(arm,n,panel,'fixed_panel_Retention',items(subset,True))
                            def unrelated(r):
                                q=r['source']['question'];a=reviewed_attribute(q)
                                return not any(normalized(q)==normalized(t['native']['question']) or a is not None and a==reviewed_attribute(t['native']['question']) for t in stream['tasks'][:n])
                            subset=[r for r in subset if unrelated(r)]
                        add(arm,n,panel,'Retention',items(subset,True));add(arm,n,panel,'Fix',items(subset,False))
            pairs=[]
            for pack in stream['anchor_packages']:
                nr=d[query_id(pack['training']['native'])];hs=[d[query_id(r)] for r in pack['evaluation'] if r['role']=='H_eval'];vs=[correct(r) for r in hs];v=correct(nr)
                pairs.append((v*mean(vs) if v is not None and vs and all(x is not None for x in vs) else None,pack['training']['native']['source_group']))
            add(arm,n,'old_anchors','PairCorrect',pairs,'edit_macro_not_independent_H')
            add(arm,n,'activation_diagnostic','accuracy',items([r for r in d.values() if r['route']['activated']]))
            transitions=Counter()
            for q in set(prev)&set(d):
                a,b=prev[q],d[q];oldroute=(a['route']['logical_edit_id'],a['route']['activated']);newroute=(b['route']['logical_edit_id'],b['route']['activated']);x,y=correct(a),correct(b)
                if x is not None and y is not None:transitions[f'{int(x)}->{int(y)}|switch={oldroute!=newroute}']+=1
                details.append(dict(arm=arm,N=n,query=q,from_route=oldroute,to_route=newroute,correct_from=x,correct_to=y))
            routes.append(dict(arm=arm,N=n,queries=len(d),activated=sum(r['route']['activated'] for r in d.values()),transitions=dict(transitions)))
            prev=d
    differences=[]
    for a,b in [('NO_H_HSIC','Stage20_FACT_H'),('NO_H_HSIC','Stage20_BE')]:
        for r in [r for r in metrics if r['arm']==a]:
            other=next((x for x in metrics if x['arm']==b and all(x[k]==r[k] for k in ['N','panel','metric'])),None)
            if other and r['value'] is not None and other['value'] is not None:
                assert r['denominator']==other['denominator'] or r['panel']=='activation_diagnostic'
                differences.append(dict(A=a,B=b,N=r['N'],panel=r['panel'],metric=r['metric'],A_numerator=r['numerator'],B_numerator=other['numerator'],A_denominator=r['denominator'],B_denominator=other['denominator'],risk_difference=r['value']-other['value'],percentage_points=100*(r['value']-other['value'])))
    comparisons=[]
    for n in (11,19,32,45):
        arms=[a for a,k in panels if k==n]
        for a,b in itertools.combinations(arms,2):
            aa,bb=panels[(a,n)],panels[(b,n)]
            comparisons.append(dict(A=a,B=b,N=n,queries=len(aa),same_routing=sum((aa[q]['route']['logical_edit_id'],aa[q]['route']['activated'])==(bb[q]['route']['logical_edit_id'],bb[q]['route']['activated']) for q in aa),different_tokens=sum(aa[q]['output']['raw_token_ids']!=bb[q]['output']['raw_token_ids'] for q in aa),different_text=sum(aa[q]['output']['raw_answer']!=bb[q]['output']['raw_answer'] for q in aa)))
    matched=[]
    for a,b in itertools.combinations(sorted({d['arm'] for d in details}),2):
        aa={(d['N'],d['query']):d for d in details if d['arm']==a};bb={(d['N'],d['query']):d for d in details if d['arm']==b};cnt=Counter()
        for k in set(aa)&set(bb):
            x,y=aa[k],bb[k]
            if x['from_route']!=y['from_route'] or x['to_route']!=y['to_route']:continue
            switch=x['from_route']!=x['to_route']
            for side,d in [('A',x),('B',y)]:
                if d['correct_from'] is not None and d['correct_to'] is not None:cnt[f"{side}|switch={switch}|{int(d['correct_from'])}->{int(d['correct_to'])}"]+=1
        matched.append(dict(A=a,B=b,counts=dict(cnt)))
    write(pub/'ROUTING_MECHANISM_ANALYSIS.json',dict(aggregate=routes,paired_same_transitions=matched,output_differences=comparisons,natural_R0=True,conditioned_results_do_not_replace_main=True))
    write(p/'ROUTING_ASSIGNMENTS_PRIVATE.json',dict(rows=details))
    result=[r for r in metrics if r['arm']=='NO_H_HSIC'];write(pub/'NO_H_PURE45_RESULTS.json',dict(results=result,descriptive_Wilson_only=True,p_values=None,old_H_sources=2))
    if result:
        with (pub/'NO_H_PURE45_RESULTS.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader();w.writerows(result)
    write(pub/'RISK_DIFFERENCES.json',dict(rows=differences))
    write(pub/'COMPARISON_DERIVED.json',dict(results=metrics,Stage20_labels_unchanged=True,original_Stage20_tables_not_overwritten=True))
    complete=all(score_key(r['source'],r['output']) in scores for r in current if r['mode'] in ('insertion','endpoint'))
    highest=max([r['N'] for r in result if r['panel']=='native' and r['value'] is not None],default=0)
    gpu=read(pub/'GPU_BUDGET_LEDGER.json');budget=read(pub/'BUDGET_LEDGER.json')
    write(pub/'REPORT_STATUS.json',dict(A_highest_scored_N=highest,A_all_generated_outputs_scored=complete,B='PENDING_LAYER_CLARIFICATION',stage_complete=False))
    table='|方法|native|insertion|old_H Retention|new_H Retention|old_U Retention|new_U Retention|PairCorrect|positive image|\n|---|---|---|---|---|---|---|---|---|\n'
    columns=[('native','accuracy'),('insertion','accuracy'),('old_H_eval','Retention'),('new_H_eval','Retention'),('old_U_eval','Retention'),('new_U_eval','Retention'),('old_anchors','PairCorrect'),('positive_positive_image','accuracy')]
    for arm in ['Stage20_FACT_H','NO_H_HSIC','FACT_FIXED_CONTROL_PENDING','Stage20_BE']:
        cells=[]
        for panel,m in columns:
            r=next((x for x in metrics if x['arm']==arm and x['N']==45 and x['panel']==panel and x['metric']==m),None)
            cells.append(f"{r['numerator']:g}/{r['denominator']} ({r['value']:.1%})" if r and r['value'] is not None else 'NA')
        table+='|'+arm+'|'+'|'.join(cells)+'|\n'
    brief='# Stage21 导师更新（进行中）\n\nNO_H 最高已评分端点 N='+str(highest)+'。Stage20 原始表和判定均保持不变。B 等待层位置澄清：真实 BE 是 L31 up_proj；原计划 FACT-L21 down_proj 不能称为同层对照。\n\n'+table+'\nHSIC：CURRENT HSIC DYNAMIC-LAYER CLAIM NOT SUPPORTED（保存的45/45均选L30）。其余机制问题须待完整两臂结果，当前 INCONCLUSIVE。PairCorrect 为11锚点关联计权，不是11份独立H。old_H仅2来源，不宣称显著；Wilson仅描述性边际区间，不表示同来源QA独立。患者/研究 UNKNOWN；非临床验证；45条为小规模纯流，不能支持100/146长序列结论。\n\nStage21 GPU 已关闭会话 '+str(round(sum(s.get('seconds',0) for s in gpu['sessions']),1))+'/12600秒；新增判定 '+str(budget['new_judgment_items_dispatched'])+'/800。后续所有阶段另共用28800秒/2000项，不能逐阶段重置。费用未由提供商暴露，记NA，tokens保留账单。\n'
    (pub/'ADVISOR_UPDATE_ZH.md').write_text(brief)
    formal=read(p/'STAGE20_FORMAL_BASELINE.json')['files'];oldroot=Path(read(p/'STAGE20_FORMAL_BASELINE.json')['source_directory'])
    assert all(hashlib.sha256((oldroot/name).read_bytes()).hexdigest()==sha for name,sha in formal.items())
    write(pub/'STAGE20_IMMUTABILITY_CHECK.json',dict(unchanged=True,formal_files=len(formal),checks='Exact saved small-formal-file digests; no writes to Stage20'))


if __name__=='__main__':report(sys.argv[1])
