"""Bounded semantic scoring and separate old/new-source Stage20 reports."""
from collections import Counter,defaultdict
from pathlib import Path
import csv,json,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.medtrace.astra_judge_bundle import read,write_new
from scripts.medtrace.stage17_prepare import digest,PROTOCOL
from scripts.medtrace.stage18_score import query_id,score_key
from scripts.medtrace.stage19_fasttrack_budget import write


def judge(root,records,name,category='main'):
    from scripts.medtrace.stage18_base_judge import run
    root=Path(root);p=root/'private';path=p/'QUALIFIED_SCORE_CACHE.json'
    original=read(path if path.exists() else p/'EXISTING_SCORE_CACHE.json')
    if original['protocol']!=PROTOCOL:raise ValueError('Semantic protocol changed')
    cache=original['scores'];receipt=p/(name+'_SCORE_RECEIPT.json')
    binding=digest(records)
    if receipt.exists():
        if read(receipt)['consumer_binding']!=binding:raise ValueError('Score consumers changed')
        return cache
    unique={score_key(r['source'],r['output']):dict(query_id=score_key(r['source'],r['output']),source=r['source'],output=dict(raw_answer=r['output']['raw_answer'])) for r in records}
    novel=[r for k,r in unique.items() if k not in cache]
    ledgerpath=root/'public/BUDGET_LEDGER.json';ledger=read(ledgerpath)
    key=category+'_items_dispatched';limit=ledger[category+'_limit']
    if ledger.get(key,0)+len(novel)>limit or ledger['new_judgment_items_dispatched']+len(novel)>2000:raise ValueError('Category or total judgment budget exceeded')
    ledger['exact_reuse_count']+=len(unique)-len(novel);write(ledgerpath,ledger)
    write_new(p/(name+'_SCORE_CONSUMERS.json'),dict(binding=binding,rows=[dict(arm=r.get('arm'),prefix=r.get('prefix'),mode=r.get('mode','Base'),query_id=r['query_id'],score_key=score_key(r['source'],r['output'])) for r in records]))
    for offset in range(0,len(novel),60):
        chunk=novel[offset:offset+60];label=name+f'_{offset//60:03d}'
        source=dict(scope='STAGE20_'+label,records=chunk);inputpath=p/(label+'_JUDGE_INPUT.json');bundle=p/(label+'_JUDGE')
        write_new(inputpath,source)
        ledger=read(ledgerpath);ledger['new_judgment_items_dispatched']+=len(chunk);ledger[key]=ledger.get(key,0)+len(chunk)
        item=dict(name=label,category=category,items=len(chunk),input_binding=digest(source),status='DISPATCHED',actual_cost=None,cost_status='Provider currency billing unavailable; never treated as zero')
        ledger['judgment_batches'].append(item);write(ledgerpath,ledger)
        run(dict(base_outputs=str(inputpath),bundle=str(bundle),N_queries=len(chunk),authorization='USER_AUTHORIZED_ASTRA_DEV_REVIEW',cli='/Applications/ChatGPT.app/Contents/Resources/codex'))
        verdict=read(bundle/'operator/VERDICTS.json')
        if verdict['protocol']!=PROTOCOL or verdict['source_binding']!=digest(source):raise ValueError('Judge lineage changed')
        by={r['opaque_query_id']:r['is_correct'] for r in verdict['decisions']}
        if set(by)!={digest(r) for r in chunk} or any(type(x) is not bool for x in by.values()):raise ValueError('Judge coverage invalid')
        for row in chunk:cache[row['query_id']]=by[digest(row)]
        evidence=read(bundle/'operator/execution_evidence/b000.json');item.update(status='COMPLETE',usage=evidence.get('usage'),execution_binding=digest(evidence))
        write(path,dict(scores=cache,protocol=PROTOCOL));write(ledgerpath,ledger)
    write(path,dict(scores=cache,protocol=PROTOCOL))
    write_new(receipt,dict(consumer_binding=binding,new=len(novel),reused=len(unique)-len(novel),category=category))
    return cache


def outputs(root):
    path=Path(root)/'private/OUTPUTS.jsonl'
    if not path.exists():return []
    text=path.read_text();lines=text.splitlines()
    if not text.endswith('\n'):lines=lines[:-1]
    return [json.loads(line) for line in lines]


def expected(stream,n,final=False,ablation=False):
    rows={query_id(t['native']):t['native'] for t in stream['tasks'][:n]}
    rows.update({query_id(r):r for r in stream['core_rows']})
    if not ablation and (n>=50 or final):rows.update({query_id(r):r for r in stream['new_rows']+stream['positive_rows']})
    return rows


def score_endpoint(root,n,final=False,ablation=False):
    source=Path(root)/'private/ablation' if ablation else Path(root)
    stream=read(Path(root)/'private/STREAM.json');rows=[r for r in outputs(source) if r['mode']=='endpoint' and r['prefix']==n]
    ids=set(expected(stream,n,final,ablation))
    arms=('A','B')
    if ablation and read(source/'public/GENERATED.json').get('arms')==['A']:
        if read(Path(root)/'public/ABLATION_TOKEN_SUPPORT.json')['EXTRA_full11_supported']:raise ValueError('Missing eligible EXTRA arm')
        arms=('A',)
    if Counter(r['arm'] for r in rows)!={a:len(ids) for a in arms} or any({r['query_id'] for r in rows if r['arm']==a}!=ids for a in arms):raise ValueError('Incomplete endpoint before scoring')
    judge(root,rows,('ablation' if ablation else 'final' if final else 'prefix')+str(n),'ablation' if ablation else 'main')


def table(public,name,rows,**meta):
    write(public/(name+'.json'),dict(results=rows,**meta))
    if rows:
        with (public/(name+'.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def continuation_receipts(private):
    # CP_W0 is initialization with a different receipt schema, not continuation.
    for path in sorted((Path(private)/'edits').glob('e*/*/TRAINING.json')):
        if path.parent.name not in ('C_FACT','C_NO_H','C_EXTRA'):continue
        receipt=read(path)
        if receipt['branch']!=path.parent.name:raise ValueError('Continuation receipt branch mismatch')
        yield path,receipt


def report(root,final=False):
    root=Path(root);p=root/'private';public=root/'public';stream=read(p/'STREAM.json');scores=read(p/'QUALIFIED_SCORE_CACHE.json')['scores']
    allrows=outputs(root);end=[r for r in allrows if r['mode']=='endpoint'];result=[];newresults=[];ablation=[];routes=[];source_stats=[]
    def correct(r,base=False):return scores.get(score_key(r['source'],r['Base'] if base else r['output']))
    def metric(target,arm,n,panel,name,values,unit='QA'):
        known=[v for v in values if v is not None];den=len(values)
        target.append(dict(arm=arm,N=n,K=n,panel=panel,metric=name,numerator=sum(known),denominator=den,scored=len(known),value=sum(known)/den if den and len(known)==den else None,unit=unit,coverage=len(known)/den if den else None))
    def evaluate(rows,n,target,arms=None,scope='main'):
        from scripts.medtrace.prepare_stage2_sources import normalized,reviewed_attribute
        natives=[t['native'] for t in stream['tasks'][:n]]
        def unrelated(r):
            q=r['source']['question'];attr=reviewed_attribute(q)
            return not any(normalized(q)==normalized(t['question']) or attr is not None and attr==reviewed_attribute(t['question']) for t in natives)
        by={a:{r['query_id']:r for r in rows if r['arm']==a} for a in sorted({r['arm'] for r in rows})}
        for a,d in by.items():
            arm=arms[a] if arms else a
            native=[d[query_id(t['native'])] for t in stream['tasks'][:n]]
            metric(target,arm,n,'native','accuracy',[correct(r) for r in native]);metric(target,arm,n,'native','Fix',[correct(r) for r in native if correct(r,True) is False])
            for label,panel in [('old',stream['core_rows']),('new',stream['new_rows']),('positive',stream['positive_rows'])]:
                rr=[d[query_id(r)] for r in panel if query_id(r) in d]
                dst=newresults if scope=='main' and label=='new' else target
                for role in sorted({r['source']['role'] for r in rr}):
                    subset=[r for r in rr if r['source']['role']==role];prefix=label+'_'+role
                    metric(dst,arm,n,prefix,'accuracy',[correct(r) for r in subset])
                    if role in ('H_eval','U_eval'):
                        if role=='U_eval':
                            metric(dst,arm,n,prefix,'fixed_panel_Retention',[correct(r) for r in subset if correct(r,True) is True])
                            subset=[r for r in subset if unrelated(r)]
                        for name,value in [('Retention',True),('Fix',False)]:metric(dst,arm,n,prefix,name,[correct(r) for r in subset if correct(r,True) is value])
                        groups=defaultdict(list)
                        for r in subset:groups[r['source']['source_group']].append(r)
                        for name,base in [('accuracy',None),('Retention',True),('Fix',False)]:
                            vals=[]
                            for group,rs in sorted(groups.items()):
                                vs=[correct(r) for r in rs if base is None or correct(r,True) is base]
                                v=sum(vs)/len(vs) if vs and all(x is not None for x in vs) else None
                                if vs:vals.append(v)
                                source_stats.append(dict(scope=scope,arm=arm,N=n,panel=prefix,metric=name,source=digest(group)[:12],denominator=len(vs),value=v))
                            metric(dst,arm,n,prefix,name+'_source_macro',vals,'source')
                            valid=[v for v in vals if v is not None]
                            loo=[(sum(valid)-v)/(len(valid)-1) for v in valid] if len(valid)>1 and len(valid)==len(vals) else []
                            source_stats.append(dict(scope=scope,arm=arm,N=n,panel=prefix,metric=name+'_delete_one_range',source=None,denominator=len(vals),value=[min(loo),max(loo)] if loo else None))
            pairs=[]
            for package in stream['anchor_packages'][:min(11,n)]:
                native_score=correct(d[query_id(package['training']['native'])]);hs=[correct(d[query_id(r)]) for r in package['evaluation'] if r['role']=='H_eval']
                pairs.append(native_score*sum(hs)/len(hs) if native_score is not None and hs and all(x is not None for x in hs) else None)
            metric(target,arm,n,'old_anchors','PairCorrect',pairs,'edit_macro_not_independent_H')
            metric(target,arm,n,'activation_diagnostic','accuracy',[correct(r) for r in d.values() if r['route']['activated']])
        return by
    ns=sorted({r['prefix'] for r in end});previous={};comparisons=[]
    for n in ns:
        rows=[r for r in end if r['prefix']==n]
        # Report only completed paired prefix files; a concurrently written next endpoint is not a result.
        if not (public/f'PREFIX_{n:03d}.json').exists() and not final:continue
        if any(sum(r['arm']==a for r in rows)<n+len(stream['core_rows']) for a in ('A','B')):continue
        by=evaluate(rows,n,result)
        for a,d in by.items():
            exposure=Counter((r['selected_kind'],r.get('writer_layer'),r['route']['activated']) for r in d.values())
            transitions=Counter();switches=0;prior=previous.get(a,{})
            for q in set(prior)&set(d):
                old,now=prior[q],d[q];switch=(old['route']['logical_edit_id'],old['route']['activated'])!=(now['route']['logical_edit_id'],now['route']['activated'])
                switches+=switch
                x,y=correct(old),correct(now)
                if x is not None and y is not None:transitions[f'{x}_to_{y}_switch_{switch}']+=1
            routes.append(dict(arm=a,N=n,exposure=[dict(method=k[0],writer_layer=k[1],activated=k[2],queries=v) for k,v in exposure.items()],expert_switches=switches,correctness_transitions=dict(transitions),natural_R0=True))
            previous[a]=d
        a,b=by['A'],by['B'];comparisons.append(dict(N=n,queries=len(a),different_tokens=sum(a[q]['output']['raw_token_ids']!=b[q]['output']['raw_token_ids'] for q in a),different_text=sum(a[q]['output']['raw_answer']!=b[q]['output']['raw_answer'] for q in a),both_scored=sum(correct(a[q]) is not None and correct(b[q]) is not None for q in a)))
        for arm in ('A','B'):
            inserted=[r for r in allrows if r['mode']=='insertion' and r['arm']==arm and r['prefix']<=n]
            metric(result,arm,n,'insertion','accuracy',[correct(r) for r in inserted])
    abrows=[r for r in outputs(root/'private/ablation') if r['mode']=='endpoint' and r['prefix']==11]
    ab_counts=Counter(r['arm'] for r in abrows)
    noh_only=ab_counts==dict(A=37) and (public/'ABLATION_TOKEN_SUPPORT.json').exists() and not read(public/'ABLATION_TOKEN_SUPPORT.json')['EXTRA_full11_supported']
    if ab_counts==dict(A=37,B=37) or noh_only:
        evaluate(abrows,11,ablation,dict(A='NO_H_HSIC',B='EXTRA_HSIC'),'ablation')
        ablation += [dict(r,arm='FACT_HSIC') for r in result if r['arm']=='A' and r['N']==11 and r['panel']!='insertion']
    table(public,'RESULTS_PURE',result,patient_study='UNKNOWN',clinical_signoff=False)
    table(public,'NEW_SOURCE_RESULTS',newresults,exposure=read(p/'NEW_SOURCE_PANEL.json')['exposure'])
    abstatus=('NO_H_COMPLETE_EXTRA_UNSUPPORTED' if noh_only else 'COMPLETE') if ablation and all(r['scored']==r['denominator'] for r in ablation) else 'PENDING_OR_RESOURCE_UNSUPPORTED'
    table(public,'ABLATION_DEV11',ablation,status=abstatus,pure11_FACT_reused=True,EXTRA_full11='UNSUPPORTED_TOKEN_STRATUM' if noh_only else 'PENDING_OR_PRESENT')
    write(public/'ROUTING_EXPOSURE_AND_TRANSITIONS.json',dict(rows=routes,output_differences=comparisons,condition_forced_results_replace_main=False))
    write(public/'SOURCE_EQUAL_SENSITIVITY.json',dict(rows=source_stats,bootstrap_significance=False))
    hsic=[];compute=[]
    for path in sorted((p/'edits').glob('e*/SELECTION.json')):
        s=read(path);hsic.append(dict(position=int(path.parent.name[1:]),layer_id=s['layer_id'],scores=s.get('scores'),invalid=s.get('invalid'),diagnostics=s.get('diagnostics'),token_counts=s.get('token_counts')))
    for base,label in [(p,'main'),(p/'ablation/private','ablation')]:
        for path,t in continuation_receipts(base):
            compute.append(dict(scope=label,position=int(path.parent.parent.name[1:]),branch=t['branch'],steps=t['steps'],session_seconds=t['session_seconds'],tokens=t['tokens'],extra_gradient_nonzero_steps=t['extra_gradient_nonzero_steps']))
    write(public/'HSIC_REBUILD_SCORE_AUDIT.json',dict(rows=hsic,layer_counts=dict(Counter(x['layer_id'] for x in hsic)),rule_unchanged=True))
    write(public/'TRAINING_COMPUTE.json',dict(rows=compute,continuation_only=True,loading_initialization_failures_in_GPU_ledger=True))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for filename,source,metrics in [('PURE_PREFIX_CURVE',result,[('native','Fix'),('old_H_eval','Retention')]),('NEW_SOURCE_CURVE',newresults,[('new_H_eval','Retention'),('new_U_eval','Retention')])]:
        fig,axes=plt.subplots(1,2,figsize=(9,3.5))
        for ax,(panel,name) in zip(axes,metrics):
            for arm in ('A','B'):
                points=sorted((r for r in source if r['arm']==arm and r['panel']==panel and r['metric']==name and r['value'] is not None),key=lambda r:r['N'])
                if points:ax.plot([r['N'] for r in points],[100*r['value'] for r in points],'o-',label=arm)
            ax.set(title=panel+' '+name,xlabel='Actual common pure N',ylabel='Correct (%)',ylim=(-2,102))
            if ax.lines:ax.legend()
        fig.suptitle('Stage20 DEV; fully scored points only');fig.tight_layout();fig.savefig(public/(filename+'.png'),dpi=160);plt.close(fig)
    ledger=read(public/'BUDGET_LEDGER.json');gpu=read(public/'GPU_BUDGET_LEDGER.json');used=sum(s.get('seconds',0) for s in gpu['sessions'])
    complete=[n for n in ns if any(r['N']==n for r in result) and all(r['scored']==r['denominator'] for r in result+newresults if r['N']==n)]
    highest=max(complete,default=0)
    brief=f'# Stage20 导师更新\n\n最高已完整评分共同端点 N={highest}；当前阶段：'+('本轮闭环收口' if final else '执行中')+'。A 全部 FACT+冻结 HSIC；B 全部原配方 BalancEdit，无共享背景。\n\n|面板/指标|A|B|\n|---|---:|---:|\n'
    for panel,m in [('native','Fix'),('insertion','accuracy'),('old_anchors','PairCorrect'),('old_H_eval','accuracy'),('old_H_eval','Retention'),('new_H_eval','accuracy'),('new_H_eval','Retention'),('new_U_eval','Retention'),('old_native_text_extension','accuracy')]:
        vals=[]
        for arm in ('A','B'):
            r=next((r for r in result+newresults if r['N']==highest and r['arm']==arm and r['panel']==panel and r['metric']==m),None)
            vals.append('NA' if not r or r['value'] is None else f"{r['value']:.1%} ({r['numerator']:g}/{r['denominator']})")
        brief+=f'|{panel}/{m}|'+ '|'.join(vals)+'|\n'
    brief+=f'\n已关闭 GPU 会话累计 {used:.1f}/28800 秒；新增判定 {ledger["new_judgment_items_dispatched"]}/2000。进行中的 GPU 会话另见账本。原19权重已清理，本轮为新增编辑/评价/登记消融重建，成本计入。最终完整精度 bank 保留；空间或时间边界决定实际共同 N。\n\n旧 H 仅6个不同QA/2图，11锚点重复关联计权，PairCorrect不能代表广泛H保护。新来源单独报告，原始材料存在历史项目曝光，不称完全未见；患者/研究独立性 UNKNOWN，无临床签署。小规模纯流不作长序列或临床确认，组合差异不单独归因H/HSIC。HSIC五个相关wrapper不是独立事实，全部L30也不自动证明动态选层有效。零分母和缺失评分为NA，未以字符串或NLL替代语义评分。纯文本泛化无合法支持为NA。实际货币费用未由账户CLI暴露，tokens已记录，费用为null而非零。\n'
    (public/'ADVISOR_UPDATE_ZH.md').write_text(brief)
    if (public/'ABLATION_TOKEN_SUPPORT.json').exists() and not read(public/'ABLATION_TOKEN_SUPPORT.json')['EXTRA_full11_supported']:
        with (public/'ADVISOR_UPDATE_ZH.md').open('a') as f:f.write('\n最小消融：固定11条中第2/4条H与G目标token长度档位不匹配（3对7），EXTRA pure11不满足冻结合同，未改答案、换样本或放宽规则；其已完成1条保留为不完整工件，不计算pure11结果。NO_H无G训练槽，可独立完成固定11条；当前状态：'+abstatus+'。失败GPU耗时纳入原一小时消融及八小时总账。\n')
    write(public/'REPORT_STATUS.json',dict(highest_fully_scored_N=highest,final=final,all_requested_outputs_scored=bool(highest) and all(score_key(r['source'],r['output']) in scores for r in allrows if r['mode'] in ('endpoint','insertion'))))
