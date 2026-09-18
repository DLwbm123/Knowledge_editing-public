"""Frozen isolated Judge consumption and coverage-aware paired FASTTRACK reports."""
from collections import Counter
import csv
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.medtrace.astra_judge_bundle import read,write_new
from scripts.medtrace.stage17_prepare import digest,PROTOCOL
from scripts.medtrace.stage18_score import query_id,score_key
from scripts.medtrace.stage19_fasttrack_budget import write


def judge(root,records,name,qualification=False):
    from scripts.medtrace.stage18_base_judge import run
    root=Path(root);p=root/'private';ledger=read(root/'public/BUDGET_LEDGER.json')
    cache_path=p/'QUALIFIED_SCORE_CACHE.json';cache=read(cache_path if cache_path.exists() else p/'EXISTING_SCORE_CACHE.json')['scores']
    unique={score_key(r['source'],r['output']):dict(query_id=score_key(r['source'],r['output']),source=r['source'],output=dict(raw_answer=r['output']['raw_answer'])) for r in records}
    novel=[r for k,r in unique.items() if k not in cache]
    count=len(novel)
    limit=ledger['new_judgment_limit']
    if ledger['new_judgment_items_dispatched']+count>limit or qualification and ledger['qualification_items_dispatched']+count>ledger['qualification_limit']:
        raise ValueError('New item budget exceeded before dispatch')
    # Mandatory endpoint reservations are charged symmetrically before optional calls.
    if not qualification and name=='prefix11' and ledger['new_judgment_items_dispatched']+count+2*(len(read(p/'STREAM.json')['tasks'])+26)>limit:
        raise ValueError('Prefix would consume paired final endpoint reserve')
    source=dict(scope='FASTTRACK_'+name,records=novel)
    input_path=p/(name+'_JUDGE_INPUT.json');bundle=p/(name+'_JUDGE')
    receipt=p/(name+'_SCORE_RECEIPT.json')
    if receipt.exists():raise ValueError('Existing score receipt; no repeated semantic call')
    write_new(input_path,source)
    ledger['new_judgment_items_dispatched']+=count
    if qualification:ledger['qualification_items_dispatched']+=count
    ledger['exact_reuse_count']+=len(unique)-count
    item=dict(name=name,items=count,reused=len(unique)-count,input_binding=digest(source),status='DISPATCHED')
    ledger['judgment_batches'].append(item);write(root/'public/BUDGET_LEDGER.json',ledger)
    if novel:
        run(dict(base_outputs=str(input_path),bundle=str(bundle),N_queries=count,authorization='USER_AUTHORIZED_ASTRA_DEV_REVIEW',cli='/Applications/ChatGPT.app/Contents/Resources/codex'))
        verdict=read(bundle/'operator/VERDICTS.json')
        if verdict['protocol']!=PROTOCOL or verdict['source_binding']!=digest(source):raise ValueError('Judge lineage changed')
        by={r['opaque_query_id']:r['is_correct'] for r in verdict['decisions']}
        for row in novel:cache[row['query_id']]=by[digest(row)]
        evidence=read(bundle/'operator/execution_evidence/b000.json');item['usage']=evidence.get('usage');item['execution_binding']=digest(evidence)
    item['status']='COMPLETE';write(root/'public/BUDGET_LEDGER.json',ledger)
    write(cache_path,dict(scores=cache,protocol=PROTOCOL))
    write_new(receipt,dict(new=count,reused=len(unique)-count,source_binding=digest(source)))
    return cache


def qualification(root):
    root=Path(root);records=read(root/'private/FRESH_BASE_OUTPUTS.json')['records']
    old={r['query_id']:r for r in read(root/'private/EXISTING_BASE_OUTPUTS.json')['records']}
    if any(r['output']['binding']!=old[r['query_id']]['output']['binding'] for r in records):
        raise ValueError('Runtime migration changed the input/generation binding')
    report=dict(queries=len(records),tokens_identical=sum(r['output']['raw_token_ids']==old[r['query_id']]['output']['raw_token_ids'] for r in records),raw_text_identical=sum(r['output']['raw_answer']==old[r['query_id']]['output']['raw_answer'] for r in records),runtime_binding=read(root/'private/FRESH_BASE_OUTPUTS.json')['runtime'])
    judge(root,records,'qualification',qualification=True)
    write_new(root/'public/MIGRATION_PARITY.json',report)


def endpoint_rows(root):
    rows=[json.loads(line) for line in (Path(root)/'private/OUTPUTS.jsonl').read_text().splitlines()]
    return [r for r in rows if r['mode']=='endpoint']


def score_endpoint(root,prefix,final=False):
    root=Path(root);stream=read(root/'private/STREAM.json');rows=[r for r in endpoint_rows(root) if r['prefix']==prefix]
    expected=prefix+26
    if Counter(r['arm'] for r in rows)!=dict(A=expected,B=expected):raise ValueError('Incomplete paired endpoint')
    if len({(r['arm'],r['query_id']) for r in rows})!=2*expected:raise ValueError('Repeated endpoint input')
    if not final and prefix==50:
        included={query_id(t['native']) for t in stream['tasks'][:11]}|{query_id(r) for r in stream['core_rows'] if r['role'] in ('H_eval','U_eval')}
        rows=[r for r in rows if r['query_id'] in included]
    judge(root,rows,('final' if final else 'prefix')+str(prefix))


def report(root):
    root=Path(root);p=root/'private';public=root/'public';stream=read(p/'STREAM.json');done=read(public/'GENERATED.json')
    rows=endpoint_rows(root);scores=read(p/'QUALIFIED_SCORE_CACHE.json')['scores'];tasks=stream['tasks'];results=[]
    def correct(r):return scores.get(score_key(r['source'],r['output']))
    def base_correct(r):return scores.get(score_key(r['source'],r['Base']))
    def result(arm,n,metric,values,total,unit='QA'):
        known=[v for v in values if v is not None]
        results.append(dict(arm=arm,N=n,K=min(11,n) if stream['track']=='S' else n,track=stream['track'],metric=metric,numerator=sum(known),denominator=total,scored=len(known),coverage=len(known)/total if total else None,
                            value=sum(known)/total if total and len(known)==total else None,unit=unit,status='COMPLETE' if len(known)==total else 'PARTIALLY_SCORED'))
    by={}
    for n in done['prefixes']:
        for arm in ('A','B'):
            selected=[r for r in rows if r['prefix']==n and r['arm']==arm];d={r['query_id']:r for r in selected};by[arm,n]=d
            if len(d)!=n+26:raise ValueError('Final reporting endpoint is incomplete')
            native=[d[query_id(t['native'])] for t in tasks[:n]]
            result(arm,n,'native_accuracy',[correct(r) for r in native],len(native))
            h=[d[query_id(r)] for r in stream['core_rows'] if r['role']=='H_eval']
            u=[d[query_id(r)] for r in stream['core_rows'] if r['role']=='U_eval']
            text=[d[query_id(r)] for r in stream['core_rows'] if r['role']=='native_text_extension']
            result(arm,n,'H_accuracy',[correct(r) for r in h],len(h))
            for name,baseline in [('H_retention',True),('H_fix',False)]:
                subset=[r for r in h if base_correct(r) is baseline];result(arm,n,name,[correct(r) for r in subset],len(subset))
            inserted={t['canonical_edit_id'] for t in tasks[:n]}
            converted={r['query_id'] for r in stream['role_transitions'] if set(r['U_scope_overlap_edits'])&inserted}
            strict_u=[r for r in u if r['query_id'] not in converted and base_correct(r) is True]
            result(arm,n,'U_retention',[correct(r) for r in strict_u],len(strict_u))
            result(arm,n,'U_fixed_panel_accuracy',[correct(r) for r in u],len(u))
            result(arm,n,'native_rephrase_accuracy',[correct(r) for r in text],len(text))
            pairs=[]
            for package in stream['anchor_packages'][:min(11,n)]:
                native_score=correct(d[query_id(package['training']['native'])]);hs=[correct(d[query_id(r)]) for r in package['evaluation'] if r['role']=='H_eval']
                pairs.append(native_score*sum(hs)/len(hs) if native_score is not None and all(v is not None for v in hs) else None)
            result(arm,n,'anchor_PairCorrect',[v for v in pairs],len(pairs),'edit_macro')
    with (public/'RESULTS.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
    write(public/'RESULTS.json',dict(results=results,methods=dict(A='C_FACT+HSIC Top1 anchors',B='BalancEdit original anchors'),N=done['N'],K=done['K'],track=stream['track'],patient_study='UNKNOWN',clinical_signoff=False))
    diagnostics=[]
    for n in done['prefixes']:
        for arm in ('A','B'):
            d=by[arm,n];fixed={query_id(t['native']) for t in tasks[:min(11,n)]}|{query_id(r) for r in stream['core_rows']}
            subset=[d[q] for q in fixed];counts=Counter('BASE' if not r['route']['activated'] else 'ANCHOR' if r['route']['logical_edit_id'] in stream['anchors'] else 'NON_ANCHOR' if stream['track']=='P' else 'BACKGROUND' for r in subset)
            item=dict(arm=arm,N=n,queries=len(subset),selected=dict(counts),all_endpoint_selected_kinds=dict(Counter(r['selected_kind'] for r in d.values())),switch_is_not_necessarily_error=True)
            if n!=done['prefixes'][0]:
                first=by[arm,done['prefixes'][0]];common=set(first)&set(d);transitions=Counter();switched_transitions=Counter();switches=0
                for q in common:
                    a,b=correct(first[q]),correct(d[q]);changed=first[q]['route']['logical_edit_id']!=d[q]['route']['logical_edit_id'];switches+=changed
                    if a is not None and b is not None:
                        transition=str(bool(a))+'_to_'+str(bool(b));transitions[transition]+=1
                        if changed:switched_transitions[transition]+=1
                item.update(expert_switches=switches,common_inputs=len(common),scored_correctness_transitions=dict(transitions),switched_only_correctness_transitions=dict(switched_transitions))
            diagnostics.append(item)
    write(public/'ROUTE_COMPETITION.json',dict(rows=diagnostics,oracle_matrix_run=False))
    comparisons=[]
    for n in done['prefixes']:
        a,b=by['A',n],by['B',n]
        comparisons.append(dict(N=n,inputs=len(a),same_logical_expert=sum(a[q]['route']['logical_edit_id']==b[q]['route']['logical_edit_id'] for q in a),
            same_weight_binding=sum(a[q]['weight_binding']==b[q]['weight_binding'] for q in a),
            same_text=sum(a[q]['output']['raw_answer']==b[q]['output']['raw_answer'] for q in a),
            same_tokens=sum(a[q]['output']['raw_token_ids']==b[q]['output']['raw_token_ids'] for q in a),
            same_correctness=sum(correct(a[q])==correct(b[q]) for q in a)))
    write(public/'OUTPUT_AGREEMENT.json',dict(rows=comparisons,natural_routing=True))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for filename,metrics in [('native_curve',['native_accuracy']),('anchor_h_curve',['anchor_PairCorrect','H_retention','H_fix']),('u_generalization_curve',['U_retention','native_rephrase_accuracy'])]:
        fig,ax=plt.subplots(figsize=(7,4))
        for arm in ('A','B'):
            for metric in metrics:
                points=[r for r in results if r['arm']==arm and r['metric']==metric and r['value'] is not None]
                if points:ax.plot([r['N'] for r in points],[100*r['value'] for r in points],marker='o',label=arm+' '+metric)
        ax.set(xlabel='Actual expert-bank N',ylabel='Correct (%)',ylim=(-2,102),title='DEV; only fully scored points; lines guide the eye')
        ax.legend(fontsize=8);fig.tight_layout();fig.savefig(public/(filename+'.png'),dpi=160);plt.close(fig)
    ledger=read(public/'BUDGET_LEDGER.json');gpu=read(public/'GPU_BUDGET_LEDGER.json');ledger['gpu_seconds_used']=sum(s['seconds'] for s in gpu['sessions']);ledger['gpu_sessions']=gpu['sessions'];write(public/'BUDGET_LEDGER.json',ledger)
    final={(r['arm'],r['metric']):r for r in results if r['N']==done['N']}
    val=lambda a,m:'NA' if final[a,m]['value'] is None else f"{100*final[a,m]['value']:.2f}% ({final[a,m]['scored']}/{final[a,m]['denominator']} scored)"
    brief=f"""# Stage19 FASTTRACK 导师汇报\n\n本轮完成共同端点 N={done['N']}，轨道 {stream['track']}，FACT 覆盖 K={done['K']}。A 为 FACT+HSIC Top1，B 为原配方 BalancEdit；S 轨其余专家为两臂精确共享、只训练一次的 C_NO_H-R4/L21。不能将总库 N 写成 N 次完整 FACT 编辑。\n\n|指标|A|B|\n|---|---:|---:|\n"""
    for label,m in [('全 native 正确性','native_accuracy'),('11 锚点 PairCorrect','anchor_PairCorrect'),('H 保持','H_retention'),('H 修复','H_fix'),('U 保持','U_retention'),('未训练改写','native_rephrase_accuracy')]:brief+=f'|{label}|{val("A",m)}|{val("B",m)}|\n'
    a=final['A','anchor_PairCorrect']['value'];b=final['B','anchor_PairCorrect']['value']
    if a is not None and b is not None:
        brief+=f'\n共同终点的锚点 PairCorrect 差为 {(a-b)*100:+.2f} 个百分点。该值只描述本 DEV 面板，尚不足以确认跨来源稳健收益；具体失败比例见完整正确性与保持率。\n'
    brief+=f"\nGPU 累计 {ledger['gpu_seconds_used']:.1f}/28800 秒；新判定 {ledger['new_judgment_items_dispatched']}/500 项，其中资格 {ledger['qualification_items_dispatched']}/100 项。插入时输出全部保存，未逐项判定的写入率不作全量语义指标。真实曲线仅绘完整评分点；路由切换与正确性转移单独报告。\n\n这是探索性 DEV，H 仅6 QA/2图，患者与研究独立性 UNKNOWN，不作患者级显著性结论。旧 DEV16 +25pp 完全依赖未核实参考，统一排除后四方法相同；此处不沿用该优势宣传。新结果是组合系统比较，不能单独归因于 H 或 HSIC；零空间、低曲率、轨迹不变性未接入。\n\n已达共同端点结果无论正负均保留。公开表完整列出分母与覆盖；原始QA、图像、回答、tokens、权重及身份映射只私有保存。\n"
    (public/'ADVISOR_BRIEF_ZH.md').write_text(brief)
    if stream['track']=='P':
        brief=brief.replace('Stage19 FASTTRACK 导师汇报','Stage19 小规模纯流对照：导师汇报')
        brief=brief.replace('A 为 FACT+HSIC Top1，B 为原配方 BalancEdit；S 轨其余专家为两臂精确共享、只训练一次的 C_NO_H-R4/L21。不能将总库 N 写成 N 次完整 FACT 编辑。','A 的全部编辑为 FACT+HSIC Top1，B 的全部编辑为原配方 BalancEdit；不存在共享背景writer。固定评价锚点为11条，此处仅为小规模纯流，不是长序列确认。')
        brief=brief.replace('/500 项',f"/{ledger['new_judgment_limit']} 项").replace('/100 项',f"/{ledger['qualification_limit']} 项")
        brief+='\nPairCorrect先在每个固定锚点内平均关联H正确性，再跨11锚点平均；6个unique H QA重复关联17个槽位，不能当成17个独立H样本。H unique-QA正确性及保持/Fix必须与计权结果并列，不仅凭PairCorrect宣称广泛H保护优势。\n'
        for n in done['prefixes']:
            for arm in ('A','B'):
                h=next(r for r in results if r['N']==n and r['arm']==arm and r['metric']=='H_accuracy')
                brief+=f"N={n}，{arm}：H unique-QA {h['numerator']}/{h['denominator']}，评分覆盖 {h['scored']}/{h['denominator']}。\n"
        (public/'ADVISOR_BRIEF_ZH.md').write_text(brief)
    write(public/'CLOSEOUT.json',dict(status='SCORED_REPORTED_PUBLICATION_PENDING',N=done['N'],K=done['K'],rows=len(results),code=read(p/'DISPATCH_train.json')['code_commit']))


if __name__=='__main__':
    operation=sys.argv[1];root=Path(sys.argv[2])
    if operation=='qualification':qualification(root)
    elif operation=='score':score_endpoint(root,int(sys.argv[3]),final=len(sys.argv)>4)
    elif operation=='report':report(root)
    else:raise ValueError('Unknown closeout operation')
