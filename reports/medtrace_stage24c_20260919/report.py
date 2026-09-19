"""Frozen DEV79 panels, source-level metrics and explicit promotion gates."""
import sys,json
from pathlib import Path
from statistics import mean
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.medtrace.stage19_fasttrack_budget import read,write
from scripts.medtrace.stage20_closeout import outputs
from scripts.medtrace.stage18_score import query_id,score_key
from scripts.medtrace.prepare_stage2_sources import normalized,reviewed_attribute
from reports.medtrace_stage22_20260919.report import metric


def report(root,common):
    root=Path(root);common=Path(common);scores=read(common/'private/QUALIFIED_SCORE_CACHE.json')['scores'];rows=outputs(root);stream=read(common/'private/STREAM.json');panel=read(root/'private/PANEL_MEMBERSHIPS.json');union=read(root/'private/DEV_UNION.json');expected={query_id(r) for r in union['rows']};result={};by={}
    def clean(x):return {k:v for k,v in x.items() if k!='per_source'}
    def correct(r):return scores.get(score_key(r['source'],r['output']))
    def unrelated(r):
        q=r['source']['question'];a=reviewed_attribute(q)
        return not any(normalized(q)==normalized(t['native']['question']) or a is not None and a==reviewed_attribute(t['native']['question']) for t in stream['tasks'][:19])
    for arm in ('E0','E2','R_H'):
        rr=[r for r in rows if r['arm']==arm and r['mode']=='DEV79'];d={r['query_id']:r for r in rr}
        if len(rr)!=79 or set(d)!=expected or any(correct(r) is None for r in rr):continue
        by[arm]=d;out={}
        for name,ids in panel.items():
            sub=[d[q] for q in ids]
            out[name+'_accuracy']=clean(metric(sub,scores))
            if name=='native':out['native_Fix']=clean(metric(sub,scores,False))
            if 'H' in name or 'U' in name:
                out[name+'_Retention']=clean(metric(sub,scores,True));out[name+'_Fix']=clean(metric(sub,scores,False))
                if 'U' in name:out[name+'_strict_Retention']=clean(metric([r for r in sub if unrelated(r)],scores,True))
        pairs=[]
        for pack in stream['anchor_packages'][:11]:
            ns=correct(d[query_id(pack['training']['native'])]);h=[correct(d[query_id(x)]) for x in pack['evaluation'] if x['role']=='H_eval'];pairs.append(float(ns)*mean(h))
        out['PairCorrect11']=dict(value=mean(pairs),N=11,unit='edit_macro_not_independent_H');result[arm]=out
    comparisons=[]
    if set(by)=={'E0','E2','R_H'}:
        for control in ('E0','E2'):
            a,b=by['R_H'],by[control];changes={}
            for name,ids in panel.items():
                count={f'{i}->{j}':0 for i in (0,1) for j in (0,1)}
                for q in ids:count[f'{int(correct(b[q]))}->{int(correct(a[q]))}']+=1
                changes[name]=count
            comparisons.append(dict(control=control,orientation='control to R_H',same_route=sum((a[q]['route']['logical_edit_id'],a[q]['route']['activated'])==(b[q]['route']['logical_edit_id'],b[q]['route']['activated']) for q in a),different_tokens=sum(a[q]['output']['raw_token_ids']!=b[q]['output']['raw_token_ids'] for q in a),correctness_transitions=changes))
        a,b=result['R_H'],result['E2'];comparands=[]
        for name in panel:
            if 'H' in name:comparands.extend([(name+'_Retention','source_macro'),(name+'_accuracy','accuracy')])
            elif 'U' in name:comparands.append((name+'_strict_Retention','source_macro'))
            elif 'rewrite' in name or 'positive' in name:comparands.append((name+'_accuracy','accuracy'))
        noninferior=all(a[k][v] is not None and b[k][v] is not None and a[k][v]>=b[k][v] for k,v in comparands)
        improved=any(a[k][v]>b[k][v] for k,v in comparands if a[k][v] is not None and b[k][v] is not None and ('Retention' in k or 'rewrite' in k))
        gate=dict(native19=a['native_accuracy']['correct']==19,all_panels_noninferior=noninferior,strict_improvement=improved,DEV_engineering_qualified=a['native_accuracy']['correct']==19 and noninferior and improved,new_CONFIRM='INSUFFICIENT_NEW_CONFIRM_SUPPORT',stage25_started=False,tie_prefers='E2')
    else:gate=dict(status='WAIT_COMPLETE_SCORED_ARMS',stage25_started=False)
    insert=[r for r in rows if r['arm']=='R_H' and r['mode']=='insertion'];insertion=clean(metric(insert,scores))
    write(root/'public/RESULTS.json',dict(N=19,exposure='VIEWED_DEV',patient_study='UNKNOWN',arms=result,insertion=insertion,promotion=gate,comparisons=comparisons,p_values=None))
    write(root/'public/REPORT_STATUS.json',dict(complete_arms=list(result),complete=set(result)=={'E0','E2','R_H'} and len(insert)==19 and insertion['coverage']==19,promotion=gate))
    if set(result)=={'E0','E2','R_H'}:
        lines=['# Stage24C 固定R2 HSIC纯19对照','', '历史E0/E2复用权重，新R_H使用修复后的确定性后端；本轮不是完整重训的严格HSIC单因素因果对照。全部结果为已查看DEV；Stage25未启动，不能据此声称独立确认或临床验证。','', '| 臂 | native | DEV H正确 | DEV H保持 | DEV U严格保持 |','|---|---:|---:|---:|---:|']
        for arm,a in result.items():
            val=lambda k:f"{a[k]['correct']}/{a[k]['N']}"
            lines.append(f"| {arm} | {val('native_accuracy')} | {val('DEV_H_accuracy')} | {val('DEV_H_Retention')} | {val('DEV_U_strict_Retention')} |")
        lines+=['','原6H/9U/11改写与DEV16H/16U/正向面板分开报告。PairCorrect11为编辑计权，不代替H unique-QA或来源宏结果。', '', '工程晋级门槛：'+json.dumps(gate,ensure_ascii=False),'','所有有效臂、正负结果和完整评分覆盖均保留；尺度稳定性检查在1/80/160/320步执行，失败即停。']
        (root/'public/ADVISOR_UPDATE_ZH.md').write_text('\n'.join(lines)+'\n')
    return read(root/'public/REPORT_STATUS.json')
