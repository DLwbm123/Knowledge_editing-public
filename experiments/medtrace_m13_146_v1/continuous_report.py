"""One actual bank with historical baseline tables; no synthetic comparator."""
import json
import sqlite3
import time
from pathlib import Path
import continuous_protocol as f
import reporting
import contract

def calculate(tasks,mask,records,scores):
    assert len(records)==len(scores)==9587 and records.keys()==scores.keys()
    panels={};owners={};damage={}
    for mode,prefix in [('single',1),('bank',1),('bank',50),('bank',100),('bank',146)]:
        ts=tasks if mode=='single' else tasks[:prefix];label=f'{mode}_{prefix}';panels[label]={};owners[label]={};protected={}
        for role in ('T0','FIT','T1G','T2G','T1L','T2L'):
            values={}
            for task in ts:
                ids=[q['query_id'] for q in contract.original.fit_rows(task)] if role=='FIT' else list(dict.fromkeys(q for e in task['events'] if e['task']==role for q in e['all_probe_query_ids']))
                if role not in ('T0','FIT'):ids=[q for q in ids if mask[q]==role.endswith('L')]
                keys=[('M13P',mode,prefix,task['order'] if mode=='single' else 0,q) for q in ids]
                if mode=='bank' and role.endswith('L'):keys=[k for k in keys if not records[k]['active_target']]
                values[task['order']]=[scores[k] for k in keys]
            panels[label][role]=reporting.summarize(values);owners[label][role]=values
        for task in ts:
            for q in contract.original.ids([task]):
                key=('M13P',mode,prefix,task['order'] if mode=='single' else 0,q)
                if not records[key]['active_target']:protected[task['order'],q]=(scores['BASE','base',0,0,q],scores[key])
        damage[label]=dict(fresh_Base_correct=sum(b==1 for b,s in protected.values()),new_damage=sum(b==1 and s==0 for b,s in protected.values()),missing=sum(b is None or s is None for b,s in protected.values()))
    insertion={};retention=[]
    for task in tasks:
        q=next(e['all_probe_query_ids'][0] for e in task['events'] if e['task']=='T0')
        v=scores['M13P','insertion',task['order'],task['order'],q];insertion[task['order']]=[v]
        if v==1:retention.append(scores['M13P','bank',146,0,q])
    changes={}
    for prefix in (50,100,146):
        changes[str(prefix)]={}
        for role in ('FIT','T1G','T2G','T1L','T2L'):
            a=owners['single_1'][role];b=owners[f'bank_{prefix}'][role];delta=[];groups=[];same_support=True
            for t in tasks[:prefix]:
                x,y=a[t['order']],b[t['order']]
                if len(x)!=len(y):same_support=False;continue
                if x:delta.append(None if any(v is None for v in x+y) else (sum(y)-sum(x))/len(x));groups.append(t['native']['source_group'])
            changes[str(prefix)][role]=dict(same_support=same_support,macro_change_pp=100*sum(delta)/len(delta) if delta and all(v is not None for v in delta) else None,edit_CI_pp=reporting.interval(delta),source_CI_pp=reporting.interval(delta,groups))
    return dict(panels=panels,protection=damage,insertion=reporting.summarize(insertion),inserted_correct_retention=dict(denominator=len(retention),correct=sum(v==1 for v in retention),missing=sum(v is None for v in retention)),single_to_bank=changes,complete_no_missing=all(v is not None for v in scores.values())),owners

def main():
    import astra_queue as a
    q,run=a.q,a.RUN;root=run/'private'/f.QUEUE;assert (root/'ALL_WORKERS_COMPLETE.json').exists()
    records={};scores={}
    with sqlite3.connect('file:'+str(root/'queue.sqlite')+'?mode=ro',uri=True) as db:
        assert not db.execute("select 1 from payload where status in ('PENDING','RESERVED')").fetchone()
        for path,binding,status,score in db.execute('select c.path,c.output_binding,p.status,p.correct from consumer c join payload p on p.key=c.payload_key'):
            d=q.read(Path(path));assert q.digest(d)==binding
            k=tuple(d[x] for x in ('method','mode','prefix','owner'))+(d['binding']['input']['query_id'],);assert k not in records
            records[k]=d;scores[k]=score if status=='FORMAT_VALID' else None
    result,owners=calculate(q.read(run/'private/ORIGINAL146_TASKS.json'),q.read(run/'private/ORIGINAL_BASE_MASK146.json'),records,scores)
    result.update(scope='Original146 actual cumulative expert bank; exposed DEV, not shared-weight sequential training',historical_baselines=q.read(run/'private/HISTORICAL_BASELINES.json'),comparison_limit='Historical Astra high adapted baselines vs current Astra medium; no matched-epoch superiority or independent confirmation claim')
    q.write(run/'private/CONTINUOUS_FULL_RESULT.json',dict(result,owner_panels=owners))
    result['panels']['bank_1']={'status':'PRIVATE_SMALL_GROUP_AUDIT_ONLY'};result['protection']['bank_1']={'status':'PRIVATE_SMALL_GROUP_AUDIT_ONLY'}
    q.write(run/'public/CONTINUOUS_RESULTS.json',result)
    lines=['# M13P旧课程：146次持续编辑','','| 方法 | T0 | T1G | T2G | T1L | T2L |','|---|---:|---:|---:|---:|---:|']
    roles=('T0','T1G','T2G','T1L','T2L');p=result['panels']['bank_146']
    lines.append('| M13P_540 | '+' | '.join(str(p[k]['macro']) for k in roles)+' |')
    for method in ('BalancEdit_locked_adaptation','BELoRA_V2_effect_repaired'):
        rows=[v for v in result['historical_baselines'] if v['method']==method and v['mode']=='sequential' and int(v['prefix'])==146]
        lines.append('| '+method+' | '+' | '.join(next(v['macro_percent_display'] for v in rows if v['task']==k) for k in roles)+' |')
    lines+=['','完整性：'+str(result['complete_no_missing']),'','插入与保持：'+json.dumps(result['inserted_correct_retention']), '','原Base损伤：'+json.dumps(result['protection']['bank_146']),'','历史基线评分Astra high，当前medium，训练配方和实现不同；只作描述性历史比较。保留BELoRA缺失界；FIT和fresh-Base损伤没有匹配历史资料，不冒充同口径。T1L仅2条/1编辑、T2L原54条/31编辑；支持及active-target排除以JSON实际分母为准。']
    (run/'public/CONTINUOUS_REPORT_ZH.md').write_text('\n'.join(lines)+'\n');q.write(run/'private/CONTINUOUS_REPORT_COMPLETE.json',dict(status='COMPLETE',epoch=time.time()))
if __name__=='__main__':main()
