"""Two-method full report; unavailable M1/M3 controls never become a success."""
import json
import os
from pathlib import Path
import sqlite3
import time
import comparison146 as contract
import reporting


def calculate(tasks,base_mask,records,scores):
    assert len(records)==len(scores)==contract.CONSUMERS
    assert records.keys()==scores.keys()
    panels={m:{} for m in contract.METHODS};owners={};damage={};damage_owners={};contrasts={}
    labels=[('single',1)]+[('bank',n) for n in contract.PREFIXES]
    for mode,prefix in labels:
        ts=tasks if mode=='single' else tasks[:prefix];label=f'{mode}_{prefix}'
        for method in contract.METHODS:
            panels[method][label]={};per_role={};protected={}
            for role in ('T0','FIT','T1G','T2G','T1L','T2L'):
                values={}
                for task in ts:
                    qids=[q['query_id'] for q in contract.original.fit_rows(task)] if role=='FIT' else list(dict.fromkeys(
                        qid for event in task['events'] if event['task']==role for qid in event['all_probe_query_ids']))
                    if role not in ('T0','FIT'):qids=[q for q in qids if base_mask[q]==role.endswith('L')]
                    keys=[(method,mode,prefix,task['order'] if mode=='single' else 0,q) for q in qids]
                    if mode=='bank' and role.endswith('L'):keys=[k for k in keys if not records[k]['active_target']]
                    values[task['order']]=[scores[k] for k in keys]
                per_role[role]=values;panels[method][label][role]=reporting.summarize(values)
            owners[method,label]=per_role
            for task in ts:
                for qid in contract.original.ids([task]):
                    key=(method,mode,prefix,task['order'] if mode=='single' else 0,qid)
                    if not records[key]['active_target']:
                        protected[task['order'],qid]=(scores['BASE','base',0,0,qid],scores[key])
            damage[method,label]=dict(observations=len(protected),fresh_Base_correct=sum(b==1 for b,s in protected.values()),
                new_damage=sum(b==1 and s==0 for b,s in protected.values()),repairs=sum(b==0 and s==1 for b,s in protected.values()),
                missing=sum(b is None or s is None for b,s in protected.values()))
            d=damage[method,label]
            d['correct_retention_rate']=1-d['new_damage']/d['fresh_Base_correct'] if d['fresh_Base_correct'] and not d['missing'] else None
            damage_owners[method,label]={t['order']:dict(
                fresh_Base_correct=sum(b==1 for (owner,q),(b,s) in protected.items() if owner==t['order']),
                new_damage=sum(b==1 and s==0 for (owner,q),(b,s) in protected.items() if owner==t['order']),
                missing=sum(b is None or s is None for (owner,q),(b,s) in protected.items() if owner==t['order'])) for t in ts}
        contrasts[label]={}
        for role in ('FIT','T2G','T2L'):
            left=owners['M13',label][role];right=owners['FISHER_MIN',label][role]
            delta=[];groups=[]
            for task in ts:
                a,b=left[task['order']],right[task['order']];assert len(a)==len(b)
                if a:
                    delta.append(None if any(v is None for v in a+b) else (sum(b)-sum(a))/len(a))
                    groups.append(task['native']['source_group'])
            contrasts[label][role]=dict(FISHER_MIN_minus_M13_macro_pp=100*sum(delta)/len(delta) if delta and all(v is not None for v in delta) else None,
                edit_CI_pp=reporting.interval(delta),source_CI_pp=reporting.interval(delta,groups),
                source_unit='Original source groups; not independently verified patients',
                FISHER_MIN_preserves_each_M13_owner=reporting.preserved(left,right))
        supported=all(damage[m,label]['fresh_Base_correct']>0 and damage[m,label]['missing']==0 for m in contract.METHODS)
        contrasts[label]['FISHER_MIN_damage_no_worse_than_M13_per_owner']=reporting.damage_preserved(
            damage_owners['M13',label],damage_owners['FISHER_MIN',label]) if supported else None
    for method in contract.METHODS:
        assert panels[method]['bank_146']['T2L']['observations']==54
        assert panels[method]['bank_146']['T2L']['edit_units']==31
    complete=all(v is not None for v in scores.values())
    result=dict(status='SCORING_RESOLVED',complete_no_missing=complete,panels=panels,
        paired_native_FIT160_contrasts=contrasts,protection={m+'_'+l:d for (m,l),d in damage.items()},
        original_M1_fit_T2G_preservation146=None,original_M3_damage_preservation146=None,
        original_joint_scientific_gate_confirmed=False,
        missing_consumers=sum(v is None for v in scores.values()),original_T2L_probes=54,original_T2L_edits=31,
        unavailable_controls_not_replaced_by_historical_recipe=True,independent_confirmation=False,medical_safety_conclusion=False)
    return result,dict(role_panels={m+'_'+l:v for (m,l),v in owners.items()},
        damage={m+'_'+l:v for (m,l),v in damage_owners.items()})


def main():
    import astra_queue as a
    q,RUN=a.q,a.RUN;queue=RUN/'private/judge_native146_two_methods_astra_medium'
    assert q.read(RUN/'private/FINAL_COMPARISON_ADMISSION.json')['status']=='PASS'
    assert (queue/'ALL_WORKERS_COMPLETE.json').exists()
    assert all((queue/'workers'/str(i)/'SCORER_DONE.json').exists() for i in range(4))
    db=sqlite3.connect('file:'+str(queue/'queue.sqlite')+'?mode=ro',uri=True)
    assert not db.execute("SELECT 1 FROM payload WHERE status IN ('PENDING','RESERVED')").fetchone()
    records={};scores={}
    for path,binding,status,score in db.execute('SELECT c.path,c.output_binding,p.status,p.correct FROM consumer c JOIN payload p ON p.key=c.payload_key'):
        data=q.read(Path(path));assert q.digest(data)==binding
        key=tuple(data[k] for k in ('method','mode','prefix','owner'))+(data['binding']['input']['query_id'],)
        assert key not in records;records[key]=data;scores[key]=score if status=='FORMAT_VALID' else None
    result,owners=calculate(q.read(RUN/'private/ORIGINAL146_TASKS.json'),q.read(RUN/'private/ORIGINAL_BASE_MASK146.json'),records,scores)
    ledger=q.read(RUN/'RESOURCE_LEDGER.json');prior=q.read(RUN/'private/INHERITED_COST.json')
    assert all(v.get('ended_epoch') for v in ledger['gpu_sessions'])
    result['cost']=dict(new_GPU_process_hours=(ledger['gpu_seconds_used']-prior['gpu_seconds_used'])/3600,
        cumulative_GPU_process_hours=ledger['gpu_seconds_used']/3600,new_Judge=ledger['Judge_attempts']-prior['Judge_attempts'],
        cumulative_Judge=ledger['Judge_attempts'])
    result['historical_reference_audit']=q.read(RUN/'private/HISTORICAL_146_REUSE_AUDIT1.json')
    q.write(RUN/'private/NATIVE146_RESULTS_WITH_OWNER_AUDIT.json',dict(result,owner_panels=owners))
    public=json.loads(json.dumps(result));public['historical_reference_audit'].pop('epoch',None)
    for method in contract.METHODS:
        public['panels'][method]['bank_1']={'status':'PRIVATE_SMALL_GROUP_AUDIT_ONLY'}
        public['protection'][method+'_bank_1']={'status':'PRIVATE_SMALL_GROUP_AUDIT_ONLY'}
    public['paired_native_FIT160_contrasts']['bank_1']={'status':'PRIVATE_SMALL_GROUP_AUDIT_ONLY'}
    q.write(RUN/'public/NATIVE146_RESULTS.json',public)
    lines=['# 原146两方法完整开发验证','','M13与FISHER_MIN使用同原图＋FIT160配方；完整无缺失：'+str(result['complete_no_missing']),
        '','| 方法 | FIT宏平均 | T2G宏平均 | T2L宏平均 | 新损伤 / fresh Base正确 |','|---|---:|---:|---:|---:|']
    for method in contract.METHODS:
        p=result['panels'][method]['bank_146'];d=result['protection'][method+'_bank_146']
        lines.append(f"| {method} | {p['FIT']['macro']} | {p['T2G']['macro']} | {p['T2L']['macro']} | {d['new_damage']} / {d['fresh_Base_correct']} |")
    lines+=['','single及各prefix、逐编辑保留、双区间和缺失界见匿名JSON，owner明细保持私有。',
        '没有同配方M1/M3的146参照，其原兼顾门槛未核实，不能宣称原联合成功。旧TT88/归一化LoRA是不同预训练配方的历史参考；缺失保留，不重判。',
        '全部已暴露DEV，无独立确认或医学安全结论。']
    (RUN/'public/NATIVE146_REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    q.write(RUN/'private/NATIVE146_REPORT_COMPLETE.json',dict(status='COMPLETE',complete_no_missing=result['complete_no_missing'],epoch=time.time()))


if __name__=='__main__':main()
