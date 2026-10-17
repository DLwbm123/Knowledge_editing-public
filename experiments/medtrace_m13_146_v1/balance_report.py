"""Frozen M13B contrasts and source-only gradient diagnosis, including failures."""
import json
from pathlib import Path
import sqlite3
import statistics
import time
import balance_protocol as f
import selected_report146 as report


def main():
    import astra_queue as a
    q,run=a.q,a.RUN;root=run/'private'/f.QUEUE
    assert (root/'ALL_WORKERS_COMPLETE.json').exists()
    records={};scores={}
    with sqlite3.connect('file:'+str(root/'queue.sqlite')+'?mode=ro',uri=True) as db:
        assert not db.execute("select 1 from payload where status in ('PENDING','RESERVED')").fetchone()
        for path,binding,status,score in db.execute('select c.path,c.output_binding,p.status,p.correct from consumer c join payload p on p.key=c.payload_key'):
            d=q.read(Path(path));assert q.digest(d)==binding
            k=tuple(d[x] for x in ('method','mode','prefix','owner'))+(d['binding']['input']['query_id'],)
            assert k not in records;records[k]=d;scores[k]=score if status=='FORMAT_VALID' else None
    out={'scope':'Original146 single-expert DEV; one source-only Fisher balancing candidate; not independent confirmation','contrasts':{}};audits={}
    candidate,reference=f.CANDIDATE,f.REFERENCE
    out.update(candidate=candidate,reference=reference)
    for left,right in [('C','M13')]+[(m,candidate) for m in f.METHODS if m!=candidate]:
        subset={k:v for k,v in records.items() if k[0] in (left,right,'BASE')}
        result,owners=report.calculate(q.read(run/'private/ORIGINAL146_TASKS.json'),q.read(run/'private/ORIGINAL_BASE_MASK146.json'),subset,{k:scores[k] for k in subset},methods=(left,right),single_only=True)
        key=left+'_to_'+right;out['contrasts'][key]=result;audits[key]=owners
    primary=out['contrasts'][reference+'_to_'+candidate];terms=primary['paired_native_FIT160_contrasts']['single_1']
    out['primary_gate']=dict(complete_no_missing=primary['complete_no_missing'],FIT=terms['FIT'][candidate+'_preserves_each_'+reference+'_owner'],T2L=terms['T2L'][candidate+'_preserves_each_'+reference+'_owner'],T2G_improves=terms['T2G'][candidate+'_minus_'+reference+'_macro_pp'] is not None and terms['T2G'][candidate+'_minus_'+reference+'_macro_pp']>0,damage=terms[candidate+'_damage_no_worse_than_'+reference+'_per_owner'])
    out['primary_gate']['passed']=all(v is True for v in out['primary_gate'].values())
    old=q.read(Path(f.admit(run)['evaluation_parent'])/'public'/f.PARENT_REPORT)
    for m in ('C','M13'):assert out['contrasts']['C_to_M13']['panels'][m]['single_1']==old['contrasts']['C_to_M13']['panels'][m]['single_1']
    curves=[q.read(run/'private/results'/candidate/f'{i}.json')['curve'] for i in range(1,147)]
    if f.DIAGNOSE_ENDPOINT:
        baseline=[q.read(run/'private/baseline_gradient_diagnostics'/f'{i}.json') for i in range(1,147)]
        out['mechanism']=dict(baseline_endpoint_owners=146,baseline_cap_would_activate=sum(d['active'] for d in baseline),
            baseline_source_conflict_groups=sum(g['uncapped_source_dot']<0 for d in baseline for g in d['groups']),
            baseline_median_scale=statistics.median(d['scale'] for d in baseline),
            new_steps=23360,new_cap_active_steps=sum(s['balance']['active'] for curve in curves for s in curve),
            new_median_scale=statistics.median(s['balance']['scale'] for curve in curves for s in curve),
            limitation='Endpoint diagnostics are source-only, not direct T2G gradients or proof of causal mediation; no finite-step locality guarantee')
    else:
        assert primary['panels'][reference]==old['contrasts']['M13_to_M13B']['panels'][reference]
        groups=[g for curve in curves for s in curve for g in s['anti_parallel']['groups']]
        out['mechanism']=dict(parent_trajectory_summary=q.read(run/'private/PARENT_TRAJECTORY_SUMMARY.json'),
            new_steps=23360,projected_groups=sum(g['removed'] for g in groups),
            minimum_source_dot_ratio_after=min(g['source_dot_ratio_after'] for g in groups if g['source_dot_ratio_after'] is not None),
            zero_directions=sum(s['matching'].get('status')=='ZERO_DIRECTION' for curve in curves for s in curve),
            limitation='Source-only pre-clipping and pre-matching geometry; not finite-step generalization or locality guarantee')
    q.write(run/'public/BALANCE_RESULTS.json',out);q.write(run/'private/BALANCE_OWNER_AUDIT.json',audits)
    lines=['# '+candidate+'：完整有限实验','','联合门槛：'+str(out['primary_gate']),'', '| 方法 | FIT | T2G | T2L | damage / Base正确 |','|---|---:|---:|---:|---:|']
    for m in f.METHODS:
        pair=reference+'_to_'+candidate if m in (reference,candidate) else m+'_to_'+candidate
        panel=out['contrasts'][pair]['panels'][m]['single_1'];d=out['contrasts'][pair]['protection'][m+'_single_1']
        lines.append('| '+m+' | '+' | '.join(str(panel[k]['macro']) for k in ('FIT','T2G','T2L'))+' | '+str(d['new_damage'])+' / '+str(d['fresh_Base_correct'])+' |')
    lines+=['','T2G配对差与区间：'+json.dumps(terms['T2G'],ensure_ascii=False),'','Source机制诊断：'+json.dumps(out['mechanism'],ensure_ascii=False),'','已暴露DEV；T2L只有31支持编辑/54观测。不把拟合成功、source梯度性质或T2G正号单独解释为联合成功；无独立确认或医学安全结论。']
    (run/'public/BALANCE_REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    q.write(run/'private/BALANCE_REPORT_COMPLETE1.json',dict(status='COMPLETE',primary_gate=out['primary_gate'],epoch=time.time()))

if __name__=='__main__':main()
