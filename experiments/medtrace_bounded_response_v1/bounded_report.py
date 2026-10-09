"""Frozen paired gate for a gain-constrained candidate; no output-based selection."""
import json,os
from pathlib import Path
import probe_report as base
r=Path(os.environ['RUN_ROOT']);ARMS=('RAW','BOUNDED','MATCHED_RAW')


def decide(protection,edit,ratios,lost,response_lost):
    if protection['MATCHED_RAW']<=1e-8 or edit['RAW']<=1e-8 or edit['MATCHED_RAW']<=1e-8:
        return 'INCONCLUSIVE_SMALL_RESPONSE'
    passed=(protection['expert_CI'][1]<0 and protection['source_CI'][1]<0
        and edit['BOUNDED']>=.9*edit['RAW'] and all(v>=.9 for v in ratios)
        and lost==0 and response_lost['BOUNDED']<=response_lost['MATCHED_RAW'])
    return 'LOCAL_MECHANISM_SIGNAL' if passed else 'NO_LOCAL_SUPPORT'


def selfcheck():
    p=dict(MATCHED_RAW=1.,expert_CI=[-2.,-1.],source_CI=[-2.,-1.])
    e=dict(RAW=1.,BOUNDED=.95,MATCHED_RAW=.9);lost=dict(BOUNDED=0,MATCHED_RAW=0)
    assert decide(p,e,[.95]*8,0,lost)=='LOCAL_MECHANISM_SIGNAL'
    assert decide(p,e,[.89]+[.95]*7,0,lost)=='NO_LOCAL_SUPPORT'
    assert decide(dict(p,expert_CI=[-1.,0.]),e,[.95]*8,0,lost)=='NO_LOCAL_SUPPORT'
    assert decide(p,e,[.95]*8,1,lost)=='NO_LOCAL_SUPPORT'
    assert decide(dict(p,MATCHED_RAW=0.),e,[.95]*8,0,lost)=='INCONCLUSIVE_SMALL_RESPONSE'


def main():
    selfcheck();data=[base.read(p) for p in sorted((r/'private/results').glob('*.json'))]
    assert len(data)==8 and len({d['expert_order'] for d in data})==8
    assert not list((r/'private').glob('FAILURE*'))
    lock=base.read(r/'private/PROBE_LOCK.json')
    assert sum(d['forward_calls'] for d in data)==lock['forward_calls']==4592
    assert sum(d['backward_calls'] for d in data)==lock['backward_calls']==13144
    assert sum(d['zero_expert_Base_checks'] for d in data)==808
    assert all(d['state_restored_exact'] and d['baseline_repeat_exact'] and not d['Base_gradient'] for d in data)
    assert all(len(d['diagnostics'])==101 for d in data)
    per=[];held=[]
    for d in data:
        h=[q for q in d['diagnostics'] if q['role']=='HELDOUT_FIT' and q['semantic_correct']]
        e=[q for q in d['diagnostics'] if q['role']=='EDIT'];assert len(h)==63 and len(e)==5
        held.extend(h);m=d['function_matching'];g=d['geometry']
        assert abs(m['actual']-m['target'])<=1e-10+1e-3*m['target']
        assert g['actual_linear_edit_progress']>=.9*g['raw_linear_edit_progress']*(1-1e-6)
        assert g['BOUNDED_map_norm']<=g['RAW_map_norm']*(1+1e-6)+1e-10
        progress={a:-sum((.5 if q['index']==0 else .125)*q['candidates'][a]['NLL_change'] for q in e) for a in ARMS}
        assert progress['RAW']>1e-8
        per.append(dict(expert_order=d['expert_order'],edit_progress=progress,
            actual_edit_retention=progress['BOUNDED']/progress['RAW'],
            heldout_KL={a:base.mean([q['candidates'][a]['KL'] for q in h]) for a in ARMS},
            edit_lost_tokens={a:sum(q['candidates'][a]['lost_correct_tokens'] for q in e) for a in ARMS},
            geometry=g,matching=m,basis_audit=d['basis_audit']))
    sources=sorted({q['group'] for q in held});assert len(sources)==29
    protection={a:base.mean([d['heldout_KL'][a] for d in per]) for a in ARMS}
    protection.update(expert_CI=base.interval([d['heldout_KL']['BOUNDED']-d['heldout_KL']['MATCHED_RAW'] for d in per]),
        source_CI=base.interval([base.mean([q['candidates']['BOUNDED']['KL']-q['candidates']['MATCHED_RAW']['KL'] for q in held if q['group']==s]) for s in sources]))
    edit={a:base.mean([d['edit_progress'][a] for d in per]) for a in ARMS}
    lost={a:sum(q['candidates'][a]['lost_correct_tokens'] for q in held) for a in ARMS}
    edit_lost=sum(d['edit_lost_tokens']['BOUNDED'] for d in per)
    decision=decide(protection,edit,[d['actual_edit_retention'] for d in per],edit_lost,lost)
    all_held=[q for d in data for q in d['diagnostics'] if q['role']=='HELDOUT_FIT']
    ledger=base.read(r/'RESOURCE_LEDGER.json');inherited=base.read(r/'private/INHERITED_COST.json')
    assert len(ledger['gpu_sessions'])==6 and all(s.get('ended_epoch') for s in ledger['gpu_sessions'])
    result=dict(status='COMPLETE',decision=decision,protection=protection,edit_progress=edit,
        primary_queries=63,primary_sources=29,response_lost_tokens=lost,edit_lost_tokens=edit_lost,
        all96_secondary={a:dict(KL=base.mean([q['candidates'][a]['KL'] for q in all_held]),
            lost_correct_tokens=sum(q['candidates'][a]['lost_correct_tokens'] for q in all_held)) for a in ARMS},
        per_expert=per,resource=dict(new_GPU_process_hours=(ledger['gpu_seconds_used']-inherited['gpu_seconds_used'])/3600,
            cumulative_GPU_process_hours=ledger['gpu_seconds_used']/3600,cumulative_Judge=ledger['Judge_attempts'],
            forward_calls=4592,backward_calls=13144,temporary_updates=8,new_generations=0,new_Judge=0,new_checkpoints=0),
        proxy_is_full_KL=False,independent_confirmation=False,clinical_protection=False,full_training_success=False)
    base.write(r/'public/RESULTS.json',result)
    base.write(r/'public/COMPLETION_AUDIT.json',dict(status='PASS',experts=8,exact_Base_checks=808,restored_states=8,
        loss_definition_gradient_checks=488,forward_calls=4592,backward_calls=13144,sessions_ended=6))
    base.write(r/'public/RESEARCH_DECISION.json',{k:v for k,v in result.items() if k!='per_expert'})
    (r/'public/REPORT_ZH.md').write_text(f"# 编辑收益约束下的响应最小化\n\n结论：{decision}。\n\n主保护KL及配对区间：{protection}。\n\n真实编辑进展：{edit}。8专家实际编辑保留比例：{[d['actual_edit_retention'] for d in per]}。\n\n优化目标是逐token线性响应变化的二次代理，不是实际全词表KL；实际编辑、KL和token丢失独立评估。当前没有自由生成或临床保护证据，也没有完整训练/24/146结果。\n")
    base.write(r/'private/REPORT_COMPLETE.json',dict(status='COMPLETE',decision=decision))


if __name__=='__main__':main()
