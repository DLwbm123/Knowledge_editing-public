"""Audit fixed NLL alignment and separate initial null headroom from probe response."""
import csv
import json
import math
from pathlib import Path
from .nllmatch import ALPHA


def summarize(root):
    root=Path(root);read=lambda n:json.loads((root/n).read_text())
    result=read('RESULT.json');frozen=read('FROZEN_TARGETS.json');priors=read('PRIORS_AD.json');gate=read('CONTROL_GATE.json')
    assert result['status']=='COMPLETE' and result['denominator']==2
    assert gate['both_before_alignment'] and gate['A_steps']==8 and gate['D_steps']==4 and gate['full_D5_not_replayed']
    states={s['arm']:s for s in result['arms']};assert set(states)=={'A','D'}
    rows=[];probes=[];checks=0
    def verdict(v,start,base,extra=None):
        nonlocal checks
        drift=v['native_nll'][1]-base[1]
        expected=dict(null=abs(drift)<=.01,visual=v['native_nll'][0]<start[0],trust=math.isfinite(v['combined_native_fisher_cost']) and v['combined_native_fisher_cost']<=.001)
        expected.update(extra or {})
        assert v['gates']==expected and v['accepted']==all(expected.values())
        assert abs(v['cumulative_null_change']-drift)<1e-12;checks+=3
    def trial(t,start,base,label,stage_accepted):
        nonlocal checks
        p=t['proposal'];c=t['correction'];main=t['proposal_verdict'];v=t['final_verdict']
        assert p['native_before']==start and p['native_after']==main['native_nll']
        assert (c is not None)==(not main['gates']['null'])
        verdict(main,start,base)
        verdict(v,start,base,dict(valid_correction=False) if c is not None and c['status']!='APPLIED' else None)
        if c is None:assert v['native_nll']==p['native_after']
        elif c['status']=='APPLIED':assert c['native_before']==p['native_after'] and c['native_after']==v['native_nll']
        rows.append(dict(candidate=label,accepted_by_original_gate=v['accepted'],stage_accepted=stage_accepted,start_visual=start[0],main_gain=start[0]-p['native_after'][0],final_visual=v['native_nll'][0],final_gain=start[0]-v['native_nll'][0],cumulative_null=v['cumulative_null_change'],null_margin=.01-abs(v['cumulative_null_change']),native_main_norm=p['quantization']['native_delta_l2'],correction=c is not None,old_trust=v['combined_native_fisher_cost'],requested=p['requested_reduction'],effective=p['effective_reduction']))
        checks+=4
    for label,count in [('A',8),('D',4)]:
        s=states[label];base=s['baseline']['native'];prev=base
        assert s['base_restored'] and len(s['steps'])==count and base==priors[label]['baseline']['native']
        for i,t in enumerate(s['steps']):
            assert t['step']==i+1 and t['start_native_nll']==prev and t['accepted']
            trial(t['trial'],prev,base,f'{label}/reference/{i+1}',True)
            assert t['trial']['final_verdict']['accepted'];prev=t['trial']['final_verdict']['native_nll']
        assert s['endpoint_native_nll']==prev
    a=states['D']['alignment'];t=a['trial'];base=states['D']['baseline']['native'];start=states['D']['endpoint_native_nll'];v=t['final_verdict']
    assert a['alpha']==frozen['alpha']==ALPHA and a['tolerance']==frozen['nll_tolerance']==.005
    assert frozen['target_nll']==states['A']['endpoint_native_nll'][0]
    assert frozen['fifth_full_effective']==priors['D']['steps'][4]['trial']['proposal']['effective_reduction']
    assert a['requested']==frozen['alignment_request']==ALPHA*frozen['fifth_full_effective']
    assert t['requested_reduction']==t['proposal']['requested_reduction']==a['requested']
    assert 'native_norm_matching' not in t['proposal'] and a['start_native_nll']==start
    error=abs(v['native_nll'][0]-frozen['target_nll'])
    assert a['error']==error and a['nll_match_pass']==(error<=.005)
    assert a['accepted']==(v['accepted'] and a['nll_match_pass'])==result['alignment_success']
    assert states['D']['alignment_base_restored']
    trial(t,start,base,'D/new_alignment',a['accepted'])
    if not a['accepted']:
        assert a['rollback_pass'] and a['after_rollback_native_nll']==start
        assert not states['A']['probes'] and not states['D']['probes']
    else:assert states['D']['aligned_endpoint_native_nll']==v['native_nll']
    checks+=10
    for label,s in states.items():
        before=s.get('aligned_endpoint_native_nll',s['endpoint_native_nll']);b=s['baseline']['native']
        for key,probe in s['probes'].items():
            assert key=='matched' and probe['endpoint_restored'] and s['probe_base_restored']
            p=probe['proposal'];v=probe['verdict'];m=p['native_norm_matching'];norm=p['quantization']['native_delta_l2'];err=abs(norm/frozen['probe_native_norm']-1)
            assert p['native_before']==before and p['native_after']==v['native_nll']
            assert m['target']==frozen['probe_native_norm'] and m['applications']==1 and m['tolerance']==.01
            assert m['ratio']==m['target']/m['unscaled_native_norm'] and m['relative_error']==err
            assert m['passed']==(err<=.01 and not p['clipped'])==probe['matching_passed']
            verdict(v,before,b)
            probes.append(dict(arm=label,start_visual=before[0],start_null=before[1],initial_null_drift=before[1]-b[1],initial_null_margin=.01-abs(before[1]-b[1]),native_norm=norm,norm_error=err,matching_pass=m['passed'],gain=before[0]-v['native_nll'][0],incremental_null=v['native_nll'][1]-before[1],absolute_incremental_null=abs(v['native_nll'][1]-before[1]),final_cumulative_null=v['cumulative_null_change'],final_null_margin=.01-abs(v['cumulative_null_change']),original_gate=v['accepted'],gates=v['gates'],old_trust=v['combined_native_fisher_cost']))
            checks+=8
    eligible=a['accepted'] and len(probes)==2 and all(p['matching_pass'] for p in probes)
    comparison=dict(eligible=eligible,target_nll=frozen['target_nll'],candidate_nll=a['trial']['final_verdict']['native_nll'][0],alignment_error=error,nll_tolerance=.005,no_null_baseline_reset=True)
    if eligible:
        q={p['arm']:p for p in probes}
        comparison['D_minus_A']={k:q['D'][k]-q['A'][k] for k in ['gain','initial_null_margin','incremental_null','absolute_incremental_null','final_null_margin']}
    else:comparison['reason']='Alignment qualification or probe norm matching failed; no same-NLL outcome comparison or imputation'
    differences=[v for s in states.values() for c in s['checks'] for k,v in c.items() if k.endswith('difference')]
    summary=dict(status='COMPLETE',endpoint_denominator=2,reference_main=12,alignment_candidates=1,alignment_original_gate_pass=a['trial']['final_verdict']['accepted'],alignment_match_pass=a['nll_match_pass'],alignment_accepted=a['accepted'],alignment_target=frozen['target_nll'],alignment_candidate_nll=a['trial']['final_verdict']['native_nll'][0],alignment_error=error,probe_candidates=len(probes),comparison_eligible=eligible,corrections=sum(r['correction'] for r in rows),reference_checks=sum(len(s['checks']) for s in states.values()),reference_max_difference=max(differences),elapsed_seconds=result['elapsed_seconds'],model_forward_calls=result['model_forward_calls'],peak_allocated_bytes=result['peak_allocated_bytes'],all_base_restored=result['all_base_restored'],half_candidates=result['half_candidates'],full16_runs=result['full16_runs'],generation_calls=result['generation_calls'],judge_calls=result['judge_calls'])
    assert result['probe_count']==len(probes) and result['all_base_restored']
    assert result['half_candidates']==result['full16_runs']==result['generation_calls']==result['judge_calls']==0
    full=priors['D']['steps'][4]['trial']['proposal'];partial=a['trial']['proposal']
    predicted_fp32=full['fp32_before'][0]+ALPHA*(full['fp32_after'][0]-full['fp32_before'][0])
    diagnostics=dict(posthoc_saved_scalars_only=True,
        raw_candidate_nll=partial['native_after'][0],raw_candidate_target_error=abs(partial['native_after'][0]-frozen['target_nll']),
        correction_visual_effect=a['trial']['final_verdict']['native_nll'][0]-partial['native_after'][0],
        full_native_gain=full['native_gain'],partial_native_gain=partial['native_gain'],
        full_effective=full['effective_reduction'],partial_effective=partial['effective_reduction'],
        predicted_editable_fp32_nll=predicted_fp32,actual_editable_fp32_nll=partial['fp32_after'][0],
        editable_fp32_interpolation_residual=partial['fp32_after'][0]-predicted_fp32,
        partial_native_minus_editable_fp32_nll=partial['native_after'][0]-partial['fp32_after'][0],
        remaining_backbone_native_FP16=True,no_additional_model_calls=True)
    audit=dict(status='PASS',scalar_gate_and_structure_checks=checks,frozen_literal_alpha_verified=True,one_alignment_candidate=True,no_norm_rescaling_of_alignment=True,nll_qualification_recomputed=True,original_native_gates_recomputed=True,old_trust_scalar_gate_recomputed=True,private_weight_quadratic_not_independently_recomputed=True,no_null_baseline_reset=True,missing_comparisons_not_imputed=True,no_additional_model_calls=True)
    for name,value in [('ALIGNMENT_DIAGNOSTICS.json',diagnostics),('SUMMARY.json',summary),('PROBE_SUMMARY.json',probes),('COMPARISON.json',comparison),('PROTOCOL_AUDIT.json',audit)]:
        (root/name).write_text(json.dumps(value,indent=2)+'\n')
    with (root/'ALL_CANDIDATES.tsv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(rows)
    return summary,probes,comparison,audit
