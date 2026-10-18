"""Audit same-candidate ABC readouts and finite precision decompositions."""
import csv
import json
import math
from pathlib import Path
from .nllprecision import SELECTED_ALPHA


def summarize(root):
    root=Path(root);read=lambda n:json.loads((root/n).read_text())
    result=read('RESULT.json');frozen=read('FROZEN_TARGETS.json');prior=read('PRIOR_MATCHED.json');gate=read('CONTROL_GATE.json')
    assert result['status']=='COMPLETE' and result['denominator']==2 and result['matched_endpoints']
    assert gate['no_search'] and gate['both_before_precision'] and gate['selected_step_replayed_once']
    assert frozen['alpha']==SELECTED_ALPHA
    states={s['arm']:s for s in result['arms']};old={s['arm']:s for s in prior['arms']}
    rows=[];decomp=[];checks=0
    def verdict(v,start,base,extra=None):
        nonlocal checks
        drift=v['native_nll'][1]-base[1]
        expected=dict(null=abs(drift)<=.01,visual=v['native_nll'][0]<start[0],trust=math.isfinite(v['combined_native_fisher_cost']) and v['combined_native_fisher_cost']<=.001)
        expected.update(extra or {});assert v['gates']==expected and v['accepted']==all(expected.values())
        assert abs(v['cumulative_null_change']-drift)<1e-12;checks+=3
    def path_trial(t,start,base):
        nonlocal checks
        p=t['proposal'];c=t['correction'];m=t['proposal_verdict'];v=t['final_verdict']
        assert p['native_before']==start and p['native_after']==m['native_nll']
        assert (c is not None)==(not m['gates']['null'])
        verdict(m,start,base);verdict(v,start,base,dict(valid_correction=False) if c and c['status']!='APPLIED' else None)
        assert v['accepted']
        if c:assert c['native_before']==p['native_after'] and v['native_nll']==c['native_after']
        else:assert v['native_nll']==p['native_after']
        checks+=4
    for label,count in [('A',8),('D',4)]:
        s=states[label];base=s['baseline']['native'];prev=base
        assert s['base_restored'] and len(s['steps'])==count
        for i,t in enumerate(s['steps']):
            assert t['step']==i+1 and t['start_native_nll']==prev and t['accepted']
            path_trial(t['trial'],prev,base);prev=t['trial']['final_verdict']['native_nll']
        assert s['endpoint_native_nll']==prev
        if label=='D':
            t=s['replay'];path_trial(t,prev,base)
            expected=old['D']['search']['candidates'][-1]
            assert t['proposal']['requested_reduction']==expected['requested']
            assert s['aligned_endpoint_native_nll']==t['final_verdict']['native_nll']
            assert s['replay_base_restored'];prev=s['aligned_endpoint_native_nll']
        assert prev==old[label].get('aligned_endpoint_native_nll',old[label]['endpoint_native_nll'])
        assert abs(prev[0]-frozen['target_nll'])<=.005
        assert len(s['probes'])==1 and s['probe_base_restored']
        probe=s['probes']['matched'];p=probe['proposal'];v=probe['verdict'];m=p['native_norm_matching']
        assert probe['endpoint_restored'] and probe['matching_passed'] and m['passed']
        norm=p['quantization']['native_delta_l2'];error=abs(norm/frozen['probe_native_norm']-1)
        assert m['applications']==1 and m['relative_error']==error and error<=.01 and not p['clipped']
        assert m['ratio']==m['target']/m['unscaled_native_norm'] and m['target']==frozen['probe_native_norm']
        assert p['native_before']==prev and p['native_after']==v['native_nll'];verdict(v,prev,base)
        b=probe['branches'];assert set(b)=={'A','B','C'}
        assert b['A']==dict(before=p['fp32_before'],after=p['fp32_after'])
        assert b['B']==dict(before=p['fp32_before'],after=p['rounded_fp32_after'])
        assert b['C']==dict(before=p['native_before'],after=p['native_after'])
        values={}
        for precision in 'ABC':
            before=b[precision]['before'];after=b[precision]['after']
            assert all(math.isfinite(z) for z in before+after)
            row=dict(endpoint=label,precision=precision,visual_gain=before[0]-after[0],incremental_null=after[1]-before[1],
                visual_before=before[0],visual_after=after[0],null_before=before[1],null_after=after[1],
                original_native_gate=v['accepted'] if precision=='C' else 'NOT_APPLICABLE')
            rows.append(row);values[precision]=row;checks+=3
        effects=dict(endpoint=label)
        for metric in ['visual_gain','incremental_null']:
            vals={k:z[metric] for k,z in values.items()}
            effects[metric]=dict(unrounded_FP32=vals['A'],rounded_FP32=vals['B'],native=vals['C'],
                rounding_B_minus_A=vals['B']-vals['A'],editable_compute_C_minus_B=vals['C']-vals['B'],total_C_minus_A=vals['C']-vals['A'])
            assert abs((vals['B']-vals['A'])+(vals['C']-vals['B'])-(vals['C']-vals['A']))<1e-12;checks+=1
        decomp.append(effects);checks+=12
    contrasts={}
    for metric in ['visual_gain','incremental_null']:
        a=decomp[0][metric];d=decomp[1][metric]
        contrasts[metric]={k:d[k]-a[k] for k in a}
    diffs=[v for s in states.values() for c in s['checks'] for k,v in c.items() if k.endswith('difference')]
    summary=dict(status='COMPLETE',endpoint_denominator=2,precision_branches=3,fixed_probe_candidates=2,
        reference_main=13,search_candidates=0,extra_readonly_B_forwards=4,reference_checks=sum(len(s['checks']) for s in states.values()),reference_max_difference=max(diffs),
        corrections=sum(t['trial']['correction'] is not None for s in states.values() for t in s['steps'])+bool(states['D']['replay']['correction']),
        elapsed_seconds=result['elapsed_seconds'],model_forward_calls=result['model_forward_calls'],peak_allocated_bytes=result['peak_allocated_bytes'],
        all_base_restored=result['all_base_restored'],half_candidates=result['half_candidates'],full16_runs=result['full16_runs'],generation_calls=result['generation_calls'],judge_calls=result['judge_calls'])
    assert result['probe_count']==2 and result['all_base_restored'] and result['half_candidates']==result['full16_runs']==result['generation_calls']==result['judge_calls']==0
    audit=dict(status='PASS',scalar_gate_structure_decomposition_checks=checks,selected_endpoint_replay_no_search=True,same_parameter_candidate_in_three_branches=True,
        original_gates_only_native=True,finite_difference_decomposition_recomputed=True,old_trust_scalar_gate_recomputed=True,private_weight_quadratic_not_independently_recomputed=True,no_additional_model_calls=True)
    for name,value in [('SUMMARY.json',summary),('PRECISION_READOUTS.json',rows),('DECOMPOSITION.json',decomp),('ENDPOINT_CONTRASTS.json',contrasts),('PROTOCOL_AUDIT.json',audit)]:
        (root/name).write_text(json.dumps(value,indent=2)+'\n')
    with (root/'PRECISION_READOUTS.tsv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(rows)
    return summary,rows,decomp,contrasts,audit
