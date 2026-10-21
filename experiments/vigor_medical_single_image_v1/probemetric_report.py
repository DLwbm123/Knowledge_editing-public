"""Audit fixed existing-metric matched-progress native probes."""
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
    rows=[];checks=0
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
        assert set(s['probes'])=={'F1','F0'} and s['probe_base_restored']
        for metric in ('F1','F0'):
            probe=s['probes'][metric];p=probe['proposal'];v=probe['verdict'];m=p['native_norm_matching']
            assert probe['endpoint_restored'] and probe['matching_passed'] and m['passed']
            norm=p['quantization']['native_delta_l2'];error=abs(norm/frozen['probe_native_norm']-1)
            assert m['applications']==1 and m['relative_error']==error and error<=.01 and not p['clipped']
            assert m['ratio']==m['target']/m['unscaled_native_norm'] and m['target']==frozen['probe_native_norm']
            assert p['native_before']==prev and p['native_after']==v['native_nll'];verdict(v,prev,base)
            assert 'rounded_fp32_after' not in p
            rows.append(dict(endpoint=label,metric=metric,visual_gain=prev[0]-p['native_after'][0],
                incremental_null=p['native_after'][1]-prev[1],cumulative_null_before=prev[1]-base[1],
                cumulative_null_after=v['cumulative_null_change'],null_margin=.01-abs(v['cumulative_null_change']),
                accepted=v['accepted'],gates=v['gates'],native_norm=norm,norm_relative_error=error))
            checks+=10
    assert read('PROBE_CONTROL_GATE.json')['both_F1_before_F0']
    lookup={(r['endpoint'],r['metric']):r for r in rows};effects={}
    for metric in ('F1','F0'):
        a=lookup['A',metric];d=lookup['D',metric]
        effects[metric]={k:d[k]-a[k] for k in ('visual_gain','incremental_null','cumulative_null_after','null_margin')}
    interaction={k:effects['F0'][k]-effects['F1'][k] for k in effects['F0']}
    diffs=[v for s in states.values() for c in s['checks'] for k,v in c.items() if k.endswith('difference')]
    summary=dict(status='COMPLETE',endpoint_denominator=2,fixed_probe_candidates=4,reference_main=13,search_candidates=0,
        reference_checks=sum(len(s['checks']) for s in states.values()),reference_max_difference=max(diffs),
        corrections=sum(t['trial']['correction'] is not None for s in states.values() for t in s['steps'])+bool(states['D']['replay']['correction']),
        elapsed_seconds=result['elapsed_seconds'],model_forward_calls=result['model_forward_calls'],peak_allocated_bytes=result['peak_allocated_bytes'],
        all_base_restored=result['all_base_restored'])
    assert result['probe_count']==4 and result['all_base_restored'] and result['half_candidates']==result['full16_runs']==result['generation_calls']==result['judge_calls']==0
    audit=dict(status='PASS',scalar_gate_structure_checks=checks,selected_endpoint_replay_no_search=True,F1_controls_before_F0=True,
        once_native_norm_matching=True,probe_corrections=0,original_native_gates_recomputed=True,
        old_trust_scalar_gate_recomputed=True,private_weight_quadratic_not_independently_recomputed=True,no_additional_model_calls=True)
    for name,value in [('SUMMARY.json',summary),('PROBE_READOUTS.json',rows),('ENDPOINT_CONTRASTS.json',effects),('INTERACTION.json',interaction),('PROTOCOL_AUDIT.json',audit)]:
        (root/name).write_text(json.dumps(value,indent=2)+'\n')
    with (root/'PROBE_READOUTS.tsv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(rows)
    return summary,rows,effects,interaction,audit
