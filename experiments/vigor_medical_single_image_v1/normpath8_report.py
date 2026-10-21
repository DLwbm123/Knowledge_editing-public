"""Audit saved native norm path candidates and conditional effects, no model calls."""
import csv
import json
import math
from pathlib import Path
from .diagnose import FIRST_D


def summarize(root):
    root=Path(root);result=json.loads((root/'RESULT.json').read_text());targets=json.loads((root/'FROZEN_TARGETS.json').read_text())
    assert result['status']=='COMPLETE' and result['denominator']==3
    assert json.loads((root/'CONTROL_GATE.json').read_text())['both_before_C']
    assert [s['arm'] for s in result['arms']]==['A','B','C']
    checks=0;rows=[];overview=[];probes=[];matches=[]
    def matching(p,target,label):
        nonlocal checks
        m=p['native_norm_matching'];norm=p['quantization']['native_delta_l2'];error=abs(norm/target-1)
        assert m['target']==target and m['applications']==1 and m['tolerance']==.01
        assert m['ratio']==target/m['unscaled_native_norm'] and m['seed_requested']==FIRST_D/16
        assert p['requested_reduction']==m['unscaled_effective']*m['ratio']
        assert error==m['relative_error'] and m['trust_clipped']==p['clipped']
        assert m['passed']==(error<=.01 and not p['clipped'])
        matches.append(dict(candidate=label,target=target,actual=norm,error=error,passed=m['passed'],ratio=m['ratio'],rounded_direction_cosine=m['native_direction_cosine']))
        checks+=7
        return m['passed']
    def verdict(v,start,base,extra=None):
        nonlocal checks
        drift=v['native_nll'][1]-base[1]
        expected=dict(null=abs(drift)<=.01,visual=v['native_nll'][0]<start[0],trust=math.isfinite(v['combined_native_fisher_cost']) and v['combined_native_fisher_cost']<=.001)
        expected.update(extra or {})
        assert v['gates']==expected and v['accepted']==all(expected.values())
        assert abs(v['cumulative_null_change']-drift)<1e-12;checks+=3
    for s in result['arms']:
        base=s['baseline']['native'];prev=base;arm=s['arm'];assert s['base_restored'] and len(s['steps'])<=8
        for i,t in enumerate(s['steps']):
            assert t['step']==i+1 and t['start_native_nll']==prev
            trial=t['trial'];p=trial['proposal'];c=trial['correction'];main=trial['proposal_verdict'];v=trial['final_verdict']
            assert p['native_before']==prev and main['native_nll']==p['native_after']
            match_ok=matching(p,targets['step_native_norms'][i],f'C/{i+1}') if arm=='C' else True
            if arm!='C':assert p['requested_reduction']==FIRST_D/16
            verdict(main,prev,base)
            extra={}
            if not match_ok:assert c is None and trial['status']=='MATCH_FAILED';extra['valid_norm_match']=False
            else:assert (c is not None)==(not main['gates']['null'])
            if c is not None and c['status']!='APPLIED':extra['valid_correction']=False
            verdict(v,prev,base,extra)
            if c is None:assert v['native_nll']==p['native_after']
            elif c['status']=='APPLIED':assert c['native_before']==p['native_after'] and v['native_nll']==c['native_after']
            assert t['accepted']==v['accepted']
            rows.append(dict(arm=arm,step=i+1,accepted=t['accepted'],native_main_norm=p['quantization']['native_delta_l2'],target_norm=targets['step_native_norms'][i] if arm=='C' else None,matching_pass=match_ok,main_gain=prev[0]-p['native_after'][0],main_null_margin=.01-abs(main['cumulative_null_change']),correction=c is not None,correction_norm=c.get('quantization',{}).get('native_delta_l2') if c else None,final_gain=prev[0]-v['native_nll'][0],final_null_margin=.01-abs(v['cumulative_null_change']),old_trust=v['combined_native_fisher_cost'],requested=p['requested_reduction'],effective=p['effective_reduction'],clipped=p['clipped']))
            if t['accepted']:prev=v['native_nll']
            else:assert i==len(s['steps'])-1 and t['rollback_pass']
            checks+=6
        accepted=[t for t in s['steps'] if t['accepted']]
        assert s['accepted_steps']==len(accepted) and s['prefix_completed']==(len(accepted)==8) and s['endpoint_native_nll']==prev
        o=dict(arm=arm,status=s['status'],accepted_steps=len(accepted),prefix_completed=s['prefix_completed'],endpoint_visual=prev[0],prefix_gain=base[0]-prev[0],endpoint_null_delta=prev[1]-base[1],endpoint_null_margin=.01-abs(prev[1]-base[1]),correction_steps=[t['step'] for t in s['steps'] if t['trial']['correction']],main_path_length=sum(t['trial']['proposal']['quantization']['native_delta_l2'] for t in accepted),correction_path_length=sum(t['trial']['correction']['quantization']['native_delta_l2'] for t in accepted if t['trial']['correction'] and t['trial']['correction']['status']=='APPLIED'),accepted_effective=s['accepted_effective_reduction'],minimum_accepted_null_margin=min((.01-abs(t['trial']['final_verdict']['cumulative_null_change']) for t in accepted),default=None))
        if not s['prefix_completed']:assert not s['probes'] and s['probe_missing_reason']=='INCOMPLETE_EIGHT_STEP_PREFIX'
        for key,probe in s['probes'].items():
            p=probe['proposal'];v=probe['verdict'];assert p['native_before']==prev and p['native_after']==v['native_nll'] and probe['endpoint_restored']
            verdict(v,prev,base)
            ok=matching(p,targets['probe_native_norm'],arm+'/probe') if key=='matched' else True
            assert probe['matching_passed']==ok
            readout=dict(arm=arm,kind=key,matching_pass=ok,native_norm=p['quantization']['native_delta_l2'],gain=prev[0]-v['native_nll'][0],null_delta=v['cumulative_null_change'],null_margin=.01-abs(v['cumulative_null_change']),old_trust=v['combined_native_fisher_cost'],gate=v['accepted'],gates=v['gates'])
            probes.append(readout)
            if key==s['matched_probe_key']:
                o.update(probe_norm_match_passed=ok,probe_gain=readout['gain'],probe_null_margin=readout['null_margin'],probe_null_delta=readout['null_delta'],probe_gate=readout['gate'])
            checks+=3
        first=s['steps'][0]['trial']['proposal']
        o.update(first_main_gain=first['native_gain'],first_main_native_norm=first['quantization']['native_delta_l2'],first_gain_per_native_norm=first['native_gain']/first['quantization']['native_delta_l2'])
        overview.append(o)
    eligible=all(o['prefix_completed'] and o.get('probe_norm_match_passed') for o in overview) and all(m['passed'] for m in matches)
    effects=dict(eligible=eligible,scope='conditional direction/amplitude policies including rounding and old-secant feedback; no interaction identified')
    if eligible:
        x={o['arm']:o for o in overview}
        for key in ('probe_gain','probe_null_margin','probe_null_delta','prefix_gain','endpoint_null_margin'):
            effects[key]=dict(direction_C_minus_A=x['C'][key]-x['A'][key],amplitude_B_minus_C=x['B'][key]-x['C'][key],total_B_minus_A=x['B'][key]-x['A'][key])
    else:effects['reason']='Incomplete prefix or failed native norm matching; no endpoint imputation or causal split'
    differences=[v for s in result['arms'] for c in s['checks'] for k,v in c.items() if k.endswith('difference')]
    summary=dict(status='COMPLETE',denominator=3,completed_prefixes=sum(o['prefix_completed'] for o in overview),main_candidates=len(rows),corrections=sum(r['correction'] for r in rows),probe_candidates=len(probes),matching_attempts=len(matches),matching_passes=sum(m['passed'] for m in matches),historical_checks=sum(len(s['checks']) for s in result['arms']),historical_max_difference=max(differences),elapsed_seconds=result['elapsed_seconds'],peak_allocated_bytes=result['peak_allocated_bytes'],model_forward_calls=result['model_forward_calls'],all_base_restored=result['all_base_restored'],half_candidates=result['half_candidates'],full16_runs=result['full16_runs'],generation_calls=result['generation_calls'],judge_calls=result['judge_calls'])
    assert summary['completed_prefixes']==result['completed_prefixes'] and summary['probe_candidates']==result['probe_count'] and summary['all_base_restored']
    assert result['half_candidates']==result['full16_runs']==result['generation_calls']==result['judge_calls']==0
    audit=dict(status='PASS',scalar_gate_and_structure_checks=checks,one_ratio_no_retry=True,native_norm_errors_recomputed=True,original_native_gates_recomputed=True,old_trust_scalar_gate_recomputed=True,private_weight_quadratic_not_independently_recomputed=True,controls_before_C=True,no_additional_model_calls=True)
    for name,v in [('SUMMARY.json',summary),('ARM_SUMMARY.json',overview),('PROBE_SUMMARY.json',probes),('MATCHING.json',matches),('CONDITIONAL_EFFECTS.json',effects),('PROTOCOL_AUDIT.json',audit)]:
        (root/name).write_text(json.dumps(v,indent=2)+'\n')
    with (root/'ALL_STEPS.tsv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(rows)
    return summary,overview,probes,matches,effects,audit
