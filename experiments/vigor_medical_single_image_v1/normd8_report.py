"""Audit D and combine frozen ABC; never impute a missing eight-step endpoint."""
import csv
import json
import math
from pathlib import Path
from .diagnose import FIRST_D


def summarize(root):
    root=Path(root);read=lambda n:json.loads((root/n).read_text())
    result=read('RESULT.json');targets=read('FROZEN_TARGETS.json');prior=read('PRIORS_ABC.json');ref=read('REFERENCE.json')
    assert result['status']=='COMPLETE' and result['denominator']==1 and len(result['arms'])==1
    gate=read('CONTROL_GATE.json');assert gate['before_D'] and not gate['full_ABC_replayed']
    assert ref['status']=='PASS' and ref['base_restored']
    abc={s['arm']:s for s in prior['arms']};d=result['arms'][0]
    assert set(abc)=={'A','B','C'} and d['arm']=='D' and d['base_restored']
    assert targets['step_native_norms']==[t['trial']['proposal']['quantization']['native_delta_l2'] for t in abc['B']['steps']]
    assert targets['probe_native_norm']==abc['A']['probes']['raw']['proposal']['quantization']['native_delta_l2']
    assert d['baseline']['native']==abc['A']['baseline']['native']
    checks=0;matches=[];rows=[]
    def matching(p,target,label):
        nonlocal checks
        m=p['native_norm_matching'];actual=p['quantization']['native_delta_l2'];error=abs(actual/target-1)
        assert m['target']==target and m['applications']==1 and m['tolerance']==.01
        assert m['ratio']==target/m['unscaled_native_norm'] and m['seed_requested']==FIRST_D/16
        assert p['requested_reduction']==m['unscaled_effective']*m['ratio']
        assert m['relative_error']==error and m['trust_clipped']==p['clipped']
        assert m['passed']==(error<=.01 and not p['clipped'])
        matches.append(dict(candidate=label,target=target,actual=actual,error=error,passed=m['passed'],ratio=m['ratio'],native_direction_cosine=m['native_direction_cosine']));checks+=7
        return m['passed']
    def verdict(v,start,base,extra=None):
        nonlocal checks
        drift=v['native_nll'][1]-base[1]
        expected=dict(null=abs(drift)<=.01,visual=v['native_nll'][0]<start[0],trust=math.isfinite(v['combined_native_fisher_cost']) and v['combined_native_fisher_cost']<=.001)
        expected.update(extra or {})
        assert v['gates']==expected and v['accepted']==all(expected.values())
        assert abs(v['cumulative_null_change']-drift)<1e-12;checks+=3
    base=d['baseline']['native'];prev=base
    for i,t in enumerate(d['steps']):
        assert t['step']==i+1 and t['start_native_nll']==prev
        trial=t['trial'];p=trial['proposal'];c=trial['correction'];main=trial['proposal_verdict'];v=trial['final_verdict']
        assert p['native_before']==prev and main['native_nll']==p['native_after']
        ok=matching(p,targets['step_native_norms'][i],f'D/{i+1}')
        verdict(main,prev,base);extra={}
        if not ok:assert c is None and trial['status']=='MATCH_FAILED';extra['valid_norm_match']=False
        else:assert (c is not None)==(not main['gates']['null'])
        if c is not None and c['status']!='APPLIED':extra['valid_correction']=False
        verdict(v,prev,base,extra)
        if c is None:assert v['native_nll']==p['native_after']
        elif c['status']=='APPLIED':assert c['native_before']==p['native_after'] and v['native_nll']==c['native_after']
        assert t['accepted']==v['accepted']
        rows.append(dict(step=i+1,accepted=t['accepted'],target_native_norm=targets['step_native_norms'][i],actual_native_norm=p['quantization']['native_delta_l2'],match_error=p['native_norm_matching']['relative_error'],match_pass=ok,main_gain=prev[0]-p['native_after'][0],main_null_margin=.01-abs(main['cumulative_null_change']),main_gate=main['accepted'],correction=c is not None,correction_status=c['status'] if c else 'NA',final_gain=prev[0]-v['native_nll'][0],final_null_margin=.01-abs(v['cumulative_null_change']),final_old_trust=v['combined_native_fisher_cost'],final_gate=v['accepted'],requested=p['requested_reduction'],effective=p['effective_reduction']))
        if t['accepted']:prev=v['native_nll']
        else:assert i==len(d['steps'])-1 and t['rollback_pass']
        checks+=6
    assert d['endpoint_native_nll']==prev and d['accepted_steps']==sum(t['accepted'] for t in d['steps'])
    assert d['prefix_completed']==(d['accepted_steps']==8) and len(d['steps'])<=8
    probe=d['probes'].get('matched')
    if not d['prefix_completed']:assert not d['probes'] and d['probe_missing_reason']=='INCOMPLETE_EIGHT_STEP_PREFIX'
    if probe:
        p=probe['proposal'];v=probe['verdict'];assert p['native_before']==prev and p['native_after']==v['native_nll']
        ok=matching(p,targets['probe_native_norm'],'D/probe');verdict(v,prev,base)
        assert probe['matching_passed']==ok and probe['endpoint_restored']
        assert d['probe_accepted']==(ok and v['accepted'])
        checks+=3
    def arm_summary(s):
        accepted=[t for t in s['steps'] if t['accepted']];loss=s['endpoint_native_nll'];b=s['baseline']['native'];p=s['steps'][0]['trial']['proposal']
        o=dict(arm=s['arm'],status=s['status'],accepted_steps=s['accepted_steps'],prefix_completed=s['prefix_completed'],endpoint_visual=loss[0],prefix_gain=b[0]-loss[0],endpoint_null_margin=.01-abs(loss[1]-b[1]),main_path_length=sum(t['trial']['proposal']['quantization']['native_delta_l2'] for t in accepted),correction_path_length=sum(t['trial']['correction']['quantization']['native_delta_l2'] for t in accepted if t['trial']['correction'] and t['trial']['correction']['status']=='APPLIED'),correction_steps=[t['step'] for t in s['steps'] if t['trial']['correction']],first_main_gain=p['native_gain'],first_main_norm=p['quantization']['native_delta_l2'],first_main_null_delta=p['native_null_change'],first_main_gate=s['steps'][0]['trial']['proposal_verdict']['accepted'])
        if s['prefix_completed']:
            probe=s['probes'][s['matched_probe_key']];p=probe['proposal'];v=probe['verdict']
            o.update(probe_match_pass=probe['matching_passed'],probe_gain=loss[0]-v['native_nll'][0],probe_null_margin=.01-abs(v['cumulative_null_change']),probe_gate=v['accepted'],probe_norm=p['quantization']['native_delta_l2'])
        return o
    arms=[arm_summary(abc[k]) for k in 'ABC']+[arm_summary(d)]
    first_eligible=bool(matches and matches[0]['passed'])
    eligible=d['prefix_completed'] and d['probe_norm_match_passed'] and all(m['passed'] for m in matches)
    effects=dict(eight_step_probe_eligible=eligible,first_step_main_eligible=first_eligible)
    def contrasts(key):
        v={s['arm']:s[key] for s in arms}
        return dict(direction_small_C_minus_A=v['C']-v['A'],direction_large_B_minus_D=v['B']-v['D'],amplitude_old_D_minus_A=v['D']-v['A'],amplitude_new_B_minus_C=v['B']-v['C'],interaction_B_minus_C_minus_D_plus_A=v['B']-v['C']-v['D']+v['A'])
    if first_eligible:
        for key in ('first_main_gain','first_main_null_delta'):effects[key]=contrasts(key)
    if eligible:
        for key in ('probe_gain','probe_null_margin','prefix_gain','endpoint_null_margin'):effects[key]=contrasts(key)
    else:effects['eight_step_missing_reason']='D incomplete eight-step prefix or failed norm match; no endpoint imputation'
    differences=[v for c in ref['checks'] for k,v in c.items() if k.endswith('difference')]
    summary=dict(status='COMPLETE',new_arm_denominator=1,factorial_arm_denominator=4,D_status=d['status'],D_accepted_steps=d['accepted_steps'],D_prefix_completed=d['prefix_completed'],D_main_candidates=len(rows),D_corrections=sum(r['correction'] for r in rows),D_probe_candidates=len(d['probes']),reference_main_candidates=1,matching_attempts=len(matches),matching_passes=sum(m['passed'] for m in matches),reference_checks=len(ref['checks']),reference_max_difference=max(differences),elapsed_seconds=result['elapsed_seconds'],model_forward_calls=result['model_forward_calls'],peak_allocated_bytes=result['peak_allocated_bytes'],all_base_restored=result['all_base_restored'],half_candidates=result['half_candidates'],full16_runs=result['full16_runs'],generation_calls=result['generation_calls'],judge_calls=result['judge_calls'])
    assert result['completed_prefixes']==int(d['prefix_completed']) and result['probe_count']==len(d['probes'])
    assert result['all_base_restored'] and result['half_candidates']==result['full16_runs']==result['generation_calls']==result['judge_calls']==0
    audit=dict(status='PASS',scalar_gate_and_structure_checks=checks,native_norm_errors_recomputed=True,one_ratio_no_retry=True,original_native_gates_recomputed=True,old_trust_scalar_gate_recomputed=True,private_weight_quadratic_not_independently_recomputed=True,reference_before_D=True,ABC_frozen_prior_results=True,no_additional_model_calls=True)
    for name,value in [('SUMMARY.json',summary),('ARMS_ABCD.json',arms),('MATCHING.json',matches),('FACTORIAL.json',effects),('PROTOCOL_AUDIT.json',audit)]:
        (root/name).write_text(json.dumps(value,indent=2)+'\n')
    with (root/'ALL_D_STEPS.tsv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(rows)
    return summary,arms,matches,effects,audit
