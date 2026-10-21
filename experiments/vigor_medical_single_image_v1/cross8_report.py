"""Audit fixed S03 factorial trajectories using saved scalars, without model calls."""
import csv
import json
import math
from pathlib import Path
from .cross8 import ARMS
from .diagnose import FIRST_D


def summarize(root):
    root=Path(root);result=json.loads((root/'RESULT.json').read_text());rows=[];overview=[];checks=0
    assert result['status']=='COMPLETE' and result['denominator']==4
    gate=json.loads((root/'CONTROL_GATE.json').read_text())
    assert gate['status']=='PASS' and gate['both_before_mixed']
    assert [s['arm'] for s in result['arms']]==list(ARMS)
    for s in result['arms']:
        base=s['baseline']['native'];prev=base
        assert s['base_restored'] and len(s['steps'])<=8
        for i,step in enumerate(s['steps']):
            assert step['step']==i+1 and step['start_native_nll']==prev
            t=step['trial'];p=t['proposal'];c=t['correction'];main=t['proposal_verdict'];final=t['final_verdict']
            assert p['requested_reduction']==FIRST_D/16 and p['native_before']==prev
            assert (c is not None)==(not main['gates']['null'])
            assert main['native_nll']==p['native_after']
            if c is None:assert final['native_nll']==p['native_after']
            elif c['status']=='APPLIED':assert c['native_before']==p['native_after'] and final['native_nll']==c['native_after']
            for v in (main,final):
                drift=v['native_nll'][1]-base[1]
                expected=dict(null=abs(drift)<=.01,visual=v['native_nll'][0]<prev[0],trust=math.isfinite(v['combined_native_fisher_cost']) and v['combined_native_fisher_cost']<=.001)
                if v is final and c is not None and c['status']!='APPLIED':expected['valid_correction']=False
                assert v['gates']==expected and v['accepted']==all(expected.values())
                assert abs(v['cumulative_null_change']-drift)<1e-12;checks+=3
            assert step['accepted']==final['accepted']
            if step['accepted']:prev=final['native_nll']
            else:assert i==len(s['steps'])-1 and step['rollback_pass']
            rows.append(dict(arm=s['arm'],step=i+1,accepted=step['accepted'],main_gain=step['start_native_nll'][0]-p['native_after'][0],main_null_margin=.01-abs(main['cumulative_null_change']),main_norm=p['quantization']['native_delta_l2'],correction=c is not None,correction_status=c['status'] if c else 'NA',correction_norm=c.get('quantization',{}).get('native_delta_l2') if c else None,final_gain=step['start_native_nll'][0]-final['native_nll'][0],final_null_margin=.01-abs(final['cumulative_null_change']),old_trust=final['combined_native_fisher_cost'],effective=p['effective_reduction'],clipped=p['clipped']))
            checks+=6
        assert s['endpoint_native_nll']==prev
        assert s['accepted_steps']==sum(t['accepted'] for t in s['steps'])
        assert s['prefix_completed']==(s['accepted_steps']==8)
        probe=s['probe'];o=dict(arm=s['arm'],status=s['status'],accepted_steps=s['accepted_steps'],prefix_completed=s['prefix_completed'],endpoint_visual=prev[0],endpoint_null_delta=prev[1]-base[1],endpoint_null_margin=.01-abs(prev[1]-base[1]),corrections=sum(t['trial']['correction'] is not None for t in s['steps']),main_path_length=sum(t['trial']['proposal']['quantization']['native_delta_l2'] for t in s['steps'] if t['accepted']),correction_path_length=sum(t['trial']['correction']['quantization']['native_delta_l2'] for t in s['steps'] if t['accepted'] and t['trial']['correction'] and t['trial']['correction']['status']=='APPLIED'),accepted_requested=s['accepted_requested_reduction'],accepted_effective=s['accepted_effective_reduction'],all_main_unclipped=s['all_main_unclipped'])
        if s['prefix_completed']:
            p=probe['proposal'];v=probe['verdict'];drift=v['native_nll'][1]-base[1]
            assert probe['endpoint_restored'] and p['native_before']==prev and p['requested_reduction']==FIRST_D/16
            expected=dict(null=abs(drift)<=.01,visual=v['native_nll'][0]<prev[0],trust=math.isfinite(v['combined_native_fisher_cost']) and v['combined_native_fisher_cost']<=.001)
            assert v['gates']==expected and v['accepted']==all(expected.values()) and p['native_after']==v['native_nll']
            o.update(probe_gain=prev[0]-v['native_nll'][0],probe_null_delta=drift,probe_null_margin=.01-abs(drift),probe_old_trust=v['combined_native_fisher_cost'],probe_gate=v['accepted'],probe_gates=v['gates'],probe_norm=p['quantization']['native_delta_l2']);checks+=6
        else:assert probe==dict(status='NOT_RUN',reason='INCOMPLETE_EIGHT_STEP_PREFIX')
        o.update(prefix_gain=base[0]-prev[0],minimum_accepted_null_margin=min(.01-abs(t['trial']['final_verdict']['cumulative_null_change']) for t in s['steps'] if t['accepted']),correction_steps=[t['step'] for t in s['steps'] if t['trial']['correction'] is not None])
        overview.append(o)
    eligible=all(o['prefix_completed'] and o['all_main_unclipped'] for o in overview) and len({o['accepted_effective'] for o in overview})==1
    effects=dict(eligible=eligible,outcome='fixed F1 d/16 main-only native gain')
    if eligible:
        g={o['arm']:o['probe_gain'] for o in overview}
        effects.update(main_given_old_correction=g['10']-g['00'],main_given_new_correction=g['11']-g['01'],correction_given_old_main=g['01']-g['00'],correction_given_new_main=g['11']-g['10'],interaction=g['11']-g['10']-g['01']+g['00'],both_minus_old=g['11']-g['00'])
        effects['other_outcomes']={}
        for key in ('endpoint_visual','endpoint_null_margin','probe_null_delta','probe_null_margin'):
            x={o['arm']:o[key] for o in overview}
            effects['other_outcomes'][key]=dict(main_given_old_correction=x['10']-x['00'],main_given_new_correction=x['11']-x['01'],correction_given_old_main=x['01']-x['00'],correction_given_new_main=x['11']-x['10'],interaction=x['11']-x['10']-x['01']+x['00'])
    else:effects['reason']='Incomplete eight-step prefix or unmatched/clipped effective reduction; no endpoint imputation'
    differences=[v for s in result['arms'] for c in s['checks'] for k,v in c.items() if k.endswith('difference')]
    summary=dict(status='COMPLETE',denominator=4,completed_prefixes=sum(o['prefix_completed'] for o in overview),probe_count=sum('probe_gain' in o for o in overview),main_candidates=len(rows),corrections=sum(r['correction'] for r in rows),historical_checks=sum(len(s['checks']) for s in result['arms']),historical_max_difference=max(differences),elapsed_seconds=result['elapsed_seconds'],model_forward_calls=result['model_forward_calls'],peak_allocated_bytes=result['peak_allocated_bytes'],all_base_restored=result['all_base_restored'],generation_calls=0,judge_calls=0,half_candidates=result['half_candidates'],full16_runs=result['full16_runs'])
    assert summary['completed_prefixes']==result['completed_prefixes'] and summary['probe_count']==result['probe_count'] and summary['all_base_restored']
    assert result['generation_calls']==result['judge_calls']==result['half_candidates']==result['full16_runs']==0
    audit=dict(status='PASS',scalar_gate_and_structure_checks=checks,original_native_scalar_gates_recomputed=True,old_trust_scalar_gate_recomputed=True,private_weight_quadratic_not_independently_recomputed=True,control_before_mixed=True,no_additional_model_calls=True)
    for name,value in [('SUMMARY.json',summary),('PROTOCOL_AUDIT.json',audit),('FACTORIAL.json',effects),('ARM_SUMMARY.json',overview)]:
        (root/name).write_text(json.dumps(value,indent=2)+'\n')
    with (root/'ALL_STEPS.tsv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(rows)
    return summary,overview,effects,audit
