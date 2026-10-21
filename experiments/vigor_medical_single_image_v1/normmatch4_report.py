"""Audit saved four-point ABC results without model calls."""
import csv
import json
import math
from pathlib import Path
from .diagnose import FIRST_D
from .normmatch4 import POINTS


def summarize(root):
    root=Path(root);result=json.loads((root/'RESULT.json').read_text());rows=[];paired=[];checks=0
    assert result['status']=='COMPLETE' and result['denominator']==4 and len(result['points'])==4
    assert json.loads((root/'RECONSTRUCTION_GATE.json').read_text())['all_four_before_evaluation']
    assert json.loads((root/'CONTROL_GATE.json').read_text())['all_four_before_C']
    for state,point in zip(result['points'],POINTS):
        assert (state['point'],state['sample'],state['step'],state['source'])==point
        assert state['status']=='COMPLETE' and state['reconstruction']=='PASS' and state['target_control']=='PASS'
        assert state['reconstruction_base_restored'] and set(state['arms'])=={'A','B','C'}
        start=state['start_native_nll'];base=state['baseline']['native']
        norms={k:v['proposal']['quantization']['native_delta_l2'] for k,v in state['arms'].items()}
        ratio=norms['A']/norms['B'];match=state['matching']
        assert match['ratio']==ratio and match['applications']==1 and match['relative_tolerance']==.01
        error=abs(norms['C']/norms['A']-1)
        assert match['relative_error']==error and match['passed']==(error<=.01)
        assert state['arms']['C']['proposal']['requested_reduction']==state['arms']['B']['proposal']['effective_reduction']*ratio
        for label,trial in state['arms'].items():
            p=trial['proposal'];c=trial['correction'];v=trial['final_verdict'];main=trial['proposal_verdict']
            assert trial['same_start_verified'] and trial['base_restored'] and p['native_before']==start
            assert (c is not None)==(not main['gates']['null'])
            if label!='C':assert p['requested_reduction']==FIRST_D/16
            for verdict in (main,v):
                drift=verdict['native_nll'][1]-base[1]
                expected=dict(null=abs(drift)<=.01,visual=verdict['native_nll'][0]<start[0],
                    trust=math.isfinite(verdict['combined_native_fisher_cost']) and verdict['combined_native_fisher_cost']<=.001)
                if verdict is v and c is not None and c['status']!='APPLIED':expected['valid_correction']=False
                assert verdict['gates']==expected and verdict['accepted']==all(expected.values())
                assert abs(verdict['cumulative_null_change']-drift)<1e-12
                checks+=3
            assert main['native_nll']==p['native_after']
            if c is not None:
                assert c['native_before']==p['native_after']
                if c['status']=='APPLIED':assert v['native_nll']==c['native_after']
            else:assert v['native_nll']==p['native_after']
            rows.append(dict(point=state['point'],sample=state['sample'],source=state['source'],step=state['step'],arm=label,
                requested=p['requested_reduction'],effective=p['effective_reduction'],clipped=p['clipped'],
                native_main_norm=norms[label],norm_match_valid=match['passed'] if label=='C' else 'NA',
                norm_match_error=error if label=='C' else 'NA',native_main_gain=start[0]-p['native_after'][0],
                fp32_main_gain=p['fp32_before'][0]-p['fp32_after'][0],native_main_null_delta=p['native_after'][1]-start[1],
                native_main_cumulative_null=main['cumulative_null_change'],main_null_margin=.01-abs(main['cumulative_null_change']),
                main_old_fisher_cost=main['combined_native_fisher_cost'],main_gate=main['accepted'],
                correction=c is not None,correction_status=c['status'] if c else 'NA',
                final_gain=start[0]-v['native_nll'][0],final_null_margin=.01-abs(v['cumulative_null_change']),
                final_old_fisher_cost=v['combined_native_fisher_cost'],final_gate=v['accepted']))
            checks+=4
        values={r['arm']:r for r in rows if r['point']==state['point']}
        paired.append(dict(point=state['point'],sample=state['sample'],source=state['source'],step=state['step'],
            match_valid=match['passed'],match_error=error,ratio=ratio,
            A_main_gain=values['A']['native_main_gain'],B_main_gain=values['B']['native_main_gain'],C_main_gain=values['C']['native_main_gain'],
            A_null_margin=values['A']['main_null_margin'],B_null_margin=values['B']['main_null_margin'],C_null_margin=values['C']['main_null_margin'],
            C_gain_minus_A=values['C']['native_main_gain']-values['A']['native_main_gain'],
            C_gain_minus_B=values['C']['native_main_gain']-values['B']['native_main_gain']))
    summary=dict(status='COMPLETE',denominator=4,matched=sum(p['match_valid'] for p in paired),
        new_main_candidates=len(rows),new_corrections=sum(r['correction'] for r in rows),
        historical_checks=sum(len(p['checks']) for p in result['points']),
        historical_max_difference=max(v for p in result['points'] for c in p['checks'] for k,v in c.items() if k.endswith('difference')),
        prefix_accepted_steps=sum(len(p['prefix']) for p in result['points']),
        prefix_main_attempts=sum(1+bool(r.get('half')) for p in result['points'] for r in p['prefix']),
        all_base_restored=result['all_base_restored'],elapsed_seconds=result['elapsed_seconds'],
        peak_allocated_bytes=result['peak_allocated_bytes'],model_forward_calls=result['model_forward_calls'],
        original_final_gate_passes={k:sum(r['arm']==k and r['final_gate'] for r in rows) for k in 'ABC'},
        generation_calls=result['generation_calls'],judge_calls=result['judge_calls'],full16_runs=result['full16_runs'])
    assert summary['matched']==result['matched'] and len(rows)==12 and summary['all_base_restored']
    audit=dict(status='PASS',posthoc=True,scalar_gate_and_structure_checks=checks,
        matching_from_actual_native_main_norms=True,one_ratio_only=True,tolerance=.01,all_failed_matches_retained=True,
        same_start_verified=True,original_native_gates_recomputed=True,old_trust_scalar_gate_recomputed=True,
        private_weight_quadratic_not_independently_recomputed=True,combined_update_norm_not_matched=True,
        no_additional_model_calls=True)
    for name,value in [('SUMMARY.json',summary),('PROTOCOL_AUDIT.json',audit),('PAIRED_SUMMARY.json',paired)]:
        (root/name).write_text(json.dumps(value,indent=2)+'\n')
    with (root/'ALL_CANDIDATES.tsv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t',lineterminator='\n')
        writer.writeheader();writer.writerows(rows)
    return summary,paired,audit
