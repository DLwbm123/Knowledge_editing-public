"""Aggregate the completed token constraint screen without changing its decision."""
import json
import os
from pathlib import Path
r=Path(os.environ['RUN_ROOT'])
read=lambda p:json.loads(p.read_text())
data=[read(p) for p in sorted((r/'private/results').glob('*.json'))]
x=read(r/'public/RESULTS.json')
assert len(data)==8 and not list((r/'private').glob('FAILURE*'))
assert sum(d['forward_calls'] for d in data)==4552
assert sum(d['backward_calls'] for d in data)==13104
assert sum(d['zero_expert_Base_checks'] for d in data)==808
assert all(d['state_restored_exact'] and d['baseline_repeat_exact'] and not d['Base_gradient'] for d in data)
assert all(d['basis_audit']['mean_gradient_checks']==61 for d in data)
assert all(s.get('ended_epoch') for s in read(r/'RESOURCE_LEDGER.json')['gpu_sessions'])
geometry=[]
for d in data:
    g=d['geometry'];s=g['singular_values'];m=d['function_matching']
    assert abs(m['actual']-m['target'])<=1e-10+1e-3*m['target']
    geometry.append(dict(expert_order=d['expert_order'],rank=g['rank'],zero_rows=g['zero_rows'],
        smallest_retained_relative_singular_value=s[g['rank']-1]/s[0],
        next_relative_singular_value=s[g['rank']]/s[0],
        retained_parameter_fraction=g['retained_parameter_norm']/g['original_parameter_norm'],
        map_norms=d['map_norms'],matching_absolute_error=abs(m['actual']-m['target']),
        matching_relative_error=m['relative_error']))
held=[q for d in data for q in d['diagnostics'] if q['role']=='HELDOUT_FIT' and q['semantic_correct']]
assert len(held)==504
result=dict(status='PASS',decision_unchanged=x['decision'],interpretation='TOKEN_CONSTRAINTS_EXHAUST_ZERO_STATE_WRITABLE_TANGENT',
    zero_state_G1_coordinates=512,total_TT_parameters=7168,geometry=geometry,
    mean_gradient_parity_checks=488,maximum_loss_definition_gradient_error=x['maximum_mean_gradient_relative_error'],
    maximum_individual_aggregation_error=x['maximum_individual_gradient_aggregation_error'],
    meaningful_nonzero_function_match=False,matching_absolute_floor_dominates=True,
    projected_and_matched_primary_KL_zero=x['protection']['projected_KL']==x['protection']['matched_KL']==0,
    projected_and_matched_edit_progress_zero=x['edit_NLL_progress']['PROJECTED']==x['edit_NLL_progress']['MATCHED_RAW']==0,
    raw_positive_progress_experts=sum(e['edit_NLL_progress']['RAW']>0 for e in x['per_expert']),
    all_six_sessions_ended=True,science_gate_changed=False,mechanical_gate_explicitly_revised=True,
    independent_confirmation=False,clinical_protection=False,full_training_registered=False,
    recovery_GPU_process_hours=x['resource']['new_GPU_process_hours'],
    cumulative_GPU_process_hours=x['resource']['cumulative_GPU_process_hours'],
    cumulative_Judge=x['resource']['cumulative_Judge'])
assert all(g['rank']==512 for g in geometry)
assert result['projected_and_matched_primary_KL_zero'] and result['projected_and_matched_edit_progress_zero']
(r/'public/REVIEW_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
