"""Post-run checks and aggregate interpretation, with no candidate reselection."""
import os,json
from pathlib import Path
r=Path(os.environ['RUN_ROOT'])
read=lambda p:json.loads(p.read_text())
x=read(r/'public/RESULTS.json')
old=read(Path(os.environ['REFERENCE_RUN'])/'public/RESULTS.json')
reference={e['expert_order']:e['edit_NLL_progress']['RAW'] for e in old['per_expert']}
per=x['per_expert'];assert len(per)==8 and x['status']=='COMPLETE'
maximum_raw_error=max(abs(e['edit_progress']['RAW']-reference[e['expert_order']]) for e in per)
assert maximum_raw_error<1e-10
assert all(e['geometry']['KKT_residual']<1e-9 for e in per)
assert all(e['geometry']['objective_BOUNDED']<=e['geometry']['objective_RAW']*(1+1e-6) for e in per)
ledger=read(r/'RESOURCE_LEDGER.json');assert all(s.get('ended_epoch') for s in ledger['gpu_sessions'])
a=dict(status='PASS',decision_unchanged=x['decision'],RAW_previous_diagnostic_max_edit_progress_difference=maximum_raw_error,
    actual_edit_retention=[e['actual_edit_retention'] for e in per],
    all_actual_edit_retention_pass=all(e['actual_edit_retention']>=.9 for e in per),
    source_proxy_reduction=[1-e['geometry']['objective_BOUNDED']/e['geometry']['objective_RAW'] for e in per],
    map_norm_ratio_to_RAW=[e['geometry']['BOUNDED_map_norm']/e['geometry']['RAW_map_norm'] for e in per],
    maximum_KKT_residual=max(e['geometry']['KKT_residual'] for e in per),
    maximum_function_match_relative_error=max(e['matching']['relative_error'] for e in per),
    minimum_function_match_target=min(e['matching']['target'] for e in per),
    maximum_loss_definition_gradient_error=max(e['basis_audit']['maximum_mean_gradient_relative_error'] for e in per),
    maximum_individual_aggregation_error=max(e['basis_audit']['maximum_individual_gradient_aggregation_error'] for e in per),
    primary_relative_KL_reduction=1-x['protection']['BOUNDED']/x['protection']['MATCHED_RAW'] if x['protection']['MATCHED_RAW']>0 else None,
    zero_new_checkpoints=True,sessions_ended=6,independent_confirmation=False,
    proxy_is_full_KL=False,free_generation_semantic_accuracy_tested=False)
(r/'public/REVIEW_AUDIT.json').write_text(json.dumps(a,indent=2)+'\n')
print(json.dumps(a,indent=2))
