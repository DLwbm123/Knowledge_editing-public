"""Offline audit of the fixed-B diagnostic; no model calls or missing-value filling."""
import math


def summarize(result,ended,lifecycle):
    errors=[];checks=result.get('checks',[])
    for row in checks:
        if row.get('passed') is False:errors.append(row['field'])
        for key in ('absolute_difference','relative_difference'):
            if key in row and (not math.isfinite(row[key]) or row[key]>row['tolerance']):errors.append(row['field'])
    restore_keys=['endpoint_restored','base_weight_restored','base_losses_restored','hooks_restored','flags_restored','generation_config_restored']
    restoration={k:result.get(k) for k in restore_keys}
    d=result.get('diagnostic',{});parts=d.get('decomposition')
    costs=dict(forward=result.get('model_forward_calls'),diagnostic_forward=result.get('diagnostic_forward_calls'),seconds=ended.get('resident_seconds'),peak_allocated_bytes=ended.get('peak_allocated_bytes'),new_disk_bytes=lifecycle['new_disk_bytes'],checkpoint_count=lifecycle['checkpoint_count'],generation=result.get('generation_calls'),judge=result.get('judge_calls'),new_cost_matching=result.get('new_cost_matching'),new_secant=result.get('new_secant'),new_edit_candidates=result.get('new_edit_candidates'))
    bounds=dict(forward=600,diagnostic_forward=20,seconds=900,new_disk_bytes=50*1024**2,checkpoint_count=0,generation=0,judge=0,new_cost_matching=0,new_secant=0,new_edit_candidates=0)
    budget={k:costs[k] is not None and costs[k]<=v for k,v in bounds.items()}
    interpretation='INCOMPLETE_NO_MECHANISM_ATTRIBUTION'
    if result['status']=='COMPLETE' and parts:
        eps=2e-5  # two independently checked loss readings in each increment
        if parts['P']>=0:interpretation='PREDICTION_ALIGNMENT_FAILED'
        elif parts['delta_F_unrounded']>eps:interpretation='UNROUNDED_FIXED_CHAIN_ALREADY_REVERSES_DESCENT'
        elif parts['delta_F_unrounded']<-eps and parts['delta_F_rounded']>eps:interpretation='PARAMETER_ROUNDING_TRIGGERS_REVERSAL_AT_THIS_INCREMENT'
        else:interpretation='MIXED_OR_UNRESOLVED_SIGN'
    return dict(status=result['status'],check_count=len(checks),failed_checks=sorted(set(errors)),restoration=restoration,costs=costs,budget_checks=budget,process_exited=lifecycle['process_exited'],gpu_process_exited=lifecycle['gpu_process_exited'],decomposition=parts,interpretation=interpretation,historical_weight_direct_comparison=None,limitations=['F is editable-layer FP32 with the original remaining mixed chain','R is not pure curvature; Q includes finite-response interactions','No new accepted edit, generation result, T0 performance or safe-B claim'])
