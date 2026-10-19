"""Offline H outcome and budget accounting; absent stages remain NA."""
import math


def summarize(result,ended,life):
    h=result.get('hybrid',{});failed=[]
    for c in result['checks']:
        if c.get('passed') is False:failed.append(c['field'])
        for key in ('absolute_difference','relative_difference'):
            if key in c and (not math.isfinite(c[key]) or c[key]>c['tolerance']):failed.append(c['field'])
    restoration={k:result.get(k) for k in ('endpoint_restored','base_weight_restored','dtype_restored','base_losses_restored','hooks_restored','flags_restored','generation_config_restored')}
    restoration['H_endpoint_restored']=h.get('endpoint_restored')
    costs=dict(forward=result.get('model_forward_calls'),extra_forward=result.get('extra_forward_calls'),seconds=ended['resident_seconds'],new_secant=result.get('new_secant'),native_match_attempts=result.get('new_cost_matching'),generation=result.get('generation_calls'),judge=result.get('judge_calls'),candidates=result.get('new_edit_candidates'),disk_bytes=life['new_disk_bytes'],checkpoint_count=life['checkpoint_count'])
    limits=dict(forward=600,extra_forward=64,seconds=900,new_secant=1,native_match_attempts=1,generation=1,judge=0,candidates=1,disk_bytes=50*1024**2,checkpoint_count=0)
    return dict(status=result['status'],recorded_checks=len(result['checks']),failed_engineering_checks=sorted(set(failed)),restoration=restoration,budget_checks={k:costs[k] is not None and costs[k]<=v for k,v in limits.items()},costs=costs,
        candidate_formulated=h.get('candidate_formulated',False),candidate_written=h.get('candidate_written',False),candidate_evaluated=h.get('candidate_evaluated',False),analytic=h.get('analytic'),matching=h.get('matching'),main_gate=h.get('main_gate'),main_margin=h.get('main_margin'),main_conditions=h.get('main_conditions'),main_margin_gain=h.get('main_margin_gain'),correction_status=h.get('correction',{}).get('status') if h.get('correction') else None,final_gate=h.get('final_gate'),final_margin=h.get('final_margin'),final_conditions=h.get('final_conditions'),final_margin_gain=h.get('final_margin_gain'),numerically_accepted=h.get('accepted',False),official_generation=h.get('generation'),local_flip=h.get('local_flip'),round_completed_and_accepted=result['status']=='COMPLETE' and h.get('accepted',False),process_exited=life['process_exited'],gpu_process_exited=life['gpu_process_exited'],historical_weight_direct_comparison=None)
