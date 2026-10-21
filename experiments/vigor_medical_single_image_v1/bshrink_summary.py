"""Offline bounded-search accounting, including rejected candidates and NA."""
import math


def summarize(result,ended,lifecycle):
    shrink=result.get('shrink',{});rows=shrink.get('candidates',[]);failed=[]
    for row in result['checks']:
        if row.get('passed') is False:failed.append(row['field'])
        for key in ('absolute_difference','relative_difference'):
            if key in row and (not math.isfinite(row[key]) or row[key]>row['tolerance']):failed.append(row['field'])
    restore={k:result.get(k) for k in ('endpoint_restored','base_weight_restored','dtype_restored','base_losses_restored','hooks_restored','flags_restored','generation_config_restored')}
    costs=dict(forward=result.get('model_forward_calls'),extra_forward=result.get('extra_forward_calls'),seconds=ended['resident_seconds'],new_secant=result.get('new_secant'),new_cost_matching=result.get('new_cost_matching'),generation=result.get('generation_calls'),judge=result.get('judge_calls'),candidates=len(rows),disk_bytes=lifecycle['new_disk_bytes'],checkpoint_count=lifecycle['checkpoint_count'])
    limits=dict(forward=600,extra_forward=64,seconds=900,new_secant=1,new_cost_matching=0,generation=1,judge=0,candidates=2,disk_bytes=50*1024**2,checkpoint_count=0)
    budget={k:costs[k] is not None and costs[k]<=bound for k,bound in limits.items()}
    table=[]
    for row in rows:
        main=row.get('main_conditions',{});final=row.get('final_conditions',{})
        table.append(dict(alpha=row['alpha'],status=row['status'],main_NLL=row.get('main_gate',{}).get('native_nll',[None])[0],main_actual_NLL_gain=main.get('actual_NLL_gain'),NLL_floor=main.get('NLL_floor'),main_NLL_pass=main.get('NLL_pass'),main_margin=row.get('main_margin',{}).get('margin'),main_margin_gain=main.get('margin_gain'),main_margin_pass=main.get('margin_pass'),main_gate=row.get('main_gate'),main_F0_cost=row.get('native_increment',{}).get('F0_cost'),secant_applied=row.get('correction',{}).get('status') if row.get('correction') else None,final_NLL=row.get('final_gate',{}).get('native_nll',[None])[0],final_actual_NLL_gain=final.get('actual_NLL_gain'),final_margin=row.get('final_margin',{}).get('margin'),final_margin_gain=final.get('margin_gain'),final_gate=row.get('final_gate'),accepted=row['accepted'],generation=row.get('generation'),local_flip=row.get('local_flip'),endpoint_restored=row.get('endpoint_restored')))
    order=[row['alpha'] for row in rows]
    selection=shrink.get('selected_scale');preselected=shrink.get('preselected_scale')
    return dict(status=result['status'],recorded_checks=len(result['checks']),failed_engineering_checks=sorted(set(failed)),restoration=restore,costs=costs,budget_checks=budget,preselected_scale=preselected,selected_scale=selection,candidates=table,frozen_order_pass=order in ([],[.5],[.5,.25]),first_eligible_search_closed=preselected is None or (bool(rows) and rows[-1]['alpha']==preselected),all_candidate_restores_pass=all(row.get('endpoint_restored') is True for row in rows),process_exited=lifecycle['process_exited'],gpu_process_exited=lifecycle['gpu_process_exited'],generation_missing_reason='NO_FINAL_ELIGIBLE_CANDIDATE' if not result.get('generation_calls') else None)
