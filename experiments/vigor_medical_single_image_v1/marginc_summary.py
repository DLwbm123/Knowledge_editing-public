"""Offline C outcome, retaining NA when the frozen pre-candidate stop fires."""


def summarize(result):
    arm=result.get('arms',{}).get('C',{});p=arm.get('proposal',{});a=p.get('analytic',{});match=p.get('matching')
    return dict(status=result['status'],constraint_active=a.get('constraint_active'),unconstrained_linear_NLL_change=a.get('unconstrained_linear_NLL_change'),NLL_floor=a.get('NLL_descent_floor'),
        candidate_formulated=p.get('candidate_formulated',False),candidate_written=p.get('candidate_written',False),matching=match,
        main_actual_NLL_gain=arm.get('actual_main_NLL_gain'),final_actual_NLL_gain=arm.get('actual_final_NLL_gain'),main_margin=arm.get('main_margin'),final_margin=arm.get('final_margin'),
        final_gate=arm.get('final_gate'),generation=arm.get('generation'),local_flip=arm.get('local_flip'),safe_local_flip=arm.get('safe_local_flip'),
        endpoint_restored=arm.get('endpoint_restored'),gradient_checks=p.get('historical_gradient_checks'),new_secant_calls=int(arm.get('correction') is not None))
