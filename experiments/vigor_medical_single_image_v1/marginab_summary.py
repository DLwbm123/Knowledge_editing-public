"""Offline comparison, preserving incomplete arms and cost-match failures."""


def summarize(result):
    origin=result.get('endpoint_margin',{}).get('margin');arms={}
    for name in ['A','B']:
        arm=result.get('arms',{}).get(name,{})
        main=arm.get('main_margin',{});final=arm.get('final_margin',{});g=arm.get('final_gate',{});gen=arm.get('generation',{});proposal=arm.get('proposal',{})
        cost=proposal.get('native_update',{}).get('fisher_cost') if name=='A' else proposal.get('actual_native_F0_cost')
        arms[name]=dict(available=bool(arm),main_cost=cost,main_margin=main.get('margin'),main_margin_gain=None if main.get('margin') is None or origin is None else main['margin']-origin,
            final_margin=final.get('margin'),final_margin_gain=None if final.get('margin') is None or origin is None else final['margin']-origin,
            final_target_probability=final.get('probability'),final_target_rank=final.get('rank'),final_NLL=final.get('mean_nll'),
            final_combined_cost=g.get('combined_native_fisher_cost'),final_null_change=g.get('cumulative_null_change'),original_gates=g.get('gates'),original_gate_accepted=arm.get('original_gate_accepted'),margin_augmented_accepted=arm.get('margin_augmented_accepted'),
            correction_applied=arm.get('correction') is not None,generation_selected_target=gen.get('selected_matches_target'),local_flip=arm.get('local_flip'),safe_local_flip=arm.get('safe_local_flip'),endpoint_restored=arm.get('endpoint_restored'))
    matching=result.get('arms',{}).get('B',{}).get('proposal',{}).get('matching')
    def difference(field):
        a,b=arms['A'].get(field),arms['B'].get(field)
        return None if a is None or b is None else b-a
    return dict(status=result['status'],origin_margin=origin,arms=arms,B_matching=matching,B_minus_A_main_margin_gain=difference('main_margin_gain'),B_minus_A_final_margin_gain=difference('final_margin_gain'),final_costs_are_not_matched=True)
