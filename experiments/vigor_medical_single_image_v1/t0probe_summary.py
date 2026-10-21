"""Offline additive token-loss decomposition for the frozen E01 audit."""
import math


def summarize(result):
    a=result['endpoints']['BASE'];b=result['endpoints']['POST'];old=a['teacher_forced']['tokens'];new=b['teacher_forced']['tokens']
    if len(old)!=len(new) or any((x['position'],x['is_eos'])!=(y['position'],y['is_eos']) for x,y in zip(old,new)):raise ValueError('Target alignment changed')
    groups={'first_content':[],'later_content':[],'eos':[]}
    for x,y in zip(old,new):
        key='eos' if x['is_eos'] else ('first_content' if x['position']==1 else 'later_content')
        groups[key].append((x,y))
    total=a['teacher_forced']['mean_nll']-b['teacher_forced']['mean_nll'];n=len(old);parts={}
    for name,rows in groups.items():
        before=sum(x['nll'] for x,y in rows);after=sum(y['nll'] for x,y in rows);gain=before-after
        parts[name]=dict(tokens=len(rows),summed_nll_before=before,summed_nll_after=after,summed_nll_reduction=gain,contribution_to_mean_reduction=gain/n,share_of_mean_reduction=None if not total else gain/n/total)
    error=abs(sum(p['contribution_to_mean_reduction'] for p in parts.values())-total)
    if error>1e-8:raise ValueError('Loss contribution identity failed')
    return dict(mean_nll_before=a['teacher_forced']['mean_nll'],mean_nll_after=b['teacher_forced']['mean_nll'],mean_nll_reduction=total,parts=parts,additive_identity_error=error,
        BASE_first_token=a['prefix_only'],POST_first_token=b['prefix_only'],first_token_probability_ratio=b['prefix_only']['probability']/a['prefix_only']['probability'],
        paths={k:{n:v[n] for n in ['teacher_vs_prefill_max_logit_difference','prefill_vs_generation_max_logit_difference','generation_raw_vs_processed_max_logit_difference','selected_matches_target','selected_matches_historical_first','selected_is_processed_argmax']} for k,v in [('BASE',a),('POST',b)]})
