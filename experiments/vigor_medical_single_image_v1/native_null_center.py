"""One deterministic margin-per-null greedy deletion on existing native values."""
import numpy as np
import torch

def center(base,proposal,gm,g0,base_bias):
    q=proposal.float()-base.float()
    null_terms=(g0.double()*q.double()).flatten()
    margin_terms=(gm.double()*q.double()).flatten()
    predicted_bias=base_bias+float(null_terms.sum())
    side=1 if predicted_bias>0 else -1
    required=abs(predicted_bias)
    info=dict(target_original_base_null=0.,original_base_bias=base_bias,initial_predicted_bias=predicted_bias,required_null_recovery=required,excursion_sign=side,sort='stable_ascending_margin_loss_per_same_side_null_recovery_then_flat_index',subset_search=False)
    if not np.isfinite(required) or required<=0:
        info['status']='STOPPED_SIGNED_CENTER_NO_NONZERO_EXCURSION';return None,info
    indices=torch.nonzero(side*null_terms>0,as_tuple=False).flatten()
    recovery=(side*null_terms[indices]).detach().cpu().numpy()
    loss=margin_terms[indices].detach().cpu().numpy()
    if len(recovery)==0 or not np.isfinite(recovery).all() or not np.isfinite(loss).all() or (loss<0).any():
        info['status']='STOPPED_SIGNED_CENTER_INVALID_CONTRIBUTIONS';return None,info
    order=np.argsort(loss/recovery,kind='stable')
    cumulative=np.cumsum(recovery[order],dtype=np.float64)
    info['available_same_side_null_recovery']=float(cumulative[-1])
    if cumulative[-1]<required:
        info['status']='STOPPED_SIGNED_CENTER_INSUFFICIENT_RECOVERY';return None,info
    count=int(np.searchsorted(cumulative,required,side='left'))+1
    chosen=torch.as_tensor(order[:count].copy(),device=indices.device,dtype=torch.long)
    dropped=indices[chosen]
    selected=proposal.clone();selected.view(-1)[dropped]=base.view(-1)[dropped]
    info.update(status='FORMULATED',dropped_elements=count,removed_margin_linear=float(margin_terms[dropped].sum()),removed_same_side_null_linear=float((side*null_terms[dropped]).sum()),last_negative_null_quantum=float((side*null_terms[dropped[-1]])),max_removed_null_quantum=float((side*null_terms[dropped]).max()),greedy_fractional_bound_not_integer_optimum=True)
    return selected,info
