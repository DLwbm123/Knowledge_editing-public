"""Replacement protection callback for an isolated EXTRA80 worker, no extra KL forwards."""
from mechanisms import ResidualCapture,negative_slot

def protector(runtime,layer,expert,old_u,new_u,background,hard,branch,seed,kl_fn,scale=None,lambda_res=0.):
 if branch not in ['A0','AK','AU','AH','AHS']:raise ValueError('Unknown branch')
 if not old_u or not new_u:raise ValueError('Original protection rows required')
 if branch in ['AU','AH','AHS'] and (not background or not hard):raise ValueError('Matched expanded pool incomplete')
 if branch=='AHS' and (scale is None or lambda_res not in [.01,.1]):raise ValueError('Frozen scale and predeclared lambda required')
 def apply(hook,step):
  row=negative_slot(branch,new_u,background,hard,seed,step);rows=[(x,.005/len(old_u)) for x in old_u]+[(row,.005)]
  total=0.;res=0.;calls=0
  for (kw,labels,mask,teacher_logp),weight in rows:
   hook.set_teacher_routing(labels) # Inherited forced-on predictor routing, not a new contribution.
   capture=ResidualCapture(layer,expert.A,expert.B,mask,scale) if branch=='AHS' else None
   try:
    logits=runtime.model(**kw).logits;calls+=1
    kl=kl_fn(logits[mask],teacher_logp.detach());loss=(4 if branch=='AK' else 1)*weight*kl
    if capture:
     if capture.loss is None:raise RuntimeError('Protection layer was not executed')
     penalty=lambda_res*capture.loss/len(rows);loss=loss+penalty;res+=float(penalty.detach())
    loss.backward();total+=float(kl.detach())/len(rows)
   finally:
    if capture:capture.close()
  return dict(U_KL=total,residual_penalty=res,KL_forward_calls=calls,extra_residual_forward_calls=0)
 return apply
