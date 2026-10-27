"""Preserve Base's conditional distribution over non-reference tokens at source positions."""
import torch

def log_distribution(logits,labels):
 target=labels[:,1:].reshape(-1);mask=target!=-100
 assert labels.shape[0]==1 and mask.any()
 z=logits[:,:-1].reshape(-1,logits.shape[-1])[mask].double();t=target[mask]
 assert torch.isfinite(z).all() and z.shape[-1]>1
 logp=z.scatter(1,t[:,None],float('-inf')).log_softmax(-1).scatter(1,t[:,None],0.)
 assert torch.isfinite(logp).all()
 return logp,t

def divergence(logits,labels,teacher):
 current,target=log_distribution(logits,labels);old=teacher['logp'].to(current);t=teacher['target'].to(target.device)
 assert torch.equal(target,t) and current.shape==old.shape and not old.requires_grad
 probability=old.exp().scatter(1,target[:,None],0.)
 kl=(probability*(old-current)).sum(-1).mean()
 assert torch.isfinite(kl) and float(kl.detach())>=-1e-10
 return kl.clamp_min(0.)

def selfcheck():
 base=torch.tensor([[[1.,2.,3.,4.],[2.,4.,1.,3.],[0.,1.,3.,2.]]],dtype=torch.float64)
 labels=torch.tensor([[-100,2,1]]);lp,t=log_distribution(base,labels);teacher=dict(logp=lp.detach(),target=t)
 assert divergence(base,labels,teacher).item()==0
 changed=base.clone();changed[0,0,2]+=20;changed[0,1,1]-=10
 torch.testing.assert_close(divergence(changed,labels,teacher),torch.zeros((),dtype=torch.float64),atol=1e-12,rtol=0)
 altered=base.clone().requires_grad_(True)
 with torch.no_grad():altered[0,0,0]+=1
 kl=divergence(altered,labels,teacher);assert kl>0
 g=torch.autograd.grad(kl,altered)[0];assert g[0,0,2]==0 and g[0,1,1]==0 and torch.isfinite(g).all()
 probabilities=lp.exp().scatter(1,t[:,None],0.);torch.testing.assert_close(probabilities.sum(-1),torch.ones(len(t),dtype=torch.float64))
 return dict(status='PASS',causal_target_mask=True,zero_at_Base=True,target_logit_shift_invariant=True,non_target_change_positive=True,target_gradient_zero=True,conditional_normalization=True,real_model_calls=0)
