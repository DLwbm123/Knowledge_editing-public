"""Base next-token KL on a fixed image-swapped source prompt, without answer inputs."""
import torch
from contextlib import contextmanager

def prefix_kwargs(batch,p):
 kw={k:v for k,v in batch.forward_kwargs().items() if k!='labels'};kw['use_cache']=False
 for k in ('inputs_embeds','attention_mask','position_ids'):
  if isinstance(kw.get(k),torch.Tensor):kw[k]=kw[k][:,:p]
 assert kw['inputs_embeds'].shape[1]==p
 return kw

def divergence(logits,teacher_logp):
 assert teacher_logp.grad_fn is None and not teacher_logp.requires_grad
 logp=logits.double().log_softmax(-1)
 loss=torch.nn.functional.kl_div(logp,teacher_logp,log_target=True,reduction='sum')
 assert torch.isfinite(loss)
 return loss

@contextmanager
def anchor_loss(runtime,batch,lay,hook):
 assert hook.handle is None
 kwargs=prefix_kwargs(batch,lay['prefix_length'])
 with torch.no_grad():
  out=runtime.model(**kwargs);teacher_logits=out.logits[0,-1].detach().clone();del out
 teacher_logp=teacher_logits.double().log_softmax(-1).detach()
 original=runtime.compute_loss
 def loss(b):
  if b is not batch:return original(b)
  assert hook.current==lay
  with hook.generation_request():out=runtime.model(**kwargs)
  result=divergence(out.logits[0,-1],teacher_logp);del out
  return result
 runtime.compute_loss=loss
 try:yield teacher_logits
 finally:runtime.compute_loss=original

def selfcheck():
 x=torch.tensor([.2,-.3,1.],requires_grad=True);target=x.detach().double().log_softmax(-1)
 z=divergence(x,target);g=torch.autograd.grad(z,x)[0];assert abs(float(z))<1e-6 and g.abs().max()<1e-6
 y=(x.detach()+torch.tensor([.4,0.,0.])).requires_grad_();z=divergence(y,target);g=torch.autograd.grad(z,y)[0]
 assert float(z)>0 and torch.isfinite(g).all() and g.norm()>0 and not target.requires_grad
 return dict(status='PASS',zero_equal=True,positive_nonzero=True,finite_gradient=True,teacher_detached=True)
