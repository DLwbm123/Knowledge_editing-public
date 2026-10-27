"""Write only at the last prompt predictor and subsequent answer positions."""
import torch
from token_hook import TokenHook as Parent
class TokenHook(Parent):
 def __init__(self,module,expert,arm='ALL_TOKENS'):
  super().__init__(module,expert,'ALL_TOKENS')
 def position_mask(self,n,device,offset=0):
  assert self.current is not None
  return (torch.arange(n,device=device)+offset>=self.current['prefix_length']-1).unsqueeze(0)
def selfcheck():
 from torch import nn
 class Expert(nn.Module):
  def __init__(self):super().__init__();self.w=nn.Parameter(torch.ones(3))
  def residual(self,x):return x*self.w
 ex=Expert();module=nn.Identity();h=TokenHook(module,ex);h.set_layout(dict(prefix_length=4,visual_start=1,visual_count=2));x=torch.ones(1,7,3);labels=torch.tensor([[-100]*4+[1,2,3]])
 h.attach()
 try:
  h.teacher(labels);out=module(x)
  assert torch.equal(out[:,:3],x[:,:3]) and torch.equal(out[:,3:],2*x[:,3:])
  g=torch.autograd.grad(out.sum(),ex.w)[0];assert torch.equal(g,torch.full((3,),4.))
  with h.generation_request():
   a=module(x[:,:4]);b=module(x[:,:1]);c=module(x[:,:1])
  assert torch.equal(a[:,:3],x[:,:3]) and torch.equal(a[:,3:],2*x[:,3:4])
  assert torch.equal(b,2*x[:,:1]) and torch.equal(c,b)
  assert [r['active_predictor_count'] for r in h.last_generation_trace]==[1,1,1]
 finally:h.detach()
 assert torch.equal(module(x),x)
 return dict(status='PASS',teacher_boundary_exact=True,prefill_single_last=True,decode_all=True,gradient_nonzero=True,detach_restores=True,real_model_calls=0)
