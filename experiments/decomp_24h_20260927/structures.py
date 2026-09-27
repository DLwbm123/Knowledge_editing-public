"""Rank-four normalized residuals; no CP-only rho interface for Tucker."""
import math
import torch
from torch import nn
class LR4(nn.Module):
 def __init__(self,din=14336,dout=4096,rank=4):
  super().__init__();self.A=nn.Parameter(torch.randn(rank,din)/math.sqrt(din));self.B=nn.Parameter(torch.zeros(dout,rank));self.epsilon=1e-6
 def factors(self):return self.B,self.A
 def residual(self,h):
  h=h.float();h=h/(h.square().mean(-1,keepdim=True).sqrt()+self.epsilon);b,a=self.factors();return (h@a.T)@b.T
 @torch.no_grad()
 def normalize_factors_(self,**kwargs):
  norms=self.A.norm(dim=1).clamp_min(self.epsilon);self.A.div_(norms[:,None]);self.B.mul_(norms[None,:])
 def parameter_groups(self):return [self.A],[self.B]
class TuckerC4(nn.Module):
 def __init__(self):
  super().__init__();self.u1=nn.Parameter(torch.randn(64,4)/8);self.u2=nn.Parameter(torch.randn(64,4)/8);self.u3=nn.Parameter(torch.randn(112,4)/math.sqrt(112));self.u4=nn.Parameter(torch.randn(128,4)/math.sqrt(128));self.L=nn.Parameter(torch.zeros(16,4));self.R=nn.Parameter(torch.randn(16,4)/4);self.epsilon=1e-6
 def factors(self):return torch.kron(self.u1,self.u2)@self.L,(torch.kron(self.u3,self.u4)@self.R).T
 def residual(self,h):
  h=h.float();h=h/(h.square().mean(-1,keepdim=True).sqrt()+self.epsilon);b,a=self.factors();return (h@a.T)@b.T
 def normalize_factors_(self,**kwargs):pass # No parameter rescaling or optimizer-state reparameterization.
 def parameter_groups(self):return [self.u3,self.u4,self.R],[self.u1,self.u2,self.L]
def convert(expert,seed,rank=None):
 from methods.medtrace import AsymmetricCPExpert
 rank=rank or (expert.A.shape[0] if isinstance(expert,LR4) else 4)
 device=next(expert.parameters()).device
 with torch.random.fork_rng(devices=[device.index] if device.type=='cuda' else []):
  torch.manual_seed(seed);out=LR4(rank=rank).to(next(expert.parameters()).device)
 with torch.no_grad():
  if isinstance(expert,AsymmetricCPExpert):b=expert.output_basis()*expert.rho*(expert.beta/math.sqrt(expert.rank));a=expert.input_basis().T
  else:b,a=expert.factors()
  out.B.zero_();out.A[:a.shape[0]].copy_(a);out.B[:,:b.shape[1]].copy_(b)
 return out

def optimizer(expert,base,stage):
 from methods.medtrace import AsymmetricCPExpert
 assert not any(p.requires_grad for p in base.parameters())
 if stage in ['native','A2']:return torch.optim.AdamW(expert.parameters(),lr=1e-3,weight_decay=0)
 if isinstance(expert,AsymmetricCPExpert):a,b=[expert.u_in,expert.v_in],[expert.u_out,expert.v_out,expert.rho]
 else:a,b=expert.parameter_groups()
 return torch.optim.Adam([dict(params=a,lr=1e-4),dict(params=b,lr=1e-3)],betas=(.9,.999),eps=1e-8,weight_decay=0)
