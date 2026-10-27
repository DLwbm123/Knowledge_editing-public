"""Fixed output-tail subspace from source CE gradients at a zero writer."""
import torch
from torch import nn

class OutputProbe(nn.Module):
 def __init__(self,a):
  super().__init__()
  assert a.shape==(4,14336) and torch.isfinite(a).all()
  self.register_buffer('a',a.detach().clone())
  self.B=nn.Parameter(a.new_zeros(4096,4))
 def residual(self,activation):
  x=activation.to(self.a.dtype)
  x=x/(x.square().mean(-1,keepdim=True).sqrt()+1e-6)
  return (x@self.a.T)@self.B.T

def output_core(gradient,original):
 assert gradient.shape==(4096,4) and original.shape==(8,64,4)
 g=gradient.detach().cpu().double().reshape(64,256)
 assert torch.isfinite(g).all() and g.norm()>0,'INVALID_SOURCE_GRADIENT'
 _,s,vh=torch.linalg.svd(g,full_matrices=False)
 v=vh[:8].clone()
 # Fix the sign of each basis vector without selecting another subspace.
 indices=v.abs().argmax(dim=1)
 signs=torch.where(v[torch.arange(8),indices]<0,-torch.ones(8,dtype=v.dtype),torch.ones(8,dtype=v.dtype))
 v*=signs[:,None]
 old=original.detach().cpu().double().reshape(8,256)
 old_basis=torch.linalg.qr(old.T,mode='reduced')[0]
 total=g.square().sum();best=(g@v.T).square().sum()/total
 prior=(g@old_basis).square().sum()/total
 assert torch.isfinite(v).all() and float(best)+1e-10>=float(prior)
 return v.reshape(8,64,4).float(),dict(gradient_norm=float(g.norm()),retained_gradient_energy=float(best),original_tail_gradient_energy=float(prior),singular_values=s.tolist())

def selfcheck():
 rng=torch.Generator().manual_seed(690023)
 a=torch.randn(4,14336,generator=rng);p=OutputProbe(a)
 x=torch.randn(1,2,14336,generator=rng);y=p.residual(x)
 assert y.count_nonzero()==0
 y.sum().backward();assert torch.isfinite(p.B.grad).all() and p.B.grad.norm()>0 and p.a.grad is None
 g=torch.randn(4096,4,generator=rng);old=torch.randn(8,64,4,generator=rng)
 v,m=output_core(g,old);vv,mm=output_core(g,old)
 assert torch.equal(v,vv) and m==mm
 torch.testing.assert_close(v.reshape(8,256)@v.reshape(8,256).T,torch.eye(8),atol=1e-6,rtol=1e-6)
 assert m['retained_gradient_energy']>=m['original_tail_gradient_energy']
 try:output_core(torch.zeros_like(g),old)
 except AssertionError:pass
 else:raise AssertionError('Zero gradient admitted')
 return dict(status='PASS',zero_writer=True,gradient_finite_nonzero=True,detached_input=True,svd_deterministic=True,orthonormal=True,zero_gradient_rejected=True,real_model_calls=0)
