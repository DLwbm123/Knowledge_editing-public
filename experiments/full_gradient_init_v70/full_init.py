"""Right-to-left TT-SVD source-gradient tail, with zero deployed root."""
import torch
from torch import nn

class FullProbe(nn.Module):
 def __init__(self,device):
  super().__init__();self.W=nn.Parameter(torch.zeros(4096,14336,device=device,dtype=torch.float32))
 def residual(self,activation):
  x=activation.to(self.W.dtype)
  x=x/(x.square().mean(-1,keepdim=True).sqrt()+1e-6)
  return x@self.W.T

def right_tail(matrix,dims=(64,64,112,128),ranks=(8,4,8)):
 x=matrix.detach().cpu().double().reshape(-1,dims[-1]);total=x.square().sum()
 assert torch.isfinite(x).all() and total>0,'INVALID_SOURCE_GRADIENT'
 cores={};right=1;energy=[]
 for axis in (3,2,1):
  width=dims[axis]*right;x=x.reshape(-1,width);rank=ranks[axis-1]
  # The Gram matrix avoids materializing the very tall first left-SVD factor.
  gram=x.T@x;ev,vec=torch.linalg.eigh(gram);v=vec[:,-rank:].flip(1).T.contiguous()
  idx=v.abs().argmax(1);sgn=torch.where(v[torch.arange(rank),idx]<0,-torch.ones(rank,dtype=v.dtype),torch.ones(rank,dtype=v.dtype));v*=sgn[:,None]
  x=x@v.T;cores['G'+str(axis+1)]=v.reshape(rank,dims[axis],right).float();right=rank
  energy.append(float(x.square().sum()/total))
 assert all(torch.isfinite(v).all() for v in cores.values())
 return cores,dict(full_gradient_norm=float(total.sqrt()),right_to_left_retained_energy=energy,final_tail_projection_energy=energy[-1])

def selfcheck():
 rng=torch.Generator().manual_seed(700023);g=torch.randn(6,20,generator=rng);dims=(2,3,4,5);ranks=(2,2,2)
 cs,m=right_tail(g,dims,ranks);ds,n=right_tail(g,dims,ranks)
 assert m==n and all(torch.equal(cs[k],ds[k]) for k in cs)
 for k,v in cs.items():
  q=v.reshape(v.shape[0],-1);torch.testing.assert_close(q@q.T,torch.eye(q.shape[0]),rtol=1e-6,atol=1e-6)
 tail=torch.einsum('ajb,bkc,cld->ajkld',cs['G2'].double(),cs['G3'].double(),cs['G4'].double()).reshape(2,-1)
 projected=g.double().reshape(2,-1)@tail.T
 assert abs(float(projected.square().sum()/g.double().square().sum())-m['final_tail_projection_energy'])<1e-6
 assert all(a+1e-12>=b for a,b in zip([1.]+m['right_to_left_retained_energy'],m['right_to_left_retained_energy']))
 try:right_tail(torch.zeros_like(g),dims,ranks)
 except AssertionError:pass
 else:raise AssertionError('Zero gradient admitted')
 # Same zero full-writer operation in small dimensions; no production-sized allocation needed.
 w=torch.zeros(3,5,requires_grad=True);x=torch.randn(2,5,generator=rng);y=(x/(x.square().mean(-1,keepdim=True).sqrt()+1e-6))@w.T
 assert y.count_nonzero()==0;y.sum().backward();assert torch.isfinite(w.grad).all() and w.grad.norm()>0
 return dict(status='PASS',zero_writer=True,finite_gradient=True,right_TT_orthogonal=True,projection_energy_reconstructed=True,deterministic=True,zero_rejected=True,real_model_calls=0)
