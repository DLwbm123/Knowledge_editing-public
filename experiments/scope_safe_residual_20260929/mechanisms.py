"""CPU-testable protection and scope gate primitives; no training admission side effects."""
import math,random
import torch

def residual_energy(a,b,h,mask,epsilon=1e-6):
 if mask.dtype!=torch.bool or mask.shape!=h.shape[:-1] or not mask.any():raise ValueError('Nonempty aligned predictor mask required')
 z=h[mask].detach().float();z=z/(z.square().mean(-1,keepdim=True).sqrt()+epsilon)
 r=(z@a.T)@b.T
 if not torch.isfinite(r).all():raise ValueError('Nonfinite effective residual')
 return r.square().sum(-1).mean()

def fixed_scale(a,b,h,base_output,mask):
 if base_output.shape[:-1]!=mask.shape:raise ValueError('Base layer output alignment mismatch')
 with torch.no_grad():
  e=residual_energy(a,b,h,mask);v=base_output[mask].detach().float().square().sum(-1).mean()
  if not torch.isfinite(v):raise ValueError('Nonfinite Base output')
  return torch.maximum(torch.maximum(e,1e-6*v),e.new_tensor(1e-12)).detach()

class ResidualCapture:
 """Capture existing KL forward at the unedited layer; never add a backbone forward."""
 def __init__(self,module,a,b,mask,scale):
  if scale.requires_grad or not torch.isfinite(scale) or scale<=0:raise ValueError('Frozen positive c_i required')
  self.a,self.b,self.mask,self.scale=a,b,mask,scale;self.loss=None
  self.handle=module.register_forward_pre_hook(self.capture)
 def capture(self,module,args):self.loss=residual_energy(self.a,self.b,args[0],self.mask)/self.scale
 def close(self):self.handle.remove()

def negative_slot(branch,old_new,background,hard,seed,step):
 if not old_new:raise ValueError('No U_new')
 rng=random.Random(seed*1000003+step)
 if branch in ['A0','AK']:return old_new[rng.randrange(len(old_new))]
 pool=background if branch=='AU' else hard
 if not pool:raise ValueError('No certified background; block the paired experiment')
 # Exact alternating mixture prevents branch-specific mixture-rate drift.
 selected=old_new if step%2 else pool
 return selected[rng.randrange(len(selected))]

def scope_gate(z,candidate,r0_active,positive,negative,tau):
 if candidate is None or not r0_active:return dict(candidate=candidate,active=False,status='R0_OFF')
 if not negative.numel():return dict(candidate=candidate,active=bool(r0_active),status='UNSUPPORTED_NO_NEGATIVE_PROTOTYPES')
 if not positive.numel():raise ValueError('Positive prototypes required')
 if not all(torch.isfinite(x).all() for x in [z,positive,negative]):raise ValueError('Nonfinite feature')
 if torch.cdist(positive.double(),negative.double()).min()<=1e-12:raise ValueError('SCOPE_CONFLICT_OVERLAPPING_PROTOTYPES')
 dp=torch.linalg.vector_norm(positive-z,dim=-1).min();dn=torch.linalg.vector_norm(negative-z,dim=-1).min();q=float((dn-dp)/(dn+dp+1e-12))
 return dict(candidate=candidate,active=q>=tau,status='SUPPORTED',q=q)

def source_folds(rows,n=3):
 import hashlib
 # Hash whole source groups; caller must report empty folds rather than impute.
 return [int(hashlib.sha256(r['source'].encode()).hexdigest(),16)%n for r in rows]

def choose_lambda(check):
 if set(check)!={.01,.1} or any(x is None or not math.isfinite(x) for x in check.values()):raise ValueError('Both isolated CHECK scores required')
 return min(check,key=lambda k:(-check[k],k))

if __name__=='__main__':
 torch.manual_seed(9);a=torch.randn(4,12,requires_grad=True);b=torch.randn(8,4,requires_grad=True);h=torch.randn(2,5,12);mask=torch.zeros(2,5,dtype=torch.bool);mask[:,1:4]=True
 base=torch.nn.Linear(12,8);base.requires_grad_(False);o=base(h);c=fixed_scale(a,b,h,o,mask);e=residual_energy(a,b,h,mask)
 q=torch.eye(4)+.05*torch.randn(4,4)
 assert torch.allclose(e,residual_energy(q@a,b@torch.linalg.inv(q),h,mask),rtol=2e-5,atol=1e-5)
 capture=ResidualCapture(base,a,b,mask,c);base(h);assert torch.allclose(capture.loss,e/c);capture.loss.backward();capture.close()
 assert all(x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum()>0 for x in [a,b]);assert all(p.grad is None for p in base.parameters()) and not c.requires_grad
 assert fixed_scale(a.detach()*0,b.detach()*0,h,o*0,mask)==1e-12
 try:residual_energy(a,b,h,mask*False);raise AssertionError('empty mask accepted')
 except ValueError:pass
 p=torch.tensor([[0.,0.]]);n=torch.tensor([[2.,0.]])
 assert scope_gate(torch.zeros(2),'j',True,p,n,.1)['active'];assert not scope_gate(torch.tensor([2.,0.]),'j',True,p,n,.1)['active']
 assert scope_gate(torch.zeros(2),'j',True,p,torch.empty(0,2),.1)['status'].startswith('UNSUPPORTED')
 try:scope_gate(torch.zeros(2),'j',True,p,p,0);raise AssertionError('collision accepted')
 except ValueError:pass
 assert choose_lambda({.01:1.,.1:1.})==.01
 assert [negative_slot('AU',['old'],['bg'],['hard'],3,s) for s in range(1,5)]==['old','bg','old','bg']
 print('PASS: effective BA gauge, fixed scale, aligned mask, finite gradients, frozen Base, gate conflict/empty/tie, exact mixed sampling')
