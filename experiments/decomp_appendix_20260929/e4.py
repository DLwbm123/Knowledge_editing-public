"""Pre-registered paired rank capacity expansion; no change to original directions."""
import copy
import torch

@torch.no_grad()
def expand(raw,seed):
 from structures import LR4
 b,a=raw.factors();device=a.device
 with torch.random.fork_rng(devices=[device.index] if device.type=='cuda' else []):
  torch.manual_seed(seed);out=LR4(din=a.shape[1],dout=b.shape[0],rank=8).to(device)
 out.A[:4].copy_(a);out.B.zero_();out.B[:,:4].copy_(b)
 assert (out.A[4:].norm(dim=1)>0).all() and not out.B[:,4:].count_nonzero()
 return out

def jobs():
 rows=[]
 for o in range(25,49):
  for kind,m in [('CP','M1'),('TK','M4')]:
   rows.append(dict(id=f'E4-{kind}-{o}',block='E4_REG24',mode='train',kind=kind,methods=[m+'_E4_R4',m+'_E4_R8'],seed=20260927,order=o,e4_canary=o==25,requires=[] if o==25 else ['E4_CANARY_PASS.json'],status='PENDING'))
 for m in ['M1_E4_R4','M1_E4_R8','M4_E4_R4','M4_E4_R8']:
  rows.append(dict(id=f'E4-REG24-{m}-bank',block='E4_REG24',mode='sequential',seed=20260927,method=m,orders=list(range(25,49)),prefixes=[12,24],requires=['E4_CANARY_PASS.json']+[f'adapters/s20260927/{m}/e{o:03d}.pt' for o in range(25,49)],status='PENDING'))
 return rows

if __name__=='__main__':
 from structures import LR4
 torch.manual_seed(20260927);x=LR4(din=32,dout=16)
 with torch.no_grad():x.B.normal_()
 y=expand(x,20260927);h=torch.randn(8,32)
 assert torch.equal(x.A,y.A[:4]) and torch.equal(x.B,y.B[:,:4])
 assert torch.allclose(x.residual(h),y.residual(h),atol=1e-6,rtol=1e-6)
 y.residual(h).square().sum().backward()
 assert y.B.grad[:,4:].abs().sum()>0 and torch.isfinite(y.B.grad).all()
 assert len(jobs())==52 and sum(len(j.get('methods',[])) for j in jobs())==96
 print('PASS: rank8 equal initial function, preserved rank4, nonzero extra-direction B gradient, 52 jobs')
