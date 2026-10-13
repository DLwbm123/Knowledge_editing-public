"""Pre-registered function-preserving CPU float64 SVD gauge control."""
import copy
import torch

@torch.no_grad()
def gauge(raw):
 b,a=raw.factors();b=b.detach().double().cpu();a=a.detach().double().cpu()
 qb,rb=torch.linalg.qr(b,mode='reduced');qa,ra=torch.linalg.qr(a.T,mode='reduced')
 u,s,vh=torch.linalg.svd(rb@ra.T,full_matrices=False)
 ac=vh@qa.T;bc=(qb@u)*s[None,:]
 # Fix each row by its largest-magnitude coordinate; ties use the first index.
 signs=torch.where(ac[torch.arange(len(s)),ac.abs().argmax(dim=1)]<0,-1.,1.)
 ac*=signs[:,None];bc*=signs[None,:]
 out=copy.deepcopy(raw);out.A.copy_(ac);out.B.copy_(bc)
 assert torch.isfinite(out.A).all() and torch.isfinite(out.B).all()
 return out,dict(torch_version=str(torch.__version__),backend='CPU float64 torch.linalg.qr + svd on 4x4 core',singular_values=s.tolist(),singular_gaps=(s[:-1]-s[1:]).tolist(),rank_directions_kept=len(s),sign_convention='Largest absolute A-row coordinate nonnegative; first index on ties',nonuniqueness='Repeated or zero singular values do not define a unique basis; no truncation or score-based choice')

def jobs():
 rows=[]
 for o in range(25,49):
  for kind,m in [('CP','M1'),('TK','M4')]:
   rows.append(dict(id=f'E3-{kind}-{o}',block='E3_REG24',mode='train',kind=kind,methods=[m+'_E3_RAW',m+'_E3_SVDGAUGE'],seed=20260927,order=o,e3_canary=o==25,requires=[] if o==25 else ['E3_CANARY_PASS.json'],status='PENDING'))
 for m in ['M1_E3_RAW','M1_E3_SVDGAUGE','M4_E3_RAW','M4_E3_SVDGAUGE']:
  rows.append(dict(id=f'E3-REG24-{m}-bank',block='E3_REG24',mode='sequential',seed=20260927,method=m,orders=list(range(25,49)),prefixes=[12,24],requires=['E3_CANARY_PASS.json']+[f'adapters/s20260927/{m}/e{o:03d}.pt' for o in range(25,49)],status='PENDING'))
 return rows

if __name__=='__main__':
 from structures import LR4
 torch.manual_seed(20260927)
 for case in ['full','zero','repeated']:
  x=LR4(din=32,dout=16)
  with torch.no_grad():
   x.B.normal_()
   if case=='zero':x.B.zero_()
   if case=='repeated':x.A.copy_(torch.linalg.qr(x.A.T)[0].T);x.B.copy_(torch.linalg.qr(x.B)[0])
  y,meta=gauge(x);h=torch.randn(8,32)
  assert torch.allclose(x.residual(h),y.residual(h),atol=1e-5,rtol=1e-5)
  assert torch.allclose(y.A@y.A.T,torch.eye(4),atol=1e-6)
  y.residual(h).square().sum().backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in y.parameters())
 assert len(jobs())==52 and sum(len(j.get('methods',[])) for j in jobs())==96
 print('PASS: full/zero/repeated spectrum, function and gradients, 52 paired jobs')
