"""Damped left-interface Gram preconditioning, preserving each gradient-group norm."""
import torch

KEYS=('G1','G2','G3','G4')
GROUPS=(('G1','G2'),('G3','G4'))

def next_gram(core,left):
 result=torch.einsum('aib,ac,cid->bd',core,left,core)
 return (result+result.T)/2

def direction(state,gradient):
 assert list(state)==list(KEYS) and gradient.numel()==7168
 assert gradient.dtype==torch.float64 and gradient.device.type=='cpu' and torch.isfinite(gradient).all()
 cores={k:state[k].detach().cpu().double() for k in KEYS};gs={};off=0
 for k in KEYS:
  n=cores[k].numel();gs[k]=gradient[off:off+n].reshape_as(cores[k]);off+=n
 left=torch.ones(1,1,dtype=torch.float64);out={};dampings=[]
 for k in KEYS:
  if k=='G1':out[k]=gs[k].clone()
  else:
   damping=1e-4*float(torch.trace(left))/left.shape[0]+1e-12
   out[k]=torch.linalg.solve(left+damping*torch.eye(left.shape[0],dtype=torch.float64),gs[k].flatten(1)).reshape_as(gs[k])
   dampings.append(damping)
  left=next_gram(cores[k],left)
 stats=[]
 for group in GROUPS:
  g=torch.cat([gs[k].flatten() for k in group]);h=torch.cat([out[k].flatten() for k in group]);gn=g.norm();hn=h.norm()
  if gn==0:
   for k in group:out[k].zero_()
   cosine=None
  else:
   assert torch.isfinite(hn) and hn>0
   scale=gn/hn
   for k in group:out[k]*=scale
   cosine=float(torch.dot(g,h)/(gn*hn));assert cosine>0
  stats.append(dict(source_norm=float(gn),direction_cosine=cosine))
 result=torch.cat([out[k].flatten() for k in KEYS]);assert torch.isfinite(result).all()
 return result,dict(dampings=dampings,groups=stats)

def selfcheck():
 from alignment import TT,snapshot
 import module_math as mm
 rng=torch.Generator().manual_seed(720023);prefix=torch.randn(5,2,generator=rng,dtype=torch.float64);core=torch.randn(2,3,4,generator=rng,dtype=torch.float64)
 expanded=torch.einsum('xa,aib->xib',prefix,core).reshape(15,4)
 torch.testing.assert_close(next_gram(core,prefix.T@prefix),expanded.T@expanded,rtol=1e-12,atol=1e-12)
 st=mm.canonicalize(snapshot(TT(225034559)))[0];g=torch.zeros(7168,dtype=torch.float64);g[:512]=torch.linspace(-1,1,512,dtype=torch.float64)
 a,stats=direction(st,g);torch.testing.assert_close(a,g,rtol=1e-12,atol=1e-12)
 st['G1'].fill_(.01);gen=torch.Generator().manual_seed(720023);g=torch.randn(7168,generator=gen,dtype=torch.float64);copy=g.clone();h,stats=direction(st,g)
 for lo,hi in [(0,2560),(2560,7168)]:
  torch.testing.assert_close(h[lo:hi].norm(),g[lo:hi].norm(),rtol=1e-12,atol=1e-12)
  assert torch.dot(h[lo:hi],g[lo:hi])>0
 assert torch.equal(g,copy)
 z,_=direction(st,torch.zeros_like(g));assert z.count_nonzero()==0
 try:direction(st,torch.full_like(g,float('nan')))
 except AssertionError:pass
 else:raise AssertionError('Nonfinite gradient admitted')
 return dict(status='PASS',left_Gram_matches_explicit_contraction=True,zero_root_first_step_unchanged=True,group_norms_preserved=True,positive_source_inner_product=True,rank_deficient_Gram_finite=True,zero_gradient=True,no_alias=True,nonfinite_rejected=True,real_model_calls=0)
