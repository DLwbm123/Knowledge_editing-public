"""Subgradient of the maximum of two complete-source-answer cross entropies."""
import math
import torch

def choose(first,second):
 a,g=first;b,h=second
 assert math.isfinite(a) and math.isfinite(b) and torch.isfinite(g).all() and torch.isfinite(h).all() and g.shape==h.shape
 if a>b:return g.clone(),'native'
 if b>a:return h.clone(),'view'
 return (g+h)/2,'tie_mean'

def selfcheck():
 g=torch.tensor([1.,-2.],dtype=torch.float64);h=torch.tensor([-4.,3.],dtype=torch.float64)
 a,branch=choose((2.,g),(1.,h));assert branch=='native' and torch.equal(a,g) and a.data_ptr()!=g.data_ptr()
 a,branch=choose((1.,g),(2.,h));assert branch=='view' and torch.equal(a,h)
 a,branch=choose((1.,g),(1.,h));assert branch=='tie_mean' and torch.equal(a,(g+h)/2)
 a,_=choose((0.,torch.zeros_like(g)),(0.,torch.zeros_like(h)));assert a.count_nonzero()==0
 try:choose((float('nan'),g),(1.,h))
 except AssertionError:pass
 else:raise AssertionError('NaN accepted')
 return dict(status='PASS',larger_complete_answer_loss_selected=True,tie_symmetric_subgradient=True,no_alias=True,zero_gradient=True,nonfinite_rejected=True,real_model_calls=0)
