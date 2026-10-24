"""Fixed native-priority conflict projection, in canonical parameter coordinates."""
import torch

def combine(native,view):
 assert native.dtype==view.dtype==torch.float64 and native.device.type==view.device.type=='cpu'
 assert native.shape==view.shape and torch.isfinite(native).all() and torch.isfinite(view).all()
 dot=float(torch.dot(native,view));norm2=float(torch.dot(native,native));other2=float(torch.dot(view,view))
 projected=view
 if dot<0:
  assert norm2>0
  projected=view-(dot/norm2)*native
 result=(native+projected)/2
 assert torch.isfinite(result).all()
 return result,dict(conflict=dot<0,dot=dot,native_norm2=norm2,view_norm2=other2,projected_dot=float(torch.dot(native,projected)))

def selfcheck():
 a=torch.tensor([2.,0.],dtype=torch.float64);b=torch.tensor([-1.,3.],dtype=torch.float64)
 g,m=combine(a,b);assert torch.equal(g,torch.tensor([1.,1.5],dtype=torch.float64)) and m['conflict'] and m['projected_dot']==0
 for v in [a,torch.zeros_like(a),torch.tensor([1.,2.],dtype=torch.float64)]:
  g,m=combine(a,v);assert torch.equal(g,(a+v)/2) and not m['conflict']
 z=torch.zeros_like(a);g,m=combine(z,b);assert torch.equal(g,b/2)
 assert torch.equal(a,torch.tensor([2.,0.],dtype=torch.float64)) and torch.equal(b,torch.tensor([-1.,3.],dtype=torch.float64))
 try:combine(a,torch.full_like(a,float('nan')))
 except AssertionError:pass
 else:raise AssertionError('NaN admitted')
 return dict(status='PASS',conflict=True,nonconflict=True,zero_native=True,no_mutation=True,nonfinite_rejected=True)
