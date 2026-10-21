"""Deterministic CPU checks; no data access or model loading."""
import json
import torch
from .rank_reachability import project, retract, dense

torch.manual_seed(42)
u=torch.linalg.qr(torch.randn(17,4,dtype=torch.float64))[0];v=torch.linalg.qr(torch.randn(23,4,dtype=torch.float64))[0];g=torch.randn(17,23,dtype=torch.float64)
y=(u*torch.arange(1,5,dtype=torch.float64))@v.T
for fixed in (True,False):
 p=project(g,u,v,fixed)
 assert torch.allclose(project(p,u,v,fixed),p,atol=1e-10)
 assert abs(float(((g-p)*p).sum()))<1e-10
 trial=y+.1*p
 s=retract(trial,u,v,4,fixed);got=dense(s)
 a,b,c=torch.linalg.svd(trial,full_matrices=False);expected=(a[:,:4]*b[:4])@c[:4]
 assert torch.allclose(got,expected,atol=1e-10)
 assert torch.linalg.matrix_rank(got)==4
 if fixed: assert torch.allclose(got,u@(u.T@got@v)@v.T,atol=1e-10)
# Verify the Fisher-coordinate chain rule and cost with nonuniform metric.
a=torch.rand(23,dtype=torch.float64)+.1;b=torch.rand(17,dtype=torch.float64)+.1
x=y.detach().requires_grad_(True);scale=.04472135955
loss=(scale*x/b.sqrt()[:,None]/a.sqrt()[None,:]*g).sum()
actual=torch.autograd.grad(loss,x)[0]
assert torch.allclose(actual,scale*g/b.sqrt()[:,None]/a.sqrt()[None,:])
delta=scale*y/b.sqrt()[:,None]/a.sqrt()[None,:]
assert abs(float((delta.square()*b[:,None]*a[None,:]).sum()/2)-scale**2*float(y.square().sum())/2)<1e-10
print(json.dumps(dict(status='PASS',checks=['fixed and tangent projection idempotence and orthogonality','compact retraction versus full SVD both arms','rank and fixed span','whitened chain rule','Fisher cost identity'],model_forwards=0)))
