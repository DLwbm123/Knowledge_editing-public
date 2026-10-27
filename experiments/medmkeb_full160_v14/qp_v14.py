"""Finite halfspace projection in one frozen parameter frame; CPU float64 only."""
import torch

def project(direction, matrix, bounds, max_sweeps=5000):
    d=direction.detach().cpu().double();a=matrix.detach().cpu().double();b=bounds.detach().cpu().double()
    assert d.ndim==1 and a.ndim==2 and a.shape[1]==d.numel() and b.shape==(a.shape[0],)
    assert torch.isfinite(d).all() and torch.isfinite(a).all() and torch.isfinite(b).all()
    norms=a.norm(dim=1);nonzero=norms>1e-12
    if bool((b[~nonzero]<0).any()):return d.clone(),dict(status='INFEASIBLE_ZERO_ROW',sweeps=0)
    a=a[nonzero]/norms[nonzero,None];b=b[nonzero]/norms[nonzero]
    if not len(a) or bool((a@d<=b).all()):return d.clone(),dict(status='PASS',sweeps=0,max_normalized_violation=0.,identity=True)
    gram=a@a.T;rhs=a@d-b;dual=torch.zeros(len(a),dtype=torch.float64)
    for sweep in range(1,max_sweeps+1):
        change=0.
        for j in range(len(a)):
            old=float(dual[j]);dual[j]=max(0.,old+float((rhs[j]-gram[j]@dual)/gram[j,j]));change=max(change,abs(float(dual[j])-old))
        delta=d-a.T@dual;violation=float((a@delta-b).max())
        if violation<=1e-8 and change<=1e-10:
            return delta,dict(status='PASS',sweeps=sweep,max_normalized_violation=violation,max_dual_change=change,active_duals=int((dual>1e-10).sum()),stationarity_residual=float((delta-d+a.T@dual).norm()),complementarity_residual=float((dual*(a@delta-b)).abs().max()),identity=False)
    return delta,dict(status='NONCONVERGED',sweeps=max_sweeps,max_normalized_violation=violation,max_dual_change=change)

def selfcheck():
    d=torch.tensor([1.,1.],dtype=torch.float64);x,a=project(d,torch.tensor([[1.,0.]]),torch.tensor([0.]));assert a['status']=='PASS' and torch.allclose(x,torch.tensor([0.,1.],dtype=x.dtype),atol=1e-10) and x.sum()>0
    z,a=project(d,torch.zeros((2,2)),torch.ones(2));assert a['status']=='PASS' and torch.equal(d,z)
    _,a=project(d,torch.tensor([[1.,0.],[-1.,0.]]),torch.tensor([-1.,-1.]),max_sweeps=20);assert a['status']=='NONCONVERGED' and a['sweeps']==20
    return dict(status='PASS',remove_conflict_keep_learning=True,zero_constraint_exact=True,contradiction_bounded_exit=True,new_model_calls=0)
if __name__=='__main__':print(selfcheck())
