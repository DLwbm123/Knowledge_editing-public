"""Projection of the actual Adam displacement; at most six constraints."""
import itertools
import torch


def project(d, gradients, slack, metric):
    # Solve on the small Gram matrix in float64; never form a parameter Hessian.
    d, g, s, m = [x.detach().double().cpu() for x in (d, gradients, slack, metric)]
    if not all(torch.isfinite(x).all() for x in (d,g,s,m)) or not (m>0).all():
        raise FloatingPointError('NUMERICAL_FAILURE: nonfinite input or nonpositive metric')
    inv=1/m; norms=(g.square()*inv).sum(1).sqrt(); zero=norms<1e-14
    if (s[zero]<-1e-10).any():raise FloatingPointError('NUMERICAL_FAILURE: infeasible zero constraint')
    g=g[~zero]/norms[~zero,None];s=s[~zero]/norms[~zero]
    gram=(g*inv)@g.T; residual=g@d-s; best=None
    for n in range(len(s)+1):
        for inds in itertools.combinations(range(len(s)),n):
            ids=list(inds);lam=torch.zeros(len(s),dtype=torch.float64)
            if ids:
                a=gram[ids][:,ids];rhs=residual[ids]
                sol=torch.linalg.pinv(a,rtol=1e-12,hermitian=True)@rhs
                if (sol < -1e-8).any() or (a@sol-rhs).abs().max()>1e-8:continue
                lam[ids]=sol
            candidate=d-inv*(g.T@lam)
            if len(s) and (g@candidate-s).max()>1e-8:continue
            objective=float(((candidate-d).square()*m).sum())
            if best is None or objective<best[0]:best=(objective,candidate,lam,ids)
    if best is None:raise FloatingPointError('NUMERICAL_FAILURE: no verified small-QP solution')
    _,out,lam,ids=best
    return out,dict(active=ids,dual=lam.tolist(),normalized_primal_residual=float((g@out-s).max()) if len(s) else 0.,weighted_distance=best[0]**.5,projection_norm_ratio=float(out.norm()/d.norm()) if d.norm()>0 else 0.)


def adam_metric(expert,opt):
    found={}
    for group in opt.param_groups:
        beta=group['betas'][1]
        for p in group['params']:
            state=opt.state[p];v=state['exp_avg_sq']/(1-beta**float(state['step']))
            found[id(p)]=((v.sqrt()+group['eps'])/group['lr']).clamp_min(1e-8).flatten()
    return torch.cat([found[id(p)] for p in expert.parameters()])


def flat(expert):return torch.cat([p.detach().flatten() for p in expert.parameters()]).clone()


def assign(expert,v):
    with torch.no_grad():
        offset=0
        for p in expert.parameters():p.copy_(v[offset:offset+p.numel()].reshape_as(p));offset+=p.numel()
    assert offset==v.numel()


def spectrum(expert):
    with torch.no_grad():
        b,a=expert.factors();_,rb=torch.linalg.qr(b,mode='reduced');_,ra=torch.linalg.qr(a.T,mode='reduced')
        sv=torch.linalg.svdvals(rb@ra.T)
    return dict(singular_values=sv.tolist(),effective_rank_relative_1e_5=int((sv>sv.max()*1e-5).sum()),core_norms_coordinate_dependent={k:float(p.norm()) for k,p in expert.named_parameters()})


def match(expert,before,d,target,activations,fb,cap=4.,grid=32,bisections=16):
    """Bounded search samples nonlinear TT functions, including all grid brackets."""
    if target==0:assign(expert,before);return dict(scale=0.,achieved=0.,target=0.,relative_error=0.,evaluations=0)
    points=[]
    def evaluate(a):
        assign(expert,before+a*d)
        with torch.no_grad():value=float((expert.residual(activations)-fb).norm())
        if not torch.isfinite(torch.tensor(value)):raise FloatingPointError('NUMERICAL_FAILURE: matching function')
        points.append((a,value));return value
    for k in range(grid+1):evaluate(cap*k/grid)
    brackets=[(a,b,va,vb) for (a,va),(b,vb) in zip(points,points[1:]) if (va-target)*(vb-target)<=0]
    if brackets:
        lo,hi,vl,vh=brackets[0]
        for _ in range(bisections):
            mid=(lo+hi)/2;vm=evaluate(mid)
            if (vl-target)*(vm-target)<=0:hi,vh=mid,vm
            else:lo,vl=mid,vm
    alpha,value=min((x for x in points if x[0]>0),key=lambda x:(abs(x[1]-target),x[0]))
    assign(expert,before+alpha*d)
    return dict(scale=alpha,achieved=value,target=target,relative_error=abs(value-target)/max(target,1e-12),evaluations=len(points),brackets=len(brackets))


def selfcheck():
    d=torch.tensor([2.,-3.],dtype=torch.float64);g=torch.tensor([[1.,0.],[0.,-1.]],dtype=torch.float64)
    out,r=project(d,g,torch.tensor([.5,1.]),torch.tensor([4.,2.]))
    assert torch.allclose(out,torch.tensor([.5,-1.],dtype=torch.float64),atol=1e-10)
    assert (g@out<=torch.tensor([.5,1.])+1e-10).all()
    # Correlated and duplicate constraints test the singular Gram path.
    out,_=project(torch.ones(2),torch.tensor([[1.,1.],[2.,2.]]),torch.tensor([0.,0.]),torch.tensor([1.,4.]))
    assert torch.allclose(out,torch.tensor([-.6,.6],dtype=torch.float64),atol=1e-10)
    out,_=project(-torch.ones(2),torch.eye(2),torch.zeros(2),torch.ones(2));assert torch.equal(out,-torch.ones(2,dtype=torch.float64))
    try:project(torch.tensor([float('nan')]),torch.ones(1,1),torch.zeros(1),torch.ones(1))
    except FloatingPointError:pass
    else:raise AssertionError('nonfinite swallowed')
    return dict(status='PASS',analytic_double_projection=True,duplicate_constraints=True,identity_feasible=True,nonfinite_stops=True)


if __name__=='__main__':print(selfcheck())
