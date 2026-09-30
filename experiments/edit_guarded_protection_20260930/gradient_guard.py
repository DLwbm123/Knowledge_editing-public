"""One-sided raw-gradient surgery, applied before the unchanged clip/Adam."""
import math
import torch

def aggregate(positive,protection,rho,projection=True,cap=True):
    if not math.isfinite(rho) or rho<=0:raise ValueError('rho must be finite and positive')
    assert len(positive)==len(protection) and positive
    def dot(a,b):return sum((x.double()*y.double()).sum() for x,y in zip(a,b,strict=True))
    eps=1e-12;p2=dot(positive,positive);m2=dot(protection,protection);raw=dot(positive,protection)
    trigger=bool(projection and raw<0)
    factor=raw/(p2+eps) if trigger else raw.new_zeros(())
    projected=[m-factor.to(p.dtype)*p for p,m in zip(positive,protection,strict=True)]
    projected2=dot(projected,projected);pn=p2.sqrt();mn=projected2.sqrt()
    scale=min(1.,float(rho*pn/(mn+eps))) if cap else 1.
    guarded=[scale*m for m in projected];final=[p+m for p,m in zip(positive,guarded,strict=True)]
    postdot=dot(positive,projected);guard2=dot(guarded,guarded);final2=dot(final,final)
    tolerance=max(1e-8,2e-6*float(pn*mn))
    if projection:assert float(postdot)>=-tolerance,'Projection algebra failed'
    if cap:assert float(guard2.sqrt())<=rho*float(pn)+max(1e-8,2e-6*rho*float(pn)),'Trust-ratio algebra failed'
    assert all(torch.isfinite(v).all() for v in final)
    fp=dot(positive,final)
    record=dict(positive_norm=float(pn),protection_norm=float(m2.sqrt()),projected_norm=float(mn),guarded_norm=float(guard2.sqrt()),final_norm=float(final2.sqrt()),dot=float(raw),post_projection_dot=float(postdot),conflict=bool(raw<0),projection_active=trigger,cap_active=bool(cap and scale<1.),cap_scale=scale,rho=rho,norm_ratio_before=float(m2.sqrt()/pn) if pn else None,norm_ratio_after=float(guard2.sqrt()/pn) if pn else None,positive_protection_cosine=float(raw/(pn*m2.sqrt())) if pn and m2 else None,final_positive_cosine=float(fp/(pn*final2.sqrt())) if pn and final2 else None,D_plus=-float(fp),raw_first_order_only=True,numerical_tolerance=tolerance)
    return final,record

def config(method):
    if method in ['SMOKE_S','SMOKE_SP']:return dict(rho=1.,projection=False,cap=False,beta=.1 if method=='SMOKE_SP' else 0.)
    if method in ['SMOKE_GUARD','SMOKE_RESUME']:return dict(rho=1.,projection=True,cap=True,beta=.1)
    arm,value=method.rsplit('_',1);assert arm in ['CAP','EGP','EGP_A'] and value in ['1','2']
    return dict(rho=float(value),projection=arm!='CAP',cap=True,beta=.1 if arm=='EGP_A' else 0.)

if __name__=='__main__':
    p=[torch.tensor([1.,0.],dtype=torch.float64)];m=[torch.tensor([-2.,3.],dtype=torch.float64)]
    for rho in [1.,2.]:
        f,d=aggregate(p,m,rho);assert d['post_projection_dot']>=-1e-8 and d['guarded_norm']<=rho+1e-8 and d['projection_active']
    f,d=aggregate(p,[torch.zeros_like(p[0])],1.);assert torch.equal(f[0],p[0])
    f,d=aggregate(p,[p[0].clone()],1.);assert not d['projection_active']
    for rho in [0.,-1.,float('nan'),float('inf')]:
        try:aggregate(p,m,rho)
        except ValueError:pass
        else:raise AssertionError('Invalid rho accepted')
    print('PASS: rho validation, one-sided projection, cap and zero-protection exactness')
