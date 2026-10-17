"""One source-only cap on the frozen Fisher coefficient; no searched threshold."""
import math
import torch


def cap_gradient(source,fisher,groups):
    assert set(source)==set(fisher)=={k for group in groups for k in group}
    stats=[];scale=1.
    for group in groups:
        g=torch.cat([source[k].detach().double().flatten() for k in group])
        f=torch.cat([fisher[k].detach().to(g).flatten() for k in group])
        assert torch.isfinite(g).all() and torch.isfinite(f).all()
        gn,fn=float(g.norm()),float(f.norm());dot=float(g@f)
        if fn:scale=min(scale,gn/fn)
        stats.append(dict(source_norm=gn,Fisher_norm=fn,ratio=fn/gn if gn else None,
            cosine=dot/(gn*fn) if gn and fn else None,uncapped_source_dot=gn*gn+dot))
    assert math.isfinite(scale) and 0<=scale<=1
    extra={k:v*scale for k,v in fisher.items()}
    for group,s in zip(groups,stats):
        dot=sum(float((source[k].double()*(source[k]+extra[k]).double()).sum()) for k in group)
        assert dot>=-1e-6*max(s['source_norm']**2,1e-30)
        s['capped_source_dot']=dot
    return extra,dict(scale=scale,active=scale<1,groups=stats)


def selfcheck():
    for g,f,expected in [([1.,0.],[-4.,0.],.25),([1.,0.],[0.,.5],1.),([0.,0.],[1.,0.],0.),([0.,0.],[0.,0.],1.)]:
        source={'a':torch.tensor(g,dtype=torch.float64),'b':torch.tensor([2.],dtype=torch.float64)}
        fish={'a':torch.tensor(f,dtype=torch.float64),'b':torch.tensor([0.],dtype=torch.float64)}
        out,a=cap_gradient(source,fish,(('a',),('b',)))
        assert a['scale']==expected
        torch.testing.assert_close(out['a'],fish['a']*expected)
    try:cap_gradient({'a':torch.tensor([float('nan')])},{'a':torch.zeros(1)},(('a',),))
    except AssertionError:pass
    else:raise AssertionError('nonfinite accepted')
    return dict(status='PASS',conflict=True,nonbinding_identity=True,zero_source=True,zero_Fisher=True,nonfinite_rejected=True)

if __name__=='__main__':print(selfcheck())
