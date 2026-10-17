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


def remove_opposing(source,capped,groups):
    """Remove only negative source-parallel components after the unchanged norm cap."""
    assert set(source)==set(capped)=={k for group in groups for k in group}
    result={k:v.clone() for k,v in capped.items()};stats=[]
    for group in groups:
        g=torch.cat([source[k].detach().double().flatten() for k in group])
        h=torch.cat([capped[k].detach().to(g).flatten() for k in group])
        assert torch.isfinite(g).all() and torch.isfinite(h).all()
        square=float(g@g);dot=float(g@h)
        coefficient=min(dot,0.)/square if square else 0.
        for k in group:result[k]=(capped[k].double()-coefficient*source[k].double()).to(capped[k])
        after=torch.cat([result[k].double().flatten() for k in group])
        final_dot=float(g@after)
        assert final_dot>=-1e-6*max(square,1e-30)
        assert float(after.norm())<=float(h.norm())*(1+1e-6)+1e-30
        stats.append(dict(removed=dot<0 and square>0,source_norm=square**.5,
            penalty_norm_before=float(h.norm()),penalty_norm_after=float(after.norm()),
            source_dot_ratio_before=1+dot/square if square else None,
            source_dot_ratio_after=1+final_dot/square if square else None))
    return result,dict(groups=stats,removed_groups=sum(v['removed'] for v in stats))


def direction_selfcheck():
    for dtype in (torch.float32,torch.float64):
        for g,h,expected in [([1.,0.],[-2.,3.],[0.,3.]),([1.,0.],[2.,3.],[2.,3.]),
                             ([1.,0.],[0.,3.],[0.,3.]),([0.,0.],[0.,0.],[0.,0.]),
                             ([1e-20,0.],[-2e-20,3e-20],[0.,3e-20])]:
            source={'a':torch.tensor(g,dtype=dtype),'b':torch.tensor([1.],dtype=dtype)}
            fish={'a':torch.tensor(h,dtype=dtype),'b':torch.tensor([.25],dtype=dtype)}
            value,a=remove_opposing(source,fish,(('a',),('b',)))
            torch.testing.assert_close(value['a'],torch.tensor(expected,dtype=dtype),atol=1e-26,rtol=1e-6)
            torch.testing.assert_close(value['b'],fish['b'],rtol=0,atol=0)
    try:remove_opposing({'a':torch.ones(1)},{'a':torch.tensor([float('inf')])},(('a',),))
    except AssertionError:pass
    else:raise AssertionError('nonfinite accepted')
    return dict(status='PASS',opposing_projection=True,aligned_and_orthogonal_identity=True,zero_and_tiny_source=True,groups_independent=True,nonfinite_rejected=True)


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
