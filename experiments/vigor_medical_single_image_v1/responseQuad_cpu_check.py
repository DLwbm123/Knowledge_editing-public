"""Exact finite-response fit, feasible and infeasible bounded solves."""
from .responseNcontrol_cpu_check import run as previous
from .responseQuad_model import fit,value,solve


def run():
    v=previous();linear=[[-1.,-.6],[1.,1.2],[0.,0.]]
    expected=[dict(linear=g,quadratic=r) for g,r in zip(linear,[[.2,.1,.05],[-.1,-.2,-.03],[0.,0.,0.]])]
    axes=[[value(m,[1,0]),value(m,[0,1])] for m in expected]
    mix=[value(m,[.5,.5]) for m in expected];fitted=fit(linear,axes,mix)
    for a,b in zip(fitted['models'],expected):assert max(abs(x-y) for x,y in zip(a['quadratic'],b['quadratic']))<1e-12
    solved=solve(fitted,0.,[[1.,.2],[.2,1.]],.5,.1,.4453125);assert solved['status']=='ELIGIBLE',solved
    failed=solve(fitted,0.,[[1.,.2],[.2,1.]],.5,10.,.4453125);assert failed['status']!='ELIGIBLE'
    return dict(v,three_site_quadratic_fit=True,single_bounded_solver_feasible=True,infeasible_stops_no_restart=True)
