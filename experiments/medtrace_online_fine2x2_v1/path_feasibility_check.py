"""A synthetic counterexample: exhausted restore paths do not imply infeasibility."""
from math import hypot


def dot(a,b):return sum(x*y for x,y in zip(a,b))


def main():
    raw=(1.,0.);initial=(.9,0.);feasible=(.9,.09)
    source=((3.,0.),(-1.,10.))
    group=((1.,5.),(1.,0.))
    fine_rhs=[.9*max(dot(g,raw),0) for g in source]
    group_rhs=[.9*max(dot(g,raw),0) for g in group]
    def accepted(z):
        return hypot(*z)<=1+1e-12 and all(dot(g,z)>=b-1e-12 for g,b in zip(group,group_rhs)) and all(dot(g,z)>=b-1e-12 for g,b in zip(source,fine_rhs))
    # H is positive definite; H*initial=90*group[0], proving its group-QP KKT point.
    H=((100.,500.),(500.,3000.))
    assert H[0][0]>0 and H[0][0]*H[1][1]-H[0][1]*H[1][0]>0
    assert all(abs(dot(h,initial)-90*g)<1e-12 for h,g in zip(H,group[0]))
    assert all(abs(dot(g,initial)-b)<1e-12 for g,b in zip(group,group_rhs))
    assert not accepted(initial) and not accepted(raw)
    assert all(not accepted(tuple(v*2**(-k) for v in initial)) for k in range(1,17))
    assert accepted(feasible)
    print('PASS: group-optimal proposal and restore rays fail; another feasible direction exists')


if __name__=='__main__':main()
