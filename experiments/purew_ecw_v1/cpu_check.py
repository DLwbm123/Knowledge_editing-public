"""CPU synthetic algebra checks, including a deliberately loose residual bound."""
import json
import torch
from core import Memory, inner


def main():
    torch.manual_seed(19)
    left=torch.linalg.qr(torch.randn(7,2,dtype=torch.float64)).Q
    right=torch.linalg.qr(torch.randn(5,2,dtype=torch.float64)).Q
    memory=Memory(left,right,(7,5));theta=torch.zeros((7,5),dtype=torch.float64)
    history=[];peak_looseness=0.
    for _ in range(6):
        rows=torch.randn(3,7,5,dtype=torch.float64)*.03
        history.append((rows.clone(),theta.clone(),memory.x.clone(),memory.residual.clone()))
        oldq,oldv=memory.q,memory.v.clone();memory.consolidate(rows)
        assert memory.q==oldq and torch.equal(memory.v,oldv)
        for __ in range(4):
            dx=torch.randn(4,dtype=torch.float64)*.02
            full=torch.randn(7,5,dtype=torch.float64)*.001
            actual=memory.embed(dx)+full-memory.embed(memory.project(full))
            _,_,event=memory.inspect(actual)
            theta+=actual;memory.advance(actual,actual)
            exact=sum(.5*float(((r.flatten(1)@(theta-t).flatten())**2).sum()) for r,t,_,_ in history)
            projected=sum(.5*float(((torch.stack([memory.project(z) for z in r])@(memory.x-x))**2).sum()) for r,_,x,_ in history)
            assert abs(projected-memory.q)<1e-10 and exact<=event['native_risk_bound']+1e-10
            direct=sum(inner(r,r)*inner(memory.residual-e,memory.residual-e) for r,_,_,e in history)
            assert abs(direct-memory.tail(memory.residual))<1e-10
            if exact>0:peak_looseness=max(peak_looseness,event['native_risk_bound']/exact)
    state=memory.state();copy=Memory.restore(state,'cpu');copy.prepare()
    delta,info=copy.solve(torch.randn(7,5),epsilon=max(.01,copy.q+copy.tail(copy.residual)+.1))
    assert torch.isfinite(delta).all()
    memory.prepare();memory.N+=torch.eye(memory.m,dtype=torch.float64)*.7
    resumed=Memory.restore(memory.state(),'cpu');g=torch.randn(7,5)
    first,_=memory.solve(g,epsilon=.2);second,_=resumed.solve(g,epsilon=.2)
    assert torch.equal(first,second),'Frozen within-source metric must survive restore'
    # A residual can lie in the true probe nullspace while the Frobenius envelope is positive.
    eye=torch.eye(3,dtype=torch.float64);counter=Memory(eye[:,:1],eye[:,:1],(3,3))
    row=torch.zeros(1,3,3,dtype=torch.float64);row[0,1,1]=1.;counter.consolidate(row)
    actual=torch.zeros(3,3,dtype=torch.float64);actual[2,2]=.2
    _,_,bad=counter.inspect(actual);exact=0.
    assert bad['native_risk_bound']>.01 and exact==0
    print(json.dumps(dict(status='PASS',multi_birth_identity=True,Welford_residual_moments=True,
        sampled_linear_native_envelope=True,restore_and_solver_finite=True,
        residual_nullspace_counterexample=True,counterexample_exact=exact,
        counterexample_bound=bad['native_risk_bound'],synthetic_max_bound_ratio=peak_looseness,
        can_certify_nonfreezing=False,CUDA_calls=0,model_calls=0)))


if __name__=='__main__':main()
