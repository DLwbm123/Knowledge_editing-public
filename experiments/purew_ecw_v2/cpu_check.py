"""Independent dense oracle for cumulative functional statistics and native quantization."""
import json,torch
from core import Memory,dot

def main():
    torch.manual_seed(38);w=torch.randn(7,5).half();r=Memory(w);theta=w.double().clone();history=[]
    for source in range(3):
        rows=torch.randn(4,7,5,dtype=torch.float64)*.1;history.append((rows.clone(),theta.clone()));q,v=r.q,r.v.clone()
        r.consolidate(rows);assert r.q==q and torch.equal(r.v,v);r.prepare()
        for slot in range(6):
            native=(theta.float()+torch.randn(7,5)*.003).half();actual=native.double()-theta
            event=r.inspect(actual);oracle=sum(.5*float((a.flatten(1)@(native.double()-birth).flatten()).square().sum()) for a,birth in history)
            assert abs(oracle-event['sampled_native_risk'])<=event['numeric_envelope']+1e-10
            r.advance(actual);theta=native.double()
        g=torch.randn(7,5);d,e=r.solve(g,epsilon=r.q+.002);B=torch.cat([x.flatten(1) for x,_ in history]);H=B.T@B
        oracle=-.01*torch.linalg.solve(torch.eye(35,dtype=torch.float64)+.01*e['multiplier']*H,g.flatten().double()+e['multiplier']*r.v)
        if float(oracle.norm())>.05:oracle*=.05/float(oracle.norm())
        assert torch.allclose(d,oracle,atol=1e-9) and r.inspect(d)['sampled_native_risk']<=r.q+.002+1e-8
        restored=Memory.restore(r.state(),'cpu');dd,_=restored.solve(g,epsilon=r.q+.002);assert torch.equal(dd,d)
    w=torch.zeros(3,3).half();r=Memory(w);rows=torch.zeros(1,3,3,dtype=torch.float64);rows[0,1,1]=1;r.consolidate(rows);r.prepare()
    actual=torch.zeros(3,3,dtype=torch.float64);actual[2,2]=.2;event=r.inspect(actual)
    assert event['native_risk_bound']<1e-12,'Old Frobenius nullspace counterexample must no longer reject'
    actual[1,1]=.2;assert r.inspect(actual)['native_risk_bound']>.01,'Observed harmful direction must still reject'
    g=torch.randn(3,3)
    r=Memory(w);r.prepare();d,_=r.solve(g);assert torch.allclose(d,-.01*g.flatten().double())
    print(json.dumps(dict(status='PASS',dense_multi_birth_native_oracle=True,unchanged_threshold_harm_rejected=True,
        old_nullspace_false_rejection_removed=True,full_gradient_retained=True,dense_KKT_oracle=True,restore_next_proposal_exact=True,
        numerical_envelope=True,CUDA_calls=0,model_calls=0)))

if __name__=='__main__':main()
