"""Cumulative sampled quadratic, rank-four FD completion, native candidate gate."""
import math
import torch


def dot(x, y):
    return torch.dot(x.flatten().double(), y.flatten().double())


class Risk:
    def __init__(self, native):
        w = native.flatten()
        self.B = torch.empty((0, w.numel()), device=w.device, dtype=torch.float64)
        self.v = torch.zeros_like(w, dtype=torch.float64)
        self.rho = self.q = 0.0
        self.ev = torch.zeros_like(self.v)
        self.erho = self.eq = 0.0
        self.compressions = 0

    def H(self, d):
        d = d.flatten().double()
        return self.B.T @ (self.B @ d) + self.rho*d

    def value(self, d):
        d = d.flatten().double()
        return self.q + float(dot(self.v,d) + .5*dot(d,self.H(d)))

    def advance(self, delta):
        d = delta.flatten().double()
        hd = self.H(d)
        self.q += float(dot(self.v,d)+.5*dot(d,hd))
        self.v += hd
        self.eq += float(dot(self.ev,d)+.5*self.erho*dot(d,d))
        self.ev += self.erho*d
        assert self.q >= -1e-9 and self.eq >= -1e-8

    def consolidate(self, rows):
        rows = rows.reshape(rows.shape[0],-1).double()
        assert 1 <= rows.shape[0] <= 4 and torch.isfinite(rows).all()
        stack = torch.cat((self.B,rows),0)
        assert len(stack) <= 8
        if len(stack) <= 4:
            self.B=stack; return dict(tau=0., rows=len(rows))
        # Small Gram eigensolve; the sketch remains in FP64 to bound numerical completion.
        gram = stack@stack.T
        vals,U=torch.linalg.eigh(gram)
        vals=vals.flip(0).clamp_min(0);U=U.flip(1)
        shrink=float(vals[4])
        scales=((vals[:4]-shrink).clamp_min(0)/vals[:4].clamp_min(1e-300)).sqrt()
        new = scales[:,None]*(U[:,:4].T@stack)
        # ||Gram rounding|| is covered conservatively, without claiming exact-real arithmetic.
        safety=128*torch.finfo(torch.float64).eps*max(float(vals.sum()),1.)*stack.shape[1]
        tau=shrink+safety
        self.B=new;self.rho+=tau;self.erho+=tau;self.compressions+=1
        # Completion is centered at the CURRENT native point: neither q nor v is reset.
        return dict(tau=tau,shrink=shrink,numerical_completion=safety,rows=len(rows))

    def state(self):
        return dict(B=self.B.cpu(),v=self.v.cpu(),rho=self.rho,q=self.q,ev=self.ev.cpu(),erho=self.erho,
                    eq=self.eq,compressions=self.compressions)

    def restore(self,state,device):
        assert set(state)=={'B','v','rho','q','ev','erho','eq','compressions'}
        for k,x in state.items():setattr(self,k,x.to(device) if isinstance(x,torch.Tensor) else x)
        assert len(self.B)<=4 and self.q>=-1e-9 and self.rho>=0

    def solve(self,g,eta=.01,epsilon=.01):
        g=g.flatten().double();B=self.B;v=self.v
        gg=float(dot(g,g));gv=float(dot(g,v));vv=float(dot(v,v))
        bg=B@g;bv=B@v;gram=B@B.T
        I=torch.eye(len(B),device=g.device,dtype=torch.float64)
        def reduced(lam,materialize=False):
            a=1+eta*lam*self.rho;t=eta*lam/a
            bh=bg+lam*bv
            z=torch.linalg.solve(I+t*gram,bh) if len(B) else bh
            # d=-eta/a*(g+lambda*v-t*B.T*z).
            norm=(gg+2*lam*gv+lam*lam*vv-2*t*float(dot(z,bh))+t*t*float(dot(z,gram@z)))*(eta/a)**2
            vd=-eta/a*(gv+lam*vv-t*float(dot(bv,z)))
            bd=-eta/a*(bh-t*(gram@z))
            risk=self.q+vd+.5*(float(dot(bd,bd))+self.rho*max(0.,norm))
            if materialize:return -eta/a*(g+lam*v-t*(B.T@z))
            return risk
        if reduced(0)<=epsilon:
            return reduced(0,True),dict(multiplier=0.,root_iterations=0,constrained=False)
        low,high=0.,1.;iterations=0
        while iterations<24 and reduced(high)>epsilon:
            low=high;high*=2;iterations+=1
        if reduced(high)>epsilon:
            return torch.zeros_like(g),dict(multiplier=high,root_iterations=iterations,solver_rejected=True)
        while iterations<48:
            mid=(low+high)/2
            if reduced(mid)>epsilon:low=mid
            else:high=mid
            iterations+=1
        return reduced(high,True),dict(multiplier=high,root_iterations=iterations,constrained=True)


def native_candidate(master,native,gradient,risk,eta=.01,epsilon=.01):
    delta,event=risk.solve(gradient,eta,epsilon)
    event['proposal_norm']=float(delta.norm())
    before=native.detach().float()
    for attempt in range(8):
        candidate_master=master+delta.reshape_as(master).float()*(.5**attempt)
        candidate_native=candidate_master.half()
        actual=candidate_native.float()-before
        q=risk.value(actual)
        proximal=float(dot(gradient,actual)+dot(actual,actual)/(2*eta))
        changed=int(torch.count_nonzero(actual))
        if changed and q<=epsilon and proximal<=0:
            risk.advance(actual)
            event.update(accepted=True,native_backtracks=attempt,actual_native_norm=float(actual.norm()),
                         changed_elements=changed,risk_after=risk.q,proximal_objective=proximal,
                         native_master_discrepancy=float((candidate_native.float()-candidate_master).norm()))
            return candidate_master,candidate_native,event
    event.update(accepted=False,reason='ZERO_NATIVE_OR_FINITE_NATIVE_GATE_REJECT',native_backtracks=8,
                 actual_native_norm=0.,changed_elements=0,risk_after=risk.q,
                 native_master_discrepancy=float((before-master).norm()))
    return master,native.detach(),event


def selfcheck():
    torch.manual_seed(83)
    w=torch.randn(17).half();r=Risk(w);dense=torch.zeros(17,17,dtype=torch.float64)
    for _ in range(6):
        rows=torch.randn(4,17,dtype=torch.float64)*.03
        oldq,oldv=r.q,r.v.clone();prior=r.B.T@r.B+r.rho*torch.eye(17,dtype=torch.float64)
        e=r.consolidate(rows);new=r.B.T@r.B+r.rho*torch.eye(17,dtype=torch.float64)
        added=new-prior-rows.T@rows
        assert torch.linalg.eigvalsh(added).min()>=-1e-10
        assert r.q==oldq and torch.equal(r.v,oldv)
        d=torch.randn(17,dtype=torch.float64)*.01
        q=r.value(d);r.advance(d);assert abs(q-r.q)<1e-12
    g=torch.randn(17,dtype=torch.float64)
    d,e=r.solve(g,epsilon=r.q+.002)
    H=r.B.T@r.B+r.rho*torch.eye(17,dtype=torch.float64)
    oracle=-.01*torch.linalg.solve(torch.eye(17,dtype=torch.float64)+.01*e['multiplier']*H,g+e['multiplier']*r.v)
    assert torch.allclose(d,oracle,atol=1e-9) and r.value(d)<=r.q+.002+1e-10
    s=r.state();rr=Risk(w);rr.restore(s,'cpu');assert torch.equal(rr.H(d),r.H(d))
    zero=Risk(w);master=w.float();m,n,event=native_candidate(master,w,torch.zeros_like(master),zero)
    assert not event['accepted'] and torch.equal(m,master) and torch.equal(n,w)
    return dict(status='PASS',FD_PSD_completion=True,cumulative_q_v_preserved=True,dense_KKT_oracle=True,
                native_zero_discards_master=True,fixed_rank_rows=4,CPU_GPU_updates=0)


if __name__=='__main__':
    import json
    print(json.dumps(selfcheck()))
