"""Complete finite sampled curvature; actual native displacement, no isotropic rounding bound."""
import math
import torch
from basis import basis
U=torch.finfo(torch.float64).eps/2

def dot(a,b):return float(torch.dot(a.flatten().double(),b.flatten().double()))
def gamma(n):
    assert n*U<1
    return n*U/(1-n*U)

class Memory:
    def __init__(self,native,restricted=False):
        self.shape=tuple(native.shape);self.n=native.numel();self.restricted=restricted
        self.B=torch.empty((0,self.n),device=native.device,dtype=torch.float64)
        self.v=torch.zeros(self.n,device=native.device,dtype=torch.float64)
        self.q=self.eq=self.ev=self.eH=0.
        self.left,self.right=basis(native) if restricted else (None,None)
        self.prepared=None

    def project(self,x):
        return (self.left.T@x.reshape(self.shape).double()@self.right).flatten() if self.restricted else x.flatten().double()

    def embed(self,x):
        return (self.left@x.reshape(self.left.shape[1],self.right.shape[1])@self.right.T).flatten() if self.restricted else x.flatten()

    def prepare(self):
        a=torch.stack([self.project(x) for x in self.B]) if self.restricted and len(self.B) else self.B
        if self.restricted and not len(self.B):a=self.B.new_empty((0,self.left.shape[1]*self.right.shape[1]))
        self.prepared=(a,a@a.T)

    def inspect(self,actual):
        d=actual.flatten().double();dn=float(d.norm());z=self.B@d
        vd=dot(self.v,d);zz=dot(z,z);value=self.q+vd+.5*zz
        bnorm=float(self.B.norm());vnorm=float(self.v.norm());ez=gamma(self.n)*bnorm*dn
        arithmetic=gamma(self.n)*vnorm*dn+.5*(2*float(z.norm())*ez+ez*ez)+gamma(max(3,len(z)))*(abs(self.q)+abs(vd)+.5*zz)
        error=self.eq+self.ev*dn+.5*self.eH*dn*dn+arithmetic
        assert math.isfinite(value+error) and value>=-error-1e-10
        return dict(sampled_native_risk=max(0.,value),signed_quadratic=value,numeric_envelope=error,
                    native_risk_bound=max(0.,value)+error,actual_native_norm=dn,
                    quantization_Frobenius_envelope_used=False,numeric_envelope_dominates=error>=.0001)

    def advance(self,actual):
        d=actual.flatten().double();event=self.inspect(d);z=self.B@d;hd=self.B.T@z
        dn=float(d.norm());bnorm=float(self.B.norm());ez=gamma(self.n)*bnorm*dn
        self.ev+=self.eH*dn+bnorm*ez+gamma(max(1,len(self.B)))*bnorm*float(z.norm())+U*(float(self.v.norm())+float(hd.norm()))
        self.eq=event['numeric_envelope'];self.q=event['sampled_native_risk'];self.v+=hd
        return event

    def consolidate(self,rows,block='answer'):
        # Full rotation mixes source sensitivities; no source-indexed root rows survive.
        assert block in ('answer','evidence') and torch.isfinite(rows).all()
        previous=self.B.cpu();self.B=self.B.new_empty((0,self.n))
        stack=torch.cat((previous,rows.reshape(len(rows),self.n).double().cpu()),0);del previous
        assert len(stack)<=16,'Only frozen first-three finite support; not a 146 implementation'
        _,rotation=torch.linalg.eigh(stack@stack.T)
        err=float((rotation.T@rotation-torch.eye(len(rotation),dtype=torch.float64)).abs().max())
        assert err<1e-12
        trace=float(stack.square().sum());r=len(stack);du=r*err;eta=gamma(r)*r
        self.eH+=(du+2*math.sqrt(1+du)*eta+eta*eta)*trace
        rotated=torch.empty_like(stack)
        for start in range(0,self.n,262144):rotated[:,start:start+262144]=rotation.T@stack[:,start:start+262144]
        self.B=rotated.to(self.v.device);self.prepared=None
        return dict(block=block,rows_added=len(rows),retained_parameter_statistic_rows=len(self.B),
                    covariance_trace=trace,orthogonal_error=err,numeric_operator_error=self.eH,
                    compression='NONE_COMPLETE_FINITE_SUPPORT')

    def solve(self,gradient,eta=.01,epsilon=.01):
        assert self.prepared is not None
        a,gram=self.prepared;g=self.project(gradient);v=self.project(self.v)
        gg,gv,vv=dot(g,g),dot(g,v),dot(v,v);ag=a@g;av=a@v
        eye=torch.eye(len(a),device=g.device,dtype=torch.float64)
        def reduced(lam,materialize=False):
            t=eta*lam;h=ag+lam*av;z=torch.linalg.solve(eye+t*gram,h) if len(a) else h
            if materialize:return -eta*(g+lam*v-t*(a.T@z))
            vd=-eta*(gv+lam*vv-t*dot(av,z));ad=-eta*(h-t*(gram@z))
            return self.q+vd+.5*dot(ad,ad)
        iterations=0;lo=0.;hi=1.
        if reduced(0)<=epsilon:lam=0.
        else:
            while iterations<24 and reduced(hi)>epsilon:lo=hi;hi*=2;iterations+=1
            if reduced(hi)>epsilon:
                return torch.zeros_like(self.v),dict(solver_reason='FINITE_ROOT_REJECT',root_iterations=iterations)
            while iterations<48:
                mid=(lo+hi)/2
                if reduced(mid)>epsilon:lo=mid
                else:hi=mid
                iterations+=1
            lam=hi
        d=self.embed(reduced(lam,True));norm=float(d.norm())
        if norm>.05:d*=.05/norm
        return d,dict(multiplier=lam,root_iterations=iterations,proposal_norm=float(d.norm()),
            gradient_retained_norm_fraction=float(g.norm())/max(float(gradient.double().norm()),1e-30),
            writable_space='SVD32x32' if self.restricted else 'FULL_ORIGINAL_W',proximal_metric='IDENTITY_BOTH_ARMS')

    def state(self):
        return {k:(v.detach().cpu().clone() if isinstance(v,torch.Tensor) else v) for k,v in self.__dict__.items() if k!='prepared'}

    @classmethod
    def restore(cls,state,device):
        obj=cls.__new__(cls)
        for k,v in state.items():setattr(obj,k,v.to(device) if isinstance(v,torch.Tensor) else v)
        obj.prepared=None;assert len(obj.B)<=16 and obj.q>=-1e-9
        obj.prepare();return obj
