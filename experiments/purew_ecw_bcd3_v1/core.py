"""Independent finite quadratic blocks on CPU; only column tiles reach the GPU."""
import math
import numpy as np
import torch
from scipy.optimize import minimize
U=torch.finfo(torch.float64).eps/2
TILE=262144

def gamma(n):
    assert n*U<1
    return n*U/(1-n*U)

def dot(a,b):return float(torch.dot(a.flatten().double(),b.flatten().double()))

class Block:
    def __init__(self,n):
        self.B=torch.empty((0,n),dtype=torch.float64);self.v=torch.zeros(n,dtype=torch.float64)
        self.q=self.eq=self.ev=self.eH=0.
    def inspect(self,d):
        dn=float(d.norm());z=self.B@d;vd=dot(self.v,d);zz=dot(z,z);value=self.q+vd+.5*zz
        bn=float(self.B.norm());ez=gamma(d.numel())*bn*dn
        arithmetic=gamma(d.numel())*float(self.v.norm())*dn+.5*(2*float(z.norm())*ez+ez*ez)+gamma(max(3,len(z)))*(abs(self.q)+abs(vd)+.5*zz)
        error=self.eq+self.ev*dn+.5*self.eH*dn*dn+arithmetic
        assert math.isfinite(value+error) and value>=-error-1e-10
        return dict(sampled_native_risk=max(0.,value),signed_quadratic=value,numeric_envelope=error,
            native_risk_bound=max(0.,value)+error,numeric_envelope_dominates=error>=.0001)
    def advance(self,d):
        event=self.inspect(d);z=self.B@d;hd=self.B.T@z;dn=float(d.norm());bn=float(self.B.norm())
        self.ev+=self.eH*dn+bn*gamma(d.numel())*bn*dn+gamma(max(1,len(self.B)))*bn*float(z.norm())+U*(float(self.v.norm())+float(hd.norm()))
        self.eq=event['numeric_envelope'];self.q=event['sampled_native_risk'];self.v+=hd
        return event
    def add(self,rows):
        rows=rows.detach().reshape(len(rows),-1).double().cpu();assert torch.isfinite(rows).all()
        stack=torch.cat((self.B,rows));self.B=torch.empty((0,stack.shape[1]),dtype=torch.float64)
        gram=stack@stack.T;_,rotation=torch.linalg.eigh(gram)
        err=float((rotation.T@rotation-torch.eye(len(rotation),dtype=torch.float64)).abs().max());assert err<1e-12
        trace=float(torch.trace(gram));du=len(stack)*err;rounding=gamma(len(stack))*len(stack)
        self.eH+=(du+2*math.sqrt(1+du)*rounding+rounding**2)*trace
        # In-place column rotation bounds host transient storage to one small tile.
        for i in range(0,stack.shape[1],TILE):stack[:,i:i+TILE]=rotation.T@stack[:,i:i+TILE]
        self.B=stack
        return dict(rows_added=len(rows),retained_rows=len(stack),covariance_trace=trace,orthogonal_error=err,numeric_operator_error=self.eH,compression='NONE_COMPLETE_FINITE_SUPPORT')

class Memory:
    def __init__(self,native,arm='B'):
        self.shape=tuple(native.shape);self.n=native.numel();self.arm=arm
        self.blocks={'Base':Block(self.n),'history':Block(self.n)};self.prepared=None
        self.path=0.;self.birth_moments=[0.,0.,0.,0.]
    @property
    def q(self):return {k:b.q for k,b in self.blocks.items()}
    def consolidate(self,rows,block='history'):
        if block not in self.blocks:self.blocks[block]=Block(self.n)
        result=self.blocks[block].add(rows)
        assert sum(len(b.B) for b in self.blocks.values())<={'B':32,'C':46,'D':68}[self.arm]
        self.birth_moments[0]+=1.;self.prepared=None
        return dict(block=block,**result)
    def prepare(self):
        names=list(self.blocks);self.prepared=dict(names=names,sizes=[len(self.blocks[k].B) for k in names])
        bs=[self.blocks[k].B for k in names];vs=torch.stack([self.blocks[k].v for k in names]);m=sum(len(b) for b in bs)
        gram=torch.zeros((m,m),dtype=torch.float64);bv=torch.zeros((m,len(names)),dtype=torch.float64)
        for i in range(0,self.n,TILE):
            tile=torch.cat([b[:,i:i+TILE] for b in bs]);gram+=tile@tile.T;bv+=tile@vs[:,i:i+TILE].T
        self.prepared.update(gram=gram,bv=bv,vv=vs@vs.T)
    def inspect(self,actual):
        d=actual.detach().flatten().double().cpu()
        return {k:b.inspect(d) for k,b in self.blocks.items()}
    def advance(self,actual):
        d=actual.detach().flatten().double().cpu();out={k:b.advance(d) for k,b in self.blocks.items()}
        length=float(d.norm());a0,a1,a2,a3=self.birth_moments
        self.birth_moments=[a0,a1+length*a0,a2+2*length*a1+length**2*a0,a3+3*length*a2+3*length**2*a1+length**3*a0]
        self.path+=length;self.prepared=None
        return out
    def solve(self,gradient,eta=.01,epsilon=.01):
        if self.prepared is None:self.prepare()
        p=self.prepared;names=p['names'];g=gradient.detach().flatten().double().cpu();bs=[self.blocks[k].B for k in names]
        v=torch.stack([self.blocks[k].v for k in names]);bg=torch.cat([b@g for b in bs]);vg=v@g;gg=dot(g,g)
        q=torch.tensor([self.blocks[k].q for k in names],dtype=torch.float64);eye=torch.eye(len(bg),dtype=torch.float64)
        groups=torch.repeat_interleave(torch.arange(len(names)),torch.tensor(p['sizes']))
        def reduced(x,materialize=False):
            lam=torch.as_tensor(x,dtype=torch.float64);s=lam[groups].sqrt();bh=bg+p['bv']@lam
            z=torch.linalg.solve(eye+eta*s[:,None]*p['gram']*s[None,:],s*bh) if len(s) else s
            t=s*z;bd=-eta*(bh-eta*p['gram']@t);vd=-eta*(vg+p['vv']@lam-eta*p['bv'].T@t)
            risks=q+vd
            for j in range(len(names)):risks[j]+=.5*dot(bd[groups==j],bd[groups==j])
            hh=gg+2*dot(lam,vg)+dot(lam,p['vv']@lam)
            dual=-.5*eta*(hh-eta*dot(s*bh,z))+dot(lam,q-epsilon)
            if not materialize:return -dual,-(risks-epsilon).numpy()
            d=-eta*(g+lam@v)
            offset=0
            for b in bs:
                coeff=t[offset:offset+len(b)];offset+=len(b)
                for i in range(0,self.n,TILE):d[i:i+TILE]+=eta**2*(b[:,i:i+TILE].T@coeff)
            return d,risks
        fit=minimize(reduced,np.zeros(len(names)),jac=True,bounds=[(0,None)]*len(names),method='L-BFGS-B',
            options=dict(maxiter=48,maxls=8,ftol=1e-15,gtol=1e-10))
        d,risks=reduced(fit.x,True);projected=np.where(fit.x>1e-8,fit.jac,np.minimum(fit.jac,0))
        kkt=float(np.max(np.abs(projected)))
        # Dual convergence is not primal feasibility: retract along this same ray
        # to the analytic quadratic boundary, without changing any risk threshold.
        factor=1.
        vd=torch.tensor([dot(self.blocks[k].v,d) for k in names],dtype=torch.float64)
        quadratic=risks-q-vd
        for j in range(len(names)):
            if risks[j]>epsilon:
                a=max(0.,float(quadratic[j]));z=float(vd[j]);gap=epsilon-float(q[j])
                assert gap>=-1e-12
                if a>0:
                    disc=math.sqrt(z*z+4*a*max(0.,gap))
                    root=2*max(0.,gap)/(disc+z) if z>=0 and disc+z>0 else (-z+disc)/(2*a)
                else:root=max(0.,gap)/z if z>0 else 1.
                factor=min(factor,max(0.,root)*(1-32*torch.finfo(torch.float64).eps))
        d*=factor
        risks=q+factor*vd+factor**2*quadratic
        feasible=bool((risks<=epsilon).all())
        event=dict(solver_reason='PASS' if fit.success and feasible and kkt<=1e-7 else 'FINITE_DUAL_REJECT',
            primal_ray_retraction=factor,dual_iterations=int(fit.nit),dual_evaluations=int(fit.nfev),dual_status=int(fit.status),multipliers=dict(zip(names,fit.x.tolist())),
            predicted_risks=dict(zip(names,risks.tolist())),projected_KKT_residual=kkt,writable_space='FULL_ORIGINAL_W',proximal_metric='IDENTITY',CPU_streamed_FP64=True)
        if event['solver_reason']!='PASS':d.zero_()
        norm=float(d.norm())
        if norm>.05:d*=.05/norm
        event['proposal_norm']=float(d.norm())
        return d,event
    def state(self):
        return dict(shape=self.shape,n=self.n,arm=self.arm,blocks={k:dict(b.__dict__) for k,b in self.blocks.items()},path=self.path,birth_moments=self.birth_moments)
    @classmethod
    def restore(cls,state,device=None):
        obj=cls.__new__(cls);obj.shape=state['shape'];obj.n=state['n'];obj.arm=state['arm'];obj.path=state['path'];obj.birth_moments=list(state['birth_moments']);obj.prepared=None;obj.blocks={}
        for k,s in state['blocks'].items():
            block=Block.__new__(Block);block.__dict__.update(s);assert block.B.device.type=='cpu' and block.v.device.type=='cpu';obj.blocks[k]=block
        assert all(math.isfinite(b.q+b.eq+b.ev+b.eH) and b.q>=-1e-9 for b in obj.blocks.values())
        return obj
