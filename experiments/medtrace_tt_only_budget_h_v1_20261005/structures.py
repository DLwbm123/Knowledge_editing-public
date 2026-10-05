"""Only four TT cores are independent parameters, including deployment."""
import torch
from torch import nn

RANKS={'TT44':(4,4),'TT84':(8,4),'TT48':(4,8),'TT88':(8,8)}
COUNTS={'TT44':3584,'TT84':4864,'TT48':5888,'TT88':7168}

def orthogonal(x):
    q,r=torch.linalg.qr(x,mode='reduced')
    if not torch.isfinite(r).all() or torch.any(torch.diag(r).abs()<1e-8):raise FloatingPointError('rank deficient initialization')
    return q*torch.where(torch.diag(r)<0,-torch.ones_like(torch.diag(r)),torch.ones_like(torch.diag(r)))

class TT4(nn.Module):
    def __init__(self,seed,rL=4,rR=4):
        super().__init__();assert (rL,rR) in RANKS.values()
        self.d_in,self.d_out,self.rank,self.epsilon=14336,4096,4,1e-6
        rng=torch.Generator().manual_seed(seed)
        self.G1=nn.Parameter(torch.zeros(1,64,rL))
        for name,shape in [('G2',(rL,64,4)),('G3',(4,112,rR)),('G4',(rR,128,1))]:
            a=torch.randn(shape[1]*shape[2],shape[0],generator=rng)
            setattr(self,name,nn.Parameter(orthogonal(a).T.reshape(shape).contiguous()))
    def factors(self):
        return (torch.einsum('ia,ajb->ijb',self.G1[0],self.G2).reshape(4096,4),
                torch.einsum('aib,bj->aij',self.G3,self.G4[:,:,0]).reshape(4,14336))
    def residual(self,activation):
        b,a=self.factors();x=activation.to(a.dtype);x=x/(x.square().mean(-1,keepdim=True).sqrt()+self.epsilon)
        return (x@a.T)@b.T
    def normalize_factors_(self,**_):pass
    def parameter_groups(self):return [self.G3,self.G4],[self.G1,self.G2]

def optimizer_for(e,base,scale=1.):
    assert isinstance(e,TT4) and not any(p.requires_grad for p in base.parameters())
    assert set(dict(e.named_parameters()))=={'G1','G2','G3','G4'} and all(p.dtype==torch.float32 for p in e.parameters())
    inputs,outputs=e.parameter_groups()
    opt=torch.optim.Adam([{'params':inputs,'lr':scale*1e-4},{'params':outputs,'lr':scale*1e-3}],betas=(.9,.999),eps=1e-8,weight_decay=0)
    assert {id(p) for g in opt.param_groups for p in g['params']}=={id(p) for p in e.parameters()}
    return opt
