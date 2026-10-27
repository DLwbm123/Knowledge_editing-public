"""Only the 512 root coordinates move; canonical tails stay bitwise fixed."""
import torch
from path_v18 import proposal

def root_step(r,before,gradient,scalar):
 assert list(before)==['G1','G2','G3','G4'] and gradient.numel()==7168
 masked=torch.zeros_like(gradient);n=before['G1'].numel();assert n==512
 masked[:n]=gradient[:n]
 raw,updated,_=proposal(r,before,masked,scalar,list(before))
 # Fixed right-canonical tails need no re-framing after a root-only update.
 assert all(torch.equal(raw[k],before[k]) for k in ('G2','G3','G4'))
 return raw,updated

def selfcheck():
 import module_math as mm
 from alignment import TT,snapshot
 from types import SimpleNamespace
 r=SimpleNamespace(mm=mm);initial=mm.canonicalize(snapshot(TT(225034559)))[0]
 grad=torch.arange(7168,dtype=torch.float64)/7168;copy=grad.clone();s=dict(step=0,second_moments=[0.,0.]);a,aa=root_step(r,initial,grad,s)
 b,bb=root_step(r,initial,torch.cat([grad[:512],torch.zeros_like(grad[512:])]),s)
 assert aa==bb and all(torch.equal(a[k],b[k]) for k in a)
 assert a['G1'].norm()>0 and aa['second_moments'][1]==0 and s==dict(step=0,second_moments=[0.,0.]) and torch.equal(grad,copy)
 c,cc=root_step(r,a,grad,aa);assert cc['step']==2
 assert all(torch.equal(initial[k],c[k]) for k in ('G2','G3','G4'))
 z,zz=root_step(r,initial,torch.zeros_like(grad),s);assert all(torch.equal(z[k],initial[k]) for k in z)
 return dict(status='PASS',active_root512=True,tails_bitwise_fixed=True,tail_gradients_ignored=True,RMS_group2_zero=True,zero_step=True,no_alias=True,real_model_calls=0)
