"""CPU check: original loss/update parity and diagnostics leave optimization unchanged."""
import sys,copy
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parent))
from training import update
import torch
from scripts.medtrace.stage18_cfact import update as original

class Expert(torch.nn.Module):
    def __init__(self):
        super().__init__();self.w=torch.nn.Parameter(torch.tensor([[.2,.1,-.3],[.1,-.2,.4]],dtype=torch.float32))
    def normalize_factors_(self,**kwargs):pass

class Runtime:
    def __init__(self,e):self.e=e;self.model=self
    def compute_loss(self,b):return torch.nn.functional.cross_entropy(b.x@self.e.w,b.y)
    def __call__(self,**kw):return SimpleNamespace(logits=(kw['x']@self.e.w).unsqueeze(0))

def check():
    hook=SimpleNamespace(set_teacher_routing=lambda labels:None)
    def batch(x,y):return SimpleNamespace(x=torch.tensor([x]),y=torch.tensor([y]),labels=torch.tensor([[y]]),target_token_ids=[y])
    native,fit,h=batch([1.,2.],0),batch([2.,1.],1),batch([-1.,2.],2)
    teacher=(dict(x=torch.tensor([[.3,.4]])),torch.tensor([[0]]),torch.tensor([[True]]),torch.tensor([[.1,.2,.3]]).log_softmax(-1),dict(role='U_fit'))
    for weight in (0.,1.):
        a,b=Expert(),Expert();oa,ob=torch.optim.Adam(a.parameters(),lr=.001),torch.optim.Adam(b.parameters(),lr=.001)
        extra=(h,dict(role='H_fit')) if weight else None
        ra=original(Runtime(a),hook,a,oa,native,fit,teacher,extra)
        rb=update(Runtime(b),hook,b,ob,native,fit,teacher,extra,extra_weight=weight,diagnostic=True)
        assert torch.equal(a.w,b.w) and ra['terms']==rb['terms'] and ra['gradient_norm']==rb['gradient_norm']
    for H,U in ((.25,.01),(0.,.05),(1.,.05),(.25,.05)):
        a,b=Expert(),Expert();oa,ob=torch.optim.Adam(a.parameters(),lr=.001),torch.optim.Adam(b.parameters(),lr=.001);extra=(h,dict(role='H_fit')) if H else None
        update(Runtime(a),hook,a,oa,native,fit,teacher,extra,extra_weight=H,U_weight=U)
        r=update(Runtime(b),hook,b,ob,native,fit,teacher,extra,extra_weight=H,U_weight=U,diagnostic=True)
        assert torch.equal(a.w,b.w)
        if H:assert all(-1.00001<=r['gradient_cosines'][k]<=1.00001 for k in ('H_U','H_edit'))
    from report import select,WEIGHTS
    assert select({})['status']=='WAIT_ALL_SIX_VALID_ARMS'
    def metrics():return {'native_accuracy':{'accuracy':1},'U_Retention':{'source_macro':1},'H_accuracy':{'accuracy':.5,'source_macro':.5},'H_Retention':{'source_macro':.5},'positive_accuracy':{'accuracy':1}}
    arms={k:metrics() for k in WEIGHTS};assert select(arms)['selected']=='E2'
    for k in ('E1','E2','E4','E5'):arms[k]['U_Retention']['source_macro']=0
    assert select(arms)['status']=='NO_FEASIBLE_H_WEIGHT'
    print('PASS original H1/NOH update parity; all candidate weights; diagnostics do not change update')

if __name__=='__main__':check()
