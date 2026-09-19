"""Small CPU check for differentiable update diagnostics and JSON persistence."""
import sys,json
from pathlib import Path
import torch
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).parent))
import regularizer
from training import update


def check():
    torch.manual_seed(24001)
    class Expert(torch.nn.Module):
        def __init__(self):super().__init__();self.p=torch.nn.Parameter(torch.randn(5,3))
        def normalize_factors_(self,**kw):pass
    e=Expert();o=torch.optim.Adam(e.parameters(),lr=.01);hook=SimpleNamespace(set_teacher_routing=lambda x:None)
    runtime=SimpleNamespace(compute_loss=lambda b:e.p.square().mean(),model=lambda **kw:SimpleNamespace(logits=e.p[None,:,:]))
    batch=SimpleNamespace(labels=torch.ones(1,5,dtype=torch.long),target_token_ids=[1,2])
    teacher=({},batch.labels,torch.ones(1,5,dtype=torch.bool),torch.zeros(5,3).log_softmax(-1),{})
    original=regularizer.capture
    regularizer.capture=lambda *args:({'input':torch.arange(15.).reshape(5,3),'patch':e.p,'final':e.p.square()},[])
    try:
        r=update(runtime,hook,e,o,batch,batch,teacher,extra=(batch,{}),extra_weight=.25,diagnostic=True,regularize=True,batches=[batch]*5)
        json.dumps(r);assert r['regularization']['x_gradient_norm']>0 and r['regularization']['y_gradient_norm']>0
    finally:regularizer.capture=original
    print('PASS: real regularizer gradient diagnostics serialize as data')

if __name__=='__main__':check()
