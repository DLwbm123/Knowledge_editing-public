"""Production CPU checks in a disposable remote directory, no GPU/model/Judge."""
import io,json,os,shutil,sys,tempfile,time,copy,importlib.util,contextlib,random
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
payload=json.loads(sys.stdin.readline())
root=Path(tempfile.mkdtemp(prefix='e.',dir='/data/bmw/tmp'));tools=root/'private/tools';tools.mkdir(parents=True)
parent=Path('/data/bmw/Knowledge_editing/outputs/purew-objective-scope-v1-20261015/run/private')
for path in (parent/'tools').glob('*.py'):shutil.copy2(path,tools/path.name)
for name,text in payload['files'].items():(tools/name).write_text(text)
os.environ.update(RUN_ROOT=str(root),ACTION='CPU_PREPARATION',ARM='PM',CUDA_VISIBLE_DEVICES='',TMPDIR=str(root),SCOPE_DIAGNOSTICS='0')
sys.path[:0]=[str(tools),str(parent/'source')]
(root/'private/FROZEN_PROTOCOL.json').write_text(json.dumps(payload['protocol']))
import torch,numpy as np
torch.set_num_threads(4);torch.manual_seed(310);started=time.monotonic()
def forbidden(*a,**kw):raise AssertionError('CPU cannot call CUDA')
torch.cuda.init=torch.cuda._lazy_init=forbidden;torch.cuda.is_available=lambda:False
import worker,source_loss,core,runtime,medical
from core import Memory
def exact(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
    return a==b

# The production split supports extreme logits without probability floor.
z0=torch.tensor([[2.,1.,-3.],[-20.,30.,2.]],dtype=torch.float64)
y=torch.tensor([0,2]);prefix0=torch.tensor([[2.,1.,-1.]],dtype=torch.float64)
anchor=source_loss.teacher(z0,y,prefix0)
z=torch.tensor([[.4,.2,-.2],[.3,-.1,.8]],dtype=torch.float64,requires_grad=True)
pr=torch.tensor([[.1,.3,-.1]],dtype=torch.float64,requires_grad=True)
parts=source_loss.components(z,y,pr,anchor)
q0=anchor['log_q'].exp();star=torch.empty_like(z0)
mask=torch.ones_like(star,dtype=torch.bool).scatter_(1,y[:,None],False)
star[mask]=(anchor['log_rest'].exp()[:,None]*q0).flatten()
star.scatter_(1,y[:,None],anchor['log_a'].exp()[:,None])
oracle=(star*(star.log()-torch.log_softmax(z,-1))).sum(-1).mean()
assert torch.allclose(parts['marginal_KL']+parts['conditional_KL'],oracle,atol=1e-12,rtol=0)
g=torch.autograd.grad(parts['marginal_KL']+parts['conditional_KL'],z,retain_graph=True)[0]
assert torch.allclose(g,(z.softmax(-1)-star)/len(y),atol=1e-12,rtol=0)
gp=torch.autograd.grad(parts['marginal_KL'],pr,allow_unused=True)[0];assert gp is None
assert torch.autograd.grad(parts['prefix_KL'],pr)[0].abs().max()>0
ext=torch.tensor([[1000.,-1000.,-999.],[-1000.,1000.,999.]],dtype=torch.float64,requires_grad=True)
ref=source_loss.teacher(ext.detach(),y,prefix0)
v=source_loss.components(ext,y,pr,ref);extg=torch.autograd.grad(v['marginal_KL']+v['conditional_KL'],ext)[0]
assert torch.isfinite(extg).all()
assert ref['log_a'][0].exp()==1. and ref['log_rest'][0]<-1000
# Realized image expansion and special-token exclusions, shifted labels unchanged.
raw=torch.tensor([[1,10,-200,11,12,20,2]])
rawlabels=torch.tensor([[-100,-100,-100,-100,-100,20,2]])
labels=torch.tensor([[-100,-100,-100,-100,-100,-100,-100,20,2]])
prepared=SimpleNamespace(labels=labels,raw_input_ids=raw,raw_labels=rawlabels,attention_mask=torch.ones_like(labels),image_token_start=2,image_token_end=5)
pm=source_loss.prefix_mask(prepared,[1,2]);assert pm.tolist()==[[False,False,False,False,False,True,False,False]]
bad=copy.deepcopy(prepared);bad.labels[0,-1]=3
try:source_loss.prefix_mask(bad,[1,2])
except AssertionError:pass
else:raise AssertionError('Corrupted mask mapping admitted')
# Actual loss selector: PM has no prefix gradient, PU does, both same reference.
fake=SimpleNamespace(_projection_cache=[anchor]*5)
for arm in ('PM','PU'):
    za=z.detach().clone().requires_grad_(True);zp=pr.detach().clone().requires_grad_(True)
    with patch.object(source_loss,'ARM',arm),patch.object(source_loss,'position_logits',return_value=(za,y,zp)):
        total,ce,_=source_loss.loss(fake,[None]*5,[],[],1)
        grads=torch.autograd.grad(total,(za,zp),allow_unused=True)
        assert (grads[1] is None)==(arm=='PM')
        assert abs(ce-float(torch.nn.functional.cross_entropy(za,y)))<1e-12

# Same old SH00 candidate path bitwise for an identical supplied objective.
spec=importlib.util.spec_from_file_location('historical_worker',parent/'tools/worker.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
weight=torch.ones((4,5),dtype=torch.float16,requires_grad=True);native=weight.detach().clone();master=weight.detach().float().clone()
risk=Memory(weight,'J');risk.consolidate(torch.ones((1,20),dtype=torch.float64)*.01,'Base');initial=copy.deepcopy(risk.state())
fake_loss=lambda *args:(weight.float().square().mean(),float(weight.detach().float().square().mean()),0.)
for module in (worker,old):
    module.loss=fake_loss;module.backward=lambda value,w:torch.autograd.grad(value,w)[0].float()
worker.b.call=lambda *a:None;worker.b.counts=lambda:None
outputs=[]
for module in (worker,old):
    with torch.no_grad():weight.copy_(native)
    mem=Memory.restore(copy.deepcopy(initial));out,event=module.attempt(None,weight,master.clone(),mem,[],[],[],1,False)
    outputs.append((weight.detach().clone(),out,event,mem.state()))
assert exact(*outputs)
accepted=outputs[0][2];assert accepted['accepted']
# The candidate gate rejects without master/native/risk changes; no committed count.
with torch.no_grad():weight.copy_(native)
mem=Memory.restore(copy.deepcopy(initial));count=worker.b.COUNTS['committed_updates']
with patch.object(mem,'inspect',return_value={'Base':dict(native_risk_bound=.02,sampled_native_risk=.02,numeric_envelope_dominates=False)}):
    out,event=worker.attempt(None,weight,master.clone(),mem,[],[],[],1,False)
assert not event['accepted'] and torch.equal(weight,native) and torch.equal(out,master)
assert exact(mem.state(),initial) and worker.b.COUNTS['committed_updates']==count
# State and source clock round-trip; next proposal exact.
stream=io.BytesIO();torch.save(mem.state(),stream);stream.seek(0)
restored=Memory.restore(torch.load(stream,weights_only=True))
gradient=torch.arange(20,dtype=torch.float32).reshape(4,5)/20
assert exact(mem.solve(gradient),restored.solve(gradient))
# Diagnostic losses and modes may not consume training RNG.
net=torch.nn.Sequential(torch.nn.Linear(2,2),torch.nn.Dropout(.5));net[0].eval()
modes=[m.training for m in net.modules()];rng=worker.c.rng();nrng=np.random.get_state()
try:
    with worker.diagnostic(SimpleNamespace(model=net)):
        net.eval();torch.rand(1);random.random();np.random.rand();raise ValueError('fixture')
except ValueError:pass
assert exact(rng,worker.c.rng()) and np.array_equal(nrng[1],np.random.get_state()[1])
assert modes==[m.training for m in net.modules()]
