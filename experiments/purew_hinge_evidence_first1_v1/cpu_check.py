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
os.environ.update(RUN_ROOT=str(root),ACTION='CPU_PREPARATION',ARM='HE',CUDA_VISIBLE_DEVICES='',TMPDIR=str(root),SCOPE_DIAGNOSTICS='0')
sys.path[:0]=[str(tools),str(parent/'source')]
(root/'private/FROZEN_PROTOCOL.json').write_text(json.dumps(payload['protocol']))
import torch,numpy as np
torch.set_num_threads(4);torch.manual_seed(310);started=time.monotonic()
def forbidden(*a,**kw):raise AssertionError('CPU cannot call CUDA')
torch.cuda.init=torch.cuda._lazy_init=forbidden;torch.cuda.is_available=lambda:False
assert not torch.cuda.is_initialized()
assert not any(x in Path('/proc/self/cmdline').read_bytes().decode() for x in ('wangbomin','Knowledge_editing','purew','ecw'))
import worker,source_loss,core,runtime,medical
from core import Memory
# Regression: the learner calls the module context manager before terminal margin reporting.
import symtable
learner_scope=next(x for x in symtable.symtable(payload['files']['worker.py'],'worker.py','exec').get_children() if x.get_name()=='learner')
assert learner_scope.lookup('diagnostic').is_global(), 'Learner shadows diagnostic context manager'
assert learner_scope.lookup('margin_diagnostic').is_local()
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
# Hinge derivative, zero boundary subgradient, and high teacher threshold.
for offset in (-.3,0.,.3):
    y1=torch.tensor([0]);base=torch.tensor([[5.,0.,-1.]],dtype=torch.float64)
    ref1=source_loss.teacher(base,y1,prefix0)
    logq=ref1['log_q'];odds=(ref1['log_a']-ref1['log_rest']).item()
    zz=torch.cat((torch.tensor([[odds+offset]],dtype=torch.float64),logq),1).requires_grad_()
    v1=source_loss.components(zz,y1,pr,ref1)
    gg=torch.autograd.grad(v1['odds_hinge'],zz)[0]
    if offset<0:
        assert abs(float(v1['odds_hinge'])+offset)<1e-12
        assert torch.allclose(gg,torch.cat((-torch.ones((1,1),dtype=torch.float64),logq.exp()),1),atol=1e-12,rtol=0)
    else:assert float(v1['odds_hinge'])==0 and not gg.any()
    assert ref1['log_a'].exp().item()>.95
v=source_loss.components(ext,y,pr,ref)
assert torch.isfinite(torch.autograd.grad(v['odds_hinge'],ext)[0]).all()
with patch.object(source_loss,'position_logits',return_value=(z,y,pr)):
    m=source_loss.measurements(SimpleNamespace(_projection_cache=[anchor]*5,_evidence_indices=[[[0],[1]] for _ in range(5)]),[None]*5)
    assert 'target_gap_min' not in m
    assert m['target_gap_global_min']==m['target_gap_view_min_weighted']
    assert m['target_pass_fraction']<=m['alpha_pass_fraction']
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
# Actual HE selector: independent concept-margin oracle and both-side gradients.
fake=SimpleNamespace(_projection_cache=[anchor]*5,_evidence_indices=[[[0],[1]] for _ in range(5)])
full=list(range(5));removed=list(range(5,10))
zfull=[(z.detach()+i*.07*torch.tensor([[1.,0.,-1.],[0.,1.,-1.]])).requires_grad_() for i in range(5)]
zremoved=[(z.detach()+i*.11*torch.tensor([[0.,1.,-1.],[1.,0.,-1.]])).requires_grad_() for i in range(5)]
zprefix=pr.detach().clone().requires_grad_()
anchors=[torch.softmax(z0+i*.13,-1).detach() for i in range(5)]
def position(_,k):return zfull[k],y,zprefix
def removed_logits(_,k):return zremoved[k-5],y
for slot in range(1,5):
    with patch.object(source_loss,'position_logits',side_effect=position),patch.object(source_loss.b,'logits',side_effect=removed_logits):
        total,ce,kl=source_loss.loss(fake,full,removed,anchors,slot)
        oracle=0.
        for k in (0,slot):
            la,lr,_=source_loss.split_logits(zfull[k],y)
            hinge=torch.relu(anchor['log_a']-anchor['log_rest']-la+lr).mean()
            lp=torch.log_softmax(zfull[k],-1).gather(1,y[:,None]).flatten()
            lr=torch.log_softmax(zremoved[k],-1).gather(1,y[:,None]).flatten()
            direct=torch.relu(__import__('math').log(1.01)-(lp-lr)).mean()
            oracle+=.5*(hinge+direct)
        assert torch.allclose(total,oracle,atol=1e-12,rtol=0)
        params=tuple(zfull+zremoved+[zprefix])
        actual=torch.autograd.grad(total,params,allow_unused=True,retain_graph=True)
        expected=torch.autograd.grad(oracle,params,allow_unused=True)
        assert all((a is None and b is None) or (a is not None and b is not None and torch.allclose(a,b,atol=1e-12,rtol=0)) for a,b in zip(actual,expected))
        assert actual[-1] is None and actual[5].abs().max()>0
        assert all(actual[k] is None for k in range(1,5) if k!=slot)
        assert abs(ce-sum(.5*float(torch.nn.functional.cross_entropy(zfull[k],y)) for k in (0,slot)))<1e-12
# Unequal concept lengths, finite differences, inactive hinges, and exact boundary.
y3=torch.tensor([0,1,2]);zf=torch.tensor([[.2,.5,-.1],[.1,.3,.6],[.4,-.2,.7]],dtype=torch.float64,requires_grad=True)
zr=(zf.detach()+torch.tensor([[.4,0.,0.],[0.,.4,0.],[0.,0.,.4]])).requires_grad_()
ix=[[0,1],[2]]
v,mm=source_loss.evidence_components(zf,y3,zr,y3,ix)
lf=torch.log_softmax(zf,-1).gather(1,y3[:,None]).flatten();lr=torch.log_softmax(zr,-1).gather(1,y3[:,None]).flatten()
assert torch.equal(mm,torch.stack(((lf[:2]-lr[:2]).mean(),lf[2]-lr[2])))
assert torch.allclose(v,torch.relu(source_loss.EVIDENCE_MARGIN-mm).mean(),atol=1e-12)
gf,gr=torch.autograd.grad(v,(zf,zr));assert gf.abs().max()>0 and gr.abs().max()>0
for side,grad in ((0,gf),(1,gr)):
 plus=[zf.detach().clone(),zr.detach().clone()];minus=[x.clone() for x in plus]
 plus[side][0,0]+=1e-6;minus[side][0,0]-=1e-6
 fd=(source_loss.evidence_components(plus[0],y3,plus[1],y3,ix)[0]-source_loss.evidence_components(minus[0],y3,minus[1],y3,ix)[0])/2e-6
 assert abs(float(fd-grad[0,0]))<1e-8
with patch.object(source_loss,'EVIDENCE_MARGIN',0.):
 same=zf.detach().clone().requires_grad_();val,_=source_loss.evidence_components(same,y3,same,y3,ix)
 assert float(val)==0 and not torch.autograd.grad(val,same)[0].any()
# A target-dominant full view makes both hinges inactive.
hi=torch.eye(3,dtype=torch.float64).mul(20).requires_grad_();lo=torch.zeros_like(hi,requires_grad=True)
val,_=source_loss.evidence_components(hi,y3,lo,y3,ix)
assert float(val)==0 and all(not g.any() for g in torch.autograd.grad(val,(hi,lo)))
for bad_ix in ([[0],[0]],[[0],[]],[[0],[3]]):
 try:source_loss.evidence_components(zf,y3,zr,y3,bad_ix)
 except AssertionError:pass
 else:raise AssertionError('Invalid concept admitted')
try:source_loss.evidence_components(zf,y3,zr,y3.flip(0),ix)
except AssertionError:pass
else:raise AssertionError('Mismatched answer labels admitted')
# Production candidate calls the same HE loss for gradient and Armijo, including rollback on failure.
weight=torch.tensor([[.4,.2,-.2],[.3,-.1,.8]],dtype=torch.float16,requires_grad=True)
native=weight.detach().clone();master=native.float();initial_memory=Memory(weight,'J').state()
real_backward=worker.backward;real_loss=worker.loss
worker.backward=lambda value,w:torch.autograd.grad(value,w)[0].float()
worker.b.call=lambda *a:None;worker.b.counts=lambda:None
model=SimpleNamespace(_projection_cache=[anchor]*5,_evidence_indices=[[[0],[1]] for _ in range(5)])
def wp(_,k):return weight.double(),y,zprefix
def wr(_,k):return weight.double()*1.3,y
with patch.object(source_loss,'position_logits',side_effect=wp),patch.object(source_loss.b,'logits',side_effect=wr):
    mem=Memory.restore(copy.deepcopy(initial_memory))
    out,event=worker.attempt(model,weight,master.clone(),mem,full,removed,anchors,1,False)
    assert event['accepted'] and event['accepted_loss']==float(source_loss.loss(model,full,removed,anchors,1)[0])
    with torch.no_grad():weight.copy_(native)
    before_counts=worker.b.COUNTS['committed_updates']
    mem=Memory.restore(copy.deepcopy(initial_memory))
    # A candidate loss exception must restore native and leave master/statistics untouched.
    calls=0
    def broken(*args):
        global calls
        calls+=1
        if calls==2:raise RuntimeError('candidate fixture')
        return real_loss(*args)
    with patch.object(worker,'loss',side_effect=broken):
        try:worker.attempt(model,weight,master.clone(),mem,full,removed,anchors,1,False)
        except RuntimeError as exc:assert str(exc)=='candidate fixture'
        else:raise AssertionError('Candidate error swallowed')
    assert torch.equal(weight,native) and exact(mem.state(),initial_memory)
    assert worker.b.COUNTS['committed_updates']==before_counts
# Concept scalar details preserve the original subtraction-before-mean gate arithmetic.
indices=[[[0],[1]] for _ in range(5)]
with patch.object(worker.b,'logits',side_effect=lambda _,k:(zfull[k] if k<5 else zremoved[k-5],y)),torch.no_grad():
    details=medical.contrast_details(worker.b,None,full,removed,indices)
    margins=medical.contrasts(worker.b,None,full,removed,indices)
    assert [v['margin'] for v in details['values']]==[float(v) for v in margins]
    assert len(details['values'])==10 and all(v['finite'] for v in details['values'])
# Gradient diagnostics use ten counted backwards and preserve Python/NumPy/Torch RNG and modes.
net=torch.nn.Sequential(torch.nn.Linear(2,2),torch.nn.Dropout(.5));net[0].eval()
model.model=net;modes=[m.training for m in net.modules()]
state=worker.c.rng();numpy_state=np.random.get_state();count=worker.b.COUNTS['backwards']
def random_wp(*args):
    torch.rand(1);np.random.rand();random.random();net.eval()
    return wp(*args)
with patch.object(source_loss,'position_logits',side_effect=random_wp),patch.object(source_loss.b,'logits',side_effect=wr):
    diag=source_loss.gradient_diagnostics(model,weight,full,removed,anchors,real_backward)
assert diag['components']==['odds_hinge','evidence_hinge'] and worker.b.COUNTS['backwards']==count+10
assert exact(state,worker.c.rng()) and np.array_equal(numpy_state[1],np.random.get_state()[1])
assert modes==[m.training for m in net.modules()]

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
    if 'evidence_loss' in event:event['scope_loss']=event.pop('evidence_loss')
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
