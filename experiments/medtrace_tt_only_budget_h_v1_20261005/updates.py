"""One Adam candidate, bounded acceptance, and trajectory-neutral diagnostics."""
from contextlib import contextmanager
import copy
import random
import torch

HOOK_FIELDS=('enabled','token_mask','generation_routing','generation_boundary','generation_trace','last_generation_trace','anchor_reference','anchor_loss')

def rng():
    return (torch.get_rng_state(),random.getstate(),torch.cuda.get_rng_state() if torch.cuda.is_available() else None)
def restore_rng(s):
    torch.set_rng_state(s[0].cpu());random.setstate(s[1])
    if s[2] is not None:torch.cuda.set_rng_state(s[2].cpu())
def hook_state(h):return {k:copy.deepcopy(getattr(h,k)) for k in HOOK_FIELDS}
def restore_hook(h,s):
    device=next(h.expert.parameters()).device
    for k,v in s.items():setattr(h,k,v.to(device) if isinstance(v,torch.Tensor) else v)
@contextmanager
def preserve(h,e=None):
    r=rng();hs=hook_state(h);grads=[None if p.grad is None else p.grad.clone() for p in e.parameters()] if e is not None else None
    try:yield
    finally:
        restore_rng(r);restore_hook(h,hs)
        if e is not None:
            for p,g in zip(e.parameters(),grads):p.grad=g

def finite(values):
    if not all(torch.isfinite(torch.as_tensor(x)).all() for x in values):raise FloatingPointError('nonfinite constraint or candidate')

def protection(runtime,hook,batches,teachers):
    from scripts.medtrace.stage18_cfact import full_vocab_kl
    values=[]
    with preserve(hook),torch.no_grad():
        for b in batches:
            hook.set_teacher_routing(b.labels);values.append(float(runtime.compute_loss(b)))
        for kwargs,labels,mask,logp,row in teachers:
            hook.set_teacher_routing(labels);values.append(float(full_vocab_kl(runtime.model(**kwargs).logits[mask],logp)))
    finite(values);return values

def limits(values):
    assert len(values)>=6
    return [v+max(.02,.05*v) if i<5 else v+max(.01,.10*v) for i,v in enumerate(values)]

def guarded_candidate(expert,opt,update,check,thresholds):
    """update must execute exactly one clipped Adam step; checks never optimize."""
    before={k:v.detach().clone() for k,v in expert.state_dict().items()};oldopt=copy.deepcopy(opt.state_dict())
    item=update();candidate={k:v.detach().clone() for k,v in expert.state_dict().items()};newopt=copy.deepcopy(opt.state_dict())
    finite(candidate.values());trials=[];chosen=None
    for alpha in (1.,.5,.25,.125):
        # alpha1 is the actual Adam candidate, avoiding roundoff from a-d+a.
        expert.load_state_dict(candidate if alpha==1 else {k:v+alpha*(candidate[k]-v) for k,v in before.items()})
        values=check();finite(values);assert len(values)==len(thresholds)
        slack=[cap-v for cap,v in zip(thresholds,values)];fail=[i for i,s in enumerate(slack) if s<0]
        trials.append(dict(alpha=alpha,values=values,slack=slack,failed_indices=fail))
        if not fail:chosen=alpha;break
    if chosen is None:expert.load_state_dict(before);opt.load_state_dict(oldopt)
    else:opt.load_state_dict(newopt)
    item['guard']=dict(accepted=chosen is not None,alpha=chosen,trials=trials,extra_forwards=len(trials)*len(thresholds),adam_candidate_steps=1,adam_committed_steps=int(chosen is not None),status='ACCEPTED' if chosen is not None else 'REJECTED')
    return item

def vector(e):return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).detach().flatten().clone() for p in e.parameters()])
def diagnostic(runtime,hook,e,batches,teachers,hbatch,hweight,activations):
    """Additional fixed training-only passes; restore RNG, routing and all grads."""
    from scripts.medtrace.stage18_cfact import full_vocab_kl
    terms={};vectors={}
    with preserve(hook,e):
        jobs=[('native',batches[0],.5),('fit',batches[1],.5),('H',hbatch,hweight)]
        for name,b,w in jobs:
            e.zero_grad(set_to_none=True);hook.set_teacher_routing(b.labels);loss=runtime.compute_loss(b);loss.backward();v=vector(e)
            finite([loss,v]);terms[name]=dict(raw_loss=float(loss.detach()),weighted_loss=w*float(loss.detach()),weight=w,raw_gradient_norm=float(v.norm()),weighted_gradient_norm=float((w*v).norm()))
            vectors[name]=v;del loss
        e.zero_grad(set_to_none=True);kwargs,labels,mask,logp,row=teachers[0];hook.set_teacher_routing(labels)
        loss=full_vocab_kl(runtime.model(**kwargs).logits[mask],logp);loss.backward();v=vector(e);finite([loss,v])
        terms['U']=dict(raw_loss=float(loss.detach()),weighted_loss=.01*float(loss.detach()),weight=.01,raw_gradient_norm=float(v.norm()),weighted_gradient_norm=float((.01*v).norm()));vectors['U']=v
        cos=lambda a,b:float(torch.nn.functional.cosine_similarity(a[None],b[None])) if a.norm()>0 and b.norm()>0 else None
        with torch.no_grad():f=e.residual(activations)
        return dict(terms=terms,H_edit_cosine=cos(vectors['H'],.5*(vectors['native']+vectors['fit'])),H_U_cosine=cos(vectors['H'],vectors['U']),H_protection_cosine=cos(vectors['H'],.5*(vectors['native']+vectors['fit'])+.01*vectors['U']),core_norms={k:float(p.norm()) for k,p in e.named_parameters()},function_norm=float(f.norm()),H_diagnostic_only=hweight==0),f.detach().clone()

def capture_training_activations(runtime,hook,batches,teachers,layer):
    captured=[]
    def take(module,args):
        mask=hook.token_mask
        if mask is not None:captured.append(args[0][mask].detach().float()[:16].clone())
    handle=runtime.get_module(layer).register_forward_pre_hook(take)
    try:protection(runtime,hook,batches,teachers)
    finally:handle.remove()
    assert len(captured)==len(batches)+len(teachers)
    return torch.cat(captured)
