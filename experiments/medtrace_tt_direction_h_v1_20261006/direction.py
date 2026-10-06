"""TT-only update policies. All direction/matching inputs are training data."""
import copy
import time
import torch
import updates
from structures import TT4
from geometry import flat,assign,project,adam_metric,match


class ExpandedTT(TT4):
    def __init__(self,seed,rL,rM,rR):
        torch.nn.Module.__init__(self)
        self.d_in,self.d_out,self.rank,self.epsilon=14336,4096,rM,1e-6
        for name,shape in zip(('G1','G2','G3','G4'),((1,64,rL),(rL,64,rM),(rM,112,rR),(rR,128,1))):
            setattr(self,name,torch.nn.Parameter(torch.zeros(shape)))
    def factors(self):
        return (torch.einsum('ia,ajb->ijb',self.G1[0],self.G2).reshape(4096,self.rank),torch.einsum('aib,bj->aij',self.G3,self.G4[:,:,0]).reshape(self.rank,14336))


def clone(state,seed,device):
    e=ExpandedTT(seed,state['G1'].shape[2],state['G2'].shape[2],state['G4'].shape[0]).to(device);e.load_state_dict(state);return e


def embed(state,kind,seed):
    ranks={'OUTER':(8,4,8),'MIDDLE':(4,8,4)}[kind];e=ExpandedTT(seed,*ranks)
    rng=torch.Generator().manual_seed(seed+104729)
    with torch.no_grad():
        e.G1[:,:,:4].copy_(state['G1']);e.G2[:4,:,:4].copy_(state['G2']);e.G3[:4,:,:4].copy_(state['G3']);e.G4[:4].copy_(state['G4'])
        if kind=='OUTER':
            e.G2[4:].copy_(.01*torch.randn(e.G2[4:].shape,generator=rng));e.G4[4:].copy_(.01*torch.randn(e.G4[4:].shape,generator=rng))
        else:e.G3[4:].copy_(.01*torch.randn(e.G3[4:].shape,generator=rng))
    return e


def linearization(runtime,hook,e,batches,teachers,hbatch):
    from scripts.medtrace.stage18_cfact import full_vocab_kl
    values=[];grads=[]
    with updates.preserve(hook,e):
        for b in batches:
            e.zero_grad(set_to_none=True);hook.set_teacher_routing(b.labels);loss=runtime.compute_loss(b);loss.backward();values.append(float(loss.detach()));grads.append(updates.vector(e));del loss
        for kwargs,labels,mask,logp,row in teachers:
            e.zero_grad(set_to_none=True);hook.set_teacher_routing(labels);loss=full_vocab_kl(runtime.model(**kwargs).logits[mask],logp);loss.backward();values.append(float(loss.detach()));grads.append(updates.vector(e));del loss
        e.zero_grad(set_to_none=True);hook.set_teacher_routing(hbatch.labels);hloss=runtime.compute_loss(hbatch);hloss.backward();h=float(hloss.detach());gh=updates.vector(e);del hloss
    g=torch.stack(grads);updates.finite([g,gh,*values,h]);return g,values,gh,h


def hvalue(runtime,hook,b):
    with updates.preserve(hook),torch.no_grad():hook.set_teacher_routing(b.labels);value=float(runtime.compute_loss(b))
    updates.finite([value]);return value


def step(runtime,hook,e,opt,update,batches,teachers,hbatch,thresholds,activations,kind,target=None):
    before=flat(e);oldopt=copy.deepcopy(opt.state_dict());before_rng=updates.rng();before_hook=updates.hook_state(hook)
    with torch.no_grad():fb=e.residual(activations).detach().clone()
    g,values,gh,hbefore=linearization(runtime,hook,e,batches,teachers,hbatch) if kind=='DIR' else (None,None,None,hvalue(runtime,hook,hbatch))
    item=update();after=flat(e);dA=after-before;newopt=copy.deepcopy(opt.state_dict());candidate_rng=updates.rng();candidate_hook=updates.hook_state(hook)
    updates.finite([dA]);h_candidate=hvalue(runtime,hook,hbatch)
    item['actual_adam']=dict(norm=float(dA.norm()),H_candidate_delta=h_candidate-hbefore,H_dot_dA=float(gh@dA) if gh is not None else None,protection_dot_dA=(g@dA).tolist() if g is not None else None)
    accepted=True;extra=0
    if kind=='DIR':
        metric=adam_metric(e,opt);started=time.perf_counter()
        d,qp=project(dA,g,torch.tensor(thresholds,device=g.device)-torch.tensor(values,device=g.device),metric)
        d=d.to(device=before.device,dtype=before.dtype);qp['seconds']=time.perf_counter()-started;qp['H_dot_projected']=float(gh@d)
        qp['actual_linear_residual']=(g@d+torch.tensor(values,device=g.device)-torch.tensor(thresholds,device=g.device)).tolist()
        item['projection']=qp;trials=[];alpha=None
        for a in (1.,.5,.25,.125):
            assign(e,after if a==1 and torch.equal(d,dA) else before+a*d);v=updates.protection(runtime,hook,batches,teachers);extra+=len(v)
            failed=[i for i,(x,limit) in enumerate(zip(v,thresholds)) if x>limit]
            trials.append(dict(alpha=a,values=v,failed_indices=failed))
            if not failed:alpha=a;break
        accepted=alpha is not None
        item['guard']=dict(accepted=accepted,alpha=alpha,trials=trials,extra_forwards=extra,adam_candidate_steps=1,adam_committed_steps=int(accepted))
    elif kind=='MATCH':
        item['match']=match(e,before,dA,target,activations,fb)
        accepted=target>0
    elif kind!='JOINT':raise ValueError(kind)
    if accepted:opt.load_state_dict(newopt);updates.restore_rng(candidate_rng);updates.restore_hook(hook,candidate_hook)
    else:assign(e,before);opt.load_state_dict(oldopt);updates.restore_rng(before_rng);updates.restore_hook(hook,before_hook)
    ha=hvalue(runtime,hook,hbatch)
    with torch.no_grad():fn=float((e.residual(activations)-fb).norm());pn=float((flat(e)-before).norm())
    item.update(H_before=hbefore,H_after=ha,H_committed_delta=ha-hbefore,H_descent=ha<hbefore-1e-8,accepted=accepted,nonzero=pn>0,function_update_norm=fn,parameter_update_norm=pn,protection_extra_forwards=extra)
    return item
