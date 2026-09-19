"""Stage24: original E2 update plus five-observation differentiable regularizer."""
import sys,time,random
from pathlib import Path
from dataclasses import replace
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from methods.medtrace import MedTraceLayerHook
from methods.medtrace.selective_write import optimizer_for,full_vocab_kl
from scripts.medtrace.stage18_cfact import (validate_task,state_hash,source_batch,extra_roles,extra_schedule,resume,rng_state,backward_term,BRANCHES)
from scripts.medtrace.stage17_prepare import digest
from scripts.medtrace.run_selective_write import save
from scripts.medtrace import stage15
LAYER='model.layers.30.mlp.down_proj'


def update(runtime,hook,expert,optimizer,native,fit,teacher,extra=None,extra_weight=1.,U_weight=.01,diagnostic=False,regularize=False,batches=None):
    optimizer.zero_grad(set_to_none=True);terms={};vectors={};params=list(expert.parameters())
    def term(name,loss,weight,tokens,sample):
        before=[p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params] if diagnostic else None
        terms[name]=backward_term(expert,loss,weight,tokens,sample)
        if diagnostic:vectors[name]=torch.cat([(p.grad-before[i]).detach().flatten() for i,p in enumerate(params)])
    for name,batch,weight in (('native',native,.5),('fit',fit,.5)):
        hook.set_teacher_routing(batch.labels);term(name,runtime.compute_loss(batch),weight,len(batch.target_token_ids),name)
    kwargs,labels,mask,logp,row=teacher;hook.set_teacher_routing(labels)
    term('U',full_vocab_kl(runtime.model(**kwargs).logits[mask],logp),U_weight,int(mask.sum()),digest(row))
    if extra is not None:
        batch,row=extra;hook.set_teacher_routing(batch.labels);term('extra',runtime.compute_loss(batch),extra_weight,len(batch.target_token_ids),digest(row))
    cosine=None
    if diagnostic and 'extra' in vectors:
        def cos(a,b):
            denominator=a.norm()*b.norm()
            return float((a@b)/denominator) if float(denominator)>0 else None
        cosine=dict(H_U=cos(vectors['extra'],vectors['U']),H_edit=cos(vectors['extra'],vectors['native']+vectors['fit']),definition='cosine of actual weighted incremental gradients before clipping',zero_norm='NA')
    regularization=None
    if regularize:
        from regularizer import capture,loss,kernel_diagnostics
        features,masks=capture(runtime,hook,batches);reg,dx,dy=loss(features)
        regularization=dict(D_x=float(dx.detach()),D_y=float(dy.detach()),loss=float(reg.detach()))
        if diagnostic:
            def vector(v):
                gs=torch.autograd.grad(v,params,retain_graph=True,allow_unused=True)
                return torch.cat([(g if g is not None else torch.zeros_like(p)).float().flatten() for p,g in zip(params,gs)])
            gx,gy=vector(.001*dx),vector(-.001*dy);g=vector(reg)
            def reg_cosine(a,b):
                den=a.norm()*b.norm()
                return float(a@b/den) if float(den)>1e-20 else None
            regularization.update(x_gradient_norm=float(gx.norm()),y_gradient_norm=float(gy.norm()),combined_gradient_norm=float(g.norm()),cosines={name:reg_cosine(g,v.float()) for name,v in vectors.items()},kernels=kernel_diagnostics(features),masks=masks)
            assert torch.isfinite(gx).all() and torch.isfinite(gy).all()
        before_reg=[p.grad.detach().clone() for p in params] if diagnostic else None
        reg.backward()
        if diagnostic:
            actual=torch.cat([(p.grad-before_reg[i]).float().flatten() for i,p in enumerate(params)])
            regularization['accumulated_gradient_increment_norm']=float(actual.norm())
            regularization['accumulated_gradient_changed_elements']=int((actual!=0).sum())
        del reg,dx,dy,features
    grad=torch.nn.utils.clip_grad_norm_(expert.parameters(),1.)
    if not torch.isfinite(grad) or any(p.grad is None for p in params):raise FloatingPointError('Invalid writer gradient')
    optimizer.step();expert.normalize_factors_(verify_dense=False)
    if any(not torch.isfinite(p).all() for p in params):raise FloatingPointError('Invalid writer state')
    return dict(terms=terms,gradient_norm=float(grad),gradient_cosines=cosine,regularization=regularization)

