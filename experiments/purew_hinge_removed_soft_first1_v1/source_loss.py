"""HR: current-window odds hinge plus unit-weight removed full-vocabulary KL."""
import math
from dataclasses import replace
import torch
import torch.nn.functional as F
from runtime import b, c, ARM, R

ALPHA = .95
COEFFICIENTS = (.5,.125,.125,.125,.125)

class Batch(tuple):
    pass


def prefix_mask(prepared, special_ids):
    """Map raw language tokens through the realized one-image expansion."""
    labels=prepared.labels.detach().cpu()
    raw=prepared.raw_input_ids.detach().cpu()
    raw_labels=prepared.raw_labels.detach().cpu()
    attention=prepared.attention_mask
    attention=torch.ones_like(labels,dtype=torch.bool) if attention is None else attention.detach().cpu().bool()
    assert labels.shape[0]==1 and attention.all(), 'Only verified unpadded single-source batches admitted'
    start,end=prepared.image_token_start,prepared.image_token_end
    assert 0<=start<end<=labels.shape[1] and raw[0,start]<0
    assert (raw<0).sum()==1 and raw.shape[1]+end-start-1==labels.shape[1]
    ids=torch.cat((raw[:,:start],torch.full((1,end-start),-200,dtype=raw.dtype),raw[:,start+1:]),1)
    expanded_labels=torch.cat((raw_labels[:,:start],torch.full((1,end-start),-100,dtype=raw_labels.dtype),raw_labels[:,start+1:]),1)
    assert torch.equal(expanded_labels,labels), 'Raw/expanded answer labels differ'
    ordinary=ids>=0
    for token in special_ids:ordinary &= ids!=token
    answer=labels[:,1:]!=-100
    prefix=ordinary[:,:-1]&ordinary[:,1:]&(labels[:,1:]==-100)&attention[:,:-1]&attention[:,1:]
    assert answer.any() and prefix.any() and not (answer&prefix).any()
    assert not prefix[:,max(0,start-1):end].any(), 'Image prediction position in prefix'
    return prefix


def batch_cpu(runtime,task,question=None):
    record=c.record(task)
    if question is not None:record=replace(record,question=question)
    prepared=runtime.build_edit_batch(record)
    kwargs={k:v.detach().cpu() if hasattr(v,'detach') else v for k,v in prepared.forward_kwargs().items()}
    assert kwargs['inputs_embeds'].dtype==torch.float16
    out=Batch((kwargs,prepared.labels.detach().cpu()))
    out.prefix=prefix_mask(prepared,runtime.adapter.tokenizer.all_special_ids)
    return out


def position_logits(model,batch):
    kwargs,labels=batch
    kwargs={k:v.to(model.device) if hasattr(v,'to') else v for k,v in kwargs.items()}
    labels=labels.to(model.device)[:,1:];answer=labels!=-100
    z=model.model(**kwargs).logits[:,:-1].float()
    assert z.shape[:2]==labels.shape and torch.isfinite(z).all()
    return z[answer].double(),labels[answer],z[batch.prefix.to(model.device)].double()


def split_logits(z,y):
    target=z.gather(1,y[:,None]).squeeze(1)
    mask=torch.ones_like(z,dtype=torch.bool).scatter_(1,y[:,None],False)
    other=z[mask].reshape(z.shape[0],z.shape[1]-1)
    rest=torch.logsumexp(other,-1)
    return -F.softplus(rest-target),-F.softplus(target-rest),F.log_softmax(other,-1)


def teacher(z,y,prefix):
    la,lr,lq=split_logits(z,y)
    keep=la>=math.log(ALPHA)
    ls=torch.maximum(la,torch.full_like(la,math.log(ALPHA)))
    lsr=torch.where(keep,lr,torch.full_like(lr,math.log1p(-ALPHA)))
    return dict(log_a=ls.detach().cpu(),log_rest=lsr.detach().cpu(),
                log_q=lq.detach().cpu(),prefix=F.log_softmax(prefix,-1).detach().cpu(),
                labels=y.detach().cpu())


def components(z,y,prefix,anchor):
    assert torch.equal(y.cpu(),anchor['labels'])
    la,lr,lq=split_logits(z,y)
    ls=anchor['log_a'].to(z.device);lsr=anchor['log_rest'].to(z.device)
    lq0=anchor['log_q'].to(z.device);lp0=anchor['prefix'].to(z.device)
    marginal=(ls.exp()*(ls-la)+lsr.exp()*(lsr-lr)).mean()
    q_per=(lq0.exp()*(lq0-lq)).sum(-1)
    conditional=(lsr.exp()*q_per).mean()
    prefix_kl=(lp0.exp()*(lp0-F.log_softmax(prefix,-1))).sum(-1).mean()
    ce=F.cross_entropy(z,y)
    hinge=F.relu((ls-lsr)-(la-lr)).mean()
    values=dict(odds_hinge=hinge,marginal_KL=marginal,conditional_KL=conditional,
                unweighted_conditional_KL=q_per.mean(),prefix_KL=prefix_kl,CE=ce,
                target_gap_mean=(la.exp()-ls.exp()).mean(),
                target_gap_min=(la.exp()-ls.exp()).min(),overshoot_fraction=(la>ls).double().mean(),
                alpha_pass_fraction=(la>=math.log(ALPHA)).double().mean(),
                target_pass_fraction=(la>=ls).double().mean())
    assert all(torch.isfinite(v) for v in values.values()), 'Nonfinite projection loss'
    return values


def prepare(model,full):
    cache=[]
    with torch.no_grad():
        for batch in full:
            z,y,prefix=position_logits(model,batch)
            cache.append(teacher(z,y,prefix))
            # Same answer positions/precision as the original production path.
            old,old_y=b.logits(model,batch)
            assert torch.equal(old_y,y) and torch.equal(old.double(),z)
    model._projection_cache=cache
    report=dict(status='PASS',views=len(full),answer_counts=[len(a['labels']) for a in cache],
                prefix_counts=[len(a['prefix']) for a in cache],answer_prefix_disjoint=True,
                image_and_special_excluded=True,original_answer_logits_exact=True,
                teacher_bytes=sum(t.numel()*t.element_size() for a in cache for t in a.values()))
    c.write(R/'public'/('PROJECTION_MASK_'+ARM+'.json'),report)
    # Persist only this source-window teacher for an exact cooperative checkpoint.
    b.save(R/'private/window'/ARM/'teacher.pt',cache)
    return cache


def removed_kl(model,batch,anchor):
    z,y=b.logits(model,batch)
    assert z.shape==anchor.shape and torch.isfinite(anchor).all()
    return F.kl_div(F.log_softmax(z.double(),-1),anchor.to(z.device),reduction='batchmean'),y


def loss(model,full,removed,anchors,slot):
    assert ARM=='HR' and len(full)==len(removed)==len(anchors)==5
    value=0.;ce=0.;protect=0.
    for k in (0,1+(slot-1)%4):
        z,y,prefix=position_logits(model,full[k])
        parts=components(z,y,prefix,model._projection_cache[k])
        kl,yr=removed_kl(model,removed[k],anchors[k])
        assert torch.equal(y,yr) and torch.isfinite(kl)
        value=value+.5*(parts['odds_hinge']+kl)
        ce+=.5*float(parts['CE'].detach());protect+=.5*float(kl.detach())
    return value,ce,protect


def measurements(model,full):
    allparts=[]
    for k,batch in enumerate(full):
        z,y,prefix=position_logits(model,batch)
        allparts.append({n:float(v.detach()) for n,v in components(z,y,prefix,model._projection_cache[k]).items()})
    out={n:sum(w*v[n] for w,v in zip(COEFFICIENTS,allparts)) for n in allparts[0]}
    out['target_gap_view_min_weighted']=out.pop('target_gap_min')
    out['target_gap_global_min']=min(v['target_gap_min'] for v in allparts)
    out['weighted_all_FIT_CE']=out['CE']
    return out


def gradient_diagnostics(model,weight,full,removed,anchors,backward):
    state=c.rng();nrng=__import__('numpy').random.get_state()
    modes=[(m,m.training) for m in model.model.modules()]
    try:
        gs=[]
        for name in ('odds_hinge','removed_KL'):
            g=torch.zeros(weight.numel(),dtype=torch.float64)
            for k,batch in enumerate(full):
                if name=='odds_hinge':
                    z,y,prefix=position_logits(model,batch)
                    value=components(z,y,prefix,model._projection_cache[k])[name]
                else:value=removed_kl(model,removed[k],anchors[k])[0]
                g+=COEFFICIENTS[k]*backward(value,weight).flatten().double().cpu()
            gs.append(g)
        norms=[float(g.norm()) for g in gs]
        cosine=float(torch.dot(gs[0],gs[1]))/(norms[0]*norms[1]) if all(norms) else None
        return dict(components=['odds_hinge','removed_KL'],norms=norms,cosines={'0-1':cosine})
    finally:
        c.restore_rng(state);__import__('numpy').random.set_state(nrng)
        for m,training in modes:m.training=training


def scope_anchors(model,removed):
    with torch.no_grad():return [b.logits(model,batch)[0].double().softmax(-1).detach() for batch in removed]
