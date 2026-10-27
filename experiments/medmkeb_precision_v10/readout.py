"""Same-logit causal CE readout, all medical content stays private."""
import torch
import torch.nn.functional as F

def read(logits,labels):
    assert logits.ndim==3 and labels.shape==logits.shape[:2]
    mask=labels[:,1:]!=-100;assert bool(mask.any())
    chosen=logits[:,:-1,:][mask];targets=labels[:,1:][mask].long()
    assert bool((targets>=0).all()) and bool((targets<logits.shape[-1]).all())
    original=F.cross_entropy(logits[:,:-1,:].contiguous().float().view(-1,logits.shape[-1]),labels[:,1:].contiguous().view(-1))
    return float(original),float(F.cross_entropy(chosen.double(),targets))

def selfcheck():
    logits=torch.tensor([[[.1,.2,.3],[.3,.1,.2],[.4,.2,.1]]]);labels=torch.tensor([[-100,2,-100]])
    a,b=read(logits,labels);expected=float(F.cross_entropy(logits[0,:1].float(),torch.tensor([2])));assert a==expected and abs(a-b)<1e-6
    try:read(logits,torch.full_like(labels,-100))
    except AssertionError:pass
    else:raise AssertionError('Empty target mask accepted')
    return dict(status='PASS',causal_shift=True,content_mask=True,readout_precision=True,empty_mask_rejected=True,new_language_calls=0)
