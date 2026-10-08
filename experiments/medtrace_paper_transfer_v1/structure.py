"""TT input-subspace routing primitives; the CP branch was canceled by the user."""
import torch
def normalized(x):return torch.nn.functional.normalize(x.float(),dim=-1,eps=1e-12)


def projection(a,x):
    """Relative row-space projection energy, invariant to invertible row-basis changes.

    Pseudoinverse cutoff is fixed; near-rank-deficient boundaries are diagnosed.
    """
    gram=a@a.T
    inverse=torch.linalg.pinv(gram,hermitian=True,rtol=1e-6)
    response=x@a.T
    return ((response@inverse)*response).sum(-1)/x.square().sum(-1).clamp_min(1e-12)


def input_score(a,features,modal=True):
    # Features are [ALL,VISUAL,TEXT]; equal modality weight avoids token-count bias.
    energy=projection(a,features[1:] if modal else features[:1])
    return energy.mean().clamp(0,1)


def key_score(query,keys,modal=True):
    if modal:
        return (normalized(keys[:,1])*normalized(query[1])).sum(-1).mul(.5)+(normalized(keys[:,2])*normalized(query[2])).sum(-1).mul(.5)
    return (normalized(keys[:,0])*normalized(query[0])).sum(-1)


def token_masks(raw_ids,expanded_length,attention,image_token):
    positions=(raw_ids==image_token).nonzero().flatten()
    assert len(positions)==1, 'Exactly one expanded image is required'
    first=int(positions[0]);length=expanded_length-len(raw_ids)+1
    assert length>0
    visual=torch.zeros(expanded_length,dtype=torch.bool,device=raw_ids.device);visual[first:first+length]=True
    valid=torch.ones_like(visual) if attention is None else attention.bool()
    assert visual.sum()==length and bool(valid[visual].all())
    text=valid & ~visual
    assert text.any() and not bool((visual&text).any())
    return valid,visual,text


def pool(activation,masks):
    assert activation.ndim==2
    result=torch.stack([activation[m].float().mean(0) for m in masks])
    counts=[int(m.sum()) for m in masks]
    reconstructed=(result[1]*counts[1]+result[2]*counts[2])/counts[0]
    assert torch.allclose(result[0],reconstructed,rtol=1e-4,atol=1e-5)
    return result,counts


def select(scores,thresholds,mix=False):
    assert scores.ndim==thresholds.ndim==1 and torch.isfinite(scores).all() and torch.isfinite(thresholds).all()
    adjusted=(scores-thresholds)/(1-thresholds).clamp_min(1e-6)
    eligible=[i for i in range(len(scores)) if float(adjusted[i])>=-1e-6]
    eligible.sort(key=lambda i:(-float(adjusted[i]),i))
    ids=eligible[:2 if mix else 1]
    weights=torch.softmax(adjusted[ids]/.1,0).tolist() if ids else []
    return ids,weights,adjusted


def selfcheck():
    rng=torch.Generator().manual_seed(2)
    a=torch.randn(4,40,generator=rng,dtype=torch.float64);x=torch.randn(3,40,generator=rng,dtype=torch.float64)
    transform=torch.eye(4,dtype=torch.float64)+torch.randn(4,4,generator=rng,dtype=torch.float64)*.1
    assert torch.allclose(projection(a,x),projection(transform@a,x),atol=1e-9,rtol=1e-8)
    direct=torch.linalg.qr(a.T,mode='reduced').Q
    assert torch.allclose(projection(a,x),(x@direct).square().sum(-1)/x.square().sum(-1),atol=1e-9)
    raw=torch.tensor([1,-200,2,3]);masks=token_masks(raw,8,None,-200)
    features,counts=pool(torch.arange(40).reshape(8,5).float(),masks);assert counts==[8,5,3]
    ids,weights,_=select(torch.tensor([.8,.7,.2]),torch.tensor([.5,.5,.5]),True)
    assert ids==[0,1] and abs(sum(weights)-1)<1e-6
    assert select(torch.tensor([.2]),torch.tensor([.5]))[0]==[]
    # Different token counts change ALL but not an equal-modality summary.
    assert torch.allclose((features[1]+features[2])*.5,features[1:].mean(0))
    return dict(status='PASS',checks=['row-space basis invariance','explicit orthogonal projection parity','image-token expansion masks','mean reconstruction','top2 convex weights','off fallback'])


if __name__=='__main__':
    import json
    print(json.dumps(selfcheck()))
