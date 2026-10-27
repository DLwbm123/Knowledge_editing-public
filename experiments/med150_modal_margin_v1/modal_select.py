"""Fixed radius-respecting MED150 modal reranking; no answers or labels."""
import torch


def select(activated, nearest, distances, radii, scores, margin=0.02):
    assert distances.ndim==radii.ndim==scores.ndim==1 and distances.shape==radii.shape==scores.shape
    assert torch.isfinite(distances).all() and torch.isfinite(radii).all() and torch.isfinite(scores).all()
    assert bool((radii>=0).all()) and margin==0.02
    if not activated:return None
    assert isinstance(nearest,int) and 0<=nearest<len(scores)
    candidates=[i for i in range(len(scores)) if float(distances[i])<=float(radii[i])]
    assert nearest in candidates
    best=max(candidates,key=lambda i:(float(scores[i]),-i))
    return best if float(scores[best])-float(scores[nearest])>=margin else nearest


def selfcheck():
    d=torch.tensor([1.,1.,2.]);r=torch.tensor([1.,1.,1.]);s=torch.tensor([0.,.03,.9])
    assert select(False,0,d,r,s) is None
    assert select(True,0,d,r,s)==1 # high score outside radius cannot win
    assert select(True,0,d,r,torch.tensor([0.,.019,.9]))==0
    assert select(True,0,d,r,torch.tensor([0.,.02,.9],dtype=torch.float64))==1
    assert select(True,0,d,r,torch.tensor([.1,.1,.9]))==0
    try:select(True,2,d,r,s)
    except AssertionError:pass
    else:raise AssertionError('R0 nearest outside candidates')
    return dict(status='PASS',radius_constraint=True,off_preserved=True,tie_original_order=True,threshold_fixed=0.02)

if __name__=='__main__':print(selfcheck())
