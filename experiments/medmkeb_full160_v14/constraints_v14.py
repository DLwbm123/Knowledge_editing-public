"""Absolute Base-anchored remaining budgets; never accumulate per-step slack."""
import torch

def build(gradients,current,anchor,epsilon=1e-4):
    assert gradients.ndim==2 and gradients.shape[0]==54 and current.shape==anchor.shape==(54,)
    assert all(torch.isfinite(x).all() for x in (gradients,current,anchor))
    matrix=[];bounds=[]
    for j in range(0,54,2):
        matrix.extend([gradients[j],gradients[j]-gradients[j+1]])
        bounds.extend([anchor[j]+epsilon-current[j],(current[j+1]-current[j])-(anchor[j+1]-anchor[j])+epsilon])
    return torch.stack(matrix),torch.stack(bounds)

def selfcheck():
    anchor=torch.tensor([1.,2.]*27,dtype=torch.float64);current=anchor.clone();current[0]+=5e-5;gradient=torch.arange(108,dtype=torch.float64).reshape(54,2)
    a,b=build(gradient,current,anchor);assert torch.allclose(b[:2],torch.tensor([5e-5,5e-5],dtype=b.dtype),atol=1e-12) and torch.equal(a[0],gradient[0]) and torch.equal(a[1],gradient[0]-gradient[1])
    current[0]=anchor[0]+1.1e-4;_,b=build(gradient,current,anchor);assert float(b[0])<0 and float(b[1])<0
    return dict(status='PASS',remaining_slack_not_reset=True,margin_sign=True,over_budget_not_clipped=True,new_model_calls=0)
if __name__=='__main__':print(selfcheck())
