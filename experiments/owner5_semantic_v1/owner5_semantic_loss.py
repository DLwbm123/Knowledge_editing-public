"""Frozen semantic pairs and exact piecewise-linear protection gradient."""
import math
ANSWERS=(
    'No pleural effusion is present', 'Pleural effusion is present',
    'There is no evidence of pleural effusion', 'There is evidence of pleural effusion',
    'The image does not show pleural effusion', 'The image shows pleural effusion',
)
GRADIENT_WEIGHTS=(1/3,-1/3,1/3,-1/3,1/3,-1/3)

def summary(losses):
    assert len(losses)==6 and all(math.isfinite(v) for v in losses)
    margins=[losses[i+1]-losses[i] for i in (0,2,4)]
    return dict(score=sum(margins)/3,pair_margins=margins,candidate_mean_nll=losses)

def selfcheck():
    import torch
    for offset in (-1.,0.,1.):
        x=torch.tensor([1.,2.,2.,4.,3.,6.],dtype=torch.float64,requires_grad=True)
        s=sum(x[i+1]-x[i] for i in (0,2,4))/3
        s0=float(s.detach())+offset
        torch.relu(s0-s).backward()
        expected=torch.tensor(GRADIENT_WEIGHTS,dtype=x.dtype) if offset>0 else torch.zeros_like(x)
        assert torch.equal(x.grad,expected)
        assert summary(x.detach().tolist())['score']==float(s.detach())
    return dict(status='PASS',active_hinge_gradient_exact=True,inactive_and_boundary_zero=True)

if __name__=='__main__':print(selfcheck())
