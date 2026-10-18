"""No model calls: shifted CE accounting and relative-hinge shift invariance."""
import math
import torch

def token_metrics(logits,labels):
    mask=labels[:,1:]!=-100
    selected=logits[:,:-1][mask].float();targets=labels[:,1:][mask]
    nll=torch.nn.functional.cross_entropy(selected,targets,reduction='none')
    assert len(nll)>0 and torch.isfinite(nll).all()
    return nll,targets

def selfcheck():
    for s0 in (-2.,-.45938539505004883,0.,2.):
        for delta in (-.25,0.,.25):
            scores=[];grads=[]
            for shift in (0.,10.):
                s=torch.tensor(s0+delta+shift,dtype=torch.float64,requires_grad=True)
                loss=torch.relu(torch.tensor(s0+shift,dtype=s.dtype)-s);loss.backward()
                scores.append(float(loss));grads.append(float(s.grad))
            assert abs(scores[0]-scores[1])<1e-14 and grads[0]==grads[1]
    logits=torch.randn(1,7,13,generator=torch.Generator().manual_seed(18))
    labels=torch.tensor([[-100,-100,-100,2,4,7,-100]])
    nll,targets=token_metrics(logits,labels);assert targets.tolist()==[2,4,7]
    expected=torch.nn.functional.cross_entropy(logits[:,:-1].reshape(-1,13),labels[:,1:].reshape(-1))
    torch.testing.assert_close(nll.mean(),expected,rtol=1e-6,atol=1e-7)
    assert len(nll)==3 and abs(float(nll.sum())-float(nll.mean())*3)<1e-6
    return dict(status='PASS',negative_zero_positive_reference_supported=True,relative_hinge_offset_invariant=True,causal_shift_and_ignore_index=True,mean_sum_from_same_forward=True,model_calls=0)
if __name__=='__main__':print(selfcheck())
