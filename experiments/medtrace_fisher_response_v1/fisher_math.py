"""Unbiased categorical Fisher probes at fixed teacher-forced Base prefixes."""
import itertools
import torch


def categorical_probe(probability,signs):
    value=probability.sqrt()*signs
    return value-probability*value.sum(-1,keepdim=True)


def selfcheck():
    p=torch.tensor([.1,.2,.3,.4],dtype=torch.float64)
    signs=torch.tensor(list(itertools.product((-1.,1.),repeat=4)),dtype=torch.float64)
    probes=categorical_probe(p,signs)
    covariance=probes.T@probes/len(probes)
    expected=torch.diag(p)-p[:,None]*p[None,:]
    assert torch.allclose(covariance,expected,atol=1e-14)
    assert probes.sum(-1).abs().max()<1e-14
    jacobian=torch.tensor([[1.,2.],[-1.,1.],[.5,0.],[2.,-1.]],dtype=torch.float64)
    projected=probes@jacobian
    assert torch.allclose(projected.T@projected/len(probes),jacobian.T@expected@jacobian,atol=1e-14)
    return dict(status='PASS',exact_all_sign_covariance=True,parameter_pullback=True)


if __name__=='__main__':print(selfcheck())
