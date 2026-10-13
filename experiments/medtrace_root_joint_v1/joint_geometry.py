"""Exact fixed-tail operator coordinates, including FP32 canonicalization error."""
import torch
import module_math as mm


def coordinates(before, candidate):
    state = {k:v.detach().cpu().double().clone() for k,v in candidate.items()}
    previous = {k:v.detach().cpu().double() for k,v in before.items()}
    state['G1'].requires_grad_(True)
    b,a = mm.factors(state)
    b0,a0 = mm.factors(previous)
    inner = ((b.T@b0)*(a@a0.T)).sum()
    pullback = torch.autograd.grad(inner,state['G1'])[0].flatten()
    tail = torch.einsum('ajb,bc,djc->ad',state['G2'],a@a.T,state['G2'])
    values,vectors = torch.linalg.eigh(tail)
    assert values.min() > 1e-10*values.max(), 'Degenerate fixed tail'
    square = (vectors*values.sqrt())@vectors.T
    inverse = (vectors/values.sqrt())@vectors.T
    identity = torch.eye(state['G1'].shape[1],dtype=torch.float64)
    transform, inverse = torch.kron(identity,square.contiguous()),torch.kron(identity,inverse.contiguous())
    center = inverse@pullback
    norm_squared = float(((b0.T@b0)*(a0@a0.T)).sum())
    orthogonal = norm_squared-float(center@center)
    assert orthogonal >= -1e-10*max(1.,norm_squared)
    return transform.detach(),inverse.detach(),center.detach(),max(0.,orthogonal)


def selfcheck():
    rng=torch.Generator().manual_seed(20261013)
    shapes=((1,3,2),(2,4,2),(2,5,2),(2,6,1))
    before={k:torch.randn(s,generator=rng,dtype=torch.float64) for k,s in zip(mm.KEYS,shapes)}
    candidate=mm.canonicalize({k:v+.01*torch.randn(v.shape,generator=rng,dtype=v.dtype) for k,v in before.items()})[0]
    candidate['G2']*=1.001
    transform,inverse,center,orthogonal=coordinates(before,candidate)
    root=candidate['G1'].flatten()
    b,a=mm.factors(candidate);b0,a0=mm.factors(before)
    expected=float(((b@a)-(b0@a0)).square().sum())
    actual=float((transform@root-center).square().sum())+orthogonal
    assert abs(actual-expected)<1e-9*max(1.,expected)
    torch.testing.assert_close(inverse@transform,torch.eye(len(root),dtype=torch.float64))
    return dict(status='PASS', affine_tail_offset=True, exact_operator_metric=True, noncanonical_roundoff_covered=True)
