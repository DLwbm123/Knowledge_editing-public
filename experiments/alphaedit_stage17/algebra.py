"""Official AlphaEdit solve with a compact cumulative post-edit key cache."""
import torch


def update(key, residual, basis, history, l2=10):
    if history.ndim!=2 or history.shape[0]!=key.numel() or l2<=0:
        raise ValueError('invalid history or regularization')
    projected_key=basis@(basis.T@key)
    if history.shape[1]==0:
        solution=projected_key/(l2+key@projected_key)
    else:
        keys=torch.cat((key[:,None],history),dim=1)
        projected=basis@(basis.T@keys)
        gram=keys.T@projected
        coefficients=torch.linalg.solve(gram+l2*torch.eye(gram.shape[0],device=key.device,dtype=key.dtype),
                                        keys.T@projected_key)
        solution=(projected_key-projected@coefficients)/l2
    result=torch.outer(residual,solution)
    if not torch.isfinite(result).all():raise FloatingPointError('nonfinite AlphaEdit solve')
    return result
