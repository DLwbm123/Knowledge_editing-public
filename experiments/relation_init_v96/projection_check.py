"""CPU-only check of the proposed row-space exclusion at a zero TT root."""
import json
import torch


def check():
    torch.set_num_threads(1)
    gen = torch.Generator().manual_seed(960024)
    source = torch.randn(64,256,generator=gen,dtype=torch.float64)
    protected = torch.randn(2,64,256,generator=gen,dtype=torch.float64)
    _, s, vh = torch.linalg.svd(protected.reshape(-1,256), full_matrices=False)
    q = vh[s>s[0]*1e-10].T
    projected = source-(source@q)@q.T
    _,_,tails = torch.linalg.svd(projected,full_matrices=False)
    tail = tails[:8].clone().requires_grad_()
    root = torch.zeros(64,8,dtype=torch.float64,requires_grad=True)
    opt = torch.optim.Adam([root,tail],lr=1e-3,eps=1e-8)
    initial = root@tail
    initial_loss = (initial*source).sum()
    initial_loss.backward()
    assert tail.grad.count_nonzero()==0
    opt.step()
    first = (root@tail).detach()
    dots1 = torch.einsum('kij,ij->k',protected,first)
    assert dots1.abs().max()<1e-10
    # An actual second factor update need not preserve the initial row-space.
    opt.zero_grad(set_to_none=True)
    ((root@tail)*source).sum().backward();opt.step()
    second=(root@tail).detach()
    dots2=torch.einsum('kij,ij->k',protected,second)
    assert dots2.abs().max()>1e-10
    # Orthogonality to a gradient does not remove curvature.
    c0=protected[0];direction=first
    eps=1e-4
    lp=lambda x:(c0*x).sum()+.5*x.square().sum()
    derivative=(lp(eps*direction)-lp(-eps*direction))/(2*eps)
    assert abs(float(derivative))<1e-10
    assert float(lp(direction))>0
    # Opposed facts would disappear if averaged before building a row-space.
    opposite=torch.stack([protected[0],-protected[0]])
    assert opposite.mean(0).count_nonzero()==0 and opposite.norm()>0
    return dict(status='PASS',device='CPU',model_calls=0,source_matrix_shape=[64,256],
        protected_rank=int(q.shape[1]),projection_dot_max=float(torch.einsum('kij,ij->k',protected,projected).abs().max()),
        first_step_protection_derivative_max=float(dots1.abs().max()),
        second_step_nonzero_protection_derivative_max=float(dots2.abs().max()),
        finite_difference=float(derivative),nonlinear_protection_loss_increase=float(lp(direction)),
        conclusion='Initial first-order exclusion only; neither later-factor updates nor nonlinear protection are guaranteed.')

if __name__=='__main__': print(json.dumps(check(),indent=2))
