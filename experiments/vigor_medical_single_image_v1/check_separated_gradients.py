"""Small chain-rule regression: no model loading, optimization or GPU calls."""
import torch
from .dual_solver import Budget, pair_objective, separated_gradients


def main():
    c=dict(null_abs_budget=.01,source_normalizer_floor=1e-6,constraint_penalty=100.,
           diversity_min=.2,joint_diversity_penalty=1.,model_gradient_paths_max=144,
           backward_API_max=140,work_seconds=100,KILL_seconds=100)
    for arm in ('IND2','JOINT2'):
        zs=[torch.tensor([[.03]],requires_grad=True),torch.tensor([[-.03]],requires_grad=True)]
        losses=[[z.square().sum()+1,(2*z).square().sum()+2] for z in zs]
        expected=torch.autograd.grad(pair_objective(losses,zs,[1.,1.],arm,c),zs,retain_graph=True)
        budget=Budget(c)
        _,got=separated_gradients(losses,zs,[1.,1.],arm,c,budget)
        assert all(torch.allclose(a,b,rtol=1e-6) for a,b in zip(expected,got))
        assert budget.backwards==4 and budget.gradient_paths==4
    x=torch.tensor(1.,requires_grad=True)
    loss=x.half().float().square()
    bad=torch.autograd.grad(1e6*loss,x,retain_graph=True)[0]
    unit=torch.autograd.grad(loss,x)[0]
    assert not torch.isfinite(bad) and unit.float()*1e6==2e6
    print('Separate backward: both objectives, coupling, cost and FP16 overflow PASS')


if __name__=='__main__':main()
