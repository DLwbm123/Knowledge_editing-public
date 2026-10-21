"""Isolate source/null gradient overlap before the fixed rank-four restriction."""
import torch
from .brc_pair import solve as constrained_solve
from .dual_solver import source_basis
from .editor import editable_fp32, finite, nll


def residual_gradient(source, null, relative_min):
    # Double reductions avoid cancellation; the model path stays FP32/FP16.
    square = null.double().square().sum()
    if float(square) == 0:
        raise ValueError('Null gradient unavailable for contrast')
    alpha = (source.double()*null.double()).sum()/square
    residual = finite(source-alpha.float()*null)
    relative = float(residual.double().norm()/source.double().norm())
    if not torch.isfinite(torch.tensor(relative)) or relative <= relative_min:
        raise ValueError('No identifiable source-specific gradient')
    return residual, relative


def build_bases(model,layer,pair,gradient,a,b,config,budget,progress):
    control=source_basis(gradient,a,b,config)
    with editable_fp32(model,layer,readonly=True):
        null=finite(budget.grad(nll(model,pair[1]),(layer.weight,),1)[0].detach())
    residual,relative=residual_gradient(gradient,null,config['residual_relative_min'])
    progress(dict(status='FULL_GRADIENT_RESIDUAL_RATIO',objective=relative,forwards=budget.forwards,backward_API=budget.backwards,model_gradient_paths=budget.gradient_paths))
    contrast=source_basis(residual,a,b,config)
    return dict(SOURCE4=control,CONTRAST4=contrast)


def solve(runtime,layer,pair,factor,config,admission,progress,*,budget):
    if config['version']!='PUREW_CONTRAST_BASIS_V1':
        raise PermissionError('Frozen basis contrast required')
    return constrained_solve(runtime,layer,pair,factor,config,admission,progress,budget=budget,basis_builder=build_bases)
