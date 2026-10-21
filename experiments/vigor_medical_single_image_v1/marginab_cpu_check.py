"""Synthetic projected margin, one ratio, and quantized zero-write checks."""
import torch
from .editor import direction
from .marginab_ops import one_match,fisher_cost
from .t0probe_cpu_check import run as previous


def run():
    previous();a=torch.tensor([1.,2.,3.]);b=torch.tensor([1.,2.]);g=torch.tensor([[1.,2.,-1.],[2.,0.,1.]]);g0=torch.ones_like(g)
    tangent,geometry=direction(g,g0,a,b)
    assert tangent is not None and abs(float((tangent*g0).sum()))<1e-6
    original=torch.zeros_like(g,dtype=torch.float16);scale,match=one_match(original,tangent,geometry['E'],.0001,a,b)
    delta=(original.float()-scale*tangent).half().float()-original.float()
    assert match['passed'] and match['ratio_applications']==1 and abs(fisher_cost(delta,a,b)/.0001-1)<=.01
    assert float((g*delta).sum())<0
    try:one_match(torch.ones_like(original),tangent,geometry['E'],1e-30,a,b)
    except ValueError:pass
    else:raise AssertionError('Quantized zero seed must stop, not retry')
    return dict(status='PASS',F0_projection=True,one_ratio_native_cost=True,descent_sign=True,zero_seed_stops=True,previous_token_and_restore=True)
