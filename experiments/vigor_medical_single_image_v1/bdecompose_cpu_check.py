"""Synthetic decomposition identities and real weight-guard restoration checks."""
from types import SimpleNamespace
import torch
from .bdecompose_ops import decomposition,weight_guard
from .editor import editable_fp32


def run():
    # Distinct finite response, rounding and execution terms, including an alignment gap.
    d=decomposition(2.,2.2,2.3,2.01,2.32,-.1,.03,pold=-.12)
    assert abs(d['R']-.3)<1e-12 and abs(d['Q']-.1)<1e-12
    assert abs(d['Q_rem']-.07)<1e-12 and abs(d['E']-.01)<1e-12
    assert abs(d['D']-.02)<1e-12 and abs(d['closure_error'])<1e-12 and abs(d['historical_gap_closure_error'])<1e-12
    layer=torch.nn.Linear(2,1,bias=False).half();model=torch.nn.Sequential(layer).eval()
    native=layer.weight.detach().clone();x=torch.tensor([[1.,2.]],dtype=torch.float16)
    flags=[p.requires_grad for p in model.parameters()];rows=[]
    with editable_fp32(model,layer,readonly=True):
        expected=native.float()+.00001
        with torch.no_grad():layer.weight.copy_(expected)
        with weight_guard(layer,expected,'F_u',rows,lambda:None):out=model(x)
        assert out.dtype==torch.float16 and rows[-1]['passed'] and rows[-1]['input_dtype_is_FP32']
        wrong=expected+1
        try:
            with weight_guard(layer,wrong,'bad',rows,lambda:None):model(x)
        except ValueError:pass
        else:raise AssertionError('Guard must stop before computing a mismatched weight')
        with torch.no_grad():layer.weight.copy_(native.float())
    assert torch.equal(layer.weight,native) and layer.weight.dtype==torch.float16
    assert not layer._forward_pre_hooks and not layer._forward_hooks and flags==[p.requires_grad for p in model.parameters()]
    try:
        with editable_fp32(model,layer,readonly=True):
            with torch.no_grad():layer.weight.add_(1)
            raise ValueError('frozen stop')
    except ValueError:pass
    assert torch.equal(layer.weight,native) and not layer._forward_pre_hooks and not layer._forward_hooks
    return dict(status='PASS',two_decomposition_identities=True,alignment_gap_retained=True,rounding_linear_remainder=True,actual_forward_weight_guard=True,guard_failure_stops=True,readonly_and_exception_restore=True)
