"""Check the new estimator and solve independently, and retain the old lifecycle tests."""
from unittest import TestCase
import torch
from .backoff16_cpu_check import run as prior_checks
from .cpu_check import Runtime,record
from .editor import calibrate,direction,editable_fp32,nll
from .diagnose import candidate
from .direct_fisher16 import estimate


def run():
    prior_checks()
    torch.manual_seed(42)
    rt=Runtime(torch.float16);name='model.layers.0.mlp.down_proj'
    records=[record('a',image=1.,target='2'),record('b',image=.2,target='3')]
    old=calibrate(rt,records,[name],train_ids=['a','b'],excluded_ids=['test'],binding={'model':'synthetic'})['factors'][name]
    base=rt.model.layer.weight.detach().clone()
    gradients=[]
    with editable_fp32(rt.model,rt.model.layer,readonly=True):
        for r in records:
            gradients.append(torch.autograd.grad(nll(rt.model,rt.build_edit_batch(r)),rt.model.layer.weight)[0].detach().double())
    expected=torch.stack([g.square() for g in gradients]).mean(0).float()
    expected=(expected+1e-3*expected.mean()).clamp_min(1e-12)
    expected*=float(old['a'].double().sum()*old['b'].double().sum())/float(expected.double().sum())
    f,metadata=estimate(rt,records,old,lambda event:None)
    assert torch.allclose(f,expected,rtol=1e-5,atol=0)
    assert not torch.allclose(torch.stack(gradients).mean(0).square().float(),torch.stack([g.square() for g in gradients]).mean(0).float())
    assert metadata['relative_trace_error']<1e-5 and not metadata['saved_factor']
    assert torch.equal(base,rt.model.layer.weight)
    gv=torch.randn_like(f);g0=torch.randn_like(f)
    tangent,geometry=direction(gv,g0,old['a'],old['b'],diagonal=f)
    delta=-.1/geometry['E']*tangent
    constraint=torch.stack([gv.flatten(),g0.flatten()]).double()
    inv=torch.diag(1/f.flatten().double())
    reference=inv@constraint.T@torch.linalg.solve(constraint@inv@constraint.T,torch.tensor([-.1,0.],dtype=torch.double))
    assert torch.allclose(delta.flatten().double(),reference,rtol=1e-4,atol=1e-6)
    for scale in (1.,100.):
        r=candidate(rt,record('test'),0,old,.02,solve_factor=f*scale)
        assert r['fp32_update']['fisher_cost']<=.001*(1+1e-4)
        assert r['native_update']['fisher_cost']>=0
        assert torch.equal(base,rt.model.layer.weight)
    with TestCase().assertRaises(TimeoutError):estimate(rt,records,old,lambda e:None,deadline=0)
    assert torch.equal(base,rt.model.layer.weight)
    return dict(status='PASS',prior_backoff_checks=True,per_example_square_before_average=True,
        dense_KKT_agreement=True,original_metric_trust_clipping=True,trace_matching=True,
        calibration_readonly=True,deadline_restoration=True,new_configurations=1)
