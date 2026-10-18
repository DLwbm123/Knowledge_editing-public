"""One frozen alignment request, mismatch/gate rollback, snapshot and probe recovery."""
from unittest.mock import patch
from unittest import TestCase
import torch
from .normd8_cpu_check import run as prior_checks
from .cpu_check import Runtime,record
from .editor import calibrate,paired_batches
from .direct_fisher16 import estimate
from .diagnose import candidate,FIRST_D
from .finite_null import native_losses
from .batch8 import baseline_check
from .nllmatch import ALPHA,align,reconstruct,probe_endpoint


def run():
    prior_checks();torch.manual_seed(42)
    rt=Runtime(torch.float16);r=record('test');name='model.layers.0.mlp.down_proj'
    train=[record('a'),record('b',image=.2,target='3')]
    f=calibrate(rt,train,[name],train_ids=['a','b'],excluded_ids=['test'],binding={'model':'synthetic'})['factors'][name]
    diagonal,_=estimate(rt,train,f,lambda e:None)
    pair=paired_batches(rt,r);base=rt.model.layer.weight.detach().clone();base_losses=native_losses(rt,pair)
    baseline=baseline_check(rt,r,f)
    with torch.no_grad():rt.model.layer.weight.add_(.002)
    prefix=rt.model.layer.weight.detach().clone();losses=native_losses(rt,pair)
    with torch.no_grad():rt.model.layer.weight.add_(.01)
    endpoint=rt.model.layer.weight.detach().clone();target=native_losses(rt,pair)[0]
    with torch.no_grad():rt.model.layer.weight.copy_(base)
    for gate_pass,nll_target,expect_success in [(True,target+.1,False),(False,target,False),(True,target,True)]:
        calls=[]
        def attempt(*args,**kw):
            assert args[11] is None and kw['correction_factor'] is None and 'native_norm_target' not in kw
            assert args[7]==ALPHA*.2;calls.append(args[7])
            with torch.no_grad():rt.model.layer.weight.add_(.01)
            v=dict(accepted=gate_pass,native_nll=native_losses(rt,pair))
            args[9].update(final_verdict=v)
            return v
        state=dict(arm='D',baseline=baseline,endpoint_native_nll=losses,probes={})
        with patch('experiments.vigor_medical_single_image_v1.nllmatch._attempt',side_effect=attempt):
            snapshot=align(rt,r,f,prefix,state,nll_target,.2,lambda:None,None)
        assert len(calls)==1 and (snapshot is not None)==expect_success
        assert state['alignment']['accepted']==expect_success and state['alignment_base_restored']
        if expect_success:assert torch.equal(snapshot,endpoint)
        else:assert state['alignment']['rollback_pass'] and state['alignment']['after_rollback_native_nll']==losses
        assert torch.equal(base,rt.model.layer.weight)
    state=dict(arm='A',baseline=baseline,endpoint_native_nll=base_losses,probes={})
    p=candidate(rt,r,0,f,FIRST_D/16,solve_factor=diagonal)
    target_norm=p['quantization']['native_delta_l2']
    from .normpath8 import run_probe
    with patch('experiments.vigor_medical_single_image_v1.nllmatch.run_probe',wraps=run_probe) as probe:
        result=probe_endpoint(rt,r,f,diagonal,base,state,target_norm,lambda:None,None)
        assert probe.call_count==1 and result['matching_passed'] and result['endpoint_restored'] and state['probe_base_restored']
    assert torch.equal(base,rt.model.layer.weight)
    def broken(*args,**kw):
        with torch.no_grad():rt.model.layer.weight.add_(.01)
        raise RuntimeError('synthetic reconstruction failure')
    state=dict(arm='A',checks=[])
    fake=dict(baseline=baseline,steps=[dict(step=1,start_native_nll=base_losses,trial=dict(proposal={}))])
    with TestCase().assertRaises(RuntimeError),patch('experiments.vigor_medical_single_image_v1.nllmatch._attempt',side_effect=broken):
        reconstruct(rt,r,f,fake,1,state,lambda:None,None)
    assert state['base_restored'] and torch.equal(base,rt.model.layer.weight)
    with TestCase().assertRaises(TimeoutError):align(rt,r,f,prefix,{},target,.2,lambda:None,0)
    assert torch.equal(base,rt.model.layer.weight)
    return dict(status='PASS',prior_checks=True,literal_frozen_alpha=True,one_alignment_candidate_only=True,
        mismatch_and_gate_failure_roll_back_stop=True,accepted_snapshot_checked=True,no_norm_rescale_in_alignment=True,
        one_matched_probe_per_endpoint=True,probe_and_base_restore=True,reconstruction_exception_restores=True,
        no_half_or_retry=True,deadline_checked=True)
