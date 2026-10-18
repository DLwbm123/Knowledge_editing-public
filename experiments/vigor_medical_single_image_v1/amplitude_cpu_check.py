"""Minimal checks for compensation placement, invalid ratio, and native rollback."""
import copy
from unittest.mock import patch
import torch
from .precision_abc_cpu_check import run as prior_checks
from .cpu_check import Runtime,record
from .batch8 import baseline_check
from .editor import calibrate,paired_batches,editable_fp32,gradient_pair,direction
from .finite_null import trajectory,correction,native_losses,update_metrics
from .precision_abc import reconstruct
from .amplitude_compensation import prepare,evaluate


def run():
    prior_checks()
    torch.manual_seed(42);rt=Runtime(torch.float16);r=record('trial');name='model.layers.0.mlp.down_proj'
    f=calibrate(rt,[record('train')],[name],train_ids=['train'],excluded_ids=['trial'],binding={'model':'synthetic'})['factors'][name]
    f=dict(f,a=f['a']*.001);layer=rt.model.layer;base=layer.weight.detach().clone()
    baseline=baseline_check(rt,r,f);old=trajectory(rt,r,f,lambda rows:None)
    target=next(s['step'] for s in old['steps'] if s['correction'])
    prefix=copy.deepcopy(old);prefix['steps']=prefix['steps'][:target];prefix['steps'][-1]['status']='REJECTED_AND_STOPPED'
    history=dict(baseline=baseline,method=prefix);state=dict(checks=[],prefix=[])
    snap=reconstruct(rt,r,f,history,target,state,lambda:None)
    prepare(rt,r,f,snap,history,state,lambda:None)
    pair=paired_batches(rt,r)
    with torch.no_grad():layer.weight.copy_(snap['weight'])
    residual=native_losses(rt,pair)[1]-baseline['native'][1]
    with editable_fp32(rt.model,layer,readonly=True):
        _,gs=gradient_pair(rt,layer,pair);t,g=direction(gs[1],gs[0],f['a'],f['b'])
        d=-residual/g['E']*t
        rounded=(layer.weight.detach()+d).half().float()-snap['weight'].float()
        rho=update_metrics(rounded,*gs,f['a'],f['b'])['null_linear_change']/(-residual)
        expected=(snap['weight'].float()+d/rho).half()
    c=correction(rt,r,f,baseline['native'][1],compensate=True)
    assert torch.equal(expected,layer.weight) and layer.weight.dtype==torch.float16
    assert c['compensation']['rho']==rho
    with torch.no_grad():layer.weight.copy_(snap['weight'])
    calls=[]
    def invalid(*args):
        m=update_metrics(*args);calls.append(1)
        if len(calls)==2:m['null_linear_change']=residual
        return m
    with patch('experiments.vigor_medical_single_image_v1.finite_null.update_metrics',side_effect=invalid):
        bad=correction(rt,r,f,baseline['native'][1],compensate=True)
    assert bad['status']=='INVALID_COMPENSATION' and torch.equal(layer.weight,snap['weight'])
    with torch.no_grad():layer.weight.copy_(base)
    evaluate(rt,r,f,snap,history,state,lambda:None)
    assert state['candidate_count']==1 and state['base_restored'] and torch.equal(layer.weight,base)
    assert state['candidate']['verdict']['combined_native_fisher_cost']>=0
    # Rejection handling is exercised deterministically without an extra scientific candidate.
    with patch('experiments.vigor_medical_single_image_v1.amplitude_compensation.correction',return_value=bad) as m:
        evaluate(rt,r,f,snap,history,state,lambda:None)
        assert m.call_count==1 and not state['passed'] and state['candidate']['rejected_step_rollback']
    return dict(status='PASS',prior_reconstruction_checks=True,one_candidate_from_same_start=True,
        native_fp16=True,invalid_ratio_no_write=True,original_gates=True,rejection_rollback=True,
        base_restoration=True,generation_calls=0,judge_calls=0)
