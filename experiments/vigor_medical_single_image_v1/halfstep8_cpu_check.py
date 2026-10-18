"""Same-start half-step, no-correction target, and rejected candidate restoration."""
import copy
from unittest import TestCase
from unittest.mock import patch
import torch
from .secant_stress16_cpu_check import run as prior_checks
from .cpu_check import Runtime,record
from .editor import calibrate
from .batch8 import run_sample
from .diagnose import candidate,FIRST_D
from .precision_abc import reconstruct
from .halfstep8 import evaluate,prepare_control


def run():
    prior_checks()
    torch.manual_seed(42);rt=Runtime(torch.float16);r=record('trial');name='model.layers.0.mlp.down_proj'
    f=calibrate(rt,[record('train')],[name],train_ids=['train'],excluded_ids=['trial'],binding={'model':'synthetic'})['factors'][name]
    f=dict(f,a=f['a']*.001);base=rt.model.layer.weight.detach().clone()
    old=dict(sample='S01');run_sample(rt,r,f,old,lambda s:None,compensate='secant',steps=16)
    old=copy.deepcopy(old);old['method']['steps']=old['method']['steps'][:1]
    first=old['method']['steps'][0]
    # Synthetic no-correction rejection exercises the real S04 plumbing without a new sample.
    first['correction']=None
    first['final_verdict']=copy.deepcopy(first['proposal_verdict'])
    first['status']='REJECTED_AND_STOPPED';first['final_verdict']['accepted']=False
    first['final_verdict']['gates']['visual']=False
    state=dict(sample='S01',checks=[],prefix=[])
    snapshot=reconstruct(rt,r,f,old,1,state,lambda:None,compensate='secant',require_target_correction=False)
    assert torch.equal(snapshot['start_weight'],base.cpu()) and state['reconstruction_base_restored']
    with patch('experiments.vigor_medical_single_image_v1.halfstep8.gates',return_value=first['final_verdict']),patch('experiments.vigor_medical_single_image_v1.halfstep8.correction') as corr:
        prepare_control(rt,r,f,snapshot,old,state,lambda:None)
        assert corr.call_count==0 and state['control_pass'] and state['control_base_restored']
    with patch('experiments.vigor_medical_single_image_v1.halfstep8.candidate',wraps=candidate) as main:
        evaluate(rt,r,f,snapshot,old,state,lambda:None)
    assert main.call_count==1 and main.call_args.args[4]==FIRST_D/32
    assert state['candidate']['proposal']['native_before']==snapshot['start_losses']
    assert state['candidate_count']==1 and state['new_correction_count']<=1
    assert state['base_restored'] and torch.equal(base,rt.model.layer.weight)
    rejected=dict(native_nll=snapshot['start_losses'],cumulative_null_change=.02,
        combined_native_fisher_cost=0,gates=dict(null=False,visual=False,trust=True),accepted=False)
    with patch('experiments.vigor_medical_single_image_v1.halfstep8.gates',return_value=rejected),patch('experiments.vigor_medical_single_image_v1.halfstep8.correction',return_value=dict(status='INVALID_SECANT')) as corr:
        evaluate(rt,r,f,snapshot,old,state,lambda:None)
    assert corr.call_count==1 and not state['passed'] and state['candidate']['rejected_step_rollback']
    assert state['base_restored'] and torch.equal(base,rt.model.layer.weight)
    with patch('experiments.vigor_medical_single_image_v1.halfstep8.candidate',side_effect=RuntimeError('synthetic')):
        with TestCase().assertRaises(RuntimeError):evaluate(rt,r,f,snapshot,old,state,lambda:None)
    assert torch.equal(base,rt.model.layer.weight)
    return dict(status='PASS',prior_checks=True,uncorrected_target_supported=True,half_main_from_step_start=True,
        one_d32_candidate=True,no_other_stepsizes=True,invalid_correction_rolls_back=True,
        exception_restores_base=True,generation_calls=0,judge_calls=0)
