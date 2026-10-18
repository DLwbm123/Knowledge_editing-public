"""Bounded same-start fallback, next-step full preference, and safe stop/restoration."""
from unittest import TestCase
from unittest.mock import patch
import torch
from .halfstep8_cpu_check import run as prior_checks
from .cpu_check import Runtime,record
from .editor import calibrate
from .diagnose import candidate,FIRST_D
from .finite_null import gates
from .backoff16 import trajectory


def run():
    prior_checks()
    torch.manual_seed(42);rt=Runtime(torch.float16);r=record('trial');name='model.layers.0.mlp.down_proj'
    f=calibrate(rt,[record('train')],[name],train_ids=['train'],excluded_ids=['trial'],binding={'model':'synthetic'})['factors'][name]
    f=dict(f,a=f['a']*.001);base=rt.model.layer.weight.detach().clone()
    starts=[];after_proposal=[];correction_starts=[];requested=[];gate_calls=[]
    def main(*args,**kwargs):
        starts.append(rt.model.layer.weight.detach().clone());requested.append(args[4])
        value=candidate(*args,**kwargs);after_proposal.append(rt.model.layer.weight.detach().clone());return value
    def controlled_gates(*args):
        value=gates(*args);gate_calls.append(1)
        is_final=len(gate_calls)%2==0
        accept=(len(gate_calls)//2==2) if is_final else False
        value['gates']=dict(null=accept,visual=accept,trust=True);value['accepted']=accept
        return value
    def same_position_correction(*args,**kwargs):
        assert kwargs['compensate']=='secant'
        correction_starts.append(rt.model.layer.weight.detach().clone())
        return dict(status='APPLIED')
    with patch('experiments.vigor_medical_single_image_v1.backoff16.candidate',side_effect=main),patch('experiments.vigor_medical_single_image_v1.backoff16.gates',side_effect=controlled_gates),patch('experiments.vigor_medical_single_image_v1.backoff16.correction',side_effect=same_position_correction):
        result=trajectory(rt,r,f,lambda rows:None,edit_threshold=100,steps=16)
    assert requested==[FIRST_D/16,FIRST_D/32,FIRST_D/16,FIRST_D/32]
    assert torch.equal(starts[0],starts[1]) and torch.equal(starts[2],starts[3])
    assert all(torch.equal(a,b) for a,b in zip(after_proposal,correction_starts))
    assert result['accepted_steps']==1 and result['attempted_steps']==2
    assert result['half_proposals']==2 and result['accepted_half_steps']==1
    assert result['primary_corrections']==2 and result['half_corrections']==2
    assert result['steps'][0]['primary']['rollback_pass'] and result['steps'][1]['half']['rollback_pass']
    assert result['accepted_requested_reduction']==FIRST_D/32 and result['equivalent_d16_steps']==.5
    assert torch.equal(base,rt.model.layer.weight)
    def pass_gate(*args):
        value=gates(*args);value['gates']=dict(null=True,visual=True,trust=True);value['accepted']=True;return value
    with patch('experiments.vigor_medical_single_image_v1.backoff16.gates',side_effect=pass_gate),patch('experiments.vigor_medical_single_image_v1.backoff16.correction') as corr:
        full=trajectory(rt,r,f,lambda rows:None,edit_threshold=100,steps=2)
    assert full['accepted_primary_steps']==2 and full['half_proposals']==0 and corr.call_count==0
    assert torch.equal(base,rt.model.layer.weight)
    with TestCase().assertRaises(TimeoutError):trajectory(rt,r,f,lambda rows:None,edit_threshold=100,deadline_epoch=0)
    with patch('experiments.vigor_medical_single_image_v1.backoff16.candidate',side_effect=RuntimeError('synthetic')):
        with TestCase().assertRaises(RuntimeError):trajectory(rt,r,f,lambda rows:None,edit_threshold=100)
    assert torch.equal(base,rt.model.layer.weight)
    return dict(status='PASS',prior_checks=True,one_half_only=True,full_rejection_rollback_before_half=True,
        recomputed_correction_at_each_candidate=True,next_logical_step_prefers_full=True,
        failed_half_stops_without_third_scale=True,accepted_edit_amount_accounted=True,
        full_acceptance_skips_half=True,deadline_and_exception_restore_base=True,generation_calls=0,judge_calls=0)
