"""Sixteen logical steps: one full attempt, then at most one same-start half attempt."""
import time
import torch
from . import batch8
from .diagnose import candidate,FIRST_D
from .editor import paired_batches
from .finite_null import correction,gates,native_losses

AUTH=dict(samples=8,steps=16,reference_steps=1,primary_denominator=16,fallback_denominator=32,
    max_fallbacks_per_step=1,max_corrections_per_candidate=1,tune=False,replace=False,
    expand=False,generation=False,judge=False,variant='secant_single_half_backoff16')


def _attempt(runtime,record,factor,pair,start_weight,start_losses,base_null,requested,anchor,trial,emit):
    trial.update(requested_reduction=requested,status='PROPOSING',correction=None);emit()
    trial['proposal']=candidate(runtime,record,0,factor,requested,anchor,retain=True)
    trial['proposal_verdict']=gates(runtime,pair,factor,start_weight,start_losses,base_null)
    trial['status']='PROPOSED';emit()
    if not trial['proposal_verdict']['gates']['null']:
        trial['correction']=correction(runtime,record,factor,base_null,compensate='secant')
        trial['status']='CORRECTED';emit()
    verdict=gates(runtime,pair,factor,start_weight,start_losses,base_null)
    if trial['correction'] is not None and trial['correction']['status']!='APPLIED':
        verdict['accepted']=False;verdict['gates']['valid_correction']=False
    trial['final_verdict']=verdict
    trial['status']='ACCEPTED' if verdict['accepted'] else 'REJECTED';emit()
    return verdict


def _rollback(layer,pair,runtime,weight,losses):
    with torch.no_grad():layer.weight.copy_(weight)
    return torch.equal(layer.weight,weight) and native_losses(runtime,pair)==losses


def trajectory(runtime,record,factor,emit,anchor=None,first_reference=None, *,
        edit_threshold,compensate='secant',steps=16,prefix_reference=None,deadline_epoch=None):
    if compensate!='secant':raise PermissionError('Frozen backoff requires the secant correction')
    layer=runtime.get_module('model.layers.0.mlp.down_proj');original=layer.weight.detach().clone()
    pair=paired_batches(runtime,record);baseline=native_losses(runtime,pair);rows=[];accepted=[]
    def check_time():
        if deadline_epoch is not None and time.time()>=deadline_epoch:
            raise TimeoutError('Frozen backoff trajectory time budget exhausted')
    try:
        for i in range(steps):
            check_time()
            start_weight=layer.weight.detach().clone();start_losses=native_losses(runtime,pair)
            row=dict(step=i+1,start_native_nll=start_losses,status='PROPOSING',primary={},half=None)
            rows.append(row)
            progress=lambda:emit(rows)
            verdict=_attempt(runtime,record,factor,pair,start_weight,start_losses,baseline[1],
                FIRST_D/16,anchor if i==0 else None,row['primary'],progress)
            if i==0 and first_reference is not None:
                if any(abs(a-b)>1e-5 for a,b in zip(row['primary']['proposal']['native_after'],first_reference)):
                    raise ValueError('First full proposal reference mismatch')
            selected='primary'
            if not verdict['accepted']:
                row['primary']['rollback_pass']=_rollback(layer,pair,runtime,start_weight,start_losses)
                row['status']='FULL_REJECTED_ROLLED_BACK';progress()
                if not row['primary']['rollback_pass']:raise ValueError('Full-step rollback failed')
                check_time()
                row['half']={}
                verdict=_attempt(runtime,record,factor,pair,start_weight,start_losses,baseline[1],
                    FIRST_D/32,anchor if i==0 else None,row['half'],progress)
                selected='half'
            row['selected_attempt']=selected;row['final_verdict']=verdict
            if verdict['accepted']:
                chosen=row[selected]
                row['status']='ACCEPTED'
                accepted.append(dict(step=i+1,selected_attempt=selected,native_nll=verdict['native_nll'],
                    requested_reduction=chosen['proposal']['requested_reduction'],
                    effective_reduction=chosen['proposal']['effective_reduction'],
                    cumulative_visual_gain=baseline[0]-verdict['native_nll'][0],
                    cumulative_null_change=verdict['cumulative_null_change']))
                progress()
            else:
                row['half']['rollback_pass']=_rollback(layer,pair,runtime,start_weight,start_losses)
                row['rollback_pass']=row['half']['rollback_pass'];row['status']='REJECTED_AND_STOPPED';progress()
                if not row['rollback_pass']:raise ValueError('Half-step rollback failed')
                break
        final=native_losses(runtime,pair);complete=len(accepted)==steps
        edit=final[0]<=edit_threshold and final[0]<baseline[0]
        null=bool(accepted) and all(abs(s['cumulative_null_change'])<=.01 for s in accepted)
        full_corr=sum(r['primary'].get('correction') is not None for r in rows)
        half_corr=sum(r.get('half') is not None and r['half'].get('correction') is not None for r in rows)
        full_accepted=sum(s['selected_attempt']=='primary' for s in accepted)
        half_accepted=sum(s['selected_attempt']=='half' for s in accepted)
        return dict(status='COMPLETE' if complete else 'STOPPED_REJECTED',baseline=baseline,steps=rows,
            accepted_trajectory=accepted,attempted_steps=len(rows),accepted_steps=len(accepted),
            primary_proposals=len(rows),half_proposals=sum(r.get('half') is not None for r in rows),
            accepted_primary_steps=full_accepted,accepted_half_steps=half_accepted,
            primary_corrections=full_corr,half_corrections=half_corr,correction_attempts=full_corr+half_corr,
            accepted_requested_reduction=sum(s['requested_reduction'] for s in accepted),
            accepted_effective_reduction=sum(s['effective_reduction'] for s in accepted),
            equivalent_d16_steps=full_accepted+half_accepted/2,
            four_steps_completed=len(accepted)>=4,trajectory_completed=complete,requested_steps=steps,
            final_accepted_native_nll=final,edit_threshold_met=edit,null_gate_on_accepted_states=null,
            joint_gate=complete and edit and null,generation_calls=0,judge_calls=0)
    finally:
        with torch.no_grad():layer.weight.copy_(original)
        if not torch.equal(layer.weight,original) or native_losses(runtime,pair)!=baseline or layer._forward_hooks or layer._forward_pre_hooks:
            raise ValueError('Final backoff base restoration failed')


def main(config):
    return batch8.main(config,trajectory_fn=trajectory,authorization=AUTH)
