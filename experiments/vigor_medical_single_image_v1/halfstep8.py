"""One d/32 candidate from the exact start of each frozen d/16 rejected step."""
import torch
from . import precision_abc
from .diagnose import candidate,FIRST_D
from .editor import paired_batches
from .finite_null import correction,gates,native_losses

POINTS=(('S01',12),('S02',13),('S03',13),('S04',16),('S05',14),('S06',11),('S07',10),('S08',16))
AUTH=dict(points=[list(p) for p in POINTS],extra_steps=8,max_new_corrections_per_point=1,
    candidates_per_point=1,tune=False,replace=False,generation=False,judge=False,variant='fixed_d32_from_step_start')


def restore(runtime,layer,pair,weight,losses):
    with torch.no_grad():layer.weight.copy_(weight)
    return torch.equal(layer.weight.cpu(),weight.cpu()) and native_losses(runtime,pair)==losses


def prepare_control(runtime,record,factor,snapshot,old,state,emit):
    layer=runtime.get_module('model.layers.0.mlp.down_proj');base=layer.weight.detach().clone()
    pair=paired_batches(runtime,record);base_losses=native_losses(runtime,pair)
    try:
        with torch.no_grad():layer.weight.copy_(snapshot['weight'])
        last=old['method']['steps'][-1];c=None
        if last['correction'] is not None:
            c=correction(runtime,record,factor,snapshot['baseline']['native'][1],compensate='secant')
            precision_abc.compare(c,last['correction'],'control.correction',state['checks'])
        verdict=gates(runtime,pair,factor,snapshot['start_weight'].to(layer.weight.device),snapshot['start_losses'],snapshot['baseline']['native'][1])
        precision_abc.compare(dict(native_after=verdict['native_nll']),dict(native_after=last['final_verdict']['native_nll']),'control.final',state['checks'])
        if verdict['accepted'] or verdict['gates']!=last['final_verdict']['gates']:
            raise ValueError('Original full-step rejection not reproduced')
        state['control']=dict(correction=c,verdict=verdict);state['control_pass']=True;emit()
    finally:
        state['control_base_restored']=restore(runtime,layer,pair,base,base_losses);emit()
        if not state['control_base_restored']:raise ValueError('Control base restoration failed')


def evaluate(runtime,record,factor,snapshot,old,state,emit):
    layer=runtime.get_module('model.layers.0.mlp.down_proj');base=layer.weight.detach().clone()
    pair=paired_batches(runtime,record);base_losses=native_losses(runtime,pair)
    try:
        # Recompute the main update and correction from the old STEP START, not its proposal.
        with torch.no_grad():layer.weight.copy_(snapshot['start_weight'])
        start_losses=native_losses(runtime,pair)
        precision_abc.compare(dict(native_before=start_losses),dict(native_before=snapshot['start_losses']),'candidate.start',state['checks'])
        if start_losses!=snapshot['start_losses']:raise ValueError('Step start changed')
        state['new_main_candidate_count']=1;state['new_correction_count']=0
        result=dict(requested_reduction=FIRST_D/32,correction=None)
        state['candidate']=result;emit()
        result['proposal']=candidate(runtime,record,0,factor,FIRST_D/32,retain=True)
        result['proposal_verdict']=gates(runtime,pair,factor,snapshot['start_weight'].to(layer.weight.device),start_losses,snapshot['baseline']['native'][1]);emit()
        if not result['proposal_verdict']['gates']['null']:
            state['new_correction_count']=1
            result['correction']=correction(runtime,record,factor,snapshot['baseline']['native'][1],compensate='secant')
            emit()
        verdict=gates(runtime,pair,factor,snapshot['start_weight'].to(layer.weight.device),start_losses,snapshot['baseline']['native'][1])
        if result['correction'] is not None and result['correction']['status']!='APPLIED':
            verdict['accepted']=False;verdict['gates']['valid_correction']=False
        changed=int(torch.count_nonzero(layer.weight.detach().cpu()!=snapshot['start_weight']))
        result.update(verdict=verdict,native_changed_elements=changed,null_margin=.01-abs(verdict['cumulative_null_change']),
            visual_gain_from_step_start=start_losses[0]-verdict['native_nll'][0],
            visual_gain_per_requested_reduction=(start_losses[0]-verdict['native_nll'][0])/(FIRST_D/32))
        if not verdict['accepted']:
            result['rejected_step_rollback']=restore(runtime,layer,pair,snapshot['start_weight'],start_losses)
            if not result['rejected_step_rollback']:raise ValueError('Rejected half-step rollback failed')
        state.update(status='COMPLETE',passed=verdict['accepted'] and changed>0,candidate_count=1);emit()
    finally:
        state['base_restored']=restore(runtime,layer,pair,base,base_losses)
        state['base_restored']=state['base_restored'] and not layer._forward_hooks and not layer._forward_pre_hooks
        emit()
        if not state['base_restored']:raise ValueError('Half-step base restoration failed')


def main(config):
    return precision_abc.main(config,evaluation=evaluate,authorization=AUTH,prepare=prepare_control,
        label='FIXED_D32',points=POINTS,prefix_compensate='secant',target_correction_required=False)
