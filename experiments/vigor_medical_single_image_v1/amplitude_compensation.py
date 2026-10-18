"""One deterministic amplitude compensation at each frozen failure point."""
import torch
from . import precision_abc
from .editor import paired_batches
from .finite_null import correction,gates,native_losses

AUTH=dict(precision_abc.AUTH,variant='one_shot_amplitude',candidates_per_point=1)


def point(runtime,record,factor,snapshot,old,state,emit, *, compensated, mode=None):
    layer=runtime.get_module('model.layers.0.mlp.down_proj');base=layer.weight.detach().clone()
    pair=paired_batches(runtime,record);base_losses=native_losses(runtime,pair)
    key='candidate' if compensated else 'control'
    try:
        with torch.no_grad():layer.weight.copy_(snapshot['weight'])
        c=correction(runtime,record,factor,snapshot['baseline']['native'][1],compensate=mode if mode is not None else compensated)
        state[key]=dict(correction=c);emit()
        if not compensated:
            precision_abc.compare(c,old['method']['steps'][-1]['correction'],'target.control',state['checks'])
        verdict=gates(runtime,pair,factor,snapshot['start_weight'].to(layer.weight.device),snapshot['start_losses'],snapshot['baseline']['native'][1])
        if c['status']!='APPLIED':
            verdict['accepted']=False;verdict['gates']['valid_correction']=False
        state[key]['verdict']=verdict
        state[key]['null_margin']=.01-abs(verdict['cumulative_null_change'])
        state[key]['visual_gain_from_step_start']=snapshot['start_losses'][0]-verdict['native_nll'][0]
        if not compensated and verdict['gates']!=old['method']['steps'][-1]['final_verdict']['gates']:
            raise ValueError('Original control gate changed')
        if not verdict['accepted']:
            with torch.no_grad():layer.weight.copy_(snapshot['start_weight'])
            restored=torch.equal(layer.weight.cpu(),snapshot['start_weight']) and native_losses(runtime,pair)==snapshot['start_losses']
            state[key]['rejected_step_rollback']=restored
            if not restored:raise ValueError('Rejected step rollback failed')
        if compensated:
            state['status']='COMPLETE';state['passed']=verdict['accepted'];state['candidate_count']=1
        emit()
    finally:
        with torch.no_grad():layer.weight.copy_(base)
        restored=torch.equal(layer.weight,base) and native_losses(runtime,pair)==base_losses and not layer._forward_hooks and not layer._forward_pre_hooks
        state[key+'_base_restored']=restored
        if compensated:state['base_restored']=restored
        emit()
        if not restored:raise ValueError('Base restoration failed')


def prepare(*args):point(*args,compensated=False)
def evaluate(*args):point(*args,compensated=True)
def main(config):precision_abc.main(config,evaluation=evaluate,authorization=AUTH,prepare=prepare,label='AMPLITUDE')
