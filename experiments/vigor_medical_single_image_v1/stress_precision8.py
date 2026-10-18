"""Same compensated corrections at all eight frozen first rejections: A/B/C only."""
from functools import partial
import torch
from . import precision_abc
from .editor import paired_batches
from .finite_null import correction,gates,native_losses

POINTS=(('S01',6),('S02',11),('S03',9),('S04',6),('S05',8),('S06',5),('S07',5),('S08',7))
AUTH=dict(points=[list(p) for p in POINTS],extra_steps=0,extra_corrections=0,
    tune=False,replace=False,generation=False,judge=False,variant='compensated_stress_ABC')


def prepare_control(runtime,record,factor,snapshot,old,state,emit):
    layer=runtime.get_module('model.layers.0.mlp.down_proj');base=layer.weight.detach().clone()
    pair=paired_batches(runtime,record);losses=native_losses(runtime,pair)
    try:
        with torch.no_grad():layer.weight.copy_(snapshot['weight'])
        c=correction(runtime,record,factor,snapshot['baseline']['native'][1],compensate=True)
        precision_abc.compare(c,old['method']['steps'][-1]['correction'],'control.correction',state['checks'])
        verdict=gates(runtime,pair,factor,snapshot['start_weight'].to(layer.weight.device),snapshot['start_losses'],snapshot['baseline']['native'][1])
        if verdict['accepted'] or verdict['gates']!=old['method']['steps'][-1]['final_verdict']['gates']:
            raise ValueError('Frozen scientific rejection not reproduced')
        state['control']=c;state['control_verdict']=verdict;state['control_pass']=True;emit()
    finally:
        with torch.no_grad():layer.weight.copy_(base)
        if not torch.equal(layer.weight,base) or native_losses(runtime,pair)!=losses:
            raise ValueError('Control base restoration failed')


def main(config):
    return precision_abc.main(config,evaluation=partial(precision_abc.evaluate,compensate=True),
        authorization=AUTH,prepare=prepare_control,label='COMPENSATED_ABC8',points=POINTS,prefix_compensate=True)
