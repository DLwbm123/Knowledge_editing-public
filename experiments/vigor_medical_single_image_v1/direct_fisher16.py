"""One fixed per-example Fisher diagonal; original native gates and trust metric."""
import time
from pathlib import Path
import json
import torch
from . import batch8,backoff16
from .editor import damp,direction,editable_fp32,finite,gradient_pair,nll,paired_batches
from .prepare import write

LAYER='model.layers.0.mlp.down_proj'
AUTH=dict(backoff16.AUTH,variant='direct_empirical_diagonal_backoff16',calibration_samples=32)


def check_time(deadline):
    if deadline is not None and time.time()>=deadline:
        raise TimeoutError('Direct diagonal experiment budget exhausted')


def estimate(runtime,records,factor,progress,deadline=None):
    layer=runtime.get_module(LAYER);base=layer.weight.detach().clone();started=time.time()
    accumulator=torch.zeros(layer.weight.shape,dtype=torch.float64,device=layer.weight.device)
    with editable_fp32(runtime.model,layer,readonly=True):
        for i,record in enumerate(records):
            check_time(deadline)
            batch=runtime.build_edit_batch(record)
            loss=nll(runtime.model,batch)
            gradient=finite(torch.autograd.grad(loss,layer.weight)[0].detach()).double()
            accumulator.addcmul_(gradient,gradient)
            del gradient,loss,batch
            progress(dict(stage='DIRECT_FISHER',completed=i+1,total=len(records)))
    if not torch.equal(layer.weight,base):raise ValueError('Calibration changed native base')
    accumulator.div_(len(records))
    raw_sum=float(accumulator.sum());raw_mean=float(accumulator.mean())
    if raw_sum<=0:raise ValueError('Zero direct empirical Fisher')
    diagonal=damp(accumulator.float().flatten()).view(layer.weight.shape)
    del accumulator
    old_trace=float(factor['a'].double().sum()*factor['b'].double().sum())
    damped_trace=float(diagonal.double().sum());scale=old_trace/damped_trace
    diagonal.mul_(scale);finite(diagonal)
    if not (diagonal>0).all():raise ValueError('Direct metric lost positivity')
    new_trace=float(diagonal.double().sum());trace_error=abs(new_trace/old_trace-1)
    if trace_error>1e-5:raise ValueError('Direct metric scale calibration failed')
    torch.cuda.synchronize() if diagonal.is_cuda else None
    return diagonal,dict(status='PASS',samples=len(records),shape=list(diagonal.shape),
        estimator='mean_of_squared_per_example_full_weight_gradients',loss='mean_target_token_nll',
        editable_layer_precision='float32',remaining_backbone='native_float16',
        damping_relative_mean=1e-3,pre_scale_floor=1e-12,scale_matching='trace_of_original_parameter_diagonal',
        raw_trace=raw_sum,raw_mean=raw_mean,damped_trace=damped_trace,original_trace=old_trace,
        multiplier=scale,final_trace=new_trace,relative_trace_error=trace_error,
        diagonal_min=float(diagonal.min()),diagonal_max=float(diagonal.max()),
        in_memory_bytes=diagonal.numel()*diagonal.element_size(),saved_factor=False,
        base_restored=True,elapsed_seconds=time.time()-started)


def prepare_solver(runtime,factor,calibration,config):
    from m3bench_repro.editors.llava_runtime import EditorRecord
    root=Path(config['output']);manifest=json.loads(Path(config['manifest']).read_text())
    raw=json.loads(Path(config['records']).read_text())
    ids=[r['record_id'] for r in raw]
    if len(ids)!=32 or len(set(ids))!=32 or ids!=manifest['calibration_ids'] or ids!=calibration['calibration_ids']:
        raise ValueError('Original calibration records changed')
    records=[EditorRecord(**dict(r,image_path=Path(r['image_path']))) for r in raw]
    diagonal,metadata=estimate(runtime,records,factor,
        lambda event:write(root/'public/CALIBRATION_PROGRESS.json',event),min(config['deadline_epoch'],time.time()+120))
    metadata.update(original_calibration_ids_match=True,calibration_binding_match=True,
        original_factors_unchanged=True,new_training_records=0,new_evaluation_records=0)
    write(root/'public/DIRECT_FISHER.json',metadata)

    def trajectory(runtime,record,factor,emit,anchor=None,**kwargs):
        check_time(config['deadline_epoch'])
        layer=runtime.get_module(LAYER);pair=paired_batches(runtime,record)
        a,b=(factor[k].to(layer.weight.device) for k in ('a','b'))
        repeats=[]
        with editable_fp32(runtime.model,layer,readonly=True):
            for _ in range(2):
                losses,gradients=gradient_pair(runtime,layer,pair)
                tangent,geometry=direction(*gradients,a,b,diagonal=diagonal)
                if tangent is None:raise ValueError('Direct solver baseline degenerate')
                repeats.append(dict(visual_nll=losses[0],null_nll=losses[1],E=geometry['E']))
        differences={k:abs(repeats[1][k]-repeats[0][k]) for k in ('visual_nll','null_nll')}
        erel=abs(repeats[1]['E']/repeats[0]['E']-1)
        if max(differences.values())>1e-5 or erel>1e-5:
            raise ValueError('Direct solver baseline is nondeterministic')
        del tangent,gradients
        result=backoff16.trajectory(runtime,record,factor,emit,anchor=repeats[0],solve_factor=diagonal,**kwargs)
        result['solver_baseline']=dict(status='PASS',repeats=repeats,NLL_absolute_differences=differences,
            E_relative_difference=erel,old_E_not_used_as_new_metric_anchor=True)
        return result
    return trajectory,metadata


def main(config):
    return batch8.main(config,authorization=AUTH,prepare_solver=prepare_solver)
