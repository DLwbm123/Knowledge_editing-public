"""Bounded three-stage engineering qualification; persist before rejecting."""
from contextlib import contextmanager
from dataclasses import replace
import math
import torch
from .editor import nll,finite,paired_batches
from .finite_null import native_losses
from .direct_fisher16 import check_time
from .chainprecision_grid import floating_tensors


def cache_items(model):
    result={}
    for name,m in model.named_modules():
        if 'vision_tower' in name or 'mm_projector' in name:continue
        for k,v in m._buffers.items():
            if v is not None:result[name+'.buffer.'+k]=v
        for k,v in vars(m).items():
            if isinstance(v,torch.Tensor) and k not in m._buffers and k not in m._parameters:result[name+'.attr.'+k]=v
            elif k in ('max_seq_len_cached','original_max_seq_len','attention_scaling','rope_type'):result[name+'.scalar.'+k]=v
    return result


def snapshot(model):
    return {k:v.detach().cpu().clone() if isinstance(v,torch.Tensor) else v for k,v in cache_items(model).items()}


def changes(model,reference):
    current=cache_items(model);diff=[]
    for k in sorted(set(current)|set(reference)):
        a=current.get(k);b=reference.get(k)
        if isinstance(a,torch.Tensor) and isinstance(b,torch.Tensor):
            a=a.detach().cpu()
            if a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b):continue
            d=dict(field=k,dtype_before=str(b.dtype),dtype_after=str(a.dtype),shape_equal=a.shape==b.shape)
            if a.shape==b.shape:d['max_absolute_difference']=float((a.double()-b.double()).abs().max()) if a.numel() else 0.
            diff.append(d)
        elif type(a)!=type(b) or a!=b:diff.append(dict(field=k,changed=True))
    return diff


@contextmanager
def complete_fp32(model,roots,out,emit):
    params=list(dict.fromkeys(p for r in roots for p in r.parameters()));saved=[(p,p.dtype,p.requires_grad) for p in params]
    flags=[(p,p.requires_grad) for p in model.parameters()];modes=[(m,m.training) for m in model.modules()]
    buffers=[(m,k,v,v.detach().clone()) for r in roots for m in r.modules() for k,v in m._buffers.items() if v is not None]
    attrs=[(m,k,v,v.detach().clone()) for r in roots for m in r.modules() for k,v in vars(m).items() if isinstance(v,torch.Tensor) and k not in m._buffers and k not in m._parameters]
    scalars=[(m,k,getattr(m,k)) for r in roots for m in r.modules() for k in ('max_seq_len_cached','original_max_seq_len','attention_scaling','rope_type') if hasattr(m,k)]
    tf32=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32);hooks=[]
    out.update(float_output_count=0,non_fp32_output_count=0,roots_include_rotary=True)
    def hook(m,args,value):
        values=list(floating_tensors(value));out['float_output_count']+=len(values)
        bad=sum(v.dtype!=torch.float32 for v in values);out['non_fp32_output_count']+=bad
        if bad:emit();raise ValueError('Non-FP32 executed floating output')
    try:
        model.eval()
        for p,_ in flags:p.requires_grad_(False)
        for r in roots:r.float()
        for m,k,orig,_ in attrs:setattr(m,k,orig.float())
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        if any(p.dtype!=torch.float32 for p in params):raise ValueError('Incomplete parameter precision switch')
        for m in model.modules():hooks.append(m.register_forward_hook(hook))
        with torch.autocast(device_type='cuda',enabled=False),torch.autocast(device_type='cpu',enabled=False):yield
    finally:
        for h in hooks:h.remove()
        for p,dtype,flag in saved:p.data=p.detach().to(dtype);p.requires_grad_(flag)
        for m,k,orig,value in buffers:
            with torch.no_grad():orig.copy_(value)
            m._buffers[k]=orig
        for m,k,orig,value in attrs:
            with torch.no_grad():orig.copy_(value)
            setattr(m,k,orig)
        for m,k,value in scalars:setattr(m,k,value)
        for p,flag in flags:p.requires_grad_(flag)
        for m,mode in modes:m.training=mode
        torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=tf32
        out['precision_context_exited']=True;emit()


def qualify(runtime,record,endpoint,update,output,emit,deadline,*,roots=None,expected_native=None):
    layer=runtime.get_module('model.layers.0.mlp.down_proj');native=layer.weight.detach().clone()
    pair=paired_batches(runtime,record);native_losses_before=native_losses(runtime,pair)
    parameter_state=[(p,p.dtype,p.requires_grad) for p in runtime.model.parameters()]
    mode_state=[(m,m.training,len(m._forward_hooks),len(m._forward_pre_hooks)) for m in runtime.model.modules()]
    native_state=snapshot(runtime.model);output.update(status='RUNNING',events=[],stages_completed=[])
    base=endpoint.to(layer.weight.device).float();u=update.to(base.device);first=None
    fp_pair=tuple(replace(b,inputs_embeds=b.inputs_embeds.float()) for b in pair)
    inputs=[{k:v.detach().clone() for k,v in b.forward_kwargs().items() if isinstance(v,torch.Tensor)} for b in fp_pair]
    if roots is None:roots=[runtime.model.model.layers,runtime.model.model.norm,runtime.model.model.rotary_emb,runtime.model.lm_head]
    if layer.weight.is_cuda:
        torch.cuda.empty_cache();free,_=torch.cuda.mem_get_info()
        extra=sum(p.numel()*(4-p.element_size()) for p in dict.fromkeys(p for r in roots for p in r.parameters()))
        if free<extra+12*1024**3:raise MemoryError('FP32 headroom insufficient')
    def event(label,reference,expected_weight,gradient=False):
        check_time(deadline)
        before=changes(runtime.model,reference)
        equal=torch.equal(layer.weight,expected_weight)
        row=dict(label=label,weight_equal_before=equal,weight_max_abs_before=float((layer.weight.detach()-expected_weight).abs().max()),cache_changes_before=before)
        output['events'].append(row);emit()
        if not equal or before:raise ValueError('Pre-read weight/cache qualification failed')
        if gradient:
            loss=nll(runtime.model,fp_pair[1]);grad=finite(torch.autograd.grad(loss,layer.weight)[0].detach())
            row.update(null_loss=float(loss.detach()),gradient_norm=float(torch.linalg.vector_norm(grad)));del loss,grad
        else:
            with torch.no_grad():row['losses']=[float(nll(runtime.model,b)) for b in fp_pair]
        row.update(weight_equal_after=torch.equal(layer.weight,expected_weight),cache_changes_after=changes(runtime.model,reference),
            input_unchanged=all(torch.equal(v,inputs[i][k]) for i,b in enumerate(fp_pair) for k,v in b.forward_kwargs().items() if isinstance(v,torch.Tensor)),
            training_modules=sum(m.training for m in runtime.model.modules()),tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32,
            attention_implementation=getattr(getattr(runtime.model,'config',None),'_attn_implementation',None),weight_dtype=str(layer.weight.dtype))
        if not gradient:
            row['baseline_difference']=None if first is None else max(abs(x-y) for x,y in zip(first,row['losses']))
        emit()
        if not row['weight_equal_after'] or row['cache_changes_after'] or not row['input_unchanged'] or row['training_modules'] or row['tf32_matmul'] or row['tf32_cudnn']:raise ValueError('Post-read state qualification failed')
        vals=[row['null_loss']] if gradient else row['losses']
        if not all(math.isfinite(x) for x in vals):raise ValueError('Nonfinite readout')
        if not gradient and row['baseline_difference'] is not None and row['baseline_difference']>1e-5:raise ValueError('Baseline readout mismatch')
        return row
    try:
        with torch.no_grad():layer.weight.copy_(endpoint)
        for entry in (1,2):
            audit={};output['context_'+str(entry)]=audit
            with complete_fp32(runtime.model,roots,audit,emit):
                with torch.no_grad():layer.weight.copy_(base)
                layer.weight.requires_grad_(True);reference=snapshot(runtime.model)
                if entry==1:
                    first=event('S1.fixed.first',reference,base)['losses']
                    event('S1.fixed.repeat',reference,base)
                    event('S1.null_gradient',reference,base,True)
                    event('S1.after_gradient',reference,base)
                    output['stages_completed'].append(1);emit()
                    with torch.no_grad():layer.weight.copy_(base+u);layer.weight.copy_(base)
                    event('S2.roundtrip.first',reference,base);event('S2.roundtrip.repeat',reference,base)
                    output['stages_completed'].append(2);emit()
                else:
                    event('S3.reentry.first',reference,base);event('S3.reentry.repeat',reference,base)
                    output['stages_completed'].append(3);emit()
            with torch.no_grad():layer.weight.copy_(endpoint)
            row=dict(label='native_exit_'+str(entry),weight_equal=torch.equal(layer.weight,endpoint.to(layer.weight.device)),cache_changes=changes(runtime.model,native_state))
            output['events'].append(row);emit()
            if not row['weight_equal'] or row['cache_changes']:raise ValueError('Native state restoration mismatch')
            if entry==1:
                read=native_losses(runtime,pair);row['losses']=read;row['difference']=max(abs(x-y) for x,y in zip(read,expected_native if expected_native is not None else native_losses_before));emit()
                if row['difference']>1e-5:raise ValueError('Native endpoint loss restoration mismatch')
        output['status']='COMPLETE';emit()
    except BaseException as e:
        output.update(status='STOPPED',error_type=type(e).__name__,failure=str(e));emit();raise
    finally:
        with torch.no_grad():layer.weight.copy_(native)
        restored=native_losses(runtime,pair)
        precision_restored=all(p.dtype==d and p.requires_grad==flag for p,d,flag in parameter_state) and all(m.training==mode and len(m._forward_hooks)==h and len(m._forward_pre_hooks)==pre for m,mode,h,pre in mode_state)
        output.update(precision_flags_hooks_restored=precision_restored,base_weight_restored=torch.equal(layer.weight,native),base_losses_restored=restored==native_losses_before,base_cache_changes=changes(runtime.model,native_state));emit()
        if not precision_restored or not output['base_weight_restored'] or not output['base_losses_restored'] or output['base_cache_changes']:raise ValueError('Final base restoration failed')
