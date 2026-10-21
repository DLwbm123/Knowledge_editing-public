"""Fixed parameter trajectory readout; no direction or scale selection."""
from contextlib import contextmanager
from dataclasses import replace
import math
import torch
from .editor import editable_fp32,finite,nll,paired_batches
from .finite_null import native_losses
from .direct_fisher16 import check_time


def floating_tensors(x):
    if isinstance(x,torch.Tensor):
        if x.is_floating_point():yield x
    elif isinstance(x,dict):
        for value in x.values():yield from floating_tensors(value)
    elif isinstance(x,(tuple,list)):
        for value in x:yield from floating_tensors(value)


@contextmanager
def chain_fp32(model,modules):
    params=list(dict.fromkeys(p for m in modules for p in m.parameters()))
    originals=[(p,p.dtype,p.requires_grad) for p in params]
    buffers=[(m,k,b.detach().clone()) for root in modules for m in root.modules() for k,b in m._buffers.items() if b is not None]
    flags=[(p,p.requires_grad) for p in model.parameters()]
    modes=[(m,m.training) for m in model.modules()]
    audit=dict(executed_module_calls=0,floating_outputs=0,non_fp32_outputs=0,parameter_count=sum(p.numel() for p in params))
    hooks=[]
    tf32=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
    def check(m,args,out):
        values=list(floating_tensors(out));audit['executed_module_calls']+=1;audit['floating_outputs']+=len(values)
        bad=sum(t.dtype!=torch.float32 for t in values);audit['non_fp32_outputs']+=bad
        if bad:raise ValueError('Executed floating output is not FP32')
    try:
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model.eval()
        for p,_ in flags:p.requires_grad_(False)
        for m in modules:m.float()
        if any(p.dtype!=torch.float32 for p in params):raise ValueError('Incomplete FP32 parameter cast')
        for m in model.modules():hooks.append(m.register_forward_hook(check))
        with torch.autocast(device_type='cuda',enabled=False),torch.autocast(device_type='cpu',enabled=False):yield audit
        if audit['floating_outputs']==0:raise ValueError('No executed FP32 outputs audited')
    finally:
        torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=tf32
        for h in hooks:h.remove()
        for p,dtype,flag in originals:
            p.data=p.detach().to(dtype);p.requires_grad_(flag)
        for m,k,b in buffers:m._buffers[k]=b
        for p,flag in flags:p.requires_grad_(flag)
        for m,mode in modes:m.training=mode


def grid(runtime,pair,layer,base,u,deadline):
    values={};linear=None;gradient_norm=None
    for t in (0,.5,1):
        check_time(deadline)
        with torch.no_grad():layer.weight.copy_(base+t*u)
        if t==0:
            layer.weight.requires_grad_(True)
            loss=nll(runtime.model,pair[1]);g=finite(torch.autograd.grad(loss,layer.weight)[0].detach())
            linear=float((g.double()*u.double()).sum());gradient_norm=float(torch.linalg.vector_norm(g))
            del g,loss
        with torch.no_grad():values[str(t)]=[float(nll(runtime.model,b)) for b in pair]
    with torch.no_grad():layer.weight.copy_(base)
    with torch.no_grad():again=[float(nll(runtime.model,b)) for b in pair]
    repeat=max(abs(x-y) for x,y in zip(values['0'],again))
    if repeat>1e-5:raise ValueError('Baseline repeat unstable')
    if not all(math.isfinite(v) for row in values.values() for v in row):raise ValueError('Nonfinite grid')
    return dict(losses=values,null_linear_full=linear,null_gradient_norm=gradient_norm,
        fp32_update_norm=float(torch.linalg.vector_norm(u)),baseline_repeat_error=repeat,positions=[0,.5,1])


def read_grid(runtime,record,snapshot,update,proposal,deadline,emit,*,modules=None,output=None):
    layer=runtime.get_module('model.layers.0.mlp.down_proj');original=layer.weight.detach().clone()
    pair=paired_batches(runtime,record);original_losses=native_losses(runtime,pair)
    answer={} if output is None else output
    try:
        with torch.no_grad():layer.weight.copy_(snapshot)
        endpoint=layer.weight.detach().clone();base=endpoint.float();u=update.to(base.device)
        if float(torch.linalg.vector_norm(u))!=proposal['quantization']['fp32_delta_l2']:raise ValueError('Captured effective update changed')
        with editable_fp32(runtime.model,layer,readonly=True):
            answer['mixed']=grid(runtime,pair,layer,base,u,deadline)
        for t,key in [('0','fp32_before'),('1','fp32_after')]:
            if max(abs(x-y) for x,y in zip(answer['mixed']['losses'][t],proposal[key]))>1e-5:raise ValueError('Existing mixed readout changed')
        if abs(answer['mixed']['null_linear_full']-proposal['fp32_update']['null_linear_change'])>1e-6:raise ValueError('Mixed gradient derivative changed')
        check_time(deadline)
        if layer.weight.is_cuda:
            torch.cuda.empty_cache();free,_=torch.cuda.mem_get_info()
            roots=[runtime.model.model.layers,runtime.model.model.norm,runtime.model.lm_head] if modules is None else modules
            extra=sum(p.numel()*(4-p.element_size()) for p in dict.fromkeys(p for m in roots for p in m.parameters()))
            if free<extra+12*1024**3:raise MemoryError('FP32 chain margin unavailable; no downgrade')
        else:roots=modules
        fp_pair=tuple(replace(b,inputs_embeds=b.inputs_embeds.float()) for b in pair)
        with chain_fp32(runtime.model,roots) as audit:
            try:answer['chain_fp32']=grid(runtime,fp_pair,layer,base,u,deadline)
            finally:
                with torch.no_grad():layer.weight.copy_(base)
        answer['chain_audit']=audit
        if not torch.equal(layer.weight,endpoint):raise ValueError('Endpoint changed after FP32 context')
        if max(abs(x-y) for x,y in zip(native_losses(runtime,pair),proposal['native_before']))>1e-5:raise ValueError('Native endpoint readout restoration failed')
        answer['endpoint_restored']=True
        return answer
    finally:
        with torch.no_grad():layer.weight.copy_(original)
        answer['base_restored']=torch.equal(layer.weight,original) and native_losses(runtime,pair)==original_losses
        emit()
        if not answer['base_restored']:raise ValueError('Base restoration failed')
