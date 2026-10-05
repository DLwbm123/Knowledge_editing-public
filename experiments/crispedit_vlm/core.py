"""Independent K-FAC adapter; the pinned upstream ProjectedAdam runs unmodified."""
import importlib.util
from pathlib import Path
import torch
import torch.nn.functional as F

UPSTREAM_COMMIT = '09035f16695998f3a71ec6006245d99e8cc648c8'


def optimizer_class(upstream):
    path=Path(upstream)/'easyeditor/models/crispedit/projected_adam.py'
    spec=importlib.util.spec_from_file_location('pinned_projected_adam',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.ProjectedAdam


@torch.no_grad()
def projection(a, b, energy=.9):
    """Match upstream strict mask and cumulative energy threshold (including ties)."""
    if not 0 < energy < 1 or a.ndim!=2 or b.ndim!=2:
        raise ValueError('invalid projection input')
    if any(not torch.isfinite(t).all() for t in (a,b)):
        raise ValueError('nonfinite factors')
    sa,ua=torch.linalg.eigh(a);sb,ub=torch.linalg.eigh(b)
    spectrum=torch.outer(sa,sb)
    ordered=torch.sort(spectrum.flatten(),descending=True).values
    total=ordered.sum()
    if not float(total)>0:raise ValueError('zero capability curvature')
    rank=int(torch.searchsorted(ordered.cumsum(0)/total,energy))+1
    threshold=ordered[rank-1] if rank<=ordered.numel() else ordered.new_tensor(0.)
    mask=spectrum<threshold
    receipt=dict(energy=energy,rank=rank,total_directions=mask.numel(),
                 retained_directions=int(mask.sum()),threshold=float(threshold),
                 input_min=float(sa.min()),output_min=float(sb.min()))
    return dict(Ua=ua,Ub=ub,M=mask),receipt


def factors(model, batches, layers, progress=None):
    """Upstream empirical CE-gradient factor estimator, with explicit valid masks.

    Batches supply native inputs and labels. For plain text every next token is
    valid; VLM edit histories use answer-predictor positions only. CE is summed
    as upstream, so B is not accidentally divided by sequence length squared.
    No gradients or parameter updates are accumulated on the frozen model.
    """
    moments={};captures={};handles=[];tokens=0;documents=0
    for layer in layers:
        module=model.get_submodule(f'model.layers.{layer}.mlp.down_proj')
        out_dim,in_dim=module.weight.shape
        moments[layer]=dict(A=torch.zeros(in_dim,in_dim,device=module.weight.device),
                            B=torch.zeros(out_dim,out_dim,device=module.weight.device))
        def capture(_,args,output,layer=layer):
            entry=dict(x=args[0].detach());captures[layer]=entry
            output.requires_grad_(True)
            output.register_hook(lambda grad,entry=entry:entry.update(g=grad.detach()))
        handles.append(module.register_forward_hook(capture))
    try:
        for inputs,labels in batches:
            mask=labels[:,1:]!=-100
            count=int(mask.sum())
            if count<1:raise ValueError('empty factor batch')
            logits=model(**inputs).logits
            loss=F.cross_entropy(logits[:,:-1].float().reshape(-1,logits.shape[-1]),
                                 labels[:,1:].reshape(-1),ignore_index=-100,reduction='sum')
            if not torch.isfinite(loss):raise ValueError('nonfinite factor loss')
            loss.backward()
            with torch.no_grad():
                if set(captures)!=set(layers):raise RuntimeError('missing factor hook')
                for layer,v in captures.items():
                    x=v['x'][:,:-1][mask].float();g=v['g'][:,:-1][mask].float()
                    if not torch.isfinite(g).all():raise ValueError('nonfinite factor derivative')
                    moments[layer]['A'].addmm_(x.T,x)
                    moments[layer]['B'].addmm_(g.T,g)
            documents+=labels.shape[0];tokens+=count
            captures.clear();del logits,loss,x,g
            if progress is not None:progress(documents,tokens)
    finally:
        for h in handles:h.remove()
        captures.clear()
    if not tokens:raise ValueError('empty capability data')
    result={}
    for layer in layers:
        pair=moments.pop(layer)
        result[layer]=dict(A=(pair['A']/tokens).cpu(),B=(pair['B']/tokens).cpu(),N=tokens)
    return result


def combine(left,right):
    """Token-weighted sufficient statistics, matching upstream sequential cache."""
    if left is None:return right
    result={}
    for layer,x in left.items():
        y=right[layer];n=x['N']+y['N']
        result[layer]=dict(A=(x['A']*x['N']+y['A']*y['N'])/n,
                           B=(x['B']*x['N']+y['B']*y['N'])/n,N=n)
    return result
