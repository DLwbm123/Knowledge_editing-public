"""One fixed-initial-point probe; no optimizer updates or generated answers."""
import json
import math
from pathlib import Path
import torch
from .coefficient_runtime import coefficient_forward, compose
from .dual_solver import source_basis, pair_objective
from .dual_execution import ExecutionBudget, Journal
from .editor import nll, editable_fp32


def diagnose(runtime, layer, record, pair, factor, config, manifest, root):
    model = runtime.model
    budget = ExecutionBudget(config, manifest['start_epoch'])
    journal = Journal(root, config['disk_bytes_max'])
    base = layer.weight.detach().clone()
    hook = model.register_forward_pre_hook(budget.before_forward)
    stats = []
    baseline = None
    def summary(x):
        finite = torch.isfinite(x)
        return dict(finite=bool(finite.all()), nonfinite=int((~finite).sum()),
                    max_abs=float(x[finite].abs().max()) if finite.any() else None)
    try:
        with torch.no_grad(): baseline = [float(nll(model, b)) for b in pair]
        with editable_fp32(model, layer, readonly=True):
            gradient = budget.grad(nll(model, pair[0]), (layer.weight,), 1)[0].detach()
        a,b = (factor[k].to(base.device).float() for k in ('a','b'))
        u,v = source_basis(gradient,a,b,config)
        del gradient
        zs=[]; losses=[]; incoming=[]
        for sign in config['init_Z_E11']:
            z=torch.zeros((4,4),device=base.device);z[0,0]=sign;z.requires_grad_(True);zs.append(z)
            coefficient=z*math.sqrt(2*config['trust_budget'])
            candidate=compose(base,u,coefficient,v)
            stats.append(dict(delta_norm=float((candidate-base.float()).norm()),base_norm=float(base.float().norm()),candidate_max=float(candidate.abs().max())))
            with coefficient_forward(model,layer,u,coefficient,v):
                def observe(module,args,out):
                    out.register_hook(lambda g: incoming.append(summary(g)))
                h=layer.register_forward_hook(observe)
                try:losses.append([nll(model,batch) for batch in pair])
                finally:h.remove()
        objective=pair_objective(losses,zs,baseline,'IND2',config)
        flat=[x for row in losses for x in row]
        adjoints=torch.autograd.grad(objective,flat,retain_graph=True)
        scale=1/max(1.,max(abs(float(x)) for x in adjoints))
        journal.private('INITIAL.json',dict(baseline=baseline,losses=[[float(x) for x in row] for row in losses],objective=float(objective),adjoints=[float(x) for x in adjoints],scale=scale,physical=stats,factor_a=summary(a),factor_b=summary(b)))
        for label,multiplier in [('unscaled',1.),('normalized_adjoint',scale)]:
            budget.check_time();budget.backwards+=1;budget.gradient_paths+=4
            incoming.clear()
            grads=torch.autograd.grad(objective*multiplier,zs,retain_graph=label=='unscaled')
            journal.private(label+'.json',dict(incoming=list(incoming),gradients=[summary(g) for g in grads],unscaled_gradients=[summary(g/multiplier) for g in grads]))
    finally:
        budget.restoring=True
        with torch.no_grad():
            layer.weight.copy_(base)
            if baseline is not None:
                actual=[float(nll(model,b)) for b in pair]
                assert all(abs(x-y)<=1e-5 for x,y in zip(actual,baseline))
        hook.remove()
        journal.event(dict(status='RESTORED',forwards=budget.forwards,backward_API=budget.backwards,model_gradient_paths=budget.gradient_paths))
