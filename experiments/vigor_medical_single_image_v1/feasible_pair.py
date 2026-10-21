"""Paired coefficient descent with actual native feasibility at every accepted step."""
import math
import torch
from .coefficient_runtime import coefficient_forward, compose, validate
from .dual_solver import source_basis, endpoint_native, fisher_cost
from .editor import editable_fp32, finite, nll


def tangent_direction(gradient, normal):
    # Projection happens only in the 16-dimensional coefficient space.
    normal = normal.double()
    direction = -gradient.double()
    square = normal.square().sum()
    if float(square) > 0:
        direction -= normal * (normal.flatten() @ direction.flatten()) / square
    direction /= max(float(direction.norm()), 1.)
    return finite(direction.float())


def merit(values, zs, baseline, arm, config):
    value = sum(v[0] for v in values) / max(baseline[0], config['source_normalizer_floor'])
    if arm == 'JOINT2':
        distance = float((zs[0]-zs[1]).norm())
        value += config['joint_diversity_penalty'] * max(0., (config['diversity_min']-distance)/config['diversity_min'])**2
    return value


def solve(runtime, layer, pair, factor, config, admission, progress, *, budget):
    if config['version'] != 'PUREW_FEASIBLE_PAIR_V1' or not all(admission.values()):
        raise PermissionError('Frozen feasible-pair admission required')
    validate(runtime.model, layer)
    model = runtime.model
    base = layer.weight.detach().clone()
    modes = [(m,m.training) for m in model.modules()]
    flags = [(p,p.requires_grad) for p in model.parameters()]
    previous_restoring = budget.restoring
    result = {'arms': {}}
    baseline = None
    try:
        model.eval()
        with torch.no_grad(): baseline = [float(nll(model,b)) for b in pair]
        with editable_fp32(model,layer,readonly=True):
            gradient = finite(budget.grad(nll(model,pair[0]),(layer.weight,),1)[0].detach())
        a,b = (factor[k].to(base.device).float() for k in ('a','b'))
        u,v = source_basis(gradient,a,b,config)
        del gradient
        def evaluate(z):
            with endpoint_native(model,layer,base,u,v,z,config), torch.no_grad():
                losses = [float(nll(model,batch)) for batch in pair]
                delta = layer.weight.float()-base.float()
                cost = fisher_cost(delta,a,b)
                changed = bool(torch.count_nonzero(delta))
            return dict(native_nll=losses,fisher_cost=cost,changed=changed,
                protection_feasible=abs(losses[1]-baseline[1])<=config['null_abs_budget'] and cost<=config['trust_budget'],
                numeric_feasible=losses[0]<=config['source_NLL_target'] and abs(losses[1]-baseline[1])<=config['null_abs_budget'] and cost<=config['trust_budget'])
        seeds = []
        for sign in config['init_Z_E11']:
            z=torch.zeros((config['rank'],config['rank']),device=base.device)
            z[0,0]=sign;seeds.append(z)
        for trial in range(config['seed_backtracks']):
            current=[z*(.5**trial) for z in seeds]
            initial=[evaluate(z) for z in current]
            first=compose(base,u,current[0]*math.sqrt(2*config['trust_budget']),v).half()
            second=compose(base,u,current[1]*math.sqrt(2*config['trust_budget']),v).half()
            distinct=not torch.equal(first,second)
            progress(dict(status='SEED_ACCEPT' if all(x['protection_feasible'] and x['changed'] for x in initial) and distinct else 'SEED_REJECT',step=trial,pre_update_losses=[x['native_nll'] for x in initial],forwards=budget.forwards))
            del first,second
            if all(x['protection_feasible'] and x['changed'] for x in initial) and distinct:break
        else:raise RuntimeError('No admissible common seed pair; no replacement')
        for arm in config['arms']:
            zs=[z.clone() for z in current]
            values=initial
            trace=[]
            for step in range(config['steps']):
                zs=[z.detach().requires_grad_(True) for z in zs]
                direct=[torch.zeros_like(z) for z in zs]
                if arm=='JOINT2':
                    penalty=config['joint_diversity_penalty']*torch.relu((config['diversity_min']-(zs[0]-zs[1]).norm())/config['diversity_min']).square()
                    direct=list(torch.autograd.grad(penalty,zs))
                directions=[]
                for i,z in enumerate(zs):
                    with coefficient_forward(model,layer,u,z*math.sqrt(2*config['trust_budget']),v):
                        source=nll(model,pair[0])
                        gs=finite(budget.grad(source,(z,),1)[0].detach())
                    # Separate coefficient contexts avoid a shared freed graph.
                    with coefficient_forward(model,layer,u,z*math.sqrt(2*config['trust_budget']),v):
                        null=nll(model,pair[1])
                        gn=finite(budget.grad(null,(z,),1)[0].detach())
                    directions.append(tangent_direction(gs/max(baseline[0],config['source_normalizer_floor'])+direct[i],gn))
                old=merit([x['native_nll'] for x in values],zs,baseline,arm,config)
                accepted=False
                for trial in range(config['step_backtracks']):
                    candidates=[(z+config['lr']*(.5**trial)*d).detach() for z,d in zip(zs,directions)]
                    proposed=[evaluate(z) for z in candidates]
                    new=merit([x['native_nll'] for x in proposed],candidates,baseline,arm,config)
                    accepted=all(x['protection_feasible'] for x in proposed) and all(x['native_nll'][0]<=oldx['native_nll'][0] for x,oldx in zip(proposed,values)) and new<old
                    progress(dict(arm=arm,step=step+1,endpoint=trial,status='STEP_ACCEPT' if accepted else 'STEP_REJECT',pre_update_losses=[x['native_nll'] for x in proposed],objective=new,forwards=budget.forwards,backward_API=budget.backwards,model_gradient_paths=budget.gradient_paths))
                    if accepted:
                        zs=candidates;values=proposed;break
                trace.append(dict(step=step+1,accepted=accepted,backtracks=trial+1))
                if not accepted:break
            first=compose(base,u,zs[0].detach()*math.sqrt(2*config['trust_budget']),v).half()
            second=compose(base,u,zs[1].detach()*math.sqrt(2*config['trust_budget']),v).half()
            distance=math.sqrt(fisher_cost(first.float()-second.float(),a,b)/config['trust_budget'])
            result['arms'][arm]=dict(trace=trace,finals=values,native_distinct=not torch.equal(first,second),native_distance=distance,numeric_pair_pass=all(x['numeric_feasible'] for x in values) and not torch.equal(first,second) and distance>=config['diversity_min'],coefficients_private=[z.detach().clone() for z in zs])
            del first,second
        result.update(baseline=baseline,basis_private=(u,v))
        return result
    finally:
        budget.restoring=True
        try:
            with torch.no_grad():
                layer.weight.copy_(base)
                if baseline is not None:
                    actual=[float(nll(model,b)) for b in pair]
                    if any(abs(x-y)>1e-5 for x,y in zip(actual,baseline)):raise RuntimeError('Feasible-pair W0 restoration failed')
        finally:
            for p,flag in flags:p.requires_grad_(flag)
            for m,mode in modes:m.training=mode
            budget.restoring=previous_restoring
