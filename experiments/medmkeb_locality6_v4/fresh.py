"""Training-only Fisher and current-task reference; never read historical Fisher/curves."""
import math
import copy
import torch

def source_step(r,before,gradient,scalar):
    parameters={k:v.detach().clone().requires_grad_(True) for k,v in before.items()}
    for k,v in parameters.items():v.grad=gradient[k].detach().clone()
    norm=torch.nn.utils.clip_grad_norm_(list(parameters.values()),1.)
    assert torch.isfinite(norm)
    proposed,updated=r.mm.canonical_rms_step(before,{k:v.grad.detach() for k,v in parameters.items()},copy.deepcopy(scalar))
    selected=r.mm.canonicalize(proposed)[0]
    target=r.fm.map_norm(before,selected);assert math.isfinite(target) and target>0
    return selected,updated,target

def fisher(runtime,r,expert,hook,batches,counts,budget):
    assert len(batches)==8
    capture={};xs=[];ys=[];token_counts=[]
    def tap(_module,args,output):
        x=args[0].detach()[hook.token_mask].to(next(expert.parameters()).dtype)
        capture['x']=x/(x.square().mean(-1,keepdim=True).sqrt()+expert.epsilon)
        capture['output']=output
    handle=runtime.get_module(r.c.LAYER).register_forward_hook(tap)
    from fisher_math import categorical_probe
    try:
        for index,batch in enumerate(batches):
            budget();hook.set_teacher_routing(batch.labels);output=runtime.model(**batch.forward_kwargs())
            mask=batch.labels[:,1:]!=-100;logits=output.logits[:,:-1][mask].float();probability=logits.detach().softmax(-1);n=len(logits)
            generator=torch.Generator(device=logits.device).manual_seed(20261009+index);vectors=[]
            for sample in range(32):
                budget();signs=torch.randint(0,2,logits.shape,device=logits.device,dtype=torch.int8,generator=generator).float()*2-1
                vector=categorical_probe(probability,signs);assert vector.sum(-1).abs().max()<=2e-6
                value=(logits*vector).sum()/math.sqrt(8*32*n)
                params=list(expert.parameters()) if index==sample==0 else []
                gradients=torch.autograd.grad(value,[capture['output']]+params,retain_graph=sample!=31)
                y=gradients[0].detach()[hook.token_mask];counts['backwards']+=1
                if params:
                    b,a=expert.factors();proxy=((capture['x']@a.T)*(y@b)).sum();pulled=torch.autograd.grad(proxy,params)
                    for direct,composed in zip(gradients[1:],pulled):torch.testing.assert_close(direct,composed,rtol=1e-4,atol=1e-7)
                vectors.append(y.cpu())
            xs.append(capture['x'].cpu());ys.append(torch.stack(vectors,1));token_counts.append(n);capture.clear()
            del output,logits,probability,gradients,value,vector,signs
        result=dict(x=torch.cat(xs),y=torch.cat(ys),token_counts=token_counts)
        assert all(torch.isfinite(result[k]).all() for k in ('x','y'))
        assert len(token_counts)==8 and result['y'].shape[1]==32
        return result
    finally:handle.remove()
