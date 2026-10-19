"""One fixed-competitor margin proposal; no search and no alternate scales."""
import math
import torch
from .editor import direction,editable_fp32,nll,finite
from .t0probe import token_stat


def fisher_cost(delta,a,b):
    return float((delta.double().square()*b.double()[:,None]*a.double()[None,:]).sum())/2


def one_match(original,tangent,energy,target,a,b):
    """One analytic seed, one native-cost ratio, one rebuilt final proposal."""
    if not math.isfinite(target) or target<=0 or not math.isfinite(energy) or energy<=0:raise ValueError('Undefined matching target/energy')
    seed=math.sqrt(2*target/energy)
    trial=(original.float()-seed*tangent).half().float()-original.float()
    cost=fisher_cost(trial,a,b)
    if not math.isfinite(cost) or cost<=0:raise ValueError('Zero/nonfinite native seed')
    ratio=math.sqrt(target/cost)
    final=(original.float()-seed*ratio*tangent).half().float()-original.float()
    actual=fisher_cost(final,a,b);error=abs(actual/target-1)
    return seed*ratio,dict(target_native_F0_cost=target,seed_scale=seed,seed_native_F0_cost=cost,ratio=ratio,ratio_applications=1,actual_native_F0_cost=actual,relative_error=error,tolerance=.01,passed=math.isfinite(error) and error<=.01)


def read_margin(runtime,batch,position=None,competitor=None):
    with torch.no_grad():
        output=runtime.model(**batch.forward_kwargs());labels=batch.labels[:,1:];mask=labels[0]!=-100
        positions=torch.where(mask)[0];targets=labels[0,positions];vectors=output.logits[0,positions].float()
        eos=runtime.adapter.tokenizer.eos_token_id
        if position is None:
            unchosen=[i for i in range(len(targets)) if int(targets[i])!=eos and int(vectors[i].argmax())!=int(targets[i])]
            if not unchosen:raise ValueError('No eligible unfavored content token')
            position=unchosen[0]
        target=int(targets[position]);v=vectors[position];others=v.detach().clone();others[target]=-torch.inf;j=int(others.argmax())
        mean=nll_from_logits(output.logits,batch.labels)
        data=dict(target_position=position+1,mean_nll=mean,margin=float(v[target]-v[j]),**token_stat(v,target))
        if competitor is not None:data['fixed_competitor_margin']=float(v[target]-v[competitor]);data['strongest_competitor_changed']=j!=competitor
        return data,(position,target,j,int(positions[position]))


def nll_from_logits(logits,labels):
    return float(torch.nn.functional.cross_entropy(logits[:,:-1].float().reshape(-1,logits.shape[-1]),labels[:,1:].reshape(-1),ignore_index=-100))


def propose(runtime,layer,pair,factor,identity,target_cost,save):
    original=layer.weight.detach().clone();a,b=(factor[k].to(layer.weight.device) for k in ('a','b'));pos,target,j,absolute=identity
    with editable_fp32(runtime.model,layer):
        output=runtime.model(**pair[0].forward_kwargs());v=output.logits[0,absolute].float()
        loss=v[j]-v[target];gm=finite(torch.autograd.grad(loss,layer.weight)[0].detach());loss_value=float(loss.detach());del output,v,loss
        null_loss=nll(runtime.model,pair[1]);g0=finite(torch.autograd.grad(null_loss,layer.weight)[0].detach());null_value=float(null_loss.detach());del null_loss
        tangent,geometry=direction(gm,g0,a,b)
        data=dict(status='PROPOSING',fixed_competitor_loss=loss_value,fp32_null_nll=null_value,geometry=geometry)
        save(data)
        if tangent is None:raise ValueError('Degenerate margin direction')
        scale,matching=one_match(original,tangent,geometry['E'],target_cost,a,b)
        data['matching']=matching;data['status']='MATCHED' if matching['passed'] else 'MATCH_FAILED';save(data)
        if not matching['passed']:raise ValueError('One-shot native F0 cost matching failed')
        with torch.no_grad():layer.weight.copy_(finite(original.float()-scale*tangent))
    actual=layer.weight.detach().float()-original.float();data['actual_native_F0_cost']=fisher_cost(actual,a,b)
    data['committed_matches_preview']=abs(data['actual_native_F0_cost']/matching['actual_native_F0_cost']-1)<=1e-10;save(data)
    if not data['committed_matches_preview']:raise ValueError('Native commit differs from matching preview')
    return data
