"""Current-frame candidate identity and diagnostic comparisons, no model dependency."""
import torch

def direct(before,delta,scale,keys):
    if scale==0 or not bool(delta.any()):return {k:v.clone() for k,v in before.items()}
    out={};offset=0
    for k in keys:
        n=before[k].numel();out[k]=before[k]+scale*delta[offset:offset+n].reshape(before[k].shape).to(before[k]);offset+=n
    assert offset==delta.numel()
    return out

def distance(a,b):return max(abs(x-y) for x,y in zip([a['source_loss']]+a['support_nll'],[b['source_loss']]+b['support_nll']))

def selfcheck():
    before={'x':torch.tensor([1.,2.],dtype=torch.float32)};d=torch.tensor([.2,-.1],dtype=torch.float64)
    assert torch.equal(direct(before,d,0.,['x'])['x'],before['x'])
    assert torch.equal(direct(before,torch.zeros_like(d),1.,['x'])['x'],before['x'])
    state={k:v.double().requires_grad_() for k,v in before.items()};loss=state['x'].square().sum();g=torch.autograd.grad(loss,state['x'])[0]
    step=1e-5;actual=((state['x'].detach()+step*d).square().sum()-loss.detach())/step
    assert abs(float(actual-g@d))<1e-5
    return dict(status='PASS',zero_scale_exact=True,zero_direction_exact=True,finite_difference_fixed_frame=True,new_language_calls=0)

def capped(before,delta,scale,keys,limit,norm):
    state=direct(before,delta,scale,keys)
    if norm(before,state)<=limit:return state,scale
    lo,hi=0.,scale;safe=direct(before,delta,0.,keys)
    for _ in range(32):
        mid=(lo+hi)/2;trial=direct(before,delta,mid,keys)
        if norm(before,trial)<=limit:lo,safe=mid,trial
        else:hi=mid
    assert norm(before,safe)<=limit
    return safe,lo
