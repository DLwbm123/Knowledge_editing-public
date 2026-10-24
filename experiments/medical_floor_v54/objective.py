"""Fixed role CE plus Base-relative target-answer log-probability constraints."""
import torch

def target_score(logits, labels, eos):
    y=labels[:,1:];mask=(y!=-100)&(y!=eos)
    assert mask.any(), 'Non-EOS answer tokens required'
    return logits[:,:-1].log_softmax(-1)[mask].gather(-1,y[mask].unsqueeze(-1)).mean()

def combine(ce, ce_grad, positive, positive_grad, negative, negative_grad, base, enabled):
    dp=positive-base[0];dn=negative-base[1]
    pair=max(0.,.1-dp+dn);upper=max(0.,dn);positive_floor=max(0.,.1-dp)
    gradient=ce_grad.clone()
    if enabled:
        if .1-dp+dn>0:gradient=gradient-positive_grad+negative_grad
        if dn>0:gradient=gradient+negative_grad
        if .1-dp>0:gradient=gradient-positive_grad
    return ce+(pair+upper+positive_floor if enabled else 0.),gradient,dict(delta_positive=dp,delta_negative=dn,pair_hinge=pair,negative_hinge=upper,positive_hinge=positive_floor)

def selfcheck():
    x=torch.tensor([.2,-.4],dtype=torch.float64,requires_grad=True)
    for base in [(0.,0.),(-1.,-2.),(1.,1.)]:
        ce=x.square().mean();p=x[0]+2*x[1];n=2*x[0]-x[1]
        gs=[torch.autograd.grad(v,x,retain_graph=True)[0] for v in (ce,p,n)]
        for enabled in [False,True]:
            loss=ce+int(enabled)*(torch.relu(.1-(p-base[0])+(n-base[1]))+torch.relu(n-base[1])+torch.relu(.1-(p-base[0])))
            value,g,_=combine(float(ce.detach()),gs[0],float(p.detach()),gs[1],float(n.detach()),gs[2],base,enabled)
            assert abs(value-float(loss.detach()))<1e-12 and torch.allclose(g,torch.autograd.grad(loss,x,retain_graph=True)[0],atol=1e-12,rtol=0)
    logits=torch.zeros(1,4,3,dtype=torch.float64,requires_grad=True);labels=torch.tensor([[-100,1,2,-100]])
    score=target_score(logits,labels,2);g=torch.autograd.grad(score,logits)[0]
    assert abs(float(score.detach())+__import__('math').log(3))<1e-12
    assert g[0,0,1]>0 and g[0,1:].count_nonzero()==0
    _,zero,stats=combine(0.,torch.zeros(2),0.,torch.tensor([1.,0.]),0.,torch.tensor([0.,1.]),(0.,0.),True)
    assert torch.equal(zero,torch.tensor([-2.,1.])) and stats['pair_hinge']==.1 and stats['negative_hinge']==0
    _,_,gap=combine(0.,torch.zeros(2),-.5,torch.tensor([1.,0.]),-1.9,torch.tensor([0.,1.]),(0.,0.),True)
    assert gap['pair_hinge']==0 and gap['negative_hinge']==0 and abs(gap['positive_hinge']-.6)<1e-12
    return dict(positive_decline_loophole_detected=True,status='PASS',analytic_gradient=True,causal_shift_EOS_mask=True,zero_increment_margin=True,CE_zero_constraint=True)
