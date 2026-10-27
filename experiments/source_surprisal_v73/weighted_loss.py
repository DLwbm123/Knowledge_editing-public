"""Fixed Base-surprisal weights on legal source-answer tokens only."""
import torch
import torch.nn.functional as F

def token_nll(logits, labels):
 assert logits.shape[:2]==labels.shape and labels.shape[0]==1
 target=labels[:,1:].reshape(-1);mask=target!=-100
 assert mask.any()
 return F.cross_entropy(logits[:,:-1].float().reshape(-1,logits.shape[-1])[mask],target[mask],reduction='none')

def weights(nll):
 x=nll.detach().cpu().double()
 assert torch.isfinite(x).all() and (x>=0).all() and x.sum()>0
 return x/x.mean()

def loss(logits,labels,weight):
 nll=token_nll(logits,labels);w=weight.to(device=nll.device,dtype=nll.dtype)
 assert w.shape==nll.shape and torch.isfinite(w).all() and (w>=0).all() and w.sum()>0
 return (nll*w).sum()/w.sum()

def selfcheck():
 logits=torch.tensor([[[1.,2.,3.],[2.,1.,0.],[0.,1.,3.],[1.,0.,0.]]],requires_grad=True)
 labels=torch.tensor([[-100,-100,2,0]])
 n=token_nll(logits,labels);assert n.numel()==2
 expected=F.cross_entropy(logits[:,:-1].reshape(-1,3),labels[:,1:].reshape(-1),ignore_index=-100)
 uniform=loss(logits,labels,torch.ones(2));torch.testing.assert_close(uniform,expected)
 g=torch.autograd.grad(uniform,logits,retain_graph=True)[0];h=torch.autograd.grad(expected,logits,retain_graph=True)[0];assert torch.equal(g,h)
 w=weights(n);assert not w.requires_grad and w.dtype==torch.float64
 torch.testing.assert_close(w.mean(),torch.ones((),dtype=torch.float64))
 torch.testing.assert_close(loss(logits,labels,w),loss(logits,labels,3*w))
 assert torch.isfinite(torch.autograd.grad(loss(logits,labels,w),logits)[0]).all()
 for bad in [torch.tensor([float('nan'),1.]),torch.zeros(2),torch.tensor([-1.,2.])]:
  try:weights(bad)
  except AssertionError:pass
  else:raise AssertionError('Invalid weights admitted')
 return dict(status='PASS',causal_mask=True,uniform_loss_gradient_equivalence=True,detached_weights=True,normalized_mean=True,scale_invariance=True,invalid_rejected=True,real_model_calls=0)
