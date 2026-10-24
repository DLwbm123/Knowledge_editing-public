"""Fixed uniform label smoothing on existing causal answer positions only."""
import torch

def components(logits,labels):
 target=labels[:,1:].reshape(-1);mask=target!=-100
 assert mask.any()
 z=logits[:,:-1].reshape(-1,logits.shape[-1])[mask].float();t=target[mask]
 assert torch.isfinite(z).all() and z.shape[-1]>1
 lp=z.log_softmax(-1)
 nll=-lp.gather(1,t[:,None]).mean();uniform=-lp.mean()
 loss=.9*nll+.1*uniform
 assert torch.isfinite(loss)
 return loss,nll,uniform

def selfcheck():
 x=torch.tensor([[[1.,2.,3.,4.],[2.,4.,1.,3.],[0.,1.,3.,2.]]],requires_grad=True)
 y=torch.tensor([[-100,2,1]])
 loss,nll,u=components(x,y)
 expected=torch.nn.functional.cross_entropy(x[:,:-1].reshape(-1,4),y[:,1:].reshape(-1),label_smoothing=.1)
 torch.testing.assert_close(loss,expected,rtol=1e-6,atol=1e-6)
 g=torch.autograd.grad(loss,x)[0];target=torch.full((2,4),.1/4);target[0,2]+=.9;target[1,1]+=.9
 torch.testing.assert_close(g[0,:2],(x[0,:2].softmax(-1)-target)/2)
 assert g[0,2].count_nonzero()==0
 masked=y.clone();masked[0,1]=-100
 a=components(x,masked)[0];b=x.detach().clone();b[0,0]+=99;b[0,2]-=33
 assert a.item()==components(b,masked)[0].item()
 return dict(status='PASS',fixed_epsilon=.1,causal_mask=True,masked_positions_ignored=True,analytic_gradient=True,torch_cross_entropy_parity=True,real_model_calls=0)
