import torch
from torch import nn
from train import TrainableBridge
class Fixed(nn.Module):
 def __init__(self,value):super().__init__();self.v=nn.Parameter(torch.tensor(value))
 def residual(self,x):return self.v*x
m=TrainableBridge(Fixed(0.0),Fixed(2.0));x=torch.tensor([1.,2.]);before=m.logit.detach().clone()
loss=(m.residual(x)-torch.tensor([2.,4.])).square().mean();loss.backward()
assert m.logit.grad is not None and m.logit.grad.item()<0
assert all(p.grad is None for n,p in m.named_parameters() if n!='logit')
torch.optim.Adam([m.logit],lr=.01).step();assert not torch.equal(before,m.logit)
print('TRAINABLE_BRIDGE_SYNTHETIC_PASS')
