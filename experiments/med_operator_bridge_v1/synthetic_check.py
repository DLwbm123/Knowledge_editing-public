import torch
from torch import nn
from bridge import OperatorBridge

class Toy(nn.Module):
    def __init__(self,scale):
        super().__init__()
        self.register_buffer('scale',torch.tensor(scale))
    def residual(self,x):return x*self.scale

x=torch.tensor([[1.,-2.,3.]])
a,b=Toy(2.),Toy(-4.)
assert torch.equal(OperatorBridge(a,b,0).residual(x),a.residual(x))
assert torch.equal(OperatorBridge(a,b,1).residual(x),b.residual(x))
assert torch.equal(OperatorBridge(a,b).residual(x),0.5*(a.residual(x)+b.residual(x)))
try:OperatorBridge(a,b,0.6)
except ValueError:pass
else:raise AssertionError('unfrozen coefficient accepted')
try:OperatorBridge(a,Toy(float('nan'))).residual(x)
except ValueError:pass
else:raise AssertionError('nonfinite residual accepted')
print('SYNTHETIC_OPERATOR_PASS')
