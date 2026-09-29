import torch
from types import SimpleNamespace
from mechanisms import ResidualCapture,scope_gate
from protection import protector
from calibration import calibrate

def main():
 torch.manual_seed(3)
 expert=torch.nn.Module();expert.A=torch.nn.Parameter(torch.randn(2,4));expert.B=torch.nn.Parameter(torch.randn(3,2))
 class Model(torch.nn.Module):
  def __init__(self):super().__init__();self.layer=torch.nn.Linear(4,3);self.layer.requires_grad_(False);self.calls=0
  def forward(self,h):
   self.calls+=1;z=h/(h.square().mean(-1,keepdim=True).sqrt()+1e-6)
   return SimpleNamespace(logits=self.layer(h)+(z@expert.A.T)@expert.B.T)
 model=Model();runtime=SimpleNamespace(model=model);hook=SimpleNamespace(set_teacher_routing=lambda labels:None)
 h=torch.randn(1,3,4);mask=torch.tensor([[False,True,True]]);row=({'h':h},torch.ones(1,3),mask,torch.zeros(2,3));before=[p.detach().clone() for p in model.parameters()]
 callback=protector(runtime,model.layer,expert,[row],[row],[row],[row],'AHS',1,lambda x,y:(x-y).square().mean(),torch.tensor(3.),.01)
 result=callback(hook,2);assert model.calls==2 and result['extra_residual_forward_calls']==0
 assert all(torch.equal(x,p) and p.grad is None for x,p in zip(before,model.parameters()))
 assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in expert.parameters())
 assert calibrate([],['e'])['status']=='UNSUPPORTED_CAL_SCOPE'
 try:scope_gate(torch.tensor([float('nan'),0]),'e',True,torch.zeros(1,2),torch.ones(1,2),0);raise AssertionError()
 except ValueError:pass
 print('PASS: protection reuses two KL forwards, Base unchanged/no gradients, expert finite gradients, calibration fail-closed')
if __name__=='__main__':main()
