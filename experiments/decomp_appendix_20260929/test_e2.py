"""CPU expansion and bank-dispatch regression check; no dense matrix."""
import torch
from e2 import structure,deploy,jobs
from structures import TuckerC4,LR4
from methods.medtrace import AsymmetricCPExpert

def main():
 q=jobs();assert len(q)==52 and sum(j.get('e2_canary',False) for j in q)==2
 assert sum(len(j.get('methods',[])) for j in q)==96
 assert all('E2_CANARY_PASS.json' in j['requires'] for j in q if not j.get('e2_canary'))
 torch.manual_seed(41)
 for method,expert,kind in [('M1_STRUCT',AsymmetricCPExpert(14336,4096,4),'CP'),('M4_STRUCT',TuckerC4(),'TK')]:
  with torch.no_grad():
   for p in expert.parameters():p.add_(torch.randn_like(p)*.01)
  free=deploy(expert,method,42);h=torch.randn(3,14336,requires_grad=True)
  a=expert.residual(h);b=free.residual(h)
  assert structure(method)==kind and isinstance(free,LR4)
  torch.testing.assert_close(a,b,rtol=1e-4,atol=1e-5)
  ga=torch.autograd.grad(a.square().sum(),h,retain_graph=True)[0];gb=torch.autograd.grad(b.square().sum(),h)[0]
  torch.testing.assert_close(ga,gb,rtol=2e-4,atol=1e-5)
  assert deploy(free,method.split('_')[0],42) is free
 print('PASS: CP/Tucker nonzero residual and input-gradient expansion; STRUCT bank mapping')
if __name__=='__main__':main()
