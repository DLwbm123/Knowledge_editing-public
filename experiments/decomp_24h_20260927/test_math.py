"""CPU checks against actual imported scientific implementations, no medical data."""
import os,sys,json
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
r=Path(os.environ['RUN_ROOT']);sys.path[:0]=[str(r/'source_patch'),str(r/'source')]
import torch
from methods.medtrace import AsymmetricCPExpert
from methods.medtrace.selective_write import full_vocab_kl,predictor_mask
from structures import TuckerC4,LR4,convert

def main():
 torch.set_num_threads(2);torch.manual_seed(42);results={}
 for name,ex in [('CP',AsymmetricCPExpert(14336,4096,4)),('TK',TuckerC4()),('LR',LR4())]:
  h=torch.randn(3,14336);target=torch.randn(3,4096);opt=torch.optim.Adam(ex.parameters(),lr=.001)
  assert torch.count_nonzero(ex.residual(h))==0
  norms=[]
  for step in range(3):
   opt.zero_grad();(ex.residual(h)-target).square().mean().backward();norms.append({k:float(p.grad.norm()) for k,p in ex.named_parameters()});opt.step();ex.normalize_factors_(verify_dense=False)
  assert any(x>0 for x in norms[0].values()) and all(x>0 for x in norms[-1].values())
  lr=convert(ex,9);y=ex.residual(h);z=lr.residual(h);err=float((y-z).abs().max());assert torch.allclose(y,z,atol=1e-6,rtol=1e-5)
  lr8=convert(ex,9,8);assert torch.allclose(y,lr8.residual(h),atol=1e-6,rtol=1e-5)
  results[name]=dict(parameters=sum(p.numel() for p in ex.parameters()),conversion_max_abs=err,gradient_norms=norms)
 teacher=torch.randn(5,11).log_softmax(-1);student=torch.randn(5,11,requires_grad=True)
 kl=full_vocab_kl(student,teacher,chunk=2);ref=(teacher.exp()*(teacher-student.log_softmax(-1))).sum(-1).mean();assert torch.allclose(kl,ref,atol=1e-7)
 grad=torch.autograd.grad(kl,student)[0];expected=(student.softmax(-1)-teacher.exp())/5;assert torch.allclose(grad,expected,atol=1e-7)
 labels=torch.tensor([[-100,-100,3,4,2,-100]]);mask=predictor_mask(labels);assert mask.tolist()==[[False,True,True,True,False,False]]
 results['KL']=dict(direction='teacher||student',normalization='token mean, full vocabulary',value=float(kl.detach()),gradient_max_abs=float((grad-expected).abs().max()),shift_and_EOS=True)
 # The frozen uniform sampler estimates .5 old + .5 mean(new), independent of support count.
 import random
 values=[1.,3.,5.,7.];samples=[values[random.Random(17*1000003+i).randrange(4)] for i in range(1,10001)];assert abs(sum(samples)/len(samples)-4)<.1
 results['sampler']=dict(mean=sum(samples)/len(samples),expected=4.)
 from training import protector,teacher_bytes
 from types import SimpleNamespace
 parameter=torch.randn(1,3,11,requires_grad=True)
 runtime=SimpleNamespace(model=lambda **kw:SimpleNamespace(logits=parameter+kw['offset']))
 hook=SimpleNamespace(set_teacher_routing=lambda labels:None)
 def row(offset):return ({'offset':torch.tensor(offset)},torch.tensor([[-100,3,2]]),torch.ones(1,3,dtype=torch.bool),torch.randn(3,11).log_softmax(-1))
 old_u=[row(.1),row(.2)];new_u=[row(.3+i) for i in range(4)];kd=[row(.4),row(.5)];step=13;seed=17
 terms=protector(runtime,[old_u,new_u],'M3',seed,kd)(hook,step);actual=parameter.grad.clone();parameter.grad=None
 selected=new_u[random.Random(seed*1000003+step).randrange(4)];selected_kd=kd[random.Random(seed*1000033+step).randrange(2)]
 def term(r):return full_vocab_kl((parameter+r[0]['offset'])[r[2]],r[3])
 reference=.005*sum(term(r) for r in old_u)/len(old_u)+.005*term(selected)+.10*term(selected_kd);reference.backward()
 err=float((actual-parameter.grad).abs().max());assert err<1e-7
 assert abs(terms['U_KL']-float((.5*sum(term(r) for r in old_u)/len(old_u)+.5*term(selected)).detach()))<1e-6
 results['callback']=dict(gradient_max_abs=err,weighted_KL_log_matches=True,total_teacher_tensor_bytes=teacher_bytes([old_u,new_u,kd]))
 print(json.dumps(dict(status='PASS',results=results)))
if __name__=='__main__':main()
