"""Small CPU checks for the new protection contract and missing-score semantics."""
import importlib.util
import inspect
import sys
from pathlib import Path
import torch
from methods.medtrace.selective_write import full_vocab_kl,predictor_mask
from scripts.medtrace import stage15,run_dev16,run_generality_ablation

for f in [stage15.initialize,stage15.train_steps,run_dev16.run_event,run_generality_ablation.train_condition]:
    assert 'protection' in inspect.signature(f).parameters
labels=torch.tensor([[-100,-100,4,2]])
assert predictor_mask(labels).tolist()==[[False,True,True,False]]
s=torch.tensor([[.1,-.3,1.],[.3,.1,-.7]],requires_grad=True)
q=torch.tensor([[.2,.3,.5],[.5,.2,.3]])
a=full_vocab_kl(s,q.log(),chunk=1)
expected=(q*(q.log()-s.log_softmax(-1))).sum()/2
assert torch.allclose(a,expected,atol=1e-7)
a.backward();assert torch.isfinite(s.grad).all() and s.grad.norm()>0
s2=s.detach().clone().requires_grad_(True)
mean=sum(full_vocab_kl(s2,q.log()) for _ in range(8))/8
mean.backward();assert torch.allclose(s.grad,s2.grad,atol=1e-7)
print('PASS: callbacks, causal shift, Base||student KL, U mean normalization, live gradient')
