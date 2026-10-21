"""Explicit target-layer integration and legacy-default compatibility, CPU only."""
import torch
from types import SimpleNamespace
from unittest.mock import patch
from .cpu_check import Runtime,record
from .editor import editable_fp32,paired_batches
from .finite_null import gates,correction,native_losses
from .nll_control import run_checks as controller_checks

def run_checks():
 torch.manual_seed(42);rt=Runtime(torch.float16);rt.model.extra=torch.nn.Linear(4,6,bias=False).half()
 def forward(inputs_embeds,**kwargs):
  x=torch.tanh(inputs_embeds.cumsum(1)*rt.model.frozen)
  return SimpleNamespace(logits=rt.model.layer(x)+rt.model.extra(x))
 rt.model.forward=forward;seen=[];zero='model.layers.0.mlp.down_proj';six='model.layers.6.mlp.down_proj'
 def get_module(name):
  seen.append(name)
  return {zero:rt.model.layer,six:rt.model.extra}[name]
 rt.get_module=get_module;z=rt.model.layer;s=rt.model.extra;z0=z.weight.detach().clone();s0=s.weight.detach().clone()
 pair=paired_batches(rt,record('source'));before=native_losses(rt,pair);factor=dict(a=torch.ones(4),b=torch.ones(6))
 with editable_fp32(rt.model,s):
  with torch.no_grad():s.weight.add_(.01)
 assert s.weight.dtype==s0.dtype and not torch.equal(s.weight,s0) and torch.equal(z.weight,z0)
 seen.clear();v=gates(rt,pair,factor,s0,before,before[1],module_name=six)
 expected=float((s.weight.detach().float()-s0.float()).double().square().sum())/2
 assert seen==[six] and abs(v['combined_native_fisher_cost']-expected)<1e-12
 with torch.no_grad():s.weight.copy_(s0)
 seen.clear();a=gates(rt,pair,factor,z0,before,before[1]);b=gates(rt,pair,factor,z0,before,before[1],module_name=zero)
 assert a==b and seen==[zero,zero]
 # Both layers influence the toy forward; a wrong matrix cannot pass by aliasing.
 seen.clear();v=correction(rt,record('source'),factor,before[1]+.1,module_name=six)
 assert v['status']=='APPLIED' and seen==[six] and torch.equal(z.weight,z0)
 assert s.weight.dtype==s0.dtype and not torch.equal(s.weight,s0)
 with torch.no_grad():s.weight.copy_(s0)
 seen.clear();flags=[p.requires_grad for p in rt.model.parameters()];modes=[m.training for m in rt.model.modules()]
 with patch('experiments.vigor_medical_single_image_v1.finite_null.gradient_pair',side_effect=RuntimeError('injected')):
  try:correction(rt,record('source'),factor,before[1]+.1,compensate='secant',module_name=six)
  except RuntimeError:pass
  else:raise AssertionError('Expected propagated correction failure')
 assert seen==[six] and torch.equal(z.weight,z0) and torch.equal(s.weight,s0)
 assert not s._forward_hooks and not s._forward_pre_hooks and s.weight.dtype==s0.dtype
 assert flags==[p.requires_grad for p in rt.model.parameters()] and modes==[m.training for m in rt.model.modules()]
 # The unchanged default path must match an explicit layer0 request exactly.
 a=correction(rt,record('source'),factor,before[1]+.1);za=z.weight.detach().clone()
 with torch.no_grad():z.weight.copy_(z0)
 b=correction(rt,record('source'),factor,before[1]+.1,module_name=zero)
 assert a==b and torch.equal(z.weight,za) and torch.equal(s.weight,s0)
 with torch.no_grad():z.weight.copy_(z0)
 controller_checks()
 return dict(status='PASS',checks=['explicit_layer6_gate_metric','legacy_gate_default_exact',
  'correction_changes_layer6_only','native_commit','correction_exception_restore',
  'flags_modes_hooks_restore','legacy_correction_default_exact','unchanged_N_controller'])
