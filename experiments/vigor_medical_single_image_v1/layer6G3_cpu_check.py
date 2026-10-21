"""Predictive-screen ablation checks; actual protection remains unchanged."""
import math
import torch
from types import SimpleNamespace
from unittest.mock import patch
from .nll_control import admission,run_checks as prior_checks
from .finite_null import gates

def run_checks():
 prior_checks()
 for cost in [0.,.5,1.,1.02,float('inf'),float('nan')]:
  for source in [-1.,-1e-5,0.,float('nan')]:
   for null in [-.02,0.,.02,float('inf')]:
    for bias in [-.02,0.,.02,float('nan')]:
     linear=[source,1.,null]
     old=all(math.isfinite(x) for x in [cost,*linear,bias]) and 0<cost<=1.01 and source<-1e-5 and abs(bias+null)<=.01
     assert admission(cost,linear,1.,bias)==old
     assert admission(cost,linear,1.,bias,check_null_prediction=True)==old
     new=admission(cost,linear,1.,bias,check_null_prediction=False)
     expected=all(math.isfinite(x) for x in [cost,*linear,bias]) and 0<cost<=1.01 and source<-1e-5
     assert new==expected
 assert not admission(1.,[-1.,1.,.02],1.,0.)
 assert admission(1.,[-1.,1.,.02],1.,0.,check_null_prediction=False)
 layer=torch.nn.Linear(2,2,bias=False).half();base=layer.weight.detach().clone();seen=[]
 def get_module(name):seen.append(name);assert name=='model.layers.6.mlp.down_proj';return layer
 rt=SimpleNamespace(get_module=get_module);factor=dict(a=torch.ones(2),b=torch.ones(2))
 with patch('experiments.vigor_medical_single_image_v1.finite_null.native_losses',return_value=[.5,1.02]):
  v=gates(rt,None,factor,base,[1.,1.],1.,module_name='model.layers.6.mlp.down_proj')
 assert not v['accepted'] and not v['gates']['null'] and v['gates']['visual'] and v['gates']['trust']
 assert torch.equal(layer.weight,base) and seen==['model.layers.6.mlp.down_proj']
 with torch.no_grad():layer.weight.add_(.1)
 with patch('experiments.vigor_medical_single_image_v1.finite_null.native_losses',return_value=[.5,1.]):
  v=gates(rt,None,factor,base,[1.,1.],1.,module_name='model.layers.6.mlp.down_proj')
 assert not v['accepted'] and not v['gates']['trust']
 with torch.no_grad():layer.weight.copy_(base)
 assert torch.equal(layer.weight,base) and layer.weight.dtype==torch.float16
 return dict(status='PASS',checks=['legacy_admission_exact','explicit_prediction_only_bypass',
  'finite_cost_source_prediction_still_required','actual_null_guard_unchanged',
  'actual_trust_guard_unchanged','native_restore','original_N_controller'])
