"""Fixed accepted-trace B/N reconstruction for new evaluation, never new search."""
import math,time
import torch
import torch.nn.functional as F
from .editor import editable_fp32,direction,nll,finite
from .finite_null import gates,correction
from .marginab_ops import fisher_cost
from .factF3_ops import read

def rebuild(rt,record,pair,factor,a,b,base,before,groups,oldarm,check,near,emit,entry):
 start=time.time();layer=rt.get_module('model.layers.0.mlp.down_proj');last=before;mm,identity=read(rt,pair[0],groups);count=0;secants=0;backwards=0
 for oldstep in oldarm['steps']:
  trials=[t for t in oldstep['trials'] if t['status']=='ACCEPTED']
  if not trials:continue
  if len(trials)!=1:raise ValueError('Ambiguous historical accepted trace')
  t=trials[0];original=layer.weight.detach().clone();row=dict(step=count+1,old_scale=t['scale'],status='RECONSTRUCTING');entry['steps'].append(row);emit()
  with editable_fp32(rt.model,layer,readonly=True):
   out=rt.model(**pair[0].forward_kwargs());logits=out.logits.float();_,target,j,absolute=identity;loss=F.cross_entropy(logits[:,:-1].reshape(-1,logits.shape[-1]),pair[0].labels[:,1:].reshape(-1),ignore_index=-100);lm=logits[0,absolute,target]-logits[0,absolute,j]
   gm=finite(torch.autograd.grad(lm,layer.weight,retain_graph=True)[0].detach());gn=finite(torch.autograd.grad(loss,layer.weight)[0].detach());del out,logits,loss,lm,gm
   ln=nll(rt.model,pair[1]);g0=finite(torch.autograd.grad(ln,layer.weight)[0].detach());del ln;backwards+=3
   qn,geometry=direction(gn,g0,a,b)
   if qn is None:raise ValueError('Historical direction no longer nondegenerate')
   delta=finite(-math.sqrt(2*5.368434395105091e-13/geometry['E'])*qn)
  check(torch.equal(layer.weight,original),'rebuild_gradient_restore')
  proposal=(original.float()+t['scale']*delta).half();cost=fisher_cost(proposal.float()-original.float(),a,b);check(abs(cost/t['cost']-1)<=1e-5,'rebuild_native_cost_binding')
  with torch.no_grad():layer.weight.copy_(proposal)
  verdict=gates(rt,pair,factor,original,last,before[1]);main,mi=read(rt,pair[0],groups,identity[0],identity[2]);row.update(main_gate=verdict,main_margin=main);emit()
  for x,y in zip(verdict['native_nll'],t['main_gate']['native_nll']):near(x,y,'rebuild_historical_main_loss')
  near(main['margin'],t['main_margin']['margin'],'rebuild_historical_main_margin')
  row['correction']=None
  if t.get('correction') is not None:
   secants+=1;backwards+=2;row['correction']=correction(rt,record,factor,before[1],compensate='secant');check(row['correction']['status']=='APPLIED','rebuild_secant_status');final=gates(rt,pair,factor,original,last,before[1]);fm,fi=read(rt,pair[0],groups,identity[0],identity[2])
  else:final,fm,fi=verdict,main,mi
  for x,y in zip(final['native_nll'],t['final_gate']['native_nll']):near(x,y,'rebuild_historical_final_loss')
  near(fm['margin'],t['final_margin']['margin'],'rebuild_historical_final_margin');check(final['accepted'] and all(final['gates'].values()),'rebuild_final_original_gates')
  last=final['native_nll'];mm=fm;identity=fi;count+=1;row.update(status='BOUND_TO_HISTORICAL_SCALARS',final_gate=final,final_margin=fm);entry['accepted_steps']=count;emit()
 check(count==oldarm['accepted_steps'],'historical_step_count_binding');cost=fisher_cost(layer.weight.detach().float()-base.float(),a,b);check(abs(cost/oldarm['last_accepted']['cumulative_F0_cost']-1)<=1e-5,'historical_endpoint_cost_binding')
 return dict(status='RECONSTRUCTION_COMPLETE',accepted_steps=count,backward_calls=backwards,secant_calls=secants,reconstruction_seconds=time.time()-start,last_accepted=dict(native_nll=last,margin=mm,cumulative_F0_cost=cost),historical_weight_tensor_comparison=None,scalar_and_cost_binding=True,new_search_trials=0)
