"""Matched FP32 generation with explicitly permitted protected zero residuals."""
from contextlib import nullcontext
import torch

def generate(runtime,hook,raw,expanded):
 model=runtime.llava_model();original=model.prepare_inputs_labels_for_multimodal;used=[0]
 def prepared(*args,**kwargs):
  if args[0] is not None and torch.equal(args[0],raw['input_ids']):
   assert args[3] is None and args[4] is None;used[0]+=1;assert used[0]==1
   return tuple(x.to(device=runtime.device,dtype=torch.float32 if x.is_floating_point() else x.dtype) if isinstance(x,torch.Tensor) else x for x in expanded)
  return original(*args,**kwargs)
 model.prepare_inputs_labels_for_multimodal=prepared
 try:
  with torch.inference_mode(),(hook.generation_request() if hook else nullcontext()):answer=runtime.adapter.generate_prepared_with_result(raw,runtime.generation_config)
 finally:model.prepare_inputs_labels_for_multimodal=original
 assert used[0]==1 and answer.raw_token_ids
 trace=list(hook.last_generation_trace) if hook else []
 if hook is not None:
  assert trace and all(x['active_predictor_count']>0 and x['active_residual_norm']>=0 and torch.isfinite(torch.tensor(x['active_residual_norm'])) for x in trace)
  if not getattr(hook.expert,'_conditional_projection',False):assert any(x['active_residual_norm']>0 for x in trace)
 return answer,trace
