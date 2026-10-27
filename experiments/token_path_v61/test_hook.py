"""Small random CPU model: mask boundaries, causal gradients and KV consistency."""
import torch
from torch import nn
from transformers import MistralConfig,MistralForCausalLM
from token_hook import TokenHook,ARMS

def check():
 torch.manual_seed(61)
 c=MistralConfig(vocab_size=37,hidden_size=16,intermediate_size=32,num_hidden_layers=3,num_attention_heads=2,num_key_value_heads=2,head_dim=8,attention_dropout=0.,max_position_embeddings=32)
 c._attn_implementation='eager';m=MistralForCausalLM(c).double().eval()
 for p in m.parameters():p.requires_grad_(False)
 class Expert(nn.Module):
  def __init__(self):super().__init__();self.weight=nn.Parameter(torch.randn(32,16,dtype=torch.float64)*.03)
  def residual(self,x):return x@self.weight
 ids=torch.tensor([[1,3,5,7,9,11,13,15,2]]);labels=ids.clone();labels[:,:6]=-100
 expert=Expert();result=[]
 for arm in ARMS:
  h=TokenHook(m.model.layers[1].mlp.down_proj,expert,arm);h.set_layout(dict(prefix_length=6,visual_start=1,visual_count=3));h.attach()
  try:
   h.teacher(labels);y=m(input_ids=ids,labels=labels,use_cache=False);g=torch.autograd.grad(y.loss,expert.weight)[0];assert g.norm()>0
   with torch.no_grad(),h.generation_request():
    a=m(input_ids=ids[:,:6],use_cache=True);b=m(input_ids=ids[:,6:7],past_key_values=a.past_key_values,use_cache=True)
   torch.testing.assert_close(a.logits[:,-1],y.logits[:,5],rtol=1e-10,atol=1e-10)
   torch.testing.assert_close(b.logits[:,-1],y.logits[:,6],rtol=1e-10,atol=1e-10)
   trace=h.last_generation_trace;expected={'ALL_TOKENS':[6,1],'PREFILL_ONLY':[6,0],'VISUAL_ONLY':[3,0]}[arm]
   assert [x['active_predictor_count'] for x in trace]==expected
   result.append(dict(arm=arm,gradient_norm=float(g.norm()),trace=list(trace)))
  finally:h.detach()
 with torch.no_grad():base=m(input_ids=ids).logits
 h=TokenHook(m.model.layers[1].mlp.down_proj,expert,'VISUAL_ONLY');h.set_layout(dict(prefix_length=9,visual_start=0,visual_count=0));h.attach()
 try:
  with torch.no_grad(),h.generation_request():text=m(input_ids=ids).logits
  assert torch.equal(base,text)
 finally:h.detach()
 return dict(status='PASS',real_model_calls=0,synthetic_random_Mistral=True,masks_and_KV=True,text_only_exact=True,results=result)
if __name__=='__main__':
 import json;print(json.dumps(check(),indent=2))
