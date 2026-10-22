import os,json,time,sys,signal
from pathlib import Path
r=Path(os.environ['V68_ROOT']);START=time.time();counts=dict(forwards=0,backwards=0,updates=0,generations=0,vision_forwards=0)
os.environ.update({k:str(v) for k,v in json.loads((r/'ENV.json').read_text()).items()});p=Path(os.environ['PARENT_RUN']);sys.path[:0]=[str(r/'tools'),str(p/'private/tools'),str(p/'private/source'),str(p/'private/official_llava')]
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('diagnostic deadline')));signal.alarm(180)
def write(name,x):(r/name).write_text(json.dumps(x,indent=2)+'\n')
try:
 import torch
 from types import SimpleNamespace
 from scripts.medtrace.run_realmodel_core import load_real_runtime
 import module_run as m
 from alignment import TT,snapshot
 from token_hook import TokenHook,layout
 from anchor import prefix_kwargs,divergence
 m.c.budget=lambda:None
 torch.use_deterministic_algorithms(True);torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True);torch.backends.cuda.matmul.allow_tf32=False
 runtime=load_real_runtime(SimpleNamespace(cpu_gate=p/'private/cpu_gate'));runtime.model.eval();m.old.fp32(runtime)
 runtime.model.register_forward_pre_hook(lambda *_:counts.__setitem__('forwards',counts['forwards']+1));runtime.model.get_vision_tower().register_forward_pre_hook(lambda *_:counts.__setitem__('vision_forwards',counts['vision_forwards']+1))
 row=json.loads((r/'SWAPS_FROZEN.json').read_text())['rows'][7];assert row['owner']==8
 bindings=json.loads((p/'private/EVAL_BINDINGS.json').read_text());template=json.loads((p/'private/M13_SOURCE_PACKET.json').read_text())['original_tasks'][4]
 runtime.model.half()
 try:item=m.s.prepare(runtime,bindings,template,[row['swapped']],True)[0]
 finally:m.old.fp32(runtime)
 b=m.old.batch_on(runtime,item[1]);lay=layout(item[2],item[3]);kw=prefix_kwargs(b,lay['prefix_length'])
 with torch.no_grad():out=runtime.model(**kw);base=out.logits[0,-1].detach().clone();del out
 ex=TT(225034559).to(runtime.device);ex.load_state_dict(m.mm.canonicalize(snapshot(ex))[0]);ex.G1.data.fill_(.001)
 h=TokenHook(runtime.get_module('model.layers.21.mlp.down_proj'),ex,'ALL_TOKENS');h.set_layout(lay);h.attach()
 try:
  with h.generation_request():out=runtime.model(**kw)
  logits=out.logits[0,-1];kl32=divergence(logits,base.float().log_softmax(-1).detach());g32=torch.autograd.grad(kl32,tuple(ex.parameters()),retain_graph=True);counts['backwards']+=1
  kl64=torch.nn.functional.kl_div(logits.double().log_softmax(-1),base.double().log_softmax(-1),log_target=True,reduction='sum');g64=torch.autograd.grad(kl64,tuple(ex.parameters()));counts['backwards']+=1
  a=torch.cat([z.detach().double().flatten() for z in g32]);c=torch.cat([z.detach().double().flatten() for z in g64]);diff=(logits.detach().double()-base.double())
  result=dict(owner=8,root_value=.001,kl32=float(kl32.detach()),kl64=float(kl64.detach()),gradient32_norm=float(a.norm()),gradient64_norm=float(c.norm()),finite_gradients=bool(torch.isfinite(a).all() and torch.isfinite(c).all()),logit_max_abs_change=float(diff.abs().max()),logit_changed_elements=int(diff.count_nonzero()),counts=counts,seconds=time.time()-START)
  result['numerical_failure_identified']=result['kl32']<=0<result['kl64'] and result['gradient32_norm']>0 and result['gradient64_norm']>0 and result['finite_gradients'] and result['logit_max_abs_change']>0
  write('DIAGNOSTIC_RESULT.json',result);print(json.dumps(result))
 finally:h.detach()
except BaseException:
 import traceback
 write('DIAGNOSTIC_FAILURE.json',dict(error=traceback.format_exc(),counts=counts,seconds=time.time()-START));raise
finally:write('DIAGNOSTIC_EXIT.json',dict(counts=counts,seconds=time.time()-START,time=time.time()))
