"""Fixed source-only versus joint positive/negative CE, with untouched retention tests."""
import os,json,time,signal,sys,traceback,shutil
from pathlib import Path
ROOT=Path(os.environ['V59_ROOT']);PART=int(os.environ['PARTITION']);OUT=ROOT/str(PART);P=Path(os.environ['PARENT_RUN']);START=time.time()
counts=dict(forwards=0,backwards=0,updates=0,generations=0,admission_backwards=0,vision_forwards=0)
limits=json.loads((ROOT/'RESOURCE_LIMITS.json').read_text());last_space=0
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def budget():
 global last_space
 assert time.time()<limits['hard_end']-300 and not (ROOT/'STOP').exists()
 assert counts['backwards']<=326 and counts['updates']<=160 and counts['generations']<=18 and counts['forwards']<40000
 assert time.time()-read(ROOT/'RESOURCE_GUARD_HEALTH.json')['time']<120
 if time.time()-last_space>30:
  assert shutil.disk_usage(ROOT).free>8*1024**3
  assert sum(x.stat().st_size for x in ROOT.rglob('*') if x.is_file() and not x.is_symlink())<1000000000
  last_space=time.time()
def main():
 with (OUT/'ONCE').open('x'):pass
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Bounded original window')))
 signal.alarm(max(1,min(1800,int(limits['hard_end']-time.time()-300))))
 sys.path[:0]=[str(ROOT/'tools'),str(P/'private/tools'),str(P/'private/source'),str(P/'private/official_llava')]
 import torch
 import module_run as r
 from scripts.medtrace.run_realmodel_core import load_real_runtime
 from types import SimpleNamespace
 from alignment import TT,Hook,snapshot
 from path_v18 import proposal,construct
 from v51_cache import PrefixCache
 r.c.budget=budget
 torch.use_deterministic_algorithms(True);torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True);torch.backends.cuda.matmul.allow_tf32=False
 runtime=load_real_runtime(SimpleNamespace(cpu_gate=P/'private/cpu_gate'));runtime.model.eval();r.old.fp32(runtime)
 assert not any(v.requires_grad for v in runtime.model.parameters())
 cfg=read(ROOT/'DATA_ADMISSION.json');lock=read(ROOT/'PROTOCOL_LOCK.json');arm=lock['arms'][PART]
 template=read(P/'private/M13_SOURCE_PACKET.json')['original_tasks'][4];bindings=read(P/'private/EVAL_BINDINGS.json');assert runtime.generation_config==next(iter(bindings.values()))['generation']
 def count(*_):budget();counts['forwards']+=1
 fh=runtime.model.register_forward_pre_hook(count)
 vh=runtime.model.get_vision_tower().register_forward_pre_hook(lambda *_:counts.__setitem__('vision_forwards',counts['vision_forwards']+1))
 def prepare(rows,training=False):
  runtime.model.half()
  try:return r.s.prepare(runtime,bindings,template,rows,training)
  finally:r.old.fp32(runtime)
 train=prepare(cfg['train_rows'],True);batches=[r.old.batch_on(runtime,x[1]) for x in train];evaluation=prepare(cfg['eval_rows'])
 assert len(batches)==2 and len(evaluation)==17
 for item in evaluation:assert item[4]==cfg['Base_outputs'][item[0]['query_id']]['binding'],'Exact Base input/runtime binding'
 ex=TT(lock['seed']).to(runtime.device);ex.load_state_dict(r.mm.canonicalize(snapshot(ex))[0]);initial=snapshot(ex)
 h=Hook(runtime.get_module('model.layers.21.mlp.down_proj'),ex,True);h.attach()
 def value(b):
  h.teacher(b.labels,b.forward_kwargs().get('attention_mask'));loss=runtime.compute_loss(b);assert torch.isfinite(loss)
  gs=torch.autograd.grad(loss,tuple(ex.parameters()));counts['backwards']+=1
  return float(loss.detach()),torch.cat([g.detach().cpu().double().flatten() for g in gs])
 def emit(item,smoke=False):
  counts['generations']+=1
  if smoke:
   assert ex.G1.count_nonzero()==0
   with h.generation_request():answer,trace=r.s.fit.short.generate(runtime,None,item[2],item[3])
   assert h.last_generation_trace and all(x['active_residual_norm']==0 for x in h.last_generation_trace)
  else:answer,trace=r.s.fit.short.generate(runtime,h,item[2],item[3])
  tokens=list(answer.raw_token_ids);x=dict(binding=item[4],raw_answer=answer.decoded_text,raw_token_ids=tokens,EOS=tokens[-1]==runtime.adapter.tokenizer.eos_token_id,at_cap=len(tokens)>=1024)
  write(OUT/('smoke' if smoke else 'outputs')/(item[0]['query_id']+'.json'),x);return x
 try:
  smoke=emit(evaluation[0],True);base=cfg['Base_outputs'][evaluation[0][0]['query_id']];assert smoke['raw_token_ids']==base['raw_token_ids'] and smoke['raw_answer']==base['raw_answer']
  old=[value(b) for b in batches];counts['admission_backwards']+=2
  with PrefixCache(runtime,batches,21):
   for _ in range(2):
    for b,ref in zip(batches,old):
     cur=value(b);counts['admission_backwards']+=1;assert cur[0]==ref[0] and torch.equal(cur[1],ref[1])
  assert all(torch.equal(v,initial[k]) for k,v in ex.state_dict().items())
  write(OUT/'STAGE0_READY.json',dict(status='PASS',arm=arm,zero_Base_generation_exact=True,cache_CE_gradient_exact=True,Base_bindings=17,counts=counts,time=time.time()))
  while not (ROOT/'STAGE1_GO.json').exists():budget();time.sleep(1)
  scalar=dict(step=0,second_moments=[0.,0.]);curve=[]
  with PrefixCache(runtime,batches,21):
   for step in range(1,161):
    budget();before=snapshot(ex);values=[value(batches[0]),value(batches[0] if arm=='SOURCE_ONLY' else batches[1])]
    gradient=(values[0][1]+values[1][1])/2;keys=list(before);raw,scalar,direction=proposal(r,before,gradient,scalar,keys);ex.load_state_dict(construct(r,before,raw,direction,direction,1.,keys));counts['updates']+=1
    curve.append(dict(step=step,losses=[x[0] for x in values]))
    if step%20==0:write(OUT/'PROGRESS.json',dict(arm=arm,step=step,counts=counts,time=time.time()))
  torch.save(snapshot(ex),OUT/'delta.pt');loaded=torch.load(OUT/'delta.pt',map_location=runtime.device,weights_only=True);assert all(torch.equal(v,loaded[k]) for k,v in ex.state_dict().items());ex.load_state_dict(loaded)
  for item in evaluation:emit(item)
  write(OUT/'MODEL_COMPLETE.json',dict(status='PASS',arm=arm,curve=curve,counts=counts,save_load_exact=True,initial_seed=lock['seed'],writer_parameters=7168,time=time.time()))
 finally:h.detach();fh.remove();vh.remove()
if __name__=='__main__':
 code=0
 try:main()
 except BaseException:
  code=1;write(OUT/'FAILURE.json',dict(error=traceback.format_exc(),counts=counts,time=time.time()));write(ROOT/'STOP',dict(reason='V59_FAILURE'));raise
 finally:write(OUT/'EXIT.json',dict(code=code,counts=counts,seconds=time.time()-START))
