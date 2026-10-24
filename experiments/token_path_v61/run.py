"""Three matched native-only TT writers; only token-position mask differs."""
import os,json,time,signal,sys,traceback,shutil
from pathlib import Path
ROOT=Path(os.environ['V61_ROOT']);PART=int(os.environ['PARTITION']);OUT=ROOT/str(PART);P=Path(os.environ['PARENT_RUN']);START=time.time()
counts=dict(forwards=0,backwards=0,updates=0,generations=0,admission_backwards=0,vision_forwards=0)
limits=json.loads((ROOT/'RESOURCE_LIMITS.json').read_text());last_space=0

def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def budget():
 global last_space
 assert time.time()<limits['hard_end']-300 and not (ROOT/'STOP').exists()
 assert counts['backwards']<=1660 and counts['updates']<=1600 and counts['generations']<=405 and counts['forwards']<340000
 assert time.time()-read(ROOT/'RESOURCE_GUARD_HEALTH.json')['time']<120
 if time.time()-last_space>30:
  assert shutil.disk_usage(ROOT).free>8*1024**3
  assert sum(x.stat().st_size for x in ROOT.rglob('*') if x.is_file() and not x.is_symlink())<3000000000
  last_space=time.time()
def main():
 with (OUT/'ONCE').open('x'):pass
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Finite original window')))
 signal.alarm(max(1,min(10800,int(limits['hard_end']-time.time()-300))))
 sys.path[:0]=[str(ROOT/'tools'),str(P/'private/tools'),str(P/'private/source'),str(P/'private/official_llava')]
 import torch
 import module_run as r
 from scripts.medtrace.run_realmodel_core import load_real_runtime
 from types import SimpleNamespace
 from alignment import TT,Hook,snapshot
 from token_hook import TokenHook,layout
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
 ex=TT(lock['seed']).to(runtime.device);ex.load_state_dict(r.mm.canonicalize(snapshot(ex))[0]);initial=snapshot(ex)
 module=runtime.get_module('model.layers.21.mlp.down_proj');h=TokenHook(module,ex,arm)
 def value(b):
  h.teacher(b.labels,b.forward_kwargs().get('attention_mask'));loss=runtime.compute_loss(b);assert torch.isfinite(loss)
  gs=torch.autograd.grad(loss,tuple(ex.parameters()));counts['backwards']+=1
  g=torch.cat([g.detach().cpu().double().flatten() for g in gs]);assert torch.isfinite(g).all()
  return float(loss.detach()),g
 def emit(item,dest,active=False,zero=False):
  assert not dest.exists();counts['generations']+=1;h.set_layout(layout(item[2],item[3]))
  if zero:
   assert ex.G1.count_nonzero()==0
   with h.generation_request():answer,trace=r.s.fit.short.generate(runtime,None,item[2],item[3])
   trace=list(h.last_generation_trace);assert all(x['active_residual_norm']==0 for x in trace)
  else:answer,trace=r.s.fit.short.generate(runtime,h if active else None,item[2],item[3])
  tokens=list(answer.raw_token_ids);x=dict(binding=item[4],raw_answer=answer.decoded_text,raw_token_ids=tokens,EOS=tokens[-1]==runtime.adapter.tokenizer.eos_token_id,at_cap=len(tokens)>=1024,generation_trace=trace)
  write(dest,x);return x
 def mechanical(item,b):
  h.set_layout(layout(item[2],item[3]));first=h.current['prefix_length'];kwargs=b.forward_kwargs()
  assert first==int(torch.where(b.labels[0]!=-100)[0][0])
  assert torch.equal(kwargs['inputs_embeds'][:,:first].cpu(),item[3][4].float()),'Teacher/raw prefix embeddings must agree'
  with torch.no_grad():base=runtime.model(**kwargs).logits[0,first-1].detach().cpu().double()
  h.attach()
  try:
   h.teacher(b.labels,kwargs.get('attention_mask'))
   with torch.no_grad():zero=runtime.model(**kwargs).logits[0,first-1].detach().cpu().double()
   assert torch.equal(base,zero)
   with torch.no_grad():ex.G1.fill_(.001)
   h.teacher(b.labels,kwargs.get('attention_mask'));out=runtime.model(**kwargs)
   gs=torch.autograd.grad(out.loss,tuple(ex.parameters()));counts['backwards']+=1
   grad=torch.cat([g.detach().cpu().double().flatten() for g in gs]);assert torch.isfinite(grad).all() and grad.norm()>0
   reference=(float(out.loss.detach()),grad);expected=out.logits[0,first-1:first+1].detach().cpu().double();del out,gs
   with PrefixCache(runtime,[b],21):
    for _ in range(2):
     got=value(b);assert got[0]==reference[0] and torch.equal(got[1],reference[1])
   prefix={k:v for k,v in kwargs.items() if k!='labels'};prefix['use_cache']=True
   for name in ('inputs_embeds','attention_mask','position_ids'):
    if isinstance(prefix.get(name),torch.Tensor):prefix[name]=prefix[name][:,:first]
   with torch.no_grad(),h.generation_request():
    out=runtime.model(**prefix);actual=out.logits[0,-1].detach().cpu().double()
    decode=dict(prefix,past_key_values=out.past_key_values);decode['inputs_embeds']=kwargs['inputs_embeds'][:,first:first+1]
    if isinstance(kwargs.get('attention_mask'),torch.Tensor):decode['attention_mask']=kwargs['attention_mask'][:,:first+1]
    if isinstance(kwargs.get('position_ids'),torch.Tensor):decode['position_ids']=kwargs['position_ids'][:,first:first+1]
    out=runtime.model(**decode);actual_next=out.logits[0,-1].detach().cpu().double()
   torch.testing.assert_close(actual,expected[0],rtol=1e-5,atol=5e-4)
   torch.testing.assert_close(actual_next,expected[1],rtol=1e-5,atol=5e-4)
   trace=list(h.last_generation_trace);n=h.current['visual_count'] if arm=='VISUAL_ONLY' else first
   assert [t['active_predictor_count'] for t in trace]==[n,1 if arm=='ALL_TOKENS' else 0]
   assert trace[0]['active_residual_norm']>0
   if arm=='ALL_TOKENS':
    h.detach();legacy=Hook(module,ex,True);legacy.attach()
    try:
     legacy.teacher(b.labels,kwargs.get('attention_mask'));loss=runtime.compute_loss(b)
     g=torch.autograd.grad(loss,tuple(ex.parameters()));counts['backwards']+=1
     assert float(loss.detach())==reference[0] and torch.equal(torch.cat([v.detach().cpu().double().flatten() for v in g]),reference[1])
    finally:legacy.detach();h.attach()
   if arm=='VISUAL_ONLY':
    h.detach();ids=item[2]['input_ids'];ids=ids[:,ids[0]!=-200]
    with torch.no_grad():textbase=runtime.model(input_ids=ids,use_cache=False).logits.detach().cpu()
    h.set_layout(dict(prefix_length=ids.shape[1],visual_start=0,visual_count=0));h.attach()
    with torch.no_grad(),h.generation_request():textnew=runtime.model(input_ids=ids,use_cache=False).logits.detach().cpu()
    assert torch.equal(textbase,textnew);h.set_layout(layout(item[2],item[3]))
   return dict(status='PASS',arm=arm,source_gradient_norm=float(grad.norm()),zero_logits_exact=True,cache_CE_gradient_exact=True,teacher_prefix_max_abs=float((actual-expected[0]).abs().max()),teacher_KV_max_abs=float((actual_next-expected[1]).abs().max()),layout=layout(item[2],item[3]),trace=trace,legacy_A_exact=arm=='ALL_TOKENS',pure_text_identity=arm=='VISUAL_ONLY')
  finally:h.detach();ex.load_state_dict(initial)
 try:
  first=prepare([cfg['tasks'][0]['source']],True)[0];batch=r.old.batch_on(runtime,first[1]);check=mechanical(first,batch);counts['admission_backwards']=counts['backwards']
  h.set_layout(layout(first[2],first[3]));h.attach()
  try:smoke=emit(first,OUT/'ZERO_SMOKE.json',zero=True)
  finally:h.detach()
  write(OUT/'STAGE0_READY.json',dict(check,counts=dict(counts),time=time.time()))
  while not (ROOT/'STAGE1_GO.json').exists():
   budget()
   if PART==0 and all((ROOT/str(i)/'STAGE0_READY.json').exists() for i in range(3)):
    assert all(read(ROOT/str(i)/'STAGE0_READY.json')['status']=='PASS' for i in range(3))
    assert all(read(ROOT/str(i)/'ZERO_SMOKE.json')['raw_token_ids']==smoke['raw_token_ids'] for i in range(3))
    write(ROOT/'STAGE1_GO.json',dict(status='PASS',all_three_zero_tokens_exact=True,time=time.time()))
   time.sleep(.5)
  if PART==0:
   queries=read(ROOT/'QUERY_ROWS.json');qid=first[0]['query_id'];write(ROOT/'base'/(qid+'.json'),smoke)
   pending=[row for key,row in sorted(queries.items()) if key!=qid]
   for start in range(0,len(pending),12):
    for item in prepare(pending[start:start+12]):emit(item,ROOT/'base'/(item[0]['query_id']+'.json'))
    write(OUT/'PROGRESS.json',dict(phase='BASE',completed=min(start+12,len(pending))+1,counts=counts,time=time.time()))
   write(ROOT/'BASE_COMPLETE.json',dict(unique_inputs=len(queries),counts=counts,time=time.time()))
  for task in cfg['tasks']:
   folder=OUT/task['key'];folder.mkdir(exist_ok=True)
   with (folder/'ONCE').open('x'):pass
   ex.load_state_dict(initial);item=prepare([task['source']],True)[0];b=r.old.batch_on(runtime,item[1]);h.set_layout(layout(item[2],item[3]));h.attach()
   scalar=dict(step=0,second_moments=[0.,0.]);curve=[]
   try:
    # Each source has its own prefix cache; affected layer21 and all later layers stay live.
    with PrefixCache(runtime,[b],21):
     for step in range(1,161):
      before=snapshot(ex);loss,g=value(b);assert g.norm()>0
      raw,scalar,direction=proposal(r,before,g,scalar,list(before));ex.load_state_dict(construct(r,before,raw,direction,direction,1.,list(before)));counts['updates']+=1
      curve.append(dict(step=step,loss=loss))
      if step%40==0:write(OUT/'PROGRESS.json',dict(phase='TRAIN',task=task['key'],arm=arm,step=step,counts=counts,time=time.time()))
   finally:h.detach()
   torch.save(snapshot(ex),folder/'delta.pt');loaded=torch.load(folder/'delta.pt',map_location=runtime.device,weights_only=True);assert all(torch.equal(v,loaded[k]) for k,v in ex.state_dict().items());ex.load_state_dict(loaded)
   evaluation=prepare(task['eval_rows']);h.attach()
   try:
    for e in evaluation:emit(e,folder/'outputs'/(e[0]['query_id']+'.json'),active=True)
   finally:h.detach()
   write(folder/'COMPLETE.json',dict(arm=arm,curve=curve,save_load_exact=True,layout=layout(item[2],item[3]),outputs=len(evaluation),counts=dict(counts),time=time.time()))
   write(OUT/'PROGRESS.json',dict(phase='EDIT_COMPLETE',task=task['key'],arm=arm,counts=counts,time=time.time()))
  write(OUT/'MODEL_COMPLETE.json',dict(status='PASS',arm=arm,counts=counts,edits=10,updates=1600,formal_generations=219,time=time.time()))
 finally:h.detach();fh.remove();vh.remove()
if __name__=='__main__':
 code=0
 try:main()
 except BaseException:
  code=1;write(OUT/'FAILURE.json',dict(error=traceback.format_exc(),counts=counts,time=time.time()));write(ROOT/'STOP',dict(reason='V61_FAILURE'));raise
 finally:write(OUT/'EXIT.json',dict(code=code,counts=counts,seconds=time.time()-START))
