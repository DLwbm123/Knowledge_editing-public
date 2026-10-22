"""Two fixed writer trajectories; source-only checkpoint selection versus step160."""
import os,json,time,signal,sys,traceback,shutil
from pathlib import Path
ROOT=Path(os.environ['V62_ROOT']);PART=int(os.environ['PARTITION']);OUT=ROOT/str(PART);P=Path(os.environ['PARENT_RUN']);START=time.time()
counts=dict(forwards=0,backwards=0,updates=0,generations=0,admission_backwards=0,vision_forwards=0)
limits=json.loads((ROOT/'RESOURCE_LIMITS.json').read_text());last_space=0

def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def budget():
 global last_space
 assert time.time()<limits['hard_end']-300 and not (ROOT/'STOP').exists()
 assert counts['backwards']<=1700 and counts['updates']<=1600 and counts['generations']<=520 and counts['forwards']<500000
 assert time.time()-read(ROOT/'RESOURCE_GUARD_HEALTH.json')['time']<120
 if time.time()-last_space>30:
  assert shutil.disk_usage(ROOT).free>8*1024**3
  assert sum(x.stat().st_size for x in ROOT.rglob('*') if x.is_file() and not x.is_symlink())<3000000000
  last_space=time.time()
def main():
 with (OUT/'ONCE').open('x'):pass
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Finite original window')))
 signal.alarm(max(1,min(7200,int(limits['hard_end']-time.time()-300))))
 sys.path[:0]=[str(ROOT/'tools'),str(P/'private/tools'),str(P/'private/source'),str(P/'private/official_llava')]
 import torch
 import module_run as r
 from scripts.medtrace.run_realmodel_core import load_real_runtime
 from types import SimpleNamespace
 from alignment import TT,Hook,snapshot
 from token_hook import TokenHook,layout
 from path_v18 import proposal,construct
 from v51_cache import PrefixCache
 from structure import LoRA,lora_step,preserve_rng,target_margin,exact_source_pass
 r.c.budget=budget
 torch.use_deterministic_algorithms(True);torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True);torch.backends.cuda.matmul.allow_tf32=False
 runtime=load_real_runtime(SimpleNamespace(cpu_gate=P/'private/cpu_gate'));runtime.model.eval();r.old.fp32(runtime)
 assert not any(v.requires_grad for v in runtime.model.parameters())
 cfg=read(ROOT/'DATA_ADMISSION.json');lock=read(ROOT/'PROTOCOL_LOCK.json');structure=lock['structures'][PART];arm='ALL_TOKENS'
 template=read(P/'private/M13_SOURCE_PACKET.json')['original_tasks'][4];bindings=read(P/'private/EVAL_BINDINGS.json');assert runtime.generation_config==next(iter(bindings.values()))['generation']
 def count(*_):budget();counts['forwards']+=1
 fh=runtime.model.register_forward_pre_hook(count)
 vh=runtime.model.get_vision_tower().register_forward_pre_hook(lambda *_:counts.__setitem__('vision_forwards',counts['vision_forwards']+1))
 def prepare(rows,training=False):
  runtime.model.half()
  try:return r.s.prepare(runtime,bindings,template,rows,training)
  finally:r.old.fp32(runtime)
 ex=(TT(lock['seed']) if PART==0 else LoRA(lock['seed'])).to(runtime.device)
 if PART==0:ex.load_state_dict(r.mm.canonicalize(snapshot(ex))[0])
 initial=snapshot(ex)
 zero_parameter=ex.G1 if PART==0 else ex.B
 module=runtime.get_module('model.layers.21.mlp.down_proj');h=TokenHook(module,ex,arm)
 def value(b):
  h.teacher(b.labels,b.forward_kwargs().get('attention_mask'));loss=runtime.compute_loss(b);assert torch.isfinite(loss)
  gs=torch.autograd.grad(loss,tuple(ex.parameters()));counts['backwards']+=1
  g=torch.cat([g.detach().cpu().double().flatten() for g in gs]);assert torch.isfinite(g).all()
  return float(loss.detach()),g
 def emit(item,dest,active=False,zero=False):
  assert not dest.exists();counts['generations']+=1;h.set_layout(layout(item[2],item[3]))
  if zero:
   assert zero_parameter.count_nonzero()==0
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
   with torch.no_grad():zero_parameter.fill_(.001)
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
   if PART==0 and all((ROOT/str(i)/'STAGE0_READY.json').exists() for i in range(2)):
    assert all(read(ROOT/str(i)/'STAGE0_READY.json')['status']=='PASS' for i in range(2))
    assert all(read(ROOT/str(i)/'ZERO_SMOKE.json')['raw_token_ids']==smoke['raw_token_ids'] for i in range(2))
    write(ROOT/'STAGE1_GO.json',dict(status='PASS',both_zero_tokens_exact=True,time=time.time()))
   time.sleep(.5)
  old_root=Path(os.environ['V61_PARENT'])
  assert smoke['raw_token_ids']==read(old_root/'0/ZERO_SMOKE.json')['raw_token_ids']
  write(OUT/'BASE_REUSE_ADMISSION.json',dict(status='PASS',zero_source_tokens_exact=True,base_unique_inputs=183,old_semantic_labels_reused=False))
  for task in cfg['tasks']:
   folder=OUT/task['key'];folder.mkdir(exist_ok=True)
   with (folder/'ONCE').open('x'):pass
   ex.load_state_dict(initial);item=prepare([task['source']],True)[0];b=r.old.batch_on(runtime,item[1]);h.set_layout(layout(item[2],item[3]));h.attach()
   scalar=dict(step=0,second_moments=[0.,0.]);curve=[];checks=[];selected=None;selected_state=None
   train_seconds=0.;check_seconds=0.
   try:
    with PrefixCache(runtime,[b],21):
     for step in range(1,161):
      began=time.time();before=snapshot(ex);loss,g=value(b);assert g.norm()>0
      if PART==0:
       raw,scalar,direction=proposal(r,before,g,scalar,list(before));state=construct(r,before,raw,direction,direction,1.,list(before))
      else:state,scalar=lora_step(before,g,scalar)
      ex.load_state_dict(state);counts['updates']+=1;train_seconds+=time.time()-began
      curve.append(dict(step=step,loss=loss))
      if step in lock['checks'] and selected is None:
       check_start=time.time();before_check=snapshot(ex);scalar_before=json.dumps(scalar,sort_keys=True);cc=dict(counts)
       with preserve_rng():
        h.teacher(b.labels,b.forward_kwargs().get('attention_mask'))
        with torch.no_grad():out=runtime.model(**b.forward_kwargs());margin,target=target_margin(out.logits,b.labels)
        del out
        assert target==list(b.target_token_ids) and target[-1]==runtime.adapter.tokenizer.eos_token_id
        generated=None;passed=False
        if margin>0:
         generated=emit(item,folder/'checks'/('step%03d.json'%step),active=True)
         passed=exact_source_pass(generated['raw_token_ids'],target,runtime.adapter.tokenizer.eos_token_id,generated['at_cap']) and generated['EOS']
       assert all(torch.equal(v,before_check[k]) for k,v in ex.state_dict().items()) and json.dumps(scalar,sort_keys=True)==scalar_before
       elapsed=time.time()-check_start;check_seconds+=elapsed
       checks.append(dict(step=step,min_target_margin=margin,teacher_all_strict_top1=margin>0,greedy_checked=generated is not None,source_pass=passed,seconds=elapsed,counts={k:counts[k]-cc[k] for k in counts},trajectory_state_unchanged=True))
       if passed:
        selected=step;selected_state=snapshot(ex);selected_training_seconds=train_seconds;selected_check_seconds=check_seconds
      if step%40==0:write(OUT/'PROGRESS.json',dict(phase='TRAIN',task=task['key'],structure=structure,step=step,selected_step=selected,counts=counts,time=time.time()))
   finally:h.detach()
   fixed=snapshot(ex)
   if selected is None:selected=160;selected_state=fixed;selected_training_seconds=train_seconds;selected_check_seconds=check_seconds
   if PART==0:
    old_curve=read(old_root/'0'/task['key']/'COMPLETE.json')['curve']
    assert curve==old_curve,'V61 fixed TT training replay differs; no automatic repair or retuning'
   selection=dict(structure=structure,selected_step=selected,fallback=not any(c['source_pass'] for c in checks),checks=checks,curve=curve,source_only=True,selected_before_any_evaluation=True,train_seconds=train_seconds,check_seconds=check_seconds,selected_training_seconds=selected_training_seconds,selected_check_seconds=selected_check_seconds,time=time.time())
   write(folder/'SELECTION_FROZEN.json',selection)
   for name,state in [('FIXED160',fixed),('SOURCE_STOP',selected_state)]:
    d=folder/name;d.mkdir(exist_ok=True);torch.save(state,d/'delta.pt');loaded=torch.load(d/'delta.pt',map_location=runtime.device,weights_only=True)
    assert all(torch.equal(v,loaded[k]) for k,v in state.items());ex.load_state_dict(loaded)
   evaluation=prepare(task['eval_rows']);outputs_reused=0;tt_tokens_checked=0
   for name,state in [('FIXED160',fixed),('SOURCE_STOP',selected_state)]:
    ex.load_state_dict(state);h.attach()
    try:
     for e in evaluation:
      dest=folder/name/'outputs'/(e[0]['query_id']+'.json')
      if name=='SOURCE_STOP' and selected==160:
       assert all(torch.equal(v,fixed[k]) for k,v in selected_state.items())
       write(dest,read(folder/'FIXED160'/'outputs'/(e[0]['query_id']+'.json')));outputs_reused+=1
      else:
       generated=emit(e,dest,active=True)
       if PART==0 and name=='FIXED160':
        prior=read(old_root/'0'/task['key']/'outputs'/(e[0]['query_id']+'.json'))
        assert generated['raw_token_ids']==prior['raw_token_ids'],'V61 fixed TT output replay differs'
        assert {k:v for k,v in generated['binding'].items() if k!='image_path'}=={k:v for k,v in prior['binding'].items() if k!='image_path'}
        tt_tokens_checked+=1
    finally:h.detach()
   write(folder/'COMPLETE.json',dict(structure=structure,selected_step=selected,save_load_exact=True,curve_replay_exact=PART==0,TT160_tokens_checked=tt_tokens_checked,outputs_per_state=len(evaluation),same_state_outputs_reused=outputs_reused,counts=dict(counts),time=time.time()))
   write(OUT/'PROGRESS.json',dict(phase='EDIT_COMPLETE',task=task['key'],structure=structure,selected_step=selected,counts=counts,time=time.time()))
  write(OUT/'MODEL_COMPLETE.json',dict(status='PASS',structure=structure,counts=counts,edits=10,updates=1600,formal_output_consumers=438,time=time.time()))
 finally:h.detach();fh.remove();vh.remove()
if __name__=='__main__':
 code=0
 try:main()
 except BaseException:
  code=1;write(OUT/'FAILURE.json',dict(error=traceback.format_exc(),counts=counts,time=time.time()));write(ROOT/'STOP',dict(reason='V62_FAILURE'));raise
 finally:write(OUT/'EXIT.json',dict(code=code,counts=counts,seconds=time.time()-START))
