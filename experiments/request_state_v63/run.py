"""Frozen six-arm experiment, with CPU/real admission and exact A replay barriers."""
import os,json,time,signal,sys,traceback,shutil,gc
from pathlib import Path
ROOT=Path(os.environ['V63_ROOT']);P=Path(os.environ['PARENT_RUN']);OLD=Path(os.environ['V62_PARENT']);START=time.time()
counts=dict(forwards=0,backwards=0,updates=0,generations=0,vision_forwards=0,prefix_extractions=0)

def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
limits=read(ROOT/'RESOURCE_LIMITS.json');last_space=0.
def budget():
 global last_space
 assert time.time()<limits['hard_end']-300 and not (ROOT/'STOP').exists(),'WINDOW_STOP'
 assert counts['backwards']<=26000 and counts['updates']<=12800 and counts['generations']<=1880 and counts['forwards']<2200000
 assert time.time()-read(ROOT/'RESOURCE_GUARD_HEALTH.json')['time']<120,'GUARD_STALE'
 if time.time()-last_space>30:
  assert shutil.disk_usage(ROOT).free>8*1024**3
  assert sum(x.stat().st_size for x in ROOT.rglob('*') if x.is_file() and not x.is_symlink())<2500000000
  last_space=time.time();write(ROOT/'COUNTS.json',dict(counts,time=time.time()))

def main():
 with (ROOT/'WORKER_ONCE').open('x'):pass
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('V63 finite deadline')))
 signal.alarm(max(1,int(limits['hard_end']-time.time()-300)))
 sys.path[:0]=[str(ROOT/'tools'),str(P/'private/tools'),str(P/'private/source'),str(P/'private/official_llava')]
 import torch,numpy as np
 import module_run as r
 from types import SimpleNamespace
 from scripts.medtrace.run_realmodel_core import load_real_runtime
 from alignment import TT,snapshot
 from token_hook import TokenHook,layout
 from path_v18 import proposal,construct
 from v51_cache import PrefixCache
 from v62_structure import LoRA,lora_step
 from state_patch import intervention,random_direction
 assert read(ROOT/'CPU_ADMISSION.json')['status']=='PASS'
 r.c.budget=budget
 torch.use_deterministic_algorithms(True);torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True);torch.backends.cuda.matmul.allow_tf32=False
 runtime=load_real_runtime(SimpleNamespace(cpu_gate=P/'private/cpu_gate'));runtime.model.eval();r.old.fp32(runtime)
 assert not any(v.requires_grad for v in runtime.model.parameters())
 cfg=read(ROOT/'DATA_ADMISSION.json');views=read(ROOT/'VIEWS_FROZEN.json')['views'];template=read(P/'private/M13_SOURCE_PACKET.json')['original_tasks'][4]
 bindings=read(P/'private/EVAL_BINDINGS.json');assert runtime.generation_config==next(iter(bindings.values()))['generation']
 layer=runtime.model.model.layers[21];module=runtime.get_module('model.layers.21.mlp.down_proj');eos=runtime.adapter.tokenizer.eos_token_id
 def count(*_):budget();counts['forwards']+=1
 fh=runtime.model.register_forward_pre_hook(count)
 vh=runtime.model.get_vision_tower().register_forward_pre_hook(lambda *_:counts.__setitem__('vision_forwards',counts['vision_forwards']+1))
 def prepare(rows,training=False):
  runtime.model.half()
  try:return r.s.prepare(runtime,bindings,template,rows,training)
  finally:r.old.fp32(runtime)
 def batches(task):
  rows=[task['source']]
  vv=sorted([v for v in views if v['owner']==task['order']],key=lambda v:v['view']);assert len(vv)==2
  for v in vv:
   assert v['source']==task['source'] and v['target_reference']==task['source']['reference']
   rows.append(dict(task['source'],question=v['derived_question'],query_id='v63-%s-%s'%(task['order'],v['view'])))
  return prepare(rows,True)
 def prefix_kwargs(b,p):
  kw={k:v for k,v in b.forward_kwargs().items() if k!='labels'};kw['use_cache']=False
  for k in ('inputs_embeds','attention_mask','position_ids'):
   if isinstance(kw.get(k),torch.Tensor):kw[k]=kw[k][:,:p]
  return kw
 def capture(kw,p):
  values=[]
  handle=layer.register_forward_pre_hook(lambda m,a:values.append(a[0][0,p-1].detach().cpu().clone()))
  try:
   with torch.no_grad():result=runtime.model(**kw)
   del result
  finally:handle.remove()
  assert len(values)==1;return values[0]
 # All data/token boundaries and all 30 prefix-only states are frozen before any expert training.
 all_directions={};metadata=[]
 for task in cfg['tasks']:
  items=batches(task);states=[];targets=[];layouts=[]
  for k,item in enumerate(items):
   b=r.old.batch_on(runtime,item[1]);kw=b.forward_kwargs();lay=layout(item[2],item[3]);p=lay['prefix_length']
   target=b.labels[0][b.labels[0]!=-100].detach().cpu();targets.append(target)
   assert target[-1]==eos and int((target==eos).sum())==1,'TARGET_EOS'
   assert p==int(torch.where(b.labels[0]!=-100)[0][0]) and lay['visual_start']+lay['visual_count']<=p-1
   assert torch.equal(kw['inputs_embeds'][:,:p].cpu(),item[3][4].float()),'PREFIX_EMBEDDING_BINDING'
   assert kw['inputs_embeds'].shape[1]<runtime.model.config.max_position_embeddings,'CONTEXT_TRUNCATION'
   pre=capture(prefix_kwargs(b,p),p);counts['prefix_extractions']+=1
   teacher=capture(kw,p);assert torch.equal(pre,teacher),'PREFIX_TEACHER_STATE_NOT_EXACT'
   states.append(pre);layouts.append(dict(lay,target_length=len(target),sequence_length=kw['inputs_embeds'].shape[1]))
   del b,kw
  assert all(torch.equal(targets[0],x) for x in targets[1:]),'TARGET_TOKEN_MISMATCH'
  ds=[(v-states[0]).float().detach() for v in states[1:]]
  assert all(torch.isfinite(d).all() and d.norm()>0 for d in ds),'ZERO_OR_NONFINITE_DIRECTION'
  rs=[random_direction(d,task['order'],k+1) for k,d in enumerate(ds)]
  all_directions[task['key']]=dict(directions=ds,random=rs)
  metadata.append(dict(owner=task['order'],layouts=layouts,direction_norms=[float(d.norm()) for d in ds],state_norms=[float(h.norm()) for h in states],cosine=[float(torch.nn.functional.cosine_similarity(states[0][None],v[None])) for v in states[1:]],cancellation_ratio=[float(d.norm()/states[0].norm()) for d in ds]))
  write(ROOT/'PREPARATION_PROGRESS.json',dict(owner=task['order'],counts=counts,time=time.time()))
  del items,states;gc.collect()
 torch.save(all_directions,ROOT/'DIRECTIONS_FROZEN.pt')
 write(ROOT/'TOKEN_STATE_ADMISSION.json',dict(status='PASS',rows=metadata,numpy_version=np.__version__,counts=dict(counts),time=time.time()))
 assert counts['prefix_extractions']==30
 def new_ex(part):
  ex=(TT(225034559) if part==0 else LoRA(225034559)).to(runtime.device)
  if part==0:ex.load_state_dict(r.mm.canonicalize(snapshot(ex))[0])
  return ex
 def update(part,before,g,scalar):
  if part==0:
   raw,ns,direction=proposal(r,before,g,scalar,list(before));return construct(r,before,raw,direction,direction,1.,list(before)),ns
  return lora_step(before,g,scalar)
 def value(ex,h,b,lay,d=None,explicit=False):
  h.set_layout(lay);h.teacher(b.labels,b.forward_kwargs().get('attention_mask'))
  with intervention(layer,lay['prefix_length']-1,d,explicit):loss=runtime.compute_loss(b)
  assert torch.isfinite(loss)
  gs=torch.autograd.grad(loss,tuple(ex.parameters()));counts['backwards']+=1
  g=torch.cat([v.detach().cpu().double().flatten() for v in gs]);assert torch.isfinite(g).all()
  return float(loss.detach()),g
 def logits(h,b,lay,d=None):
  h.set_layout(lay);h.teacher(b.labels,b.forward_kwargs().get('attention_mask'))
  with torch.no_grad(),intervention(layer,lay['prefix_length']-1,d):out=runtime.model(**b.forward_kwargs())
  p=lay['prefix_length'];result=out.logits[0,p-1:p+1].detach().cpu().double();del out;return result
 def emit(h,item,dest):
  assert not dest.exists() and not layer._forward_pre_hooks
  counts['generations']+=1
  if h:h.set_layout(layout(item[2],item[3]))
  answer,trace=r.s.fit.short.generate(runtime,h,item[2],item[3]);tokens=list(answer.raw_token_ids)
  out=dict(binding=item[4],raw_answer=answer.decoded_text,raw_token_ids=tokens,EOS=bool(tokens) and tokens[-1]==eos,at_cap=len(tokens)>=1024,generation_trace=trace,state_intervention_events=0)
  write(dest,out);return out
 def same(a,b):return a[0]==b[0] and torch.equal(a[1],b[1])
 # 360 backward calls: 20 structure/edit pairs x 2 root states x 9 uncached/cached/explicit checks.
 for part in (0,1):
  ex=new_ex(part);initial=snapshot(ex);h=TokenHook(module,ex,'ALL_TOKENS');root=ex.G1 if part==0 else ex.B
  try:
   for task in cfg['tasks']:
    ex.load_state_dict(initial);item=prepare([task['source']],True)[0];b=r.old.batch_on(runtime,item[1]);lay=layout(item[2],item[3]);p=lay['prefix_length']
    ds=[x.to(runtime.device) for x in all_directions[task['key']]['directions']]
    dpath=ROOT/'admission'/str(part)/task['key'];dpath.mkdir(parents=True,exist_ok=True)
    h.set_layout(lay);base=logits(h,b,lay)
    basegen=emit(None,item,dpath/'BASE_SOURCE.json')
    h.attach()
    try:
     assert torch.equal(base,logits(h,b,lay)),'ZERO_BASE_LOGITS'
     zero=emit(h,item,dpath/'ZERO_SOURCE.json');assert zero['raw_token_ids']==basegen['raw_token_ids'],'ZERO_BASE_GENERATION'
     if task['order']==1:assert zero['raw_token_ids']==read(OLD/str(part)/'ZERO_SMOKE.json')['raw_token_ids'],'PARENT_BASE_TOKEN_BINDING'
     for d in ds:
      h.detach();base_d=logits(h,b,lay,d);h.attach();assert torch.equal(base_d,logits(h,b,lay,d)),'ZERO_INTERVENED_BASE'
     gradients=[];rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state().clone()
     for nonzero in (False,True):
      ex.load_state_dict(initial)
      if nonzero:
       with torch.no_grad():root.fill_(.001)
      refs=[value(ex,h,b,lay,d) for d in (None,*ds)]
      assert all(v[1].norm()>0 for v in refs)
      if not nonzero:
       named=list(ex.named_parameters());offset=0
       for name,param in named:
        n=param.numel()
        if param is root:assert refs[0][1][offset:offset+n].norm()>0,'ACTIVE_ROOT_ZERO_GRAD'
        offset+=n
      saved=snapshot(ex)
      with PrefixCache(runtime,[b],21):
       clean1=value(ex,h,b,lay);clean2=value(ex,h,b,lay)
       assert same(refs[0],clean1) and same(clean1,clean2),'CLEAN_CACHE_GRAD'
       averaged=(clean1[1]+clean2[1])/2;assert torch.equal(averaged,refs[0][1])
       scalar=dict(step=0,second_moments=[0.,0.]);one,ss=update(part,saved,refs[0][1],scalar);two,tt=update(part,saved,averaged,scalar)
       assert ss==tt and all(torch.equal(one[k],two[k]) for k in one) and scalar==dict(step=0,second_moments=[0.,0.])
       for i,d in enumerate(ds):
        assert same(refs[i+1],value(ex,h,b,lay,d)),'INJECTED_CACHE_GRAD'
        assert same(refs[i+1],value(ex,h,b,lay,d,True)),'EXPLICIT_HOOK_GRAD'
      assert all(torch.equal(v,saved[k]) for k,v in snapshot(ex).items()),'BRANCH_MUTATES_EXPERT'
      if nonzero:
       # Frozen teacher versus prefix plus real KV at first two prediction positions.
       for d in (None,*ds):
        expected=logits(h,b,lay,d);pre=prefix_kwargs(b,p);pre['use_cache']=True
        with torch.no_grad(),h.generation_request():
         with intervention(layer,p-1,d):out=runtime.model(**pre)
         first=out.logits[0,-1].detach().cpu().double();decode=dict(pre,past_key_values=out.past_key_values)
         kw=b.forward_kwargs();decode['inputs_embeds']=kw['inputs_embeds'][:,p:p+1]
         for k in ('attention_mask','position_ids'):
          if isinstance(kw.get(k),torch.Tensor):decode[k]=kw[k][:,:p+1] if k=='attention_mask' else kw[k][:,p:p+1]
         out=runtime.model(**decode);second=out.logits[0,-1].detach().cpu().double();del out
        torch.testing.assert_close(first,expected[0],rtol=1e-5,atol=5e-4)
        torch.testing.assert_close(second,expected[1],rtol=1e-5,atol=5e-4)
      gradients.append([float(x[1].norm()) for x in refs])
     ex.load_state_dict(initial)
     assert torch.equal(rng,torch.get_rng_state()) and torch.equal(cuda_rng,torch.cuda.get_rng_state()),'RNG_CHANGED'
     torch.save(snapshot(ex),dpath/'state.pt');loaded=torch.load(dpath/'state.pt',map_location=runtime.device,weights_only=True)
     assert all(torch.equal(v,loaded[k]) for k,v in initial.items())
     assert torch.equal(base,logits(h,b,lay)) and not layer._forward_pre_hooks
     write(dpath/'PASS.json',dict(status='PASS',gradient_norms=gradients,counts=dict(counts),time=time.time()))
     # Save/load consumed; no later consumer of temporary admission state.
     size=(dpath/'state.pt').stat().st_size;(dpath/'state.pt').unlink();write(dpath/'RETIREMENT.json',dict(bytes=size,reason='Admission save-load consumed; exact initialization reproducible',count=1))
    finally:h.detach()
    del b,item;gc.collect()
  finally:h.detach()
  del ex;gc.collect();torch.cuda.empty_cache()
 assert counts['backwards']==360 and counts['generations']==40
 write(ROOT/'REAL_ADMISSION.json',dict(status='PASS',counts=dict(counts),time=time.time(),all10_both_structures=True,Base_unique_bound_reuse=183))
 # Full A replay of BOTH structures is a barrier before either B or C.
 for arm in ('A_NATIVE','B_VIEW_CE','C_STATE'):
  if arm!='A_NATIVE':assert read(ROOT/'A_REPLAY_BARRIER.json')['status']=='PASS'
  for part in (0,1):
   ex=new_ex(part);initial=snapshot(ex);h=TokenHook(module,ex,'ALL_TOKENS')
   try:
    for task in cfg['tasks']:
     dest=ROOT/str(part)/task['key']/arm;dest.mkdir(parents=True,exist_ok=True)
     with (dest/'ONCE').open('x'):pass
     ex.load_state_dict(initial);items=batches(task) if arm=='B_VIEW_CE' else prepare([task['source']],True)
     bs=[r.old.batch_on(runtime,i[1]) for i in items];lays=[layout(i[2],i[3]) for i in items]
     ds=[d.to(runtime.device) for d in all_directions[task['key']]['directions']]
     scalar=dict(step=0,second_moments=[0.,0.]);curve=[];h.attach();began=time.time()
     try:
      with PrefixCache(runtime,bs,21):
       for step in range(1,161):
        k=1+(step-1)%2
        first=value(ex,h,bs[0],lays[0])
        second=value(ex,h,bs[k] if arm=='B_VIEW_CE' else bs[0],lays[k] if arm=='B_VIEW_CE' else lays[0],ds[k-1] if arm=='C_STATE' else None)
        if arm=='A_NATIVE':assert same(first,second),'DUPLICATE_BRANCH_CHANGED'
        state,scalar=update(part,snapshot(ex),(first[1]+second[1])/2,scalar);ex.load_state_dict(state);counts['updates']+=1
        curve.append(dict(step=step,loss=first[0],second_loss=second[0],objective=.5*(first[0]+second[0])))
     finally:h.detach()
     if arm=='A_NATIVE':
      old=read(OLD/str(part)/task['key']/'SELECTION_FROZEN.json')['curve']
      assert [dict(step=x['step'],loss=x['loss']) for x in curve]==old,'A_TRAJECTORY_REPLAY_FAILED'
     write(dest/'TRAIN.json',dict(curve=curve,train_seconds=time.time()-began,counts=dict(counts)))
     torch.save(snapshot(ex),dest/'state.pt');loaded=torch.load(dest/'state.pt',map_location=runtime.device,weights_only=True)
     assert all(torch.equal(v,loaded[k]) for k,v in snapshot(ex).items());ex.load_state_dict(loaded)
     del bs,items;gc.collect()
     evaluation=prepare(task['eval_rows']);h.attach()
     try:
      for item in evaluation:
       output=emit(h,item,dest/'outputs'/(item[0]['query_id']+'.json'))
       if arm=='A_NATIVE':
        prior=read(OLD/str(part)/task['key']/'FIXED160/outputs'/(item[0]['query_id']+'.json'))
        assert output['raw_token_ids']==prior['raw_token_ids'],'A_OUTPUT_TOKEN_REPLAY_FAILED'
        assert {k:v for k,v in output['binding'].items() if k!='image_path'}=={k:v for k,v in prior['binding'].items() if k!='image_path'},'A_OUTPUT_BINDING_FAILED'
     finally:h.detach()
     write(dest/'COMPLETE.json',dict(status='PASS',outputs=len(evaluation),save_load_exact=True,replay=arm=='A_NATIVE',counts=dict(counts),time=time.time()))
     write(ROOT/'PROGRESS.json',dict(arm=arm,structure=part,owner=task['order'],counts=dict(counts),time=time.time()))
     del evaluation;gc.collect()
   finally:h.detach()
   del ex;gc.collect();torch.cuda.empty_cache()
  if arm=='A_NATIVE':write(ROOT/'A_REPLAY_BARRIER.json',dict(status='PASS',both_structures=True,each_160_curve_all10=True,each_219_raw_tokens=True,time=time.time()))
 assert counts['updates']==9600 and counts['backwards']==19560 and counts['generations']==1354
 write(ROOT/'MAIN_COMPLETE.json',dict(status='MAIN_OUTPUTS_COMPLETE_SCORING_PENDING',counts=dict(counts),time=time.time(),conditional_R='PENDING_FROZEN_BLIND_DECISION'))
 fh.remove();vh.remove()

if __name__=='__main__':
 code=0
 try:main()
 except BaseException:
  code=1;write(ROOT/'FAILURE.json',dict(error=traceback.format_exc(),counts=counts,time=time.time()));write(ROOT/'STOP',dict(reason='V63_DEPENDENCY_FAILURE'));raise
 finally:
  write(ROOT/'EXIT.json',dict(code=code,counts=counts,seconds=time.time()-START,time=time.time()))
  write(ROOT/'RESOURCE_SCOPE_CLOSED.json',dict(time=time.time(),status='WORKER_EXITED',conditional_stage='Requires scoring decision and same original experiment deadline'))
