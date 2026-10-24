"""Source-informed full TT with fixed Base-surprisal token loss with fixed view CE and all-token writer."""
import os,json,time,signal,sys,traceback,shutil,gc
from pathlib import Path
ROOT=Path(os.environ['V73_ROOT']);P=Path(os.environ['PARENT_RUN']);OLD=Path(os.environ['V62_PARENT']);START=time.time()
counts=dict(forwards=0,backwards=0,updates=0,generations=0,vision_forwards=0,initialization_backwards=0)
if (ROOT/'PREVIOUS_COUNTS.json').exists():counts.update(json.loads((ROOT/'PREVIOUS_COUNTS.json').read_text()))
prior_counts=dict(counts)

def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
limits=read(ROOT/'RESOURCE_LIMITS.json');last_space=0.
def budget():
 global last_space
 assert time.time()<limits['hard_end']-300 and not (ROOT/'STOP').exists(),'WINDOW_STOP'
 assert counts['backwards']<=3400 and counts['updates']<=1600 and counts['generations']<=230 and counts['forwards']<400000
 assert time.time()-read(ROOT/'RESOURCE_GUARD_HEALTH.json')['time']<120,'GUARD_STALE'
 if time.time()-last_space>30:
  assert shutil.disk_usage(ROOT).free>8*1024**3
  assert sum(x.stat().st_size for x in ROOT.rglob('*') if x.is_file() and not x.is_symlink())<2500000000
  last_space=time.time();write(ROOT/'COUNTS.json',dict(counts,time=time.time()))

def main():
 with (ROOT/'WORKER_ONCE').open('x'):pass
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('V73 finite deadline')))
 signal.alarm(max(1,int(limits['hard_end']-time.time()-300)))
 sys.path[:0]=[str(ROOT/'tools'),str(P/'private/tools'),str(P/'private/source'),str(P/'private/official_llava')]
 import torch
 import module_run as r
 from types import SimpleNamespace
 from scripts.medtrace.run_realmodel_core import load_real_runtime
 from alignment import TT,snapshot
 from token_hook import TokenHook,layout
 from path_v18 import proposal,construct
 from v51_cache import PrefixCache
 from state_patch import intervention
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
 # No hidden-state extraction or injection in this independent input-coverage experiment.
 metadata=[];frozen_weights={}
 from weighted_loss import token_nll,weights,loss as weighted_loss
 weight_lookup={}
 for task in cfg['tasks']:
  items=batches(task);targets=[];layouts=[];task_weights=[]
  for item in items:
   b=r.old.batch_on(runtime,item[1]);kw=b.forward_kwargs();lay=layout(item[2],item[3]);p=lay['prefix_length']
   target=b.labels[0][b.labels[0]!=-100].detach().cpu();targets.append(target)
   with torch.no_grad():
    out=runtime.model(**kw);nll=token_nll(out.logits,b.labels)
    torch.testing.assert_close(nll.mean(),out.loss,rtol=1e-6,atol=1e-7)
    task_weights.append(weights(nll))
    del out,nll
   assert target[-1]==eos and int((target==eos).sum())==1
   assert p==int(torch.where(b.labels[0]!=-100)[0][0]) and lay['visual_start']+lay['visual_count']<=p-1
   assert torch.equal(kw['inputs_embeds'][:,:p].cpu(),item[3][4].float())
   assert kw['inputs_embeds'].shape[1]<runtime.model.config.max_position_embeddings
   layouts.append(dict(lay,target_length=len(target),sequence_length=kw['inputs_embeds'].shape[1]))
   del b,kw
  assert all(torch.equal(targets[0],x) for x in targets[1:]),'TARGET_TOKEN_MISMATCH'
  frozen_weights[task['key']]=task_weights
  metadata.append(dict(owner=task['order'],layouts=layouts))
  del items;gc.collect()
 torch.save(frozen_weights,ROOT/'TOKEN_WEIGHTS_FROZEN.pt')
 write(ROOT/'TOKEN_WEIGHTS_FROZEN.json',dict(status='PASS',source='Unedited Base teacher-forced legal source answers only',rows=[dict(owner=t['order'],views=[dict(count=len(w),min=float(w.min()),max=float(w.max()),zero_tokens=int((w==0).sum()),mean=float(w.mean())) for w in frozen_weights[t['key']]]) for t in cfg['tasks']],time=time.time(),counts=dict(counts)))
 write(ROOT/'TOKEN_ADMISSION.json',dict(status='PASS',rows=metadata,time=time.time(),counts=dict(counts)))
 def new_ex(part):
  assert part==0
  ex=TT(225034559).to(runtime.device)
  ex.load_state_dict(r.mm.canonicalize(snapshot(ex))[0])
  return ex
 # Reuse the immutable source-supervised initializers from V69, not trained weights.
 assert read(ROOT/'INITIALIZER_SOURCE_DATA.json')==cfg
 assert read(ROOT/'INITIALIZER_SOURCE_VIEWS.json')['views']==views
 initializers=torch.load(ROOT/'INITIALIZERS_INPUT.pt',map_location='cpu',weights_only=True)
 assert set(initializers)=={t['key'] for t in cfg['tasks']}
 for state in initializers.values():
  assert state['G1'].count_nonzero()==0 and sum(v.numel() for v in state.values())==7168
  assert all(torch.isfinite(v).all() for v in state.values())
 torch.save(initializers,ROOT/'INITIALIZERS_FROZEN.pt')
 write(ROOT/'INITIALIZATION_FROZEN.json',dict(status='PASS',reused_source='V69_INITIALIZERS_FROZEN',all10=True,counts=dict(counts),time=time.time()))
 def update(part,before,g,scalar):
  assert part==0
  raw,ns,delta=proposal(r,before,g,scalar,list(before))
  return construct(r,before,raw,delta,delta,1.,list(before)),ns
 def value(ex,h,b,lay,d=None,explicit=False):
  h.set_layout(lay);h.teacher(b.labels,b.forward_kwargs().get('attention_mask'))
  with intervention(layer,lay['prefix_length']-1,d,explicit):
   output=runtime.model(**b.forward_kwargs());loss=weighted_loss(output.logits,b.labels,weight_lookup[id(b)]);del output
  assert torch.isfinite(loss)
  gs=torch.autograd.grad(loss,tuple(ex.parameters()));counts['backwards']+=1
  g=torch.cat([v.detach().cpu().double().flatten() for v in gs]);assert torch.isfinite(g).all()
  return float(loss.detach()),g
 def logits(h,b,lay,d=None):
  h.set_layout(lay);h.teacher(b.labels,b.forward_kwargs().get('attention_mask'))
  with torch.no_grad(),intervention(layer,lay['prefix_length']-1,d):out=runtime.model(**b.forward_kwargs())
  p=lay['prefix_length'];result=out.logits[0,p-1:p+1].detach().cpu().double();del out;return result
 def emit(h,item,dest,zero=False):
  assert not dest.exists() and not layer._forward_pre_hooks
  counts['generations']+=1
  if h:h.set_layout(layout(item[2],item[3]))
  if zero:
   assert h is not None and h.expert.G1.count_nonzero()==0
   with h.generation_request():answer,_=r.s.fit.short.generate(runtime,None,item[2],item[3])
   trace=list(h.last_generation_trace);assert trace and all(x['active_residual_norm']==0 for x in trace)
  else:answer,trace=r.s.fit.short.generate(runtime,h,item[2],item[3])
  tokens=list(answer.raw_token_ids)
  out=dict(binding=item[4],raw_answer=answer.decoded_text,raw_token_ids=tokens,EOS=bool(tokens) and tokens[-1]==eos,at_cap=len(tokens)>=1024,generation_trace=trace,state_intervention_events=0)
  write(dest,out);return out
 def same(a,b):return a[0]==b[0] and torch.equal(a[1],b[1])
 # All10 edits x all3 view bindings x 2 root states x 3 evaluations =180 backwards.
 ex=new_ex(0);initial=snapshot(ex);h=TokenHook(module,ex,'ALL_TOKENS');root=ex.G1
 try:
  for task in cfg['tasks']:
   initial={k:v.to(runtime.device) for k,v in initializers[task['key']].items()}
   items=batches(task);bs=[r.old.batch_on(runtime,i[1]) for i in items];lays=[layout(i[2],i[3]) for i in items]
   weight_lookup={id(b):w for b,w in zip(bs,frozen_weights[task['key']])}
   dest=ROOT/'admission'/task['key'];dest.mkdir(parents=True,exist_ok=True)
   ex.load_state_dict(initial);base=[logits(h,b,lay) for b,lay in zip(bs,lays)]
   if task['order']==1:basegen=emit(None,items[0],dest/'BASE_SOURCE.json')
   h.attach()
   try:
    assert all(torch.equal(ref,logits(h,b,lay)) for ref,b,lay in zip(base,bs,lays)),'ZERO_BASE_LOGITS'
    if task['order']==1:
     zero=emit(h,items[0],dest/'ZERO_SOURCE.json',zero=True)
     assert zero['raw_token_ids']==basegen['raw_token_ids']==read(OLD/'0/ZERO_SMOKE.json')['raw_token_ids']
    rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state().clone()
    for nonzero in (False,True):
     ex.load_state_dict(initial)
     if nonzero:
      with torch.no_grad():root.fill_(.001)
     refs=[value(ex,h,b,lay) for b,lay in zip(bs,lays)]
     assert all(v[1].norm()>0 for v in refs)
     if not nonzero:
      offset=0
      for name,param in ex.named_parameters():
       n=param.numel()
       if param is root:assert all(v[1][offset:offset+n].norm()>0 for v in refs)
       offset+=n
     saved=snapshot(ex)
     with PrefixCache(runtime,bs,21):
      for b,lay,ref in zip(bs,lays,refs):
       a=value(ex,h,b,lay);c=value(ex,h,b,lay)
       assert same(ref,a) and same(a,c),'CACHE_LOSS_GRADIENT_FAILED'
       avg=(a[1]+c[1])/2;assert torch.equal(avg,ref[1])
       scalar=dict(step=0,second_moments=[0.,0.]);one,ss=update(0,saved,ref[1],scalar);two,tt=update(0,saved,avg,scalar)
       assert ss==tt and all(torch.equal(one[k],two[k]) for k in one)
       assert scalar==dict(step=0,second_moments=[0.,0.])
     assert all(torch.equal(v,saved[k]) for k,v in snapshot(ex).items())
     if nonzero:
      for b,lay in zip(bs,lays):
       p=lay['prefix_length'];expected=logits(h,b,lay);pre=prefix_kwargs(b,p);pre['use_cache']=True
       with torch.no_grad(),h.generation_request():
        out=runtime.model(**pre);first=out.logits[0,-1].detach().cpu().double()
        kw=b.forward_kwargs();decode=dict(pre,past_key_values=out.past_key_values,inputs_embeds=kw['inputs_embeds'][:,p:p+1])
        for k in ('attention_mask','position_ids'):
         if isinstance(kw.get(k),torch.Tensor):decode[k]=kw[k][:,:p+1] if k=='attention_mask' else kw[k][:,p:p+1]
        out=runtime.model(**decode);second=out.logits[0,-1].detach().cpu().double();del out
       torch.testing.assert_close(first,expected[0],rtol=1e-5,atol=5e-4)
       torch.testing.assert_close(second,expected[1],rtol=1e-5,atol=5e-4)
    ex.load_state_dict(initial)
    assert torch.equal(rng,torch.get_rng_state()) and torch.equal(cuda_rng,torch.cuda.get_rng_state())
    torch.save(snapshot(ex),dest/'state.pt');loaded=torch.load(dest/'state.pt',map_location=runtime.device,weights_only=True)
    assert all(torch.equal(v,loaded[k]) for k,v in initial.items())
    assert all(torch.equal(ref,logits(h,b,lay)) for ref,b,lay in zip(base,bs,lays))
    size=(dest/'state.pt').stat().st_size;(dest/'state.pt').unlink()
    write(dest/'PASS.json',dict(status='PASS',all_three_bindings=True,save_load_exact=True,temporary_state_retired_bytes=size,counts=dict(counts),time=time.time()))
   finally:h.detach()
   write(ROOT/'PROGRESS.json',dict(phase='REAL_ADMISSION',owner=task['order'],counts=dict(counts),time=time.time()))
   del bs,items;gc.collect()
 finally:h.detach()
 del ex;gc.collect();torch.cuda.empty_cache()
 assert counts['backwards']-prior_counts['backwards']==180 and counts['generations']-prior_counts['generations']==2
 write(ROOT/'REAL_ADMISSION.json',dict(status='PASS',counts=dict(counts),time=time.time(),all10_all3_bindings=True))
 # Original A/B outputs are frozen historical comparators.
 for arm in ('B_VIEW_CE',):
  assert read(ROOT/'PARENT_COMPARISON_ADMISSION.json')['status']=='PASS'
  for part in (0,):
   ex=new_ex(part);initial=snapshot(ex);h=TokenHook(module,ex,'ALL_TOKENS')
   try:
    for task in cfg['tasks']:
     initial={k:v.to(runtime.device) for k,v in initializers[task['key']].items()}
     dest=ROOT/str(part)/task['key']/arm;dest.mkdir(parents=True,exist_ok=True)
     with (dest/'ONCE').open('x'):pass
     ex.load_state_dict(initial);items=batches(task) if arm=='B_VIEW_CE' else prepare([task['source']],True)
     bs=[r.old.batch_on(runtime,i[1]) for i in items];lays=[layout(i[2],i[3]) for i in items]
     weight_lookup={id(b):w for b,w in zip(bs,frozen_weights[task['key']])}
     scalar=dict(step=0,second_moments=[0.,0.]);curve=[];h.attach();began=time.time()
     try:
      with PrefixCache(runtime,bs,21):
       for step in range(1,161):
        k=1+(step-1)%2
        first=value(ex,h,bs[0],lays[0])
        second=value(ex,h,bs[k] if arm=='B_VIEW_CE' else bs[0],lays[k] if arm=='B_VIEW_CE' else lays[0])
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
  if arm=='A_NATIVE':write(ROOT/'A_REPLAY_BARRIER.json',dict(status='PASS',TT_only=True,each_160_curve_all10=True,each_219_raw_tokens=True,time=time.time()))
 assert counts['updates']-prior_counts['updates']==1600 and counts['backwards']-prior_counts['backwards']==3380 and counts['generations']-prior_counts['generations']==221
 write(ROOT/'MAIN_COMPLETE.json',dict(status='MAIN_OUTPUTS_COMPLETE_SCORING_PENDING',counts=dict(counts),time=time.time(),state_injection_tested=False))
 fh.remove();vh.remove()

if __name__=='__main__':
 code=0
 try:main()
 except BaseException:
  code=1;write(ROOT/'FAILURE.json',dict(error=traceback.format_exc(),counts=counts,time=time.time()));write(ROOT/'STOP',dict(reason='V73_DEPENDENCY_FAILURE'));raise
 finally:
  write(ROOT/'EXIT.json',dict(code=code,counts=counts,seconds=time.time()-START,time=time.time()))
  write(ROOT/'RESOURCE_SCOPE_CLOSED.json',dict(time=time.time(),status='WORKER_EXITED',scoring='PENDING' ))
