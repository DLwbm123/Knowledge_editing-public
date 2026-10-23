"""Fixed TT and BalancEdit bank validation with frozen shared R0 routing."""
import os,json,time,signal,sys,traceback,shutil,gc
from pathlib import Path
ROOT=Path(os.environ['V80_ROOT']);P=Path(os.environ['PARENT_RUN']);OLD=Path(os.environ['V62_PARENT']);V69=Path(os.environ['V69_PARENT']);START=time.time()
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
 assert counts['backwards']<=4200 and counts['updates']<=2100 and counts['generations']<=1500 and counts['forwards']<1700000
 assert time.time()-read(ROOT/'RESOURCE_GUARD_HEALTH.json')['time']<120,'GUARD_STALE'
 if time.time()-last_space>30:
  assert shutil.disk_usage(ROOT).free>8*1024**3
  assert sum(x.stat().st_size for x in ROOT.rglob('*') if x.is_file() and not x.is_symlink())<limits['soft_stop_bytes']
  last_space=time.time();write(ROOT/'COUNTS.json',dict(counts,time=time.time()))

def main():
 with (ROOT/'WORKER_ONCE').open('x'):pass
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('V80 finite deadline')))
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
 assert read(ROOT/'UNIFIED_RESULTS.json')['status']=='UNIFIED_FULL_TEXT_SCORED'
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
 metadata=[]
 for task in cfg['tasks']:
  items=batches(task);targets=[];layouts=[]
  for item in items:
   b=r.old.batch_on(runtime,item[1]);kw=b.forward_kwargs();lay=layout(item[2],item[3]);p=lay['prefix_length']
   target=b.labels[0][b.labels[0]!=-100].detach().cpu();targets.append(target)
   assert target[-1]==eos and int((target==eos).sum())==1
   assert p==int(torch.where(b.labels[0]!=-100)[0][0]) and lay['visual_start']+lay['visual_count']<=p-1
   assert torch.equal(kw['inputs_embeds'][:,:p].cpu(),item[3][4].float())
   assert kw['inputs_embeds'].shape[1]<runtime.model.config.max_position_embeddings
   layouts.append(dict(lay,target_length=len(target),sequence_length=kw['inputs_embeds'].shape[1]))
   del b,kw
  assert all(torch.equal(targets[0],x) for x in targets[1:]),'TARGET_TOKEN_MISMATCH'
  metadata.append(dict(owner=task['order'],layouts=layouts))
  del items;gc.collect()
 write(ROOT/'TOKEN_ADMISSION.json',dict(status='PASS',rows=metadata,time=time.time(),counts=dict(counts)))
 def new_ex(part):
  assert part==0, 'No LoRA arm authorized'
  ex=TT(225034559).to(runtime.device)
  if part==0:ex.load_state_dict(r.mm.canonicalize(snapshot(ex))[0])
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
  with intervention(layer,lay['prefix_length']-1,d,explicit):loss=runtime.compute_loss(b)
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
   assert h is not None and (h.expert.G1 if hasattr(h.expert,'G1') else h.expert.B).count_nonzero()==0
   with h.generation_request():answer,_=r.s.fit.short.generate(runtime,None,item[2],item[3])
   trace=list(h.last_generation_trace);assert trace and all(x['active_residual_norm']==0 for x in trace)
  else:answer,trace=r.s.fit.short.generate(runtime,h,item[2],item[3])
  tokens=list(answer.raw_token_ids)
  out=dict(binding=item[4],raw_answer=answer.decoded_text,raw_token_ids=tokens,EOS=bool(tokens) and tokens[-1]==eos,at_cap=len(tokens)>=1024,generation_trace=trace,state_intervention_events=0)
  write(dest,out);return out
 def same(a,b):return a[0]==b[0] and torch.equal(a[1],b[1])
 def admit(part):
  # All10 edits x all3 view bindings x 2 root states x 3 evaluations =180 backwards.
  ex=new_ex(part);h=TokenHook(module,ex,'ALL_TOKENS');root=ex.G1 if part==0 else ex.B
  admission_before=dict(counts)
  try:
   for task in cfg['tasks']:
    initial={k:v.to(runtime.device) for k,v in initializers[task['key']].items()} if part==0 else snapshot(new_ex(part))
    items=batches(task);bs=[r.old.batch_on(runtime,i[1]) for i in items];lays=[layout(i[2],i[3]) for i in items]
    dest=ROOT/'admission'/str(part)/task['key'];dest.mkdir(parents=True,exist_ok=True)
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
      with PrefixCache(runtime,bs,21) as cache:
       for b,lay,ref in zip(bs,lays,refs):
        a=value(ex,h,b,lay);c=value(ex,h,b,lay)
        assert same(ref,a) and same(a,c),'CACHE_LOSS_GRADIENT_FAILED'
        avg=(a[1]+c[1])/2;assert torch.equal(avg,ref[1])
        scalar=dict(step=0,second_moments=[0.,0.]);one,ss=update(part,saved,ref[1],scalar);two,tt=update(part,saved,avg,scalar)
        assert ss==tt and all(torch.equal(one[k],two[k]) for k in one)
        assert scalar==dict(step=0,second_moments=[0.,0.])
      assert cache.misses==3 and cache.hits==3,'REAL_CACHE_NOT_EXERCISED'
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
     write(dest/'PASS.json',dict(status='PASS',all_three_bindings=True,save_load_exact=True,temporary_state_retired_bytes=size,cache_misses_per_state=3,cache_hits_per_state=3,counts=dict(counts),time=time.time()))
    finally:h.detach()
    write(ROOT/'PROGRESS.json',dict(phase='REAL_ADMISSION',owner=task['order'],counts=dict(counts),time=time.time()))
    del bs,items;gc.collect()
  finally:h.detach()
  del ex;gc.collect();torch.cuda.empty_cache()
  assert counts['backwards']-admission_before['backwards']==180 and counts['generations']-admission_before['generations']==2
  write(ROOT/('REAL_ADMISSION_%d.json'%part),dict(status='PASS',counts=dict(counts),time=time.time(),all10_all3_bindings=True))
 # Reconstruct retired TT states for the registered bank consumer. No separate replay experiment.
 admit(0)
 for task in cfg['tasks']:
  dest=ROOT/'TT_V69'/task['key'];dest.mkdir(parents=True,exist_ok=True)
  with (dest/'ONCE').open('x'):pass
  ex=new_ex(0);ex.load_state_dict(initializers[task['key']]);h=TokenHook(module,ex,'ALL_TOKENS')
  items=batches(task);bs=[r.old.batch_on(runtime,i[1]) for i in items];lays=[layout(i[2],i[3]) for i in items];scalar=dict(step=0,second_moments=[0.,0.]);curve=[];began=time.time();h.attach()
  try:
   with PrefixCache(runtime,bs,21) as cache:
    for step in range(1,161):
     k=1+(step-1)%2;first=value(ex,h,bs[0],lays[0]);second=value(ex,h,bs[k],lays[k]);state,scalar=update(0,snapshot(ex),(first[1]+second[1])/2,scalar);ex.load_state_dict(state);counts['updates']+=1
     curve.append(dict(step=step,loss=first[0],second_loss=second[0],objective=.5*(first[0]+second[0])))
  finally:h.detach()
  assert cache.misses==3 and cache.hits==317
  assert curve==read(V69/'0'/task['key']/'B_VIEW_CE/TRAIN.json')['curve'],'REBUILD_TRAJECTORY_MISMATCH'
  torch.save(snapshot(ex),dest/'state.pt');loaded=torch.load(dest/'state.pt',map_location='cpu',weights_only=True);assert all(torch.equal(v,loaded[k]) for k,v in snapshot(ex).items())
  write(dest/'TRAIN.json',dict(status='PASS',curve=curve,cache_misses=cache.misses,cache_hits=cache.hits,save_load_exact=True,train_seconds=time.time()-began,counts=dict(counts)))
  write(ROOT/'PROGRESS.json',dict(phase='TT_RECONSTRUCTION',owner=task['order'],counts=dict(counts),time=time.time()))
  del ex,h,bs,items;gc.collect();torch.cuda.empty_cache()
 # Standard original BalancEdit only, no shared-support CE or comparator tuning.
 import be_v80
 from m3bench_repro.editors.methods import BalanceEditPaperSpecEditor
 from m3bench_repro.editors.routing import MemoryRouter,decision_as_json
 from dataclasses import replace
 BEOLD=Path(os.environ['V22_PARENT'])
 def save(p,x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);torch.save(x,p)
 for task in cfg['tasks']:
  dest=ROOT/'BALANCEDIT'/task['key'];source=prepare([task['source']],True);batch=r.old.batch_on(runtime,source[0][1]);began=time.time()
  be_v80.train(runtime,r,template,task,batch,[],0,dest,counts,budget,save,write)
  tr=read(dest/'BALANCEDIT_TRAIN.json');old=read(BEOLD/str((task['order']-1)%5)/task['key']/'BALANCEDIT_TRAIN.json')
  assert tr['curve']==old['curve'] and tr['parameter_count']==58720256 and tr['config']==old['config'],'STANDARD_BE_RECONSTRUCTION_MISMATCH'
  tr['train_seconds_with_anchors_and_save']=time.time()-began;write(dest/'BALANCEDIT_TRAIN.json',tr)
  write(ROOT/'PROGRESS.json',dict(phase='BE_RECONSTRUCTION',owner=task['order'],counts=dict(counts),time=time.time()))
  del source,batch;gc.collect();torch.cuda.empty_cache()
 assert counts['updates']==2100 and counts['backwards']==3880
 # Common Base-only keys; answer text and evaluation owner never enter route selection.
 query_rows={q['query_id']:q for t in cfg['tasks'] for q in t['eval_rows']};assert len(query_rows)==183
 editor=BalanceEditPaperSpecEditor(runtime);target=editor.target;keys={};key_seconds={};anchors=[]
 try:
  assert target=='model.layers.31.mlp.up_proj'
  for task in cfg['tasks']:
   st=torch.load(ROOT/'BALANCEDIT'/task['key']/'BALANCEDIT.pt',map_location='cpu',weights_only=True)
   assert len(st['router']['entries'])==1;entry=st['router']['entries'][0];assert entry['logical_edit_id']==task['key'];anchors.append(entry)
  for q,row in sorted(query_rows.items()):
   budget();began=time.perf_counter();record=r.s.record(template,row);key=editor._question_key(record);torch.cuda.synchronize();key_seconds[q]=time.perf_counter()-began;keys[q]=key.detach().cpu()
  for task,entry in zip(cfg['tasks'],anchors):assert float((keys[task['source']['query_id']]-entry['key'].cpu()).abs().max())<=1e-6,'R0_NATIVE_KEY_MISMATCH'
 finally:runtime.replace_module(target,editor.wrapper.base)
 del editor;gc.collect();torch.cuda.empty_cache()
 torch.save(dict(keys=keys,anchors=anchors),ROOT/'ROUTE_KEYS.pt');write(ROOT/'KEY_COST.json',dict(query_count=len(keys),seconds=key_seconds,reference_or_owner_in_route=False))
 routes={}
 for n in (1,5,10):
  router=MemoryRouter('euclidean')
  for e in anchors[:n]:router.add(e['logical_edit_id'],e['key'],e['radius'],e['label'])
  qs=sorted({q['query_id'] for t in cfg['tasks'][:n] for q in t['eval_rows']})
  routes[n]={q:decision_as_json(router.route(keys[q])) for q in qs}
 write(ROOT/'ROUTES_FROZEN.json',dict(status='FROZEN',prefixes=routes,route='Original R0 nearest then own-radius alpha.2',time=time.time()))
 # For each arm and prefix, all inserted experts actually reside on GPU; no offload simulation.
 bank_cost=[]
 for arm in ('TT_V69','BALANCEDIT'):
  for n in (1,5,10):
   budget();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();began=time.time();base_allocated=torch.cuda.memory_allocated()
   members={};ed=None;hooks={};member_bytes=0
   if arm=='TT_V69':
    for task in cfg['tasks'][:n]:
     ex=new_ex(0);ex.load_state_dict(torch.load(ROOT/arm/task['key']/'state.pt',map_location='cpu',weights_only=True));ex.requires_grad_(False);members[task['key']]=ex;hooks[task['key']]=TokenHook(module,ex,'ALL_TOKENS');member_bytes+=sum(v.numel()*v.element_size() for v in ex.parameters())
    del ex
   else:
    ed=BalanceEditPaperSpecEditor(runtime)
    for task in cfg['tasks'][:n]:
     state=torch.load(ROOT/arm/task['key']/'BALANCEDIT.pt',map_location='cpu',weights_only=True)
     e=ed.wrapper.add_edit(task['key']);e.load_state_dict(state['wrapper']['edits'][task['key']]);e.requires_grad_(False);member_bytes+=sum(v.numel()*v.element_size() for v in e.parameters())
    del state,e
   resident_allocated=torch.cuda.memory_allocated();outroot=ROOT/'bank'/arm/str(n)
   def bank_prepare(rows):
    # Temporarily detach full-linear bank before FP16 preprocessing so stored FP32 experts are never rounded.
    if ed:runtime.replace_module(target,ed.wrapper.base)
    try:return prepare(rows)
    finally:
     if ed:runtime.replace_module(target,ed.wrapper)
   def generate_selected(selected,item,path):
    if arm=='TT_V69':
     h=hooks.get(selected)
     if h:h.attach()
     try:return emit(h,item,path)
     finally:
      if h:h.detach()
    ed.wrapper.set_active(selected)
    try:return emit(None,item,path)
    finally:ed.wrapper.set_active(None)
   cache_outputs={};consumer=[];route_n=routes[n];qowners={q:[] for q in route_n}
   for task in cfg['tasks'][:n]:
    for q in task['eval_rows']:qowners[q['query_id']].append(task['key'])
   for qi,q in enumerate(sorted(route_n)):
    item=bank_prepare([query_rows[q]])[0];selected=route_n[q]['logical_edit_id'];choices=list(dict.fromkeys(qowners[q]+[selected]))
    for chosen in choices:
     dest=outroot/'outputs'/('BASE' if chosen is None else chosen)/(q+'.json');torch.cuda.synchronize();t=time.perf_counter();o=generate_selected(chosen,item,dest);torch.cuda.synchronize();elapsed=time.perf_counter()-t
     cache_outputs[chosen,q]=(o,dest);write(dest.with_suffix('.cost.json'),dict(generation_seconds_prepared_input=elapsed,tokens=len(o['raw_token_ids'])))
     if chosen in qowners[q]:
      ref=(V69/'0'/chosen/'B_VIEW_CE/outputs'/(q+'.json')) if arm=='TT_V69' else (BEOLD/str((int(chosen[1:])-1)%5)/chosen/'outputs/BALANCEDIT'/(q+'.json'))
      if ref.exists():
       prior=read(ref);assert prior['raw_token_ids']==o['raw_token_ids'] and all(prior['binding'][k]==o['binding'][k] for k in ['question','reference','image_sha256','prompt_ids','generation']),'DESIGNATED_BANK_SINGLE_MISMATCH'
     elif chosen is None:
      prior=read(OLD/'base'/(q+'.json'));assert prior['raw_token_ids']==o['raw_token_ids'],'BANK_BASE_MISMATCH'
    for owner in qowners[q]:consumer.append(dict(owner=owner,query=q,condition='DESIGNATED',selected=owner,output=str(cache_outputs[owner,q][1].relative_to(ROOT))))
    consumer.append(dict(query=q,condition='NATURAL',selected=selected,output=str(cache_outputs[selected,q][1].relative_to(ROOT))))
    # No model call: same R0 route, source answer for activation and the same Base fallback.
    answer=next(t['source']['reference'] for t in cfg['tasks'][:n] if t['key']==selected) if selected else cache_outputs[None,q][0]['raw_answer']
    write(outroot/'retrieval'/(q+'.json'),dict(query=q,selected=selected,raw_answer=answer,binding=item[4],route=route_n[q],oracle=False,model_calls=0,Base_fallback=selected is None))
    del item
    if qi%10==0:write(ROOT/'PROGRESS.json',dict(phase='REAL_BANK',arm=arm,prefix=n,query=qi,total=len(route_n),counts=dict(counts),time=time.time()))
   write(outroot/'CONSUMERS.json',consumer)
   # Actual uncached end-to-end requests: source and first/last T2G, fixed before observing outputs.
   first=cfg['tasks'][0];last=cfg['tasks'][n-1];latqs=[first['source']['query_id'],next(q['query_id'] for q in first['eval_rows'] if 'T2G' in q['roles']),[q['query_id'] for q in last['eval_rows'] if 'T2G' in q['roles']][-1]];assert len(set(latqs))==3
   lat=[]
   probe=BalanceEditPaperSpecEditor(runtime) if ed is None else ed
   # TT hook is inactive here; all routing features are extracted with the Base branch.
   for j,q in enumerate([latqs[0]]+latqs):
    torch.cuda.synchronize();t=time.perf_counter();k=probe._question_key(r.s.record(template,query_rows[q]));router=MemoryRouter('euclidean')
    for e in anchors[:n]:router.add(e['logical_edit_id'],e['key'],e['radius'],e['label'])
    decision=router.route(k.detach().cpu());assert decision.logical_edit_id==routes[n][q]['logical_edit_id'];item=bank_prepare([query_rows[q]])[0]
    o=generate_selected(decision.logical_edit_id,item,outroot/'latency'/('%02d.json'%j));torch.cuda.synchronize();seconds=time.perf_counter()-t;assert o['raw_token_ids']==cache_outputs[decision.logical_edit_id,q][0]['raw_token_ids']
    lat.append(dict(query=q,warmup=j==0,seconds=seconds,tokens=len(o['raw_token_ids']),components='uncached Base key+R0+FP16 visual preparation+greedy generation; no disk write excluded'))
   if ed is None:runtime.replace_module(target,probe.wrapper.base)
   del probe,item,k,router
   cost=dict(arm=arm,prefix=n,resident_parameter_bytes=member_bytes,base_allocated=base_allocated,resident_allocated=resident_allocated,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),phase_seconds=time.time()-began,latency=lat,all_members_on_gpu=True,consumer_count=len(consumer),counts=dict(counts))
   bank_cost.append(cost);write(outroot/'COMPLETE.json',dict(status='PASS',cost=cost,all_single_designated_checked=True))
   if ed:runtime.replace_module(target,ed.wrapper.base)
   hooks.clear();members.clear();ed=None;cache_outputs.clear();gc.collect();torch.cuda.empty_cache()
 write(ROOT/'BANK_COST.json',bank_cost)
 assert counts['updates']==2100 and counts['backwards']==3880
 write(ROOT/'MAIN_COMPLETE.json',dict(status='ALL_BANK_CONSUMERS_COMPLETE_SCORING_PENDING',counts=dict(counts),time=time.time(),prefixes=[1,5,10],arms=['TT_V69','BALANCEDIT'],retrieval=True,LoRA=False))
 fh.remove();vh.remove()

if __name__=='__main__':
 code=0
 try:main()
 except BaseException:
  code=1;write(ROOT/'FAILURE.json',dict(error=traceback.format_exc(),counts=counts,time=time.time()));write(ROOT/'STOP',dict(reason='V80_DEPENDENCY_FAILURE'));raise
 finally:
  write(ROOT/'EXIT.json',dict(code=code,counts=counts,seconds=time.time()-START,time=time.time()));write(ROOT/'RESOURCE_SCOPE_CLOSED.json',dict(time=time.time(),status='WORKER_EXITED',scoring='PENDING'))
