"""One SOURCE expert or one disjoint Base feature shard; fixed bounded consumers."""
import os,sys,json,time,traceback,subprocess,shutil,copy
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(os.environ['V20_ROOT']);OUT=Path(os.environ['OUTPUT_RUN']);P=Path(os.environ['PARENT_RUN']);START=time.time();last_space=0.
MODE=os.environ['V20_MODE'];INDEX=int(os.environ['V20_INDEX'])
counts=dict(forwards=0,backwards=0,updates=0,generations=0,vision_forwards=0,query_keys=0,anchor_keys=0)
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
def budget():
 global last_space
 assert time.time()<float(os.environ['DEADLINE_EPOCH']) and not (ROOT/'STOP').exists()
 assert shutil.disk_usage(ROOT).free>8*1024**3
 if MODE=='SOURCE':assert counts['forwards']<=35000 and counts['backwards']<=161 and counts['updates']<=160 and counts['generations']<=28
 else:assert MODE=='KEYS' and counts['forwards']<=20 and counts['generations']==counts['backwards']==counts['updates']==0
 if time.time()-last_space>60:
  assert sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())+1000000<16000000000
  last_space=time.time()
def main():
 with (OUT/'ATTEMPT').open('x'):pass
 sys.path[:0]=[str(ROOT/'tools'),str(P/'private/tools'),str(P/'private/source'),str(P/'private/official_llava')]
 import torch
 import module_run as r
 from scripts.medtrace.run_realmodel_core import load_real_runtime
 from m3bench_repro.editors.llava_runtime import seed_everything
 from path_v20 import flatten,proposal,construct,same
 from cache_v20 import PrefixCache
 r.c.budget=budget
 torch.use_deterministic_algorithms(True);torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True);torch.backends.cuda.matmul.allow_tf32=False
 budget();runtime=load_real_runtime(SimpleNamespace(cpu_gate=P/'private/cpu_gate'));runtime.model.eval();r.old.fp32(runtime)
 assert not any(p.requires_grad for p in runtime.model.parameters())
 def count(*_):
  budget();assert counts['forwards']<(35000 if MODE=='SOURCE' else 20);counts['forwards']+=1
 h=runtime.model.register_forward_pre_hook(count);vh=runtime.model.get_vision_tower().register_forward_pre_hook(lambda *_:counts.__setitem__('vision_forwards',counts['vision_forwards']+1))
 write(OUT/'MODEL_READY.json',dict(time=time.time(),mode=MODE,gpu=os.environ['CUDA_VISIBLE_DEVICES'],command=subprocess.check_output(['ps','-ww','-p',str(os.getpid()),'-o','args='],text=True).strip()))
 config=read(ROOT/'MATRIX_CONFIG.json');template=read(P/'private/M13_SOURCE_PACKET.json')['original_tasks'][4];bindings=read(P/'private/EVAL_BINDINGS.json');assert runtime.generation_config==next(iter(bindings.values()))['generation']
 def binding_same(a,b):
  assert all(a[k]==b[k] for k in ('image_sha256','question','reference','prompt_ids','generation'))
 def save(p,v):
  budget();t=p.with_suffix('.tmp');torch.save(v,t);t.replace(p)
 try:
  if MODE=='KEYS':
   from dataclasses import replace
   from PIL import Image
   from m3bench_repro.editors.routing import balanced_radius
   keys={};anchors={};kb={}
   def extract(row,image=None,question=None):
    budget();record=r.s.record(template,row);b=runtime.build_question_batch(record,image_path=image,question=question)
    a=runtime.extract_layer_input_features(b,module_path=runtime.target_lock['balancedit']['targets'][0]);mask=torch.ones(a.shape[0],dtype=torch.bool,device=a.device) if b.attention_mask is None else b.attention_mask[0].bool();im=mask.clone();im[:b.image_token_start]=False;im[b.image_token_end:]=False;txt=mask&~im
    assert im.any() and txt.any();v={k:a[m].mean(0).detach().cpu() for k,m in [('joint',mask),('image',im),('text',txt)]};assert all(torch.isfinite(z).all() for z in v.values());return v,b
   for row in config['rows'][INDEX::2]:
    v,b=extract(row);base=read(config['Base_outputs'][row['id']])['binding'];assert b.image_sha256==base['image_sha256'] and b.raw_input_ids[0].tolist()==base['prompt_ids'] and b.preprocessing_seed==base['preprocessing_seed'];keys[row['id']]=v;kb[row['id']]=base;counts['query_keys']+=1
    save(OUT/'KEYS_ACTIVE.pt',dict(keys=keys,bindings=kb,anchors=anchors,counts=copy.deepcopy(counts)))
   # Native keys assigned to the other shard are read after its query pass, no duplicate model call.
   write(OUT/'QUERIES_READY.json',dict(status='PASS',counts=counts,time=time.time()))
   other=ROOT/'features'/str(1-INDEX)
   while not (other/'QUERIES_READY.json').exists():
    budget();assert not (other/'FAILURE.json').exists(),'Other key shard failed';time.sleep(2)
   allkeys=dict(torch.load(other/'KEYS_ACTIVE.pt',map_location='cpu',weights_only=True)['keys']);allkeys.update(keys)
   for task in config['tasks'][INDEX::2]:
    row=task['source'];record=r.s.record(template,row);positive,_=extract(row,question='Please answer this question: '+row['question'])
    black=runtime.make_black_image(record,OUT/'black_images')
    negative,_=extract(row,image=black);native=allkeys[row['id']]['joint'];radius=balanced_radius(native.to(runtime.device),positive['joint'].to(runtime.device),negative['joint'].to(runtime.device),alpha=.2,distance='euclidean');anchors[task['key']]=dict(native=native,positive=positive['joint'],negative=negative['joint'],radius=float(radius));counts['anchor_keys']+=2
   save(OUT/'KEYS_FINAL.pt',dict(keys=keys,bindings=kb,anchors=anchors,counts=copy.deepcopy(counts)))
   write(OUT/'ANCHORS_READY.json',dict(status='PASS',counts=counts))
   if INDEX==0:
    from dataclasses import asdict
    from m3bench_repro.editors.routing import MemoryRouter
    while not (other/'ANCHORS_READY.json').exists():
     budget();assert not (other/'FAILURE.json').exists();time.sleep(2)
    otherdata=torch.load(other/'KEYS_FINAL.pt',map_location='cpu',weights_only=True);allanchors=dict(otherdata['anchors']);allanchors.update(anchors);router=MemoryRouter('euclidean')
    for task in config['tasks']:
     a=allanchors[task['key']];router.add(task['key'],a['native'].to(runtime.device),a['radius'],())
    decisions={row['id']:asdict(router.route(allkeys[row['id']]['joint'].to(runtime.device))) for row in config['rows']}
    write(ROOT/'R0_DECISIONS.json',dict(decisions=decisions,device='Original FP32 GPU MemoryRouter',extra_language_forward=0))
   write(OUT/'COMPLETE.json',dict(status='KEY_CONSUMERS_COMPLETE',counts=counts,seconds=time.time()-START))
  else:
   task=config['tasks'][INDEX];source=task['source'];seed_everything(template['seed'])
   runtime.model.half();items=r.s.prepare(runtime,bindings,template,[source],True);r.old.fp32(runtime)
   row,sourcecpu,raw,_,binding=items[0];binding_same(binding,read(config['Base_outputs'][row['id']])['binding']);batch=r.old.batch_on(runtime,sourcecpu)
   # Base CE is measured before a zero-residual expert; first training forward independently checks parity.
   with torch.no_grad():base_ce=float(runtime.compute_loss(batch))
   initial=r.p.tr.TT4(template['seed'],8,8).state_dict();base=r.mm.canonicalize(initial)[0];expert=r.p.expert(initial,template['seed'],runtime.device);expert.load_state_dict(base);ks=list(base);params=dict(expert.named_parameters());hook=r.q.scoped.replay.MedTraceLayerHook(runtime.get_module(r.c.LAYER),expert);curve=[];scalar=dict(step=0,second_moments=[0.,0.]);hook.attach()
   try:
    with PrefixCache(runtime,[batch]):
     for step in range(1,161):
      budget();before=r.old.snapshot(expert);hook.set_teacher_routing(batch.labels);loss=runtime.compute_loss(batch);assert torch.isfinite(loss);val=float(loss.detach())
      if step==1:
       assert abs(val-base_ce)<=1e-6;write(OUT/'ENGINEERING_ADMISSION.json',dict(status='PASS',zero_expert_Base_CE_difference=abs(val-base_ce),binding=True,recipe='SOURCE_160_original_RMS_clip_canonicalize_EOS'))
      gs=torch.autograd.grad(loss,tuple(params[k] for k in ks),allow_unused=True);counts['backwards']+=1;g=torch.cat([(z if z is not None else torch.zeros_like(params[k])).detach().cpu().double().reshape(-1) for k,z in zip(ks,gs)])
      rawstate,updated,direction=proposal(r,before,g,scalar,ks);state=construct(r,before,rawstate,direction,direction,1.,ks);assert all(torch.isfinite(z).all() for z in state.values());expert.load_state_dict(state);scalar=updated;counts['updates']+=1;curve.append(dict(step=step,source_loss_before=val,actual_norm=r.fm.map_norm(before,state)))
      save(OUT/'active.pt',dict(state=r.old.snapshot(expert),scalar=scalar,curve=curve,counts=copy.deepcopy(counts)))
      if step%10==0:write(OUT/'PROGRESS.json',dict(step=step,task=task['key'],counts=counts,time=time.time()))
     with torch.no_grad():hook.set_teacher_routing(batch.labels);final_ce=float(runtime.compute_loss(batch))
   finally:hook.detach()
   final=r.old.snapshot(expert);save(OUT/'final.pt',dict(state=final));assert same(final,torch.load(OUT/'final.pt',map_location=runtime.device,weights_only=True)['state'])
   write(OUT/'TRAIN_COMPLETE.json',dict(status='COMPLETE',steps=160,source=source['id'],source_CE_initial=base_ce,source_CE_final=final_ce,curve=curve,save_load_exact=True,counts=copy.deepcopy(counts)))
   runtime.model.half();evaluation=r.s.prepare(runtime,bindings,template,config['rows'],False);r.old.fp32(runtime);hook.attach()
   try:
    for row,_,raw,expanded,binding in evaluation:
     budget();qid=row['id'];binding_same(binding,read(config['Base_outputs'][qid])['binding']);dest=OUT/'outputs'/ (qid+'.json');assert not dest.exists()
     write(OUT/'attempts'/(qid+'.json'),dict(attempt=1,time=time.time()));assert counts['generations']<28;counts['generations']+=1;answer,trace=r.s.fit.short.generate(runtime,hook,raw,expanded);tokens=list(answer.raw_token_ids)
     write(dest,dict(expert=task['key'],query=qid,binding=dict(binding,preprocessing_seed=raw['preprocessing_seed']),raw_answer=answer.decoded_text,raw_token_ids=tokens,trace=trace,EOS=bool(tokens) and tokens[-1]==runtime.adapter.tokenizer.eos_token_id,at_cap=len(tokens)>=1024))
     write(OUT/'PROGRESS.json',dict(task=task['key'],generated=counts['generations'],counts=counts,time=time.time()))
   finally:hook.detach()
   assert counts['generations']==len(config['rows']);write(OUT/'COMPLETE.json',dict(status='SOURCE_AND_ALL_MATRIX_MODEL_CONSUMERS_COMPLETE',task=task['key'],save_load_exact=True,counts=counts,seconds=time.time()-START))
 finally:h.remove();vh.remove()
if __name__=='__main__':
 code=0
 try:main()
 except BaseException:code=1;write(OUT/'FAILURE.json',dict(error=traceback.format_exc(),counts=counts,automatic_retry=False));raise
 finally:write(OUT/'EXIT.json',dict(code=code,counts=counts,seconds=time.time()-START))
