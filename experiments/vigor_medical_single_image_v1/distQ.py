"""One offline qualification run, no editor or judge; partial failures are terminal."""
import json,os,sys,time,signal,subprocess,traceback,math
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from .prepare import write
from .editor import editable_fp32
from .distQ_ops import AUTH,finite,select,probe,log_distribution,KL,fit_directions,risks,summarize

def main(c):
 if c['authorization']!=AUTH:raise PermissionError('Frozen DQ authorization')
 root=Path(c['output']);started=time.time();calls=backs=0;restoring=False;rt=layer=base=hook=layer0=base0=None;batches=[];log0=[];statCPU=[];references=[]
 state=dict(status='LOADING',checks=[],smoke=[],construction=[],measurements=[],backward_calls=0,editor_calls=0,judge_calls=0,complete_generations=0)
 def emit(enforce=True):
  state['model_forward_calls']=calls;state['backward_calls']=backs;write(root/'public/RESULT.json',state)
  if enforce and sum(p.stat().st_size for p in root.rglob('*') if p.is_file())>AUTH['disk_max']:raise OSError('Disk ceiling')
  if enforce and rt is not None and torch.cuda.max_memory_allocated()>40*1024**3:raise MemoryError('Own GPU allocation ceiling')
 def check(ok,name):
  state['checks'].append(dict(field=name,passed=bool(ok)));emit()
  if not ok:raise ValueError(name)
 def count(m,args):
  nonlocal calls
  if calls>=AUTH['forward_max'] or (not restoring and (calls>=AUTH['work_forward_max'] or time.time()>=c['deadline_epoch'])):raise TimeoutError('Forward/time ceiling')
  calls+=1
 def backward():
  nonlocal backs
  if backs>=774:raise TimeoutError('VJP ceiling')
  backs+=1;state['backward_calls']=backs
 def interrupt(s,f):raise TimeoutError('Hard phase deadline')
 signal.signal(signal.SIGTERM,interrupt)
 def selected(batch):
  kw=batch.forward_kwargs();kw['labels']=None
  return select(rt.model(**kw).logits,batch.labels)
 try:
  with (root/'STARTED.json').open('x') as f:json.dump(dict(pid=os.getpid(),epoch=started),f)
  torch.use_deterministic_algorithms(True);check(os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8','deterministic_workspace')
  sys.path.insert(0,c['source_root']);check(subprocess.check_output(['git','rev-parse','HEAD'],cwd=c['source_root'],text=True).strip()==c['source_commit'],'runtime_source')
  from scripts.medtrace.run_realmodel_core import load_real_runtime
  from m3bench_repro.editors.llava_runtime import EditorRecord
  rt=load_real_runtime(SimpleNamespace(cpu_gate=Path(c['cpu_gate'])));hook=rt.model.register_forward_pre_hook(count);layer=rt.get_module(AUTH['layer']);base=layer.weight.detach().clone();flags=[p.requires_grad for p in rt.model.parameters()];modes=[m.training for m in rt.model.modules()];genconfig=rt.model.generation_config.to_dict();layer0=rt.get_module('model.layers.0.mlp.down_proj');base0=layer0.weight.detach().clone()
  check(base.dtype==torch.float16 and tuple(base.shape)==(4096,14336) and layer.bias is None,'native_matrix_binding')
  check(not any(m._forward_hooks or m._forward_pre_hooks for m in rt.model.modules() if m is not rt.model),'no_mutating_hooks')
  check(sum(p.data_ptr()==layer.weight.data_ptr() for _,p in rt.model.named_parameters(remove_duplicate=False))==1,'unshared_matrix')
  calpath=Path(c['calibration']);cm=json.loads((calpath.parent/'manifest.json').read_text());records=json.loads((calpath.parent/'records.json').read_text());cal=torch.load(calpath,map_location='cpu',weights_only=True);factor=cal['factors'][AUTH['layer']]
  check(cal['binding']==c['binding'] and cal['calibration_ids']==cm['calibration_ids'],'old_calibration_binding')
  groups=[x['dataset']+':'+x['relative_image_path'].split('/')[0] for x in records]
  check(len(records)==32 and [x['record_id'] for x in records]==cm['calibration_ids'] and set(groups)==set(cm['calibration_source_groups']) and len(set(groups))==32 and not set(groups)&set(cm['excluded_source_groups']),'frozen_24_8_disjoint_groups')
  write(root/'private/SPLIT.json',dict(build_ids=cm['calibration_ids'][:24],heldout_ids=cm['calibration_ids'][24:],build_groups=groups[:24],heldout_groups=groups[24:]))
  state['split']=dict(build=24,heldout=8,groups=32,patient_independence_verified=False);state['baseline']=[];state['status']='PREPARING_OFFLINE_BASELINES';emit();prep_seconds=0
  for i,row in enumerate(records):
   t=time.time();record=EditorRecord(**dict(row,image_path=Path(row['image_path'])))
   with torch.no_grad():batch=rt.build_edit_batch(record)
   prep_seconds+=time.time()-t;batches.append(batch)
   with torch.no_grad():s=selected(batch);log=log_distribution(s).cpu();loss=float(F.cross_entropy(s,batch.labels[:,1:][batch.labels[:,1:]!=-100]))
   log0.append(log);check(s.shape[1]==32000 and int(batch.labels[batch.labels!=-100][-1])==rt.adapter.tokenizer.eos_token_id,'full_vocab_EOS_complete_mask');state['baseline'].append(dict(alias=('C' if i<24 else 'H')+str(i+1 if i<24 else i-23).zfill(3),answer_positions=s.shape[0],expanded_positions=batch.inputs_embeds.shape[1],native_NLL=loss));del s,log,record
   if i<2:
    with editable_fp32(rt.model,layer,readonly=True):
     try:
      s=selected(batch);bridge=KL(log0[i],s);backward();g=probe(s,layer.weight,AUTH['fit_seed']+i*1000)
      check(float(g.norm())>0 and bridge<=1e-4,'smoke_full_distribution_VJP_dtype_bridge');state['smoke'].append(dict(record=i+1,bridge_KL=bridge,positions=s.shape[0],vocab=s.shape[1],gradient_norm=float(g.norm())));del g,s
     finally:layer.weight.grad=None
    with torch.no_grad():same=KL(log0[i],selected(batch))
    check(torch.equal(layer.weight,base) and abs(same)<=1e-12 and not layer._forward_hooks and not layer._forward_pre_hooks and all(p.grad is None for p in rt.model.parameters()),'smoke_native_restore')
    check(torch.cuda.max_memory_allocated()<=40*1024**3,'smoke_peak_memory')
   emit()
  del records,row,groups
  state['preprocessing']=dict(records=32,seconds=prep_seconds,vision_preparations=32);state['smoke_status']='PASS';state['status']='BUILDING_FROZEN_STATISTICS';emit()
  check(torch.cuda.mem_get_info()[0]>=12*1024**3,'formal_GPU_workspace');host=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024;check(host>=32*1024**3,'formal_host_reserve')
  statCPU=[torch.zeros_like(base,device='cpu',dtype=torch.float32) for _ in range(32)]
  for i in range(24):
   with editable_fp32(rt.model,layer,readonly=True):
    try:
     s=selected(batches[i]);labels=batches[i].labels[:,1:][batches[i].labels[:,1:]!=-100]
     if i<4:
      backward();g=torch.autograd.grad(F.cross_entropy(s,labels),layer.weight,retain_graph=True)[0];references.append(finite(g.detach().float().cpu()));del g
     for k in range(32):
      seed=(AUTH['fit_seed'] if k<16 else AUTH['audit_seed'])+i*1000+k%16;backward();g=probe(s,layer.weight,seed,retain=k<31);statCPU[k].add_(g.cpu());del g
      state['construction'].append(dict(record=i+1,domain='fit' if k<16 else 'audit',probe=k%16+1));emit()
     del s,labels
    finally:layer.weight.grad=None
   check(torch.equal(layer.weight,base) and all(p.grad is None for p in rt.model.parameters()),'offline_statistic_W0_restore')
  check(backs==774,'all_VJP_costs_including_smoke_reference');state['status']='FREEZING_DIRECTIONS';emit()
  directions,aliases,geometry=fit_directions(statCPU[:16],references,base.device);del references
  check(len(directions)==12 and aliases==['R1','R2','R3','R4','G1','G2','G3','G4','B1','B2','B3','B4'],'fixed_direction_order');state['direction_freeze']=dict(aliases=aliases,geometry=geometry,audit_queries_before_freeze=0,heldout_used_for_directions=False);emit()
  statistics=torch.empty((32,base.numel()),device=base.device,dtype=torch.float32)
  for k,g in enumerate(statCPU):statistics[k].copy_(g.reshape(-1))
  del statCPU,g
  generator=torch.Generator(device=base.device).manual_seed(AUTH['mis_seed']);sign=torch.randint(0,2,(base.numel(),),device=base.device,generator=generator,dtype=torch.int8).float().mul_(2).sub_(1)
  a=factor['a'].to(base.device,dtype=torch.float64);b=factor['b'].to(base.device,dtype=torch.float64)
  from scipy.stats import chi2
  point=0;state['status']='MEASURING_FROZEN_POINTS';emit()
  for name,direction in zip(aliases,directions):
   d=direction.to(base.device,dtype=torch.float64)
   for scale in AUTH['scales']:
    for side in [-1,1]:
     point+=1;check(torch.equal(layer.weight,base),'independent_point_W0')
     with torch.no_grad():layer.weight.copy_((base.double()+side*scale*AUTH['work_norm']*d).to(base.dtype))
     delta=finite(layer.weight.detach().double()-base.double());risk=risks(statistics,delta,sign);norm2=float(delta.square().sum());f0=float((delta.square()*b[:,None]*a[None,:]).sum()/2)
     risk.update(point=point,direction=name,scale=scale,sign=side,native_norm=math.sqrt(norm2),norm_squared=norm2,F0=f0,record_KL=[],C_KL=None,H_KL=None)
     risk['ideal_local_KL_pointwise95']=[16*risk['audit']/float(chi2.ppf(.975,16)),16*risk['audit']/float(chi2.ppf(.025,16))];risk['ideal_local_KL_Bonferroni72']=[16*risk['audit']/float(chi2.ppf(1-.05/144,16)),16*risk['audit']/float(chi2.ppf(.05/144,16))];risk['interval_scope']='ideal_F_C_local_only_not_H_or_native_exact'
     state['measurements'].append(risk);emit()
     for i,batch in enumerate(batches):
      with torch.no_grad():value=KL(log0[i],selected(batch))
      risk['record_KL'].append(value);emit()
      if not math.isfinite(value) or value < -1e-12:raise ValueError('Numerical full-vocabulary KL')
     risk['C_KL']=sum(risk['record_KL'][:24])/24;risk['H_KL']=sum(risk['record_KL'][24:])/8
     with torch.no_grad():layer.weight.copy_(base)
     risk['W0_restored']=torch.equal(layer.weight,base);emit();del delta
   del d
  state['qualification']=summarize(state['measurements']);state['status']='COMPLETE';emit()
 except BaseException as e:
  state.update(status='FAILED_NO_RETRY',error_type=type(e).__name__);write(root/'private/FAILURE.json',dict(traceback=traceback.format_exc()));emit(False)
 finally:
  restoring=True
  if layer is not None:
   try:
    with torch.no_grad():layer.weight.data=base.clone();layer.weight.grad=None
    restored=[]
    for i,batch in enumerate(batches[:len(log0)]):
     with torch.no_grad():restored.append(KL(log0[i],selected(batch)))
    state['restoration']=dict(weight=torch.equal(layer.weight,base),layer0_unchanged=torch.equal(layer0.weight,base0),dtype=layer.weight.dtype==base.dtype,full_native_distributions=bool(restored) and all(abs(x)<=1e-12 for x in restored),restored_records=len(restored),flags=flags==[p.requires_grad for p in rt.model.parameters()],modes=modes==[m.training for m in rt.model.modules()],hooks=not layer._forward_hooks and not layer._forward_pre_hooks,gradients=all(p.grad is None for p in rt.model.parameters()),generation_config=genconfig==rt.model.generation_config.to_dict())
    if not all(x for k,x in state['restoration'].items() if k!='restored_records'):state['status']='RESTORE_FAILED'
   except BaseException:
    state['status']='RESTORE_FAILED';write(root/'private/RESTORE_FAILURE.json',dict(traceback=traceback.format_exc()))
  if hook is not None:hook.remove()
  emit(False);write(root/'ENDED.json',dict(status=state['status'],epoch=time.time(),resident_seconds=time.time()-started,model_forward_calls=calls,backward_calls=backs,peak_allocated_bytes=torch.cuda.max_memory_allocated() if rt else None,checkpoint=0,judge_calls=0,editor_calls=0))
