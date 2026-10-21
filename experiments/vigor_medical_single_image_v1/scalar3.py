"""Scalar null repair inside the fixed successful source-action class."""
import os,json,sys,time,signal,traceback,subprocess,math
from pathlib import Path
from types import SimpleNamespace
import torch
from .prepare import write
from .editor import paired_batches
from .finite_null import native_losses,gates
from .factF3 import evaluate_panel
from .ftmq_ops import fit
from .teacherL3_ops import fit_matches,source_matches
from .equiv3_ops import basis,relnorm
from .scalar3_ops import AUTH,repair
from .hpse2_ops import raw_forward

def main(c):
 if c['authorization']!=AUTH:raise PermissionError('Frozen teacher qualification only')
 root=Path(c['output']);started=time.time();rt=layer=base=hook=pair=before=untouched=baseline0=None
 calls=gcalls=0;generating=restoring=fitting=False;fit_first=0
 state=dict(status='LOADING',checks=[],teachers=[],student_edits=0)
 def emit():
  write(root/'public/RESULT.json',state)
  if sum(p.stat().st_size for p in root.rglob('*') if p.is_file())>AUTH['disk_max']:
   raise OSError('Disk ceiling')
 def check(ok,k):
  state['checks'].append(dict(field=k,passed=bool(ok)));emit()
  if not ok:raise ValueError(k)
 def count(model,args):
  nonlocal calls,gcalls
  if calls>=AUTH['forward_max'] or (not restoring and (calls>=AUTH['work_forward_max'] or time.time()>=c['deadline_epoch'])):
   raise TimeoutError('Forward/time ceiling')
  if fitting and calls-fit_first>=26:raise TimeoutError('Teacher reconstruction forward ceiling')
  if generating:
   if gcalls>=AUTH['per_generation_forward_max']:raise TimeoutError('Incomplete generation; no retry')
   gcalls+=1
  calls+=1
 def interrupt(signum,frame):raise TimeoutError('Hard phase deadline')
 signal.signal(signal.SIGTERM,interrupt)
 try:
  with (root/'STARTED.json').open('x') as f:json.dump(dict(pid=os.getpid(),epoch=started),f)
  torch.use_deterministic_algorithms(True);check(os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8','workspace')
  sys.path.insert(0,c['source_root']);check(subprocess.check_output(['git','rev-parse','HEAD'],cwd=c['source_root'],text=True).strip()==c['source_commit'],'runtime_source')
  from scripts.medtrace.run_realmodel_core import load_real_runtime
  from m3bench_repro.editors.llava_runtime import EditorRecord
  prior=Path(c['prior']);manifest=json.loads((prior/'private/MANIFEST.json').read_text());bindings=json.loads((prior/'private/BINDINGS.json').read_text());old=json.loads((prior/'public/SAMPLES.json').read_text())
  rt=load_real_runtime(SimpleNamespace(cpu_gate=Path(c['cpu_gate'])));hook=rt.model.register_forward_pre_hook(count)
  layer=rt.get_module(AUTH['layer']);base=layer.weight.detach().clone()
  flags=[p.requires_grad for p in rt.model.parameters()];modes=[m.training for m in rt.model.modules()];genconfig=rt.model.generation_config.to_dict()
  untouched=rt.get_module('model.layers.0.mlp.down_proj');baseline0=untouched.weight.detach().clone()
  cal=torch.load(c['calibration'],map_location='cpu',weights_only=True);check(cal['binding']==c['binding'],'old_F0_binding');factor=cal['factors'][AUTH['layer']]
  check(cal['selected_layer']=='model.layers.0.mlp.down_proj' and len(cal['factors'])==32 and len(cal['calibration_ids'])==32,'original_train_calibration_immutable')
  check(tuple(factor['a'].shape)==(14336,) and tuple(factor['b'].shape)==(4096,) and factor['positions']==19237,'layer6_factor_identity')
  tq=Path(c['teacher_prior']);tql=Path(c['teacher_locality_prior']);check(json.loads((tq/'ENDED.json').read_text())['status']=='COMPLETE','original_teacher_terminal');old_teachers=json.loads((tq/'public/RESULT.json').read_text())['teachers']
  check(type(layer).__name__=='Linear' and type(rt.get_module('model.layers.6')).__name__=='MistralDecoderLayer','standard_feedforward_structure')
  check(sum(p.data_ptr()==layer.weight.data_ptr() for _,p in rt.model.named_parameters(remove_duplicate=False))==1,'unshared_selected_weight')
  check(not any(m._forward_hooks or m._forward_pre_hooks for m in rt.model.modules() if m is not rt.model),'no_mutating_module_hooks')
  cm=json.loads((Path(c['calibration']).parent/'manifest.json').read_text());panel=json.loads((Path(c['master'])/'private/LOCALITY_PANEL.json').read_text())['rows']
  queries=[manifest['queries'][next(e['query'] for e in t['panel'] if e['task']=='T0')] for t in manifest['tasks']]+[x['query'] for x in panel]
  groups={x['source_group'] for x in queries};ids={x['query_id'] for x in queries}
  check(set(cal['calibration_ids'])==set(cm['calibration_ids']) and not groups.intersection(cm['calibration_source_groups']) and groups.issubset(cm['excluded_source_groups']) and not ids.intersection(cm['calibration_ids']) and ids.issubset(cm['excluded_ids']),'training_source_evaluation_isolation')
  check(torch.isfinite(factor['a']).all() and bool((factor['a']>0).all()),'fixed_positive_input_metric')
  check(torch.cuda.mem_get_info()[0]>=12*1024**3,'twelve_GiB_workspace_reserve')
  check(base.dtype==torch.float16,'native_FP16');check(layer.bias is None,'bias_free_teacher_matrix')
  state['teacher_parameters']=base.numel();state['teacher_native_bytes']=base.numel()*base.element_size();emit()
  for index,sample in enumerate(AUTH['samples']):
   check(torch.equal(layer.weight,base),'independent_W0_teacher')
   task=manifest['tasks'][index];check(task['alias']==sample,'fixed_case_order')
   row=manifest['queries'][next(e['query'] for e in task['panel'] if e['task']=='T0')]
   record=EditorRecord(**dict(task['record'],image_path=Path(task['record']['image_path'])))
   check(record.target==row['reference'] and record.question==row['question'],'source_binding')
   pair=paired_batches(rt,record);before=native_losses(rt,pair)
   check(all(math.isfinite(x) and abs(x-y)<=1e-5 for x,y in zip(before,old[index]['baseline']['native'])),'original_base_native_loss')
   raw=rt.adapter.prepare_inputs(record.image_path,record.question,None);bound=bindings[row['opaque_Base_id']]
   check(raw['input_ids'].tolist()==[bound['prompt_ids']] and raw['attention_mask'].tolist()==[bound['attention_mask']] and raw['image_sha256']==bound['image_sha256'],'original_question_image_no_reference')
   check(rt.generation_config==bound['generation'],'original_generation_config')
   entry=dict(sample=sample,status='FITTING',base_native_nll=before,steps=[]);state['teachers'].append(entry);state['status']='FITTING_'+sample;emit();first=calls;fit_first=calls
   def record_step(x):entry['steps'].append(x);emit()
   fitting=True
   try:fit(rt.model,layer,pair[0],record_step)
   finally:fitting=False
   entry['fit_forwards']=calls-first;entry['backward_calls']=sum(s['optimizer_step'] for s in entry['steps'])
   check(layer.weight.dtype==base.dtype,'native_teacher_dtype');check(all(p.grad is None for p in rt.model.parameters()),'fit_grad_cleanup')
   teacher_losses=native_losses(rt,pair);check(all(math.isfinite(x) for x in teacher_losses),'native_teacher_finite')
   entry['teacher_native_nll']=teacher_losses
   old_teacher=next(x for x in old_teachers if x['sample']==sample)
   check(fit_matches(entry,old_teacher),'fixed_teacher_fit_binding')
   check(sum(x.get('fit_forwards',0) for x in state['teachers'])<=78,'total_fit_ceiling')
   endpoint=gates(rt,pair,factor,base,before,before[1],module_name=AUTH['layer']);entry['endpoint']=endpoint;entry['allowed_endpoint_submission']=endpoint['accepted']
   check(all(abs(x-y)<=1e-5 for x,y in zip(endpoint['native_nll'],teacher_losses)),'endpoint_native_repeat')
   entry['teacher_delta_norm']=float((layer.weight.detach().float()-base.float()).double().norm());entry['native_changed']=not torch.equal(layer.weight,base);emit()
   original_tql=next(x for x in json.loads((tql/'public/RESULT.json').read_text())['teachers'] if x['sample']==sample)
   check(all(abs(x-y)<=1e-5 for x,y in zip(endpoint['native_nll'],original_tql['endpoint']['native_nll'])) and abs(endpoint['combined_native_fisher_cost']/original_tql['endpoint']['combined_native_fisher_cost']-1)<=1e-5,'T_endpoint_binding')
   keys=[]
   kh=layer.register_forward_pre_hook(lambda m,args:keys.append(args[0].detach().clone()))
   state['status']='GENERATING_'+sample;entry['status']='GENERATING';gcalls=0;generating=True;captured=[]
   def capture(model,args,out):captured.append(int(out.logits[0,-1].argmax()))
   gh=rt.model.register_forward_hook(capture)
   write(root/('private/'+sample+'_GENERATION_STARTED.json'),dict(epoch=time.time()))
   try:
    with torch.inference_mode():value=rt.adapter.generate_prepared_with_result(raw,rt.generation_config)
    generated=dict(status='COMPLETE',raw_answer=value.decoded_text,raw_token_ids=list(value.raw_token_ids));write(root/('private/'+sample+'_GENERATION.json'),generated)
    check(source_matches(generated,json.loads((tq/'private'/(sample+'_GENERATION.json')).read_text())),'complete_original_source_output_binding')
    entry['source_output_exact_binding']=True
    check(bool(value.decoded_text.strip()) and len(value.raw_token_ids)<rt.generation_config['max_new_tokens'],'natural_complete_source_answer')
    entry.update(status='COMPLETE',generation_forwards=gcalls,raw_token_count=len(value.raw_token_ids),qualification=None)
   except BaseException:
    write(root/('private/'+sample+'_INCOMPLETE.json'),dict(partial_greedy_ids=captured));entry.update(status='FAILED_GENERATION_NO_RETRY',generation_forwards=gcalls);raise
   finally:gh.remove();kh.remove();generating=False
   check(len(keys)==entry['generation_forwards'] and keys[0].shape[0]==1 and keys[0].shape[1]>raw['input_ids'].shape[1] and all(x.shape[1]==1 for x in keys[1:]),'complete_visual_prefill_and_cached_decode_keys')
   entry['key_coverage']=dict(prefill_positions=keys[0].shape[1],decode_calls=len(keys)-1,all_decode_positions=sum(x.shape[1] for x in keys[1:]),all_positions=sum(x.shape[1] for x in keys))
   K=torch.cat([x.reshape(-1,x.shape[-1]) for x in keys]).T.double();keys.clear()
   teacher=layer.weight.detach().clone();delta=teacher.double()-base.double();q,meta=basis(K);entry['basis']=meta;entry['candidates']=[];emit()
   tokens=list(value.raw_token_ids)
   def fixed_prefix():
    captured=[]
    h=layer.register_forward_hook(lambda m,args,out:captured.append((args[0].detach().clone(),out.detach().clone())))
    try:
     with torch.no_grad():logits=raw_forward(rt,raw,tokens[:-1])[0,-len(tokens):].detach()
    finally:h.remove()
    check(len(captured)==1,'fixed_prefix_single_invocation')
    return logits,*captured[0]
   teacher_logits,reference_inputs,teacher_outputs=fixed_prefix()
   for name in ['S']:
    entry['repair_steps']=[]
    def repair_emit(row):entry['repair_steps'].append(row);emit()
    state['status']='REPAIRING_'+sample;emit()
    result=repair(rt.model,layer,base,delta,K,q,pair[1],before[1],lambda:gates(rt,pair,factor,base,before,before[1],module_name=AUTH['layer']),repair_emit)
    entry['repair_status']=result['status'];entry['repair_backwards']=result['repair_backwards'];entry['accepted_updates']=result['accepted_updates'];entry['initial_I_endpoint']=result['initial_endpoint']
    native_error=result['final_native_action_error']
    candidate_logits,actual_inputs,actual_outputs=fixed_prefix()
    check(torch.equal(reference_inputs,actual_inputs),'fixed_prefix_upstream_inputs_independent')
    logits_diff=candidate_logits-teacher_logits;output_diff=actual_outputs.float()-teacher_outputs.float()
    arm=dict(name=name,source_action_error=result['final_action_error'],native_action_relative_error=native_error,actual_layer_output_relative_error=relnorm(output_diff,teacher_outputs.float()),actual_layer_output_max_abs=float(output_diff.abs().max()),fixed_prefix_logits_max_abs=float(logits_diff.abs().max()),fixed_prefix_logits_relative_error=relnorm(logits_diff,teacher_logits))
    arm['endpoint']=gates(rt,pair,factor,base,before,before[1],module_name=AUTH['layer']);arm['allowed_endpoint_submission']=arm['endpoint']['accepted'];entry['candidates'].append(arm);emit()
    del candidate_logits,actual_inputs,actual_outputs,logits_diff,output_diff
    generating=True;gcalls=0
    try:
     with torch.inference_mode():g=rt.adapter.generate_prepared_with_result(raw,rt.generation_config)
     private=dict(status='COMPLETE',raw_answer=g.decoded_text,raw_token_ids=list(g.raw_token_ids));write(root/('private/'+sample+'_'+name+'_GENERATION.json'),private)
     check(bool(g.decoded_text.strip()) and len(g.raw_token_ids)<rt.generation_config['max_new_tokens'],'complete_candidate_source')
     arm.update(generation_status='COMPLETE',generation_forwards=gcalls,source_exact_teacher=source_matches(private,generated))
    finally:generating=False
    def reset_generation():
     nonlocal gcalls
     gcalls=0
    generating=True
    try:evaluate_panel(root,Path(c['master']),rt,record,sample,name,state,emit,reset_generation,lambda:gcalls)
    finally:generating=False
    with torch.no_grad():layer.weight.copy_(base)
    check(torch.equal(layer.weight,base),'independent_candidate_W0_restored');emit()
   del K,q,delta,teacher,teacher_logits,reference_inputs,teacher_outputs
   with torch.no_grad():layer.weight.copy_(base)
   restored=native_losses(rt,pair);check(torch.equal(layer.weight,base) and all(abs(x-y)<=1e-5 for x,y in zip(before,restored)),'teacher_retired_base_restored');entry['base_restored']=True;emit()
  state['status']='COMPLETE';emit()
 except BaseException as ex:
  state.update(status='FAILED_NO_RETRY',error_type=type(ex).__name__);write(root/'private/FAILURE.json',dict(traceback=traceback.format_exc()));emit()
 finally:
  restoring=True;generating=False
  if layer is not None:
   try:
    with torch.no_grad():layer.weight.copy_(base)
    reread=native_losses(rt,pair) if pair is not None else None
    state['restoration']=dict(unselected_layer0_unchanged=untouched is not None and torch.equal(untouched.weight,baseline0),weight=torch.equal(layer.weight,base),dtype=layer.weight.dtype==base.dtype,
      native_loss=reread is not None and all(abs(x-y)<=1e-5 for x,y in zip(before,reread)),
      flags=flags==[p.requires_grad for p in rt.model.parameters()],modes=modes==[m.training for m in rt.model.modules()],
      hooks=not layer._forward_hooks and not layer._forward_pre_hooks,
      gradients=all(p.grad is None for p in rt.model.parameters()),generation_config=genconfig==rt.model.generation_config.to_dict())
    if not all(state['restoration'].values()):state['status']='RESTORE_FAILED'
   except BaseException:state['status']='RESTORE_FAILED'
  if hook is not None:hook.remove()
  state['model_forward_calls']=calls;emit()
  write(root/'ENDED.json',dict(status=state['status'],epoch=time.time(),resident_seconds=time.time()-started,
    model_forward_calls=calls,peak_allocated_bytes=torch.cuda.max_memory_allocated() if rt else None,checkpoint=0))
