"""One-shot P/R target generation from the immutable original model, no editing."""
import json,os,sys,time,subprocess,traceback,signal
from pathlib import Path
from types import SimpleNamespace
import torch
from .prepare import write
from .editor import paired_batches
from .finite_null import native_losses
from .revision3_ops import AUTH,prompt

def main(c):
 if c['authorization']!=AUTH:raise PermissionError('Frozen preparation only')
 root=Path(c['output']);started=time.time();calls=0;gcalls=0;generating=False;restoring=False;rt=None;layer=None;base=None;hook=None;pair=None;before=None
 state=dict(status='LOADING',checks=[],sources=[],outputs=[],editing_candidates=0,backward_calls=0)
 def emit():
  write(root/'public/RESULT.json',state)
  if sum(p.stat().st_size for p in root.rglob('*') if p.is_file())>AUTH['disk_max']:raise OSError('Disk ceiling')
 def check(ok,field):
  state['checks'].append(dict(field=field,passed=bool(ok)));emit()
  if not ok:raise ValueError(field)
 def count(model,args):
  nonlocal calls,gcalls
  if calls>=AUTH['forward_max'] or (not restoring and (calls>=AUTH['work_forward_max'] or time.time()>=c['deadline_epoch'])):raise TimeoutError('Time/forward budget')
  if generating:
   if gcalls>=128:raise TimeoutError('Per-request budget; no partial target')
   gcalls+=1
  calls+=1
 def interrupt(signum,frame):raise TimeoutError('Hard phase deadline')
 signal.signal(signal.SIGTERM,interrupt)
 try:
  with (root/'STARTED.json').open('x') as f:json.dump(dict(pid=os.getpid(),epoch=started),f)
  torch.use_deterministic_algorithms(True);check(os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8','workspace');sys.path.insert(0,c['source_root'])
  check(subprocess.check_output(['git','rev-parse','HEAD'],cwd=c['source_root'],text=True).strip()==c['source_commit'],'runtime_source')
  from scripts.medtrace.run_realmodel_core import load_real_runtime
  from m3bench_repro.editors.llava_runtime import EditorRecord
  prior=Path(c['prior']);manifest=json.loads((prior/'private/MANIFEST.json').read_text());bindings=json.loads((prior/'private/BINDINGS.json').read_text());sources=json.loads((root/'private/SOURCES.json').read_text())['sources']
  rt=load_real_runtime(SimpleNamespace(cpu_gate=Path(c['cpu_gate'])));hook=rt.model.register_forward_pre_hook(count);layer=rt.get_module('model.layers.0.mlp.down_proj');base=layer.weight.detach().clone();flags=[p.requires_grad for p in rt.model.parameters()];modes=[m.training for m in rt.model.modules()];genconfig=rt.model.generation_config.to_dict();check(base.dtype==torch.float16,'native_FP16')
  old=json.loads((prior/'public/SAMPLES.json').read_text())
  for index,s in enumerate(sources):
   t=manifest['tasks'][index];check(t['alias']==s['alias']==AUTH['samples'][index],'fixed_case_order')
   row=manifest['queries'][next(e['query'] for e in t['panel'] if e['task']=='T0')];record=EditorRecord(**dict(t['record'],image_path=Path(t['record']['image_path'])))
   check(record.target==s['reference']==row['reference'] and record.question==s['question']==row['question'],'source_reference_binding')
   pair=paired_batches(rt,record);before=native_losses(rt,pair);check(all(abs(x-y)<=1e-5 for x,y in zip(before,old[index]['baseline']['native'])),'native_base_loss')
   raw=rt.adapter.prepare_inputs(record.image_path,record.question,None);bound=bindings[row['opaque_Base_id']]
   check(raw['input_ids'].tolist()==[bound['prompt_ids']] and raw['attention_mask'].tolist()==[bound['attention_mask']] and raw['image_sha256']==bound['image_sha256'],'cached_draft_prompt_image_binding');check(rt.generation_config==bound['generation'],'original_generation_config')
   state['sources'].append(dict(sample=s['alias'],draft_reused=True,base_native_nll=before));emit()
   for arm in ['P','R']:
    alias=s['alias']+'_'+arm;state['status']='GENERATING_'+alias;emit();question=prompt(s['question'],s['reference'],arm,s['draft'] if arm=='R' else None);prepared=rt.adapter.prepare_inputs(record.image_path,question,None);check(prepared['image_sha256']==bound['image_sha256'],'same_source_image')
    private=dict(source_alias=s['alias'],arm=arm,source_question=s['question'],source_reference=s['reference'],prepared_prompt=question,prompt_ids=prepared['input_ids'].tolist(),image_sha256=prepared['image_sha256'],generation_config=rt.generation_config)
    write(root/('private/'+alias+'_INPUT.json'),private);write(root/('private/'+alias+'_STARTED.json'),dict(epoch=time.time()));gcalls=0;generating=True
    try:
     with torch.inference_mode():value=rt.adapter.generate_prepared_with_result(prepared,rt.generation_config)
     write(root/('private/'+alias+'_GENERATION.json'),dict(status='COMPLETE',raw_answer=value.decoded_text,raw_token_ids=list(value.raw_token_ids)))
     check(bool(value.decoded_text.strip()) and len(value.raw_token_ids)<rt.generation_config['max_new_tokens'],'nonempty_natural_completion')
     state['outputs'].append(dict(alias=alias,sample=s['alias'],arm=arm,status='COMPLETE',generation_forwards=gcalls,raw_token_count=len(value.raw_token_ids),source_qualification=None));emit()
    except BaseException as ex:
     state['outputs'].append(dict(alias=alias,sample=s['alias'],arm=arm,status='FAILED_NO_RETRY',generation_forwards=gcalls,error_type=type(ex).__name__));emit();raise
    finally:generating=False
    check(torch.equal(layer.weight,base),'read_only_selected_weight')
   reread=native_losses(rt,pair);check(all(abs(x-y)<=1e-5 for x,y in zip(before,reread)),'source_pair_native_return')
  state['status']='COMPLETE';emit()
 except BaseException as ex:
  state.update(status='FAILED_NO_RETRY',error_type=type(ex).__name__);write(root/'private/FAILURE.json',dict(traceback=traceback.format_exc()));emit()
 finally:
  restoring=True
  if layer is not None:
   try:
    with torch.no_grad():layer.weight.copy_(base)
    reread=native_losses(rt,pair)
    state['restoration']=dict(weight=torch.equal(layer.weight,base),native_loss=all(abs(x-y)<=1e-5 for x,y in zip(before,reread)),dtype=layer.weight.dtype==base.dtype,flags=flags==[p.requires_grad for p in rt.model.parameters()] and modes==[m.training for m in rt.model.modules()],generation_config=genconfig==rt.model.generation_config.to_dict(),hooks=not layer._forward_hooks and not layer._forward_pre_hooks)
    if not all(state['restoration'].values()):state['status']='RESTORE_FAILED'
   except BaseException:state['status']='RESTORE_FAILED'
  if hook is not None:hook.remove()
  state['model_forward_calls']=calls;emit();write(root/'ENDED.json',dict(status=state['status'],epoch=time.time(),resident_seconds=time.time()-started,model_forward_calls=calls,peak_allocated_bytes=torch.cuda.max_memory_allocated() if rt else None,checkpoint=0))
