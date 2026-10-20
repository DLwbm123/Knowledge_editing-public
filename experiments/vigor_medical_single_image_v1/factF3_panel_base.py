"""First execution of locality Base generation after CPU-only exact token binding."""
import os,json,time,sys,subprocess,traceback
from pathlib import Path
from types import SimpleNamespace
import torch
from .prepare import write
from .editor import paired_batches
from .finite_null import native_losses
from .factF3_ops import bind_groups,read

def main(c):
 root=Path(c['output']);master=Path(c['master']);started=time.time();calls=0;gen_calls=0;generating=False;rt=None;hook=None;layer=None;base=None;pair=None;before=None;state=dict(status='LOADING',checks=[],facts=[],outputs=[])
 def emit():
  write(root/'public/RESULT.json',state)
  if sum(p.stat().st_size for p in master.rglob('*') if p.is_file())>268435456:raise RuntimeError('Experiment disk bound')
 def check(ok,field):
  state['checks'].append(dict(field=field,passed=bool(ok)));emit()
  if not ok:raise ValueError(field)
 def count(m,args):
  nonlocal calls,gen_calls
  if calls>=3496 or time.time()>c['deadline_epoch']:raise TimeoutError('Preparation forward/time ceiling')
  if generating:
   if gen_calls>=128:raise RuntimeError('Generation128 cap')
   gen_calls+=1
  calls+=1
 try:
  with (root/'STARTED.json').open('x') as f:json.dump(dict(pid=os.getpid(),epoch=started),f)
  torch.use_deterministic_algorithms(True);check(os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8','determinism');sys.path.insert(0,c['source_root'])
  check(subprocess.check_output(['git','rev-parse','HEAD'],cwd=c['source_root'],text=True).strip()==c['source_commit'],'runtime_source')
  from scripts.medtrace.run_realmodel_core import load_real_runtime
  from m3bench_repro.editors.llava_runtime import EditorRecord
  rt=load_real_runtime(SimpleNamespace(cpu_gate=Path(c['cpu_gate'])));hook=rt.model.register_forward_pre_hook(count);layer=rt.get_module('model.layers.0.mlp.down_proj');base=layer.weight.detach().clone();flags=[p.requires_grad for p in rt.model.parameters()];modes=[m.training for m in rt.model.modules()];gc=rt.model.generation_config.to_dict();check(base.dtype==torch.float16,'native_half')
  pre=json.loads((master/'public/FACT_TOKEN_BINDINGS.json').read_text());check(pre['status']=='PASS' and pre['model_forward']==0,'CPU_fact_binding_frozen');state['facts']=pre['facts'];state['old_source_forward_not_replayed']=True;emit()
  candidates=json.loads((master/'private/LOCALITY_CANDIDATES.json').read_text())['rows'];state['status']='BASE_GENERATION';emit()
  for item in candidates:
   row=item['query'];alias=item['alias'];raw=rt.adapter.prepare_inputs(Path(row['image_path']),row['question'],None);check(raw['image_sha256']==row['image_sha256'],'candidate_image_binding')
   write(root/('private/'+alias+'_STARTED.json'),dict(epoch=time.time()));gen_calls=0;generating=True
   try:
    with torch.inference_mode():value=rt.adapter.generate_prepared_with_result(raw,rt.generation_config)
    private=dict(status='COMPLETE',raw_answer=value.decoded_text,raw_token_ids=list(value.raw_token_ids));write(root/('private/'+alias+'_GENERATION.json'),private);entry=dict(alias=alias,status='COMPLETE',raw_tokens=len(value.raw_token_ids),generation_forwards=gen_calls)
   except BaseException as ex:
    write(root/('private/'+alias+'_GENERATION.json'),dict(status='FAILED',error_type=type(ex).__name__));state['outputs'].append(dict(alias=alias,status='FAILED',generation_forwards=gen_calls));emit();raise
   finally:generating=False
   state['outputs'].append(entry);emit();check(torch.equal(layer.weight,base),'read_only_Base_unchanged')
  state['status']='COMPLETE';emit()
 except BaseException as ex:
  state.update(status='FAILED_NO_RETRY',error_type=type(ex).__name__);write(root/'private/FAILURE.json',dict(traceback=traceback.format_exc()));emit()
 finally:
  if layer is not None and base is not None:
   try:
    with torch.no_grad():layer.weight.data=base.clone()
    state['restoration']=dict(weight=torch.equal(layer.weight,base),source_loss_not_replayed=pair is None,dtype=layer.weight.dtype==base.dtype,flags=flags==[p.requires_grad for p in rt.model.parameters()] and modes==[m.training for m in rt.model.modules()],generation_config=gc==rt.model.generation_config.to_dict(),hooks=not layer._forward_hooks and not layer._forward_pre_hooks)
    if not all(state['restoration'].values()):state['status']='RESTORE_FAILED'
   except BaseException:state['status']='RESTORE_FAILED'
  if hook is not None:hook.remove()
  state['model_forward_calls']=calls;emit();write(root/'ENDED.json',dict(status=state['status'],epoch=time.time(),resident_seconds=time.time()-started,model_forward_calls=calls,peak_allocated_bytes=torch.cuda.max_memory_allocated() if rt else None))
