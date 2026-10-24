import os,sys,json,time,traceback,subprocess
from pathlib import Path
from types import SimpleNamespace
from contextlib import nullcontext
P=Path(os.environ['PARENT_RUN']);OUT=Path(os.environ['OUTPUT_RUN'])
def read(p):return json.loads(p.read_text())
def write(n,x):(OUT/n).write_text(json.dumps(x,indent=2,default=str)+'\n')
assert not (OUT/'ATTEMPT.json').exists(),'No retry'
write('ATTEMPT.json',dict(start=time.time(),pid=os.getpid(),maximum_generations=2))
sys.path[:0]=[str(P/'private/source'),str(P/'private/official_llava')]
import torch
def generate(runtime,hook,raw,expanded):
    model=runtime.llava_model();original=model.prepare_inputs_labels_for_multimodal;used=[0]
    def prepared(*args,**kwargs):
        if args[0] is not None and torch.equal(args[0],raw['input_ids']):
            assert args[3] is None and args[4] is None;used[0]+=1;assert used[0]==1
            return tuple(x.to(device=runtime.device,dtype=torch.float32 if x.is_floating_point() else x.dtype) if isinstance(x,torch.Tensor) else x for x in expanded)
        return original(*args,**kwargs)
    model.prepare_inputs_labels_for_multimodal=prepared
    try:
        with torch.inference_mode(),(hook.generation_request() if hook else nullcontext()):answer=runtime.adapter.generate_prepared_with_result(raw,runtime.generation_config)
    finally:model.prepare_inputs_labels_for_multimodal=original
    assert used[0]==1 and answer.raw_token_ids
    trace=list(hook.last_generation_trace) if hook else []
    assert hook is None or trace and any(x['active_residual_norm']>0 for x in trace)
    return answer,trace
def fp32(runtime):
    runtime.model.float()
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert all(v.dtype==torch.float32 for v in runtime.model.parameters() if v.is_floating_point())

try:
 torch.use_deterministic_algorithms(True)
 from scripts.medtrace.run_realmodel_core import load_real_runtime
 runtime=load_real_runtime(SimpleNamespace(cpu_gate=P/'private/cpu_gate'));runtime.model.eval()
 assert not any(p.requires_grad for p in runtime.model.parameters())
 assert all(p.dtype==torch.float16 for p in runtime.model.parameters() if p.is_floating_point())
 assert not runtime.get_module('model.layers.21.mlp.down_proj')._forward_hooks
 bindings=read(P/'private/EVAL_BINDINGS.json');assert runtime.generation_config==next(iter(bindings.values()))['generation']
 write('RUNTIME.json',dict(generation=runtime.generation_config,inherited_runtime=next(iter(bindings.values()))['runtime'],model_config=runtime.model.config.to_dict(),image_processor=runtime.adapter.image_processor.to_dict(),torch=torch.__version__,actual_precision='MATH_FP32_FROZEN_FP16_PREFILL',training=False,expert=False))
 commands=subprocess.check_output(['ps','-ww','-p',str(os.getpid()),'-o','pid=,args='],text=True)
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,process_name,used_memory','--format=csv,noheader'],text=True)
 write('PROCESS.json',dict(commands=commands,gpu_processes=apps))
 rows=read(OUT/'inputs.json');assert len(rows)==2
 prepared=[]
 for row in rows:
  path=Path(os.environ['DATA_ROOT'])/row['image_path'].split('/DataP/',1)[1]
  raw=runtime.adapter.prepare_inputs(path,row['question'],None)
  assert raw['image_sha256']==row['image_sha256']
  with torch.no_grad():expanded=runtime.llava_model().prepare_inputs_labels_for_multimodal(raw['input_ids'],None,raw['attention_mask'],None,None,raw['images'],image_sizes=None)
  assert expanded[4].dtype==torch.float16
  expanded=tuple(x.detach().cpu() if isinstance(x,torch.Tensor) else x for x in expanded)
  key=row['image_id'].split('.')[0]
  def cpu(x):
   if isinstance(x,torch.Tensor):return x.detach().cpu()
   if isinstance(x,list):return [cpu(v) for v in x]
   return x
  torch.save(dict(raw={k:cpu(v) for k,v in raw.items()},expanded=expanded),OUT/(key+'_inputs.pt'))
  write(key+'_binding.json',dict(source=row,resolved_image=str(path),prompt=raw['prompt'],prompt_ids=raw['input_ids'][0].tolist(),attention_mask=raw['attention_mask'][0].tolist(),image_sha256=raw['image_sha256'],preprocessing_seed=raw['preprocessing_seed'],generation=runtime.generation_config,padding_side=runtime.adapter.tokenizer.padding_side,eos_token_id=runtime.adapter.tokenizer.eos_token_id))
  prepared.append((key,raw,expanded))
 fp32(runtime)
 for key,raw,expanded in prepared:
  write(key+'_attempt.json',dict(start=time.time(),attempt=1))
  answer,trace=generate(runtime,None,raw,expanded)
  write(key+'_result.json',dict(raw_answer=answer.decoded_text,raw_token_ids=list(answer.raw_token_ids),EOS=answer.raw_token_ids[-1]==runtime.adapter.tokenizer.eos_token_id,at_cap=len(answer.raw_token_ids)>=1024,generation_trace=trace,attempts=1))
  print(key,repr(answer.decoded_text),flush=True)
 write('COMPLETE.json',dict(status='COMPLETE',generations=2,training_steps=0,finish=time.time()))
except BaseException:
 write('FAILURE.json',dict(traceback=traceback.format_exc(),finish=time.time()));raise
