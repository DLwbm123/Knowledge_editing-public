"""Bounded numeric forensic worker, separate immutable lineage."""
import os,sys,json,time,traceback,subprocess
from pathlib import Path
from types import SimpleNamespace
P=Path(os.environ['PARENT_RUN']);OUT=Path(os.environ['OUTPUT_RUN'])
sys.path[:0]=[str(OUT/'tools'),str(P/'private/tools'),str(P/'private/source'),str(P/'private/official_llava')]
import torch
import module_run as r
from scripts.medtrace.run_realmodel_core import load_real_runtime

def write(name,v):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
counts=dict(updates=0,backwards=0,operator_backwards=0,generations=0,forwards=0,candidate_checks=0,logical_transitions=0,reused_transitions=0,vision_forwards=0)
def budget():
    assert time.time()<float(os.environ['DEADLINE_EPOCH']) and not (OUT.parent/'STOP').exists()
r.c.budget=budget
start=time.time();code=0
try:
    with (OUT/'ATTEMPT').open('x'):pass
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True);torch.backends.cuda.matmul.allow_tf32=False
    runtime=load_real_runtime(SimpleNamespace(cpu_gate=P/'private/cpu_gate'));runtime.model.eval();r.old.fp32(runtime)
    assert not any(p.requires_grad for p in runtime.model.parameters())
    handle=runtime.model.register_forward_pre_hook(lambda *_:counts.__setitem__('forwards',counts['forwards']+1))
    vision_handle=runtime.model.get_vision_tower().register_forward_pre_hook(lambda *_:counts.__setitem__('vision_forwards',counts['vision_forwards']+1))
    write('MODEL_READY.json',dict(time=time.time(),gpu=os.environ['CUDA_VISIBLE_DEVICES'],commands=subprocess.check_output(['ps','-ww','-p',str(os.getpid()),'-o','pid=,args='],text=True),gpu_processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,process_name,used_memory','--format=csv,noheader'],text=True)))
    from study import run
    run(runtime,r,OUT,int(os.environ['GPU']),counts)
    handle.remove();vision_handle.remove()
except BaseException:
    code=1;write('FAILURE.json',dict(error=traceback.format_exc(),counts=counts,automatic_retry=False));raise
finally:write('EXIT.json',dict(code=code,counts=counts,seconds=time.time()-start))
