"""Fresh formal worker on the already frozen V18 Base admission."""
import os,sys,json,time,traceback,subprocess
from pathlib import Path
from types import SimpleNamespace
OUT=Path(os.environ['OUTPUT_RUN']);P=Path(os.environ['PARENT_RUN']);start=time.time()
counts=dict(forwards=0,backwards=0,updates=0,generations=0,vision_forwards=0,candidate_checks=0)
def write(name,value):
    path=OUT/name;path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def main():
    with (OUT/'FORMAL_ATTEMPT').open('x'):pass
    sys.path[:0]=[str(OUT/'tools'),str(P/'private/tools'),str(P/'private/source'),str(P/'private/official_llava')]
    import torch
    import module_run as r
    from scripts.medtrace.run_realmodel_core import load_real_runtime
    torch.use_deterministic_algorithms(True);torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True);torch.backends.cuda.matmul.allow_tf32=False
    def budget():assert time.time()<float(os.environ['DEADLINE_EPOCH']) and not (OUT.parent/'STOP').exists()
    r.c.budget=budget;budget();runtime=load_real_runtime(SimpleNamespace(cpu_gate=P/'private/cpu_gate'));runtime.model.eval();r.old.fp32(runtime);assert not any(p.requires_grad for p in runtime.model.parameters())
    def count(*_):r.c.budget();assert counts['forwards']<3000000;counts['forwards']+=1
    handle=runtime.model.register_forward_pre_hook(count);vision=runtime.model.get_vision_tower().register_forward_pre_hook(lambda *_:counts.__setitem__('vision_forwards',counts['vision_forwards']+1))
    write('FORMAL_MODEL_READY.json',dict(time=time.time(),gpu=os.environ['CUDA_VISIBLE_DEVICES'],commands=subprocess.check_output(['ps','-ww','-p',str(os.getpid()),'-o','pid=,args='],text=True)))
    import study_v18
    assert Path(study_v18.__file__).resolve()==(OUT/'tools/study_v18.py').resolve()
    try:study_v18.run(runtime,r,OUT,int(os.environ['GPU']),counts)
    finally:handle.remove();vision.remove()
if __name__=='__main__':
    code=0
    try:main()
    except BaseException:
        code=1;write('FORMAL_FAILURE.json',dict(error=traceback.format_exc(),counts=counts,automatic_retry=False));raise
    finally:write('FORMAL_EXIT.json',dict(code=code,counts=counts,seconds=time.time()-start))
