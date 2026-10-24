"""Five disjoint benchmark shards; no wall-clock deadline or implicit retries."""
import os,sys,json,time,traceback,subprocess,shutil
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(os.environ['V22_ROOT']);OUT=Path(os.environ['OUTPUT_RUN']);P=Path(os.environ['PARENT_RUN']);PART=int(os.environ['PARTITION']);START=time.time();last_space=0.
counts=dict(forwards=0,backwards=0,updates=0,generations=0,vision_forwards=0,candidate_checks=0,query_keys=0)
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
def budget():
    global last_space
    assert not (ROOT/'STOP').exists()
    assert time.time()-START<16*3600,'Finite GPU-process cost budget exhausted; not a calendar deadline'
    assert counts['updates']<=1800 and counts['backwards']<=60000 and counts['forwards']<=600000 and counts['generations']<=800
    if time.time()-last_space>60:
        assert shutil.disk_usage(ROOT).free>8*1024**3
        assert sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())<14000000000
        last_space=time.time()
def main():
    with (OUT/'COMPACT_ATTEMPT').open('x'):pass
    sys.path[:0]=[str(ROOT/'tools'),str(P/'private/tools'),str(P/'private/source'),str(P/'private/official_llava')]
    import torch
    import module_run as r
    from scripts.medtrace.run_realmodel_core import load_real_runtime
    assert read(ROOT/'FORMAL_CPU_ADMISSION.json')['status']=='PASS'
    r.c.budget=budget
    torch.use_deterministic_algorithms(True);torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True);torch.backends.cuda.matmul.allow_tf32=False
    budget();runtime=load_real_runtime(SimpleNamespace(cpu_gate=P/'private/cpu_gate'));runtime.model.eval();r.old.fp32(runtime)
    assert not any(p.requires_grad for p in runtime.model.parameters())
    def count(*_):budget();assert counts['forwards']<600000;counts['forwards']+=1
    h=runtime.model.register_forward_pre_hook(count);vh=runtime.model.get_vision_tower().register_forward_pre_hook(lambda *_:counts.__setitem__('vision_forwards',counts['vision_forwards']+1))
    write(OUT/'COMPACT_MODEL_READY.json',dict(time=time.time(),partition=PART,gpu=os.environ['CUDA_VISIBLE_DEVICES'],command=subprocess.check_output(['ps','-ww','-p',str(os.getpid()),'-o','args='],text=True)))
    try:
        import study_compact_v22
        assert Path(study_compact_v22.__file__).resolve()==ROOT/'tools/study_compact_v22.py'
        template=read(P/'private/M13_SOURCE_PACKET.json')['original_tasks'][4];bindings=read(P/'private/EVAL_BINDINGS.json');queries=read(ROOT/'QUERY_ROWS.json');keys={}
        assert runtime.generation_config==next(iter(bindings.values()))['generation']
        assert all((ROOT/str(i)/'BASE_COMPLETE.json').exists() for i in range(5))
        assert len(list((ROOT/'base').glob('*/*.json')))==len(queries)
        keys=torch.load(OUT/'QUERY_KEYS.pt',map_location='cpu',weights_only=True)
        write(OUT/'COMPACT_REUSE_ADMISSION.json',dict(status='PASS',OFF_answers=len(queries),query_keys=len(keys),new_Base_forward=0,new_generation=0,counts=counts))
        study_compact_v22.run(runtime,r,OUT,PART,counts)
    finally:h.remove();vh.remove()
if __name__=='__main__':
    code=0
    try:main()
    except BaseException:
        code=1;write(OUT/'COMPACT_FAILURE.json',dict(error=traceback.format_exc(),counts=counts,automatic_retry=False));raise
    finally:write(OUT/'COMPACT_EXIT.json',dict(code=code,counts=counts,seconds=time.time()-START))
