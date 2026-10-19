"""One-attempt Base qualification only; no training or duplicate generation."""
import os,sys,json,time,traceback,subprocess,shutil
from pathlib import Path
from types import SimpleNamespace
OUT=Path(os.environ['OUTPUT_RUN']);P=Path(os.environ['PARENT_RUN']);start=time.time()
counts=dict(forwards=0,backwards=0,updates=0,generations=0,vision_forwards=0)
def write(name,value):
    path=OUT/name;path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def budget():
    assert time.time()<float(os.environ['BASE_DEADLINE_EPOCH']) and not (OUT.parent/'STOP').exists()
    assert counts['forwards']<=400000 and counts['generations']<=325 and counts['backwards']==counts['updates']==0
    assert shutil.disk_usage(OUT).free>8*1024**3

def main():
    with (OUT/'ATTEMPT').open('x'):pass
    (OUT/'attempts').mkdir(exist_ok=True)
    sys.path[:0]=[str(OUT/'tools'),str(P/'private/tools'),str(P/'private/source'),str(P/'private/official_llava')]
    import torch
    import module_run as r
    from scripts.medtrace.run_realmodel_core import load_real_runtime
    r.c.budget=budget
    torch.use_deterministic_algorithms(True);torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True);torch.backends.cuda.matmul.allow_tf32=False
    rows=json.loads((OUT/'CANDIDATES.json').read_text());assert 1<=len(rows)<=325 and len({x['query_id'] for x in rows})==len(rows)
    budget();runtime=load_real_runtime(SimpleNamespace(cpu_gate=P/'private/cpu_gate'));runtime.model.eval();r.old.fp32(runtime)
    assert not any(p.requires_grad for p in runtime.model.parameters())
    def count(*_):
        budget();assert counts['forwards']<400000;counts['forwards']+=1
    handle=runtime.model.register_forward_pre_hook(count);vision=runtime.model.get_vision_tower().register_forward_pre_hook(lambda *_:counts.__setitem__('vision_forwards',counts['vision_forwards']+1))
    write('MODEL_READY.json',dict(time=time.time(),gpu=os.environ['CUDA_VISIBLE_DEVICES'],commands=subprocess.check_output(['ps','-ww','-p',str(os.getpid()),'-o','pid=,args='],text=True)))
    bindings=json.loads((P/'private/EVAL_BINDINGS.json').read_text());assert runtime.generation_config==next(iter(bindings.values()))['generation']
    task=json.loads((P/'private/M13_SOURCE_PACKET.json').read_text())['original_tasks'][4]
    runtime.model.half();items=r.s.prepare(runtime,bindings,task,rows,False);r.old.fp32(runtime)
    try:
        for row,_,raw,expanded,binding in items:
            budget();qid=row['query_id'];assert not (OUT/'outputs/BASE'/(qid+'.json')).exists()
            with (OUT/'attempts'/(qid+'.json')).open('x') as f:json.dump(dict(attempt=1,time=time.time()),f)
            counts['generations']+=1;answer,trace=r.s.fit.short.generate(runtime,None,raw,expanded);tokens=list(answer.raw_token_ids)
            write('outputs/BASE/'+qid+'.json',dict(binding=dict(binding,preprocessing_seed=raw['preprocessing_seed']),raw_answer=answer.decoded_text,raw_token_ids=tokens,EOS=bool(tokens) and tokens[-1]==runtime.adapter.tokenizer.eos_token_id,at_cap=len(tokens)>=1024))
            write('PROGRESS.json',dict(last=qid,completed=counts['generations'],total=len(rows),counts=counts,time=time.time()))
        assert counts['generations']==len(rows)
        write('COMPLETE.json',dict(status='BASE_QUALIFICATION_COMPLETE',counts=counts,total=len(rows),seconds=time.time()-start))
    finally:handle.remove();vision.remove()

if __name__=='__main__':
    code=0
    try:main()
    except BaseException:
        code=1;write('FAILURE.json',dict(error=traceback.format_exc(),counts=counts,automatic_retry=False));raise
    finally:write('EXIT.json',dict(code=code,counts=counts,seconds=time.time()-start))
