"""Frozen MedMKEB Base qualification and conditional shared-runtime six-arm handoff."""
import os,sys,json,time,traceback,subprocess,copy
from pathlib import Path
from types import SimpleNamespace
P=Path(os.environ['PARENT_RUN']);OUT=Path(os.environ['OUTPUT_RUN'])
sys.path[:0]=[str(OUT/'tools'),str(P/'private/tools'),str(P/'private/source'),str(P/'private/official_llava')]
import torch
import module_run as r
import trajectory,training_math
import owner5_semantic_loss as semantic
from prefix_cache import PrefixCache
from scripts.medtrace.run_realmodel_core import load_real_runtime
from m3bench_repro.editors.llava_runtime import seed_everything

def read(p):return json.loads(Path(p).read_text())
def write(name,value):
    path=OUT/name;path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,default=str)+'\n');tmp.replace(path)
def save(path,value):
    tmp=path.with_suffix('.tmp');torch.save(value,tmp);tmp.replace(path)
def budget():
    assert not (OUT/'STOP').exists(),'Explicit stop'
    import shutil
    assert shutil.disk_usage(OUT).free>8*1024**3
    assert time.time()<float(os.environ['DEADLINE_EPOCH']),'Finite chain deadline'
    assert counts['updates']<=960 and counts['backwards']<=9000 and counts['operator_backwards']<=550 and counts['generations']<=325
    assert counts['forwards']<=400000

def prepare(runtime,bindings,task,rows,training):
    runtime.model.half()
    items=r.s.prepare(runtime,bindings,task,rows,training)
    for row,batch,raw,expanded,binding in items:
        name=row['query_id']
        write(('training/' if training else 'evaluation/')+'bindings/'+name+'.json',dict(binding,prompt=raw['prompt'],preprocessing_seed=raw['preprocessing_seed']))
        save(OUT/(('train_inputs_' if training else 'eval_inputs_')+name+'.pt'),dict(expanded=expanded,prompt_ids=raw['input_ids'].cpu(),images=raw['images'].cpu() if isinstance(raw['images'],torch.Tensor) else [x.cpu() for x in raw['images']]))
    r.old.fp32(runtime)
    return items

def main():
    with (OUT/'ATTEMPT').open('x') as f:f.write(str(os.getpid()))
    torch.use_deterministic_algorithms(True)
    runtime=load_real_runtime(SimpleNamespace(cpu_gate=P/'private/cpu_gate'));runtime.model.eval()
    assert not any(p.requires_grad for p in runtime.model.parameters())
    assert not runtime.get_module(r.c.LAYER)._forward_hooks
    handle=runtime.model.register_forward_pre_hook(lambda *_:counts.__setitem__('forwards',counts['forwards']+1))
    bindings=read(P/'private/EVAL_BINDINGS.json')
    assert runtime.generation_config==next(iter(bindings.values()))['generation']
    task=read(P/'private/M13_SOURCE_PACKET.json')['original_tasks'][4]
    write('MODEL_READY.json',dict(time=time.time(),gpu=os.environ['CUDA_VISIBLE_DEVICES']))
    while not (OUT/'CANDIDATES.json').exists():
        budget()
        if (OUT/'DATA_GATE_FAILED.json').exists():write('COMPLETE.json',dict(status='NO_QUALIFIED_CANDIDATES',counts=counts));return
        time.sleep(10)
    rows=read(OUT/'CANDIDATES.json');assert 1<=len(rows)<=24
    items=prepare(runtime,bindings,task,rows,False)
    write('PROCESS.json',dict(commands=subprocess.check_output(['ps','-ww','-p',str(os.getpid()),'-o','pid=,args='],text=True),gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,process_name,used_memory','--format=csv,noheader'],text=True)))
    try:
        for row,_,raw,expanded,binding in items:
            budget();dest='outputs/BASE/'+row['query_id']+'.json';assert not (OUT/dest).exists()
            write('attempts/'+row['query_id']+'.json',dict(attempt=1,time=time.time()));counts['generations']+=1
            answer,trace=r.s.fit.short.generate(runtime,None,raw,expanded)
            write(dest,dict(binding=binding,raw_answer=answer.decoded_text,raw_token_ids=list(answer.raw_token_ids),EOS=answer.raw_token_ids[-1]==runtime.adapter.tokenizer.eos_token_id,at_cap=len(answer.raw_token_ids)>=1024))
        assert counts['generations']==len(rows) and counts['updates']==counts['backwards']==0
        write('COMPLETE.json',dict(status='BASE_QUALIFICATION_COMPLETE',counts=counts,seconds=time.time()-START))
        while not (OUT.parent/'DATA_ADMISSION.json').exists():
            budget();write('RESIDENCY.json',dict(time=time.time(),status='BASE_QUALIFICATION_COMPLETE_MODEL_RESIDENT'));time.sleep(10)
        admission=read(OUT.parent/'DATA_ADMISSION.json')
        if admission['status']=='PASS':
            from study import run
            run(runtime,r,OUT,int(os.environ['CUDA_VISIBLE_DEVICES']),counts)
            completion=read(OUT/'STUDY_COMPLETE.json')
            write('FINAL_COMPLETE.json',dict(status='SIX_ARM_PARTITION_COMPLETE' if completion['status']=='COMPLETE' else completion['status'],counts=counts))
        else:write('FINAL_COMPLETE.json',dict(status='DATA_GATE_NOT_MET',counts=counts))
    finally:handle.remove()

START=time.time();counts=dict(updates=0,backwards=0,operator_backwards=0,generations=0,forwards=0)
r.c.budget=budget
code=0
try:main()
except BaseException:
    code=1;write('FAILURE.json',dict(error=traceback.format_exc(),counts=counts,automatic_retry=False));raise
finally:write('EXIT.json',dict(code=code,counts=counts,seconds=time.time()-START))
