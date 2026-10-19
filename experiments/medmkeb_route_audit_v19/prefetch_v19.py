"""Consume the existing 28-key budget early, without any expert or generation."""
import os,sys,json,time,traceback,subprocess
from pathlib import Path
from types import SimpleNamespace
OUT=Path(os.environ['OUTPUT_RUN']);V18=Path(os.environ['V18_RUN']);P=Path(os.environ['PARENT_RUN']);began=time.time();counts=dict(forwards=0,vision_forwards=0,generations=0,backwards=0,updates=0,route_keys=0)
def read(p):return json.loads(Path(p).read_text())
def write(name,value):
    p=OUT/name;tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p)
def main():
    with (OUT/'PREFETCH_ATTEMPT').open('x'):pass
    sys.path[:0]=[str(OUT/'tools'),str(V18/os.environ['PARTITION']/'tools'),str(P/'private/tools'),str(P/'private/source'),str(P/'private/official_llava')]
    import torch
    import module_run as r
    from scripts.medtrace.run_realmodel_core import load_real_runtime
    def budget():assert time.time()<float(os.environ['PREFETCH_DEADLINE']) and counts['forwards']<=14 and counts['vision_forwards']<=14
    r.c.budget=budget;torch.use_deterministic_algorithms(True);torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True);torch.backends.cuda.matmul.allow_tf32=False
    runtime=load_real_runtime(SimpleNamespace(cpu_gate=P/'private/cpu_gate'));runtime.model.eval();r.old.fp32(runtime);assert not any(p.requires_grad for p in runtime.model.parameters())
    def count(*_):budget();assert counts['forwards']<14;counts['forwards']+=1
    handle=runtime.model.register_forward_pre_hook(count);vision=runtime.model.get_vision_tower().register_forward_pre_hook(lambda *_:counts.__setitem__('vision_forwards',counts['vision_forwards']+1))
    write('PREFETCH_MODEL_READY.json',dict(time=time.time(),gpu=os.environ['GPU'],command=subprocess.check_output(['ps','-ww','-p',str(os.getpid()),'-o','args='],text=True).strip()))
    try:
        tasks=read(V18/'DATA_ADMISSION.json')['tasks'];rows=read(V18/'EVAL_ROWS.json')+[t[k] for t in tasks for k in ('source','cross','rephrase')];rows=rows[int(os.environ['PARTITION'])::2];assert len(rows)==14
        template=read(P/'private/M13_SOURCE_PACKET.json')['original_tasks'][4];reuse=read(V18/'BASE_REUSE.json');keys={};bindings={}
        for row in rows:
            budget();qid=row['query_id'];base=read(reuse[qid])['binding'];batch=runtime.build_question_batch(r.s.record(template,row));assert batch.image_sha256==base['image_sha256'] and batch.raw_input_ids[0].tolist()==base['prompt_ids'] and batch.preprocessing_seed==base['preprocessing_seed'];assert runtime.generation_config==base['generation']
            with torch.inference_mode():key=runtime.extract_layer_input_key(batch,module_path=runtime.target_lock['balancedit']['targets'][0],pooling='mean')
            assert torch.isfinite(key).all();keys[qid]=key.detach().cpu();bindings[qid]=base;counts['route_keys']+=1
            write('PREFETCH_PROGRESS.json',dict(completed=counts['route_keys'],counts=counts,time=time.time()))
        budget();assert counts['route_keys']==14;tmp=OUT/'QUERY_CACHE.tmp';torch.save(dict(keys=keys,bindings=bindings,counts=counts),tmp);tmp.replace(OUT/'QUERY_CACHE.pt');write('PREFETCH_COMPLETE.json',dict(status='PASS',counts=counts,seconds=time.time()-began))
    finally:handle.remove();vision.remove()
if __name__=='__main__':
    code=0
    try:main()
    except BaseException:code=1;write('PREFETCH_FAILURE.json',dict(error=traceback.format_exc(),counts=counts));raise
    finally:write('PREFETCH_EXIT.json',dict(code=code,counts=counts,seconds=time.time()-began))
