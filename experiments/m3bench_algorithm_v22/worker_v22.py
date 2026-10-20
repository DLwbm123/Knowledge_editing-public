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
    assert time.time()-START<60*3600,'Finite GPU-process cost budget exhausted; not a calendar deadline'
    assert counts['updates']<=27000 and counts['backwards']<=800000 and counts['forwards']<=8000000 and counts['generations']<=6000
    if time.time()-last_space>60:
        assert shutil.disk_usage(ROOT).free>8*1024**3
        assert sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())<14000000000
        last_space=time.time()
def main():
    with (OUT/'FORMAL_ATTEMPT').open('x'):pass
    sys.path[:0]=[str(ROOT/'tools'),str(P/'private/tools'),str(P/'private/source'),str(P/'private/official_llava')]
    import torch
    import module_run as r
    from scripts.medtrace.run_realmodel_core import load_real_runtime
    assert read(ROOT/'FORMAL_CPU_ADMISSION.json')['status']=='PASS'
    r.c.budget=budget
    torch.use_deterministic_algorithms(True);torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True);torch.backends.cuda.matmul.allow_tf32=False
    budget();runtime=load_real_runtime(SimpleNamespace(cpu_gate=P/'private/cpu_gate'));runtime.model.eval();r.old.fp32(runtime)
    assert not any(p.requires_grad for p in runtime.model.parameters())
    def count(*_):budget();assert counts['forwards']<8000000;counts['forwards']+=1
    h=runtime.model.register_forward_pre_hook(count);vh=runtime.model.get_vision_tower().register_forward_pre_hook(lambda *_:counts.__setitem__('vision_forwards',counts['vision_forwards']+1))
    write(OUT/'FORMAL_MODEL_READY.json',dict(time=time.time(),partition=PART,gpu=os.environ['CUDA_VISIBLE_DEVICES'],command=subprocess.check_output(['ps','-ww','-p',str(os.getpid()),'-o','args='],text=True)))
    try:
        import study_v22
        assert Path(study_v22.__file__).resolve()==ROOT/'tools/study_v22.py'
        template=read(P/'private/M13_SOURCE_PACKET.json')['original_tasks'][4];bindings=read(P/'private/EVAL_BINDINGS.json');queries=read(ROOT/'QUERY_ROWS.json');keys={}
        assert runtime.generation_config==next(iter(bindings.values()))['generation']
        for i,qid in enumerate(sorted(queries)):
            if i%5!=PART:continue
            row=queries[qid];dest=ROOT/'base'/str(PART)/(qid+'.json');assert not dest.exists()
            runtime.model.half();item=r.s.prepare(runtime,bindings,template,[row],False)[0];r.old.fp32(runtime)
            _,_,raw,expanded,binding=item
            write(OUT/'base_attempts'/(qid+'.json'),dict(attempt=1,time=time.time()));counts['generations']+=1
            answer,trace=r.s.fit.short.generate(runtime,None,raw,expanded);tokens=list(answer.raw_token_ids)
            write(dest,dict(binding=dict(binding,preprocessing_seed=raw['preprocessing_seed']),raw_answer=answer.decoded_text,raw_token_ids=tokens,generation_trace=trace,EOS=bool(tokens) and tokens[-1]==runtime.adapter.tokenizer.eos_token_id,at_cap=len(tokens)>=1024,mode='OFF_CURRENT_PRECISION'))
            # One registered query feature forward, no answer/reference supplied to extraction.
            with torch.no_grad():
                batch=runtime.build_question_batch(r.s.record(template,row));features=runtime.extract_layer_input_features(batch,module_path=runtime.target_lock['balancedit']['targets'][0])
                mask=torch.ones(features.shape[0],dtype=torch.bool,device=features.device) if batch.attention_mask is None else batch.attention_mask[0].bool();im=mask.clone();im[:batch.image_token_start]=False;im[batch.image_token_end:]=False;txt=mask&~im
                assert im.any() and txt.any();keys[qid]={k:features[m].mean(0).detach().cpu() for k,m in [('joint',mask),('image',im),('text',txt)]}
                assert all(torch.isfinite(z).all() for z in keys[qid].values())
            counts['query_keys']+=1
            if counts['query_keys']%10==0:
                tmp=OUT/'QUERY_KEYS.tmp';torch.save(keys,tmp);tmp.replace(OUT/'QUERY_KEYS.pt');write(OUT/'PROGRESS.json',dict(stage='OFF_BASE_AND_KEYS',completed=counts['query_keys'],counts=counts,time=time.time()))
            del item,raw,expanded,batch,features
        torch.save(keys,OUT/'QUERY_KEYS.pt');write(OUT/'BASE_COMPLETE.json',dict(status='PASS',queries=len(keys),counts=counts,time=time.time()))
        # No shard changes shared reference values while another shard is still building OFF.
        while not all((ROOT/str(i)/'BASE_COMPLETE.json').exists() for i in range(5)):
            budget();assert not any((ROOT/str(i)/'FORMAL_FAILURE.json').exists() for i in range(5));time.sleep(5)
        study_v22.run(runtime,r,OUT,PART,counts)
    finally:h.remove();vh.remove()
if __name__=='__main__':
    code=0
    try:main()
    except BaseException:
        code=1;write(OUT/'FORMAL_FAILURE.json',dict(error=traceback.format_exc(),counts=counts,automatic_retry=False));raise
    finally:write(OUT/'FORMAL_EXIT.json',dict(code=code,counts=counts,seconds=time.time()-START))
