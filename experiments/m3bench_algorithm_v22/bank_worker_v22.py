"""Registered V22 SOURCE-bank consumer. No training, Base or feature extraction."""
import os,sys,json,time,traceback,shutil
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(os.environ['V22_ROOT']);PART=int(os.environ['PARTITION']);OUT=ROOT/'bank'/str(PART)
P=Path(os.environ['PARENT_RUN']);START=time.time()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
counts=dict(forwards=0,backwards=0,updates=0,generations=0,vision_forwards=0,query_keys=0)
last_space=0.
def budget():
    global last_space
    limits=read(ROOT/'BANK_ALLOCATION.json');hard=read(ROOT/'RESOURCE_LIMITS.json')
    assert not (ROOT/'STOP').exists() and time.time()<hard['hard_end']-300
    assert time.time()-START<limits['seconds_per_partition']
    assert counts['forwards']<=limits['forwards_per_partition'] and counts['generations']<=limits['generations_per_partition'][str(PART)]
    assert shutil.disk_usage(ROOT).free>8*1024**3
    if time.time()-last_space>60:
        assert sum(p.stat().st_size for p in (ROOT/'bank').rglob('*') if p.is_file())<1000000000
        last_space=time.time()

def main():
    with (OUT/'ATTEMPT').open('x'):pass
    # Launcher registers identity before opening the go barrier; no model work before it.
    assert (OUT/'GO.json').exists()
    ident=read(OUT/'LAUNCH.json');assert ident['pid']==os.getpid()
    assert any(t['pid']==ident['pid'] and str(t['start_ticks'])==str(ident['start_ticks']) for t in read(ROOT/'RESOURCE_TARGETS.json')['targets'])
    sys.path[:0]=[str(ROOT/'tools'),str(P/'private/tools'),str(P/'private/source'),str(P/'private/official_llava')]
    import torch
    import module_run as r
    from scripts.medtrace.run_realmodel_core import load_real_runtime
    from path_v18 import same
    budget();assert read(ROOT/'BANK_CPU_ADMISSION.json')['status']=='PASS'
    r.c.budget=budget;torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True);torch.backends.cuda.matmul.allow_tf32=False
    runtime=load_real_runtime(SimpleNamespace(cpu_gate=P/'private/cpu_gate'));runtime.model.eval();r.old.fp32(runtime)
    assert not any(p.requires_grad for p in runtime.model.parameters())
    def count(*_):
        budget();assert counts['forwards']<read(ROOT/'BANK_ALLOCATION.json')['forwards_per_partition'];counts['forwards']+=1
    h=runtime.model.register_forward_pre_hook(count)
    vh=runtime.model.get_vision_tower().register_forward_pre_hook(lambda *_:counts.__setitem__('vision_forwards',counts['vision_forwards']+1))
    try:
        write(OUT/'MODEL_READY.json',dict(time=time.time(),physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],counts=counts))
        plan=read(ROOT/'ROUTE_PLAN_FROZEN.json');template=read(P/'private/M13_SOURCE_PACKET.json')['original_tasks'][4];bindings=read(P/'private/EVAL_BINDINGS.json')
        query_order=sorted(read(ROOT/'QUERY_ROWS.json'));basepaths={q:ROOT/'base'/str(i%5)/(q+'.json') for i,q in enumerate(query_order)}
        expert=r.p.expert(r.p.tr.TT4(template['seed'],8,8).state_dict(),template['seed'],runtime.device)
        hook=r.q.scoped.replay.MedTraceLayerHook(runtime.get_module(r.c.LAYER),expert)
        expected=[]
        for task in plan['tasks'][PART::5]:
            key=task['key'];folder=ROOT/str((task['order']-1)%5)/key
            assert read(folder/'MODEL_CONSUMERS_COMPLETE.json')['status']=='PASS'
            state=torch.load(folder/'SOURCE/final.pt',map_location=runtime.device,weights_only=True)['state'];expert.load_state_dict(state);assert same(r.old.snapshot(expert),state)
            missing=[q for q in sorted(plan['queries']) if key+'|'+q not in plan['existing_source_outputs']]
            # At most one prepared image at a time; all baseline generation bindings must match.
            hook.attach()
            try:
                for q in missing:
                    budget();dest=ROOT/'bank/outputs'/key/(q+'.json');assert not dest.exists()
                    row=plan['queries'][q];runtime.model.half();item=r.s.prepare(runtime,bindings,template,[row],False)[0];r.old.fp32(runtime)
                    _,_,raw,expanded,binding=item;old=read(basepaths[q])['binding']
                    assert all(old[k]==binding[k] for k in ('image_sha256','question','reference','prompt_ids','generation'))
                    write(OUT/'attempts'/key/(q+'.json'),dict(attempt=1,time=time.time()));counts['generations']+=1
                    answer,trace=r.s.fit.short.generate(runtime,hook,raw,expanded);tokens=list(answer.raw_token_ids)
                    write(dest,dict(binding=binding,raw_answer=answer.decoded_text,raw_token_ids=tokens,generation_trace=trace,EOS=bool(tokens) and tokens[-1]==runtime.adapter.tokenizer.eos_token_id,at_cap=len(tokens)>=1024,scope='FORCED_SOURCE_BANK'))
                    expected.append((key,q));write(OUT/'PROGRESS.json',dict(expert=key,last_query=q,counts=counts,time=time.time()))
                    del item,raw,expanded
            finally:hook.detach()
            assert same(r.old.snapshot(expert),state),'Inference modified SOURCE state'
        assert len(expected)==read(ROOT/'BANK_ALLOCATION.json')['generations_per_partition'][str(PART)]
        write(OUT/'MODEL_CONSUMERS_COMPLETE.json',dict(status='PASS',answers=len(expected),source_state_unchanged=True,save_load_exact=True,counts=counts))
    finally:h.remove();vh.remove()

if __name__=='__main__':
    code=0
    try:main()
    except BaseException:
        code=1;write(OUT/'FAILURE.json',dict(error=traceback.format_exc(),counts=counts,automatic_retry=False));raise
    finally:write(OUT/'EXIT.json',dict(code=code,counts=counts,seconds=time.time()-START))
