"""Paired Base and fixed MED expert answers on the role-safe source CHECK."""
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

ROOT=Path(os.environ['EXPERIMENT_ROOT']);PARENT=Path(os.environ['PARENT_RUN']);MED=Path(os.environ['MED150_ROOT']);OLD=Path(os.environ['RAW_MED_ROOT'])
LOCK=json.loads((ROOT/'EVAL_LOCK.json').read_text());CLOCK=json.loads((ROOT/'CLOCK.json').read_text())
COUNTS=dict(forwards=0,generations=0,backwards=0,updates=0,vision_forwards=0)

def write(name,value):
    p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2)+'\n');t.replace(p)
def budget():
    assert time.time()<CLOCK['hard_end']-30 and not (ROOT/'EVAL_STOP.json').exists()
    assert time.time()-json.loads((ROOT/'EVAL_GUARD_HEALTH.json').read_text())['time']<45
    for k,v in LOCK['budgets'].items():
        if k in COUNTS:assert COUNTS[k]<=v,(k,COUNTS[k],v)


def main():
    with (ROOT/'EVAL_WORKER_ONCE').open('x'):pass
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(InterruptedError('Guard stop')))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Original clock')))
    signal.alarm(max(1,int(CLOCK['hard_end']-time.time()-30)))
    sys.path[:0]=[str(ROOT/'tools'),str(PARENT/'private/tools'),str(PARENT/'private/source'),str(PARENT/'private/official_llava')]
    import numpy as np
    import torch
    import module_run as r
    import med133_evaluate
    from scripts.medtrace.run_realmodel_core import load_real_runtime
    from types import SimpleNamespace
    r.c.budget=budget;torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    assert os.environ['CUDA_VISIBLE_DEVICES']==str(LOCK['physical_GPU']);torch.cuda.set_per_process_memory_fraction(.78)
    plan=json.loads((ROOT/'CHECK_INPUTS_FROZEN.json').read_text());assert len(plan['queries'])==93 and r.c.digest(plan)==LOCK['input_binding']
    pkg=torch.load(MED/'retained/MED150.pt',map_location='cpu',weights_only=True);tasks=pkg['tasks'];states=pkg['states'];hashes=pkg['state_hashes']
    assert len(tasks)==len(states)==150 and all(med133_evaluate.state_id(v)==hashes[k] for k,v in states.items())
    runtime=load_real_runtime(SimpleNamespace(cpu_gate=PARENT/'private/cpu_gate'));runtime.model.eval();r.old.fp32(runtime)
    assert not any(p.requires_grad for p in runtime.model.parameters())
    bindings=r.c.read(PARENT/'private/EVAL_BINDINGS.json');assert runtime.generation_config==next(iter(bindings.values()))['generation']
    rng=torch.load(OLD/'GENERATION_RNG.pt',map_location='cpu',weights_only=False)
    def count(_m,args,kwargs):budget();COUNTS['forwards']+=1
    fh=runtime.model.register_forward_pre_hook(count,with_kwargs=True)
    vh=runtime.model.get_vision_tower().register_forward_pre_hook(lambda *_:COUNTS.__setitem__('vision_forwards',COUNTS['vision_forwards']+1))
    image_by_group={}
    for begin in range(0,len(plan['queries']),8):
        budget();runtime.model.half();prepared=[]
        try:
            for row in plan['queries'][begin:begin+8]:
                idx=row['forced_expert_index'];assert 0<=idx<32
                prepared.append((row,tasks[idx],r.s.prepare(runtime,bindings,tasks[idx],[row])[0]))
        finally:r.old.fp32(runtime)
        for row,task,item in prepared:
            qid=row['query_id'];g=row['source_group'];b=item[4]
            if g in image_by_group:assert image_by_group[g]==b['image_sha256']
            else:image_by_group[g]=b['image_sha256']
            for arm in ('BASE','MED_FORCED'):
                budget();cid=json.dumps([arm,qid],separators=(',',':'));dest=ROOT/'outputs'/(r.c.digest(cid)+'.json');assert not dest.exists(),'No implicit retries'
                hook=None
                try:
                    if arm=='MED_FORCED':
                        state=states[task['edit_id']];expert=r.p.expert(state,task['seed'],runtime.device).requires_grad_(False)
                        hook=r.q.scoped.replay.MedTraceLayerHook(runtime.get_module(r.c.LAYER),expert);hook.attach()
                    r.c.restore_rng(rng['rng']);np.random.set_state(rng['numpy_rng']);COUNTS['generations']+=1
                    value,trace=r.s.fit.short.generate(runtime,hook,item[2],item[3]);observed=r.c.rng()
                    assert observed['python_rng']==rng['rng']['python_rng']
                    for key in ('torch_rng','cuda_rng'):assert torch.equal(observed[key],rng['rng'][key])
                    answer=dict(raw_answer=value.decoded_text,raw_token_ids=list(value.raw_token_ids));assert answer['raw_token_ids']
                finally:
                    if hook is not None:hook.detach()
                write(str(dest.relative_to(ROOT)),dict(method=arm,mode='FORCED_STRESS',prefix=150,owner=task['order'],consumer_id=cid,
                    binding=dict(input=row,judge_input=b),R0=answer,route=None,effective_expert=task['edit_id'] if arm=='MED_FORCED' else None,
                    effective_state_hash=hashes[task['edit_id']] if arm=='MED_FORCED' else 'UNEDITED_BASE',generation_trace=trace,
                    EOS=answer['raw_token_ids'][-1]==runtime.adapter.tokenizer.eos_token_id,at_cap=len(answer['raw_token_ids'])>=1024))
        write('EVAL_PROGRESS.json',dict(time=time.time(),queries=min(begin+8,93),total=93,counts=COUNTS))
    assert len(image_by_group)==31 and len(list((ROOT/'outputs').glob('*.json')))==186 and COUNTS['generations']==186
    fh.remove();vh.remove();write('EVAL_GENERATION_COMPLETE.json',dict(time=time.time(),consumers=186,queries=93,groups=31,counts=COUNTS,forced_activation=True,not_natural_routing=True,cuda_max_memory_allocated=torch.cuda.max_memory_allocated()))

if __name__=='__main__':
    code=0
    try:main()
    except BaseException:
        code=1;write('EVAL_FAILURE.json',dict(time=time.time(),error=traceback.format_exc(),counts=COUNTS));raise
    finally:write('EVAL_WORKER_EXIT.json',dict(time=time.time(),code=code,counts=COUNTS))
