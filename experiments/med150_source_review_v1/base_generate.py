"""Generate Base answers for the fixed source-annotation feasibility panel."""
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

ROOT=Path(os.environ['EXPERIMENT_ROOT']);PARENT=Path(os.environ['PARENT_RUN']);MED=Path(os.environ['MED150_ROOT']);OLD=Path(os.environ['RAW_MED_ROOT'])
LOCK=json.loads((ROOT/'BASE_LOCK.json').read_text());CLOCK=json.loads((ROOT/'CLOCK.json').read_text())
COUNTS=dict(forwards=0,generations=0,backwards=0,updates=0,vision_forwards=0)


def write(name,value):
    p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2)+'\n');t.replace(p)
def budget():
    assert time.time()<CLOCK['hard_end']-30 and not (ROOT/'BASE_STOP.json').exists()
    assert time.time()-json.loads((ROOT/'BASE_GUARD_HEALTH.json').read_text())['time']<45
    for k,v in LOCK['budgets'].items():
        if k in COUNTS:assert COUNTS[k]<=v,(k,COUNTS[k],v)


def main():
    with (ROOT/'BASE_WORKER_ONCE').open('x'):pass
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(InterruptedError('Guard stop')))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Original clock')))
    signal.alarm(max(1,int(CLOCK['hard_end']-time.time()-30)))
    sys.path[:0]=[str(ROOT/'tools'),str(PARENT/'private/tools'),str(PARENT/'private/source'),str(PARENT/'private/official_llava')]
    import numpy as np
    import torch
    import module_run as r
    from scripts.medtrace.run_realmodel_core import load_real_runtime
    from types import SimpleNamespace
    r.c.budget=budget;torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    assert os.environ['CUDA_VISIBLE_DEVICES']==str(LOCK['physical_GPU']);torch.cuda.set_per_process_memory_fraction(.78)
    plan=json.loads((ROOT/'SOURCE_INPUTS_FROZEN.json').read_text());assert len(plan['queries'])==37 and r.c.digest(plan)==LOCK['input_binding']
    pkg=torch.load(MED/'retained/MED150.pt',map_location='cpu',weights_only=True);task=pkg['tasks'][0]
    runtime=load_real_runtime(SimpleNamespace(cpu_gate=PARENT/'private/cpu_gate'));runtime.model.eval();r.old.fp32(runtime)
    assert not any(p.requires_grad for p in runtime.model.parameters())
    bindings=r.c.read(PARENT/'private/EVAL_BINDINGS.json');assert runtime.generation_config==next(iter(bindings.values()))['generation']
    rng=torch.load(OLD/'GENERATION_RNG.pt',map_location='cpu',weights_only=False)
    def count(_module,args,kwargs):budget();COUNTS['forwards']+=1
    fh=runtime.model.register_forward_pre_hook(count,with_kwargs=True)
    vh=runtime.model.get_vision_tower().register_forward_pre_hook(lambda *_:COUNTS.__setitem__('vision_forwards',COUNTS['vision_forwards']+1))
    identities={}
    for begin in range(0,37,8):
        budget();runtime.model.half()
        try:prepared=[r.s.prepare(runtime,bindings,task,[q])[0] for q in plan['queries'][begin:begin+8]]
        finally:r.old.fp32(runtime)
        for item in prepared:
            budget();q=item[0];b=item[4];g=q['group_index']
            if g in identities:assert identities[g]==b['image_sha256']
            else:identities[g]=b['image_sha256']
            dest=ROOT/'base_outputs'/(r.c.digest(q['query_id'])+'.json');assert not dest.exists(),'No implicit retry'
            r.c.restore_rng(rng['rng']);np.random.set_state(rng['numpy_rng']);COUNTS['generations']+=1
            value,trace=r.s.fit.short.generate(runtime,None,item[2],item[3]);observed=r.c.rng()
            assert observed['python_rng']==rng['rng']['python_rng']
            for k in ('torch_rng','cuda_rng'):assert torch.equal(observed[k],rng['rng'][k])
            answer=dict(raw_answer=value.decoded_text,raw_token_ids=list(value.raw_token_ids));assert answer['raw_token_ids']
            write(str(dest.relative_to(ROOT)),dict(mode='base',prefix=0,owner=g,consumer_id=q['query_id'],binding=dict(input=q,judge_input=b),R0=answer,
                route=None,effective_expert=None,effective_state_hash='UNEDITED_BASE',generation_trace=trace,
                EOS=answer['raw_token_ids'][-1]==runtime.adapter.tokenizer.eos_token_id,at_cap=len(answer['raw_token_ids'])>=1024))
        write('BASE_PROGRESS.json',dict(time=time.time(),consumers=min(begin+8,37),total=37,counts=COUNTS))
    assert len(identities)==8 and len(list((ROOT/'base_outputs').glob('*.json')))==37 and COUNTS['generations']==37
    fh.remove();vh.remove();write('BASE_GENERATION_COMPLETE.json',dict(time=time.time(),consumers=37,groups=8,counts=COUNTS,same_image_binding_per_group=True,cuda_max_memory_allocated=torch.cuda.max_memory_allocated()))


if __name__=='__main__':
    code=0
    try:main()
    except BaseException:
        code=1;write('BASE_FAILURE.json',dict(time=time.time(),error=traceback.format_exc(),counts=COUNTS));raise
    finally:write('BASE_WORKER_EXIT.json',dict(time=time.time(),code=code,counts=COUNTS))
