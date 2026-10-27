"""Produce the frozen four-edit Base panel, reusing only exact prior Base outputs."""
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

ROOT=Path(os.environ['EXPERIMENT_ROOT']);PARENT=Path(os.environ['PARENT_RUN']);OLD=Path(os.environ['RAW_MED_ROOT'])
LOCK=json.loads((ROOT/'BASE_LOCK.json').read_text());CLOCK=json.loads((ROOT/'CLOCK.json').read_text())
COUNTS=dict(forwards=0,generations=0,backwards=0,updates=0,vision_forwards=0)


def write(name,value):
    p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p)
def budget():
    assert time.time()<CLOCK['hard_end']-30 and not (ROOT/'BASE_STOP.json').exists()
    assert time.time()-json.loads((ROOT/'BASE_GUARD_HEALTH.json').read_text())['time']<45
    for k,v in LOCK['budgets'].items():
        if k in COUNTS:assert COUNTS[k]<=v,(k,COUNTS[k],v)


def main():
    with (ROOT/'BASE_WORKER_ONCE').open('x'):pass
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(InterruptedError('Resource guard stop')))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Original clock')))
    signal.alarm(max(1,int(CLOCK['hard_end']-time.time()-30)))
    sys.path[:0]=[str(ROOT/'tools'),str(PARENT/'private/tools'),str(PARENT/'private/source'),str(PARENT/'private/official_llava')]
    import numpy as np
    import torch
    import module_run as r
    from scripts.medtrace.run_realmodel_core import load_real_runtime
    from types import SimpleNamespace
    r.c.budget=budget;torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    assert os.environ['CUDA_VISIBLE_DEVICES']==str(LOCK['physical_GPU'])
    torch.cuda.set_per_process_memory_fraction(.78)
    data=json.loads((ROOT/'ADDITIONAL_INPUTS.json').read_text());old=json.loads((ROOT/'HISTORICAL_BASE_NATIVE.json').read_text())
    assert len(data['tasks'])==len(old)==4 and data['candidate_binding']==LOCK['candidate_binding']
    runtime=load_real_runtime(SimpleNamespace(cpu_gate=PARENT/'private/cpu_gate'));runtime.model.eval();r.old.fp32(runtime)
    assert not any(p.requires_grad for p in runtime.model.parameters())
    bindings=r.c.read(PARENT/'private/EVAL_BINDINGS.json');original_runtime=next(iter(bindings.values()))['runtime']
    assert runtime.generation_config==next(iter(bindings.values()))['generation']
    saved_rng=torch.load(OLD/'GENERATION_RNG.pt',map_location='cpu',weights_only=False)
    def count(_module,args,kwargs):budget();COUNTS['forwards']+=1
    forward=runtime.model.register_forward_pre_hook(count,with_kwargs=True)
    vision=runtime.model.get_vision_tower().register_forward_pre_hook(lambda *_:COUNTS.__setitem__('vision_forwards',COUNTS['vision_forwards']+1))
    inputs=[]
    for entry,prior in zip(data['tasks'],old):
        task=entry['task'];native=dict(task['native'],audit_role='T0')
        rows=[(native,prior)]+[(v,None) for v in entry['source_rows'][1:]+entry['T1G']+entry['T2G']]
        rows += [({k:v for k,v in row.items() if k!='historical_base_binding'},row['historical_base_binding']) for row in entry['T2L']]
        for row,previous in rows:inputs.append((task,row,previous))
    assert len(inputs)==55 and len({q['query_id'] for _,q,_ in inputs})==55
    reused=0
    for begin in range(0,len(inputs),16):
        budget();runtime.model.half();prepared=[]
        try:
            for task,row,previous in inputs[begin:begin+16]:
                item=r.s.prepare(runtime,bindings,task,[row])[0];prepared.append((task,row,previous,item))
        finally:r.old.fp32(runtime)
        for task,row,previous,item in prepared:
            budget();path=ROOT/'base_outputs'/(r.c.digest(row['query_id'])+'.json')
            assert not path.exists(), 'No implicit generation retry'
            if previous is not None:
                oldbind=previous['historical_binding'] if 'historical_binding' in previous else previous
                oldanswer=previous['old_base_output'] if 'old_base_output' in previous else oldbind['output']
                b=item[4]
                assert oldbind['query_id']==row['query_id'] and oldbind['question']==row['question'] and oldbind['reference']==row['reference']
                assert oldbind['image_sha256']==b['image_sha256'] and oldbind['prompt_ids']==b['prompt_ids'] and oldbind['attention_mask']==b['attention_mask']
                assert oldbind['runtime']==original_runtime and oldbind['generation']==runtime.generation_config
                assert oldanswer['prompt_token_ids']==b['prompt_ids'] and oldanswer['error'] is None and oldanswer['raw_generated_token_ids']
                answer=dict(raw_answer=oldanswer['model_answer_raw'],raw_token_ids=oldanswer['raw_generated_token_ids']);reused+=1
            else:
                r.c.restore_rng(saved_rng['rng']);np.random.set_state(saved_rng['numpy_rng']);COUNTS['generations']+=1
                value,_=r.s.fit.short.generate(runtime,None,item[2],item[3])
                answer=dict(raw_answer=value.decoded_text,raw_token_ids=list(value.raw_token_ids));assert answer['raw_token_ids']
                after=r.c.rng();assert after['python_rng']==saved_rng['rng']['python_rng']
                for key in ('torch_rng','cuda_rng'):assert torch.equal(after[key],saved_rng['rng'][key])
            write(str(path.relative_to(ROOT)),dict(owner=task['order'],query=row,judge_input=item[4],R0=answer,
                previous_Base_reused=previous is not None,EOS=answer['raw_token_ids'][-1]==runtime.adapter.tokenizer.eos_token_id,
                at_cap=len(answer['raw_token_ids'])>=1024))
        write('BASE_PROGRESS.json',dict(time=time.time(),consumers=min(begin+16,len(inputs)),total=len(inputs),
            reused=reused,counts=COUNTS));print('BASE',min(begin+16,len(inputs)),len(inputs),flush=True)
    assert len(list((ROOT/'base_outputs').glob('*.json')))==55 and reused==7
    forward.remove();vision.remove()
    write('BASE_GENERATION_COMPLETE.json',dict(time=time.time(),consumers=55,historical_exact_reused=reused,
        new_generations=COUNTS['generations'],counts=COUNTS,cuda_max_memory_allocated=torch.cuda.max_memory_allocated()))


if __name__=='__main__':
    code=0
    try:main()
    except BaseException:
        code=1;write('BASE_FAILURE.json',dict(time=time.time(),error=traceback.format_exc(),counts=COUNTS));raise
    finally:write('BASE_WORKER_EXIT.json',dict(time=time.time(),code=code,counts=COUNTS))
